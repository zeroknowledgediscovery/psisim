"""Adapter exposing a public native DTAG/LSM model to ``veritas_core``.

Uses the native ``predict_distribution`` extension (``PersistenceEvaluator``):
``score_batch`` gives native persistence P(x) = sum_i log2 phi_i(x_-i)(x_i)
with the native probability floor, and U = -(ln 2) P.

Integer code k >= 1 of item i is the k-th category of the model's own source
map (``column_strings_map[k]``), so codes round-trip exactly to the strings
the native runtime encodes.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

_W = {}


def _init(trees, run_dir, cats):
    import predict_distribution as P

    _W["P"] = P
    _W["trees"], _W["run"], _W["cats"] = trees, run_dir, cats
    _W["full"] = P.PersistenceEvaluator(trees, run_dir=run_dir)
    _W["full"].distributions([""] * len(cats))  # warm: first call loads lazily


def _decode(codes):
    cats = _W["cats"]
    return [cats[j][c - 1] if c > 0 else "" for j, c in enumerate(codes)]


def _dist_arrays(dlist):
    cats = _W["cats"]
    out = []
    for j, d in enumerate(dlist):
        out.append(np.array([d.get(c, 0.0) for c in cats[j]]) if d else None)
    return out


def _w_dists(codes):
    return _dist_arrays(_W["full"].distributions(_decode(codes)))


def _w_sequential(args):
    """Sequential completion of one row in its own random order."""
    codes, sel, seed = args
    rng = np.random.default_rng(seed)
    y = np.array(codes)
    y[sel] = 0
    cats = _W["cats"]
    for j in rng.permutation(sel):
        # A warmed full evaluator returns all conditionals in ~16 ms, far
        # cheaper than a per-tree evaluator's ~3 s lazy first call.
        d = _W["full"].distributions(_decode(y))[j]
        p = np.array([d.get(c, 0.0) for c in cats[j]])
        c = p.cumsum()
        y[j] = int((rng.random() * c[-1] > c).sum()) + 1
    return y


class NativeLSM:
    def __init__(self, model_dir: str | Path, workers: int = 4):
        model_dir = Path(model_dir)
        m = json.loads((model_dir / "source_maps" / "json_shards" / "0.json").read_text())
        self.names = [m[str(i)]["column_header"] for i in range(len(m))]
        self.cats = [m[str(i)]["column_strings_map"][1:] for i in range(len(m))]
        self.K = [len(c) for c in self.cats]
        self.trees = str(model_dir / "trees" / "binary")
        self.run = str(model_dir)
        _init(self.trees, self.run, self.cats)
        # workers=0: evaluate in-process. Use it when the caller also runs
        # joblib/loky pools; a fork pool created first can deadlock with them.
        self.pool = mp.get_context("fork").Pool(
            workers, initializer=_init, initargs=(self.trees, self.run, self.cats)) if workers else None

    def encode(self, df):
        """Strings (model categories, '' = missing) -> integer codes."""
        X = np.zeros(df.shape, np.int32)
        for j, n in enumerate(self.names):
            lut = {c: k + 1 for k, c in enumerate(self.cats[j])}
            X[:, j] = [lut.get(s, 0) for s in df[n].to_numpy()]
        return X

    def psi0(self):
        d = _W["full"].distributions([""] * len(self.names))
        return _dist_arrays(d)

    # --- adapter API --------------------------------------------------------
    def score(self, X):
        rows = [_decode(r) for r in X]
        r = _W["full"].score_batch(rows, threads=4)
        return -math.log(2.0) * np.asarray(r["persistence"], float)

    def dists(self, X):
        per_row = (self.pool.map(_w_dists, list(X), chunksize=4) if self.pool
                   else [_w_dists(r) for r in X])
        out = []
        for j, k in enumerate(self.K):
            out.append(np.stack([pr[j] if pr[j] is not None else np.full(k, 1.0 / k)
                                 for pr in per_row]))
        return out

    def dist_item(self, X, j):
        rows = []
        for r in X:
            d = _W["full"].distributions(_decode(r))[j]
            rows.append([d.get(c, 0.0) for c in self.cats[j]])
        return np.array(rows)

    def sequential(self, X, S, rng):
        seeds = rng.integers(0, 2**63 - 1, size=X.shape[0])
        args = [(X[r], np.flatnonzero(S[r]), int(seeds[r])) for r in range(X.shape[0])]
        Y = (self.pool.map(_w_sequential, args, chunksize=2) if self.pool
             else [_w_sequential(a) for a in args])
        return np.stack(Y).astype(np.int32)

    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()
