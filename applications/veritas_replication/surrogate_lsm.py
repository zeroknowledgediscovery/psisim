"""A transparent surrogate LSM: one categorical decision tree per item.

The native LSM trainer is not available outside the private ``lsm`` repo, so
the strict 80/20 holdout protocol is run with this surrogate, which has the
same structure as the native model:

  * one learned conditional predictor phi_i(x_-i) per item, trained only on
    training respondents;
  * hard partial rows as input, missing coordinates handled natively (sklearn
    trees route NaN to the side learned during training);
  * leaf distributions over the item's alphabet, here smoothed toward the
    item's training marginal (pseudo-count ``alpha``).

Features are the one-hot blocks of all items (NaN when the item is missing).
While training phi_j, block j is set entirely to NaN, so the tree cannot split
on its own target; at inference block j is therefore ignored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

PROB_FLOOR = 1e-12


# --------------------------------------------------------------------------
# coding (fit on training rows only)
# --------------------------------------------------------------------------
def fit_coding(raw: pd.DataFrame, tr: np.ndarray, max_levels: int = 25,
               n_bins: int = 5, min_obs: int = 20):
    """Return list of (name, kind, spec) codings fitted on training rows."""
    codings = []
    for c in raw.columns:
        if c.startswith("__"):
            continue
        v = raw[c].iloc[tr].dropna()
        if len(v) < min_obs:
            continue
        u = np.sort(v.unique())
        if len(u) < 2:
            continue
        if len(u) > max_levels:
            qs = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1]))
            codings.append((c, "bins", qs))
        else:
            codings.append((c, "cat", u))
    return codings


def apply_coding(raw: pd.DataFrame, codings):
    """Integer codes (0 = missing). Returns X and count of OOV cells masked."""
    X = np.zeros((len(raw), len(codings)), np.int32)
    oov = 0
    for j, (c, kind, spec) in enumerate(codings):
        v = raw[c].to_numpy(float)
        o = ~np.isnan(v)
        if kind == "bins":
            X[o, j] = np.searchsorted(spec, v[o], side="right") + 1
        else:
            idx = np.searchsorted(spec, v[o])
            idx = np.clip(idx, 0, len(spec) - 1)
            hit = spec[idx] == v[o]
            col = np.zeros(o.sum(), np.int32)
            col[hit] = idx[hit] + 1
            oov += int((~hit).sum())
            X[o, j] = col
    return X, oov


def n_levels(codings):
    return [len(s) + 1 if k == "bins" else len(s) for _, k, s in codings]


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class OneHot:
    def __init__(self, K):
        self.K = list(K)
        self.off = np.concatenate([[0], np.cumsum(self.K)]).astype(int)

    def __call__(self, X):
        n = X.shape[0]
        F = np.full((n, self.off[-1]), np.nan, np.float32)
        for j, k in enumerate(self.K):
            o = np.flatnonzero(X[:, j] > 0)
            if len(o) == 0:
                continue
            F[o, self.off[j]:self.off[j] + k] = 0.0
            F[o, self.off[j] + X[o, j] - 1] = 1.0
        return F


def _fit_one(F, y, block, k, prior, params, alpha):
    o = y > 0
    Fj = F[o].copy()
    Fj[:, block[0]:block[1]] = np.nan
    yj = y[o]
    tree = DecisionTreeClassifier(**params)
    tree.fit(Fj, yj)
    t = tree.tree_
    counts = np.zeros((t.node_count, k))
    val = t.value[:, 0, :]
    val = val / np.maximum(val.sum(1, keepdims=True), 1e-300)
    counts[:, tree.classes_.astype(int) - 1] = val * t.weighted_n_node_samples[:, None]
    table = (counts + alpha * prior[None, :]) / (counts.sum(1, keepdims=True) + alpha)
    t_light = tree.tree_  # keep only the low-level Tree for apply()
    return t_light, table.astype(np.float64)


class SurrogateLSM:
    def __init__(self, K, max_depth=8, min_samples_leaf=80, alpha=10.0,
                 n_jobs=4, seed=0):
        self.K = list(K)
        self.enc = OneHot(K)
        self.params = dict(criterion="entropy", max_depth=max_depth,
                           min_samples_leaf=min_samples_leaf, random_state=seed)
        self.alpha = alpha
        self.n_jobs = n_jobs

    def fit(self, Xtr):
        F = self.enc(Xtr)
        priors = []
        for j, k in enumerate(self.K):
            c = np.bincount(Xtr[:, j], minlength=k + 1)[1:] + 0.5
            priors.append(c / c.sum())
        res = Parallel(n_jobs=self.n_jobs, max_nbytes="1M")(
            delayed(_fit_one)(F, Xtr[:, j], (self.enc.off[j], self.enc.off[j + 1]),
                              k, priors[j], self.params, self.alpha)
            for j, k in enumerate(self.K))
        self.trees = [r[0] for r in res]
        self.tables = [r[1] for r in res]
        return self

    # --- model adapter API used by veritas_core ---------------------------
    def _leaf(self, j, F):
        return self.trees[j].apply(F)

    def score(self, X):
        F = self.enc(X)
        U = np.zeros(X.shape[0])
        for j in range(len(self.K)):
            o = np.flatnonzero(X[:, j] > 0)
            if len(o) == 0:
                continue
            p = self.tables[j][self._leaf(j, F[o]), X[o, j] - 1]
            U[o] -= np.log(np.maximum(p, PROB_FLOOR))
        return U

    def dists(self, X):
        F = self.enc(X)
        return [self.tables[j][self._leaf(j, F)] for j in range(len(self.K))]

    def dist_item(self, X, j):
        return self.tables[j][self._leaf(j, self.enc(X))]

    def n_splits_on(self):
        """Number of trees that use each item (learned dependency in-degree)."""
        owner = np.repeat(np.arange(len(self.K)), self.K)
        used = np.zeros(len(self.K), int)
        for t in self.trees:
            f = t.feature[t.feature >= 0]
            used[np.unique(owner[f])] += 1
        return used
