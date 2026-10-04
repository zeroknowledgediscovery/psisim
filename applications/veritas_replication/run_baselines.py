#!/usr/bin/env python3
"""Can non-LSM methods replicate the Veritas results?

Strict 80/20 holdout on public GSS-2018. Every detector and generator is
fitted on the 80% only. For each (mechanism, level, repeat) ONE corrupted copy
of the held-out rows is generated and scored by every detector, so all
detectors are compared on identical rows.

Detectors : lsm (surrogate LSM), marginal, pairwise, logreg, latentclass,
            pca, iforest, knn
Generators: permutation (marginal-preserving), lsm_conditional,
            lsm_sequential, logreg_conditional, latentclass_conditional,
            splice (nearest-respondent donor answers), uniform
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baselines as bl
import veritas_core as vc
from surrogate_lsm import SurrogateLSM, apply_coding, fit_coding, n_levels


class Marginal:
    name = "marginal"

    def __init__(self, K):
        self.K = K

    def fit(self, X):
        self.t = vc.marginal_table(X, self.K)
        return self

    def score(self, X):
        return vc.marginal_score(X, self.t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-seeds", default="20261002,1,2")
    ap.add_argument("--levels", default="0.005,0.01,0.02,0.05,0.10,0.20")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    levels = [float(x) for x in args.levels.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_pickle(args.raw)
    results, meta = [], []
    for split_seed in [int(s) for s in args.split_seeds.split(",")]:
        idx = np.random.default_rng(split_seed).permutation(len(raw))
        ntr = int(round(0.8 * len(raw)))
        tr, te = np.sort(idx[:ntr]), np.sort(idx[ntr:])
        cod = fit_coding(raw, tr)
        K = n_levels(cod)
        Xtr, _ = apply_coding(raw.iloc[tr], cod)
        X, _ = apply_coding(raw.iloc[te], cod)
        n_obs = int((X > 0).sum())

        det = {}
        fit_s = {}
        for name, mk in [("lsm", lambda: SurrogateLSM(K, seed=split_seed)),
                         ("marginal", lambda: Marginal(K)),
                         ("pairwise", lambda: bl.PairwiseBest(K)),
                         ("logreg", lambda: bl.LogRegPL(K)),
                         ("latentclass", lambda: bl.LatentClass(K, seed=split_seed)),
                         ("pca", lambda: bl.PCARecon(K)),
                         ("iforest", lambda: bl.IForest(K)),
                         ("knn", lambda: bl.KNNDistance(K))]:
            t = time.time()
            det[name] = mk().fit(Xtr)
            fit_s[name] = round(time.time() - t, 1)
        base = {n: d.score(X) for n, d in det.items()}
        info = dict(split_seed=split_seed, n_train=len(tr), n_heldout=len(te),
                    n_items=len(K), fit_seconds=fit_s,
                    heldout_nats_per_cell={n: float(base[n].sum() / n_obs)
                                           for n in ("lsm", "marginal", "pairwise",
                                                     "logreg")},
                    latentclass_nats_per_cell_joint=float(base["latentclass"].sum() / n_obs))
        meta.append(info)
        print(json.dumps(info), flush=True)

        gens = {
            "permutation": lambda S, g: vc.marginal_permutation(X, rho, g),
            "lsm_conditional": lambda S, g: vc.conditional_independent(det["lsm"], X, S, g),
            "lsm_sequential": lambda S, g: vc.conditional_sequential(det["lsm"], X, S, g),
            "logreg_conditional": lambda S, g: vc.conditional_independent(det["logreg"], X, S, g),
            "latentclass_conditional": lambda S, g: vc.conditional_independent(det["latentclass"], X, S, g),
            "splice": lambda S, g: vc.hotdeck_replacement(X, S, Xtr, g),
            "uniform": lambda S, g: vc.uniform_replacement(X, S, K, g),
        }
        for rho in levels:
            for rep in range(args.repeats):
                S = vc.select_cells(X, rho, np.random.default_rng([split_seed, rep, int(rho * 1e5)]))
                for gi, (gname, f) in enumerate(gens.items()):
                    Y = f(S, np.random.default_rng([split_seed, rep, int(rho * 1e5), gi]))
                    assert ((Y > 0) == (X > 0)).all()
                    realized = int((Y != X).sum()) / n_obs
                    for dname, d in det.items():
                        s1 = d.score(Y)
                        results.append(dict(split_seed=split_seed, generator=gname,
                                            detector=dname, rho_requested=rho, repeat=rep,
                                            rho_realized=realized,
                                            auc=vc.auc(base[dname], s1)))
                    row = {r["detector"]: round(r["auc"], 3) for r in results[-len(det):]}
                    print(time.strftime("%H:%M:%S"), split_seed, f"{gname:24s}", rho, rep,
                          f"real={realized:.4f}", row, flush=True)
                pd.DataFrame(results).to_csv(out / "auc_by_repeat.csv", index=False)
        (out / "splits.json").write_text(json.dumps(meta, indent=1))

    df = pd.DataFrame(results)
    s = df.groupby(["generator", "rho_requested", "detector"]).agg(
        rho_realized=("rho_realized", "mean"), auc=("auc", "mean"),
        auc_sd=("auc", "std")).reset_index()
    s.to_csv(out / "auc_summary_long.csv", index=False)
    wide = s.pivot_table(index=["generator", "rho_requested"], columns="detector",
                         values="auc").reset_index()
    real = s.groupby(["generator", "rho_requested"]).rho_realized.mean().reset_index()
    wide = real.merge(wide, on=["generator", "rho_requested"])
    wide.to_csv(out / "auc_summary_wide.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(wide.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
