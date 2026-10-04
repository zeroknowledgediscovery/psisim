#!/usr/bin/env python3
"""Generator/scorer separation test for LSM-conditional replacement.

In the README design the model that generates conditional replacements is the
same model that scores them. Low AUC can then mean either "the replacements
respect the population's conditional structure" or "a model assigns high
probability to its own samples".

Here the 80% training set is split into disjoint halves A and B; surrogate
LSMs are trained on each. All rows are scored by model B:

  conditional_self  : x'_j ~ phi^B_j(x_-S)        (README design)
  conditional_cross : x'_j ~ phi^A_j(x_-S)        (independent model, same population)
  sequential_self / sequential_cross               (README 4.3)
  permutation, hotdeck (donors = training rows), uniform

If cross ~ self, the README's structural interpretation holds; if cross moves
toward hotdeck/permutation, part of the conditional result is self-consistency.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import veritas_core as vc
from surrogate_lsm import SurrogateLSM, apply_coding, fit_coding, n_levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-seeds", default="20261002,1,2")
    ap.add_argument("--levels", default="0.01,0.02,0.05,0.10,0.20")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    levels = [float(x) for x in args.levels.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_pickle(args.raw)
    results, meta = [], []
    for split_seed in [int(s) for s in args.split_seeds.split(",")]:
        rng = np.random.default_rng(split_seed)
        idx = rng.permutation(len(raw))
        ntr = int(round(0.8 * len(raw)))
        tr, te = np.sort(idx[:ntr]), np.sort(idx[ntr:])
        half = np.random.default_rng(split_seed + 1).permutation(tr)
        trA, trB = np.sort(half[: ntr // 2]), np.sort(half[ntr // 2:])
        cod = fit_coding(raw, tr)
        K = n_levels(cod)
        Xtr, _ = apply_coding(raw.iloc[tr], cod)
        XA, _ = apply_coding(raw.iloc[trA], cod)
        XB, _ = apply_coding(raw.iloc[trB], cod)
        X, _ = apply_coding(raw.iloc[te], cod)
        A = SurrogateLSM(K, seed=split_seed).fit(XA)
        B = SurrogateLSM(K, seed=split_seed + 7).fit(XB)
        mtab = vc.marginal_table(XB, K)
        n_obs = int((X > 0).sum())
        U0 = B.score(X)
        meta.append(dict(split_seed=split_seed, nA=len(trA), nB=len(trB), n_test=len(te),
                         nats_cell_B=float(U0.sum() / n_obs),
                         nats_cell_A=float(A.score(X).sum() / n_obs),
                         nats_cell_marg=float(vc.marginal_score(X, mtab).sum() / n_obs)))
        print(meta[-1], flush=True)
        for rho in levels:
            for rep in range(args.repeats):
                r0 = np.random.default_rng([split_seed, rep, int(rho * 1e5)])
                S = vc.select_cells(X, rho, r0)
                gens = {
                    "permutation": lambda g: vc.marginal_permutation(X, rho, g),
                    "conditional_self": lambda g: vc.conditional_independent(B, X, S, g),
                    "conditional_cross": lambda g: vc.conditional_independent(A, X, S, g),
                    "sequential_self": lambda g: vc.conditional_sequential(B, X, S, g),
                    "sequential_cross": lambda g: vc.conditional_sequential(A, X, S, g),
                    "hotdeck": lambda g: vc.hotdeck_replacement(X, S, Xtr, g),
                    "uniform": lambda g: vc.uniform_replacement(X, S, K, g),
                }
                for k, (mech, f) in enumerate(gens.items()):
                    Y = f(np.random.default_rng([split_seed, rep, int(rho * 1e5), k]))
                    assert ((Y > 0) == (X > 0)).all()
                    U1 = B.score(Y)
                    ch = int((Y != X).sum())
                    dU = U1 - U0
                    results.append(dict(
                        split_seed=split_seed, mechanism=mech, rho_requested=rho,
                        repeat=rep, rho_realized=ch / n_obs, auc_lsm=vc.auc(U0, U1),
                        auc_marg=vc.auc(vc.marginal_score(X, mtab), vc.marginal_score(Y, mtab)),
                        mean_dU=float(dU.mean()), frac_dU_pos=float((dU > 0).mean()),
                        mean_dU_per_changed=float(dU.sum() / max(ch, 1))))
                    print(time.strftime("%H:%M:%S"), split_seed, mech, rho, rep,
                          round(results[-1]["rho_realized"], 4), round(results[-1]["auc_lsm"], 4),
                          flush=True)
                pd.DataFrame(results).to_csv(out / "auc_by_repeat.csv", index=False)
        (out / "splits.json").write_text(json.dumps(meta, indent=1))
    df = pd.DataFrame(results)
    s = df.groupby(["mechanism", "rho_requested"]).agg(
        rho_realized=("rho_realized", "mean"), auc_lsm=("auc_lsm", "mean"),
        auc_lsm_sd=("auc_lsm", "std"), auc_marg=("auc_marg", "mean"),
        mean_dU=("mean_dU", "mean"), dU_per_changed=("mean_dU_per_changed", "mean")).reset_index()
    s.to_csv(out / "auc_summary.csv", index=False)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
