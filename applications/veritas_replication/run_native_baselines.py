#!/usr/bin/env python3
"""Native LSM vs non-LSM detectors, all trained on GSS-2016, scored on GSS-2018.

The public native DTAG ``gss/gss_2016`` model was trained on (MAGICS) 2016
respondents. The comparison detectors are trained on the public GSS-2016
respondents, coded with exactly the same item mapping (``data_prep.py``
audit). All are scored on the same 1000 GSS-2018 respondents used by
``run_experiments.py native`` and on identical corrupted copies.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baselines as bl
import veritas_core as vc
from data_prep import public_strings
from native_adapter import NativeLSM


class Marginal:
    def __init__(self, K):
        self.K = K

    def fit(self, X):
        self.t = vc.marginal_table(X, self.K)
        return self

    def score(self, X):
        return vc.marginal_score(X, self.t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--train-pkl", required=True, help="public GSS rows of the model's wave")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--eval-rows", required=True, help="native-coded GSS-2018 rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rows", type=int, default=1000)
    ap.add_argument("--row-seed", type=int, default=20261002)
    ap.add_argument("--levels", default="0.005,0.01,0.02,0.05,0.10,0.20")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    levels = [float(x) for x in args.levels.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = NativeLSM(args.model_dir)
    K = model.K
    audit = json.loads(Path(args.audit).read_text())
    labels = pickle.loads(Path(args.labels).read_bytes())
    pub = pd.read_pickle(args.train_pkl)
    tr_df = pd.DataFrame("", index=pub.index, columns=model.names, dtype=object)
    for n in audit["accepted"]:
        mp = audit["per_item"][n]["mapping"]
        s = public_strings(pub[n.lower()], labels.get(n.lower(), {}))
        tr_df[n] = s.map(lambda v: mp.get(v, ""))
    Xtr = model.encode(tr_df)
    Xtr = Xtr[(Xtr > 0).sum(1) >= 50]

    X_all = model.encode(pd.read_pickle(args.eval_rows))
    keep = np.flatnonzero((X_all > 0).sum(1) >= 50)
    ev = np.sort(np.random.default_rng(args.row_seed).choice(
        keep, min(args.n_rows, len(keep)), replace=False))
    X = X_all[ev]
    n_obs = int((X > 0).sum())

    det, fit_s = {"lsm_native": model}, {}
    for name, mk in [("marginal", lambda: Marginal(K)), ("pairwise", lambda: bl.PairwiseBest(K)),
                     ("logreg", lambda: bl.LogRegPL(K)), ("latentclass", lambda: bl.LatentClass(K)),
                     ("pca", lambda: bl.PCARecon(K)), ("iforest", lambda: bl.IForest(K)),
                     ("knn", lambda: bl.KNNDistance(K))]:
        t = time.time()
        det[name] = mk().fit(Xtr)
        fit_s[name] = round(time.time() - t, 1)
    base = {n: d.score(X) for n, d in det.items()}
    info = dict(model=args.model_dir, n_train_public=int(len(Xtr)),
                train_observed_per_row=float((Xtr > 0).sum(1).mean()),
                n_eval=len(ev), eval_observed_per_row=float((X > 0).sum(1).mean()),
                fit_seconds=fit_s,
                nats_per_cell={n: float(base[n].sum() / n_obs)
                               for n in ("lsm_native", "marginal", "pairwise", "logreg")})
    print(json.dumps(info), flush=True)
    (out / "info.json").write_text(json.dumps(info, indent=1))

    results = []
    for rho in levels:
        for rep in range(args.repeats):
            S = vc.select_cells(X, rho, np.random.default_rng([args.row_seed, rep, int(rho * 1e5)]))
            gens = {
                "permutation": lambda g: vc.marginal_permutation(X, rho, g),
                "lsm_native_conditional": lambda g: vc.conditional_independent(model, X, S, g),
                "logreg_conditional": lambda g: vc.conditional_independent(det["logreg"], X, S, g),
                "latentclass_conditional": lambda g: vc.conditional_independent(det["latentclass"], X, S, g),
                "splice": lambda g: vc.hotdeck_replacement(X, S, Xtr, g),
                "uniform": lambda g: vc.uniform_replacement(X, S, K, g),
            }
            for gi, (gname, f) in enumerate(gens.items()):
                Y = f(np.random.default_rng([args.row_seed, rep, int(rho * 1e5), gi]))
                assert ((Y > 0) == (X > 0)).all()
                realized = int((Y != X).sum()) / n_obs
                for dname, d in det.items():
                    results.append(dict(generator=gname, detector=dname, rho_requested=rho,
                                        repeat=rep, rho_realized=realized,
                                        auc=vc.auc(base[dname], d.score(Y))))
                row = {r["detector"]: round(r["auc"], 3) for r in results[-len(det):]}
                print(time.strftime("%H:%M:%S"), f"{gname:24s}", rho, rep,
                      f"real={realized:.4f}", row, flush=True)
            pd.DataFrame(results).to_csv(out / "auc_by_repeat.csv", index=False)
    model.close()
    df = pd.DataFrame(results)
    s = df.groupby(["generator", "rho_requested", "detector"]).agg(
        rho_realized=("rho_realized", "mean"), auc=("auc", "mean"),
        auc_sd=("auc", "std")).reset_index()
    s.to_csv(out / "auc_summary_long.csv", index=False)
    wide = s.pivot_table(index=["generator", "rho_requested"], columns="detector",
                         values="auc").reset_index()
    wide = s.groupby(["generator", "rho_requested"]).rho_realized.mean().reset_index().merge(
        wide, on=["generator", "rho_requested"])
    wide.to_csv(out / "auc_summary_wide.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(wide.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
