#!/usr/bin/env python3
"""Run the Veritas replication experiments.

  surrogate : strict respondent-level 80/20 holdout on public GSS-2018 with a
              surrogate LSM trained on the 80% only, repeated over split seeds.
  native    : public native DTAG model scored on public GSS-2018 respondents
              (gss_2018: rows overlap its training set; gss_2016: different
              survey wave, so none of the scored respondents trained it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import veritas_core as vc

LEVELS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]


def log_to(path):
    fh = open(path, "a")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()
    return log


def run_surrogate(args):
    from surrogate_lsm import SurrogateLSM, apply_coding, fit_coding, n_levels

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    raw = pd.read_pickle(args.raw)
    results, meta = [], []
    for split_seed in [int(s) for s in args.split_seeds.split(",")]:
        rng = np.random.default_rng(split_seed)
        idx = rng.permutation(len(raw))
        ntr = int(round(args.train_fraction * len(raw)))
        tr, te = np.sort(idx[:ntr]), np.sort(idx[ntr:])
        cod = fit_coding(raw, tr)
        K = n_levels(cod)
        Xtr, _ = apply_coding(raw.iloc[tr], cod)
        Xte, oov = apply_coding(raw.iloc[te], cod)
        t = time.time()
        model = SurrogateLSM(K, seed=split_seed).fit(Xtr)
        mtab = vc.marginal_table(Xtr, K)
        n_cells = int((Xte > 0).sum())
        U = model.score(Xte)
        M = vc.marginal_score(Xte, mtab)
        info = dict(split_seed=split_seed, n_train=len(tr), n_heldout=len(te),
                    n_items=len(K), heldout_cells=n_cells,
                    oov_cells_masked=oov, oov_fraction=oov / (n_cells + oov),
                    fit_seconds=time.time() - t,
                    heldout_nats_per_cell_lsm=float(U.sum() / n_cells),
                    heldout_nats_per_cell_marg=float(M.sum() / n_cells),
                    train_ids_sha256=hashlib.sha256(raw["__id"].iloc[tr].to_numpy().tobytes()).hexdigest()[:16])
        meta.append(info)
        log(f"split {split_seed}: {json.dumps(info)}")
        res = vc.run_levels(model, Xte, mtab, args.levels, args.repeats,
                            seed=split_seed, mechanisms=args.mechanisms,
                            donors=Xtr, log=log)
        for r in res:
            r["split_seed"] = split_seed
        results += res
        pd.DataFrame(results).to_csv(out / "auc_by_repeat.csv", index=False)
        (out / "splits.json").write_text(json.dumps(meta, indent=1))
    summarize(out)


def run_native(args):
    from native_adapter import NativeLSM

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "log.txt")
    model = NativeLSM(args.model_dir, workers=args.workers)
    df = pd.read_pickle(args.rows)
    X_all = model.encode(df)
    keep = np.flatnonzero((X_all > 0).sum(1) >= 50)
    rng = np.random.default_rng(args.row_seed)
    ev = np.sort(rng.choice(keep, min(args.n_rows, len(keep)), replace=False))
    donors = np.setdiff1d(np.arange(len(X_all)), ev)
    X, D = X_all[ev], X_all[donors]
    mtab = [np.maximum(p, 1e-12) / np.maximum(p, 1e-12).sum() for p in model.psi0()]
    info = dict(model=str(args.model_dir), n_rows=len(ev), row_seed=args.row_seed,
                observed_cells=int((X > 0).sum()),
                mean_observed_per_row=float((X > 0).sum(1).mean()),
                items_observed=int((X > 0).any(0).sum()), n_items=len(model.K),
                heldout_nats_per_cell_lsm=float(model.score(X).sum() / (X > 0).sum()),
                heldout_nats_per_cell_marg=float(vc.marginal_score(X, mtab).sum() / (X > 0).sum()))
    log(json.dumps(info))
    (out / "native_info.json").write_text(json.dumps(info, indent=1))
    results = []
    for mech in args.mechanisms:
        levels = args.seq_levels if mech == "sequential" else args.levels
        reps = args.seq_repeats if mech == "sequential" else args.repeats
        res = vc.run_levels(model, X, mtab, levels, reps, seed=args.row_seed,
                            mechanisms=[mech], donors=D, log=log)
        for r in res:
            r["split_seed"] = 0
        results += res
        pd.DataFrame(results).to_csv(out / "auc_by_repeat.csv", index=False)
    model.close()
    summarize(out)


def summarize(out: Path):
    df = pd.read_csv(out / "auc_by_repeat.csv")
    g = df.groupby(["mechanism", "rho_requested"])
    summ = g.agg(n=("auc_lsm", "size"),
                 rho_realized=("rho_realized", "mean"),
                 auc_lsm=("auc_lsm", "mean"), auc_lsm_sd=("auc_lsm", "std"),
                 auc_marg=("auc_marg", "mean"), auc_marg_sd=("auc_marg", "std"),
                 mean_dU=("mean_dU", "mean"), frac_dU_pos=("frac_dU_pos", "mean"),
                 dU_per_changed=("mean_dU_per_changed", "mean")).reset_index()
    # split-level variability (mean over repeats within split, sd across splits)
    if df["split_seed"].nunique() > 1:
        s = df.groupby(["mechanism", "rho_requested", "split_seed"])["auc_lsm"].mean()
        sd = s.groupby(level=[0, 1]).std().rename("auc_lsm_sd_across_splits")
        summ = summ.merge(sd.reset_index(), on=["mechanism", "rho_requested"])
    summ.to_csv(out / "auc_summary.csv", index=False)
    # matched realized comparison: interpolate permutation onto other mechanisms
    rows = []
    if "permutation" in set(summ.mechanism):
        p = summ[summ.mechanism == "permutation"].sort_values("rho_realized")
        for mech in sorted(set(summ.mechanism) - {"permutation"}):
            for _, r in summ[summ.mechanism == mech].iterrows():
                x = r.rho_realized
                inside = p.rho_realized.min() <= x <= p.rho_realized.max()
                rows.append(dict(
                    mechanism=mech, rho_requested=r.rho_requested, rho_realized=x,
                    auc_perm_interp=np.interp(x, p.rho_realized, p.auc_lsm) if inside else np.nan,
                    auc_mech=r.auc_lsm,
                    dU_perm_interp=np.interp(x, p.rho_realized, p.mean_dU) if inside else np.nan,
                    dU_mech=r.mean_dU))
    pd.DataFrame(rows).to_csv(out / "matched_realized.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("surrogate")
    a.add_argument("--raw", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--split-seeds", default="20261002")
    a.add_argument("--train-fraction", type=float, default=0.8)
    a.add_argument("--repeats", type=int, default=5)
    b = sub.add_parser("native")
    b.add_argument("--model-dir", required=True)
    b.add_argument("--rows", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--n-rows", type=int, default=400)
    b.add_argument("--row-seed", type=int, default=20261002)
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--seq-repeats", type=int, default=2)
    b.add_argument("--seq-levels", default="0.02,0.05,0.10,0.20")
    b.add_argument("--workers", type=int, default=4)
    for p in (a, b):
        p.add_argument("--levels", default=",".join(map(str, LEVELS)))
        p.add_argument("--mechanisms",
                       default="permutation,conditional,sequential,hotdeck,uniform")
    args = ap.parse_args()
    args.levels = [float(x) for x in args.levels.split(",")]
    args.mechanisms = args.mechanisms.split(",")
    if args.cmd == "native":
        args.seq_levels = [float(x) for x in args.seq_levels.split(",")]
        run_native(args)
    else:
        run_surrogate(args)


if __name__ == "__main__":
    main()
