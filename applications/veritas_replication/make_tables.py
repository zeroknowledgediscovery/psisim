#!/usr/bin/env python3
"""Render result CSVs as markdown tables (written to findings/tables.md)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "findings")
OUT.mkdir(exist_ok=True)

# README section 12/13 numbers (native LSM, GSS-2018 80/20 holdout).
README = {
    "permutation": [(0.005, .0013, .52890, .50009), (0.01, .0055, .58466, .50014),
                    (0.02, .0119, .65765, .50073), (0.05, .0317, .83683, .50177),
                    (0.10, .0625, .95440, .50287), (0.20, .1206, .99718, .50604)],
    "conditional": [(0.005, .0014, .50028, .50030), (0.01, .0031, .49851, .50027),
                    (0.02, .0062, .49844, .50069), (0.05, .0158, .50384, .50098),
                    (0.10, .0331, .54626, .50278), (0.20, .0698, .69109, .50426)],
}
MECH_ORDER = ["permutation", "conditional", "sequential", "hotdeck", "uniform",
              "conditional_self", "conditional_cross", "sequential_self", "sequential_cross"]


def pct(x):
    return f"{100 * x:.2f}%"


def mech_table(s: pd.DataFrame, split_sd: bool) -> str:
    hdr = ["Mechanism", "Requested", "Realized", "AUC U_LSM", "AUC U_marg",
           "Mean ΔU (nats)", "ΔU per changed cell", "Frac ΔU>0"]
    if split_sd:
        hdr.insert(4, "SD across splits")
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---:|" * len(hdr)]
    s = s.copy()
    s["o"] = s.mechanism.map({m: i for i, m in enumerate(MECH_ORDER)})
    for _, r in s.sort_values(["o", "rho_requested"]).iterrows():
        cells = [r.mechanism, pct(r.rho_requested), pct(r.rho_realized),
                 f"{r.auc_lsm:.4f} ± {r.auc_lsm_sd:.4f}", f"{r.auc_marg:.4f}",
                 f"{r.mean_dU:.2f}", f"{r.dU_per_changed:.3f}", f"{r.frac_dU_pos:.3f}"]
        if split_sd:
            cells.insert(4, f"{r.auc_lsm_sd_across_splits:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def readme_compare(s: pd.DataFrame) -> str:
    lines = ["| Mechanism | Requested | README realized | Replication realized | README AUC | Replication AUC | README AUC_marg | Replication AUC_marg |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mech, rows in README.items():
        for rho, real, a, am in rows:
            r = s[(s.mechanism == mech) & np.isclose(s.rho_requested, rho)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            lines.append(f"| {mech} | {pct(rho)} | {pct(real)} | {pct(r.rho_realized)} | "
                         f"{a:.4f} | {r.auc_lsm:.4f} | {am:.4f} | {r.auc_marg:.4f} |")
    return "\n".join(lines)


def matched(s: pd.DataFrame, ref="permutation") -> str:
    """AUC of each mechanism vs permutation interpolated at the same realized fraction."""
    p = s[s.mechanism == ref].sort_values("rho_realized")
    lines = ["| Mechanism | Realized | AUC (mechanism) | AUC permutation, interpolated | Difference | ΔU/changed cell (mech) | ΔU/changed cell (perm, interp) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    s = s.copy()
    s["o"] = s.mechanism.map({m: i for i, m in enumerate(MECH_ORDER)})
    for _, r in s[s.mechanism != ref].sort_values(["o", "rho_requested"]).iterrows():
        x = r.rho_realized
        if not (p.rho_realized.min() <= x <= p.rho_realized.max()):
            continue
        a = np.interp(x, p.rho_realized, p.auc_lsm)
        dc = np.interp(x, p.rho_realized, p.dU_per_changed)
        lines.append(f"| {r.mechanism} | {pct(x)} | {r.auc_lsm:.4f} | {a:.4f} | {a - r.auc_lsm:+.4f} | "
                     f"{r.dU_per_changed:.3f} | {dc:.3f} |")
    return "\n".join(lines)


parts = []
for name in sorted(p.name for p in R.iterdir() if (p / "auc_by_repeat.csv").exists()):
    d = pd.read_csv(R / name / "auc_by_repeat.csv")
    split_sd = "split_seed" in d and d.split_seed.nunique() > 1
    g = d.groupby(["mechanism", "rho_requested"])
    s = g.agg(rho_realized=("rho_realized", "mean"), auc_lsm=("auc_lsm", "mean"),
              auc_lsm_sd=("auc_lsm", "std"), auc_marg=("auc_marg", "mean"),
              mean_dU=("mean_dU", "mean"), frac_dU_pos=("frac_dU_pos", "mean"),
              dU_per_changed=("mean_dU_per_changed", "mean"), n=("auc_lsm", "size")).reset_index()
    if split_sd:
        ss = d.groupby(["mechanism", "rho_requested", "split_seed"]).auc_lsm.mean()
        s = s.merge(ss.groupby(level=[0, 1]).std().rename("auc_lsm_sd_across_splits").reset_index())
    s.to_csv(OUT / f"{name}_summary.csv", index=False)
    parts.append(f"## {name}\n\nRepeats per cell: {int(s.n.min())}–{int(s.n.max())}"
                 + (f"; split seeds: {sorted(d.split_seed.unique().tolist())}" if split_sd else "")
                 + " (± is SD over all repeats).\n\n" + mech_table(s, split_sd))
    if name.startswith("surrogate") or name.startswith("native"):
        parts.append(f"### {name}: README numbers vs replication\n\n" + readme_compare(s))
    parts.append(f"### {name}: matched realized-corruption comparison\n\n" + matched(s))
(OUT / "tables.md").write_text("\n\n".join(parts) + "\n")
print((OUT / "tables.md").read_text())
