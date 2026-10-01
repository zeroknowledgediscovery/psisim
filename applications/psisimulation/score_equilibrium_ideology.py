#!/usr/bin/env python3
"""Score hard equilibrium representatives with the DTAG ideology index.

Uses the exact DTAG definition

    I(s) = [qdistance(s_L, s) - qdistance(s_R, s)] / qdistance(s_L, s_R)

where s_L and s_R are sparse hard pole vectors built from
assets/polar_vectors/polar_vectors.csv. Positive values are closer to the
right pole, negative values are closer to the left pole.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BIN = REPO_ROOT / "bin"
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))

from common import resolve_model

qd = importlib.import_module("qdistance")


def load_poles(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_fields = list(reader.fieldnames or [])
        fields = {str(x).strip().lower(): x for x in raw_fields}

        # DTAG's committed polar_vectors.csv currently uses a blank first
        # header followed by R,L:
        #
        #     ,R,L
        #
        # pandas names that first column "Unnamed: 0"; csv.DictReader keeps
        # it as the empty string.  Support both representations.
        var_col = (
            fields.get("variable")
            or fields.get("unnamed: 0")
            or fields.get("unnamed:0")
        )
        if var_col is None and raw_fields:
            first = raw_fields[0]
            if str(first).strip() == "":
                var_col = first

        left_col = fields.get("left") or fields.get("l")
        right_col = fields.get("right") or fields.get("r")

        if var_col is None or left_col is None or right_col is None:
            raise ValueError(
                "polar CSV must have a variable/index column plus left/right "
                "columns (DTAG's ,R,L format is supported)"
            )
        left, right = {}, {}
        for row in reader:
            v = str(row[var_col]).strip().lower()
            if not v:
                continue
            lv = str(row[left_col]).strip()
            rv = str(row[right_col]).strip()
            if lv:
                left[v] = lv
            if rv:
                right[v] = rv
    return left, right


def auto_polar_path() -> Path:
    candidates = [
        REPO_ROOT.parent / "DTAG" / "assets" / "polar_vectors" / "polar_vectors.csv",
        Path.home() / "Dropbox" / "ZED" / "Research" / "DTAG" / "assets" / "polar_vectors" / "polar_vectors.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


def load_representatives(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        meta = {
            "cluster",
            "cluster_size",
            "equilibrium_index",
            "source_row",
            "mean_within_cluster_qdistance",
        }
        features = [c for c in fieldnames if c not in meta]
        records = list(reader)
    return features, records


def build_pole_vector(features, mapping):
    lower_to_index = {str(v).strip().lower(): i for i, v in enumerate(features)}
    row = [""] * len(features)
    used = 0
    for var, value in mapping.items():
        j = lower_to_index.get(str(var).strip().lower())
        if j is not None:
            row[j] = str(value)
            used += 1
    return row, used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gss/gss_2018")
    parser.add_argument(
        "--representatives",
        default="results/psisimulation/gss2018_equilibrium_clusters/major_cluster_representatives.csv",
    )
    parser.add_argument("--polar-vectors", default="")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--no-fetch", action="store_true")
    args = parser.parse_args()

    model = resolve_model(args.model, fetch=not args.no_fetch)
    trees = model / "trees" / "binary"
    reps_path = Path(args.representatives).expanduser().resolve()

    polar_path = (
        Path(args.polar_vectors).expanduser().resolve()
        if args.polar_vectors
        else auto_polar_path()
    )
    if not polar_path.is_file():
        raise FileNotFoundError(
            f"polar vectors not found: {polar_path}; pass --polar-vectors"
        )

    features, records = load_representatives(reps_path)
    left_map, right_map = load_poles(polar_path)

    sL, nL = build_pole_vector(features, left_map)
    sR, nR = build_pole_vector(features, right_map)

    eq_rows = [[str(rec.get(v, "")) for v in features] for rec in records]

    all_rows = [sL, sR] + eq_rows
    D = np.asarray(
        qd.qdistance_matrix(
            str(trees),
            all_rows,
            run_dir=str(model),
            threads=args.threads,
        ),
        dtype=float,
    )

    dLR = float(D[0, 1])
    if not np.isfinite(dLR) or dLR <= 0:
        raise RuntimeError(f"invalid left/right pole distance: {dLR}")

    print("model:", model)
    print("representatives:", reps_path)
    print("polar vectors:", polar_path)
    print("usable left assignments:", nL)
    print("usable right assignments:", nR)
    print("d(L,R):", dLR)
    print()
    print(
        "cluster  source_row      d_left       d_right      ideology       polviews"
    )

    for k, rec in enumerate(records):
        idx = k + 2
        dL = float(D[0, idx])
        dR = float(D[1, idx])
        ideology = (dL - dR) / dLR
        print(
            f"{int(rec['cluster']):7d} "
            f"{int(rec['source_row']):11d} "
            f"{dL:12.8f} "
            f"{dR:12.8f} "
            f"{ideology:12.8f} "
            f"{rec.get('polviews','')}"
        )


if __name__ == "__main__":
    main()
