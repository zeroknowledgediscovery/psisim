#!/usr/bin/env python3
"""Start at Psi0 from the all-missing row and study centered-LDP relaxation."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from common import (
    psi_distance,
    resolve_model,
    resident_empty_state,
    snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gss/gss_2018")
    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--empirical-n", type=int, default=10)
    parser.add_argument(
        "--event",
        choices=["zero_action", "mode", "sample"],
        default="mode",
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--tol", type=float, default=0.0)
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument(
        "--out",
        default="results/psisimulation/empty",
    )
    args = parser.parse_args()

    model = resolve_model(args.model, fetch=not args.no_fetch)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"model: {model}")
    print("constructing Psi0 from the all-missing row...", flush=True)
    t0 = time.perf_counter()
    state = resident_empty_state(model)
    psi_initial = snapshot(state)
    print(f"initialized in {time.perf_counter()-t0:.3f} sec")

    history = []
    print()
    print(
        "sweep       sec        meanTV          maxTV   "
        "maxcol       eventTV          nD"
    )

    for sweep in range(1, args.sweeps + 1):
        t0 = time.perf_counter()
        h = dict(
            state.sweep(
                n=args.empirical_n,
                event=args.event,
                seed=args.seed,
                response_scale=1.0,
                threads=args.threads,
                random_permutation=False,
            )
        )
        dt = time.perf_counter() - t0
        history.append((sweep, dt, h))

        print(
            f"{sweep:5d} "
            f"{dt:9.3f} "
            f"{h['mean_tv']:14.10g} "
            f"{h['max_tv']:14.10g} "
            f"{h['max_col']:7d} "
            f"{h['event_tv']:13.9g} "
            f"{h['sanov_exponent']:12.6g}"
        )

        if args.tol > 0 and h["max_tv"] <= args.tol:
            print(f"stopping: maxTV <= {args.tol:g}")
            break

    psi_final = snapshot(state)
    mean_d, max_d = psi_distance(psi_initial, psi_final)
    print()
    print("Psi0 -> final:")
    print("  mean TV =", mean_d)
    print("  max  TV =", max_d)

    hard = list(state.hard_row())
    with (out / "final_hard_row.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        csv.writer(handle).writerow(hard)

    with (out / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sweep",
                "seconds",
                "mean_tv",
                "max_tv",
                "max_col",
                "event_tv",
                "sanov_exponent",
                "projected_count",
            ]
        )
        for sweep, sec, h in history:
            writer.writerow(
                [
                    sweep,
                    sec,
                    h["mean_tv"],
                    h["max_tv"],
                    h["max_col"],
                    h["event_tv"],
                    h["sanov_exponent"],
                    h["projected_count"],
                ]
            )

    print(f"written: {out}")


if __name__ == "__main__":
    main()
