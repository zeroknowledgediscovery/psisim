#!/usr/bin/env python3
"""Minimal invariant check for a downloaded model."""
from __future__ import annotations

import argparse

from common import psi_distance, resolve_model, resident_empty_state, snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gss/gss_2018")
    parser.add_argument("--column", type=int, default=0)
    parser.add_argument("--value", default="working fulltime")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--no-fetch", action="store_true")
    args = parser.parse_args()

    model = resolve_model(args.model, fetch=not args.no_fetch)
    state = resident_empty_state(model)

    before = snapshot(state)
    z = dict(
        state.sweep(
            n=10,
            event="zero_action",
            seed=1,
            response_scale=1.0,
            threads=args.threads,
            random_permutation=False,
        )
    )
    after = snapshot(state)
    mean_d, max_d = psi_distance(before, after)

    print("zero-action check")
    print("  reported meanTV =", z["mean_tv"])
    print("  reported maxTV  =", z["max_tv"])
    print("  direct meanTV   =", mean_d)
    print("  direct maxTV    =", max_d)

    if max_d > 1e-12:
        raise SystemExit("FAIL: zero-action update moved Psi")

    h = dict(
        state.hard_observe(
            source=args.column,
            value=args.value,
            response_scale=1.0,
            threads=args.threads,
            clamp=True,
        )
    )

    q = dict(state.distribution(args.column))
    if abs(float(q.get(args.value, 0.0)) - 1.0) > 1e-12:
        raise SystemExit("FAIL: hard observation did not collapse source")

    state.sweep(
        n=10,
        event="mode",
        seed=2,
        response_scale=1.0,
        threads=args.threads,
        random_permutation=False,
    )
    q2 = dict(state.distribution(args.column))
    if abs(float(q2.get(args.value, 0.0)) - 1.0) > 1e-12:
        raise SystemExit("FAIL: clamped evidence moved during relaxation")

    print("hard-observation check")
    print("  meanTV =", h["mean_tv"])
    print("  maxTV  =", h["max_tv"])
    print("  clamped_count =", state.clamped_count)
    print("PASS")


if __name__ == "__main__":
    main()
