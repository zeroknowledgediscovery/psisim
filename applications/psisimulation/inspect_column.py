#!/usr/bin/env python3
"""Inspect the empty-state alphabet/distribution for one model coordinate."""
from __future__ import annotations

import argparse

from common import psi0, resolve_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="gss/gss_2018",
        help="public key (gss/gss_2018) or local model directory",
    )
    parser.add_argument("--column", type=int, required=True)
    parser.add_argument("--no-fetch", action="store_true")
    args = parser.parse_args()

    model = resolve_model(args.model, fetch=not args.no_fetch)
    psi = psi0(model)

    if not (0 <= args.column < len(psi)):
        raise SystemExit(
            f"column {args.column} out of range [0,{len(psi)-1}]"
        )

    q = psi[args.column]
    print(f"model:  {model}")
    print(f"column: {args.column}")
    print(f"support: {len(q)}")
    print()

    for value, probability in sorted(
        q.items(), key=lambda kv: kv[1], reverse=True
    ):
        print(f"{probability: .10f}  {value!r}")


if __name__ == "__main__":
    main()
