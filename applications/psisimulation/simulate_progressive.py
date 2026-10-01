#!/usr/bin/env python3
"""Progressive hard observations followed by centered-LDP relaxation.

No empirical data row is required. The simulation starts from the all-missing
row, constructs Psi0, applies each chosen answer as persistent hard evidence,
records the immediate global response, and then performs the requested number
of relaxation sweeps before the next question.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from common import resolve_model, resident_empty_state, snapshot
from plotting import make_gif, save_state_frame


def load_choices(path: Path, default_sweeps: int):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("choice file must contain a JSON list")
    out = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each choice must be a JSON object")
        out.append(
            {
                "column": int(item["column"]),
                "value": str(item["value"]),
                "sweeps": int(item.get("sweeps", default_sweeps)),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gss/gss_2018")
    parser.add_argument(
        "--choices",
        default=str(
            Path(__file__).resolve().parent
            / "examples"
            / "gss2018_choices.json"
        ),
    )
    parser.add_argument("--sweeps", type=int, default=5)
    parser.add_argument("--empirical-n", type=int, default=10)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument(
        "--out",
        default="results/psisimulation/progressive",
    )
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()

    model = resolve_model(args.model, fetch=not args.no_fetch)
    choices = load_choices(Path(args.choices), args.sweeps)

    out = Path(args.out)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"model: {model}")
    print("constructing Psi0 from the all-missing row...", flush=True)

    t0 = time.perf_counter()
    state = resident_empty_state(model)
    print(f"Psi0 ready in {time.perf_counter()-t0:.3f} sec")

    frame_paths = []
    history = []
    observed = []
    previous = None
    frame_no = 0

    psi = snapshot(state)
    frame = frames_dir / f"{frame_no:03d}_empty.png"
    save_state_frame(
        psi,
        previous,
        frame,
        title="Psi0: empty-sample state",
        subtitle="All coordinates missing; Psi_i = phi_i(empty)",
        observed_cols=observed,
        topk=args.topk,
        dpi=args.dpi,
    )
    frame_paths.append(frame)
    previous = psi
    frame_no += 1

    for qnum, choice in enumerate(choices, 1):
        col = choice["column"]
        value = choice["value"]
        sweeps = choice["sweeps"]

        print()
        print(
            f"QUESTION {qnum}: column={col}, value={value!r}",
            flush=True,
        )

        t0 = time.perf_counter()
        h = dict(
            state.hard_observe(
                source=col,
                value=value,
                response_scale=1.0,
                threads=args.threads,
                clamp=True,
            )
        )
        sec = time.perf_counter() - t0

        print(
            f"  hard observation: {sec:.3f} sec  "
            f"meanTV={h['mean_tv']:.6g} "
            f"maxTV={h['max_tv']:.6g}",
            flush=True,
        )

        observed.append(col)
        history.append(
            {
                "question": qnum,
                "phase": "hard_observation",
                "sweep": 0,
                "column": col,
                "value": value,
                "seconds": sec,
                **h,
            }
        )

        psi = snapshot(state)
        frame = frames_dir / f"{frame_no:03d}_q{qnum}_hard.png"
        save_state_frame(
            psi,
            previous,
            frame,
            title=f"Question {qnum}: X[{col}] = {value!r}",
            subtitle=(
                "Immediate centered hard response; "
                f"meanTV={h['mean_tv']:.4g}, maxTV={h['max_tv']:.4g}"
            ),
            observed_cols=observed,
            topk=args.topk,
            dpi=args.dpi,
        )
        frame_paths.append(frame)
        previous = psi
        frame_no += 1

        for sweep in range(1, sweeps + 1):
            print(
                f"  relaxation {sweep}/{sweeps} ... ",
                end="",
                flush=True,
            )
            t0 = time.perf_counter()
            s = dict(
                state.sweep(
                    n=args.empirical_n,
                    event="mode",
                    seed=args.seed,
                    response_scale=1.0,
                    threads=args.threads,
                    random_permutation=False,
                )
            )
            sec = time.perf_counter() - t0
            print(
                f"{sec:.3f} sec "
                f"meanTV={s['mean_tv']:.6g} "
                f"maxTV={s['max_tv']:.6g}",
                flush=True,
            )

            history.append(
                {
                    "question": qnum,
                    "phase": "relaxation",
                    "sweep": sweep,
                    "column": col,
                    "value": value,
                    "seconds": sec,
                    **s,
                }
            )

            psi = snapshot(state)
            frame = (
                frames_dir
                / f"{frame_no:03d}_q{qnum}_sweep{sweep}.png"
            )
            save_state_frame(
                psi,
                previous,
                frame,
                title=f"Question {qnum}: relaxation sweep {sweep}",
                subtitle=(
                    f"meanTV={s['mean_tv']:.4g}, "
                    f"maxTV={s['max_tv']:.4g}, "
                    f"eventTV={s['event_tv']:.4g}"
                ),
                observed_cols=observed,
                topk=args.topk,
                dpi=args.dpi,
            )
            frame_paths.append(frame)
            previous = psi
            frame_no += 1

        print(f"  clamped observations: {state.clamped_count}")

    out.mkdir(parents=True, exist_ok=True)

    hard = list(state.hard_row())
    with (out / "final_hard_row.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        csv.writer(handle).writerow(hard)

    with (out / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    if args.gif:
        gif = out / "psi_dynamics.gif"
        make_gif(frame_paths, gif)
        print(f"written: {gif}")

    print(f"frames:   {frames_dir}")
    print(f"history:  {out / 'history.json'}")
    print(f"hard row: {out / 'final_hard_row.csv'}")


if __name__ == "__main__":
    main()
