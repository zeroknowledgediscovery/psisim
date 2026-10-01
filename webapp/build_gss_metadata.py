#!/usr/bin/env python3
"""Build the PsiSim-owned GSS 2018 variable metadata asset for the webapp.

The native model is authoritative for column order, variable names and the
legal response categories (from ``source_maps/``). Human-readable question
labels are borrowed from DTAG's semantic GSS maps (``maps/gss/gss_*_map.csv``)
by GSS variable name. DTAG's own ``gss_2018_map.csv`` currently carries only
variable names, so labels are taken from the nearest survey year whose map has
real text, preferring year-specific codebook entries over donor entries.

The output is committed to ``webapp/assets/gss/gss_2018_map.csv`` so the
webapp never needs a live DTAG checkout. Re-run only to refresh labels:

  python3 webapp/build_gss_metadata.py \\
      --model gss/gss_2018 \\
      --dtag /path/to/dtag \\
      --out webapp/assets/gss/gss_2018_map.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "applications" / "psisimulation"))

from common import resolve_model, tree_ids  # noqa: E402

MODEL_YEAR = 2018
FIELDS = [
    "column",
    "variable",
    "label",
    "question_text",
    "categories",
    "label_source_year",
    "label_provenance",
]


def load_source_maps(model: Path) -> dict[int, dict]:
    """Return {column: {"column_header": str, "column_strings_map": [...]}}."""
    out: dict[int, dict] = {}
    for shard in sorted((model / "source_maps" / "json_shards").glob("*.json")):
        raw = json.loads(shard.read_text(encoding="utf-8"))
        for key, value in raw.items():
            out[int(key)] = value
    if not out:
        raise RuntimeError(f"no source-map shards under {model / 'source_maps'}")
    return out


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace(" ", " ")).strip()


def load_dtag_labels(dtag: Path) -> dict[str, dict]:
    """Pick one readable label per lower-cased GSS variable name."""
    best: dict[str, tuple[tuple, dict]] = {}
    for path in sorted((dtag / "maps" / "gss").glob("gss_*_map.csv")):
        match = re.fullmatch(r"gss_(\d{4})_map\.csv", path.name)
        if not match:
            continue
        map_year = int(match.group(1))
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                var = norm(row.get("variable", ""))
                text = norm(row.get("question_text_filled") or row.get("question_text"))
                if not var or not text or text.lower() == var.lower():
                    continue
                year_specific = row.get("map_provenance") == "YEAR_SPECIFIC"
                rank = (
                    0 if year_specific else 1,
                    abs(map_year - MODEL_YEAR),
                    -map_year,
                )
                key = var.lower()
                if key not in best or rank < best[key][0]:
                    best[key] = (
                        rank,
                        {
                            "text": text,
                            "year": map_year,
                            "provenance": (
                                f"dtag:maps/gss/{path.name}"
                                f"#{row.get('map_provenance', '')}"
                            ),
                        },
                    )
    return {k: v for k, (_, v) in best.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gss/gss_2018")
    parser.add_argument("--dtag", required=True, help="path to a DTAG checkout")
    parser.add_argument(
        "--out",
        default=str(HERE / "assets" / "gss" / "gss_2018_map.csv"),
    )
    args = parser.parse_args()

    model = resolve_model(args.model, fetch=True)
    dtag = Path(args.dtag).expanduser().resolve()
    maps = load_source_maps(model)
    labels = load_dtag_labels(dtag)
    learned = set(tree_ids(model))
    width = max(learned) + 1

    rows = []
    for col in range(width):
        entry = maps.get(col, {})
        var = norm(entry.get("column_header", f"col{col}"))
        cats = [c for c in entry.get("column_strings_map", []) if c != ""]
        lab = labels.get(var.lower())
        rows.append(
            {
                "column": col,
                "variable": var,
                "label": lab["text"] if lab else "",
                "question_text": lab["text"] if lab else "",
                # JSON keeps categories containing commas/pipes unambiguous.
                "categories": json.dumps(cats, ensure_ascii=False),
                "label_source_year": lab["year"] if lab else "",
                "label_provenance": lab["provenance"] if lab else "none",
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    try:
        dtag_sha = subprocess.run(
            ["git", "-C", str(dtag), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        dtag_sha = "unknown"

    labelled = sum(1 for r in rows if r["label"])
    print(f"model:    {model}")
    print(f"dtag:     {dtag} @ {dtag_sha}")
    print(f"columns:  {len(rows)}  labelled: {labelled}")
    print(f"written:  {out}")


if __name__ == "__main__":
    main()
