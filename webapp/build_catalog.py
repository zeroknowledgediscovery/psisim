#!/usr/bin/env python3
"""Build the survey catalog and per-model item labels for the webapp.

Outputs (committed, so the webapp never needs a DTAG checkout at runtime):

  webapp/assets/catalog.json
      every model of the public DTAG model release with its survey family,
      fieldwork period and the countries it covers;
  webapp/assets/metadata/<family>/<name>.json.gz
      {variable: [short label, question text]} for that model.

Country coverage and periods follow DTAG's own model recommender
(``scripts/dtag_recommend.py``): GSS is the United States by survey year;
Afrobarometer rounds and Eurobarometer waves cover the countries listed in
their own country variable (``configs/model_coverage.json``), with round years
and the Eurobarometer fieldwork registry (``configs/eurodates.csv``) giving the
periods; the pooled WVS7 model covers countries whose centroid snaps within 2
degrees of its coordinate support. Labels come from DTAG's semantic maps
(``maps/``), resolved per model exactly as ``dtag_engine.canonical_map_key``.

  python3 webapp/build_catalog.py --dtag /path/to/DTAG
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
import re
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "applications" / "psisimulation"))
from fetch_model import DEFAULT_RELEASE, load_manifest  # noqa: E402

FAMILY_LABEL = {
    "gss": "General Social Survey",
    "afrobarometer": "Afrobarometer",
    "wvs": "World Values Survey 7",
    "eurobarometer": "Eurobarometer",
}
# Survey labels that name the same country, or are not countries at all.
COUNTRY_MERGE = {
    "bosnia": "bosnia and herzegovina",
    "germany east west": "germany", "germany total": "germany",
    "germany west only": "germany", "united germany": "germany",
    "hong kong sar": "hong kong", "macao sar": "macao", "macau": "macao",
    "taiwan roc": "taiwan", "sa o toma and pra ncipe": "sao tome and principe",
}
COUNTRY_DROP = {"other", "other countries", "not germany", "not united kingdom"}
COUNTRY_NAME = {
    "united states": "United States", "united kingdom": "United Kingdom",
    "cote d ivoire": "Côte d'Ivoire", "sao tome and principe": "São Tomé and Príncipe",
    "congo brazzaville": "Congo (Brazzaville)", "bosnia and herzegovina": "Bosnia and Herzegovina",
    "north macedonia": "North Macedonia", "northern ireland": "Northern Ireland",
    "hong kong": "Hong Kong", "eswatini": "Eswatini",
}
STUDY_TITLE = re.compile(r"^(standard |special |flash )?euro ?barometer\b", re.I)
JUNK = re.compile(r"value label missing count|valid percent|^\s*$", re.I)
MAX_TEXT = 400


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "").replace(" ", " ")).strip()
    return "" if JUNK.search(text) else text[:MAX_TEXT]


def literal(value, default):
    if isinstance(value, (list, dict)):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        return default


def read_map(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def labels_from_map(family: str, rows: list[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in rows:
        var = (r.get("variable") or "").strip()
        if not var:
            continue
        if family == "afrobarometer":
            short, _, long = (r.get("question_text") or "").partition(" | ")
            label, question = clean(short), clean(long) or clean(short)
        elif family == "wvs":
            label = clean(r.get("pdf_short_label"))
            question = clean(r.get("question_text_filled") or r.get("question_text"))
        elif family == "eurobarometer":
            label = clean(r.get("variable_label"))
            question = clean(r.get("question_text_filled") or r.get("question_text"))
            if STUDY_TITLE.match(question):     # the study title, not a question
                question = ""
        else:  # gss
            label = clean(r.get("question_text_filled") or r.get("question_text"))
            question = label
        if label.lower() == var.lower():
            label = ""
        if question.lower() == var.lower():
            question = ""
        if label or question:
            out[var] = [label or question[:120], question or label]
    return out


def gss_labels(maps_root: Path) -> dict[int, dict[str, list[str]]]:
    """GSS labels per year; years without text borrow the nearest year's."""
    per_year: dict[int, dict[str, tuple]] = {}
    for p in sorted((maps_root / "gss").glob("gss_*_map.csv")):
        year = int(re.search(r"(\d{4})", p.name).group(1))
        per_year[year] = {}
        for r in read_map(p):
            var = (r.get("variable") or "").strip()
            text = clean(r.get("question_text_filled") or r.get("question_text"))
            if var and text and text.lower() != var.lower():
                per_year[year][var.lower()] = (text, r.get("map_provenance") == "YEAR_SPECIFIC")
    out: dict[int, dict[str, list[str]]] = {}
    for year in per_year:
        best: dict[str, tuple] = {}
        for y, table in per_year.items():
            for var, (text, specific) in table.items():
                rank = (0 if specific else 1, abs(y - year), -y)
                if var not in best or rank < best[var][0]:
                    best[var] = (rank, text)
        out[year] = {var: [text, text] for var, (_, text) in best.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtag", required=True, help="path to a DTAG checkout")
    ap.add_argument("--release", default=DEFAULT_RELEASE)
    ap.add_argument("--out", default=str(HERE / "assets"))
    args = ap.parse_args()

    dtag = Path(args.dtag).expanduser().resolve()
    sys.path.insert(0, str(dtag / "scripts"))
    # DTAG's pipeline imports an LLM client at module level; the geography
    # helpers used here never call it.
    sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))
    import dtag_recommend as R  # noqa: E402
    import eurobarometer_dates as EBD  # noqa: E402
    import pipeline as core  # noqa: E402
    import pipeline_localized as localized  # noqa: E402

    manifest = load_manifest(args.release)
    models = manifest["models"]
    coverage = json.loads((dtag / "configs" / "model_coverage.json").read_text())["models"]
    waves = {w.za_id: w for w in EBD.load_registry(dtag / "configs" / "eurodates.csv")}
    maps_root = dtag / "maps"
    out = Path(args.out)
    meta_root = out / "metadata"

    countries: dict[str, dict] = {}

    def add_country(label: str, key: str) -> str | None:
        lab = str(label).strip()
        if not lab or lab.isdigit() or lab.lower() in {"none", "nan"}:
            return None
        if lab.upper() in localized.ISO_COUNTRY_NAMES:
            lab = localized.ISO_COUNTRY_NAMES[lab.upper()]
        ckey, _partial = R._label_key(lab)
        ckey = COUNTRY_MERGE.get(ckey, ckey)
        if not ckey or len(ckey) <= 2 or ckey in COUNTRY_DROP:
            return None
        name = COUNTRY_NAME.get(ckey, ckey.title().replace(" And ", " and "))
        countries.setdefault(ckey, {"name": name, "models": set()})["models"].add(key)
        return ckey

    def country_values(rec: dict) -> list[str]:
        vals = []
        for feat in ("country_values", "iso_values"):
            vals += [str(v) for v in literal(rec.get(feat), [])]
        for v in literal(rec.get("nation_features"), {}).values():
            vals += [str(x) for x in v]
        return vals

    gss_years = gss_labels(maps_root)
    wvs_cov = literal(coverage.get("wvs/wvs7_pooled", {}).get("coordinate_values"), {})

    catalog: dict[str, dict] = {}
    for key in sorted(models):
        fam, name = key.split("/", 1)
        rec = coverage.get(key, {})
        entry = {
            "family": fam,
            "family_label": FAMILY_LABEL.get(fam, fam),
            "archive_bytes": models[key].get("size_bytes"),
            "features": int(rec.get("features") or 0) or None,
        }
        labels: dict[str, list[str]] = {}
        if fam == "gss":
            year = int(re.search(r"(\d{4})", name).group(1))
            entry.update(label=f"GSS {year}", period=[year, year], date=None)
            add_country("United States", key)
            labels = gss_years.get(year) or gss_years[min(gss_years, key=lambda y: abs(y - year))]
        elif fam == "afrobarometer":
            rnd = name.lower()
            lo, hi = R.AFRO_ROUND_YEARS.get(rnd, (None, None))
            entry.update(label=f"Afrobarometer {rnd.upper()}", period=[lo, hi] if lo else None, date=None)
            for v in country_values(rec):
                add_country(v, key)
            mp = maps_root / "afromap" / f"afrobarometer_{rnd}_map.csv"
            if not mp.is_file():
                mp = maps_root / "afromap" / f"afrobaromete_{rnd}_map.csv"
            if mp.is_file():
                labels = labels_from_map(fam, read_map(mp))
        elif fam == "eurobarometer":
            za = re.match(r"(ZA\d+)", name, re.I).group(1).upper()
            w = waves.get(za)
            entry.update(
                label=f"Eurobarometer {za}",
                period=[w.start_date.year, w.end_date.year] if w else None,
                date=f"{w.start_date.isoformat()} – {w.end_date.isoformat()}" if w else None,
                auto_select=bool(w.auto_select) if w else False,
                za=za,
            )
            for v in country_values(rec):
                add_country(v, key)
            mp = maps_root / "eurobarometer" / f"{za}_map.csv"
            if mp.is_file():
                labels = labels_from_map(fam, read_map(mp))
        elif fam == "wvs":
            years = sorted(int(y) for y in literal(rec.get("year_values"), []) if str(y).isdigit()) or [2017, 2023]
            entry.update(label="WVS7 (pooled)", period=[years[0], years[-1]], date=None)
            for c in core.list_supported_countries():
                try:
                    _, lon, lat = core.resolve_country_to_coords(c)
                    lon_s = float(core._nearest_allowed_numeric_value(lon, wvs_cov["O1_LONGITUDE"], "O1_LONGITUDE"))
                    lat_s = float(core._nearest_allowed_numeric_value(lat, wvs_cov["O2_LATITUDE"], "O2_LATITUDE"))
                except (ValueError, KeyError):
                    continue
                if max(abs(lon_s - lon), abs(lat_s - lat)) <= 2.0:
                    add_country(c, key)
            mp = maps_root / "wvs7_variable_question_map.csv"
            if mp.is_file():
                labels = labels_from_map(fam, read_map(mp))
        entry["labelled"] = len(labels)
        catalog[key] = entry
        dest = meta_root / fam / f"{name}.json.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(dest, "wt", encoding="utf-8", compresslevel=9) as handle:
            json.dump(labels, handle, ensure_ascii=False, separators=(",", ":"))

    for key, entry in catalog.items():
        entry["countries"] = sorted(c for c, v in countries.items() if key in v["models"])

    try:
        sha = subprocess.run(["git", "-C", str(dtag), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "unknown"
    doc = {
        "schema": 1,
        "release": manifest.get("release", args.release),
        "source": {"dtag_commit": sha, "manifest": manifest.get("base_url")},
        "default_model": "gss/gss_2018",
        "families": FAMILY_LABEL,
        "countries": {k: {"name": v["name"], "models": sorted(v["models"])}
                      for k, v in sorted(countries.items())},
        "models": catalog,
    }
    (out / "catalog.json").write_text(json.dumps(doc, ensure_ascii=False, indent=0))
    print(f"models: {len(catalog)}  countries: {len(countries)}  dtag: {sha}")


if __name__ == "__main__":
    main()
