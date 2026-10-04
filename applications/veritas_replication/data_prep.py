#!/usr/bin/env python3
"""Prepare public GSS-2018 respondents for the Veritas replication.

Inputs (public, reachable from this environment):
  * gssr's cumulative GSS file ``gss_all.rda`` (kjhealy/gssr on GitHub), read
    once into ``gss2018_all.pkl`` (year == 2018 rows) and
    ``gss_value_labels.pkl`` (value labels) by ``extract_gssr.py``.
  * public DTAG native models (``fetch_model.py gss/gss_2018`` etc.).

Outputs:
  * ``surrogate_2018.pkl``: every model item, public labels as strings, used to
    train/evaluate the surrogate LSM under a true 80/20 split.
  * ``native_<model>.pkl``: rows encoded with the native model's own category
    strings, for every item whose public->model label mapping passes a
    marginal-agreement audit; other items are left missing. A JSON audit is
    written next to it.
"""
from __future__ import annotations

import argparse
import difflib
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

NUM_RE = re.compile(r"-?\d+(\.\d+)?")


def norm(s: str) -> str:
    s = str(s).lower().replace("'", "").replace("’", "")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def sim(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if a == b:
        return 1.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    jac = len(ta & tb) / max(1, len(ta | tb))
    # Abbreviation: every model token is a prefix of some public token.
    pre = 0.0
    if tb and all(any(x.startswith(t) or t.startswith(x) for x in ta) for t in tb):
        pre = 0.85
    return max(r, jac, pre)


def model_meta(model: Path):
    m = json.loads((model / "source_maps" / "json_shards" / "0.json").read_text())
    names = [m[str(i)]["column_header"] for i in range(len(m))]
    cats = [m[str(i)]["column_strings_map"][1:] for i in range(len(m))]
    return names, cats


def model_psi0(model: Path, width: int):
    import predict_distribution as P  # native; see README for path setup

    ev = P.PersistenceEvaluator(str(model / "trees" / "binary"), run_dir=str(model))
    return ev.distributions([""] * width)


def public_strings(col: pd.Series, labels: dict) -> pd.Series:
    """Public numeric codes -> label strings (or the number as a string)."""
    lab = {k: v for k, v in labels.items() if k == k}
    out = []
    for v in col.to_numpy():
        if v != v:
            out.append("")
        elif v in lab:
            out.append(lab[v])
        elif float(v).is_integer():
            out.append(str(int(v)))
        else:
            out.append(repr(float(v)))
    return pd.Series(out, index=col.index)


def map_variable(pub: pd.Series, cats: list[str], psi0: dict, tol: float):
    """Map one public column (strings) onto a native model's category strings.

    Returns (mapping dict, audit dict). mapping is None when rejected.
    """
    obs = pub[pub != ""]
    audit = {"n_obs": int(len(obs))}
    if len(obs) == 0:
        audit["reason"] = "no public data"
        return None, audit
    pfreq = obs.value_counts(normalize=True)
    pcats = list(pfreq.index)
    if all(NUM_RE.fullmatch(c) for c in cats):
        mapping = {c: c for c in pcats if c in cats}
        audit["kind"] = "numeric"
    elif all(re.fullmatch("[a-e]", c) for c in cats):
        audit["reason"] = "letter-binned in model (bin edges unknown)"
        return None, audit
    else:
        audit["kind"] = "text"
        S = np.array([[sim(p, c) for c in cats] for p in pcats])
        F = np.array([[abs(pfreq[p] - psi0.get(c, 0.0)) for c in cats] for p in pcats])
        r, c = linear_sum_assignment(-(S - 1.0 * F))
        mapping = {}
        sims = []
        for i, j in zip(r, c):
            if S[i, j] >= 0.34:
                mapping[pcats[i]] = cats[j]
                sims.append(S[i, j])
        audit["mean_sim"] = float(np.mean(sims)) if sims else 0.0
    covered = float(obs.isin(list(mapping)).mean())
    mapped = obs[obs.isin(list(mapping))].map(mapping)
    mf = mapped.value_counts(normalize=True)
    keys = set(mf.index) | set(psi0)
    # Model may hold explicit 'na'/'refused' categories that public data codes
    # as missing: compare after renormalising psi0 over mapped categories.
    tgt = {k: psi0.get(k, 0.0) for k in keys if k in set(mapping.values())}
    z = sum(tgt.values()) or 1.0
    tvd = 0.5 * sum(abs(mf.get(k, 0.0) - tgt.get(k, 0.0) / z) for k in keys)
    audit.update(coverage=covered, tvd=float(tvd), n_map=len(mapping),
                 n_model_cats=len(cats), n_public_cats=len(pcats))
    ok = covered >= 0.97 and tvd <= tol
    if audit.get("kind") == "text":
        ok = ok and audit["mean_sim"] >= 0.5
    if not ok:
        audit["reason"] = "failed audit"
        return None, audit
    return mapping, audit


def bin_numeric(train: pd.Series, full: pd.Series, k: int = 5) -> pd.Series:
    """Quantile-bin a many-valued numeric column with edges from training rows."""
    tr = train.dropna()
    qs = np.unique(np.quantile(tr, np.linspace(0, 1, k + 1)[1:-1]))
    out = pd.Series([""] * len(full), index=full.index, dtype=object)
    ok = full.notna()
    out[ok] = [chr(ord("a") + int(b)) for b in np.searchsorted(qs, full[ok], side="right")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="dir with gss2018_all.pkl, gss_value_labels.pkl")
    ap.add_argument("--model-root", default=str(Path("~/.cache/dtag/models").expanduser()))
    ap.add_argument("--native-models", default="gss/gss_2018:0.05,gss/gss_2016:0.10")
    args = ap.parse_args()
    work = Path(args.work)
    d = pd.read_pickle(work / "gss2018_all.pkl").sort_values("id").reset_index(drop=True)
    labels = pickle.loads((work / "gss_value_labels.pkl").read_bytes())

    for spec in args.native_models.split(","):
        key, tol = spec.split(":")
        model = Path(args.model_root) / key
        names, cats = model_meta(model)
        psi0 = model_psi0(model, len(names))
        rows = pd.DataFrame("", index=d.index, columns=names, dtype=object)
        audits = {}
        for i, n in enumerate(names):
            col = n.lower()
            if col not in d.columns:
                audits[n] = {"reason": "absent from public file"}
                continue
            pub = public_strings(d[col], labels.get(col, {}))
            mapping, audit = map_variable(pub, cats[i], psi0[i], float(tol))
            audits[n] = audit
            if mapping is not None:
                audit["mapping"] = mapping
                rows[n] = pub.map(lambda s: mapping.get(s, ""))
        acc = [n for n, a in audits.items() if "mapping" in a]
        tag = key.replace("/", "_")
        rows.to_pickle(work / f"native_{tag}.pkl")
        (work / f"native_{tag}_audit.json").write_text(json.dumps(
            {"model": key, "tol": float(tol), "n_items": len(names),
             "n_accepted": len(acc), "accepted": acc, "per_item": audits},
            indent=1, default=str))
        cnt = {}
        for a in audits.values():
            cnt[a.get("reason", "accepted")] = cnt.get(a.get("reason", "accepted"), 0) + 1
        print(key, cnt, "observed cells/row (mean):", float((rows != "").sum(1).mean()))

    # Surrogate frame: every 2018-model item with public data, label strings.
    names, _ = model_meta(Path(args.model_root) / "gss/gss_2018")
    cols = [n.lower() for n in names if n.lower() in d.columns]
    raw = d[cols].copy()
    raw.attrs["labels"] = {c: labels.get(c, {}) for c in cols}
    raw["__id"] = d["id"].astype(int)
    raw.to_pickle(work / "surrogate_2018_raw.pkl")
    print("surrogate raw frame", raw.shape)


if __name__ == "__main__":
    main()
