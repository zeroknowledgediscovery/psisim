#!/usr/bin/env python3
"""Extract GSS-2018 rows and value labels from gssr's ``data/gss_all.rda``.

    git clone --depth 1 https://github.com/kjhealy/gssr
    python3 extract_gssr.py gssr/data/gss_all.rda WORKDIR

Writes WORKDIR/gss2018_all.pkl and WORKDIR/gss_value_labels.pkl.
"""
import pickle
import sys
from pathlib import Path

import pyreadr
import rdata


def dec(v):
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8")
        except UnicodeDecodeError:
            return v.decode("latin-1")
    return v


def walk_attrs(a):
    out = {}
    while a is not None and a.info.type.name != "NILVALUE":
        tag = a.tag
        while tag is not None and tag.info.type.name == "REF":
            tag = tag.referenced_object
        out[dec(tag.value.value if hasattr(tag.value, "value") else tag.value)] = a.value[0]
        a = a.value[1]
    return out


def main(rda: str, work: str):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    df = list(pyreadr.read_r(rda).values())[0]
    df[df.year == 2018].to_pickle(work / "gss2018_all.pkl")

    obj = rdata.parser.parse_file(rda).object.value[0]
    cols = [dec(z.value) for z in walk_attrs(obj.attributes)["names"].value]
    labels = {}
    for i, c in enumerate(cols):
        v = obj.value[i]
        if v.attributes is None:
            continue
        va = walk_attrs(v.attributes)
        if "labels" not in va:
            continue
        la = va["labels"]
        names = [dec(z.value) for z in walk_attrs(la.attributes)["names"].value]
        labels[c] = dict(zip([float(x) for x in la.value], names))
    (work / "gss_value_labels.pkl").write_bytes(pickle.dumps(labels))


if __name__ == "__main__":
    main(*sys.argv[1:3])
