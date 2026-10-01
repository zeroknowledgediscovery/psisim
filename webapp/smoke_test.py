#!/usr/bin/env python3
"""End-to-end smoke test of the webapp engine on the installed GSS 2018 model.

Requires the built bindings (predict_distribution, psisim_graph) and the
model (python3 applications/psisimulation/fetch_model.py gss/gss_2018).
"""
from __future__ import annotations

import time

import engine as eng


def main() -> None:
    t0 = time.perf_counter()
    model = eng.Model(fetch=False)
    print(f"model ready in {time.perf_counter() - t0:.1f}s: "
          f"{model.width} columns, {len(model.edges)} learned links")
    assert len(model.variables) == model.width
    assert len(model.layout) == model.width
    assert model.edges, "dependency graph is empty"

    engine = eng.Engine(model)
    s = engine.create(seed=12345)
    col = next(v.column for v in model.variables if v.variable == "wrkstat")

    events = list(engine.answer(s, col, "sample", max_sweeps=1))
    kinds = [e["type"] for e in events]
    print("events:", kinds)
    assert kinds[0] == "answer" and kinds[-1] == "done", kinds
    frames = [e for e in events if e["type"] == "frame"]
    assert [f["phase"] for f in frames] == ["hard_observation", "relaxation"]
    value = events[0]["value"]
    assert s.psi[col] == {value: 1.0}, "answered coordinate must be a point mass"

    # Same seed => same sampled answer (reproducible demonstration).
    s2 = engine.create(seed=12345)
    again = next(iter(engine.answer(s2, col, "sample", max_sweeps=0)))
    assert again["value"] == value and again["u"] == events[0]["u"]

    try:
        engine.answer(s, col, "map")
    except eng.AnswerError:
        pass
    else:
        raise AssertionError("re-answering a clamped item must be refused")

    print(f"OK: {model.variables[col].variable} = {value!r}; "
          f"splash meanTV={frames[0]['summary']['mean_tv']:.4g}, "
          f"sweep meanTV={frames[1]['summary']['mean_tv']:.4g}")


if __name__ == "__main__":
    main()
