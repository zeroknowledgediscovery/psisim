#!/usr/bin/env python3
"""End-to-end smoke test of the webapp engine on the installed GSS 2018 model.

Requires the built bindings (predict_distribution, psisim_dynamics) and the
model (python3 applications/psisimulation/fetch_model.py gss/gss_2018).
"""
from __future__ import annotations

import time

import engine as eng


def main() -> None:
    t0 = time.perf_counter()
    registry = eng.ModelRegistry()
    engine = eng.Engine(registry)
    model = engine.default
    links = sum(len(t) for t in model.targets)
    print(f"model ready in {time.perf_counter() - t0:.1f}s: "
          f"{model.width} columns, {links} learned links")
    assert len(model.variables) == model.width
    assert links, "dependency graph is empty"
    assert eng.CATALOG["models"] and eng.CATALOG["countries"], "catalog missing"
    s = engine.create(seed=12345)
    assert s.dynamics == "propagation"
    assert s.psi == model.psi0 or all(
        abs(a[k] - b[k]) < 1e-12 for a, b in zip(s.psi, model.psi0) for k in a
    )
    col = next(v.column for v in model.variables if v.variable == "polviews")

    events = list(engine.answer(s, col, max_steps=3))
    kinds = [e["type"] for e in events]
    print("events:", kinds)
    assert kinds[0] == "answer" and kinds[-1] == "done", kinds
    frames = [e for e in events if e["type"] == "frame"]
    assert [f["phase"] for f in frames] == ["hard_observation", "wave", "wave", "wave"]
    value = events[0]["value"]
    assert s.psi[col] == {value: 1.0}, "answered coordinate must be a point mass"
    done = events[-1]
    assert done["stats"]["moved"] > 100, done["stats"]
    assert s.state.pending_count == 0, "an answer's wave train must not leak"

    # The webapp default: one update per question (hard observation only).
    s3 = engine.create(seed=7)
    one = list(engine.answer(s3, col))
    assert [e["phase"] for e in one if e["type"] == "frame"] == ["hard_observation"]
    assert one[-1]["stats"]["moved"] > 0, one[-1]["stats"]

    # Same seed => same sampled answer (reproducible demonstration).
    s2 = engine.create(seed=12345)
    again = next(iter(engine.answer(s2, col, max_steps=0)))
    assert again["value"] == value and again["u"] == events[0]["u"]

    try:
        engine.answer(s, col)
    except eng.AnswerError:
        pass
    else:
        raise AssertionError("re-answering a clamped item must be refused")

    # Another survey: downloaded from the public release on demand.
    other = "eurobarometer/ZA7561_v1-0-0"
    registry.start(other)
    while registry.status(other)["state"] not in ("ready", "error"):
        time.sleep(0.5)
    assert registry.status(other)["state"] == "ready", registry.status(other)
    s4 = engine.create(seed=1, model_key=other)
    m4 = s4.model
    assert m4.key == other and m4.width != model.width
    assert sum(1 for v in m4.variables if v.label) > m4.width // 2, "labels missing"
    col4 = max((v for v in m4.variables if v.learned), key=lambda v: len(m4.targets[v.column])).column
    ev4 = list(engine.answer(s4, col4))
    assert ev4[-1]["type"] == "done" and ev4[-1]["stats"]["moved"] > 0, ev4[-1]
    print(f"OK: {other}: {m4.width} items; asking {m4.variables[col4].variable} "
          f"moved {ev4[-1]['stats']['moved']} distributions")

    st = done["stats"]
    print(f"OK: {model.variables[col].variable} = {value!r}; "
          f"{st['moved']}/{st['other_topics']} other topics moved, "
          f"{st['moved_001']} by >0.01, mean TV {st['mean_tv']:.4g}; "
          f"residual {done['residual_pending_tv']:.3g}")


if __name__ == "__main__":
    main()
