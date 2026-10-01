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
    model = eng.Model(fetch=False)
    print(f"model ready in {time.perf_counter() - t0:.1f}s: "
          f"{model.width} columns, {len(model.edges)} learned links")
    assert len(model.variables) == model.width
    assert len(model.layout) == model.width
    assert model.edges, "dependency graph is empty"

    engine = eng.Engine(model)
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

    st = done["stats"]
    print(f"OK: {model.variables[col].variable} = {value!r}; "
          f"{st['moved']}/{st['other_topics']} other topics moved, "
          f"{st['moved_001']} by >0.01, mean TV {st['mean_tv']:.4g}; "
          f"residual {done['residual_pending_tv']:.3g}")


if __name__ == "__main__":
    main()
