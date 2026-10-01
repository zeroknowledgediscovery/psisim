#!/usr/bin/env python3
"""Regression tests for the propagation-only (default) PsiSim dynamics.

Requirements checked on the real GSS 2018 LSM:

1. No intervention => no motion: waves from Psi0 leave the state bit-for-bit
   unchanged.
2. A hard observation produces a non-zero first wave, identical to the
   vendored CenteredLdpPsiState.hard_observe.
3. Each later wave is driven only by the deltas newly induced by the previous
   wave (checked against an independent reconstruction from the upstream
   predictor), not by the accumulated change.
4. Clamped coordinates never change.

Run directly (python3 tests/test_propagation_dynamics.py) or with pytest.
Needs bin/predict_distribution*.so, bin/psisim_dynamics*.so and the model
(python3 applications/psisimulation/fetch_model.py gss/gss_2018).
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
sys.path.insert(0, str(ROOT / "applications" / "psisimulation"))

import psisim_dynamics as pdx  # noqa: E402
from common import load_binding, psi0, resolve_model  # noqa: E402

THREADS = 4
KERNEL_TOL = 1e-5  # upstream predict_* round phi to 1e-6 pseudocounts


@functools.lru_cache(maxsize=None)
def model() -> Path:
    return resolve_model("gss/gss_2018", fetch=True)


@functools.lru_cache(maxsize=None)
def initial_psi() -> tuple:
    return tuple(psi0(model()))


def new_state():
    m = model()
    return pdx.PropagationPsiState(
        str(m / "trees" / "binary"), str(m), list(initial_psi())
    )


@functools.lru_cache(maxsize=None)
def columns() -> dict[str, int]:
    """GSS variable name -> native column, from the native source maps."""
    import json

    out = {}
    for shard in (model() / "source_maps" / "json_shards").glob("*.json"):
        for key, value in json.loads(shard.read_text()).items():
            out[value["column_header"]] = int(key)
    return out


def diff(a: dict, b: dict) -> dict:
    keys = set(a) | set(b)
    return {k: a.get(k, 0.0) - b.get(k, 0.0) for k in keys
            if abs(a.get(k, 0.0) - b.get(k, 0.0)) > 1e-18}


def max_abs(d: dict) -> float:
    return max((abs(v) for v in d.values()), default=0.0)


def simplex_project(v: np.ndarray) -> np.ndarray:
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u - (css - 1.0) / np.arange(1, len(u) + 1) > 0)[0][-1]
    theta = (css[rho] - 1.0) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    return w / w.sum()


class KernelReference:
    """K_{j<-i}(.|s) = phi_j(x_empty with X_i = s), via the upstream binding."""

    def __init__(self):
        m = model()
        self.lsm = load_binding()
        self.trees = str(m / "trees" / "binary")
        self.run_dir = str(m)
        self.width = len(initial_psi())
        self.cache: dict[tuple[int, str], list] = {}

    def row(self, source: int, label: str, targets: list[int]) -> list:
        key = (source, label, tuple(sorted(targets)))
        if key not in self.cache:
            row = np.asarray([""] * self.width, dtype=object)
            row[source] = label
            self.cache[key] = self.lsm.predict_distributions(
                self.trees, row.astype(str), raw=True, run_dir=self.run_dir,
                tree_ids=sorted(targets),
            )
        return self.cache[key]

    def wave(self, psi: list[dict], deltas: dict[int, dict], targets_of,
             clamped: set[int], only: set[int]) -> dict[int, dict]:
        """Reference next distributions of the targets in `only`."""
        response: dict[int, dict] = {}
        for i, delta in sorted(deltas.items()):
            tg = [t for t in targets_of(i) if t in only and t not in clamped]
            if not tg:
                continue
            for s, w in delta.items():
                pred = self.row(i, s, sorted(only))
                for t in tg:
                    r = response.setdefault(t, {})
                    for c, q in pred[t].items():
                        r[c] = r.get(c, 0.0) + w * q
        out = {}
        for t, r in response.items():
            keys = sorted(set(psi[t]) | set(r))
            v = np.array([psi[t].get(k, 0.0) + r.get(k, 0.0) for k in keys])
            out[t] = dict(zip(keys, simplex_project(v)))
        return out


# ---------------------------------------------------------------------------


def test_no_intervention_no_motion():
    st = new_state()
    before = st.to_python()
    for _ in range(3):
        s = st.wave(threads=THREADS)
        assert s["sources"] == 0 and s["changed"] == 0, s
        assert s["mean_tv"] == 0.0 and s["max_tv"] == 0.0, s
    after = st.to_python()
    assert after == before, "Psi0 moved without any intervention"
    assert st.pending_count == 0
    # The resident state really is Psi0.
    for a, b in zip(after, initial_psi()):
        assert set(a) == set(b)
        assert max(abs(a[k] - b[k]) for k in a) < 1e-12


def test_first_wave_matches_upstream_hard_observe():
    col = columns()["polviews"]
    value = "extremely liberal"

    st = new_state()
    s0 = st.hard_observe(col, value, threads=THREADS)
    ours = st.to_python()

    m = model()
    up = load_binding().centered_ldp_state_from_psi(
        str(m / "trees" / "binary"), list(initial_psi()), raw=True,
        run_dir=str(m))
    h = dict(up.hard_observe(source=col, value=value, threads=THREADS,
                             clamp=True))
    theirs = [dict(q) for q in up.to_python()]

    assert s0["changed"] > 1 and s0["mean_tv"] > 0.0, s0
    assert ours[col] == {value: 1.0}
    worst = max(max_abs(diff(a, b)) for a, b in zip(ours, theirs))
    assert worst == 0.0, f"wave 0 differs from upstream hard_observe by {worst}"
    assert abs(s0["mean_tv"] - h["mean_tv"]) < 1e-15
    # Same dependency structure as the vendored centered state.
    for i in range(st.size):
        assert len(st.dependency_targets(i)) == up.dependency_count(i)


def test_waves_propagate_only_new_deltas():
    col = columns()["polviews"]
    st = new_state()
    clamped = {col}
    ref = KernelReference()

    st.hard_observe(col, "extremely liberal", threads=THREADS)
    origin = list(initial_psi())
    for k in range(1, 3):
        before = st.to_python()
        pending = {int(i): dict(d) for i, d in st.pending().items()}
        assert pending, "a non-trivial observation must leave deltas to propagate"
        assert col not in pending, "the observed coordinate is never re-propagated"

        # The independent reconstruction costs ~3 s per upstream call, so it
        # is run on wave 1 for a few targets that receive responses from
        # several sources at once; every wave checks (a).
        check = k == 1
        if check:
            incoming: dict[int, list[int]] = {}
            for i in pending:
                for t in st.dependency_targets(i):
                    if t not in clamped:
                        incoming.setdefault(t, []).append(i)
            # Targets that are also direct dependents of the observed item:
            # there, re-sending the accumulated change would double-count
            # the observation's own perturbation (checked in (c)).
            direct = set(st.dependency_targets(col))
            multi = sorted((len(v), sum(len(pending[i]) for i in v), t)
                           for t, v in incoming.items()
                           if len(v) >= 2 and t in direct)
            only = {t for _, _, t in multi[:3]}
            assert only, "expected targets hit by several sources"
            expected = ref.wave(before, pending, st.dependency_targets,
                                clamped, only)
        s = st.wave(threads=THREADS)
        after = st.to_python()
        assert s["sources"] == len(pending)

        # (a) the next wave's deltas are exactly this wave's induced change
        new_pending = {int(i): dict(d) for i, d in st.pending().items()}
        for i in range(st.size):
            d = diff(after[i], before[i])
            got = new_pending.get(i, {})
            assert max_abs(diff(got, d)) < 1e-15, f"pending delta wrong at {i}"

        if not check:
            continue

        # (b) the wave equals the independent reconstruction from new deltas
        reachable = {t for i in pending for t in st.dependency_targets(i)}
        moved = {i for i in range(st.size) if diff(after[i], before[i])}
        assert moved <= reachable, "a coordinate moved without a source"
        assert set(expected) == only
        worst = max(max_abs(diff(after[t], q)) for t, q in expected.items())
        assert worst < KERNEL_TOL, f"wave {k} deviates from reference by {worst}"

        # (c) propagating the accumulated change instead would differ
        accumulated = {i: diff(before[i], origin[i]) for i in pending}
        accumulated[col] = diff(before[col], origin[col])
        wrong = ref.wave(before, accumulated, st.dependency_targets, clamped,
                         only)
        gap = max(max_abs(diff(wrong[t], expected.get(t, before[t])))
                  for t in wrong)
        assert gap > 100 * KERNEL_TOL, "test cannot distinguish new vs accumulated"


def test_clamped_coordinates_never_change():
    cols = columns()
    first = cols["polviews"]
    st = new_state()
    st.hard_observe(first, "extremely liberal", threads=THREADS)

    # A second answer whose dependents include the first, clamped item.
    second = next(i for i in range(st.size)
                  if i != first and first in st.dependency_targets(i))
    label = min(st.distribution(second), key=st.distribution(second).get)
    st.hard_observe(second, label, threads=THREADS)

    for _ in range(4):
        st.wave(threads=THREADS)
        assert st.distribution(first) == {"extremely liberal": 1.0}
        assert st.distribution(second) == {label: 1.0}
        assert first not in st.pending() and second not in st.pending()
    assert st.clamped() == {first: "extremely liberal", second: label}

    try:
        st.hard_observe(first, "moderate", threads=THREADS)
    except ValueError:
        pass
    else:
        raise AssertionError("re-observing a clamped coordinate must fail")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}", flush=True)
    print(f"{len(tests)} passed")
