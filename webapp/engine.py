"""Native PsiSim simulation engine behind the webapp.

All mathematics runs in the vendored C++ runtime through the existing
``applications/psisimulation`` helpers. This module only:

* loads the model, variable metadata and the learned dependency graph once;
* keeps one resident ``CenteredLdpPsiState`` per browser session;
* turns ``hard_observe`` + ``sweep`` calls into a stream of snapshot frames.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import secrets
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "applications" / "psisimulation"))

from common import (  # noqa: E402
    BIN,
    load_binding,
    psi0,
    resolve_model,
    snapshot,
    tree_ids,
)

MODEL_KEY = os.environ.get("PSISIM_MODEL", "gss/gss_2018")
METADATA_CSV = Path(
    os.environ.get(
        "PSISIM_METADATA", HERE / "assets" / "gss" / "gss_2018_map.csv"
    )
)
CACHE_DIR = Path(
    os.environ.get("PSISIM_WEBAPP_CACHE", "~/.cache/psisim/webapp")
).expanduser()

# Relaxation defaults from webapp_instruction.md section 8.
DEFAULT_EMPIRICAL_N = 10
DEFAULT_MAX_SWEEPS = 5
DEFAULT_TOL = 1e-3
MAX_SWEEPS_LIMIT = 10
RESPONSE_SCALE = 1.0

# Ignore floating-point dust when deciding which coordinates changed.
CHANGE_EPS = 1e-9


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


THREADS = env_int("PSISIM_THREADS", os.cpu_count() or 1)
MAX_SESSIONS = max(1, env_int("PSISIM_MAX_SESSIONS", 4))
SESSION_TTL = env_int("PSISIM_SESSION_TTL", 1800)


def load_graph_binding():
    if str(BIN) not in sys.path:
        sys.path.insert(0, str(BIN))
    try:
        import psisim_graph
    except ImportError as exc:
        raise RuntimeError(
            "Could not import psisim_graph. Build it with:\n"
            "  cmake --build build --target predict_distribution psisim_graph"
        ) from exc
    return psisim_graph


def tv_array(a: list[dict], b: list[dict]) -> np.ndarray:
    out = np.zeros(len(a), dtype=float)
    for i, (x, y) in enumerate(zip(a, b)):
        if x is y:
            continue
        keys = set(x) | set(y)
        out[i] = 0.5 * sum(abs(x.get(k, 0.0) - y.get(k, 0.0)) for k in keys)
    return out


# ---------------------------------------------------------------------------
# Graph layout (computed once per model, cached on disk)
# ---------------------------------------------------------------------------


def force_layout(
    n: int, edges: list[tuple[int, int]], iterations: int = 300, seed: int = 7
) -> np.ndarray:
    """Readable "pond" layout of the undirected learned dependency graph.

    A degree-normalised force layout places linked items near each other;
    radii are then rank-equalised so the disc is evenly filled (densely
    connected items towards the centre, sparsely connected ones outside).
    Positions carry no model meaning: screen distance is not a metric of the
    LSM and not a propagation time.
    """
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=np.float32)
    for s, t in edges:
        adj[s, t] = adj[t, s] = 1.0
    degree = adj.sum(1)
    connected = degree > 0
    pos = np.zeros((n, 2), dtype=np.float32)

    idx = np.flatnonzero(connected)
    m = len(idx)
    if m:
        d = degree[idx]
        w = adj[np.ix_(idx, idx)] / np.sqrt(np.outer(d, d))
        p = rng.normal(size=(m, 2)).astype(np.float32)
        temp = 1.0
        for _ in range(iterations):
            delta = p[:, None, :] - p[None, :, :]
            dist = np.sqrt((delta**2).sum(-1)) + np.float32(1e-3)
            rep = 1.0 / dist**1.5
            att = w * np.log1p(dist) / dist * np.float32(0.35 * m)
            force = ((rep - att)[:, :, None] * delta).sum(1)
            force -= np.float32(0.02 * m) * p
            length = np.sqrt((force**2).sum(-1, keepdims=True)) + np.float32(1e-9)
            p += force / length * np.minimum(length, np.float32(temp))
            temp = max(0.005, temp * 0.99)
        p -= np.median(p, 0)
        radius = np.sqrt((p**2).sum(1))
        angle = np.arctan2(p[:, 1], p[:, 0])
        rank = np.argsort(np.argsort(radius))
        r_eq = np.sqrt((rank + 0.5) / m)
        pos[idx, 0] = r_eq * np.cos(angle)
        pos[idx, 1] = r_eq * np.sin(angle)

    # Coordinates without learned links sit on an outer ring.
    lonely = np.flatnonzero(~connected)
    if len(lonely):
        angle = np.linspace(0, 2 * np.pi, len(lonely), endpoint=False)
        pos[lonely, 0] = 1.15 * np.cos(angle)
        pos[lonely, 1] = 1.15 * np.sin(angle)
    return pos


# ---------------------------------------------------------------------------
# Model (shared, read-only)
# ---------------------------------------------------------------------------


@dataclass
class Variable:
    column: int
    variable: str
    label: str
    question_text: str
    categories: list[str]
    label_source_year: str
    learned: bool


class Model:
    def __init__(self, key: str = MODEL_KEY, fetch: bool = True):
        t0 = time.perf_counter()
        self.key = key
        self.path = resolve_model(key, fetch=fetch)
        self.trees_dir = self.path / "trees" / "binary"
        self.tree_ids = tree_ids(self.path)
        self.width = max(self.tree_ids) + 1
        self.lsm = load_binding()

        self.variables = self._load_metadata()

        # Psi0 = (phi_i(x_empty))_i, computed natively once and shared.
        self.psi0 = psi0(self.path)
        if len(self.psi0) != self.width:
            raise RuntimeError("Psi0 width does not match model width")
        self._align_categories(self.psi0)

        graph = load_graph_binding()
        used = graph.used_columns(str(self.trees_dir), self.tree_ids)
        learned = set(self.tree_ids)
        self.edges: list[tuple[int, int]] = sorted(
            (s, t)
            for t, sources in used.items()
            for s in sources
            if s != t and 0 <= s < self.width and s in learned
        )
        self.targets = [[] for _ in range(self.width)]
        self.sources = [[] for _ in range(self.width)]
        for s, t in self.edges:
            self.targets[s].append(t)
            self.sources[t].append(s)

        self.layout = self._layout()
        self.load_seconds = time.perf_counter() - t0

    def _load_metadata(self) -> list[Variable]:
        rows: dict[int, dict] = {}
        if METADATA_CSV.is_file():
            with METADATA_CSV.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    rows[int(row["column"])] = row
        learned = set(self.tree_ids)
        out = []
        for col in range(self.width):
            row = rows.get(col, {})
            out.append(
                Variable(
                    column=col,
                    variable=row.get("variable") or f"col{col}",
                    label=row.get("label", ""),
                    question_text=row.get("question_text", ""),
                    categories=json.loads(row.get("categories") or "[]"),
                    label_source_year=row.get("label_source_year", ""),
                    learned=col in learned,
                )
            )
        return out

    def _align_categories(self, psi: list[dict]) -> None:
        """Ensure every symbol the native state can emit has a slot."""
        for var, q in zip(self.variables, psi):
            for sym in q:
                if sym not in var.categories:
                    var.categories.append(sym)

    def _layout(self) -> np.ndarray:
        digest = hashlib.sha256(
            json.dumps([self.width, self.edges]).encode()
        ).hexdigest()[:16]
        cache = CACHE_DIR / f"layout_v2_{self.key.replace('/', '_')}_{digest}.json"
        if cache.is_file():
            try:
                return np.asarray(json.loads(cache.read_text()), dtype=float)
            except (OSError, ValueError):
                pass
        pos = force_layout(self.width, self.edges).astype(float)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(np.round(pos, 5).tolist()))
        except OSError:
            pass
        return pos

    # -- encoding ---------------------------------------------------------

    def encode(self, psi: list[dict]) -> list[list[float]]:
        """Distributions as probability arrays aligned to category order."""
        return [
            [round(float(q.get(c, 0.0)), 6) for c in var.categories]
            for var, q in zip(self.variables, psi)
        ]

    def encode_one(self, col: int, q: dict) -> list[float]:
        return [
            round(float(q.get(c, 0.0)), 6)
            for c in self.variables[col].categories
        ]

    def describe(self) -> dict:
        return {
            "model": self.key,
            "width": self.width,
            "variables": [
                {
                    "column": v.column,
                    "variable": v.variable,
                    "label": v.label,
                    "question_text": v.question_text,
                    "categories": v.categories,
                    "label_source_year": v.label_source_year,
                    "learned": v.learned,
                    "out_degree": len(self.targets[v.column]),
                    "in_degree": len(self.sources[v.column]),
                }
                for v in self.variables
            ],
            "edges": [x for e in self.edges for x in e],
            "layout": np.round(self.layout, 4).ravel().tolist(),
            "psi0": self.encode(self.psi0),
            "defaults": {
                "empirical_n": DEFAULT_EMPIRICAL_N,
                "max_sweeps": DEFAULT_MAX_SWEEPS,
                "tol": DEFAULT_TOL,
                "max_sweeps_limit": MAX_SWEEPS_LIMIT,
                "response_scale": RESPONSE_SCALE,
                "event": "mode",
                "random_permutation": False,
            },
        }

    def new_state(self):
        """A resident centered-LDP state initialised at the shared Psi0."""
        return self.lsm.centered_ldp_state_from_psi(
            str(self.trees_dir),
            self.psi0,
            raw=True,
            run_dir=str(self.path),
        )


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class AnswerError(ValueError):
    pass


@dataclass
class Session:
    id: str
    seed: int
    state: object
    psi: list[dict]
    rng: np.random.Generator
    lock: threading.Lock = field(default_factory=threading.Lock)
    observed: list[dict] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_used = time.time()


class Engine:
    def __init__(self, model: Model):
        self.model = model
        self.sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._spare = None
        self._spare_lock = threading.Lock()
        self._spare_thread: threading.Thread | None = None
        self.refill_spare()

    # -- warm spare state: building a resident state loads all trees -------

    def refill_spare(self) -> None:
        def build():
            state = self.model.new_state()
            with self._spare_lock:
                if self._spare is None:
                    self._spare = state

        with self._spare_lock:
            busy = self._spare_thread is not None and self._spare_thread.is_alive()
            if self._spare is not None or busy:
                return
            self._spare_thread = threading.Thread(target=build, daemon=True)
            self._spare_thread.start()

    def _take_state(self):
        with self._spare_lock:
            state, self._spare = self._spare, None
        if state is None:
            state = self.model.new_state()
        self.refill_spare()
        return state

    # -- session bookkeeping ----------------------------------------------

    def _evict(self) -> None:
        now = time.time()
        for sid, s in list(self.sessions.items()):
            if now - s.last_used > SESSION_TTL and not s.lock.locked():
                del self.sessions[sid]
        while len(self.sessions) >= MAX_SESSIONS:
            idle = [s for s in self.sessions.values() if not s.lock.locked()]
            if not idle:
                raise RuntimeError("all simulation slots are busy; try again shortly")
            oldest = min(idle, key=lambda s: s.last_used)
            del self.sessions[oldest.id]

    def create(self, seed: int | None = None) -> Session:
        if seed is None:
            seed = secrets.randbelow(2**31 - 1) + 1
        seed = int(seed) % (2**63)
        with self._lock:
            self._evict()
        state = self._take_state()
        session = Session(
            id=secrets.token_urlsafe(12),
            seed=seed,
            state=state,
            psi=snapshot(state),
            rng=np.random.default_rng(seed),
        )
        with self._lock:
            self.sessions[session.id] = session
        return session

    def get(self, sid: str) -> Session | None:
        with self._lock:
            session = self.sessions.get(sid)
        if session:
            session.touch()
        return session

    def drop(self, sid: str) -> None:
        with self._lock:
            self.sessions.pop(sid, None)

    def session_view(self, s: Session) -> dict:
        return {
            "session_id": s.id,
            "seed": s.seed,
            "psi": self.model.encode(s.psi),
            "observed": s.observed,
            "clamped_count": int(s.state.clamped_count),
        }

    def export(self, s: Session) -> dict:
        return {
            "model": self.model.key,
            "seed": s.seed,
            "observed": s.observed,
            "log": s.log,
            "hard_row": list(s.state.hard_row()),
        }

    # -- answering ---------------------------------------------------------

    def _choose_value(
        self, s: Session, col: int, mode: str, value: str | None
    ) -> tuple[str, dict]:
        var = self.model.variables[col]
        q = s.psi[col]
        cats = [c for c in var.categories if q.get(c, 0.0) > 0.0] or list(q)
        probs = np.asarray([q.get(c, 0.0) for c in cats], dtype=float)
        if mode == "sample":
            total = probs.sum()
            if total <= 0:
                raise AnswerError("current distribution has no mass")
            cdf = np.cumsum(probs / total)
            u = float(s.rng.random())
            k = int(min(np.searchsorted(cdf, u, side="right"), len(cats) - 1))
            return cats[k], {"u": u}
        if mode == "map":
            k = max(range(len(cats)), key=lambda j: (probs[j], cats[j]))
            return cats[k], {}
        if mode == "choose":
            if value is None or value not in var.categories:
                raise AnswerError(f"{value!r} is not a legal response for {var.variable}")
            return value, {}
        raise AnswerError(f"unknown mode {mode!r}")

    def answer(
        self,
        s: Session,
        col: int,
        mode: str,
        value: str | None = None,
        max_sweeps: int = DEFAULT_MAX_SWEEPS,
        empirical_n: int = DEFAULT_EMPIRICAL_N,
        tol: float = DEFAULT_TOL,
    ) -> Iterator[dict]:
        """Validate, then return a generator streaming animation frames.

        Validation errors raise immediately (before any native call) so the
        HTTP layer can map them to a 4xx response.
        """
        model = self.model
        if not (0 <= col < model.width):
            raise AnswerError("column out of range")
        if not model.variables[col].learned:
            raise AnswerError("this column has no learned tree in the model")
        if any(o["column"] == col for o in s.observed):
            raise AnswerError("this item has already been answered and is clamped")
        max_sweeps = max(0, min(int(max_sweeps), MAX_SWEEPS_LIMIT))
        empirical_n = max(1, min(int(empirical_n), 1000))
        tol = max(0.0, float(tol))
        if mode == "choose" and value not in model.variables[col].categories:
            raise AnswerError(
                f"{value!r} is not a legal response for "
                f"{model.variables[col].variable}"
            )
        if mode not in ("sample", "map", "choose"):
            raise AnswerError(f"unknown mode {mode!r}")
        if s.lock.locked():
            raise AnswerError("this session is still relaxing the previous answer")
        return self._run(s, col, mode, value, max_sweeps, empirical_n, tol)

    def _frame(
        self, s: Session, before: list[dict], index: int, phase: str,
        sweep: int, summary: dict, seconds: float,
    ) -> dict:
        tv = tv_array(s.psi, before)
        changed = np.flatnonzero(tv > CHANGE_EPS)
        return {
            "type": "frame",
            "frame": index,
            "phase": phase,
            "sweep": sweep,
            "seconds": round(seconds, 4),
            "summary": summary,
            "tv": [round(float(x), 6) for x in tv],
            "changed": {
                str(int(c)): self.model.encode_one(int(c), s.psi[int(c)])
                for c in changed
            },
        }

    def _run(self, s, col, mode, value, max_sweeps, empirical_n, tol):
        # The lock is taken inside the generator so that it is only ever held
        # while the generator is actually running (and released by finally).
        if not s.lock.acquire(blocking=False):
            yield {"type": "error",
                   "detail": "this session is still relaxing the previous answer"}
            return
        model = self.model
        try:
            s.touch()
            if any(o["column"] == col for o in s.observed):
                yield {"type": "error", "detail": "this item is already clamped"}
                return
            try:
                sigma, extra = self._choose_value(s, col, mode, value)
            except AnswerError as exc:
                yield {"type": "error", "detail": str(exc)}
                return
            qnum = len(s.observed) + 1
            yield {
                "type": "answer",
                "question": qnum,
                "column": col,
                "variable": model.variables[col].variable,
                "mode": mode,
                "value": sigma,
                "seed": s.seed,
                "p_before": model.encode_one(col, s.psi[col]),
                **extra,
            }

            before = s.psi
            t0 = time.perf_counter()
            h = dict(
                s.state.hard_observe(
                    source=col,
                    value=sigma,
                    response_scale=RESPONSE_SCALE,
                    threads=THREADS,
                    clamp=True,
                )
            )
            sec = time.perf_counter() - t0
            s.psi = snapshot(s.state)
            record = {"column": col, "variable": model.variables[col].variable,
                      "value": sigma, "mode": mode, "question": qnum, **extra}
            s.observed.append(record)
            s.log.append({"question": qnum, "phase": "hard_observation",
                          "sweep": 0, "column": col, "value": sigma,
                          "mode": mode, "seconds": sec, **h})
            yield self._frame(s, before, 1, "hard_observation", 0, h, sec)

            stop = "max_sweeps"
            for sweep in range(1, max_sweeps + 1):
                yield {"type": "progress", "sweep": sweep, "of": max_sweeps}
                prev = s.psi
                t0 = time.perf_counter()
                summary = dict(
                    s.state.sweep(
                        n=empirical_n,
                        event="mode",
                        seed=s.seed,
                        response_scale=RESPONSE_SCALE,
                        threads=THREADS,
                        random_permutation=False,
                    )
                )
                sec = time.perf_counter() - t0
                s.psi = snapshot(s.state)
                s.log.append({"question": qnum, "phase": "relaxation",
                              "sweep": sweep, "column": col, "value": sigma,
                              "mode": mode, "seconds": sec, **summary})
                yield self._frame(s, prev, sweep + 1, "relaxation", sweep, summary, sec)
                s.touch()
                if summary.get("mean_tv", 1.0) < tol:
                    stop = "tolerance"
                    break

            yield {
                "type": "done",
                "question": qnum,
                "stop": stop,
                "clamped_count": int(s.state.clamped_count),
            }
        finally:
            s.touch()
            s.lock.release()
