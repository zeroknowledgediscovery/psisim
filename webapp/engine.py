"""Native PsiSim simulation engine behind the webapp.

All mathematics runs in native code. This module only:

* loads the model, variable metadata and the learned dependency graph once;
* keeps one resident native state per browser session;
* turns native calls into a stream of snapshot frames.

Default dynamics: propagation-only centered waves
(``psisim_dynamics.PropagationPsiState``). No intervention => no motion, so
Psi0 is stationary until a survey answer is imposed. An answer X_i = sigma is
applied with ``hard_observe`` (wave 0, the "splash"); each later wave
propagates only the deltas newly induced by the previous wave (the "ripples").

Optional finite-n empirical dynamics (``event="mode"`` / ``"sample"`` sweeps
of the vendored ``CenteredLdpPsiState``) are available as a separate session
type. They inject a new finite-n perturbation at every variable on every
sweep, so they move Psi0 even without an answer; they are an LDP experiment,
not the default relaxation.
"""
from __future__ import annotations

import gzip
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

from fetch_model import fetch_model  # noqa: E402
from common import (  # noqa: E402
    BIN,
    load_binding,
    cached_json,
    model_fingerprint,
    psi0_cached,
    resolve_model,
    snapshot,
    tree_ids,
)

ASSETS = HERE / "assets"
CATALOG_JSON = ASSETS / "catalog.json"
METADATA_ROOT = ASSETS / "metadata"

DYNAMICS = {
    "propagation": {
        "label": "Propagation waves (default)",
        "help": "Only an answer moves the state: its perturbation spreads "
                "wave by wave through the learned links. Psi0 is stationary.",
        "finite_n": False,
    },
    "mode": {
        "label": "Finite-n mode sweeps (optional LDP experiment)",
        "help": "Every sweep replaces each marginal by its modal finite-n "
                "empirical type. This injects new perturbations everywhere and "
                "moves Psi0 even without an answer.",
        "finite_n": True,
    },
    "sample": {
        "label": "Finite-n sampled sweeps (optional LDP experiment)",
        "help": "Every sweep replaces each marginal by a sampled finite-n "
                "empirical type. Stochastic; moves Psi0 even without an answer.",
        "finite_n": True,
    },
}
DEFAULT_DYNAMICS = "propagation"
DEFAULT_MAX_STEPS = 0      # one update per question (hard observation only);
                           # >0 adds propagation waves / finite-n sweeps (API only)
MAX_STEPS_LIMIT = 20
DEFAULT_EMPIRICAL_N = 10   # finite-n dynamics only
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


def load_dynamics_binding():
    if str(BIN) not in sys.path:
        sys.path.insert(0, str(BIN))
    try:
        import psisim_dynamics
    except ImportError as exc:
        raise RuntimeError(
            "Could not import psisim_dynamics. Build it with:\n"
            "  cmake --build build --target predict_distribution psisim_dynamics"
        ) from exc
    return psisim_dynamics


def tv_array(a: list[dict], b: list[dict]) -> np.ndarray:
    out = np.zeros(len(a), dtype=float)
    for i, (x, y) in enumerate(zip(a, b)):
        if x is y:
            continue
        keys = set(x) | set(y)
        out[i] = 0.5 * sum(abs(x.get(k, 0.0) - y.get(k, 0.0)) for k in keys)
    return out


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
    learned: bool


def load_catalog() -> dict:
    try:
        return json.loads(CATALOG_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"models": {}, "countries": {}, "default_model": "gss/gss_2018"}


CATALOG = load_catalog()
DEFAULT_MODEL = os.environ.get("PSISIM_MODEL", CATALOG.get("default_model", "gss/gss_2018"))


def model_root() -> Path:
    return Path(os.environ.get("DTAG_MODEL_ROOT", "~/.cache/dtag/models")).expanduser().resolve()


def is_installed(key: str) -> bool:
    local = model_root() / key
    return (local / "source_maps").is_dir() and (local / "trees" / "binary").is_dir()


def load_labels(key: str) -> dict[str, list[str]]:
    """{variable: [short label, question text]} built from DTAG's maps."""
    fam, _, name = key.partition("/")
    path = METADATA_ROOT / fam / f"{name}.json.gz"
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def source_map_columns(model: Path) -> dict[int, dict]:
    """{column: {column_header, column_strings_map}} from the native source maps."""
    out: dict[int, dict] = {}
    for shard in sorted((model / "source_maps" / "json_shards").glob("*.json")):
        for k, v in json.loads(shard.read_text(encoding="utf-8")).items():
            out[int(k)] = v
    return out


class Model:
    def __init__(self, key: str = DEFAULT_MODEL, fetch: bool = True):
        t0 = time.perf_counter()
        self.key = key
        self.info = dict(CATALOG.get("models", {}).get(key, {}))
        self.path = resolve_model(key, fetch=fetch)
        self.trees_dir = self.path / "trees" / "binary"
        self.tree_ids = tree_ids(self.path)
        self.width = max(self.tree_ids) + 1
        self.lsm = load_binding()

        self.variables = self._load_metadata()

        # Psi0 = (phi_i(x_empty))_i: computed natively once per model and
        # kept in a persistent on-disk cache (bit-identical on reuse).
        t1 = time.perf_counter()
        fingerprint = model_fingerprint(self.path)
        self.psi0, self.psi0_from_cache = psi0_cached(self.path, fingerprint=fingerprint)
        self.psi0_seconds = time.perf_counter() - t1
        if len(self.psi0) != self.width:
            raise RuntimeError("Psi0 width does not match model width")
        self._align_categories(self.psi0)

        self.dynamics = load_dynamics_binding()
        # The learned dependency graph (which columns each tree splits on)
        # needs every tree deserialised; it is cached like Psi0.
        used_raw, _ = cached_json(
            self.path, "deps",
            lambda: {str(k): v for k, v in self.dynamics.used_columns(
                str(self.trees_dir), self.tree_ids).items()},
            fingerprint=fingerprint,
        )
        used = {int(k): v for k, v in used_raw.items()}
        learned = set(self.tree_ids)
        self.targets = [[] for _ in range(self.width)]
        self.sources = [[] for _ in range(self.width)]
        for t, srcs in used.items():
            for s in srcs:
                if s != t and 0 <= s < self.width and s in learned:
                    self.targets[s].append(t)
                    self.sources[t].append(s)
        self.load_seconds = time.perf_counter() - t0

    def _load_metadata(self) -> list[Variable]:
        cols = source_map_columns(self.path)
        labels = load_labels(self.key)
        lower = {k.lower(): v for k, v in labels.items()}
        learned = set(self.tree_ids)
        out = []
        for col in range(self.width):
            entry = cols.get(col, {})
            name = str(entry.get("column_header") or f"col{col}")
            lab = labels.get(name) or lower.get(name.lower()) or ["", ""]
            out.append(
                Variable(
                    column=col,
                    variable=name,
                    label=lab[0],
                    question_text=lab[1],
                    categories=[c for c in entry.get("column_strings_map", []) if c != ""],
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
            "info": self.info,
            "width": self.width,
            "variables": [
                {
                    "column": v.column,
                    "variable": v.variable,
                    "label": v.label,
                    "question_text": v.question_text,
                    "categories": v.categories,
                    "learned": v.learned,
                    "out_degree": len(self.targets[v.column]),
                    "in_degree": len(self.sources[v.column]),
                }
                for v in self.variables
            ],
            "psi0": self.encode(self.psi0),
            "defaults": {
                "dynamics": DEFAULT_DYNAMICS,
                "max_steps": DEFAULT_MAX_STEPS,
                "max_steps_limit": MAX_STEPS_LIMIT,
                "empirical_n": DEFAULT_EMPIRICAL_N,
                "response_scale": RESPONSE_SCALE,
            },
            "dynamics": DYNAMICS,
        }

    def new_state(self, dynamics: str = DEFAULT_DYNAMICS):
        """A resident native state initialised at the shared Psi0."""
        if dynamics == "propagation":
            return self.dynamics.PropagationPsiState(
                str(self.trees_dir), str(self.path), self.psi0
            )
        return self.lsm.centered_ldp_state_from_psi(
            str(self.trees_dir),
            self.psi0,
            raw=True,
            run_dir=str(self.path),
        )


# ---------------------------------------------------------------------------
# Model registry: download from the public release on demand, load, cache
# ---------------------------------------------------------------------------

MAX_MODELS = max(1, env_int("PSISIM_MAX_MODELS", 3))


class ModelError(ValueError):
    pass


class ModelRegistry:
    def __init__(self, fetch: bool = True):
        self.fetch = fetch
        self._models: dict[str, Model] = {}
        self._order: list[str] = []
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def validate(self, key: str) -> str:
        if key not in CATALOG.get("models", {}):
            raise ModelError(f"unknown model {key!r}")
        return key

    def status(self, key: str) -> dict:
        with self._lock:
            if key in self._models:
                return {"key": key, "state": "ready", "progress": 1.0}
            job = self._jobs.get(key)
            if job:
                return dict(job)
        return {"key": key, "state": "installed" if is_installed(key) else "remote", "progress": 0.0}

    def get(self, key: str) -> Model:
        with self._lock:
            model = self._models.get(key)
            if model is None:
                raise ModelError(f"model {key} is not loaded yet")
            self._order.remove(key)
            self._order.append(key)
            return model

    def _remember(self, model: Model) -> None:
        with self._lock:
            self._models[model.key] = model
            if model.key in self._order:
                self._order.remove(model.key)
            self._order.append(model.key)
            # Sessions keep their own reference, so evicting only drops the
            # cache entry; the default model is never evicted.
            for old in [k for k in self._order if k != DEFAULT_MODEL]:
                if len(self._order) <= MAX_MODELS:
                    break
                self._order.remove(old)
                self._models.pop(old, None)

    def load_now(self, key: str) -> Model:
        self.validate(key)
        with self._lock:
            if key in self._models:
                return self._models[key]
        model = Model(key, fetch=self.fetch)
        self._remember(model)
        return model

    def start(self, key: str) -> dict:
        """Download (if needed) and load ``key`` in the background."""
        self.validate(key)
        with self._lock:
            if key in self._models:
                return {"key": key, "state": "ready", "progress": 1.0}
            job = self._jobs.get(key)
            if job and job["state"] in ("queued", "downloading", "loading"):
                return dict(job)
            job = {"key": key, "state": "queued", "progress": 0.0, "message": ""}
            self._jobs[key] = job

        def update(**kw):
            with self._lock:
                job.update(kw)

        def run():
            try:
                if not is_installed(key):
                    if not self.fetch:
                        raise ModelError("model is not installed and downloads are disabled")
                    update(state="downloading", message="downloading from the public model release")

                    def progress(done, total):
                        update(progress=(done / total) if total else 0.0,
                               done_bytes=done, total_bytes=total)

                    fetch_model(key, root=model_root(), progress=progress)
                update(state="loading", progress=1.0, message="loading Ψ₀ (cached after the first time)")
                model = Model(key, fetch=False)
                self._remember(model)
                update(state="ready", progress=1.0, message="")
            except Exception as exc:  # reported to the client
                update(state="error", message=str(exc))

        threading.Thread(target=run, daemon=True).start()
        return dict(job)

    def catalog(self) -> dict:
        doc = dict(CATALOG)
        models = {}
        with self._lock:
            loaded = set(self._models)
        for key, info in CATALOG.get("models", {}).items():
            m = dict(info)
            m["installed"] = is_installed(key)
            m["loaded"] = key in loaded
            models[key] = m
        doc["models"] = models
        return doc


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class AnswerError(ValueError):
    pass


@dataclass
class Session:
    id: str
    seed: int
    dynamics: str
    state: object
    psi: list[dict]
    rng: np.random.Generator
    model: "Model"
    lock: threading.Lock = field(default_factory=threading.Lock)
    observed: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)  # one entry per answer
    log: list[dict] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_used = time.time()


# A coordinate counts as "moved" above the 1e-6 resolution of the streamed
# probabilities, so server statistics match what the browser can show.
MOVED_EPS = 1.25e-6  # between 1e-6 and the next value on the rounded grid


def change_stats(tv: np.ndarray, learned: np.ndarray, clamped: set[int]) -> dict:
    """Movement of the unclamped learned coordinates ("other topics")."""
    mask = learned.copy()
    for c in clamped:
        mask[c] = False
    t = tv[mask]
    return {
        "moved": int((t > MOVED_EPS).sum()),
        "moved_001": int((t > 0.01).sum()),
        "moved_005": int((t > 0.05).sum()),
        "mean_tv": float(t.mean()) if len(t) else 0.0,
        "max_tv": float(t.max()) if len(t) else 0.0,
        "other_topics": int(mask.sum()),
    }


class Engine:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.default = registry.load_now(DEFAULT_MODEL)
        self.sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._spare = None
        self._spare_lock = threading.Lock()
        self._spare_thread: threading.Thread | None = None
        self.refill_spare()

    # -- warm spare state of the default model (building one loads all trees)

    def refill_spare(self) -> None:
        def build():
            state = self.default.new_state(DEFAULT_DYNAMICS)
            with self._spare_lock:
                if self._spare is None:
                    self._spare = state

        with self._spare_lock:
            busy = self._spare_thread is not None and self._spare_thread.is_alive()
            if self._spare is not None or busy:
                return
            self._spare_thread = threading.Thread(target=build, daemon=True)
            self._spare_thread.start()

    def _take_state(self, model: Model, dynamics: str):
        if dynamics != DEFAULT_DYNAMICS or model is not self.default:
            return model.new_state(dynamics)
        with self._spare_lock:
            state, self._spare = self._spare, None
        if state is None:
            state = model.new_state(dynamics)
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

    def create(self, seed: int | None = None,
               dynamics: str = DEFAULT_DYNAMICS,
               model_key: str | None = None) -> Session:
        if dynamics not in DYNAMICS:
            raise AnswerError(f"unknown dynamics {dynamics!r}")
        model = self.registry.get(model_key or DEFAULT_MODEL)
        if seed is None:
            seed = secrets.randbelow(2**31 - 1) + 1
        seed = int(seed) % (2**63)
        with self._lock:
            self._evict()
        state = self._take_state(model, dynamics)
        session = Session(
            id=secrets.token_urlsafe(12),
            seed=seed,
            dynamics=dynamics,
            state=state,
            psi=snapshot(state),
            rng=np.random.default_rng(seed),
            model=model,
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
            "model": s.model.key,
            "seed": s.seed,
            "dynamics": s.dynamics,
            "psi": s.model.encode(s.psi),
            "observed": s.observed,
            "history": [
                {k: v for k, v in h.items() if k != "psi_raw"}
                for h in s.history
            ],
            "clamped_count": int(s.state.clamped_count),
        }

    def export(self, s: Session) -> dict:
        return {
            "model": s.model.key,
            "dynamics": s.dynamics,
            "seed": s.seed,
            "observed": s.observed,
            "answers": [
                {k: v for k, v in h.items() if k not in ("psi", "psi_raw", "tv")}
                for h in s.history
            ],
            "log": s.log,
        }

    # -- answering ---------------------------------------------------------

    def _draw(self, s: Session, col: int) -> tuple[str, float]:
        """sigma ~ p_i from the current state, with the session's seeded RNG."""
        var = s.model.variables[col]
        q = s.psi[col]
        cats = [c for c in var.categories if q.get(c, 0.0) > 0.0] or list(q)
        probs = np.asarray([q.get(c, 0.0) for c in cats], dtype=float)
        total = probs.sum()
        if total <= 0:
            raise AnswerError("current distribution has no mass")
        cdf = np.cumsum(probs / total)
        u = float(s.rng.random())
        k = int(min(np.searchsorted(cdf, u, side="right"), len(cats) - 1))
        return cats[k], u

    def answer(
        self,
        s: Session,
        col: int,
        max_steps: int = DEFAULT_MAX_STEPS,
        empirical_n: int = DEFAULT_EMPIRICAL_N,
    ) -> Iterator[dict]:
        """Validate, then return a generator streaming animation frames.

        Validation errors raise immediately (before any native call) so the
        HTTP layer can map them to a 4xx response.
        """
        model = s.model
        if not (0 <= col < model.width):
            raise AnswerError("column out of range")
        if not model.variables[col].learned:
            raise AnswerError("this column has no learned tree in the model")
        if any(o["column"] == col for o in s.observed):
            raise AnswerError("this item has already been answered and is clamped")
        max_steps = max(0, min(int(max_steps), MAX_STEPS_LIMIT))
        empirical_n = max(1, min(int(empirical_n), 1000))
        if s.lock.locked():
            raise AnswerError("this session is still propagating the previous answer")
        return self._run(s, col, max_steps, empirical_n)

    def _frame(self, s: Session, before: list[dict], index: int, phase: str,
               step: int, summary: dict, seconds: float) -> dict:
        tv = tv_array(s.psi, before)
        changed = np.flatnonzero(tv > CHANGE_EPS)
        return {
            "type": "frame",
            "frame": index,
            "phase": phase,
            "step": step,
            "seconds": round(seconds, 4),
            "summary": summary,
            "tv": [round(float(x), 6) for x in tv],
            "changed": {
                str(int(c)): s.model.encode_one(int(c), s.psi[int(c)])
                for c in changed
            },
        }

    def _steps(self, s: Session, max_steps: int, empirical_n: int):
        """Yield (summary, seconds) for each wave / finite-n sweep."""
        state = s.state
        for step in range(1, max_steps + 1):
            if s.dynamics == "propagation":
                if state.pending_count == 0:
                    return
                t0 = time.perf_counter()
                summary = dict(state.wave(threads=THREADS))
            else:
                t0 = time.perf_counter()
                summary = dict(
                    state.sweep(
                        n=empirical_n,
                        event=s.dynamics,
                        seed=s.seed,
                        response_scale=RESPONSE_SCALE,
                        threads=THREADS,
                        random_permutation=False,
                    )
                )
            yield step, summary, time.perf_counter() - t0

    def _run(self, s: Session, col: int, max_steps: int, empirical_n: int):
        # The lock is taken inside the generator so that it is only ever held
        # while the generator is actually running (and released by finally).
        if not s.lock.acquire(blocking=False):
            yield {"type": "error",
                   "detail": "this session is still propagating the previous answer"}
            return
        model = s.model
        try:
            s.touch()
            if any(o["column"] == col for o in s.observed):
                yield {"type": "error", "detail": "this item is already clamped"}
                return
            try:
                sigma, u = self._draw(s, col)
            except AnswerError as exc:
                yield {"type": "error", "detail": str(exc)}
                return
            qnum = len(s.observed) + 1
            yield {
                "type": "answer",
                "question": qnum,
                "column": col,
                "variable": model.variables[col].variable,
                "value": sigma,
                "u": u,
                "seed": s.seed,
                "dynamics": s.dynamics,
                "p_before": model.encode_one(col, s.psi[col]),
            }

            start = s.psi
            t0 = time.perf_counter()
            if s.dynamics == "propagation":
                h = dict(s.state.hard_observe(col, sigma, clamp=True,
                                              threads=THREADS))
            else:
                h = dict(s.state.hard_observe(source=col, value=sigma,
                                              response_scale=RESPONSE_SCALE,
                                              threads=THREADS, clamp=True))
            sec = time.perf_counter() - t0
            s.psi = snapshot(s.state)
            s.observed.append({"column": col,
                               "variable": model.variables[col].variable,
                               "value": sigma, "question": qnum, "u": u})
            s.log.append({"question": qnum, "phase": "hard_observation",
                          "step": 0, "column": col, "value": sigma,
                          "seconds": sec, **h})
            yield self._frame(s, start, 1, "hard_observation", 0, h, sec)

            steps_run = 0
            for step, summary, sec in self._steps(s, max_steps, empirical_n):
                yield {"type": "progress", "step": step, "of": max_steps}
                prev = s.psi
                s.psi = snapshot(s.state)
                steps_run = step
                s.log.append({"question": qnum,
                              "phase": "wave" if s.dynamics == "propagation"
                              else "sweep",
                              "step": step, "column": col, "value": sigma,
                              "seconds": sec, **summary})
                yield self._frame(s, prev, step + 1,
                                  "wave" if s.dynamics == "propagation" else "sweep",
                                  step, summary, sec)
                s.touch()

            residual = 0.0
            if s.dynamics == "propagation":
                residual = sum(
                    0.5 * sum(abs(x) for x in d.values())
                    for d in s.state.pending().values())
                if s.state.pending_count:
                    stop = "max_steps"
                    # The wave train of this answer ends here. Undamped
                    # waves at response scale 1 need not die out, so the
                    # remaining deltas are dropped (and reported) rather
                    # than leaking into the next answer's waves.
                    s.state.discard_pending()
                else:
                    stop = "settled"
            else:
                stop = "max_steps"

            # Statistics use the same 1e-6-rounded arrays the browser receives,
            # so the saved numbers equal the live counter exactly.
            after_enc = model.encode(s.psi)
            before_enc = model.encode(start)
            tv = np.array([0.5 * sum(abs(x - y) for x, y in zip(a, b))
                           for a, b in zip(after_enc, before_enc)])
            learned = np.array([v.learned for v in model.variables])
            stats = change_stats(tv, learned,
                                 {o["column"] for o in s.observed})
            entry = {
                "question": qnum,
                "column": col,
                "variable": model.variables[col].variable,
                "value": sigma,
                "steps": steps_run,
                "stop": stop,
                "residual_pending_tv": residual,
                "stats": stats,
                "tv": [round(float(x), 6) for x in tv],
                "psi": after_enc,
            }
            s.history.append(entry)
            yield {
                "type": "done",
                "question": qnum,
                "stop": stop,
                "steps": steps_run,
                "residual_pending_tv": residual,
                "stats": stats,
                "clamped_count": int(s.state.clamped_count),
            }
        finally:
            s.touch()
            s.lock.release()
