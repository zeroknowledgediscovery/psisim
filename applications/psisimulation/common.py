"""Shared helpers for Psi dynamical simulation."""
from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BIN = REPO_ROOT / "bin"


def load_binding():
    """Import the locally compiled predict_distribution binding."""
    if str(BIN) not in sys.path:
        sys.path.insert(0, str(BIN))
    try:
        return importlib.import_module("predict_distribution")
    except ImportError as exc:
        raise RuntimeError(
            "Could not import predict_distribution. Build it first:\n"
            "  cmake -S . -B build-tests -G Ninja "
            "-DCMAKE_BUILD_TYPE=Release -DLSM_BUILD_PYTHON_BINDINGS=ON\n"
            "  cmake --build build-tests --target predict_distribution -j $(nproc)"
        ) from exc


def tree_ids(model: Path) -> list[int]:
    trees = model / "trees" / "binary"
    ids = []
    for path in trees.iterdir():
        match = re.fullmatch(r"tree_(\d+)\.bin", path.name)
        if match:
            ids.append(int(match.group(1)))
    ids.sort()
    if not ids:
        raise RuntimeError(f"no tree_*.bin files under {trees}")
    return ids


def model_width(model: Path) -> int:
    return max(tree_ids(model)) + 1


def empty_row(model: Path) -> np.ndarray:
    return np.asarray([""] * model_width(model), dtype=str)


def resolve_model(
    model: str,
    fetch: bool = True,
    release: str = "v0.2.0",
) -> Path:
    """Resolve either a local model directory or a public model key."""
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        return candidate.resolve()

    if "/" not in model:
        model = f"gss/{model}"

    root = Path(
        os.environ.get("DTAG_MODEL_ROOT", "~/.cache/dtag/models")
    ).expanduser().resolve()
    local = root / model

    if (
        (local / "source_maps").is_dir()
        and (local / "trees" / "binary").is_dir()
    ):
        return local

    if not fetch:
        raise FileNotFoundError(
            f"model is not installed: {model}; expected {local}"
        )

    from fetch_model import fetch_model

    return fetch_model(model, root=root, release=release)


def psi0(model: Path):
    """Return the canonical empty-row state Psi0."""
    lsm = load_binding()
    row = empty_row(model)
    psi = lsm.row_to_psi(
        str(model / "trees" / "binary"),
        row,
        raw=True,
        run_dir=str(model),
    )
    return [dict(q) for q in psi]


def model_cache_dir() -> Path:
    return Path(os.environ.get("PSISIM_CACHE_DIR", "~/.cache/psisim")).expanduser()


def model_fingerprint(model: Path) -> str:
    """Identity of everything derived model data depends on: the model's
    trees and source maps (content) and the compiled native runtime."""
    h = hashlib.sha256()
    for sub in ("trees/binary", "source_maps"):
        root = model / sub
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            h.update(str(path.relative_to(model)).encode())
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    h.update(block)
    for lib in sorted(BIN.glob("*.so")):
        h.update(lib.name.encode())
        h.update(lib.read_bytes())
    return h.hexdigest()[:20]


def cached_json(model: Path, kind: str, compute, *, fingerprint: str | None = None,
                cache_dir: Path | None = None, valid=lambda value: True):
    """Value derived from ``model``, kept in a persistent on-disk JSON cache.

    The file is ``<cache>/<kind>/<family>/<name>-<fingerprint>.json.gz``, so a
    changed model or rebuilt runtime is recomputed automatically and stale
    entries for that model are removed. Returns (value, hit).
    """
    root = model_cache_dir() if cache_dir is None else Path(cache_dir)
    fp = fingerprint or model_fingerprint(model)
    name = model.name
    path = root / kind / model.parent.name / f"{name}-{fp}.json.gz"
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
        if valid(value):
            return value, True
    except (OSError, ValueError, TypeError):
        pass

    value = compute()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{kind}-", suffix=".tmp")
        with os.fdopen(fd, "wb") as raw, gzip.open(raw, "wt", encoding="utf-8") as handle:
            json.dump(value, handle, separators=(",", ":"))
        os.replace(tmp, path)
        for stale in path.parent.glob(f"{name}-*.json.gz"):
            if stale != path:
                stale.unlink(missing_ok=True)
    except OSError:
        pass  # caching is best effort
    return value, False


def psi0_cached(model: Path, *, fingerprint: str | None = None,
                cache_dir: Path | None = None) -> tuple[list[dict], bool]:
    """Psi0 from the persistent cache (computed and stored on first use).

    Values round-trip exactly (JSON floats use repr) and category order is
    preserved, so a cached Psi0 is bit-identical to a freshly computed one.
    """
    width = model_width(model)
    rows, hit = cached_json(
        model, "psi0",
        lambda: [list(q.items()) for q in psi0(model)],
        fingerprint=fingerprint, cache_dir=cache_dir,
        valid=lambda v: isinstance(v, list) and len(v) == width,
    )
    return [dict(r) for r in rows], hit


def resident_empty_state(model: Path):
    """Construct one resident centered-LDP state initialized at Psi0."""
    lsm = load_binding()
    p0, _ = psi0_cached(model)   # exact; computed once per model and runtime
    return lsm.centered_ldp_state_from_psi(
        str(model / "trees" / "binary"),
        p0,
        raw=True,
        run_dir=str(model),
    )


def snapshot(state) -> list[dict]:
    return [dict(q) for q in state.to_python()]


def total_variation(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(
        abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0)))
        for k in keys
    )


def psi_distance(a, b) -> tuple[float, float]:
    values = [
        total_variation(x, y)
        for x, y in zip(a, b)
    ]
    return float(np.mean(values)), float(np.max(values))


def argmax_symbol(q: dict):
    if not q:
        return None
    return max(
        q.items(),
        key=lambda kv: (float(kv[1]), str(kv[0])),
    )[0]
