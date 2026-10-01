"""Shared helpers for Psi dynamical simulation."""
from __future__ import annotations

import importlib
import os
import re
import sys
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


def resident_empty_state(model: Path):
    """Construct one resident centered-LDP state initialized at Psi0."""
    lsm = load_binding()
    p0 = psi0(model)
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
