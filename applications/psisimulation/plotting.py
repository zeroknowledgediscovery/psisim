"""Visualization helpers for probability-valued Psi states."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import total_variation


def ranked_probability_matrix(psi, topk: int = 8) -> np.ndarray:
    matrix = np.zeros((len(psi), topk), dtype=float)
    for i, q in enumerate(psi):
        values = sorted(
            (float(v) for v in q.values()),
            reverse=True,
        )[:topk]
        matrix[i, : len(values)] = values
    return matrix


def movement_vector(current, previous) -> np.ndarray:
    if previous is None:
        return np.zeros(len(current), dtype=float)
    return np.asarray(
        [
            total_variation(a, b)
            for a, b in zip(current, previous)
        ],
        dtype=float,
    )


def save_state_frame(
    psi,
    previous,
    path: str | Path,
    title: str,
    subtitle: str = "",
    observed_cols=None,
    topk: int = 8,
    dpi: int = 120,
) -> None:
    observed_cols = list(observed_cols or [])
    matrix = ranked_probability_matrix(psi, topk=topk)
    movement = movement_vector(psi, previous)
    top1 = np.asarray(
        [max(q.values()) if q else 0.0 for q in psi],
        dtype=float,
    )

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_axes([0.08, 0.48, 0.64, 0.42])
    image = ax1.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax1.set_xlabel("probability rank within variable")
    ax1.set_ylabel("LSM variable")
    ax1.set_title("Psi: ranked marginal probability mass")
    ax1.set_xticks(range(topk))
    ax1.set_xticklabels([str(i + 1) for i in range(topk)])

    for col in observed_cols:
        ax1.axhline(col, linewidth=0.8)

    cbar = fig.add_axes([0.73, 0.48, 0.02, 0.42])
    fig.colorbar(image, cax=cbar)

    ax2 = fig.add_axes([0.08, 0.12, 0.64, 0.25])
    ax2.plot(top1, linewidth=0.7)
    ax2.set_xlim(0, len(top1) - 1)
    ax2.set_ylim(0, 1.01)
    ax2.set_xlabel("variable")
    ax2.set_ylabel("max probability")
    ax2.set_title("MAP probability")
    for col in observed_cols:
        ax2.axvline(col, linewidth=0.8)

    ax3 = fig.add_axes([0.79, 0.12, 0.17, 0.78])
    ax3.plot(movement, np.arange(len(movement)), linewidth=0.7)
    ax3.invert_yaxis()
    ax3.set_xlabel("TV")
    ax3.set_ylabel("variable")
    ax3.set_title("movement\nfrom previous frame")
    for col in observed_cols:
        ax3.axhline(col, linewidth=0.8)

    fig.suptitle(title, fontsize=14)
    if subtitle:
        fig.text(0.08, 0.94, subtitle)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def make_gif(frame_paths, output: str | Path, duration_ms: int = 800) -> None:
    from PIL import Image

    frames = [Image.open(path) for path in frame_paths]
    if not frames:
        raise ValueError("no frames")
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    for frame in frames:
        frame.close()
