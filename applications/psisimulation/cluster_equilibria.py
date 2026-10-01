#!/usr/bin/env python3
"""Find and visualize dynamical equilibrium structure in a GSS model.

Workflow
--------
1. Reproducibly sample actual survey rows from a CSV.
2. Evolve every row for a fixed number of centered-LDP mode sweeps.
3. Save the terminal coordinatewise-MAP hard realizations.
4. Reload those saved equilibria from disk.
5. Compute the full native LSM qdistance matrix.
6. Embed the qdistance geometry into 2-D by classical MDS.
7. Fit k-medoids directly to qdistance for k in a configurable range.
8. Select k by mean silhouette in the native distance geometry.
9. Report one actual equilibrium medoid for each major cluster.
10. Save the plot, matrices, assignments, representatives, and diagnostics.

The 2-D embedding is ONLY a visualization. Clustering and representative
selection are performed directly on the native qdistance matrix.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BIN = REPO_ROOT / "bin"
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))

from common import resolve_model

predict_distribution = importlib.import_module("predict_distribution")
qdistance = importlib.import_module("qdistance")


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    return header, rows


def write_rows(path: Path, header, source_indices, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_row"] + list(header))
        for source, row in zip(source_indices, rows):
            writer.writerow([source] + list(row))


def reload_equilibria(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        source = []
        rows = []
        for record in reader:
            source.append(int(record[0]))
            rows.append(record[1:])
    return header[1:], source, rows


def classical_mds(D: np.ndarray, ndim: int = 2):
    """Classical metric MDS from a symmetric distance matrix.

    qdistance need not be exactly Euclidean. Negative eigenvalues are therefore
    retained in diagnostics but only positive eigen-directions are used for the
    displayed coordinates.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError("D must be square")

    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J

    eigenvalues, eigenvectors = np.linalg.eigh(B)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    positive = eigenvalues > 0
    pos_vals = eigenvalues[positive]
    pos_vecs = eigenvectors[:, positive]

    if len(pos_vals) < ndim:
        raise RuntimeError(
            f"classical MDS found only {len(pos_vals)} positive eigenvalues"
        )

    coords = pos_vecs[:, :ndim] * np.sqrt(pos_vals[:ndim])

    pos_sum = float(pos_vals.sum())
    explained = (
        pos_vals[:ndim] / pos_sum
        if pos_sum > 0
        else np.zeros(ndim)
    )
    neg_mass = float(np.abs(eigenvalues[eigenvalues < 0]).sum())
    pos_mass = float(pos_vals.sum())

    return coords, eigenvalues, explained, neg_mass, pos_mass


def initial_medoids(D: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Deterministic farthest-first initialization with seeded first point."""
    rng = random.Random(seed)
    n = D.shape[0]

    first = rng.randrange(n)
    medoids = [first]

    while len(medoids) < k:
        d_nearest = np.min(D[:, medoids], axis=1)
        d_nearest[medoids] = -1.0
        nxt = int(np.argmax(d_nearest))
        medoids.append(nxt)

    return np.asarray(medoids, dtype=int)


def assign_to_medoids(D: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    return np.argmin(D[:, medoids], axis=1)


def update_medoids(D: np.ndarray, labels: np.ndarray, k: int):
    medoids = np.empty(k, dtype=int)

    for cluster in range(k):
        members = np.flatnonzero(labels == cluster)
        if len(members) == 0:
            medoids[cluster] = -1
            continue

        sub = D[np.ix_(members, members)]
        costs = sub.mean(axis=1)
        medoids[cluster] = int(members[int(np.argmin(costs))])

    return medoids


def kmedoids(
    D: np.ndarray,
    k: int,
    seed: int,
    max_iter: int = 100,
):
    medoids = initial_medoids(D, k, seed)

    for _ in range(max_iter):
        labels = assign_to_medoids(D, medoids)
        new_medoids = update_medoids(D, labels, k)

        # Empty clusters should be extremely unlikely with distinct medoids,
        # but recover by choosing the point farthest from current medoids.
        for c in range(k):
            if new_medoids[c] >= 0:
                continue
            d_nearest = np.min(D[:, medoids], axis=1)
            new_medoids[c] = int(np.argmax(d_nearest))

        new_medoids = np.asarray(new_medoids, dtype=int)

        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids

    labels = assign_to_medoids(D, medoids)

    # Re-label clusters by medoid index for deterministic output.
    order = np.argsort(medoids)
    remap = np.empty(k, dtype=int)
    for new_label, old_label in enumerate(order):
        remap[old_label] = new_label

    labels = remap[labels]
    medoids = medoids[order]

    return labels, medoids


def silhouette_precomputed(D: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-sample silhouette using the supplied distance matrix."""
    labels = np.asarray(labels)
    unique = np.unique(labels)
    n = len(labels)

    if len(unique) < 2:
        return np.zeros(n)

    out = np.zeros(n, dtype=float)

    for i in range(n):
        own = labels[i]
        same = np.flatnonzero(labels == own)

        if len(same) <= 1:
            out[i] = 0.0
            continue

        same_wo_i = same[same != i]
        a = float(D[i, same_wo_i].mean())

        b = math.inf
        for other in unique:
            if other == own:
                continue
            members = np.flatnonzero(labels == other)
            if len(members) == 0:
                continue
            b = min(b, float(D[i, members].mean()))

        denom = max(a, b)
        out[i] = 0.0 if denom <= 0 else (b - a) / denom

    return out


def choose_k(
    D: np.ndarray,
    k_min: int,
    k_max: int,
    seed: int,
    restarts: int,
):
    results = []

    for k in range(k_min, k_max + 1):
        best = None

        for r in range(restarts):
            labels, medoids = kmedoids(
                D,
                k,
                seed=seed + 1009 * k + r,
            )
            sil = silhouette_precomputed(D, labels)
            score = float(np.mean(sil))

            candidate = {
                "k": k,
                "score": score,
                "labels": labels,
                "medoids": medoids,
                "silhouette": sil,
            }

            if best is None or score > best["score"]:
                best = candidate

        results.append(best)
        print(
            f"k={k:2d}  mean silhouette={best['score']:.6f}",
            flush=True,
        )

    # Highest native-distance silhouette wins.
    best = max(results, key=lambda item: item["score"])
    return best, results


def major_cluster_mask(
    labels: np.ndarray,
    major_fraction: float,
    min_major_size: int,
):
    n = len(labels)
    threshold = max(
        min_major_size,
        int(math.ceil(major_fraction * n)),
    )
    counts = {
        int(c): int(np.sum(labels == c))
        for c in np.unique(labels)
    }
    major = {
        c
        for c, count in counts.items()
        if count >= threshold
    }
    return major, counts, threshold


def save_matrix_csv(path: Path, D: np.ndarray, source_indices) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_row"] + list(source_indices))
        for source, row in zip(source_indices, D):
            writer.writerow([source] + [f"{x:.12g}" for x in row])


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="gss/gss_2018",
        help="public model key or local native-model directory",
    )
    parser.add_argument(
        "--data",
        default=os.environ.get("DATA", ""),
        help="GSS CSV. Defaults to $DATA.",
    )
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--sample-seed", type=int, default=271828)

    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--empirical-n", type=int, default=10)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--row-parallelism", type=int, default=0)

    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--cluster-restarts", type=int, default=5)
    parser.add_argument("--cluster-seed", type=int, default=314159)
    parser.add_argument("--major-fraction", type=float, default=0.05)
    parser.add_argument("--min-major-size", type=int, default=10)

    parser.add_argument(
        "--out",
        default="results/psisimulation/gss2018_equilibrium_clusters",
    )
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--verbose-dynamics", action="store_true")

    args = parser.parse_args()

    if not args.data:
        parser.error("--data is required unless $DATA is set")

    data_path = Path(args.data).expanduser().resolve()
    model = resolve_model(args.model, fetch=not args.no_fetch)
    trees = model / "trees" / "binary"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    header, all_rows = read_csv(data_path)

    if args.sample_size > len(all_rows):
        raise ValueError(
            f"sample size {args.sample_size} exceeds {len(all_rows)} data rows"
        )

    rng = random.Random(args.sample_seed)
    source_indices = sorted(
        rng.sample(range(len(all_rows)), args.sample_size)
    )
    rows = [all_rows[i] for i in source_indices]

    print("=" * 80)
    print("GSS DYNAMICAL EQUILIBRIUM CLUSTERING")
    print("=" * 80)
    print("model:          ", model)
    print("data:           ", data_path)
    print("sample size:    ", args.sample_size)
    print("sweeps:         ", args.sweeps)
    print("empirical n:    ", args.empirical_n)
    print("threads:        ", args.threads or "OpenMP default")
    print("row parallelism:", args.row_parallelism or "auto")
    print()

    # ------------------------------------------------------------------
    # 1. Evolve sampled observed rows toward near-equilibria.
    # ------------------------------------------------------------------

    print("Evolving sampled rows...", flush=True)
    t0 = time.perf_counter()

    dyn = dict(
        predict_distribution.centered_ldp_converge_rows(
            str(trees),
            rows,
            sweeps=args.sweeps,
            empirical_n=args.empirical_n,
            seed=12345,
            response_scale=1.0,
            random_permutation=False,
            tol=0.0,
            patience=1,
            verbose=args.verbose_dynamics,
            raw=True,
            run_dir=str(model),
            threads=args.threads,
            row_parallelism=args.row_parallelism,
        )
    )

    hard_equilibria = [list(row) for row in dyn["hard_rows"]]
    dynamics_sec = time.perf_counter() - t0

    print(f"dynamics complete: {dynamics_sec:.3f} sec")

    endpoint_path = out / "equilibria_hard.csv"
    write_rows(
        endpoint_path,
        header,
        source_indices,
        hard_equilibria,
    )
    print("saved:", endpoint_path)

    # Save convergence diagnostics before deliberately discarding/reloading rows.
    diagnostics_path = out / "convergence.csv"
    with diagnostics_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_row",
                "sweeps_run",
                "final_mean_tv",
                "final_max_tv",
                "final_max_col",
                "final_event_tv",
                "final_sanov_exponent",
                "final_projected_count",
            ]
        )
        for i, source in enumerate(source_indices):
            writer.writerow(
                [
                    source,
                    dyn["sweeps_run"][i],
                    dyn["final_mean_tv"][i],
                    dyn["final_max_tv"][i],
                    dyn["final_max_col"][i],
                    dyn["final_event_tv"][i],
                    dyn["final_sanov_exponent"][i],
                    dyn["final_projected_count"][i],
                ]
            )

    # ------------------------------------------------------------------
    # 2. Explicitly reload the saved hard equilibria.
    # ------------------------------------------------------------------

    del hard_equilibria
    _, reloaded_sources, equilibria = reload_equilibria(endpoint_path)

    if reloaded_sources != source_indices:
        raise RuntimeError("saved/reloaded source-row order changed")

    print(
        f"reloaded {len(equilibria)} hard equilibria from disk",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 3. Native pairwise qdistance matrix.
    # ------------------------------------------------------------------

    print("Computing native pairwise qdistance matrix...", flush=True)
    t0 = time.perf_counter()

    D = np.asarray(
        qdistance.qdistance_matrix(
            str(trees),
            equilibria,
            run_dir=str(model),
            threads=args.threads,
        ),
        dtype=float,
    )

    qdistance_sec = time.perf_counter() - t0

    # Clean tiny numerical asymmetry/diagonal noise defensively.
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)

    print(f"qdistance matrix complete: {qdistance_sec:.3f} sec")
    print(
        "distance range:",
        float(D[np.triu_indices_from(D, 1)].min()),
        "to",
        float(D.max()),
    )

    np.save(out / "qdistance_matrix.npy", D)
    save_matrix_csv(
        out / "qdistance_matrix.csv",
        D,
        source_indices,
    )

    # ------------------------------------------------------------------
    # 4. Classical MDS solely for visualization.
    # ------------------------------------------------------------------

    print("Computing 2-D classical MDS...", flush=True)
    coords, eigenvalues, explained, neg_mass, pos_mass = classical_mds(D)

    np.savetxt(
        out / "embedding_2d.csv",
        np.column_stack(
            [
                np.asarray(source_indices, dtype=int),
                coords,
            ]
        ),
        delimiter=",",
        header="source_row,mds1,mds2",
        comments="",
    )

    # ------------------------------------------------------------------
    # 5. Cluster DIRECTLY in qdistance geometry via k-medoids.
    # ------------------------------------------------------------------

    print()
    print("Selecting number of qdistance clusters...")
    best, cluster_runs = choose_k(
        D,
        k_min=args.k_min,
        k_max=min(args.k_max, len(equilibria) - 1),
        seed=args.cluster_seed,
        restarts=args.cluster_restarts,
    )

    labels = best["labels"]
    medoids = best["medoids"]
    silhouette = best["silhouette"]
    k = int(best["k"])

    major, counts, major_threshold = major_cluster_mask(
        labels,
        args.major_fraction,
        args.min_major_size,
    )

    print()
    print(f"selected k = {k}")
    print(f"mean silhouette = {best['score']:.6f}")
    print(f"major-cluster threshold = {major_threshold} equilibria")

    for c in range(k):
        marker = "MAJOR" if c in major else "minor"
        print(
            f"cluster {c:2d}: n={counts[c]:4d}  "
            f"medoid source row={source_indices[medoids[c]]:5d}  "
            f"{marker}"
        )

    # ------------------------------------------------------------------
    # 6. Save cluster assignments.
    # ------------------------------------------------------------------

    assignment_path = out / "cluster_assignments.csv"
    with assignment_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "equilibrium_index",
                "source_row",
                "cluster",
                "cluster_size",
                "major_cluster",
                "silhouette",
                "mds1",
                "mds2",
                "distance_to_cluster_medoid",
            ]
        )

        for i, source in enumerate(source_indices):
            c = int(labels[i])
            medoid = int(medoids[c])
            writer.writerow(
                [
                    i,
                    source,
                    c,
                    counts[c],
                    int(c in major),
                    silhouette[i],
                    coords[i, 0],
                    coords[i, 1],
                    D[i, medoid],
                ]
            )

    # ------------------------------------------------------------------
    # 7. Save ACTUAL medoid equilibrium for every major cluster.
    # ------------------------------------------------------------------

    rep_path = out / "major_cluster_representatives.csv"
    with rep_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "cluster",
                "cluster_size",
                "equilibrium_index",
                "source_row",
                "mean_within_cluster_qdistance",
            ]
            + list(header)
        )

        for c in sorted(major):
            medoid = int(medoids[c])
            members = np.flatnonzero(labels == c)
            within = float(D[medoid, members].mean())

            writer.writerow(
                [
                    c,
                    counts[c],
                    medoid,
                    source_indices[medoid],
                    within,
                ]
                + list(equilibria[medoid])
            )

    # ------------------------------------------------------------------
    # 8. 2-D plot.
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot minor clusters first with lower alpha.
    minor_mask = np.asarray(
        [int(c) not in major for c in labels],
        dtype=bool,
    )
    if np.any(minor_mask):
        ax.scatter(
            coords[minor_mask, 0],
            coords[minor_mask, 1],
            s=18,
            alpha=0.25,
            label="minor clusters",
        )

    for c in sorted(major):
        members = labels == c
        ax.scatter(
            coords[members, 0],
            coords[members, 1],
            s=28,
            alpha=0.75,
            label=f"cluster {c} (n={counts[c]})",
        )

        m = int(medoids[c])
        ax.scatter(
            [coords[m, 0]],
            [coords[m, 1]],
            marker="*",
            s=260,
            edgecolors="black",
            linewidths=0.8,
        )
        ax.annotate(
            f"C{c} medoid\nrow {source_indices[m]}",
            (coords[m, 0], coords[m, 1]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(
        f"MDS 1 ({100.0 * explained[0]:.1f}% of positive spectral mass)"
    )
    ax.set_ylabel(
        f"MDS 2 ({100.0 * explained[1]:.1f}% of positive spectral mass)"
    )
    ax.set_title(
        "GSS-2018 centered-LDP equilibrium geometry\n"
        f"native qdistance; {args.sweeps} sweeps; "
        f"k-medoids k={k}"
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out / "equilibrium_clusters_mds2d.png", dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 9. Diagnostics/provenance.
    # ------------------------------------------------------------------

    summary = {
        "model": str(model),
        "data": str(data_path),
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "sweeps": args.sweeps,
        "empirical_n": args.empirical_n,
        "dynamics_seconds": dynamics_sec,
        "qdistance_seconds": qdistance_sec,
        "selected_k": k,
        "mean_silhouette": float(best["score"]),
        "cluster_sizes": {str(c): counts[c] for c in sorted(counts)},
        "major_cluster_threshold": major_threshold,
        "major_clusters": sorted(int(c) for c in major),
        "major_cluster_medoids": {
            str(c): {
                "equilibrium_index": int(medoids[c]),
                "source_row": int(source_indices[medoids[c]]),
            }
            for c in sorted(major)
        },
        "mds_positive_explained_first2": [
            float(explained[0]),
            float(explained[1]),
        ],
        "mds_positive_spectral_mass": pos_mass,
        "mds_negative_spectral_mass": neg_mass,
        "candidate_k": [
            {
                "k": int(item["k"]),
                "mean_silhouette": float(item["score"]),
            }
            for item in cluster_runs
        ],
    }

    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print()
    print("=" * 80)
    print("WRITTEN")
    print("=" * 80)
    for name in [
        "equilibria_hard.csv",
        "convergence.csv",
        "qdistance_matrix.npy",
        "qdistance_matrix.csv",
        "embedding_2d.csv",
        "cluster_assignments.csv",
        "major_cluster_representatives.csv",
        "equilibrium_clusters_mds2d.png",
        "summary.json",
    ]:
        print(out / name)


if __name__ == "__main__":
    main()
