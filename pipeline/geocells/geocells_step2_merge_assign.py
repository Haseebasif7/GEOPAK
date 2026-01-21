"""
Geocell refinement (Step 4-5):

STEP 4 — MERGE UNDERSIZED CLUSTERS
  For each province & cluster (cluster_id >= 0):
    if cluster_size < min_samples_per_cell (40):
        find nearest valid cluster (by centroid distance)
        merge into it

STEP 5 — ASSIGN NOISE POINTS
  For each point with cluster_id = -1:
    assign to nearest cluster centroid

Inputs (from geocells_step1.py, under pipeline/geocells/):
  - province_<name>_clusters.csv  (id, x_m, y_m, cluster_id)

Outputs (under pipeline/geocells/):
  - province_<name>_clusters_step2.csv  (id, x_m, y_m, cluster_id)
  - geocell_step2_summary.md

Original merged_training_data_with_province.csv is NOT modified.

Run manually from repo root:
    python pipeline/geocells_step2_merge_assign.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

GEOCELLS_DIR = Path("pipeline/geocells")
STEP1_PATTERN = "province_*_clusters.csv"
STEP2_SUFFIX = "_clusters_step2.csv"
SUMMARY_PATH = GEOCELLS_DIR / "geocell_step2_summary.md"

MIN_SAMPLES_PER_CELL = 40  # min samples per cell used for merging


def _load_step1_files() -> Dict[str, Path]:
    """Return mapping province_name -> clusters.csv path from Step 1."""
    files = list(GEOCELLS_DIR.glob(STEP1_PATTERN))
    mapping: Dict[str, Path] = {}
    for p in files:
        # Expect filename like province_Sindh_clusters.csv
        name = p.stem  # province_Sindh_clusters
        if not name.startswith("province_") or not name.endswith("_clusters"):
            continue
        core = name[len("province_") : -len("_clusters")]
        province = core.replace("_", " ")
        mapping[province] = p
    return mapping


def _cluster_centroids(
    df: pd.DataFrame, cluster_col: str = "cluster_id"
) -> Dict[int, Tuple[float, float]]:
    """Compute centroids (x_m, y_m) per cluster_id (>= 0)."""
    centroids: Dict[int, Tuple[float, float]] = {}
    valid = df[df[cluster_col] >= 0]
    grouped = valid.groupby(cluster_col)
    for cid, g in grouped:
        centroids[int(cid)] = (float(g["x_m"].mean()), float(g["y_m"].mean()))
    return centroids


def _merge_undersized_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Merge clusters with size < MIN_SAMPLES_PER_CELL into nearest valid cluster."""
    labels = df["cluster_id"].to_numpy()

    # Compute sizes for non-noise clusters
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    sizes = dict(zip(unique.tolist(), counts.tolist()))

    # Identify small clusters
    small_clusters = [cid for cid, c in sizes.items() if c < MIN_SAMPLES_PER_CELL]

    if not small_clusters:
        return df

    centroids = _cluster_centroids(df)
    if not centroids:
        # No valid clusters, nothing to merge
        return df

    # Precompute arrays for nearest-neighbor search among valid clusters
    valid_ids = np.array(sorted(centroids.keys()), dtype=int)
    valid_centers = np.array([centroids[cid] for cid in valid_ids], dtype=float)

    arr = df[["x_m", "y_m"]].to_numpy()
    new_labels = labels.copy()

    for cid in small_clusters:
        if cid not in centroids:
            continue
        # If this is the only cluster, we can't merge it
        if len(valid_ids) <= 1:
            continue

        # Find nearest other cluster centroid
        center = np.array(centroids[cid], dtype=float)
        deltas = valid_centers - center[None, :]
        dists2 = np.einsum("ij,ij->i", deltas, deltas)

        # Exclude self from nearest calculation
        mask_other = valid_ids != cid
        if not mask_other.any():
            continue
        nearest_idx = np.argmin(dists2[mask_other])
        nearest_cid = valid_ids[mask_other][nearest_idx]

        # Reassign points from cid to nearest_cid
        new_labels[new_labels == cid] = nearest_cid

        # Optionally, update sizes (not strictly needed here)

    df = df.copy()
    df["cluster_id"] = new_labels
    return df


def _assign_noise_to_nearest(df: pd.DataFrame) -> pd.DataFrame:
    """Assign noise points (cluster_id == -1) to nearest cluster centroid."""
    labels = df["cluster_id"].to_numpy()
    centroids = _cluster_centroids(df)

    if not centroids:
        # No valid clusters to assign to
        return df

    valid_ids = np.array(sorted(centroids.keys()), dtype=int)
    valid_centers = np.array([centroids[cid] for cid in valid_ids], dtype=float)

    df = df.copy()
    coords = df[["x_m", "y_m"]].to_numpy()

    noise_mask = labels == -1
    if not noise_mask.any():
        return df

    noise_idx = np.where(noise_mask)[0]
    noise_coords = coords[noise_mask]

    # Compute distances from each noise point to all centroids
    # noise_coords: (N, 2), valid_centers: (K, 2)
    # -> (N, K)
    deltas = noise_coords[:, None, :] - valid_centers[None, :, :]
    dists2 = np.einsum("ijk,ijk->ij", deltas, deltas)
    nearest_indices = np.argmin(dists2, axis=1)
    nearest_cids = valid_ids[nearest_indices]

    # Assign
    labels = labels.copy()
    labels[noise_idx] = nearest_cids
    df["cluster_id"] = labels
    return df


def main():
    GEOCELLS_DIR.mkdir(parents=True, exist_ok=True)

    mapping = _load_step1_files()
    if not mapping:
        raise SystemExit("No Step1 cluster files found in pipeline/geocells")

    lines = []
    lines.append("# Geocell Step2 Summary\n")
    lines.append(
        f"- MIN_SAMPLES_PER_CELL (merge threshold): {MIN_SAMPLES_PER_CELL}\n"
    )

    for province, path in sorted(mapping.items()):
        print(f"\n=== Province: {province} ===")
        df = pd.read_csv(path)

        if not {"id", "x_m", "y_m", "cluster_id"}.issubset(df.columns):
            print(f"  Skipping {province}: required columns missing in {path.name}")
            continue

        total_before = len(df)
        n_noise_before = int((df["cluster_id"] == -1).sum())

        # Merge small clusters
        df_merged = _merge_undersized_clusters(df)

        # Assign noise
        df_final = _assign_noise_to_nearest(df_merged)

        n_noise_after = int((df_final["cluster_id"] == -1).sum())

        # Cluster stats after refinement
        labels = df_final["cluster_id"].to_numpy()
        unique, counts = np.unique(labels, return_counts=True)

        n_clusters = int(len(unique))  # all are valid now (no -1 expected)
        min_size = int(counts.min())
        max_size = int(counts.max())

        out_path = path.with_name(path.stem + STEP2_SUFFIX)
        df_final.to_csv(out_path, index=False)

        print(f"  Input rows: {total_before:,}")
        print(f"  Noise before: {n_noise_before:,}")
        print(f"  Noise after: {n_noise_after:,}")
        print(f"  Clusters (final): {n_clusters:,}")
        print(f"  Cell size range: {min_size:,} – {max_size:,}")

        lines.append(f"## {province}")
        lines.append(f"- Input file: {path.name}")
        lines.append(f"- Output file: {out_path.name}")
        lines.append(f"- Samples: {total_before:,}")
        lines.append(f"- Noise before: {n_noise_before:,}")
        lines.append(f"- Noise after: {n_noise_after:,}")
        lines.append(f"- Clusters (final): {n_clusters:,}")
        lines.append(f"- Cell size range: {min_size:,} – {max_size:,}\n")

    SUMMARY_PATH.write_text("\n".join(lines))
    print(f"\nSummary written to {SUMMARY_PATH}")
    print("Step 4-5 complete (offline).")


if __name__ == "__main__":
    main()


