"""Geocell refinement (Step 6-7): Best-effort radius constraint and target cell counts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

GEOCELLS_DIR = Path("pipeline/geocells")
STEP2_SUFFIX = "_clusters_step2.csv"
STEP1_SUFFIX = "_clusters.csv"
FINAL_SUFFIX = "_clusters_final.csv"
SUMMARY_PATH = GEOCELLS_DIR / "geocell_step3_summary.md"

MAX_RADIUS_M = 50_000.0  # 50 km in meters (BEST‑EFFORT target, not hard)
# Province-specific min samples per cell
MIN_SAMPLES_PER_CELL_BY_PROVINCE = {
    "Balochistan": 35,  # Lower threshold for Balochistan
}
MIN_SAMPLES_PER_CELL = 40  # default

def get_min_samples_for_province(province: str) -> int:
    """Get min samples per cell for a province, with fallback to default."""
    return MIN_SAMPLES_PER_CELL_BY_PROVINCE.get(province, MIN_SAMPLES_PER_CELL)

# Province-specific radius targets (for Balochistan, use higher threshold)
MAX_RADIUS_M_BY_PROVINCE = {
    "Balochistan": 60_000.0,  # 60 km for Balochistan (target: 55-65 km avg)
}
# Default to MAX_RADIUS_M if province not specified
def get_max_radius_for_province(province: str) -> float:
    return MAX_RADIUS_M_BY_PROVINCE.get(province, MAX_RADIUS_M)

TARGET_CELLS = {
    "Sindh": 450,
    "Punjab": 400,
    "Khyber Pakhtunkhwa": 220,
    "ICT": 100,
    "Gilgit-Baltistan": 180,
    "Balochistan": 50,  # Updated: target 48-55, using 50 as middle
    "Azad Kashmir": 80,
}


@dataclass
class ClusterStats:
    n_cells_before: int
    n_cells_after_radius: int
    n_cells_final: int
    max_radius_before_m: float
    max_radius_after_m: float


def _load_province_cluster_file(province: str) -> pd.DataFrame | None:
    """Load step2 clusters for a province, or fall back to step1 clusters."""
    stem = f"province_{province.replace(' ', '_')}"
    step2_path = GEOCELLS_DIR / f"{stem}{STEP2_SUFFIX}"
    step1_path = GEOCELLS_DIR / f"{stem}{STEP1_SUFFIX}"

    path = None
    if step2_path.exists():
        path = step2_path
    elif step1_path.exists():
        path = step1_path

    if path is None:
        return None

    df = pd.read_csv(path)
    required = {"id", "x_m", "y_m", "cluster_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing required columns {required}")

    # Ensure integer cluster ids
    df["cluster_id"] = df["cluster_id"].astype(int)
    return df


def _cluster_centroids(df: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
    centroids: Dict[int, Tuple[float, float]] = {}
    for cid, g in df.groupby("cluster_id"):
        centroids[int(cid)] = (float(g["x_m"].mean()), float(g["y_m"].mean()))
    return centroids


def _cluster_radii(df: pd.DataFrame) -> Dict[int, float]:
    """Return per-cluster max radius in meters (using x_m, y_m)."""
    radii: Dict[int, float] = {}
    for cid, g in df.groupby("cluster_id"):
        center_x = float(g["x_m"].mean())
        center_y = float(g["y_m"].mean())
        dx = g["x_m"].to_numpy() - center_x
        dy = g["y_m"].to_numpy() - center_y
        dist = np.sqrt(dx * dx + dy * dy)
        radii[int(cid)] = float(dist.max()) if len(dist) else 0.0
    return radii


def _merge_small_cells(
    df: pd.DataFrame,
    min_samples_per_cell: int,
) -> pd.DataFrame:
    """
    Final cleanup: ensure all cells have at least `min_samples_per_cell`
    points by repeatedly merging the smallest cell into its nearest neighbor.
    """
    df = df.copy()

    while True:
        sizes = df["cluster_id"].value_counts()
        min_size = int(sizes.min())
        if min_size >= min_samples_per_cell or len(sizes) <= 1:
            break

        # Smallest cell to remove
        cid_small = int(sizes.idxmin())
        centroids = _cluster_centroids(df)
        center_small = np.array(centroids[cid_small], dtype=float)

        # Find nearest neighbor cluster
        other_ids = [cid for cid in centroids.keys() if cid != cid_small]
        if not other_ids:
            break

        centers = np.array([centroids[cid] for cid in other_ids], dtype=float)
        deltas = centers - center_small[None, :]
        dists2 = np.einsum("ij,ij->i", deltas, deltas)
        nearest_idx = int(np.argmin(dists2))
        cid_target = int(other_ids[nearest_idx])

        df.loc[df["cluster_id"] == cid_small, "cluster_id"] = cid_target

    # Re-normalize cluster ids
    unique_ids = sorted(df["cluster_id"].unique())
    remap = {old: i for i, old in enumerate(unique_ids)}
    df["cluster_id"] = df["cluster_id"].map(remap).astype(int)
    return df


def _split_cluster_kmeans(df: pd.DataFrame, cid: int, next_cluster_id: int) -> int:
    """
    Split cluster `cid` into two using k-means (k=2).
    Uses x_m, y_m; returns new next_cluster_id.
    """
    mask = df["cluster_id"] == cid
    sub = df.loc[mask, ["x_m", "y_m"]].to_numpy()
    if len(sub) < 2:
        return next_cluster_id

    km = KMeans(n_clusters=2, random_state=0, n_init="auto")
    labels = km.fit_predict(sub)

    # Keep cid for label 0, assign new id for label 1.
    # Need to map cluster-local labels back to the original DataFrame index.
    new_id = next_cluster_id
    cluster_index = df.index[mask]  # index positions belonging to this cluster
    # labels is aligned with `sub`, which corresponds to cluster_index
    indices_label1 = cluster_index[labels == 1]
    df.loc[indices_label1, "cluster_id"] = new_id

    return new_id + 1


def _split_cluster_if_feasible(
    df: pd.DataFrame,
    cid: int,
    next_cluster_id: int,
    min_samples_per_cell: int,
) -> tuple[bool, pd.DataFrame, int]:
    """
    Attempt to split cluster `cid` into two using k-means (k=2), but only
    apply the split if BOTH resulting sub-clusters have at least
    `min_samples_per_cell` points.

    Returns (success, new_df, new_next_cluster_id).
    If not feasible, returns (False, original_df, next_cluster_id) and
    leaves cluster assignments unchanged.
    """
    mask = df["cluster_id"] == cid
    sub = df.loc[mask, ["x_m", "y_m"]].to_numpy()
    n = len(sub)
    if n < 2 * min_samples_per_cell:
        # Even an ideal 50/50 split cannot satisfy the min per cell
        return False, df, next_cluster_id

    km = KMeans(n_clusters=2, random_state=0, n_init="auto")
    labels = km.fit_predict(sub)

    # Count sizes of the two proposed clusters
    counts = np.bincount(labels, minlength=2)
    if counts.min() < min_samples_per_cell:
        # Would violate min_samples_per_cell; reject split
        return False, df, next_cluster_id

    # Apply split: keep cid for label 0, assign new id for label 1.
    # Map local k-means labels back to the DataFrame index.
    new_df = df.copy()
    new_id = next_cluster_id
    cluster_index = new_df.index[mask]
    indices_label1 = cluster_index[labels == 1]
    new_df.loc[indices_label1, "cluster_id"] = new_id
    return True, new_df, new_id + 1


def _enforce_radius(df: pd.DataFrame, province: str = None) -> Tuple[pd.DataFrame, ClusterStats]:
    """Split clusters until all have radius <= max_radius for province."""
    max_radius = get_max_radius_for_province(province) if province else MAX_RADIUS_M
    min_samples = get_min_samples_for_province(province) if province else MIN_SAMPLES_PER_CELL
    
    # Initial stats
    radii_before = _cluster_radii(df)
    max_radius_before = max(radii_before.values()) if radii_before else 0.0
    n_cells_before = df["cluster_id"].nunique()

    next_cluster_id = int(df["cluster_id"].max()) + 1

    # Iterate splitting worst-offending cluster until all radii satisfy constraint
    max_iterations = 100
    for _ in range(max_iterations):
        radii = _cluster_radii(df)
        if not radii:
            break

        cid_worst, r_worst = max(radii.items(), key=lambda kv: kv[1])
        if r_worst <= max_radius:
            break

        # Split worst cluster (check feasibility first)
        success, new_df, next_cluster_id = _split_cluster_if_feasible(
            df, cid_worst, next_cluster_id, min_samples_per_cell=min_samples
        )
        if not success:
            break
        df = new_df

    radii_after = _cluster_radii(df)
    max_radius_after = max(radii_after.values()) if radii_after else 0.0
    n_cells_after = df["cluster_id"].nunique()

    stats = ClusterStats(
        n_cells_before=n_cells_before,
        n_cells_after_radius=n_cells_after,
        n_cells_final=0,  # filled later
        max_radius_before_m=max_radius_before,
        max_radius_after_m=max_radius_after,
    )
    return df, stats


def _best_effort_radius_reduction(df: pd.DataFrame, province: str = None) -> pd.DataFrame:
    """
    Final pass (BEST‑EFFORT radius reduction):
      1) Try to reduce max radius toward max_radius for province by splitting worst cells,
         but ONLY when feasible under the min_samples_per_cell constraint.
      2) After each split, run min-size cleanup to keep all cells >= min_samples.
    This does NOT guarantee radius <= max_radius in extremely sparse regions.
    """
    max_radius = get_max_radius_for_province(province) if province else MAX_RADIUS_M
    min_samples = get_min_samples_for_province(province) if province else MIN_SAMPLES_PER_CELL
    
    df = df.copy()
    next_cluster_id = int(df["cluster_id"].max()) + 1
    max_iterations = 200  # Increased for Balochistan

    for _ in range(max_iterations):
        radii = _cluster_radii(df)
        if not radii:
            break

        cid_worst, r_worst = max(radii.items(), key=lambda kv: kv[1])
        if r_worst <= max_radius:
            break

        # Try to split this cluster while respecting min-samples constraint
        success, new_df, next_cluster_id = _split_cluster_if_feasible(
            df,
            cid_worst,
            next_cluster_id,
            min_samples_per_cell=min_samples,
        )
        if not success:
            # Cannot further reduce radius without violating min-samples;
            # stop and keep current configuration (soft constraint).
            break

        df = new_df
        # Ensure no sub-cluster fell below min size after the split
        df = _merge_small_cells(df, min_samples)

    return df


def _match_target_cells(df: pd.DataFrame, target_cells: int, min_samples: int = MIN_SAMPLES_PER_CELL) -> pd.DataFrame:
    """
    Adjust number of cells to match an *effective* target using split/merge
    operations, while respecting the min_samples_per_cell constraint.

    NOTE: This function assumes the caller has ALREADY applied the
    max-feasible cap: target_cells = min(spec_target, floor(N / min_samples)).
    """
    # Ensure contiguous integer ids for simplicity
    unique_ids = sorted(df["cluster_id"].unique())
    remap = {old: i for i, old in enumerate(unique_ids)}
    df = df.copy()
    df["cluster_id"] = df["cluster_id"].map(remap).astype(int)

    next_cluster_id = int(df["cluster_id"].max()) + 1

    def current_n_cells() -> int:
        return df["cluster_id"].nunique()

    # Track which clusters are unsplittable due to min_samples_per_cell
    unsplittable: set[int] = set()

    # If too few cells -> split largest clusters, but only if feasible
    while current_n_cells() < target_cells:
        sizes = df["cluster_id"].value_counts().sort_values(ascending=False)

        # Find the largest splittable cluster
        cid_candidate = None
        for cid in sizes.index:
            cid_int = int(cid)
            if cid_int not in unsplittable:
                cid_candidate = cid_int
                break

        if cid_candidate is None:
            # All clusters are unsplittable under the min-samples constraint
            break

        success, new_df, next_cluster_id = _split_cluster_if_feasible(
            df,
            cid_candidate,
            next_cluster_id,
            min_samples_per_cell=min_samples,
        )
        if not success:
            # Mark this cluster as unsplittable and try another
            unsplittable.add(cid_candidate)
            # If everything is now unsplittable, stop
            if len(unsplittable) == sizes.shape[0]:
                break
        else:
            df = new_df

    # If too many cells -> merge nearest small clusters
    while current_n_cells() > target_cells:
        # Compute centroids and sizes
        centroids = _cluster_centroids(df)
        sizes = df["cluster_id"].value_counts()

        # Order clusters by size ascending (smallest first)
        ordered_ids = sizes.sort_values(ascending=True).index.to_list()
        if len(ordered_ids) <= 1:
            break

        # Try merging smallest cluster into its nearest neighbor
        cid_small = int(ordered_ids[0])
        center_small = np.array(centroids[cid_small], dtype=float)

        other_ids = [cid for cid in ordered_ids if cid != cid_small]
        centers = np.array([centroids[cid] for cid in other_ids], dtype=float)

        deltas = centers - center_small[None, :]
        dists2 = np.einsum("ij,ij->i", deltas, deltas)
        nearest_idx = int(np.argmin(dists2))
        cid_target = int(other_ids[nearest_idx])

        df = df.copy()
        df.loc[df["cluster_id"] == cid_small, "cluster_id"] = cid_target

    # Re-normalize cluster ids to 0..K-1
    unique_ids = sorted(df["cluster_id"].unique())
    remap = {old: i for i, old in enumerate(unique_ids)}
    df["cluster_id"] = df["cluster_id"].map(remap).astype(int)

    return df


def main():
    GEOCELLS_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Geocell Step3 Summary\n")
    lines.append(f"- TARGET_MAX_RADIUS_M (best-effort): {MAX_RADIUS_M}")
    lines.append(f"- MIN_SAMPLES_PER_CELL (hard): {MIN_SAMPLES_PER_CELL}\n")

    for province, spec_target in TARGET_CELLS.items():
        print(f"\n=== Province: {province} (spec target cells: {spec_target}) ===")
        df = _load_province_cluster_file(province)
        if df is None:
            print(f"  No cluster file found for {province}, skipping.")
            continue

        n_samples = len(df)
        min_samples_prov = get_min_samples_for_province(province)
        max_feasible_cells = max(1, n_samples // min_samples_prov)
        desired_cells = min(spec_target, max_feasible_cells)

        # Step 6: enforce radius (province-specific)
        df_radius, stats = _enforce_radius(df, province=province)

        # Step 7: match target cell counts (using effective target)
        min_samples_prov = get_min_samples_for_province(province)
        df_final = _match_target_cells(df_radius, target_cells=desired_cells, min_samples=min_samples_prov)
        stats.n_cells_final = df_final["cluster_id"].nunique()

        # Final mandatory cleanups:
        # 1) Enforce min cell size via merge-only pass (province-specific)
        df_final = _merge_small_cells(df_final, min_samples_prov)
        # 2) Best-effort radius reduction with size cleanup (province-specific)
        df_final = _best_effort_radius_reduction(df_final, province=province)

        # Save final clusters
        stem = f"province_{province.replace(' ', '_')}"
        final_path = GEOCELLS_DIR / f"{stem}{FINAL_SUFFIX}"
        df_final.to_csv(final_path, index=False)

        # Recompute radii for final clusters for reporting
        radii_final = _cluster_radii(df_final)
        max_radius_final = max(radii_final.values()) if radii_final else 0.0
        sizes_final = df_final["cluster_id"].value_counts()
        min_size = int(sizes_final.min())
        max_size = int(sizes_final.max())

        lines.append(f"## {province}")
        lines.append(f"- Samples: {n_samples}")
        lines.append(f"- Spec target cells: {spec_target}")
        lines.append(f"- Max feasible cells (n/40): {max_feasible_cells}")
        lines.append(f"- Effective target cells: {desired_cells}")
        lines.append(f"- Cells before radius step: {stats.n_cells_before}")
        lines.append(f"- Cells after radius step: {stats.n_cells_after_radius}")
        lines.append(f"- Cells final: {stats.n_cells_final}")
        lines.append(
            f"- Max radius before: {stats.max_radius_before_m/1000:.2f} km"
        )
        lines.append(f"- Max radius after radius step: {stats.max_radius_after_m/1000:.2f} km")
        lines.append(f"- Max radius final: {max_radius_final/1000:.2f} km")
        lines.append(f"- Cell size range final: {min_size:,} – {max_size:,}")

        note_parts = []
        if spec_target > max_feasible_cells:
            note_parts.append("target capped by min_samples constraint")
        if stats.n_cells_final < desired_cells:
            note_parts.append("could not reach effective target due to split feasibility")
        if max_radius_final > MAX_RADIUS_M / 1000.0:
            note_parts.append(
                "radius could not be reduced to TARGET_MAX_RADIUS_M without violating min_samples"
            )

        if note_parts:
            lines.append(f"- Note: {'; '.join(note_parts)}\n")
        else:
            lines.append("- Note: targets met within constraints (size & radius)\n")

    SUMMARY_PATH.write_text("\n".join(lines))
    print(f"\nSummary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()


