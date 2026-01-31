"""Refine existing Balochistan clusters to reach target."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Dict, Tuple

GEOCELLS_DIR = Path("pipeline/geocells")
PROVINCE = "Balochistan"
STEM = f"province_{PROVINCE.replace(' ', '_')}"
INPUT_PATH = GEOCELLS_DIR / f"{STEM}_clusters_final.csv"
OUTPUT_PATH = GEOCELLS_DIR / f"{STEM}_clusters_final.csv"

TARGET_CELLS = 50  # Target 48-55, using 50
MAX_RADIUS_M = 60_000.0  # 60 km
MIN_SAMPLES_PER_CELL = 40
MIN_SAMPLES_FOR_SPLIT = 35  # Lower threshold for splitting (allows more aggressive splitting)


def _cluster_centroids(df: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
    centroids: Dict[int, Tuple[float, float]] = {}
    for cid, g in df.groupby("cluster_id"):
        centroids[int(cid)] = (float(g["x_m"].mean()), float(g["y_m"].mean()))
    return centroids


def _cluster_radii(df: pd.DataFrame) -> Dict[int, float]:
    """Return per-cluster max radius in meters."""
    radii: Dict[int, float] = {}
    for cid, g in df.groupby("cluster_id"):
        center_x = float(g["x_m"].mean())
        center_y = float(g["y_m"].mean())
        dx = g["x_m"].to_numpy() - center_x
        dy = g["y_m"].to_numpy() - center_y
        dist = np.sqrt(dx * dx + dy * dy)
        radii[int(cid)] = float(dist.max()) if len(dist) else 0.0
    return radii


def _merge_undersized_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Merge clusters with < 40 samples."""
    df = df.copy()
    
    while True:
        sizes = df["cluster_id"].value_counts()
        min_size = int(sizes.min())
        if min_size >= MIN_SAMPLES_PER_CELL or len(sizes) <= 1:
            break
        
        cid_small = int(sizes.idxmin())
        centroids = _cluster_centroids(df)
        center_small = np.array(centroids[cid_small], dtype=float)
        
        other_ids = [cid for cid in centroids.keys() if cid != cid_small]
        if not other_ids:
            break
        
        centers = np.array([centroids[cid] for cid in other_ids], dtype=float)
        deltas = centers - center_small[None, :]
        dists2 = np.einsum("ij,ij->i", deltas, deltas)
        nearest_idx = int(np.argmin(dists2))
        cid_target = int(other_ids[nearest_idx])
        
        df.loc[df["cluster_id"] == cid_small, "cluster_id"] = cid_target
    
    # Re-normalize
    unique_ids = sorted(df["cluster_id"].unique())
    remap = {old: i for i, old in enumerate(unique_ids)}
    df["cluster_id"] = df["cluster_id"].map(remap).astype(int)
    return df


def _split_cluster_if_feasible(
    df: pd.DataFrame,
    cid: int,
    next_cluster_id: int,
) -> tuple[bool, pd.DataFrame, int]:
    """Split cluster if both halves would have >= MIN_SAMPLES_FOR_SPLIT."""
    mask = df["cluster_id"] == cid
    sub = df.loc[mask, ["x_m", "y_m"]].to_numpy()
    n = len(sub)
    if n < 2 * MIN_SAMPLES_FOR_SPLIT:
        return False, df, next_cluster_id
    
    km = KMeans(n_clusters=2, random_state=0, n_init="auto")
    labels = km.fit_predict(sub)
    
    counts = np.bincount(labels, minlength=2)
    if counts.min() < MIN_SAMPLES_FOR_SPLIT:
        return False, df, next_cluster_id
    
    new_df = df.copy()
    new_id = next_cluster_id
    cluster_index = new_df.index[mask]
    indices_label1 = cluster_index[labels == 1]
    new_df.loc[indices_label1, "cluster_id"] = new_id
    return True, new_df, new_id + 1


def _aggressive_radius_reduction(df: pd.DataFrame) -> pd.DataFrame:
    """Aggressively reduce radius by splitting large clusters."""
    df = df.copy()
    next_cluster_id = int(df["cluster_id"].max()) + 1
    max_iterations = 400  # Very aggressive
    
    for iteration in range(max_iterations):
        radii = _cluster_radii(df)
        if not radii:
            break
        
        # Sort by radius descending
        sorted_radii = sorted(radii.items(), key=lambda kv: kv[1], reverse=True)
        
        # Try to split the worst clusters
        found_split = False
        for cid_worst, r_worst in sorted_radii:
            if r_worst <= MAX_RADIUS_M:
                break
            
            success, new_df, next_cluster_id = _split_cluster_if_feasible(
                df, cid_worst, next_cluster_id
            )
            if success:
                df = new_df
                df = _merge_undersized_clusters(df)
                found_split = True
                break
        
        if not found_split:
            break
    
    return df


def _split_large_clusters_aggressively(df: pd.DataFrame, target_cells: int) -> pd.DataFrame:
    """Split large clusters aggressively to reach target cell count."""
    df = df.copy()
    next_cluster_id = int(df["cluster_id"].max()) + 1
    max_iterations = 300  # More iterations
    
    for iteration in range(max_iterations):
        current_cells = df["cluster_id"].nunique()
        if current_cells >= target_cells:
            break
        
        # Get cluster sizes, sort by size descending
        sizes = df["cluster_id"].value_counts().sort_values(ascending=False)
        
        # Find largest splittable cluster (using lower threshold)
        found_split = False
        for cid, size in sizes.items():
            if size < 2 * MIN_SAMPLES_FOR_SPLIT:
                continue
            
            success, new_df, next_cluster_id = _split_cluster_if_feasible(
                df, int(cid), next_cluster_id
            )
            if success:
                df = new_df
                # Only merge if below absolute minimum (40), not the split threshold (35)
                df = _merge_undersized_clusters(df)
                found_split = True
                break
        
        if not found_split:
            break
    
    return df


def _match_target_cells(df: pd.DataFrame, target_cells: int) -> pd.DataFrame:
    """Adjust number of cells to match target."""
    unique_ids = sorted(df["cluster_id"].unique())
    remap = {old: i for i, old in enumerate(unique_ids)}
    df = df.copy()
    df["cluster_id"] = df["cluster_id"].map(remap).astype(int)
    
    next_cluster_id = int(df["cluster_id"].max()) + 1
    unsplittable: set[int] = set()
    
    # If too few cells -> split largest clusters
    while df["cluster_id"].nunique() < target_cells:
        sizes = df["cluster_id"].value_counts().sort_values(ascending=False)
        
        cid_candidate = None
        for cid in sizes.index:
            cid_int = int(cid)
            if cid_int not in unsplittable:
                cid_candidate = cid_int
                break
        
        if cid_candidate is None:
            break
        
        success, new_df, next_cluster_id = _split_cluster_if_feasible(
            df, cid_candidate, next_cluster_id
        )
        if not success:
            unsplittable.add(cid_candidate)
            if len(unsplittable) == sizes.shape[0]:
                break
        else:
            df = new_df
            df = _merge_undersized_clusters(df)
    
    # If too many cells -> merge nearest small clusters
    while df["cluster_id"].nunique() > target_cells:
        centroids = _cluster_centroids(df)
        sizes = df["cluster_id"].value_counts()
        ordered_ids = sizes.sort_values(ascending=True).index.to_list()
        
        if len(ordered_ids) <= 1:
            break
        
        cid_small = int(ordered_ids[0])
        center_small = np.array(centroids[cid_small], dtype=float)
        
        other_ids = [cid for cid in ordered_ids if cid != cid_small]
        centers = np.array([centroids[cid] for cid in other_ids], dtype=float)
        
        deltas = centers - center_small[None, :]
        dists2 = np.einsum("ij,ij->i", deltas, deltas)
        nearest_idx = int(np.argmin(dists2))
        cid_target = int(other_ids[nearest_idx])
        
        df.loc[df["cluster_id"] == cid_small, "cluster_id"] = cid_target
        df = _merge_undersized_clusters(df)
    
    # Re-normalize
    unique_ids = sorted(df["cluster_id"].unique())
    remap = {old: i for i, old in enumerate(unique_ids)}
    df["cluster_id"] = df["cluster_id"].map(remap).astype(int)
    return df


def main():
    print("REFINING EXISTING BALOCHISTAN CLUSTERS")
    
    if not INPUT_PATH.exists():
        print(f"❌ Input file not found: {INPUT_PATH}")
        print("   Please run the clustering pipeline first.")
        return
    
    # Load existing clusters
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    
    # Initial stats
    radii_before = _cluster_radii(df)
    max_radius_before = max(radii_before.values()) if radii_before else 0.0
    avg_radius_before = np.mean(list(radii_before.values())) / 1000.0
    # Initial stats
    radii_before = _cluster_radii(df)
    max_radius_before = max(radii_before.values()) if radii_before else 0.0
    avg_radius_before = np.mean(list(radii_before.values())) / 1000.0
    print(f"  Max radius: {max_radius_before/1000:.2f} km")
    print(f"  Avg radius: {avg_radius_before:.2f} km")
    
    # Step 1: Aggressively split large clusters to reach target cell count
    # Step 1: Aggressively split large clusters to reach target cell count
    print(f"Step 1: Aggressively splitting large clusters to reach {TARGET_CELLS} cells...")
    df = _split_large_clusters_aggressively(df, TARGET_CELLS)
    
    # Step 2: Aggressive radius reduction
    # Step 2: Aggressive radius reduction
    print(f"Step 2: Aggressive radius reduction (target: {MAX_RADIUS_M/1000:.0f} km max)...")
    df = _aggressive_radius_reduction(df)
    radii_after = _cluster_radii(df)
    max_radius_after = max(radii_after.values()) if radii_after else 0.0
    avg_radius_after = np.mean(list(radii_after.values())) / 1000.0
    print(f"  Clusters: {df['cluster_id'].nunique()}")
    print(f"  Max radius: {max_radius_after/1000:.2f} km")
    print(f"  Avg radius: {avg_radius_after:.2f} km")
    
    # Step 3: Fine-tune to match target cell count
    # Step 3: Fine-tune to match target cell count
    print(f"Step 3: Fine-tuning to match target cell count ({TARGET_CELLS})...")
    n_samples = len(df)
    max_feasible = max(1, n_samples // MIN_SAMPLES_PER_CELL)
    desired_cells = min(TARGET_CELLS, max_feasible)
    
    df = _match_target_cells(df, desired_cells)
    
    # Final cleanup
    print("Step 4: Final cleanup...")
    df = _merge_undersized_clusters(df)
    df = _aggressive_radius_reduction(df)
    
    # Final stats
    radii_final = _cluster_radii(df)
    max_radius_final = max(radii_final.values()) if radii_final else 0.0
    avg_radius_final = np.mean(list(radii_final.values())) / 1000.0
    n_cells_final = df["cluster_id"].nunique()
    
    # Final stats
    radii_final = _cluster_radii(df)
    max_radius_final = max(radii_final.values()) if radii_final else 0.0
    avg_radius_final = np.mean(list(radii_final.values())) / 1000.0
    n_cells_final = df["cluster_id"].nunique()

    print("FINAL RESULTS")
    print(f"Final cells: {n_cells_final}")
    print(f"Max radius: {max_radius_final/1000:.2f} km")
    print(f"Average radius: {avg_radius_final:.2f} km")
    
    # Save
    # Save
    df[["id", "x_m", "y_m", "cluster_id"]].to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
