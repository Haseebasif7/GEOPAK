"""Re-cluster Balochistan with adjusted parameters to reduce cell size."""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import hdbscan
from sklearn.cluster import KMeans
from typing import Dict, Tuple

# Config
INPUT_CSV = Path("final_cleaned_merged.csv")  # Run from repo root
OUT_DIR = Path("pipeline/geocells")
PROVINCE = "Balochistan"
EPSG = 32642  # UTM zone 42N for Balochistan

# Adjusted HDBSCAN parameters for more clusters
HDBSCAN_KWARGS = dict(
    min_cluster_size=40,  # Keep same
    min_samples=6,  # Lowered from 10 to get more clusters
    metric="euclidean",
)

# Target cells for Balochistan
TARGET_CELLS = 50  # Target 45-55, using 50 as middle
MAX_RADIUS_M = 60_000.0  # 60 km target (slightly above 50-65 avg target)
MIN_SAMPLES_PER_CELL = 40


def project_province(df: pd.DataFrame, epsg: int) -> gpd.GeoDataFrame:
    """Project province DataFrame to UTM meters."""
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    gdf = gdf.to_crs(epsg=epsg)
    gdf["x_m"] = gdf.geometry.x
    gdf["y_m"] = gdf.geometry.y
    gdf = gdf.drop(columns=["geometry"])
    return gdf


def run_hdbscan(coords: np.ndarray) -> np.ndarray:
    """Run HDBSCAN on (x_m, y_m) array."""
    clusterer = hdbscan.HDBSCAN(**HDBSCAN_KWARGS)
    return clusterer.fit_predict(coords)


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
    """Merge clusters with < 40 samples into nearest neighbor."""
    df = df.copy()
    
    while True:
        sizes = df["cluster_id"].value_counts()
        min_size = int(sizes.min())
        if min_size >= MIN_SAMPLES_PER_CELL or len(sizes) <= 1:
            break
        
        # Smallest cluster
        cid_small = int(sizes.idxmin())
        centroids = _cluster_centroids(df)
        center_small = np.array(centroids[cid_small], dtype=float)
        
        # Find nearest neighbor
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


def _assign_noise_to_nearest(df: pd.DataFrame) -> pd.DataFrame:
    """Assign noise points (cluster_id=-1) to nearest cluster."""
    df = df.copy()
    noise_mask = df["cluster_id"] == -1
    
    if not noise_mask.any():
        return df
    
    centroids = _cluster_centroids(df[~noise_mask])
    if not centroids:
        # No valid clusters, assign all noise to cluster 0
        df.loc[noise_mask, "cluster_id"] = 0
        return df
    
    noise_points = df.loc[noise_mask, ["x_m", "y_m"]].to_numpy()
    cluster_centers = np.array(list(centroids.values()), dtype=float)
    cluster_ids = np.array(list(centroids.keys()), dtype=int)
    
    for idx in df.index[noise_mask]:
        point = df.loc[idx, ["x_m", "y_m"]].to_numpy()
        deltas = cluster_centers - point[None, :]
        dists2 = np.einsum("ij,ij->i", deltas, deltas)
        nearest_idx = int(np.argmin(dists2))
        df.at[idx, "cluster_id"] = cluster_ids[nearest_idx]
    
    # Re-normalize
    unique_ids = sorted(df["cluster_id"].unique())
    remap = {old: i for i, old in enumerate(unique_ids)}
    df["cluster_id"] = df["cluster_id"].map(remap).astype(int)
    return df


def _split_cluster_if_feasible(
    df: pd.DataFrame,
    cid: int,
    next_cluster_id: int,
    min_samples_per_cell: int,
) -> tuple[bool, pd.DataFrame, int]:
    """Split cluster if both halves would have >= min_samples_per_cell."""
    mask = df["cluster_id"] == cid
    sub = df.loc[mask, ["x_m", "y_m"]].to_numpy()
    n = len(sub)
    if n < 2 * min_samples_per_cell:
        return False, df, next_cluster_id
    
    km = KMeans(n_clusters=2, random_state=0, n_init="auto")
    labels = km.fit_predict(sub)
    
    counts = np.bincount(labels, minlength=2)
    if counts.min() < min_samples_per_cell:
        return False, df, next_cluster_id
    
    new_df = df.copy()
    new_id = next_cluster_id
    cluster_index = new_df.index[mask]
    indices_label1 = cluster_index[labels == 1]
    new_df.loc[indices_label1, "cluster_id"] = new_id
    return True, new_df, new_id + 1


def _best_effort_radius_reduction(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce radius by splitting large clusters when feasible."""
    df = df.copy()
    next_cluster_id = int(df["cluster_id"].max()) + 1
    max_iterations = 200
    
    for _ in range(max_iterations):
        radii = _cluster_radii(df)
        if not radii:
            break
        
        cid_worst, r_worst = max(radii.items(), key=lambda kv: kv[1])
        if r_worst <= MAX_RADIUS_M:
            break
        
        success, new_df, next_cluster_id = _split_cluster_if_feasible(
            df, cid_worst, next_cluster_id, MIN_SAMPLES_PER_CELL
        )
        if not success:
            break
        
        df = new_df
        df = _merge_undersized_clusters(df)
    
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
            df, cid_candidate, next_cluster_id, MIN_SAMPLES_PER_CELL
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
    print("RE-CLUSTERING BALOCHISTAN")
    
    # Load data
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    baloch_df = df[df["province"] == PROVINCE].copy()
    print(f"Balochistan samples: {len(baloch_df):,}")
    
    # Step 1: Project to UTM
    print(f"\nStep 1: Projecting to UTM (EPSG:{EPSG})...")
    gproj = project_province(baloch_df, EPSG)
    
    # Step 2: HDBSCAN clustering
    labels = run_hdbscan(gproj[["x_m", "y_m"]].to_numpy())
    gproj["cluster_id"] = labels
    
    unique, counts = np.unique(labels, return_counts=True)
    noise = counts[unique == -1][0] if (-1 in unique) else 0
    n_clusters = int((unique != -1).sum())
    
    # Step 3: Merge undersized and assign noise
    print("Step 3: Merging undersized clusters and assigning noise...")
    gproj = _merge_undersized_clusters(gproj)
    gproj = _assign_noise_to_nearest(gproj)
    
    gproj = _match_target_cells(gproj, desired_cells)
    
    # Final cleanup
    print("\nStep 6: Final cleanup...")
    gproj = _merge_undersized_clusters(gproj)
    gproj = _best_effort_radius_reduction(gproj)
    
    # Final stats
    radii_final = _cluster_radii(gproj)
    max_radius_final = max(radii_final.values()) if radii_final else 0.0
    avg_radius_final = np.mean(list(radii_final.values())) / 1000.0
    n_cells_final = gproj["cluster_id"].nunique()
    
    # Final stats
    radii_final = _cluster_radii(gproj)
    max_radius_final = max(radii_final.values()) if radii_final else 0.0
    avg_radius_final = np.mean(list(radii_final.values())) / 1000.0
    n_cells_final = gproj["cluster_id"].nunique()

    print("FINAL RESULTS")
    print(f"Final cells: {n_cells_final}")
    print(f"Max radius: {max_radius_final/1000:.2f} km")
    print(f"Average radius: {avg_radius_final:.2f} km")
    print(f"Radius range: {min(radii_final.values())/1000:.2f} - {max_radius_final/1000:.2f} km")
    
    # Save output
    stem = f"province_{PROVINCE.replace(' ', '_')}"
    final_path = OUT_DIR / f"{stem}_clusters_final.csv"
    gproj[["id", "x_m", "y_m", "cluster_id"]].to_csv(final_path, index=False)
    # Also save UTM version for reference
    utm_path = OUT_DIR / f"{stem}_utm.csv"
    gproj[["id", "latitude", "longitude", "x_m", "y_m", "path"]].to_csv(utm_path, index=False)

    print("Re-clustering complete!")


if __name__ == "__main__":
    main()
