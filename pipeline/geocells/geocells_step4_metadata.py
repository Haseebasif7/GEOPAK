"""
Step 8 — Save cell metadata (per province)

Reads final per-province cluster assignments and:
  - Computes per-cell centroids in UTM and lat/lon
  - Computes per-cell radius_km (max haversine distance to centroid)
  - Computes neighbor_cell_ids via K-NN on centroids (per province)
  - Assigns global cell_id and province_id

Inputs (under pipeline/geocells/):
  - province_<name>_clusters_final.csv  (id, x_m, y_m, cluster_id)

Outputs:
  - pipeline/geocells/cell_metadata.csv
      columns:
        cell_id
        province
        province_id
        local_cluster_id
        center_lat
        center_lon
        radius_km
        neighbor_cell_ids   (comma-separated global cell_ids)

Original merged_training_data_with_province.csv is NOT modified.

Run manually from repo root:
    python pipeline/geocells_step4_metadata.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

GEOCELLS_DIR = Path("pipeline/geocells")
METADATA_PATH = GEOCELLS_DIR / "cell_metadata.csv"

# Province order and IDs (stable across pipeline)
PROVINCES = [
    "Sindh",
    "Punjab",
    "Khyber Pakhtunkhwa",
    "ICT",
    "Gilgit-Baltistan",
    "Balochistan",
    "Azad Kashmir",
]
PROVINCE_ID = {name: i for i, name in enumerate(PROVINCES)}

# Per-province UTM EPSG mapping (must match geocells_step1.py)
EPSG_MAP = {
    "Sindh": 32642,
    "Punjab": 32643,
    "Khyber Pakhtunkhwa": 32642,
    "ICT": 32643,
    "Gilgit-Baltistan": 32644,
    "Balochistan": 32642,
    "Azad Kashmir": 32643,
}

K_NEIGHBORS = 8  # number of neighbor cells per cell (best-effort)


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """Vectorized haversine distance in kilometers from arrays (lat1, lon1) to a single point (lat2, lon2)."""
    R = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def _load_final_clusters(province: str) -> pd.DataFrame | None:
    stem = f"province_{province.replace(' ', '_')}"
    path = GEOCELLS_DIR / f"{stem}_clusters_final.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"id", "x_m", "y_m", "cluster_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing required columns {required}")
    df["cluster_id"] = df["cluster_id"].astype(int)
    return df


def _compute_cell_metadata_for_province(
    province: str,
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For a single province:
      - back-project x_m/y_m to lat/lon
      - aggregate per local cluster_id into cell metadata
      - return (points_with_latlon, cell_metadata_local)
    """
    epsg = EPSG_MAP[province]
    # Convert UTM → lat/lon using GeoPandas
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["x_m"], df["y_m"]),
        crs=f"EPSG:{epsg}",
    ).to_crs("EPSG:4326")

    gdf["lat"] = gdf.geometry.y
    gdf["lon"] = gdf.geometry.x
    gdf = gdf.drop(columns=["geometry"])

    # Aggregate per local cluster_id
    rows: List[Dict] = []
    for cid, grp in gdf.groupby("cluster_id"):
        lat_arr = grp["lat"].to_numpy()
        lon_arr = grp["lon"].to_numpy()

        center_lat = float(lat_arr.mean())
        center_lon = float(lon_arr.mean())

        # Radius in km via haversine to centroid
        if len(grp) > 0:
            dists_km = haversine_km(lat_arr, lon_arr, center_lat, center_lon)
            radius_km = float(dists_km.max())
        else:
            radius_km = 0.0

        rows.append(
            dict(
                province=province,
                province_id=PROVINCE_ID[province],
                local_cluster_id=int(cid),
                center_lat=center_lat,
                center_lon=center_lon,
                radius_km=radius_km,
                # neighbor_cell_ids will be filled after global cell_ids are assigned
            )
        )

    meta_local = pd.DataFrame(rows)
    return gdf, meta_local


def main():
    GEOCELLS_DIR.mkdir(parents=True, exist_ok=True)

    # Accumulate per-province metadata
    all_meta: List[pd.DataFrame] = []
    cell_id_counter = 0

    for province in PROVINCES:
        print(f"\n=== Province: {province} ===")
        df = _load_final_clusters(province)
        if df is None:
            print("  No final clusters file found, skipping.")
            continue

        print(f"  Samples: {len(df):,}")
        print(f"  Unique local clusters: {df['cluster_id'].nunique():,}")

        points_with_latlon, meta_local = _compute_cell_metadata_for_province(province, df)

        # Assign global cell_ids for this province
        n_cells = len(meta_local)
        global_ids = np.arange(cell_id_counter, cell_id_counter + n_cells, dtype=int)
        meta_local = meta_local.copy()
        meta_local["cell_id"] = global_ids

        cell_id_counter += n_cells
        all_meta.append(meta_local)

    if not all_meta:
        raise SystemExit("No province metadata produced; check inputs.")

    meta = pd.concat(all_meta, ignore_index=True)

    # Compute neighbor_cell_ids per province (within province)
    neighbor_ids_col: List[str] = []
    for province in PROVINCES:
        mask = meta["province"] == province
        sub = meta.loc[mask].copy()
        if sub.empty:
            continue

        coords = sub[["center_lat", "center_lon"]].to_numpy()
        ids = sub["cell_id"].to_numpy()

        # Pairwise distances (in km) between cell centroids
        n = len(sub)
        dmat = np.zeros((n, n), dtype=float)
        for i in range(n):
            dmat[i] = haversine_km(
                coords[:, 0], coords[:, 1], coords[i, 0], coords[i, 1]
            )

        # For each cell, take K nearest (excluding self)
        neighbor_lists: Dict[int, str] = {}
        for i, cid in enumerate(ids):
            order = np.argsort(dmat[i])
            # exclude self (distance 0)
            order = order[order != i]
            k = min(K_NEIGHBORS, len(order))
            nbr_ids = ids[order[:k]]
            neighbor_lists[int(cid)] = ",".join(str(x) for x in nbr_ids)

        # Assign back
        meta.loc[mask, "neighbor_cell_ids"] = meta.loc[mask, "cell_id"].map(
            neighbor_lists
        )

    # Reorder columns nicely
    meta = meta[
        [
            "cell_id",
            "province",
            "province_id",
            "local_cluster_id",
            "center_lat",
            "center_lon",
            "radius_km",
            "neighbor_cell_ids",
        ]
    ].sort_values(["province_id", "cell_id"])

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(METADATA_PATH, index=False)
    print(f"\nWrote cell metadata: {METADATA_PATH} (cells: {len(meta):,})")


if __name__ == "__main__":
    main()


