"""
Province-aware geocell construction (Step 1-3):
1) Split by province
2) Project lat/lon to per-province UTM meters
3) Initial HDBSCAN clustering per province

Outputs (under pipeline/geocells/):
- province_<name>_utm.csv         (id, lat, lon, x_m, y_m, path)
- province_<name>_clusters.csv    (id, x_m, y_m, cluster_id)
- geocell_step1_summary.md        (counts, noise, EPSG, params)

Note: Does NOT modify merged_training_data_with_province.csv
Run manually from repo root:
    python pipeline/geocells_step1.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import hdbscan

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
INPUT_CSV = Path("merged_training_data_with_province_backup_before_ICT_fix.csv")
OUT_DIR = Path("pipeline/geocells")
SUMMARY_PATH = OUT_DIR / "geocell_step1_summary.md"

# Per-province UTM EPSG mapping
EPSG_MAP = {
    "Sindh": 32642,
    "Punjab": 32643,
    "Khyber Pakhtunkhwa": 32642,
    "ICT": 32643,
    "Gilgit-Baltistan": 32644,
    "Balochistan": 32642,
    "Azad Kashmir": 32643,
}

HDBSCAN_KWARGS = dict(
    min_cluster_size=40,
    min_samples=10,
    metric="euclidean",
)


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


def save_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    if "province" not in df.columns:
        raise SystemExit("province column missing in input CSV")

    # Drop rows without province just in case
    df = df.dropna(subset=["province"])
    print(f"Rows after province drop: {len(df):,}")

    summary = []
    summary.append("# Geocell Step1 Summary\n")
    summary.append(f"- Input: {INPUT_CSV}")
    summary.append(
        f"- HDBSCAN: min_cluster_size={HDBSCAN_KWARGS['min_cluster_size']}, "
        f"min_samples={HDBSCAN_KWARGS['min_samples']}, metric={HDBSCAN_KWARGS['metric']}\n"
    )

    for prov, g in df.groupby("province"):
        epsg = EPSG_MAP.get(prov)
        if epsg is None:
            print(f"Skipping province with no EPSG mapping: {prov}")
            continue

        print(f"\nProvince: {prov} (n={len(g):,}), EPSG:{epsg}")
        gproj = project_province(g, epsg)

        # Save projected points
        utm_path = OUT_DIR / f"province_{prov.replace(' ', '_')}_utm.csv"
        save_csv(utm_path, gproj[["id", "latitude", "longitude", "x_m", "y_m", "path"]])

        # HDBSCAN clustering
        labels = run_hdbscan(gproj[["x_m", "y_m"]].to_numpy())
        gproj["cluster_id"] = labels

        clusters_path = OUT_DIR / f"province_{prov.replace(' ', '_')}_clusters.csv"
        save_csv(clusters_path, gproj[["id", "x_m", "y_m", "cluster_id"]])

        # Summary stats
        unique, counts = np.unique(labels, return_counts=True)
        noise = counts[unique == -1][0] if (-1 in unique) else 0
        n_clusters = int((unique != -1).sum())

        summary.append(f"## {prov}")
        summary.append(f"- EPSG: {epsg}")
        summary.append(f"- Samples: {len(gproj):,}")
        summary.append(f"- Clusters (excl noise): {n_clusters}")
        summary.append(f"- Noise: {noise:,}\n")

    SUMMARY_PATH.write_text("\n".join(summary))
    print(f"\nSummary written to {SUMMARY_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()

