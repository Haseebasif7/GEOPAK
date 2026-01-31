"""Step 9 — Assign final training fields"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

GEOCELLS_DIR = Path("pipeline/geocells")
CELL_METADATA_PATH = GEOCELLS_DIR / "cell_metadata.csv"
INPUT_CSV = Path("final_cleaned_merged.csv")
OUTPUT_CSV = Path("final_cleaned_with_cells_1.csv")


def _load_cell_metadata() -> pd.DataFrame:
    if not CELL_METADATA_PATH.exists():
        raise SystemExit(f"Cell metadata not found at {CELL_METADATA_PATH}")
    meta = pd.read_csv(CELL_METADATA_PATH)
    required = {
        "cell_id",
        "province",
        "province_id",
        "local_cluster_id",
        "center_lat",
        "center_lon",
    }
    if not required.issubset(meta.columns):
        raise ValueError(f"{CELL_METADATA_PATH} missing required columns {required}")
    return meta


def _build_assignment_table(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Build a table mapping (province, id) -> (cell_id, center_lat, center_lon)
    by joining final cluster assignments with cell metadata.
    """
    rows = []
    for province in meta["province"].unique():
        stem = f"province_{province.replace(' ', '_')}"
        clusters_path = GEOCELLS_DIR / f"{stem}_clusters_final.csv"
        if not clusters_path.exists():
            print(f"Warning: missing final clusters for {province}, skipping.")
            continue

        df_clu = pd.read_csv(clusters_path)
        if not {"id", "cluster_id"}.issubset(df_clu.columns):
            print(f"Warning: {clusters_path} missing id/cluster_id, skipping.")
            continue
        df_clu["cluster_id"] = df_clu["cluster_id"].astype(int)

        meta_sub = meta[meta["province"] == province][
            ["cell_id", "local_cluster_id", "center_lat", "center_lon"]
        ].copy()
        meta_sub = meta_sub.rename(columns={"local_cluster_id": "cluster_id"})

        merged = df_clu[["id", "cluster_id"]].merge(
            meta_sub, on="cluster_id", how="left"
        )
        merged["province"] = province

        rows.append(merged[["id", "province", "cell_id", "center_lat", "center_lon"]])

    if not rows:
        raise SystemExit("No assignment rows produced; check cluster files.")

    assign_df = pd.concat(rows, ignore_index=True)

    # Some (province, id) pairs can appear multiple times if the source
    # cluster files contain duplicates. In that case, we collapse them
    # by taking the first cell_id / center_lat / center_lon per pair.
    assign_df = (
        assign_df.groupby(["province", "id"], as_index=False)
        .agg(
            {
                "cell_id": "first",
                "center_lat": "first",
                "center_lon": "first",
            }
        )
    )

    return assign_df


def main():
    if not INPUT_CSV.exists():
        raise SystemExit(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    if "province" not in df.columns:
        raise SystemExit("Input CSV missing 'province' column")

    # Normalize province names to the canonical forms used in cell_metadata
    # (without modifying the original source CSV).
    province_normalization = {
        "Islamabad Capital Territory": "ICT",
    }
    df["province"] = df["province"].replace(province_normalization)

    meta = _load_cell_metadata()
    assign_df = _build_assignment_table(meta)

    # Merge: left join original data with assignments on (province, id)
    # Some (province, id) pairs in the original CSV can legitimately appear
    # multiple times (e.g., duplicate images or augmented views). Each of
    # them should map to the *same* cell, so we allow a many-to-one merge.
    merged = df.merge(
        assign_df,
        on=["id", "province"],
        how="left",
        validate="m:1",
    )

    # province_id from metadata (via province)
    prov_id_map: Dict[str, int] = {
        p: int(pid)
        for p, pid in meta[["province", "province_id"]]
        .drop_duplicates()
        .to_numpy()
    }
    merged["province_id"] = merged["province"].map(prov_id_map)

    # Rename center_lat/lon columns for clarity
    merged = merged.rename(
        columns={
            "center_lat": "cell_center_lat",
            "center_lon": "cell_center_lon",
        }
    )

    # Reorder columns: keep core fields first
    core_cols = [
        "id",
        "latitude",
        "longitude",
        "province",
        "province_id",
        "cell_id",
        "cell_center_lat",
        "cell_center_lon",
    ]
    other_cols = [c for c in merged.columns if c not in core_cols]
    merged = merged[core_cols + other_cols]

    OUTPUT_CSV.write_text(merged.to_csv(index=False))
    print(f"Wrote training CSV with cells: {OUTPUT_CSV} (rows: {len(merged):,})")


if __name__ == "__main__":
    main()


