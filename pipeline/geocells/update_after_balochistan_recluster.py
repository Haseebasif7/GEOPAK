"""Update all files after Balochistan re-clustering."""

from pathlib import Path
import pandas as pd
import subprocess
import sys

GEOCELLS_DIR = Path("pipeline/geocells")
CELL_METADATA_PATH = GEOCELLS_DIR / "cell_metadata.csv"


def run_step4():
    """Run step 4 to regenerate metadata."""
    """Run step 4 to regenerate metadata."""
    print("STEP 4: Regenerating cell metadata")
    
    try:
        result = subprocess.run(
            [sys.executable, "pipeline/geocells/geocells_step4_metadata.py"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode != 0:
            print(f"⚠️  Step 4 returned code {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"❌ Error running step 4: {e}")
        return False


def run_step5():
    """Run step 5 to reassign cell_ids."""
    """Run step 5 to reassign cell_ids."""
    print("STEP 5: Reassigning global cell_ids")
    
    try:
        result = subprocess.run(
            [sys.executable, "pipeline/geocells/geocells_step5_assign_cells.py"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode != 0:
            print(f"⚠️  Step 5 returned code {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"❌ Error running step 5: {e}")
        return False


def update_train_test_csvs():
    """Update train.csv and test.csv with new cell assignments."""
    """Update train.csv and test.csv with new cell assignments."""
    print("UPDATING train.csv AND test.csv")
    
    # Load new assignments
    new_assignments_path = Path("final_cleaned_with_cells_1.csv")
    if not new_assignments_path.exists():
        print(f"❌ {new_assignments_path} not found. Run step 5 first.")
        return False
    
    new_assignments = pd.read_csv(new_assignments_path)
    print(f"Loaded new assignments: {len(new_assignments):,} rows")
    
    # Create mapping from id to cell info
    cell_mapping = new_assignments.set_index('id')[['cell_id', 'cell_center_lat', 'cell_center_lon']].to_dict('index')
    print(f"Created mapping for {len(cell_mapping):,} IDs")
    print()
    
    # Update train.csv
    train_df = pd.read_csv('train.csv')
    print(f"Train.csv: {len(train_df):,} rows")
    
    train_matched = train_df['id'].isin(cell_mapping.keys())
    print(f"Matched IDs: {train_matched.sum():,} / {len(train_df):,}")
    
    for idx, row in train_df.iterrows():
        if row['id'] in cell_mapping:
            cell_info = cell_mapping[row['id']]
            train_df.at[idx, 'cell_id'] = cell_info['cell_id']
            train_df.at[idx, 'cell_center_lat'] = cell_info['cell_center_lat']
            train_df.at[idx, 'cell_center_lon'] = cell_info['cell_center_lon']
    
    train_df.to_csv('train.csv', index=False)
    print(f"Updated train.csv")

    # Update test.csv
    test_df = pd.read_csv('test.csv')
    print(f"Test.csv: {len(test_df):,} rows")
    
    test_matched = test_df['id'].isin(cell_mapping.keys())
    print(f"Matched IDs: {test_matched.sum():,} / {len(test_df):,}")
    
    for idx, row in test_df.iterrows():
        if row['id'] in cell_mapping:
            cell_info = cell_mapping[row['id']]
            test_df.at[idx, 'cell_id'] = cell_info['cell_id']
            test_df.at[idx, 'cell_center_lat'] = cell_info['cell_center_lat']
            test_df.at[idx, 'cell_center_lon'] = cell_info['cell_center_lon']
    
    test_df.to_csv('test.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    print(f"Updated test.csv")
    
    return True


def verify_consistency():
    """Verify all files are consistent."""
    print("VERIFICATION")
    
    # Check cell_metadata
    cell_meta = pd.read_csv(CELL_METADATA_PATH)
    print(f"cell_metadata.csv: {len(cell_meta)} cells")
    
    # Check Balochistan
    baloch_meta = cell_meta[cell_meta['province'] == 'Balochistan']
    print(f"\nBalochistan in metadata:")
    print(f"  Cells: {len(baloch_meta)}")
    print(f"  Avg radius: {baloch_meta['radius_km'].mean():.2f} km")
    print(f"  Cell ID range: {baloch_meta['cell_id'].min()}-{baloch_meta['cell_id'].max()}")
    
    # Check train.csv
    train_df = pd.read_csv('train.csv')
    baloch_train = train_df[train_df['province'] == 'Balochistan']
    print(f"\nBalochistan in train.csv:")
    print(f"  Rows: {len(baloch_train):,}")
    print(f"  Unique cell_ids: {baloch_train['cell_id'].nunique()}")
    print(f"  Cell ID range: {baloch_train['cell_id'].min()}-{baloch_train['cell_id'].max()}")
    
    # Check test.csv
    test_df = pd.read_csv('test.csv')
    baloch_test = test_df[test_df['province'] == 'Balochistan']
    print(f"\nBalochistan in test.csv:")
    print(f"  Rows: {len(baloch_test):,}")
    print(f"  Unique cell_ids: {baloch_test['cell_id'].nunique()}")
    print(f"  Cell ID range: {baloch_test['cell_id'].min()}-{baloch_test['cell_id'].max()}")
    
    # Verify cell_ids match
    meta_cell_ids = set(baloch_meta['cell_id'].unique())
    train_cell_ids = set(baloch_train['cell_id'].unique())
    test_cell_ids = set(baloch_test['cell_id'].unique())
    
    if meta_cell_ids == train_cell_ids == test_cell_ids:
        print("All files are consistent!")
    else:
        print("Cell ID mismatch detected:")
        print(f"  Metadata: {sorted(meta_cell_ids)}")
        print(f"  Train: {sorted(train_cell_ids)}")
        print(f"  Test: {sorted(test_cell_ids)}")


def main():
    print("UPDATING FILES AFTER BALOCHISTAN RE-CLUSTERING")
    
    # Step 4: Regenerate metadata
    if not run_step4():
        print("❌ Step 4 failed. Please check errors above.")
        return
    
    # Step 5: Reassign cell_ids
    if not run_step5():
        print("❌ Step 5 failed. Please check errors above.")
        return
    
    # Update train.csv and test.csv
    if not update_train_test_csvs():
        print("❌ Failed to update train/test CSVs.")
        return
    
    # Verify
    verify_consistency()
    print("ALL UPDATES COMPLETE!")


if __name__ == "__main__":
    main()
