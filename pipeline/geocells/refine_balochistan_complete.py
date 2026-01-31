"""Complete Balochistan refinement pipeline."""

from pathlib import Path
import subprocess
import sys
import pandas as pd

def run_command(cmd, description):
    """Run a command and return success status."""
    """Run a command and return success status."""
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def update_train_test():
    """Update train.csv and test.csv with new assignments."""
    """Update train.csv and test.csv with new assignments."""
    print("UPDATING train.csv AND test.csv")
    
    new_assignments_path = Path("final_cleaned_with_cells_1.csv")
    if not new_assignments_path.exists():
        print(f"❌ {new_assignments_path} not found.")
        return False
    
    new_assignments = pd.read_csv(new_assignments_path)
    cell_mapping = new_assignments.set_index('id')[['cell_id', 'cell_center_lat', 'cell_center_lon']].to_dict('index')
    
    # Update train.csv
    train_df = pd.read_csv('train.csv')
    for idx, row in train_df.iterrows():
        if row['id'] in cell_mapping:
            cell_info = cell_mapping[row['id']]
            train_df.at[idx, 'cell_id'] = cell_info['cell_id']
            train_df.at[idx, 'cell_center_lat'] = cell_info['cell_center_lat']
            train_df.at[idx, 'cell_center_lon'] = cell_info['cell_center_lon']
    train_df.to_csv('train.csv', index=False)
    print(f"✅ Updated train.csv: {len(train_df):,} rows")
    
    # Update test.csv
    test_df = pd.read_csv('test.csv')
    for idx, row in test_df.iterrows():
        if row['id'] in cell_mapping:
            cell_info = cell_mapping[row['id']]
            test_df.at[idx, 'cell_id'] = cell_info['cell_id']
            test_df.at[idx, 'cell_center_lat'] = cell_info['cell_center_lat']
            test_df.at[idx, 'cell_center_lon'] = cell_info['cell_center_lon']
    test_df.to_csv('test.csv', index=False)
    print(f"✅ Updated test.csv: {len(test_df):,} rows")
    
    return True


def main():
    print("COMPLETE BALOCHISTAN REFINEMENT PIPELINE")
    
    steps = [
        ([sys.executable, "pipeline/geocells/geocells_step1.py"],
         "Step 1: HDBSCAN clustering (min_samples=5 for Balochistan)"),
        ([sys.executable, "pipeline/geocells/geocells_step2_merge_assign.py"],
         "Step 2: Merge undersized clusters and assign noise"),
        ([sys.executable, "pipeline/geocells/geocells_step3_radius_target.py"],
         "Step 3: Radius reduction and target cell matching (target=50, max_radius=60km)"),
        ([sys.executable, "pipeline/geocells/geocells_step4_metadata.py"],
         "Step 4: Generate cell metadata"),
        ([sys.executable, "pipeline/geocells/geocells_step5_assign_cells.py"],
         "Step 5: Assign global cell_ids"),
    ]
    
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            print(f"\n❌ Failed at: {desc}")
            print("Please check errors above and fix before continuing.")
            return
    
    # Update train.csv and test.csv
    if not update_train_test():
        print("\n❌ Failed to update train/test CSVs")
        return
    
    # Final verification
    # Final verification
    print("FINAL VERIFICATION")
    
    cell_meta = pd.read_csv('pipeline/geocells/cell_metadata.csv')
    baloch = cell_meta[cell_meta['province'] == 'Balochistan']
    
    print(f"Balochistan in cell_metadata.csv:")
    print(f"  Cells: {len(baloch)} (target: 48-55)")
    print(f"  Avg radius: {baloch['radius_km'].mean():.2f} km (target: 55-65 km)")
    
    train_df = pd.read_csv('train.csv')
    baloch_train = train_df[train_df['province'] == 'Balochistan']
    print(f"Balochistan in train.csv:")
    print(f"  Rows: {len(baloch_train):,}")
    print(f"  Unique cell_ids: {baloch_train['cell_id'].nunique()}")
    
    test_df = pd.read_csv('test.csv')
    baloch_test = test_df[test_df['province'] == 'Balochistan']
    print(f"Balochistan in test.csv:")
    print(f"  Rows: {len(baloch_test):,}")
    print(f"  Unique cell_ids: {baloch_test['cell_id'].nunique()}")
    
    print("Complete refinement pipeline finished!")


if __name__ == "__main__":
    main()
