import pandas as pd
from pathlib import Path
import json

# Paths
csv_path = Path(__file__).parent / "train_filtered.csv"
output_path = Path(__file__).parent / "missing_coordinates_landmarks.json"
output_txt_path = Path(__file__).parent / "missing_coordinates_landmarks.txt"

# Load CSV
print("Loading CSV file...")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows")

# Filter rows where latitude or longitude is missing
missing_coords = df[(df['latitude'].isna()) | (df['longitude'].isna())]
print(f"\nFound {len(missing_coords)} rows with missing coordinates")

# Get unique landmark names and their IDs
if len(missing_coords) > 0:
    missing_landmarks = missing_coords[['landmark_id', 'landmark_name']].drop_duplicates()
    missing_landmarks = missing_landmarks.sort_values('landmark_id')
    
    print(f"Found {len(missing_landmarks)} unique landmarks with missing coordinates:\n")
    
    # Display the landmarks
    for idx, row in missing_landmarks.iterrows():
        landmark_id = row['landmark_id']
        landmark_name = row['landmark_name']
        count = len(missing_coords[missing_coords['landmark_id'] == landmark_id])
        print(f"  ID: {landmark_id:6d} | Name: {landmark_name:50s} | Rows: {count}")
    
    # Save as JSON
    landmarks_list = [
        {
            'id': int(row['landmark_id']),
            'name': row['landmark_name'],
            'missing_rows': int(len(missing_coords[missing_coords['landmark_id'] == row['landmark_id']]))
        }
        for _, row in missing_landmarks.iterrows()
    ]
    
    with open(output_path, 'w') as f:
        json.dump(landmarks_list, f, indent=2)
    print(f"\n✓ Saved {len(landmarks_list)} landmarks to: {output_path}")
    
    # Save as simple text file (just names, one per line)
    with open(output_txt_path, 'w') as f:
        for _, row in missing_landmarks.iterrows():
            f.write(f"{row['landmark_name']}\n")
    print(f"✓ Saved landmark names to: {output_txt_path}")
    
    # Summary statistics
    total_missing_rows = len(missing_coords)
    total_rows = len(df)
    percentage = (total_missing_rows / total_rows) * 100
    
    print(f"\nSummary:")
    print(f"  Total rows: {total_rows}")
    print(f"  Rows with missing coordinates: {total_missing_rows} ({percentage:.2f}%)")
    print(f"  Unique landmarks with missing coordinates: {len(missing_landmarks)}")
    
else:
    print("✓ All landmarks have coordinates!")

