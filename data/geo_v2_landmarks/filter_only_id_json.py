import json
import pandas as pd
from pathlib import Path

# Paths
json_path = Path(__file__).parent / "pk.json"
csv_path = Path(__file__).parent / "train.csv"
output_path = Path(__file__).parent / "train_filtered.csv"

# Load JSON and extract IDs as a set for fast lookup
print("Loading JSON file...")
with open(json_path, 'r') as f:
    landmarks = json.load(f)

# Create a set of landmark IDs for O(1) lookup
landmark_ids = {item['id'] for item in landmarks}
print(f"Found {len(landmark_ids)} unique landmark IDs in JSON")

# Read CSV in chunks for memory efficiency
print("Processing CSV file...")
chunk_size = 100000  # Process 100k rows at a time
filtered_chunks = []

for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
    # Filter chunk: keep only rows where landmark_id is in our set
    filtered_chunk = chunk[chunk['landmark_id'].isin(landmark_ids)]
    if len(filtered_chunk) > 0:
        filtered_chunks.append(filtered_chunk)
    print(f"Processed chunk: {len(chunk)} rows -> {len(filtered_chunk)} matches")

# Combine all filtered chunks
if filtered_chunks:
    print("Combining filtered data...")
    filtered_df = pd.concat(filtered_chunks, ignore_index=True)
    
    # Save filtered data
    print(f"Saving filtered data ({len(filtered_df)} rows) to {output_path}...")
    filtered_df.to_csv(output_path, index=False)
    print(f"Done! Filtered CSV saved to {output_path}")
else:
    print("No matching rows found!")

