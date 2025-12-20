import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Paths
csv_path = Path(__file__).parent / "train_filtered.csv"
download_dir = Path(__file__).parent / "images"
download_dir.mkdir(exist_ok=True)

# Load the filtered CSV
print("Loading CSV file...")
df = pd.read_csv(csv_path)
print(f"Found {len(df)} images to download")

# Download function
def download_image(row):
    """Download a single image"""
    image_id = row['id']
    url = row['url']
    landmark_id = row['landmark_id']
    
    # Create filename: {image_id}_{landmark_id}.jpg (or keep original extension)
    file_ext = Path(url).suffix or '.jpg'
    filename = f"{image_id}_{landmark_id}{file_ext}"
    filepath = download_dir / filename
    
    # Skip if already downloaded
    if filepath.exists():
        return {'id': image_id, 'status': 'skipped', 'url': url}
    
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        # Save image
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return {'id': image_id, 'status': 'success', 'url': url}
    except Exception as e:
        return {'id': image_id, 'status': 'error', 'url': url, 'error': str(e)}

# Download with progress bar and parallel processing
print("Starting downloads...")
max_workers = 10  # Adjust based on your connection speed
results = []

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all download tasks
    future_to_row = {executor.submit(download_image, row): row for _, row in df.iterrows()}
    
    # Process completed downloads with progress bar
    with tqdm(total=len(df), desc="Downloading") as pbar:
        for future in as_completed(future_to_row):
            result = future.result()
            results.append(result)
            pbar.update(1)

# Print summary
success_count = sum(1 for r in results if r['status'] == 'success')
error_count = sum(1 for r in results if r['status'] == 'error')
skipped_count = sum(1 for r in results if r['status'] == 'skipped')

print(f"\nDownload Summary:")
print(f"  Success: {success_count}")
print(f"  Errors: {error_count}")
print(f"  Skipped (already exists): {skipped_count}")

# Save error log if there are errors
if error_count > 0:
    error_log = download_dir / "download_errors.txt"
    with open(error_log, 'w') as f:
        for r in results:
            if r['status'] == 'error':
                f.write(f"{r['id']}: {r['url']} - {r.get('error', 'Unknown error')}\n")
    print(f"\nError log saved to: {error_log}")