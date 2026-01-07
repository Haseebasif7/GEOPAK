import csv
from pathlib import Path
from collections import defaultdict
import os

# Paths
CSV_PATH = Path(__file__).parent / "train_filtered_pak.csv"
IMAGES_BASE_DIR = Path.home() / "Desktop" / "datasets" / "pakistan_images"
TEMP_CSV_PATH = Path(__file__).parent / "train_filtered_pak_temp.csv"

print("=" * 70)
print("Building image lookup index...")
print(f"Scanning: {IMAGES_BASE_DIR}")

# Build lookup dictionary: {image_id_without_ext: full_path}
# This is a one-time scan, much faster than searching for each ID
image_lookup = {}
image_extensions = {'.jpg'}

# Scan all subfolders (00-97) and build lookup
subfolders_scanned = 0
images_found = 0

for subfolder_num in range(98):  # 00 to 97
    subfolder = IMAGES_BASE_DIR / f"{subfolder_num:02d}"
    if not subfolder.exists():
        continue
    
    subfolders_scanned += 1
    for image_file in subfolder.iterdir():
        if image_file.is_file() and image_file.suffix in image_extensions:
            # Get ID from filename (without extension)
            image_id = image_file.stem
            # Store the full path (use first match if duplicates exist)
            if image_id not in image_lookup:
                image_lookup[image_id] = str(image_file)
                images_found += 1

print(f"Scanned {subfolders_scanned} subfolders")
print(f"Found {images_found:,} images in lookup index")
print("=" * 70)

# Now process CSV
print("Processing CSV...")
rows_processed = 0
rows_matched = 0
rows_not_found = 0

with open(CSV_PATH, 'r', encoding='utf-8') as infile, \
     open(TEMP_CSV_PATH, 'w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ['actual_path']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for row in reader:
        rows_processed += 1
        image_id = row['id']
        
        # Lookup image path
        actual_path = image_lookup.get(image_id, '')
        
        if actual_path:
            rows_matched += 1
        else:
            rows_not_found += 1
        
        # Add actual_path column
        row['actual_path'] = actual_path
        writer.writerow(row)
        
        # Progress update
        if rows_processed % 10000 == 0:
            print(f"Processed {rows_processed:,} rows... (Matched: {rows_matched:,}, Not found: {rows_not_found:,})")

# Replace original with updated file
TEMP_CSV_PATH.replace(CSV_PATH)

print("=" * 70)
print("Processing complete!")
print(f"Total rows processed: {rows_processed:,}")
print(f"Images matched: {rows_matched:,} ({rows_matched/rows_processed*100:.2f}%)")
print(f"Images not found: {rows_not_found:,} ({rows_not_found/rows_processed*100:.2f}%)")
print(f"Updated CSV saved to: {CSV_PATH}")
print("=" * 70)

