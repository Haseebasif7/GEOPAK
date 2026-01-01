import msgpack
from pathlib import Path
import os
import sys
import zipfile
import shutil

FILE_PREFIX = "shards"


# Try to import kaggle API (for individual file downloads)
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    import kaggle
    HAS_KAGGLE_API = True
except ImportError:
    print("Error: kaggle package is not installed.")
    print("Install it with: pip install kaggle")
    print("Also ensure you have kaggle.json credentials set up.")
    sys.exit(1)

# Initialize Kaggle API
api = KaggleApi()
try:
    api.authenticate()
except Exception as e:
    print(f"Error authenticating with Kaggle API: {e}")
    print("Make sure you have kaggle.json in ~/.kaggle/")
    sys.exit(1)

# Pakistan bounding box coordinates
PAKISTAN_LAT_MIN = 23.6345
PAKISTAN_LAT_MAX = 37.0841
PAKISTAN_LON_MIN = 60.8726
PAKISTAN_LON_MAX = 77.8375

def is_in_pakistan(latitude, longitude):
    """Check if coordinates are within Pakistan's bounding box"""
    if latitude is None or longitude is None:
        return False
    return (PAKISTAN_LAT_MIN <= latitude <= PAKISTAN_LAT_MAX and 
            PAKISTAN_LON_MIN <= longitude <= PAKISTAN_LON_MAX)

# Configuration
DATASET_NAME = "habedi/large-dataset-of-geotagged-images"
SHARDS_DIR = Path(__file__).parent / "shards"
OUTPUT_PATH = SHARDS_DIR / "shard_pakistan.msg"
PROGRESS_FILE = SHARDS_DIR / ".processed_shards.txt"
TOTAL_SHARDS = 142  # shard_0.msg to shard_141.msg

# Create directories
SHARDS_DIR.mkdir(parents=True, exist_ok=True)

def load_processed_shards():
    """Load list of already processed shard numbers"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return {int(line.strip()) for line in f if line.strip().isdigit()}
    return set()

def mark_shard_processed(shard_num):
    """Mark a shard as processed"""
    with open(PROGRESS_FILE, 'a') as f:
        f.write(f"{shard_num}\n")

print("=" * 70)
print("Kaggle Shard Downloader and Processor")
print("=" * 70)
print(f"Dataset: {DATASET_NAME}")
print(f"Total shards: {TOTAL_SHARDS} (shard_0.msg to shard_141.msg)")
print(f"Pakistan bounds: Lat [{PAKISTAN_LAT_MIN}, {PAKISTAN_LAT_MAX}], Lon [{PAKISTAN_LON_MIN}, {PAKISTAN_LON_MAX}]")
print(f"Output file: {OUTPUT_PATH}")
print("=" * 70)

# Global statistics
total_records_all = 0
in_pakistan_all = 0
outside_pakistan_all = 0
missing_coords_all = 0
total_pakistan_count = 0

# Open output file for writing Pakistan records
packer = msgpack.Packer()
output_file = None
columns_shown = False

import zipfile
import shutil

import zipfile

def download_shard(shard_num, shard_path):
    filename = f"shard_{shard_num}.msg"
    kaggle_file = f"{FILE_PREFIX}/{filename}"
    zip_path = SHARDS_DIR / f"{filename}.zip"

    # Resume support
    if shard_path.exists():
        print("  ℹ File already exists, skipping download")
        return True

    try:
        api.dataset_download_file(
            DATASET_NAME,
            kaggle_file,
            path=str(SHARDS_DIR),
            quiet=False
        )

        if not zip_path.exists():
            print(f"  ✗ Zip not found: {zip_path}")
            return False

        # Extract .msg
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extract(filename, path=SHARDS_DIR)

        # Delete zip immediately (save space)
        zip_path.unlink()

        return shard_path.exists()

    except Exception as e:
        print(f"  ✗ Download error: {e}")
        return False

def process_shard(shard_path, shard_num):
    """Process a single shard and extract Pakistan data"""
    global total_records_all, in_pakistan_all, outside_pakistan_all
    global missing_coords_all, total_pakistan_count, output_file, columns_shown
    
    shard_total = 0
    shard_in_pakistan = 0
    shard_outside = 0
    shard_missing = 0
    
    # Open output file if not already open
    if output_file is None:
        output_file = open(OUTPUT_PATH, "wb")
    
    try:
        with open(shard_path, "rb") as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            
            for i, record in enumerate(unpacker):
                shard_total += 1
                total_records_all += 1
                
                # Show columns from first record
                if not columns_shown and i == 0:
                    print(f"  Columns: {list(record.keys())}")
                    columns_shown = True
                
                # Get coordinates
                latitude = record.get("latitude")
                longitude = record.get("longitude")
                
                # Check if coordinates exist
                if latitude is None or longitude is None:
                    shard_missing += 1
                    missing_coords_all += 1
                elif is_in_pakistan(latitude, longitude):
                    shard_in_pakistan += 1
                    in_pakistan_all += 1
                    # Write directly to output file
                    output_file.write(packer.pack(record))
                    total_pakistan_count += 1
                else:
                    shard_outside += 1
                    outside_pakistan_all += 1
                
                # Show progress every 5000 records
                if (i + 1) % 5000 == 0:
                    print(f"    Processed {i + 1:,} records... (PK: {shard_in_pakistan:,}, Outside: {shard_outside:,})")
        
        return {
            'total': shard_total,
            'in_pakistan': shard_in_pakistan,
            'outside': shard_outside,
            'missing': shard_missing
        }
    except Exception as e:
        print(f"  ✗ Processing error: {e}")
        return None

# Check for resume capability
processed_shards = load_processed_shards()
if processed_shards:
    print(f"\n⚠ Found {len(processed_shards)} already processed shards")
    print(f"  Processed shards: {sorted(processed_shards)[:10]}{'...' if len(processed_shards) > 10 else ''}")
    response = input("  Continue from where we left off? (y/n): ").strip().lower()
    if response != 'y':
        # Clear progress and start fresh
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        if OUTPUT_PATH.exists():
            response2 = input("  Delete existing output file? (y/n): ").strip().lower()
            if response2 == 'y':
                OUTPUT_PATH.unlink()
        processed_shards = set()
        print("  ✓ Starting fresh")
    else:
        # Open in append mode if resuming
        if OUTPUT_PATH.exists():
            output_file = open(OUTPUT_PATH, "ab")
            print(f"  ✓ Will append to existing file")

# Process each shard
for shard_num in range(TOTAL_SHARDS):
    # Skip if already processed
    if shard_num in processed_shards:
        print(f"\n{'='*70}")
        print(f"Shard {shard_num + 1}/{TOTAL_SHARDS}: shard_{shard_num}.msg (already processed, skipping)")
        continue
    print(f"\n{'='*70}")
    print(f"Shard {shard_num + 1}/{TOTAL_SHARDS}: shard_{shard_num}.msg")
    print(f"{'='*70}")
    
    shard_path = SHARDS_DIR / f"shard_{shard_num}.msg"
    
    # Step 1: Download shard
    print(f"  📥 Downloading shard_{shard_num}.msg...")
    if not download_shard(shard_num, shard_path):
        print(f"  ⚠ Skipping shard_{shard_num}.msg (download failed)")
        continue
    
    file_size_mb = shard_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Downloaded ({file_size_mb:.2f} MB)")
    
    # Step 2: Process shard
    print(f"  🔄 Processing shard_{shard_num}.msg...")
    stats = process_shard(shard_path, shard_num)
    
    if stats:
        print(f"  ✓ Processed:")
        print(f"    Total: {stats['total']:,}")
        print(f"    In Pakistan: {stats['in_pakistan']:,} ({stats['in_pakistan']/stats['total']*100:.2f}%)")
        print(f"    Outside: {stats['outside']:,}")
        print(f"    Missing coords: {stats['missing']:,}")
    
    # Step 3: Delete shard to save space
    print(f"  🗑️  Deleting shard_{shard_num}.msg to save space...")
    try:
        freed_mb = shard_path.stat().st_size / (1024 * 1024)
        shard_path.unlink()
        print(f"  ✓ Deleted (freed {freed_mb:.2f} MB)")
        
        # Mark as processed
        mark_shard_processed(shard_num)
    except Exception as e:
        print(f"  ✗ Delete error: {e}")

# Close output file
if output_file:
    output_file.close()

# Print final summary
print("\n" + "=" * 70)
print("FINAL SUMMARY - ALL SHARDS PROCESSED")
print("=" * 70)
print(f"Total shards processed: {TOTAL_SHARDS}")
print(f"Total images/records: {total_records_all:,}")
if total_records_all > 0:
    print(f"Images within Pakistan bounding box: {in_pakistan_all:,} ({in_pakistan_all/total_records_all*100:.2f}%)")
    print(f"Images outside Pakistan bounding box: {outside_pakistan_all:,} ({outside_pakistan_all/total_records_all*100:.2f}%)")
    print(f"Images with missing coordinates: {missing_coords_all:,} ({missing_coords_all/total_records_all*100:.2f}%)")
else:
    print(f"Images within Pakistan bounding box: {in_pakistan_all:,} (N/A - no records processed)")
    print(f"Images outside Pakistan bounding box: {outside_pakistan_all:,} (N/A - no records processed)")
    print(f"Images with missing coordinates: {missing_coords_all:,} (N/A - no records processed)")
print("=" * 70)

if total_pakistan_count > 0:
    print(f"\n✓ Successfully created Pakistan shard: {OUTPUT_PATH}")
    print(f"  Total Pakistan records: {total_pakistan_count:,}")
    if OUTPUT_PATH.exists():
        print(f"  Output file size: {OUTPUT_PATH.stat().st_size / (1024*1024):.2f} MB")
else:
    print("\n⚠ No Pakistan records found across all shards!")

print("\n✓ All done! Original shards deleted to save space.")

