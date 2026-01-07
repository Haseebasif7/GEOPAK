"""
Extract Pakistan images from OpenStreetView-5M (OSV5M) training dataset.

✔ Downloads ONE ZIP at a time
✔ Extracts ONLY Pakistan images (unique_country == 'PK')
✔ Deletes ZIP immediately (low disk usage)
✔ Resume-safe (can stop and restart)
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # faster HF downloads

import pandas as pd
import zipfile
import time
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# ============================================================================
# CONFIG
# ============================================================================
REPO_ID = "osv5m/osv5m"
TRAIN_CSV = "datasets/OpenWorld/train.csv"
OUT_DIR = Path("datasets/OpenWorld/pakistan_images")
TMP_DIR = Path("datasets/tmp")
PROGRESS_FILE = TMP_DIR / "processed_zips.txt"

# OSV5M TRAIN IMAGES = 98 ZIP FILES: 00.zip → 97.zip
ZIP_NAMES = [f"{i:02d}.zip" for i in range(98)]

# Download settings
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
DOWNLOAD_TIMEOUT = 3600  # 1 hour timeout per download

# ============================================================================
# STEP 0: Load train.csv and extract Pakistan IDs
# ============================================================================
print("\nSTEP 0: Loading train.csv")

if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(f"{TRAIN_CSV} not found")

df = pd.read_csv(TRAIN_CSV)
pak_df = df[df["unique_country"] == "PK"]
pak_ids = set(pak_df["id"].astype(str))

print(f"Total rows in CSV: {len(df):,}")
print(f"Pakistan rows: {len(pak_df):,}")

if not pak_ids:
    raise RuntimeError("No Pakistan images found in train.csv")

# ============================================================================
# STEP 1: Prepare directories
# ============================================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Remove already extracted images from target set (resume-safe)
existing = {p.stem for p in OUT_DIR.glob("*.jpg")}
pak_ids -= existing

print(f"Already extracted images: {len(existing):,}")
print(f"Remaining images to extract: {len(pak_ids):,}")

# Load processed ZIPs (resume support)
processed_zips = set()
if PROGRESS_FILE.exists():
    with open(PROGRESS_FILE) as f:
        processed_zips = {line.strip() for line in f if line.strip()}
    print(f"Resuming: {len(processed_zips)} ZIPs already processed")

# ============================================================================
# STEP 2: ZIP-by-ZIP download → extract → delete
# ============================================================================

def download_zip_with_retry(zip_name, max_retries=MAX_RETRIES):
    """Download ZIP file with retry logic and timeout handling"""
    zip_path = None
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  📥 Downloading {zip_name} (attempt {attempt}/{max_retries})...")
            
            # Download with explicit timeout handling
            zip_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=zip_name,
                subfolder="images/train",
                local_dir=str(TMP_DIR),
                resume_download=True  # Resume if interrupted
            )
            
            # Verify file exists and is not empty
            if zip_path and os.path.exists(zip_path):
                file_size = os.path.getsize(zip_path)
                if file_size > 0:
                    size_mb = file_size / (1024 * 1024)
                    print(f"  ✓ Downloaded {zip_name} ({size_mb:.1f} MB)")
                    return zip_path
                else:
                    print(f"  ⚠️  Downloaded file is empty, retrying...")
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
            else:
                print(f"  ⚠️  Download completed but file not found, retrying...")
                
        except Exception as e:
            last_error = e
            print(f"  ✗ Download attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print(f"  ⏳ Waiting {RETRY_DELAY} seconds before retry...")
                time.sleep(RETRY_DELAY)
            # Clean up partial download if exists
            if zip_path and os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                except:
                    pass
    
    raise Exception(f"Failed to download {zip_name} after {max_retries} attempts: {last_error}")

found = 0
errors = []

for zip_idx, zip_name in enumerate(tqdm(ZIP_NAMES, desc="Processing ZIPs"), 1):
    if zip_name in processed_zips:
        tqdm.write(f"⏭️  Skipping {zip_name} (already processed)")
        continue

    if not pak_ids:
        print("\n✓ All Pakistan images extracted. Stopping early.")
        break

    print(f"\n{'='*70}")
    print(f"Processing {zip_name} ({zip_idx}/{len(ZIP_NAMES)})")
    print(f"{'='*70}")
    
    zip_path = None
    try:
        # 1️⃣ Download ONE ZIP with retry logic
        zip_path = download_zip_with_retry(zip_name)
        
        if not zip_path or not os.path.exists(zip_path):
            raise Exception(f"Downloaded ZIP file not found: {zip_path}")

        # 2️⃣ Extract ONLY Pakistan images
        print(f"  📦 Extracting Pakistan images from {zip_name}...")
        extracted_this_zip = 0
        
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                # Get total members for progress
                members = z.namelist()
                total_members = len(members)
                
                for member in tqdm(members, desc=f"    Scanning", leave=False, total=total_members):
                    base = os.path.basename(member)
                    if not base:  # Skip directories
                        continue

                    img_id = os.path.splitext(base)[0]
                    if img_id in pak_ids:
                        try:
                            z.extract(member, OUT_DIR)
                            pak_ids.discard(img_id)  # Use discard to avoid KeyError
                            extracted_this_zip += 1
                            found += 1
                        except Exception as e:
                            print(f"    ⚠️  Error extracting {member}: {e}")
                            errors.append(f"{zip_name}: {member} - {e}")
        
        except zipfile.BadZipFile:
            raise Exception(f"{zip_name} is corrupted or not a valid ZIP file")
        except Exception as e:
            raise Exception(f"Error extracting from {zip_name}: {e}")

        print(f"  ✓ Extracted {extracted_this_zip} images from {zip_name}")

    except Exception as e:
        error_msg = f"Error processing {zip_name}: {e}"
        print(f"  ✗ {error_msg}")
        errors.append(error_msg)
        # Continue to next ZIP even if this one failed
    
    finally:
        # 3️⃣ DELETE ZIP immediately (always, even on error)
        if zip_path and os.path.exists(zip_path):
            try:
                size_mb = os.path.getsize(zip_path) / (1024 * 1024)
                os.remove(zip_path)
                print(f"  🗑️  Deleted {zip_name} ({size_mb:.1f} MB freed)")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not delete {zip_name}: {e}")
                errors.append(f"{zip_name}: Could not delete - {e}")
        
        # 4️⃣ Mark ZIP as processed (even if it had errors, to avoid infinite retries)
        try:
            with open(PROGRESS_FILE, "a") as f:
                f.write(zip_name + "\n")
            processed_zips.add(zip_name)
        except Exception as e:
            print(f"  ⚠️  Warning: Could not update progress file: {e}")

    print(f"  📊 Total found so far: {found:,}")
    print(f"  📊 Remaining: {len(pak_ids):,}")
    
    # Small delay to avoid overwhelming the server
    time.sleep(0.5)

# ============================================================================
# STEP 3: Final verification
# ============================================================================
extracted_images = list(OUT_DIR.glob("*.jpg"))

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"ZIP files processed: {len(processed_zips)}/{len(ZIP_NAMES)}")
print(f"Extracted images on disk: {len(extracted_images):,}")
print(f"Expected from CSV: {len(pak_df):,}")
print(f"Missing (normal): {max(0, len(pak_df) - len(extracted_images)):,}")
print(f"Errors encountered: {len(errors)}")
print(f"Output directory: {OUT_DIR}")
print("="*70)

if errors:
    error_log_path = TMP_DIR / "extraction_errors.txt"
    with open(error_log_path, "w") as f:
        for error in errors:
            f.write(f"{error}\n")
    print(f"\n⚠️  {len(errors)} errors encountered. See: {error_log_path}")
    if len(errors) <= 10:
        for error in errors:
            print(f"  - {error}")

print("\n✓ DONE")
