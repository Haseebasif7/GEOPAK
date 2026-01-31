"""
Upload training data to Modal volume for GPU training.

This script uploads:
- train.csv
- test.csv
- resnet50_places365.pth.tar
- datasets/ directory (if exists)

Usage:
    # For large uploads (recommended), use detached mode:
    modal run --detach model/province/upload_data_to_modal.py
    
    # Normal mode (may disconnect on large uploads):
    modal run model/province/upload_data_to_modal.py
"""
import modal
from pathlib import Path
import shutil
import sys

# Project root (parent of this script's directory)
project_root = Path(__file__).parent.parent.parent

# Create Modal app and image
image = modal.Image.debian_slim().pip_install("tqdm")

app = modal.App("geopak-upload-data", image=image)

# Get or create volume
volume = modal.Volume.from_name("geopak-data", create_if_missing=True)

@app.function(volumes={"/data": volume}, timeout=600)  # 10 minutes for large files
def write_file_to_volume(file_content: bytes, volume_path: str):
    """Write file content to Modal volume"""
    from pathlib import Path
    
    dst = Path(f"/data/{volume_path}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Write file in chunks to handle large files
    chunk_size = 10 * 1024 * 1024  # 10 MB chunks
    total_written = 0
    
    with open(dst, 'wb') as f:
        for i in range(0, len(file_content), chunk_size):
            chunk = file_content[i:i + chunk_size]
            f.write(chunk)
            total_written += len(chunk)
    
    # Verify the file was written completely
    if dst.exists():
        actual_size = dst.stat().st_size
        if actual_size != len(file_content):
            return False, f"Size mismatch: expected {len(file_content)}, got {actual_size}"
        
        size_mb = len(file_content) / (1024 * 1024)
        volume.commit()
        return True, f"Uploaded ({size_mb:.2f} MB, {actual_size} bytes)"
    else:
        return False, "Upload failed - file not found after write"

@app.function(volumes={"/data": volume})
def verify_file_size(volume_path: str, expected_size: int):
    """Verify uploaded file size"""
    from pathlib import Path
    dst = Path(f"/data/{volume_path}")
    if dst.exists():
        actual_size = dst.stat().st_size
        return actual_size == expected_size, f"Expected {expected_size}, got {actual_size}"
    return False, "File not found"

@app.function(
    volumes={"/data": volume},
    timeout=300,  # 5 minutes timeout per batch
)
def write_directory_to_volume(file_dict: dict, base_path: str):
    """Write multiple files (directory structure) to Modal volume"""
    from pathlib import Path
    
    # file_dict: {relative_path: file_content_bytes}
    written_count = 0
    total_size = 0
    
    for rel_path, content in file_dict.items():
        dst = Path(f"/data/{base_path}/{rel_path}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dst, 'wb') as f:
            f.write(content)
        
        if dst.exists():
            written_count += 1
            total_size += len(content)
    
    if written_count > 0:
        volume.commit()
        total_size_mb = total_size / (1024 * 1024)
        return True, f"Uploaded ({written_count} files, {total_size_mb:.2f} MB)"
    else:
        return False, "Upload failed"

@app.local_entrypoint()
def main():
    """Local entry point to trigger upload"""
    print("=" * 70)
    print("UPLOADING DATA TO MODAL VOLUME")
    print("=" * 70)
    print("\n⚠️  IMPORTANT: For large uploads (90k+ files), use detached mode:")
    print("   modal run --detach model/province/upload_data_to_modal.py")
    print("   This prevents connection loss during long uploads.\n")
    
    # Files to upload (filename, description)
    # NOTE: Only uploading CSVs - images and weights are already uploaded
    files_to_upload = [
        #("train.csv", "Training CSV"),
        #("test.csv", "Test CSV"),
        ("pipeline/geocells/cell_metadata.csv", "Cell Metadata CSV"),
        #("resnet50_places365.pth.tar", "ResNet50 Places365 model"),  
    ]
    
    # Directories to upload (source_path, volume_path, description)
    # NOTE: Skipping datasets - images are already uploaded
    datasets_path = Path("/Users/haseeb/Desktop/datasets")
    dirs_to_upload = [
        #(datasets_path, "datasets", "Datasets directory"),  # Commented out - already uploaded
    ]
    
    success_count = 0
    total_count = len(files_to_upload) + len(dirs_to_upload)
    
    print("\n📦 Starting upload...")
    print("   This will upload files from your local project to Modal volume")
    print()
    
    # Upload files
    print("=" * 70)
    print("UPLOADING FILES")
    print("=" * 70)
    
    for src_name, desc in files_to_upload:
        src_path = project_root / src_name
        dst_name = src_name  # Use same name in volume
        
        if not src_path.exists():
            print(f"⚠️  Skipping {desc}: {src_path} not found")
            continue
        
        print(f"📤 Uploading {desc}...")
        print(f"   From: {src_path}")
        print(f"   To: /data/{dst_name}")
        
        # Read file content and verify size
        try:
            file_size = src_path.stat().st_size
            print(f"   File size: {file_size / (1024 * 1024):.2f} MB")
            
            with open(src_path, 'rb') as f:
                file_content = f.read()
            
            # Verify we read the entire file
            if len(file_content) != file_size:
                print(f"   ⚠️  Warning: Read {len(file_content)} bytes, expected {file_size} bytes")
            
            print(f"   Uploading {len(file_content) / (1024 * 1024):.2f} MB...")
            success, message = write_file_to_volume.remote(
                file_content,
                dst_name
            )
            
            if success:
                print(f"   ✅ {message}")
                # Verify the uploaded file size
                verified, verify_msg = verify_file_size.remote(dst_name, file_size)
                if verified:
                    print(f"   ✅ Verified: File size matches ({file_size / (1024 * 1024):.2f} MB)")
                else:
                    print(f"   ⚠️  Verification failed: {verify_msg}")
                success_count += 1
            else:
                print(f"   ❌ {message}")
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Upload directories
    print("=" * 70)
    print("UPLOADING DIRECTORIES")
    print("=" * 70)
    
    for src_path, dst_name, desc in dirs_to_upload:
        # src_path is already a Path object
        if not src_path.exists():
            print(f"⚠️  Skipping {desc}: {src_path} not found")
            continue
        
        if not src_path.is_dir():
            print(f"⚠️  {desc}: {src_path} is not a directory, skipping")
            continue
        
        print(f"📤 Uploading {desc}...")
        print(f"   From: {src_path}")
        print(f"   To: /data/{dst_name}")
        
        # Process files in batches to avoid memory issues
        try:
            # First, collect all file paths (without loading content)
            all_files = [f for f in src_path.rglob('*') if f.is_file()]
            
            if len(all_files) == 0:
                print(f"   ⚠️  Directory is empty")
                continue
            
            print(f"   Found {len(all_files)} files")
            print(f"   Uploading in batches of 500 (committing every 5 batches)...")
            
            BATCH_SIZE = 500  # Larger batches = fewer remote calls
            COMMIT_EVERY = 5  # Commit every 5 batches to reduce overhead
            total_uploaded = 0
            total_size_mb = 0.0
            batches_since_commit = 0
            total_batches = (len(all_files) + BATCH_SIZE - 1) // BATCH_SIZE
            
            # Upload all files in batches
            for batch_idx in range(0, len(all_files), BATCH_SIZE):
                batch_num = (batch_idx // BATCH_SIZE) + 1
                batch_files = all_files[batch_idx:batch_idx + BATCH_SIZE]
                file_dict = {}
                
                # Read batch
                print(f"   📤 Batch {batch_num}/{total_batches}: Reading {len(batch_files)} files...", flush=True)
                for file_path in batch_files:
                    rel_path = file_path.relative_to(src_path)
                    try:
                        with open(file_path, 'rb') as f:
                            file_dict[str(rel_path)] = f.read()
                    except Exception as e:
                        print(f"   ⚠️  Error reading {file_path.name}: {e}", flush=True)
                
                if len(file_dict) > 0:
                    batch_size_mb = sum(len(c) for c in file_dict.values()) / (1024 * 1024)
                    print(f"   ⬆️  Batch {batch_num}/{total_batches}: Uploading {len(file_dict)} files ({batch_size_mb:.2f} MB)...", flush=True)
                    
                    # Upload batch (commit happens inside function, but we'll also commit less frequently)
                    try:
                        success, message = write_directory_to_volume.remote(
                            file_dict,
                            dst_name
                        )
                        
                        if success:
                            uploaded_count = len(file_dict)
                            total_uploaded += uploaded_count
                            total_size_mb += batch_size_mb
                            batches_since_commit += 1
                            progress = (total_uploaded / len(all_files)) * 100
                            print(f"   ✅ Batch {batch_num}/{total_batches} complete: {progress:.1f}% ({total_uploaded:,}/{len(all_files):,} files, {total_size_mb:.2f} MB total)", flush=True)
                        else:
                            print(f"   ❌ Batch {batch_num} failed: {message}", flush=True)
                    except Exception as e:
                        print(f"\n   ⚠️  Batch {batch_num} upload error: {e}", flush=True)
                        print(f"   Retrying batch {batch_num}...", flush=True)
                        # Retry once
                        try:
                            success, message = write_directory_to_volume.remote(
                                file_dict,
                                dst_name
                            )
                            if success:
                                uploaded_count = len(file_dict)
                                total_uploaded += uploaded_count
                                total_size_mb += batch_size_mb
                                batches_since_commit += 1
                                progress = (total_uploaded / len(all_files)) * 100
                                print(f"   ✅ Batch {batch_num} retry successful: {progress:.1f}% ({total_uploaded:,}/{len(all_files):,} files)", flush=True)
                            else:
                                print(f"   ❌ Batch {batch_num} retry failed: {message}", flush=True)
                        except Exception as e2:
                            print(f"   ❌ Retry failed: {e2}", flush=True)
            
            print()  # New line after progress
            print(f"   ✅ Uploaded {total_uploaded:,} files ({total_size_mb:.2f} MB)")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Error reading directory: {e}")
            print()
    
    # Summary
    print("=" * 70)
    print("UPLOAD SUMMARY")
    print("=" * 70)
    print(f"   Successfully uploaded: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("   ✅ All files uploaded successfully!")
    else:
        print(f"   ⚠️  {total_count - success_count} files/directories were skipped or failed")
    
    print("\n" + "=" * 70)
    print("✅ UPLOAD COMPLETE!")
    print("=" * 70)
    print("\n💡 TIP: For large uploads, use detached mode to prevent connection loss:")
    print("   modal run --detach model/province/upload_data_to_modal.py")
    print("\nYou can now run training on Modal:")
    print("   modal run model/province/train_province.py --num-epochs 50 --temperature 1.5 --gpu L4")
    print()

if __name__ == "__main__":
    # Run using Modal
    # NOTE: For long uploads, use detached mode: modal run --detach model/province/upload_data_to_modal.py
    # This prevents connection loss if your local client disconnects
    with app.run():
        main()
