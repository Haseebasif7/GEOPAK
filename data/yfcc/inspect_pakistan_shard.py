import msgpack
from pathlib import Path
import json
from PIL import Image
import io

# Path to the Pakistan shard file
shard_path = Path(__file__).parent / "shards" / "shard_pakistan.msg"

if not shard_path.exists():
    print(f"✗ File not found: {shard_path}")
    exit(1)

print("=" * 70)
print("Pakistan Shard Inspector")
print("=" * 70)
print(f"File: {shard_path}")
print(f"File size: {shard_path.stat().st_size / (1024*1024):.2f} MB")
print("=" * 70)

# Read and count records
print("\n📊 Reading shard file...")
total_images = 0
columns = None
first_image = None

with open(shard_path, "rb") as f:
    unpacker = msgpack.Unpacker(f, raw=False)
    
    for i, record in enumerate(unpacker):
        total_images += 1
        
        # Get columns from first record
        if i == 0:
            columns = list(record.keys())
            first_image = record
        
        # Show progress every 10000 records
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1:,} images...")

print(f"\n{'='*70}")
print("Summary")
print(f"{'='*70}")
print(f"Total images: {total_images:,}")

if columns:
    print(f"\nColumns ({len(columns)}):")
    for col in columns:
        print(f"  - {col}")

if first_image:
    print(f"\n{'='*70}")
    print("Sample Image (First Record)")
    print(f"{'='*70}")
    
    # Pretty print the first image
    print(json.dumps(first_image, indent=2, default=str))
    
    # Show specific fields if they exist
    print(f"\n{'='*70}")
    print("Key Fields from Sample Image:")
    print(f"{'='*70}")
    
    if "id" in first_image:
        print(f"  Image ID: {first_image['id']}")
    
    if "latitude" in first_image and "longitude" in first_image:
        print(f"  Coordinates: ({first_image['latitude']}, {first_image['longitude']})")
    
    if "landmark_id" in first_image:
        print(f"  Landmark ID: {first_image['landmark_id']}")
    
    if "landmark_name" in first_image:
        print(f"  Landmark Name: {first_image['landmark_name']}")
    
    if "url" in first_image:
        print(f"  URL: {first_image['url']}")
    
    if "image" in first_image:
        img_data = first_image['image']
        if isinstance(img_data, bytes):
            print(f"  Image data: {len(img_data):,} bytes (binary)")
            
            # Save and display the image
            try:
                # Create image from bytes
                img = Image.open(io.BytesIO(img_data))
                
                # Save to output directory
                output_dir = Path(__file__).parent / "shards"
                output_dir.mkdir(parents=True, exist_ok=True)
                sample_image_path = output_dir / "sample_image.jpg"
                
                # Save the image
                img.save(sample_image_path, "JPEG")
                print(f"\n{'='*70}")
                print("Sample Image Saved")
                print(f"{'='*70}")
                print(f"  Image saved to: {sample_image_path}")
                print(f"  Image size: {img.size[0]}x{img.size[1]} pixels")
                print(f"  Image mode: {img.mode}")
                print(f"\n  To view the image, open: {sample_image_path}")
                
            except Exception as e:
                print(f"  ⚠ Could not process image: {e}")
        else:
            print(f"  Image data: {type(img_data).__name__}")

print(f"\n{'='*70}")
print("✓ Inspection complete!")
print(f"{'='*70}")

