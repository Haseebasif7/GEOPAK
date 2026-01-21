import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

# Paths
CSV_PATH = "province_to_add.csv"
PIPELINE_DIR = Path(__file__).parent
OUTPUT_PATH = "province_to_add_with_province.csv"

print("=" * 70)
print("Adding Province Column using Offline Reverse Geocoding")
print("=" * 70)

# Try to find boundary file (supports .shp, .geojson, .topojson)
boundary_file = None
possible_files = [
    PIPELINE_DIR / "geoBoundaries-PAK-ADM1_simplified.shp",
    PIPELINE_DIR / "geoBoundaries-PAK-ADM1.shp",
    PIPELINE_DIR / "geoBoundaries-PAK-ADM1_simplified.geojson",
    PIPELINE_DIR / "geoBoundaries-PAK-ADM1.geojson",
    PIPELINE_DIR / "gadm41_PAK_1.shp",
]

for file_path in possible_files:
    if file_path.exists():
        # For .shp files, check if .shx exists
        if file_path.suffix == '.shp':
            shx_path = file_path.with_suffix('.shx')
            if not shx_path.exists():
                print(f"⚠ Found {file_path.name} but missing .shx file, skipping...")
                continue
        boundary_file = file_path
        break

if not boundary_file:
    print(f"\n⚠ No boundary file found in: {PIPELINE_DIR}")
    print("Please download Pakistan provinces boundary file:")
    print("1. Visit: https://www.geoboundaries.org/")
    print("2. Search 'Pakistan' and download ADM1")
    print(f"3. Extract to: {PIPELINE_DIR}")
    exit(1)

# Load Pakistan provinces boundary file
print(f"\nLoading Pakistan provinces from: {boundary_file.name}")
pakistan_provinces = gpd.read_file(boundary_file)
print(f"Loaded {len(pakistan_provinces)} provinces")

# Find province name column (different sources use different column names)
province_col = None
possible_cols = ['shapeName', 'NAME_1', 'name', 'NAME', 'province', 'PROVINCE']
for col in possible_cols:
    if col in pakistan_provinces.columns:
        province_col = col
        break

if not province_col:
    print(f"⚠ Warning: Could not find province name column.")
    print(f"Available columns: {list(pakistan_provinces.columns)}")
    print("Using first text column as province name...")
    # Use first non-geometry column
    for col in pakistan_provinces.columns:
        if col != 'geometry':
            province_col = col
            break

print(f"Using column '{province_col}' for province names:")
for idx, row in pakistan_provinces.iterrows():
    print(f"  - {row[province_col]}")

# Load CSV
print(f"\nLoading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df):,} rows")

# Create Point geometries from coordinates
print("\nCreating point geometries...")
geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

# Ensure both GeoDataFrames use same CRS
pakistan_provinces = pakistan_provinces.to_crs('EPSG:4326')

# Spatial join to get province
print("Performing spatial join to match coordinates with provinces...")
print("(This may take a few minutes for 128k+ rows...)")
result = gpd.sjoin(gdf, pakistan_provinces[[province_col, 'geometry']], 
                   how='left', predicate='within')

# Rename province column
result = result.rename(columns={province_col: 'province'})

# Drop geometry and index columns for final CSV
output_df = result[['id', 'latitude', 'longitude', 'path', 'province']].copy()

# Count matches
matched = output_df['province'].notna().sum()
unmatched = len(df) - matched
print(f"\n✓ Matched {matched:,} rows ({matched/len(df)*100:.2f}%)")
print(f"  Unmatched: {unmatched:,} rows ({unmatched/len(df)*100:.2f}%)")

# Show province distribution
if matched > 0:
    print("\nProvince distribution:")
    province_counts = output_df['province'].value_counts()
    for province, count in province_counts.items():
        if pd.notna(province):
            print(f"  {province}: {count:,} ({count/matched*100:.2f}%)")

# Save to CSV
print(f"\nSaving to: {OUTPUT_PATH}")
output_df.to_csv(OUTPUT_PATH, index=False)

print("=" * 70)
print("Complete!")
print(f"Output saved with province column added")
print("=" * 70)

