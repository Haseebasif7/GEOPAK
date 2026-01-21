import pandas as pd
from pathlib import Path

# Paths
CSV_PATH = "province_to_add_with_province.csv"

print("=" * 70)
print("Checking Missing Provinces")
print("=" * 70)

# Load CSV
print(f"\nLoading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"Total rows: {len(df):,}")

# Check for missing provinces
missing_province = df['province'].isna()
missing_count = missing_province.sum()
has_province_count = (~missing_province).sum()

print("\n" + "=" * 70)
print("Results:")
print("=" * 70)
print(f"Rows WITH province: {has_province_count:,} ({has_province_count/len(df)*100:.2f}%)")
print(f"Rows WITHOUT province: {missing_count:,} ({missing_count/len(df)*100:.2f}%)")

if missing_count > 0:
    print("\n" + "-" * 70)
    print("Sample rows without province:")
    print("-" * 70)
    missing_rows = df[missing_province][['id', 'latitude', 'longitude', 'path']].head(10)
    print(missing_rows.to_string(index=False))
    
    if missing_count > 10:
        print(f"\n... and {missing_count - 10:,} more rows without province")
    
    # Check if coordinates are outside Pakistan bounds
    print("\n" + "-" * 70)
    print("Coordinate analysis for missing provinces:")
    print("-" * 70)
    missing_df = df[missing_province]
    if len(missing_df) > 0:
        print(f"Latitude range: {missing_df['latitude'].min():.6f} to {missing_df['latitude'].max():.6f}")
        print(f"Longitude range: {missing_df['longitude'].min():.6f} to {missing_df['longitude'].max():.6f}")
        print("\nPakistan bounds (typical):")
        print("  Latitude: 23.6345 to 37.0841")
        print("  Longitude: 60.8726 to 77.8375")
        
        # Count how many are outside bounds
        pak_lat_min, pak_lat_max = 23.6345, 37.0841
        pak_lon_min, pak_lon_max = 60.8726, 77.8375
        
        outside_bounds = (
            (missing_df['latitude'] < pak_lat_min) | 
            (missing_df['latitude'] > pak_lat_max) |
            (missing_df['longitude'] < pak_lon_min) | 
            (missing_df['longitude'] > pak_lon_max)
        ).sum()
        
        print(f"\nRows outside Pakistan bounds: {outside_bounds:,}")

# Show province distribution for rows that have province
if has_province_count > 0:
    print("\n" + "=" * 70)
    print("Province distribution (for rows WITH province):")
    print("=" * 70)
    province_counts = df[~missing_province]['province'].value_counts()
    for province, count in province_counts.items():
        print(f"  {province}: {count:,} ({count/has_province_count*100:.2f}%)")

print("\n" + "=" * 70)

