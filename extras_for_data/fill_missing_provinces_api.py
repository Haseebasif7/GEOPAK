import pandas as pd
import time
import requests
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import json

# Paths
CSV_PATH = Path(__file__).parent.parent / "merged_training_data_with_province.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "merged_training_data_with_province.csv"

# API Configuration
USE_NOMINATIM = True  # Free but slow (1 req/sec)
USE_MAPBOX = False    # Requires API key, faster
MAPBOX_API_KEY = ""   # Set your Mapbox API key if using Mapbox

# Rate limiting
NOMINATIM_DELAY = 1.1  # seconds between requests (slightly more than 1 to be safe)

print("=" * 70)
print("Filling Missing Provinces using Reverse Geocoding API")
print("=" * 70)

# Load CSV
print(f"\nLoading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"Total rows: {len(df):,}")

# Find rows without province
missing_province = df['province'].isna()
missing_count = missing_province.sum()
has_province_count = (~missing_province).sum()

print(f"\nRows WITH province: {has_province_count:,}")
print(f"Rows WITHOUT province: {missing_count:,}")

if missing_count == 0:
    print("\n✓ All rows already have province. Nothing to do!")
    exit(0)

# Initialize geocoder
if USE_NOMINATIM:
    print("\nUsing Nominatim (OpenStreetMap) - Free but slow (~1 req/sec)")
    geolocator = Nominatim(user_agent="PakistanGeoDataset/1.0")
elif USE_MAPBOX:
    if not MAPBOX_API_KEY:
        print("\n✗ Error: MAPBOX_API_KEY not set!")
        exit(1)
    print("\nUsing Mapbox Geocoding API")
    # Mapbox would use requests directly
else:
    print("\n✗ Error: No API selected!")
    exit(1)

def get_province_from_coords(lat, lon, use_nominatim=True):
    """Get province name from coordinates using reverse geocoding"""
    if use_nominatim:
        try:
            location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
            if location:
                address = location.raw.get('address', {})
                # Try different possible province fields (Pakistan-specific)
                province = (
                    address.get('state') or 
                    address.get('province') or 
                    address.get('region') or
                    address.get('state_district') or
                    None
                )
                # Clean up province name (remove common suffixes)
                if province:
                    province = province.replace(' Province', '').replace(' province', '').strip()
                return province
        except (GeocoderTimedOut, GeocoderServiceError, Exception) as e:
            return None
    else:
        # Mapbox implementation
        url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json"
        params = {
            'access_token': MAPBOX_API_KEY,
            'types': 'region,district'
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('features'):
                    # Extract province from context
                    for feature in data['features']:
                        if 'region' in feature.get('place_type', []):
                            return feature.get('text')
            return None
        except Exception as e:
            return None

# Pakistan province name mapping (to standardize different API responses)
PAKISTAN_PROVINCES = {
    'punjab': 'Punjab',
    'sindh': 'Sindh',
    'balochistan': 'Balochistan',
    'khyber pakhtunkhwa': 'Khyber Pakhtunkhwa',
    'kpk': 'Khyber Pakhtunkhwa',
    'nwfp': 'Khyber Pakhtunkhwa',
    'gilgit baltistan': 'Gilgit-Baltistan',
    'azad kashmir': 'Azad Kashmir',
    'ajk': 'Azad Kashmir',
}

def normalize_province_name(province):
    """Normalize province name to standard format"""
    if not province:
        return None
    province_lower = province.lower().strip()
    return PAKISTAN_PROVINCES.get(province_lower, province.title())

# Process rows without province
print(f"\nProcessing {missing_count:,} rows without province...")
if USE_NOMINATIM:
    estimated_minutes = missing_count * NOMINATIM_DELAY / 60
    print(f"This will take approximately {estimated_minutes:.1f} minutes with Nominatim")
print("(Press Ctrl+C to stop and save progress)\n")

filled_count = 0
failed_count = 0
progress_interval = 100

missing_df = df[missing_province].copy()

try:
    for idx, row in missing_df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        
        province = get_province_from_coords(lat, lon, use_nominatim=USE_NOMINATIM)
        
        if province:
            # Normalize province name
            province = normalize_province_name(province)
            df.at[idx, 'province'] = province
            filled_count += 1
        else:
            failed_count += 1
        
        # Progress update
        if (filled_count + failed_count) % progress_interval == 0:
            print(f"Progress: {filled_count + failed_count:,}/{missing_count:,} "
                  f"(Filled: {filled_count:,}, Failed: {failed_count:,})")
        
        # Rate limiting for Nominatim
        if USE_NOMINATIM:
            time.sleep(NOMINATIM_DELAY)
except KeyboardInterrupt:
    print("\n\n⚠ Interrupted by user. Saving progress...")
except Exception as e:
    print(f"\n\n✗ Error: {e}")
    print("Saving progress...")

# Save updated CSV
print(f"\nSaving updated CSV to: {OUTPUT_PATH}")
df.to_csv(OUTPUT_PATH, index=False)

print("=" * 70)
print("Complete!")
print(f"Filled provinces: {filled_count:,}")
print(f"Still missing: {failed_count:,}")
print(f"Total with province now: {has_province_count + filled_count:,} "
      f"({(has_province_count + filled_count)/len(df)*100:.2f}%)")
print("=" * 70)

