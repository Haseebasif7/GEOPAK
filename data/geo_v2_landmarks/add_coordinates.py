import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from pathlib import Path
from tqdm import tqdm
import time

# Paths
csv_path = Path(__file__).parent / "train_filtered.csv"
output_path = Path(__file__).parent / "train_filtered.csv"  # Overwrite original

# Initialize geocoder with rate limiter (1 request per second to respect API limits)
geolocator = Nominatim(user_agent="geopak_landmark_geocoder")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

def geocode_landmark(landmark_name):
    """
    Geocode a landmark name, adding 'Pakistan' for better accuracy
    """
    # Replace underscores with spaces and add Pakistan for better results
    query = landmark_name.replace('_', ' ') + ', Pakistan'
    
    try:
        location = geocode(query, timeout=10)
        if location:
            return {
                'latitude': location.latitude,
                'longitude': location.longitude,
                'found': True
            }
        else:
            return {
                'latitude': None,
                'longitude': None,
                'found': False
            }
    except Exception as e:
        print(f"Error geocoding '{landmark_name}': {e}")
        return {
            'latitude': None,
            'longitude': None,
            'found': False
        }

# Load CSV
print("Loading CSV file...")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows")

# Get unique landmark names
unique_landmarks = df['landmark_name'].unique()
print(f"Found {len(unique_landmarks)} unique landmarks to geocode")

# Geocode each unique landmark (with caching)
print("Geocoding landmarks (this may take a while due to rate limiting)...")
landmark_coords = {}

for landmark_name in tqdm(unique_landmarks, desc="Geocoding"):
    if pd.isna(landmark_name):
        landmark_coords[landmark_name] = {'latitude': None, 'longitude': None, 'found': False}
    else:
        result = geocode_landmark(landmark_name)
        landmark_coords[landmark_name] = result
        # Small additional delay to be safe
        time.sleep(0.1)

# Map coordinates back to dataframe
print("Mapping coordinates to dataframe...")
df['latitude'] = df['landmark_name'].map(lambda x: landmark_coords.get(x, {}).get('latitude'))
df['longitude'] = df['landmark_name'].map(lambda x: landmark_coords.get(x, {}).get('longitude'))

# Check success rate
found_count = sum(1 for coords in landmark_coords.values() if coords.get('found', False))
not_found = [name for name, coords in landmark_coords.items() 
              if not coords.get('found', False) and pd.notna(name)]

print(f"\nGeocoding Summary:")
print(f"  Successfully geocoded: {found_count}/{len(unique_landmarks)} landmarks")
if not_found:
    print(f"  Failed to geocode {len(not_found)} landmarks:")
    for name in not_found[:10]:  # Show first 10
        print(f"    - {name}")
    if len(not_found) > 10:
        print(f"    ... and {len(not_found) - 10} more")

# Save updated CSV
print(f"\nSaving updated CSV to {output_path}...")
df.to_csv(output_path, index=False)
print(f"Done! CSV now has columns: {list(df.columns)}")
print(f"Rows with coordinates: {df['latitude'].notna().sum()}/{len(df)}")

