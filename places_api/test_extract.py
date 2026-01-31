"""
Extract places data for Gilgit-Baltistan, Pakistan - Dataset Augmentation
Focus: OUTDOOR geolocation-relevant places that help model distinguish Gilgit-Baltistan visually
- Landscapes: valleys with rivers, partial glacier views, tree-lined villages
- Human presence: small settlements, stone houses, suspension bridges
- Seasonal: snowy roads, autumn colors, spring meltwater
Excludes: Indoor/retail/brand places (noisy)
Target: ~1000 places with 5 images each = ~5000 images for dataset augmentation
"""

import requests
import json
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import time

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load province mapping
PROVINCE_MAPPING_PATH = Path(__file__).parent.parent / 'model' / 'province_mapping.json'
with open(PROVINCE_MAPPING_PATH, 'r') as f:
    PROVINCE_MAPPING = json.load(f)

# Configuration - Gilgit-Baltistan, Pakistan (Dataset Augmentation - focus on valleys, villages, seasonal variation)
PROVINCES = {
    "Gilgit-Baltistan": 4,  # Gilgit-Baltistan, Pakistan
}

# Search configuration
MAX_RESULTS_PER_QUERY = 20  # Places API max per query
TARGET_PLACES_PER_PROVINCE = 1000  # Target places (5 images each = 5000 images)

# Pakistan bounding box for location restriction (to avoid India results)
# Approximate bounds for Pakistan: lat 23.5-37.0, lng 60.0-77.8
PAKISTAN_BOUNDS = {
    "south": 23.5,
    "west": 60.0,
    "north": 37.0,
    "east": 77.8
}

# Gilgit-Baltistan bounding box for validation (approx)
# Note: Coarse filter to keep results within GB region and avoid India/China.
GILGIT_BALTISTAN_BOUNDS = {
    "south": 34.5,
    "west": 73.0,
    "north": 37.5,
    "east": 77.5,
}

# Specific places to search for Gilgit-Baltistan (FOCUS: Valleys, villages, bridges, seasonal roads)
# Based on plan.txt: valleys with rivers, partial glaciers, tree-lined villages, small settlements, stone houses, suspension bridges
PROVINCE_SPECIFIC_PLACES = {
    "Gilgit-Baltistan": [
        # MAJOR VALLEYS & TOWNS (landscapes + small settlements)
        "Hunza Valley", "Hunza Valley Pakistan", "Hunza Gilgit-Baltistan",
        "Skardu Valley", "Skardu Gilgit-Baltistan", "Skardu Pakistan",
        "Gilgit city outskirts", "Gilgit outskirts", "Gilgit River valley",
        "Nagar Valley Gilgit-Baltistan", "Ghizer Valley Gilgit-Baltistan",
        "Astore Valley", "Gupis Valley", "Yasin Valley Gilgit-Baltistan",
        # RIVER & GLACIER LANDSCAPES
        "Hunza River valley", "Indus River Skardu", "Gilgit River",
        "Glacier views Hunza", "Glacier views Skardu", "Glacier valley Gilgit-Baltistan",
        # VILLAGES & STONE HOUSES
        "Tree-lined villages Gilgit-Baltistan", "Mountain village Gilgit-Baltistan",
        "Stone houses Gilgit-Baltistan", "Stone village houses Hunza",
        # BRIDGES
        "Suspension bridge Hunza", "Suspension bridge Gilgit-Baltistan",
        "Hussaini suspension bridge", "Hussaini bridge Hunza",
    ],
}

# Generic search queries for Gilgit-Baltistan (Based on plan.txt: valleys, villages, seasonal)
# Focus: Visual features that help distinguish GB but not too restrictive (still exclude retail/indoor)
GENERIC_SEARCH_QUERIES = [
    # LANDSCAPES (plan.txt)
    "{province} valleys with rivers Pakistan",
    "{province} river valley Pakistan",
    "{province} mountain valley river Pakistan",
    "{province} glacier valley Pakistan",
    "{province} partial glacier views Pakistan",
    # VILLAGES & HOUSES (plan.txt)
    "{province} tree-lined villages Pakistan",
    "{province} mountain villages Pakistan",
    "{province} stone houses Pakistan",
    "{province} small settlements Pakistan",
    "{province} hillside villages Pakistan",
    # BRIDGES (plan.txt)
    "{province} suspension bridges Pakistan",
    "{province} suspension bridge valley Pakistan",
    "{province} river suspension bridge Pakistan",
    # SEASONAL (plan.txt)
    "{province} snowy roads Pakistan",
    "{province} autumn colors Pakistan",
    "{province} autumn valley Pakistan",
    "{province} spring meltwater Pakistan",
    # GENERAL LANDSCAPES (looser, but still non-retail)
    "{province} scenic valley Pakistan",
    "{province} mountain road Pakistan",
    "{province} river road Pakistan",
]

# Rate limiting (to avoid hitting API limits and control costs)
DELAY_BETWEEN_QUERIES = 1.0  # Seconds between API queries
DELAY_BETWEEN_DOWNLOADS = 0.3  # Seconds between image downloads

# Billing notes:
# - Places API (New): $17 per 1000 requests for searchText
# - Places API (New): $7 per 1000 requests for photo media
# - Estimated cost per province: ~$0.50-1.00 (depending on results)
# - Total estimated cost for all provinces: ~$3-6

# Normalize province names to match mapping
PROVINCE_NORMALIZE = {
    "Punjab": "Punjab",
    "Khyber Pakhtunkhwa": "Khyber Pakhtunkhwa",
    "ICT": "ICT",
    "Islamabad Capital Territory": "ICT",
    "Gilgit-Baltistan": "Gilgit-Baltistan",
    "Balochistan": "Balochistan",
    "Azad Kashmir": "Azad Kashmir",
    "Azad Jammu and Kashmir": "Azad Kashmir"
}

OUTPUT_DIR = Path("/Users/haseeb/Desktop/geopak/places_api")
OUTPUT_DIR.mkdir(exist_ok=True)
IMAGES_DIR = Path("/Users/haseeb/Desktop/datasets/pakistan_images/places_api_images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "gilgit_baltistan.csv"

# Places API endpoint
PLACES_URL = "https://places.googleapis.com/v1/places:searchText"

def is_in_pakistan(location):
    """Check if location is within Pakistan bounds"""
    if not location:
        return False
    lat = location.get('latitude', 0)
    lng = location.get('longitude', 0)
    return (PAKISTAN_BOUNDS['south'] <= lat <= PAKISTAN_BOUNDS['north'] and
            PAKISTAN_BOUNDS['west'] <= lng <= PAKISTAN_BOUNDS['east'])

def is_in_gilgit_baltistan(location):
    """Check if location is within Gilgit-Baltistan bounds"""
    if not location:
        return False
    lat = location.get('latitude', 0)
    lng = location.get('longitude', 0)
    return (GILGIT_BALTISTAN_BOUNDS['south'] <= lat <= GILGIT_BALTISTAN_BOUNDS['north'] and
            GILGIT_BALTISTAN_BOUNDS['west'] <= lng <= GILGIT_BALTISTAN_BOUNDS['east'])

def is_geolocation_relevant(place):
    """
    Filter places to keep only those useful for geolocation (outdoor, landscape, infrastructure)
    Includes: Valleys with rivers, partial glaciers, villages, stone houses, suspension bridges (GB-specific)
    Excludes: Indoor/retail/brands and other noisy categories
    """
    if not place:
        return False
    
    display_name = place.get('displayName', {}).get('text', '').lower()
    formatted_address = place.get('formattedAddress', '').lower()
    full_text = f"{display_name} {formatted_address}".lower()
    types = [t.lower() for t in (place.get("types") or []) if isinstance(t, str)]
    
    # Keywords that indicate indoor/people-focused / brand / retail places (EXCLUDE)
    exclude_keywords = [
        'restaurant', 'cafe', 'café', 'hotel', 'motel', 'resort', 'lodge',
        'hospital', 'clinic', 'medical', 'pharmacy', 'drugstore',
        'indoor', 'interior', 'inside',
        'dining', 'food', 'eatery', 'fast food',
        'gym', 'fitness', 'spa', 'salon', 'barber',
        'cinema', 'theater', 'movie', 'entertainment center',
        'library', 'museum', 'gallery', 'exhibition',
        # retail / brands / stores (usually indoor + noisy for province)
        'mall', 'shopping', 'store', 'shop', 'outlet', 'boutique', 'brand',
        'plaza', 'center', 'centre', 'supermarket', 'hypermarket', 'mart',
        'cash and carry', 'cash & carry', 'electronics', 'mobile', 'phones',
        'clothing', 'fashion', 'jewelry', 'jewellery', 'shoes',
        'bakery', 'sweets', 'sweet', 'restaurant', 'cafe',
    ]
    
    # Check if place name contains any exclude keywords
    for keyword in exclude_keywords:
        if keyword in full_text:
            return False
    
    # Hard-exclude by Places types (more reliable than name matching)
    # NOTE: These are Places API "types" strings when available.
    exclude_types = {
        "shopping_mall",
        "department_store",
        "clothing_store",
        "shoe_store",
        "jewelry_store",
        "electronics_store",
        "grocery_store",
        "supermarket",
        "store",
        "convenience_store",
        "restaurant",
        "cafe",
        "bar",
        "night_club",
        "lodging",
        "hospital",
        "pharmacy",
        "drugstore",
        "gym",
        "beauty_salon",
        "spa",
        "movie_theater",
        "museum",
        "art_gallery",
        "library",
    }
    if any(t in exclude_types for t in types):
        return False

    # Keywords that indicate outdoor/geolocation-relevant places (INCLUDE) - Gilgit-Baltistan-specific
    include_keywords = [
        # Landscapes (plan.txt)
        'valley', 'valleys', 'river valley', 'river', 'rivers',
        'glacier', 'glaciers', 'glacier view', 'glacier views',
        # Villages & houses (plan.txt)
        'village', 'villages', 'tree-lined village', 'tree-lined villages',
        'stone house', 'stone houses', 'stone-built', 'stone built',
        'small settlement', 'small settlements', 'settlement', 'settlements',
        # Bridges (plan.txt)
        'suspension bridge', 'suspension bridges',
        # Seasonal (plan.txt)
        'snowy road', 'snowy roads', 'snow covered road', 'snow covered roads',
        'autumn color', 'autumn colors', 'autumn valley',
        'spring meltwater', 'meltwater',
        # General outdoor
        'landscape', 'outdoor', 'scenic', 'mountain', 'mountains',
    ]
    
    # If it contains include keywords, it's likely geolocation-relevant
    for keyword in include_keywords:
        if keyword in full_text:
            return True
    
    # Looser fallback for GB: if clearly in GB and not indoor/retail, keep it
    gb_markers = [
        'gilgit-baltistan', 'gilgit baltistan',
        'gilgit', 'skardu', 'hunza', 'nagar', 'ghizer', 'gupis',
        'yasin', 'astore', 'diamir', 'ghanche',
    ]
    if any(m in full_text for m in gb_markers):
        return True
    
    # Otherwise drop
    return False

def search_places(query_text, max_results=20, restrict_to_pakistan=True):
    """
    Search for places using a query
    Returns list of places with name, location, and photos
    Filters to Pakistan only if restrict_to_pakistan is True
    """
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        "X-Goog-FieldMask": "places.id,places.displayName,places.location,places.photos,places.formattedAddress,places.types"
    }
    
    payload = {
        "textQuery": query_text,
        "maxResultCount": min(max_results, MAX_RESULTS_PER_QUERY)
    }
    
        # Add location restriction for Pakistan (especially important for border provinces to avoid India)
    if restrict_to_pakistan:
        payload["locationRestriction"] = {
            "rectangle": {
                "low": {
                    "latitude": PAKISTAN_BOUNDS['south'],
                    "longitude": PAKISTAN_BOUNDS['west']
                },
                "high": {
                    "latitude": PAKISTAN_BOUNDS['north'],
                    "longitude": PAKISTAN_BOUNDS['east']
                }
            }
    }
    
    try:
        response = requests.post(PLACES_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code != 200:
            if response.status_code == 429:
                print(f"  ⚠️  Rate limit hit, waiting longer...")
                time.sleep(5)
                return []
            print(f"  ⚠️  API error {response.status_code}")
            return []
        
        data = response.json()
        places = data.get('places', [])
        
        # Additional filtering: verify places are in Pakistan by coordinates
        if restrict_to_pakistan:
            places = [p for p in places if is_in_pakistan(p.get('location', {}))]
        
        # Filter for geolocation-relevant places only (outdoor, no indoor/people photos)
        places = [p for p in places if is_geolocation_relevant(p)]
        
        return places
        
    except Exception as e:
        print(f"  ⚠️  Error searching: {e}")
        return []

def get_place_photo(photo_reference):
    """Get photo reference - returns the photo reference path (no API key in URL)"""
    if not photo_reference:
        return None
    # Return the photo reference as-is (it's already the full path from API)
    # API key will be in headers when downloading
    return photo_reference

def download_test_image(image_url, place_id, province_name, existing_image_filenames, image_index=1):
    """Download a high-quality test image with proper authentication
    Checks for duplicate filenames before downloading
    image_index: Index of the image (1-5) to create unique filenames
    """
    # Generate expected filename with image index
    expected_filename = f"{province_name}_{place_id[:8]}_{image_index}.jpg"
    
    # Check if image already exists
    if expected_filename in existing_image_filenames:
        output_path = IMAGES_DIR / expected_filename
        if output_path.exists():
            print(f"    ⏭️  Image already exists: {expected_filename}")
            return str(output_path)
    
    try:
        headers = {
            "X-Goog-Api-Key": GOOGLE_API_KEY
        }
        # Use higher resolution for better quality images (maxWidthPx: 1600 for high quality)
        params = {
            "maxWidthPx": 1600  # Increased from 800 for better quality
        }
        response = requests.get(image_url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')
            output_path = IMAGES_DIR / expected_filename
            
            # Double-check file doesn't exist (race condition protection)
            if output_path.exists():
                print(f"    ⏭️  Image already exists: {expected_filename}")
                return str(output_path)
            
            # Save with high quality
            img.save(output_path, 'JPEG', quality=95)
            # Add to existing set to prevent duplicates in same run
            existing_image_filenames.add(expected_filename)
            return str(output_path)
        else:
            print(f"    ⚠️  HTTP {response.status_code}: {response.text[:100]}")
    except Exception as e:
        print(f"    ⚠️  Could not download image: {e}")
    return None

def load_existing_place_ids():
    """Load place IDs from existing CSV and final.csv to skip duplicates
    Returns both all IDs and base IDs (without _1, _2 suffix) to avoid re-processing same places
    Also checks final.csv to prevent duplicates across all files
    """
    existing_ids = set()
    existing_base_ids = set()  # Base IDs without suffix (e.g., ChIJ1234 from ChIJ1234_1)
    
    # Check current CSV file
    if CSV_PATH.exists():
        try:
            df = pd.read_csv(CSV_PATH)
            if 'id' in df.columns:
                all_ids = set(df['id'].dropna().astype(str))
                existing_ids.update(all_ids)
                
                # Extract base place IDs (remove _1, _2, etc. suffix)
                for place_id in all_ids:
                    # Check if ID has suffix pattern (e.g., ChIJ1234_1)
                    if '_' in place_id:
                        # Extract base ID (everything before last underscore and number)
                        parts = place_id.rsplit('_', 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            existing_base_ids.add(parts[0])
                    else:
                        # No suffix, use as base ID
                        existing_base_ids.add(place_id)
                
                print(f"✓ Loaded {len(all_ids)} existing place IDs from {CSV_PATH.name}")
        except Exception as e:
            print(f"⚠️  Could not load existing place IDs from {CSV_PATH.name}: {e}")
    
    # Also check final.csv to prevent duplicates across files
    final_csv_path = Path(__file__).parent.parent / 'final.csv'
    if final_csv_path.exists():
        try:
            df_final = pd.read_csv(final_csv_path)
            if 'id' in df_final.columns:
                final_ids = set(df_final['id'].dropna().astype(str))
                existing_ids.update(final_ids)
                
                # Extract base place IDs from final.csv
                for place_id in final_ids:
                    if '_' in place_id:
                        parts = place_id.rsplit('_', 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            existing_base_ids.add(parts[0])
                    else:
                        existing_base_ids.add(place_id)
                
                print(f"✓ Loaded {len(final_ids)} existing place IDs from final.csv")
        except Exception as e:
            print(f"⚠️  Could not load existing place IDs from final.csv: {e}")
    
    print(f"✓ Total unique base place IDs to skip: {len(existing_base_ids)}")
    return existing_ids, existing_base_ids

def load_existing_image_filenames():
    """Load existing image filenames to prevent duplicate downloads
    Checks both the images directory and any references in CSV files
    """
    existing_images = set()
    initial_count = 0
    
    # Check images directory
    if IMAGES_DIR.exists():
        existing_images = {f.name for f in IMAGES_DIR.glob('*.jpg')}
        initial_count = len(existing_images)
        print(f"✓ Loaded {initial_count} existing image filenames from directory")
    
    # Also check final.csv for image paths to prevent duplicates
    final_csv_path = Path(__file__).parent.parent / 'final.csv'
    if final_csv_path.exists():
        try:
            df_final = pd.read_csv(final_csv_path)
            if 'path' in df_final.columns:
                # Extract filenames from paths
                additional_count = 0
                for path in df_final['path'].dropna():
                    if isinstance(path, str) and path.endswith('.jpg'):
                        filename = Path(path).name
                        if filename not in existing_images:
                            existing_images.add(filename)
                            additional_count += 1
                if additional_count > 0:
                    print(f"✓ Loaded additional {additional_count} image filenames from final.csv")
        except Exception as e:
            print(f"⚠️  Could not load image filenames from final.csv: {e}")
    
    print(f"✓ Total unique image filenames to skip: {len(existing_images)}")
    return existing_images

def extract_places_data():
    """Extract places data for Gilgit-Baltistan, Pakistan - Dataset Augmentation
    Focus: OUTDOOR geolocation-relevant places that help model distinguish Gilgit-Baltistan visually
    - Landscapes: valleys with rivers, partial glacier views, tree-lined villages
    - Human presence: small settlements, stone houses, suspension bridges
    - Seasonal: snowy roads, autumn colors, spring meltwater
    Excludes: Indoor/retail/brands (noisy)
    Target: ~1000 places with 5 images each = ~5000 images for dataset augmentation
    """
    print("=" * 70)
    print("Places API - Data Extraction for Gilgit-Baltistan, Pakistan (Dataset Augmentation)")
    print("Focus: Valleys, villages, bridges, seasonal variation")
    print("Excludes: Indoor/retail/brand places (noisy)")
    print("Target: ~1000 places × 5 images = ~5000 images")
    print("=" * 70)
    
    if not GOOGLE_API_KEY:
        print("❌ Error: GOOGLE_API_KEY not found in environment variables!")
        return
    
    print(f"✓ API Key loaded")
    
    # Load existing place IDs to skip duplicates
    existing_place_ids, existing_base_ids = load_existing_place_ids()
    
    # Load existing image filenames to prevent duplicate downloads
    existing_image_filenames = load_existing_image_filenames()
    
    print(f"\n📋 Processing: Gilgit-Baltistan, Pakistan (Dataset Augmentation)")
    print(f"   Focus: Valleys with rivers, villages, bridges, seasonal variation")
    print(f"\n⚙️  Configuration:")
    print(f"   Target places per province: {TARGET_PLACES_PER_PROVINCE}")
    print(f"   Images per place: 5")
    print(f"   Total target images: ~{TARGET_PLACES_PER_PROVINCE * 5}")
    print(f"   Max results per query: {MAX_RESULTS_PER_QUERY}")
    print(f"   Image quality: High (1600px width, 95% JPEG quality)")
    total_specific_places = sum(len(places) for places in PROVINCE_SPECIFIC_PLACES.values())
    print(f"   Specific places to search: {total_specific_places} across all provinces")
    print(f"   Generic queries per province: {len(GENERIC_SEARCH_QUERIES)}")
    print(f"   Rate limiting: {DELAY_BETWEEN_QUERIES}s between queries, {DELAY_BETWEEN_DOWNLOADS}s between downloads")
    print(f"   Existing places to skip: {len(existing_place_ids)}")
    print(f"   Pakistan bounds: (lat: {PAKISTAN_BOUNDS['south']}-{PAKISTAN_BOUNDS['north']}, lng: {PAKISTAN_BOUNDS['west']}-{PAKISTAN_BOUNDS['east']})")
    print(f"   Gilgit-Baltistan bounds: (lat: {GILGIT_BALTISTAN_BOUNDS['south']}-{GILGIT_BALTISTAN_BOUNDS['north']}, lng: {GILGIT_BALTISTAN_BOUNDS['west']}-{GILGIT_BALTISTAN_BOUNDS['east']})")
    print("\n" + "-" * 70)
    
    all_places = []
    
    for province_name, province_id in PROVINCES.items():
        print(f"\n🔍 Searching places in {province_name}...")
        print(f"   Target: {TARGET_PLACES_PER_PROVINCE} places with images")
        
        all_province_places = []
        seen_place_ids = set()
        
        # First, search for specific places in this province
        specific_places = PROVINCE_SPECIFIC_PLACES.get(province_name, [])
        if specific_places:
            print(f"   Searching {len(specific_places)} specific places...")
            for place_name in specific_places:
                if len(all_province_places) >= TARGET_PLACES_PER_PROVINCE:
                    break
                
                # Search for this specific place - include province to reduce drift
                query = f"{place_name} Gilgit-Baltistan Pakistan"
                print(f"   Query: '{query}'...")
                
                # Always restrict to Pakistan
                places = search_places(query, max_results=MAX_RESULTS_PER_QUERY, restrict_to_pakistan=True)
                time.sleep(DELAY_BETWEEN_QUERIES)  # Rate limiting between queries
                
                if not places:
                    continue
                
                # Add unique places (skip if base place ID already exists in CSV)
                for place in places:
                    place_id = place.get('id', '')
                    # Check if base place ID (without suffix) already exists
                    if place_id and place_id not in seen_place_ids and place_id not in existing_base_ids:
                        seen_place_ids.add(place_id)
                        all_province_places.append(place)
                
                print(f"   Found {len(places)} places ({len(all_province_places)} unique so far)")
        
        # Then, use generic queries to get more diverse results
        if len(all_province_places) < TARGET_PLACES_PER_PROVINCE:
            print(f"   Using generic queries to reach target...")
            for query_template in GENERIC_SEARCH_QUERIES:
                if len(all_province_places) >= TARGET_PLACES_PER_PROVINCE:
                    break
                
                query = query_template.format(province=province_name)
                print(f"   Query: '{query}'...")
                
                # Always restrict to Pakistan
                places = search_places(query, max_results=MAX_RESULTS_PER_QUERY, restrict_to_pakistan=True)
                time.sleep(DELAY_BETWEEN_QUERIES)  # Rate limiting between queries
                
                if not places:
                    continue
                
                # Add unique places (skip if base place ID already exists in CSV)
                # Also validate Gilgit-Baltistan bounds before adding
                for place in places:
                    place_id = place.get('id', '')
                    location = place.get('location', {})
                    # Check if base place ID (without suffix) already exists
                    # AND validate it's within Gilgit-Baltistan bounds
                    if place_id and place_id not in seen_place_ids and place_id not in existing_base_ids:
                        if is_in_gilgit_baltistan(location):
                            seen_place_ids.add(place_id)
                            all_province_places.append(place)
                        else:
                            display_name = place.get('displayName', {}).get('text', 'Unknown')
                            lat = location.get('latitude', 0)
                            lng = location.get('longitude', 0)
                            print(f"      ⚠️  Skipping {display_name} - outside Gilgit-Baltistan bounds (lat: {lat:.4f}, lng: {lng:.4f})")
                
                print(f"   Found {len(places)} places ({len(all_province_places)} unique so far)")
        
        # If still short, try searching with major cities from the province
        if len(all_province_places) < TARGET_PLACES_PER_PROVINCE:
            print(f"   Searching by major cities to reach target...")
            major_cities = PROVINCE_SPECIFIC_PLACES.get(province_name, [])[:10]  # Top 10 cities
            for city in major_cities:
                if len(all_province_places) >= TARGET_PLACES_PER_PROVINCE:
                    break
                
                # Try different query variations for each seed term
                city_queries = [
                    f"{city} Gilgit-Baltistan Pakistan",
                    f"{city} Gilgit-Baltistan Pakistan landscape",
                    f"{city} Gilgit-Baltistan Pakistan valley",
                ]
                
                for query in city_queries:
                    if len(all_province_places) >= TARGET_PLACES_PER_PROVINCE:
                        break
                    
                    print(f"   Query: '{query}'...")
                    # Always restrict to Pakistan
                    places = search_places(query, max_results=MAX_RESULTS_PER_QUERY, restrict_to_pakistan=True)
                    time.sleep(DELAY_BETWEEN_QUERIES)
                    
                    if not places:
                        continue
                    
                    # Add unique places (skip if already exists in CSV)
                    # Also validate Gilgit-Baltistan bounds before adding
                    for place in places:
                        place_id = place.get('id', '')
                        location = place.get('location', {})
                        if place_id and place_id not in seen_place_ids and place_id not in existing_place_ids:
                            if is_in_gilgit_baltistan(location):
                                seen_place_ids.add(place_id)
                                all_province_places.append(place)
                            else:
                                display_name = place.get('displayName', {}).get('text', 'Unknown')
                                lat = location.get('latitude', 0)
                                lng = location.get('longitude', 0)
                                print(f"      ⚠️  Skipping {display_name} - outside Gilgit-Baltistan bounds (lat: {lat:.4f}, lng: {lng:.4f})")
                    
                    print(f"   Found {len(places)} places ({len(all_province_places)} unique so far)")
        
        if not all_province_places:
            print(f"  ⚠️  No places found for {province_name}")
            continue
        
        print(f"  ✓ Total unique places found: {len(all_province_places)}")
        print(f"  📥 Processing and downloading images...")
        
        # Process places and download images (up to 5 images per place)
        MAX_IMAGES_PER_PLACE = 5
        province_places_with_images = 0
        for place in all_province_places:
            place_id = place.get('id', '')
            display_name = place.get('displayName', {}).get('text', 'Unknown')
            location = place.get('location', {})
            lat = location.get('latitude', None)
            lng = location.get('longitude', None)
            photos = place.get('photos', [])
            
            # Skip if location is invalid
            if lat is None or lng is None:
                print(f"    ⚠️  Skipping {display_name} - invalid location data")
                continue
            
            # Double-check location is in Pakistan
            if not is_in_pakistan(location):
                print(f"    ⚠️  Skipping {display_name} - location outside Pakistan bounds")
                continue
            
            # Strict validation: Must be within Gilgit-Baltistan bounds
            if not is_in_gilgit_baltistan(location):
                print(f"    ⚠️  Skipping {display_name} - location outside Gilgit-Baltistan bounds (lat: {lat:.4f}, lng: {lng:.4f})")
                continue
            
            # Normalize province name according to mapping
            normalized_province = PROVINCE_NORMALIZE.get(province_name, province_name)
            
            # Download up to 5 images per place (for diversity), but only add 1 row to CSV per place ID
            downloaded_images = []
            num_photos_to_download = min(len(photos), MAX_IMAGES_PER_PLACE)
            
            if not photos:
                print(f"    ⚠️  Skipping {display_name} - no photos available")
                continue
            
            print(f"    📥 Downloading up to {num_photos_to_download} images for {display_name}...")
            
            for photo_idx in range(num_photos_to_download):
                photo = photos[photo_idx]
                photo_ref = photo.get('name', '')
                
                if not photo_ref:
                    continue
                
                # Convert photo reference to full media URL
                if photo_ref.startswith('http'):
                    media_url = f"{photo_ref}/media" if not photo_ref.endswith('/media') else photo_ref
                elif photo_ref.startswith('places/'):
                    media_url = f"https://places.googleapis.com/v1/{photo_ref}/media"
                else:
                    # If it's just a photo ID, construct full path
                    media_url = f"https://places.googleapis.com/v1/places/{place_id}/photos/{photo_ref}/media"
                
                # Generate unique filename for each image (place_id_image_index.jpg)
                image_index = photo_idx + 1
                downloaded_path = download_test_image(media_url, place_id, normalized_province, existing_image_filenames, image_index)
                
                if downloaded_path:
                    # Store relative path from project root
                    local_path = f"datasets/pakistan_images/places_api_images/{normalized_province}_{place_id[:8]}_{image_index}.jpg"
                    downloaded_images.append(local_path)
                    print(f"    ✓ Image {image_index}/{num_photos_to_download} saved")
                else:
                    print(f"    ⚠️  Image {image_index}/{num_photos_to_download} failed to download")
                
                time.sleep(DELAY_BETWEEN_DOWNLOADS)  # Rate limiting between downloads
            
            # Add one CSV row per image, with unique ID for each (place_id_1, place_id_2, etc.)
            if downloaded_images:
                for img_idx, local_path in enumerate(downloaded_images):
                    # Create unique ID by appending image index (e.g., ChIJ1234_1, ChIJ1234_2)
                    unique_id = f"{place_id}_{img_idx + 1}"
                    
                    place_data = {
                        'id': unique_id,  # Unique ID per image
                        'latitude': lat,
                        'longitude': lng,
                        'province': normalized_province,
                        'path': local_path  # Local path to downloaded image
                    }
                    all_places.append(place_data)
                    province_places_with_images += 1
                
                print(f"    ✅ Added {len(downloaded_images)} CSV rows for {display_name} (IDs: {place_id}_1 to {place_id}_{len(downloaded_images)})")
            else:
                print(f"    ⚠️  No images downloaded for {display_name}")
            
            # Progress update every 10 places
            if province_places_with_images > 0 and province_places_with_images % 10 == 0:
                print(f"  Progress: {province_places_with_images} images downloaded...")
        
        print(f"  ✅ {province_name}: {province_places_with_images} places with images added")
    
    if not all_places:
        print("\n⚠️  No places data extracted!")
        return
    
    # Create DataFrame with only required columns
    df_new = pd.DataFrame(all_places)
    
    # Ensure only required columns: id, latitude, longitude, province, path
    required_columns = ['id', 'latitude', 'longitude', 'province', 'path']
    df_new = df_new[required_columns]
    
    # Load existing CSV if it exists
    existing_count = 0
    if CSV_PATH.exists():
        try:
            df_existing = pd.read_csv(CSV_PATH)
            existing_count = len(df_existing)
            print(f"\n📖 Found existing CSV with {existing_count} places")
            
            # Combine new and existing data
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            
            # Remove duplicates based on 'id' column (keep first occurrence)
            # Each image has unique ID (place_id_1, place_id_2, etc.), so this prevents exact duplicates
            before_dedup = len(df_combined)
            df_combined = df_combined.drop_duplicates(subset=['id'], keep='first')
            duplicates_removed = before_dedup - len(df_combined)
            
            if duplicates_removed > 0:
                print(f"   ⚠️  Removed {duplicates_removed} duplicate IDs from CSV (keeping first occurrence)")
            
            print(f"   After removing duplicates: {len(df_combined)} total places")
            print(f"   New places added: {len(df_combined) - existing_count}")
        except Exception as e:
            print(f"⚠️  Could not read existing CSV: {e}")
            print("   Creating new CSV file...")
            df_combined = df_new
    else:
        print(f"\n📝 Creating new CSV file...")
        df_combined = df_new
    
    # Save to CSV (append mode - overwrite with combined data)
    print(f"\n💾 Saving to CSV...")
    df_combined.to_csv(CSV_PATH, index=False)
    print(f"   ✅ Saved {len(df_combined)} total places to: {CSV_PATH}")
    print(f"   Columns: {', '.join(df_combined.columns)}")
    
    # Use combined dataframe for summary
    df = df_combined
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 Summary")
    print("=" * 70)
    print(f"Total places extracted: {len(df):,}")
    print(f"\nBy province:")
    for province in PROVINCES.keys():
        count = len(df[df['province'] == province])
        print(f"  {province}: {count}")
    
    print(f"\nPlaces with photos: {(df['path'] != '').sum()}")
    print(f"Places without photos: {(df['path'] == '').sum()}")
    print(f"Images downloaded: {len(list(IMAGES_DIR.glob('*.jpg')))}")
    
    # Estimate API usage and cost
    total_specific_queries = sum(len(places) for places in PROVINCE_SPECIFIC_PLACES.values())
    total_generic_queries = len(PROVINCES) * len(GENERIC_SEARCH_QUERIES)
    estimated_queries = total_specific_queries + total_generic_queries
    estimated_photos = len(df)
    estimated_cost_queries = (estimated_queries / 1000) * 17  # $17 per 1000 searchText requests
    estimated_cost_photos = (estimated_photos / 1000) * 7     # $7 per 1000 photo media requests
    total_estimated_cost = estimated_cost_queries + estimated_cost_photos
    
    print(f"\n💰 Estimated API Usage:")
    print(f"   Search queries: ~{estimated_queries}")
    print(f"   Photo downloads: ~{estimated_photos}")
    print(f"   Estimated cost: ~${total_estimated_cost:.2f}")
    
    print(f"\n✅ Complete! Check {CSV_PATH} and {IMAGES_DIR}")

if __name__ == "__main__":
    extract_places_data()
