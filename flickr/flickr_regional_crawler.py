import csv
import time
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
import requests
import os
from pathlib import Path

load_dotenv()

FLICKR_API_KEY = os.getenv("FLICKR_API_KEY")
FLICKR_API_SECRET = os.getenv("FLICKR_API_SECRET")

FLICKR_API_ENDPOINT = "https://api.flickr.com/services/rest/"


def flickr_api_call(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Low-level helper to call the Flickr REST API.
    """
    base_params = {
        "api_key": FLICKR_API_KEY,
        "format": "json",
        "nojsoncallback": 1,
    }
    response = requests.get(FLICKR_API_ENDPOINT, params={**base_params, **params}, timeout=30)
    response.raise_for_status()
    data = response.json()

    if data.get("stat") != "ok":
        raise RuntimeError(f"Flickr API error: {data}")

    return data


# Target regions with their bounding boxes and geography-relevant tags
TARGET_REGIONS = {
    "Sindh": {
        "bbox": (66.0, 23.5, 71.0, 28.5),  # Approximate bounding box
        "tags": [
            "Sindh",
            "Karachi",
            "Hyderabad",
            "Sukkur",
            "Larkana",
            "Thatta",
            "Mohenjo-daro",
            "Makli",
            "Clifton",
            "Seaview",
            "Indus River",
            "Thar Desert",
        ],
    },
    "Punjab": {
        "bbox": (70.0, 28.5, 75.0, 33.0),  # Approximate bounding box
        "tags": [
            "Punjab",
            "Lahore",
            "Multan",
            "Faisalabad",
            "Rawalpindi",
            "Gujranwala",
            "Sialkot",
            "Badshahi Mosque",
            "Lahore Fort",
            "Shalimar Gardens",
            "Wagah Border",
            "Taxila",
            "Rohtas Fort",
        ],
    },
    "Gilgit-Baltistan": {
        "bbox": (72.0, 35.0, 77.0, 37.0),  # Approximate bounding box
        "tags": [
            "Gilgit-Baltistan",
            "Gilgit",
            "Baltistan",
            "Hunza",
            "Skardu",
            "Karakoram",
            "Nanga Parbat",
            "K2",
            "Baltoro",
            "Fairy Meadows",
        ],
    },
    "Khyber Pakhtunkhwa": {
        "bbox": (70.0, 31.0, 74.0, 35.0),  # Approximate bounding box
        "tags": [
            "Khyber Pakhtunkhwa",
            "KPK",
            "Peshawar",
            "Swat",
            "Malam Jabba",
            "Kalam",
            "Chitral",
            "Hunza Valley",
            "Khyber Pass",
            "Dir",
            "Abbottabad",
            "Mardan",
        ],
    },
    "Azad Kashmir": {
        "bbox": (73.0, 33.0, 75.0, 35.0),  # Approximate bounding box
        "tags": [
            "Azad Kashmir",
            "AJK",
            "Muzaffarabad",
            "Neelum Valley",
            "Jhelum Valley",
            "Mirpur",
            "Kotli",
            "Rawalakot",
        ],
    },
    "Balochistan": {
        "bbox": (60.0, 24.0, 70.0, 32.0),  # Approximate bounding box
        "tags": [
            "Balochistan",
            "Quetta",
            "Gwadar",
            "Turbat",
            "Chaman",
            "Ziarat",
            "Hingol National Park",
            "Makran",
            "Lasbela",
        ],
    },
    "Islamabad": {
        "bbox": (72.8, 33.5, 73.2, 33.8),  # Approximate bounding box for Islamabad
        "tags": [
            "Islamabad",
            "Islamabad Capital Territory",
            "ICT",
            "Faisal Mosque",
            "Margalla Hills",
            "Daman-e-Koh",
            "Rawalpindi",
        ],
    },
}


def pick_url(photo: Dict[str, Any]) -> Optional[str]:
    """
    Pick a single URL for a photo based on priority:
    url_o → url_l → url_c → url_m.
    """
    for key in ["url_o", "url_l", "url_c", "url_m"]:
        val = photo.get(key)
        if isinstance(val, str):
            val = val.strip()
        if val:
            return val
    return None


def fetch_photos_for_region(
    region_name: str,
    region_config: Dict[str, Any],
    per_page: int = 250,
    max_pages: Optional[int] = None,
    sleep_between_pages: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Fetch photos for a specific region using bounding box and tags.
    Returns a list of photo dictionaries with photo_id, latitude, longitude, path.
    """
    bbox = region_config["bbox"]
    tags = region_config["tags"]
    
    # Format bbox as string: "min_lon,min_lat,max_lon,max_lat"
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    
    # Combine tags into a search query
    tags_query = " OR ".join(tags)
    
    extras = ",".join(
        [
            "geo",
            "date_taken",
            "date_upload",
            "owner_name",
            "url_o",
            "url_l",
            "url_c",
            "url_m",
            "tags",
        ]
    )
    
    base_params: Dict[str, Any] = {
        "method": "flickr.photos.search",
        "bbox": bbox_str,
        "has_geo": 1,
        "extras": extras,
        "content_type": 1,  # photos only
        "media": "photos",
        "per_page": per_page,
        "page": 1,
        "safe_search": 1,
        "sort": "relevance",
        "text": tags_query,  # Use tags as search text
    }
    
    # First call to discover total pages
    try:
        first_page = flickr_api_call(base_params)
        photos_meta = first_page["photos"]
        total_pages = photos_meta["pages"]
        
        if max_pages is not None:
            total_pages = min(total_pages, max_pages)
        
        print(f"  {region_name}: Total available pages: {photos_meta['pages']}, fetching: {total_pages}")
        
        all_photos = []
        
        # Process first page
        photos = photos_meta.get("photo", [])
        for p in photos:
            lat = p.get("latitude")
            lon = p.get("longitude")
            if not lat or not lon:
                continue
            
            url = pick_url(p)
            if not url:
                continue
            
            all_photos.append({
                "photo_id": p.get("id"),
                "latitude": lat,
                "longitude": lon,
                "path": url,
            })
        
        print(f"  {region_name}: Page 1 / {total_pages} processed with {len(photos)} photos ({len(all_photos)} after filtering).")
        
        # Fetch remaining pages
        for page in range(2, total_pages + 1):
            time.sleep(sleep_between_pages)
            params = dict(base_params)
            params["page"] = page
            try:
                data = flickr_api_call(params)
                photos = data["photos"].get("photo", [])
                
                page_photos = []
                for p in photos:
                    lat = p.get("latitude")
                    lon = p.get("longitude")
                    if not lat or not lon:
                        continue
                    
                    url = pick_url(p)
                    if not url:
                        continue
                    
                    page_photos.append({
                        "photo_id": p.get("id"),
                        "latitude": lat,
                        "longitude": lon,
                        "path": url,
                    })
                
                all_photos.extend(page_photos)
                print(f"  {region_name}: Page {page} / {total_pages} processed with {len(photos)} photos ({len(page_photos)} after filtering).")
            except Exception as e:
                print(f"  {region_name}: Error on page {page}: {e}")
                continue
        
        return all_photos
    
    except Exception as e:
        print(f"  {region_name}: Error fetching photos: {e}")
        return []


def fetch_regional_photos(
    output_csv: str = None,
    per_page: int = 250,
    max_pages_per_region: Optional[int] = None,
    sleep_between_pages: float = 0.5,
) -> None:
    """
    Fetch geo-tagged Flickr photos for all regions:
    - Sindh
    - Punjab
    - Gilgit-Baltistan
    - Khyber Pakhtunkhwa
    - Azad Kashmir
    - Balochistan
    - Islamabad
    
    Parameters
    ----------
    output_csv : str, optional
        Path to the CSV file that will be written. If None, defaults to flickr/regional_photos_with_geo.csv
    per_page : int
        Number of results per page (max 250 for Flickr).
    max_pages_per_region : int or None
        Maximum number of pages to fetch per region. Use None to fetch all pages.
    sleep_between_pages : float
        Seconds to sleep between page requests to be gentle on the API.
    """
    
    # Set default output path to flickr folder
    if output_csv is None:
        script_dir = Path(__file__).parent
        output_csv = str(script_dir / "regional_photos_with_geo.csv")
    
    print("=" * 70)
    print("Flickr Regional Crawler")
    print("Target regions: Sindh, Punjab, Gilgit-Baltistan, Khyber Pakhtunkhwa, Azad Kashmir, Balochistan, Islamabad")
    print("=" * 70)
    
    fieldnames = ["photo_id", "latitude", "longitude", "path"]
    
    all_photos = []
    seen_photo_ids = set()  # To avoid duplicates
    
    # Fetch photos for each target region
    for region_name, region_config in TARGET_REGIONS.items():
        print(f"\nFetching photos for {region_name}...")
        photos = fetch_photos_for_region(
            region_name,
            region_config,
            per_page=per_page,
            max_pages=max_pages_per_region,
            sleep_between_pages=sleep_between_pages,
        )
        
        # Add photos, avoiding duplicates
        for photo in photos:
            photo_id = photo["photo_id"]
            if photo_id not in seen_photo_ids:
                seen_photo_ids.add(photo_id)
                all_photos.append(photo)
        
        print(f"  {region_name}: Added {len(photos)} unique photos (total so far: {len(all_photos)})")
        time.sleep(1)  # Brief pause between regions
    
    # Write to CSV
    print(f"\nWriting {len(all_photos)} unique photos to {output_csv}...")
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_photos)
    
    print(f"\nDone. CSV saved to: {output_csv}")
    print(f"Total unique photos: {len(all_photos)}")


if __name__ == "__main__":
    # Example usage
    fetch_regional_photos(
        output_csv=None,  # Will default to flickr/regional_photos_with_geo.csv
        per_page=250,
        max_pages_per_region=None,
        sleep_between_pages=0.5,
    )
