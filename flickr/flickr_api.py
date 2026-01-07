import csv
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import requests
import os
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


def fetch_pakistan_photos(
    text: Optional[str] = "Pakistan",
    per_page: int = 250,
    max_pages: Optional[int] = None,
    min_upload_date: Optional[str] = None,
    output_csv: str = "pakistan_photos_with_geo.csv",
    sleep_between_pages: float = 0.5, 
) -> None:
    """
    Fetch geo-tagged Flickr photos taken in/around Pakistan and save to CSV.

    Parameters
    ----------
    text : str, optional
        Free-text search query (default "Pakistan").
    per_page : int
        Number of results per page (max 250 for Flickr).
    max_pages : int or None
        Maximum number of pages to fetch. Use None to fetch all pages.
    min_upload_date : str, optional
        Only fetch photos uploaded after this date (UNIX timestamp or YYYY-MM-DD).
    output_csv : str
        Path to the CSV file that will be written.
    sleep_between_pages : float
        Seconds to sleep between page requests to be gentle on the API.
    """

    # Rough bounding box for Pakistan: (min_lon, min_lat, max_lon, max_lat)
    # Values taken from approximate country bounds.
    pakistan_bbox = "60.87,23.63,77.84,37.09"

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
        "bbox": pakistan_bbox,
        "has_geo": 1,
        "extras": extras,
        "content_type": 1,  # photos only
        "media": "photos",
        "per_page": per_page,
        "page": 1,
        "safe_search": 1,
        "sort": "relevance",
    }

    if text:
        base_params["text"] = text

    if min_upload_date:
        base_params["min_upload_date"] = min_upload_date

    # First call to discover total pages
    first_page = flickr_api_call(base_params)
    photos_meta = first_page["photos"]
    total_pages = photos_meta["pages"]

    if max_pages is not None:
        total_pages = min(total_pages, max_pages)

    print(f"Total available pages: {photos_meta['pages']}, fetching: {total_pages}")

    # We only keep these final fields in the CSV.
    fieldnames = [
        "photo_id",
        "latitude",
        "longitude",
        "accuracy",
        "url",
    ]

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

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

        def write_photos(photos: List[Dict[str, Any]]) -> None:
            for p in photos:
                # Some photos might still lack geo if Flickr marks them differently
                lat = p.get("latitude")
                lon = p.get("longitude")
                if not lat or not lon:
                    continue

                # Select URL based on priority; skip photos without any usable URL.
                url = pick_url(p)
                if not url:
                    continue

                row = {
                    "photo_id": p.get("id"),
                    "latitude": lat,
                    "longitude": lon,
                    "accuracy": p.get("accuracy"),
                    "url": url,
                }
                writer.writerow(row)

        # Write first page
        write_photos(photos_meta.get("photo", []))
        print(f"Page 1 / {total_pages} written with {len(photos_meta.get('photo', []))} photos.")

        # Fetch remaining pages
        for page in range(2, total_pages + 1):
            time.sleep(sleep_between_pages)
            params = dict(base_params)
            params["page"] = page
            data = flickr_api_call(params)
            photos = data["photos"].get("photo", [])
            write_photos(photos)
            print(f"Page {page} / {total_pages} written with {len(photos)} photos.")

    print(f"Done. CSV saved to: {output_csv}")


if __name__ == "__main__":
    # Basic example usage.
    # You can customize arguments below as needed.
    fetch_pakistan_photos()


