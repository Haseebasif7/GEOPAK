"""
Download Pakistan provinces shapefile from GADM
Multiple download sources with fallback options
"""
import urllib.request
import zipfile
from pathlib import Path

# Try multiple sources
DOWNLOAD_URLS = [
    "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_PAK_1.zip",
    "https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/PAK/ADM1/geoBoundaries-PAK-ADM1-all.zip",
    "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_1_states_provinces.zip",
]

OUTPUT_DIR = Path(__file__).parent
SHAPEFILE_PATH = OUTPUT_DIR / "gadm41_PAK_1.shp"

print("=" * 70)
print("Downloading Pakistan Provinces Shapefile")
print("=" * 70)
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 70)

if SHAPEFILE_PATH.exists():
    print(f"\n✓ Shapefile already exists at: {SHAPEFILE_PATH}")
    print("Skipping download.")
else:
    print("\nAttempting to download from multiple sources...")
    
    success = False
    for i, url in enumerate(DOWNLOAD_URLS, 1):
        zip_filename = url.split('/')[-1]
        ZIP_PATH = OUTPUT_DIR / zip_filename
        
        print(f"\n[{i}/{len(DOWNLOAD_URLS)}] Trying: {url}")
        try:
            urllib.request.urlretrieve(url, ZIP_PATH)
            print(f"✓ Downloaded: {ZIP_PATH}")
            
            print("Extracting...")
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(OUTPUT_DIR)
            print(f"✓ Extracted to: {OUTPUT_DIR}")
            
            # Look for shapefile (may have different names)
            possible_names = [
                "gadm41_PAK_1.shp",
                "geoBoundaries-PAK-ADM1.shp",
                "ne_10m_admin_1_states_provinces.shp",
            ]
            
            found_shapefile = None
            for name in possible_names:
                test_path = OUTPUT_DIR / name
                if test_path.exists():
                    found_shapefile = test_path
                    break
            
            # Also search for any .shp file
            if not found_shapefile:
                shp_files = list(OUTPUT_DIR.glob("*.shp"))
                if shp_files:
                    found_shapefile = shp_files[0]
            
            if found_shapefile and found_shapefile != SHAPEFILE_PATH:
                # Rename to expected name
                found_shapefile.rename(SHAPEFILE_PATH)
                print(f"✓ Renamed to: {SHAPEFILE_PATH}")
            
            # Clean up zip file
            ZIP_PATH.unlink()
            print(f"✓ Removed zip file")
            
            if SHAPEFILE_PATH.exists():
                print(f"\n✓ Shapefile ready at: {SHAPEFILE_PATH}")
                success = True
                break
            else:
                print(f"⚠ Shapefile not found at expected location")
                
        except Exception as e:
            print(f"✗ Failed: {e}")
            if ZIP_PATH.exists():
                ZIP_PATH.unlink()
            continue
    
    if not success:
        print("\n" + "=" * 70)
        print("✗ All download sources failed")
        print("\nManual download instructions:")
        print("=" * 70)
        print("\nOption 1 - GADM (Recommended):")
        print("  1. Visit: https://gadm.org/download_country.html")
        print("  2. Select 'Pakistan' and 'Shapefile' format")
        print("  3. Download 'Level 1' (provinces)")
        print(f"  4. Extract to: {OUTPUT_DIR}")
        print("  5. Rename the .shp file to: gadm41_PAK_1.shp")
        print("\nOption 2 - geoBoundaries:")
        print("  1. Visit: https://www.geoboundaries.org/")
        print("  2. Search for 'Pakistan' and download ADM1")
        print(f"  3. Extract to: {OUTPUT_DIR}")
        print("\nOption 3 - Natural Earth:")
        print("  1. Visit: https://www.naturalearthdata.com/downloads/")
        print("  2. Download 'Admin 1 - States, Provinces'")
        print("  3. Filter for Pakistan in QGIS or similar tool")
        print(f"  4. Save to: {SHAPEFILE_PATH}")

print("=" * 70)

