# Pipeline: Add Province Column

This pipeline adds a `province` column to the merged training data CSV using offline reverse geocoding.

## Setup

1. **Install dependencies:**
   ```bash
   pip install geopandas shapely pandas
   ```

2. **Download Pakistan provinces shapefile:**
   
   **Option A - Automatic download (may fail):**
   ```bash
   cd pipeline
   python download_shapefile.py
   ```
   
   **Option B - Manual download (Recommended):**
   
   **GADM (Best quality):**
   1. Visit: https://gadm.org/download_country.html
   2. Select "Pakistan" → "Shapefile" → "Level 1" (provinces)
   3. Download and extract to `pipeline/` folder
   4. Rename the .shp file to `gadm41_PAK_1.shp`
   
   **Alternative - geoBoundaries:**
   1. Visit: https://www.geoboundaries.org/
   2. Search "Pakistan" → Download "ADM1" (Administrative Level 1)
   3. Extract to `pipeline/` folder
   4. Rename to `gadm41_PAK_1.shp`

## Usage

Run the script to add province column:

```bash
cd pipeline
python add_province.py
```

This will:
- Read `../merged_training_data.csv`
- Match coordinates to Pakistan provinces using spatial join
- Create `../merged_training_data_with_province.csv` with added `province` column

## Output

The output CSV will have columns:
- `id`
- `latitude`
- `longitude`
- `path`
- `province` (new column)

## Performance

- Processes ~128k rows in **2-5 minutes**
- No API calls or rate limits
- Works completely offline

## Notes

- Coordinates outside Pakistan boundaries will have `NaN` for province
- Uses GADM administrative level 1 (provinces) data
- Provinces include: Balochistan, Khyber Pakhtunkhwa, Punjab, Sindh, etc.

