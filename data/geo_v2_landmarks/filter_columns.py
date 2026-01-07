import csv
from pathlib import Path

csv_path = Path(__file__).parent / "train_filtered.csv"
temp_path = Path(__file__).parent / "train_filtered_temp.csv"

# Columns to keep
keep_columns = ['id', 'latitude', 'longitude', 'url']
output_fieldnames = ['id', 'latitude', 'long', 'url']

with open(csv_path, 'r', encoding='utf-8') as infile, open(temp_path, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    
    # Check if all columns exist
    fieldnames = reader.fieldnames
    if not all(col in fieldnames for col in keep_columns):
        print(f'Error: Some columns not found. Available: {fieldnames}')
        exit(1)
    
    writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
    writer.writeheader()
    
    for row in reader:
        writer.writerow({
            'id': row['id'],
            'latitude': row['latitude'],
            'long': row['longitude'],  # Rename longitude to long
            'url': row['url']
        })

# Replace original with filtered file
temp_path.replace(csv_path)
print(f'Filtered CSV saved. Kept columns: id, latitude, long, url')

