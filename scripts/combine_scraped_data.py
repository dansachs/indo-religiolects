#!/usr/bin/env python3
"""
Combine scraped data from multiple CSV files into a single dated CSV file.

Combines:
- Scraped_Catholic.csv
- Scraped_Islam.csv
- Scraped_Protestant.csv

Into a single file with timestamp in the name.
"""

import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict

BASE_DIR = Path(__file__).parent.parent.resolve()  # Go up one level from scripts/ to project root
FILE_ENCODING = 'utf-8'

# Input files
INPUT_FILES = {
    'Catholic': BASE_DIR / 'data' / 'scraped' / 'Scraped_Catholic.csv',
    'Islam': BASE_DIR / 'data' / 'scraped' / 'Scraped_Islam.csv',
    'Protestant': BASE_DIR / 'data' / 'scraped' / 'Scraped_Protestant.csv',
}

def combine_csv_files() -> Path:
    """Combine all scraped CSV files into a single dated file."""
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = BASE_DIR / 'data' / 'combined'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"religious_corpus_{timestamp}.csv"
    
    print(f"📊 Combining scraped data files...")
    print(f"   Output: {output_file.name}\n")
    
    total_rows = 0
    rows_by_religion = {}
    
    # Open output file for writing
    with open(output_file, 'w', encoding=FILE_ENCODING, newline='') as outfile:
        writer = None
        
        # Process each input file
        for religion, input_file in INPUT_FILES.items():
            if not input_file.exists():
                print(f"⚠️  Skipping {religion}: {input_file.name} not found")
                continue
            
            print(f"📂 Processing {religion}...")
            rows_count = 0
            
            with open(input_file, 'r', encoding=FILE_ENCODING) as infile:
                reader = csv.DictReader(infile)
                
                # Initialize writer with headers from first file
                if writer is None:
                    # Get fieldnames from first file
                    fieldnames = reader.fieldnames
                    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                    writer.writeheader()
                    print(f"   Headers: {', '.join(fieldnames)}")
                
                # Write all rows
                for row in reader:
                    writer.writerow(row)
                    rows_count += 1
                    total_rows += 1
                
                rows_by_religion[religion] = rows_count
                print(f"   ✅ Added {rows_count:,} rows")
    
    print(f"\n✅ Combined {total_rows:,} total rows into {output_file.name}")
    print(f"\n📊 Breakdown by religion:")
    for religion, count in rows_by_religion.items():
        print(f"   {religion}: {count:,} rows")
    
    return output_file

if __name__ == "__main__":
    try:
        output_file = combine_csv_files()
        print(f"\n✨ Success! Combined file: {output_file.name}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

