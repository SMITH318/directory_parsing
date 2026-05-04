"""
Script to sort CSV entry files by entry_id and save with _sorted postfix.
Sorts both amd_1918_city_entries.csv and amd_1918_doc_entries.csv files.
"""

import pandas as pd
from pathlib import Path

# Define the data directory
data_dir = Path(__file__).parent.parent / "data" / "04_extracted_entries_gemini_2026.03.18"

# Define input and output file paths
city_input = data_dir / "amd_1918_city_entries.csv"
doc_input = data_dir / "amd_1918_doc_entries.csv"

city_output = data_dir / "amd_1918_city_entries_sorted.csv"
doc_output = data_dir / "amd_1918_doc_entries_sorted.csv"

def sort_and_save(input_file, output_file, file_name):
    """Sort CSV by entry_id and save to output file. Check for duplicates."""
    print(f"Processing {file_name}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} rows from {input_file.name}")
    
    # Check for duplicate entry_ids
    duplicates = df[df.duplicated(subset=['entry_id'], keep=False)].sort_values('entry_id')
    if len(duplicates) > 0:
        num_duplicate_ids = len(duplicates['entry_id'].unique())
        print(f"  ⚠️  WARNING: Found {len(duplicates)} duplicate rows ({num_duplicate_ids} duplicate entry_ids)")
        print(f"      Details:")
        for dup_id in duplicates['entry_id'].unique():
            count = len(duplicates[duplicates['entry_id'] == dup_id])
            print(f"        - {dup_id}: {count} occurrences")
    else:
        print(f"  ✓ No duplicates found")
    
    # Sort by entry_id
    df_sorted = df.sort_values("entry_id").reset_index(drop=True)
    print(f"  Sorted by entry_id")
    
    # Save to output file
    df_sorted.to_csv(output_file, index=False)
    print(f"  Saved to {output_file.name}")
    print()

# Process both files
sort_and_save(city_input, city_output, "City Entries")
sort_and_save(doc_input, doc_output, "Doc Entries")

print("✓ Done! Both files have been sorted and saved with '_sorted' postfix.")
