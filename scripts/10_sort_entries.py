"""
Step 10: Sort CSV parsed entry files
Script to sort CSV entry files by entry_id and save with _sorted postfix.
Sorts both amd_1918_city_entries.csv and amd_1918_doc_entries.csv files,
checks for duplicate rows and entry_ids.
"""

import pandas as pd
import json
from pathlib import Path
import sys
import argparse
import logging
logging.basicConfig(
  handlers=[
      logging.FileHandler('10_sort_entries.log', mode='w', encoding='utf-8'),
      logging.StreamHandler(sys.stderr)
  ],
  level=logging.WARNING) ## <=================== Change logging level here
logger = logging.getLogger(__name__)

def sort_and_save(input_file, output_file, file_name):
    """Sort CSV by entry_id and save to output file. Check for duplicates."""
    logger.info(f"Processing {file_name}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    logger.info(f"  Loaded {len(df)} rows from {input_file.name}")
    
    # Check for duplicate entry_ids
    duplicates = df[df.duplicated(subset=['entry_id'], keep=False)].sort_values('entry_id')
    if len(duplicates) > 0:
        num_duplicate_ids = len(duplicates['entry_id'].unique())
        logger.warning(f"  ⚠️  WARNING: Found {len(duplicates)} duplicate rows ({num_duplicate_ids} duplicate entry_ids)")
        logger.warning(f"      Details:")
        for dup_id in duplicates['entry_id'].unique():
            count = len(duplicates[duplicates['entry_id'] == dup_id])
            logger.warning(f"        - {dup_id}: {count} occurrences")
    else:
        logger.info(f"  ✓ No duplicates found")
    
    # Sort by entry_id
    df_sorted = df.sort_values("entry_id").reset_index(drop=True)
    logger.info(f"  Sorted by entry_id")
    
    # Save to output file
    df_sorted.to_csv(output_file, index=False)
    print(output_file)

def main(dataset: str, config_path: str):
    # Define the data directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / dataset

    # Load JSON schema config
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"❌ Error: Config file not found at {config_file}")
        return 1
        
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            schema = json.load(f)
    except Exception as e:
        logger.error(f"❌ Error loading JSON config file: {e}")
        return 1

    entities = schema.get('properties', {})
    for entity, entity_info in entities.items():
        # Define input and output file paths
        input_file = data_dir / f"09_{entity.lower()}_parsed.csv"
        output_file = data_dir / f"10_{entity.lower()}_entries_sorted.csv"
        
        # Process the file
        sort_and_save(input_file, output_file, f"{entity} Entries")
    

    logger.info("✓ Done! All files have been sorted and saved with '_sorted' postfix.")
    logger.info("✓ Step completed successfully")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 10: Sort CSV parsed entry files")
    parser.add_argument("dataset", help="Name of the dataset")
    parser.add_argument("--config", help="Path to JSON schema config file", required=True)
    args = parser.parse_args()
    
    exit_code = main(args.dataset, args.config)
    sys.exit(exit_code)
