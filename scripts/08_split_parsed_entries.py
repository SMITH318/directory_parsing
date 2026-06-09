"""
Step 8: Split Parsed Entries into City and Doctor CSVs
This script takes the combined parsed entries CSV and splits it into two separate CSV files: 
one for city entries and another for doctor entries, based on their entryType. 
It also adds unique identifiers to each entry for reference in later stages of processing.
Cities following a state entry are assigned that state name, and 
doctor entries following a city entry are assigned that city ID.
"""
import pandas as pd
from pathlib import Path
import sys

import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    handlers=[
        logging.FileHandler('08_split_parsed_entries.log', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stderr)
    ],
    level=logging.WARNING
)

def pub_to_id(pub:str) -> str:
    return pub.replace("New ", "N").replace("North ", "N").replace("South ", "S").replace("West ", "W")[:4]

def main(dataset: str, pdf_dir: str = None, preprocessed_dir: str = None):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Set up file paths
    data_dir = project_root / "data" / dataset 
    input_file = data_dir / "07_entries_segmented_man_cleaned.csv"
    docs_file = data_dir / "08_doc_entries.csv" 
    cities_file = data_dir / "08_city_entries.csv"

    combined_df = pd.read_csv(input_file, encoding="utf-8")

    combined_df["entry_id"] = combined_df.apply(
        lambda row: 
            f'{pub_to_id(row["publication"])}_{row["page_number"]:03d}_{row["column"]:02d}_{row.name:06d}',
        axis = 1
    )

    cities_list = []
    docs_list = []

    currentState = None
    currentCityId = None
    for i, row in combined_df.iterrows():
        if row["entryType"] == "STATE":
            currentState = row["full_text"]
        elif row["entryType"] == "CITY":
            row["state_name"] = currentState
            currentCityId = row["entry_id"]
            cities_list.append(row)
        elif row["entryType"] == "DOC":
            row["city_id"] = currentCityId
            docs_list.append(row)
        else:
            # UNKNOWN or unknown type
            logger.warning(f"** Unexpected entry type '{row['entryType']}' Ignoring!! **")

    cities_df = pd.DataFrame(cities_list)
    docs_df = pd.DataFrame(docs_list)

    cities_df.to_csv(cities_file, encoding="utf-8", index=False)
    docs_df.to_csv(docs_file, encoding="utf-8", index=False)
    
    print(cities_file)
    print(docs_file)
    if len(docs_df) > 0 and len(cities_df) > 0:
        logger.info(f"✓ Step completed successfully ({cities_file}; {docs_file})")
        return 0
    else:
        logger.error("✗ Step failed: not enough entries were split")
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 8: Split Parsed Entries")
    parser.add_argument("dataset", help="Name of the dataset")
    args = parser.parse_args()
    
    exit_code = main(args.dataset)
    sys.exit(exit_code)


