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

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Set up file paths
data_dir = project_root / "data" / "03_processed_batch" 
input_file = data_dir / "entries_segmented_2026.03.18_man_cleaned.csv"
docs_file = data_dir / "doc_entries_2026.03.18.csv" 
cities_file = data_dir / "city_entries_2026.03.18.csv"

combined_df = pd.read_csv(input_file, encoding="utf-8")

def pub_to_id(pub:str) -> str:
    return pub.replace("New ", "N").replace("North ", "N").replace("South ", "S").replace("West ", "W")[:4]

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
        print(f"** Unexpected entry type '{row["entryType"]}' Ignoring!! **")

cities_df = pd.DataFrame(cities_list)
docs_df = pd.DataFrame(docs_list)

cities_df.to_csv(cities_file, encoding="utf-8", index=False)
docs_df.to_csv(docs_file, encoding="utf-8", index=False)

