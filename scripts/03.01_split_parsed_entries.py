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

combined_df["entry_id"] = combined_df.apply(
    lambda row: 
        f'{row["publication"][:4]}_{row["page_number"]:03d}_{row["column"]:02d}_{row.name:05d}',
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

