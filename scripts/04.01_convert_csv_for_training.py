DIDN'T HELP -- BAD!!

from pathlib import Path
import pandas as pd
import json

MAX_ENTRIES_SENT = 10

# setup file paths
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

output_dir = project_root / "data" / f"04_extracted_entries_gemini_2pgs_seeded_986234"
docs_csv = output_dir / "amd_1918_doc_entries_cleaned.csv" 
city_csv = output_dir / "amd_1918_city_entries.csv" 
reponse_json = output_dir / "extracted_entries_responses_cleaned.jsonl"

all_docs = pd.read_csv(docs_csv, encoding="utf-8")
all_city = pd.read_csv(city_csv, encoding="utf-8")
# grouped_inputs = all_inputs.groupby(["publication", "page_number", "column"])

with open(reponse_json, 'w') as response_f:
    for page in range(1,3):
        for col in range(3):
            col_docs = all_docs.loc[(all_docs["page_number"] == page) & (all_docs["column"] == col)]
            col_city = all_city.loc[(all_city["page_number"] == page) & (all_city["column"] == col)]
            all_ids = pd.concat([col_docs["entry_id"], col_city["entry_id"]])
            sorted_ids = all_ids.sort_values()
            

            next_entry = 0
            while next_entry < len(sorted_ids):
                last_entry = min(next_entry + MAX_ENTRIES_SENT, len(sorted_ids))
                curr_ids = sorted_ids.iloc[next_entry:last_entry]
                curr_docs = col_docs.loc[col_docs["entry_id"].isin(curr_ids)]
                curr_city = col_city.loc[col_city["entry_id"].isin(curr_ids)]
                response_f.write(json.dumps(   
                    {
                        "doc_entries": curr_docs.to_json(),
                        "city_entries": curr_city.to_json()
                    }
                )+ '\n')

                next_entry += MAX_ENTRIES_SENT