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
import json
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

def get_inheritance_rules(config: Path):
    if not config:
        return {}
    with open(config, 'r', encoding='utf-8') as f:
        schema_data = json.load(f)
    
    rules = {} # {child_type: [(parent_type, child_field)]}
    definitions = schema_data.get('definitions', {})
    
    properties = schema_data.get('properties', {})
    for entity_name, prop_info in properties.items():
        items_ref = prop_info.get('items', {}).get('$ref', '')
        if items_ref:
            def_name = items_ref.split('/')[-1]
            definition = definitions.get(def_name, {})
        else:
            definition = prop_info.get('items', {})
            
        entity_rules = []
        for prop_name, prop_details in definition.get('properties', {}).items():
            if 'x-inherits-id-from' in prop_details:
                parent_type = prop_details['x-inherits-id-from']
                entity_rules.append((parent_type, prop_name))
        
        if entity_rules:
            rules[entity_name] = entity_rules
            
    return rules

def main(dataset: str, config: Path):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Set up file paths
    data_dir = project_root / "data" / dataset 
    input_file = data_dir / "07_entries_segmented_man_cleaned.csv"
    completion_file = data_dir / "08_split_complete.txt"
    completion_file.unlink(missing_ok=True)

    combined_df = pd.read_csv(input_file, encoding="utf-8")

    combined_df["entry_id"] = combined_df.apply(
        lambda row: 
            f'{pub_to_id(row["publication"])}_{row["page_number"]:03d}_{row["column"]:02d}_{row.name:06d}',
        axis = 1
    )
    combined_df.to_csv(data_dir / "08_combined_entries_IDed.csv", encoding="utf-8", index=False)

    # get inheritance rules from schema if config is provided
    inheritance_rules = get_inheritance_rules(config)

    # dynamically group by entryType; keep track of previous ids by type
    output_dfs = {entry_type: [] for entry_type in combined_df["entryType"].unique() if entry_type != "UNKNOWN"}
    last_seen_id = {} # {entryType: entry_id}

    for i, row in combined_df.iterrows():
        entry_type = row["entryType"]
        if entry_type in output_dfs:
            # track this entry's id
            last_seen_id[entry_type] = row["entry_id"]
            
            # apply inheritance rules
            if entry_type in inheritance_rules:
                for parent_type, child_field in inheritance_rules[entry_type]:
                    row[child_field] = last_seen_id.get(parent_type)
                    
            output_dfs[entry_type].append(row)
        else:
            # UNKNOWN or unknown type
            logger.warning(f"** Unexpected entry type '{entry_type}' Ignoring!! **")

    # save each kind of entry into its own CSV
    output_files = []
    for entry_type, lst in output_dfs.items():
        if lst:
            df = pd.DataFrame(lst)
            file_name = data_dir / f"08_{entry_type.lower()}_entries.csv"
            df.to_csv(file_name, encoding="utf-8", index=False)
            output_files.append(file_name)
            print(file_name)
            
    # create a completion marker file for the pipeline
    with open(completion_file, "w") as f:
        f.write("complete\n")

    if len(output_files) > 0:
        logger.info(f"✓ Step completed successfully ({len(output_files)} files created)")
        return 0
    else:
        logger.error("✗ Step failed: not enough entries were split")
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 8: Split Parsed Entries according to JSON schema")
    parser.add_argument("dataset", help="Name of the dataset")
    parser.add_argument("--config", help="Path to JSON schema config", required=True)
    args = parser.parse_args()
    
    exit_code = main(args.dataset, args.config)
    sys.exit(exit_code)


