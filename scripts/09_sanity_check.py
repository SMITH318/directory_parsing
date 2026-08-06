"""
Post-line parsing (step 9) sanity checker
Analyze consistency between classified entries and parsed entries. 
Check that:
 - All classified entries are represented in parsed entries
 - No repeated ids in any parsed entries
 - No sequentially missing ids in parsed entries
 - All parsed cities and docs are consistent with each other and with classified entries
    - Parsed entries have same number of cities and docs as classified entries
    - All city entries referenced by a doc entry
    - All doc entries reference a valid city entry
"""

import pandas as pd
from pathlib import Path
from typing import List
import sys
import argparse
import logging
import json

sys.stdout.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)
logging.basicConfig(
  handlers=[
      logging.FileHandler('09_sanity_check.log', mode='w', encoding='utf-8'),
      logging.StreamHandler(sys.stderr)
  ],
  level=logging.INFO) ## <=================== Change logging level here

# expects either both split out docs and cities (with IDs) or classified entries with no IDs
START_SPLIT = True#False

def check_missing_ids(ids:pd.Series) -> List[str]:
    """
    Check for missing IDs (gaps in numeric sequences).
    Only works for numeric IDs or numeric suffixes.
    """
    
    missing = []
    
    # Try to extract numeric parts
    numeric_ids = []
    for id_str in ids:
        # Look for trailing numbers
        i = len(id_str) - 1
        while i >= 0 and id_str[i].isdigit():
            i -= 1
        if i < len(id_str) - 1:
            try:
                numeric_ids.append((id_str, int(id_str[i+1:])))
            except ValueError:
                pass
    
    if numeric_ids:
        # Sort by numeric part and check for gaps
        numeric_ids.sort(key=lambda x: x[1])
        
        min_num = numeric_ids[0][1]
        max_num = numeric_ids[-1][1]
        existing_nums = set(num for _, num in numeric_ids)
        
        for num in range(min_num, max_num + 1):
            if num not in existing_nums:
                missing.append(f"{num}")
    
    return missing

def validate_column_values(df: pd.DataFrame, col_name: str, prop_info: dict, is_required: bool) -> List[str]:
    errors = []
    
    # Determine allowed types
    allowed_types = []
    if 'type' in prop_info:
        t = prop_info['type']
        if isinstance(t, list):
            allowed_types.extend(t)
        else:
            allowed_types.append(t)
            
    null_allowed = ('null' in allowed_types) or (not is_required)
    type_set = set(allowed_types) - {'null'}
    enum_list = prop_info.get('enum', None)
    
    for idx, val in enumerate(df[col_name]):
        # Check for null/empty
        is_val_null = pd.isna(val) or val == ""
        if isinstance(val, str) and val.strip() == "":
            is_val_null = True
            
        if is_val_null:
            if not null_allowed:
                errors.append(f"Row {idx+1}: Value is missing/empty, but field is required.")
            continue
            
        # Check enum
        if enum_list is not None:
            val_str = str(val)
            # Standardize boolean strings if pandas converted them
            if isinstance(val, bool):
                val_str = "True" if val else "False"
            if val_str not in enum_list and val not in enum_list:
                errors.append(f"Row {idx+1}: Value '{val}' is not in allowed enum list {enum_list}.")
                continue
                
        # Check type
        type_matched = False
        if not type_set:
            # If no type constraint specified, treat as matched
            type_matched = True
            
        for t in type_set:
            if t == 'integer':
                try:
                    if isinstance(val, (int, float)):
                        if float(val).is_integer():
                            type_matched = True
                            break
                except ValueError:
                    pass
            elif t == 'string':
                type_matched = True
                break
            elif t == 'number':
                try:
                    float(val)
                    type_matched = True
                    break
                except ValueError:
                    pass
            elif t == 'boolean':
                if isinstance(val, bool) or str(val).lower() in ['true', 'false', '0', '1']:
                    type_matched = True
                    break
                    
        if not type_matched:
            errors.append(f"Row {idx+1}: Value '{val}' does not match any of the expected types: {list(type_set)}")
            
    return errors

def main(dataset: str, config_path: Path):
    any_warnings = False
    any_errors = False
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / dataset

    # Load JSON schema config
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"❌ Error: Config file not found at {config_path}")
        sys.exit(1)
        
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            schema = json.load(f)
    except Exception as e:
        logger.error(f"❌ Error loading JSON config file: {e}")
        sys.exit(1)

    if not START_SPLIT:
        classified_entries_path = data_dir / "07_entries_segmented_man_cleaned.csv"
    else:
        split_docs_path = data_dir / "08_doc_entries.csv"
        split_cities_path = data_dir / "08_city_entries.csv"
        if not (split_docs_path.exists() and split_cities_path.exists()):
            logger.error(
                f"❌ Error: only one input classified split provided: "
                f"{'docs' if split_cities_path else 'cities'} missing"
            )
            sys.exit(1)

    parsed_docs_path = data_dir / "09_doc_parsed.csv"
    parsed_cities_path = data_dir / "09_city_parsed.csv"
    parsed_states_path = data_dir / "09_state_parsed.csv"
    if not parsed_docs_path.exists():
        logger.error("❌ Error: step 9 output docs missing")
        sys.exit(1)
    if not parsed_cities_path.exists():
        logger.error("❌ Error: step 9 output cities missing")
        sys.exit(1)
    if not parsed_states_path.exists():
        logger.error("❌ Error: step 9 output states missing")
        sys.exit(1)

    try:
        if not START_SPLIT:
            classified_entries = pd.read_csv(classified_entries_path, encoding="utf-8")
        else:
            classified_docs = pd.read_csv(split_docs_path, encoding="utf-8")
            classified_cities = pd.read_csv(split_cities_path, encoding="utf-8")
            classified_entries = pd.concat([classified_docs, classified_cities], ignore_index=True)
    except Exception as e:
        logger.error(f"Error loading classified entries input: {str(e)}")
        sys.exit(1)
    try:
        parsed_docs = pd.read_csv(parsed_docs_path, encoding="utf-8")
        parsed_cities = pd.read_csv(parsed_cities_path, encoding="utf-8")
        parsed_states = pd.read_csv(parsed_states_path, encoding="utf-8")
    except Exception as e:
        logger.error(f"Error loading parsed entries output: {str(e)}")
        sys.exit(1)

    logger.info("\n" + "=" * 80)
    logger.info("SANITY CHECK REPORT - ENTRY PARSING CHECKS")
    logger.info("-" * 80 + "\n")

    # Verify no repeated ids in parsed entries
    all_ids = pd.concat([parsed_docs["entry_id"], parsed_cities["entry_id"], parsed_states["entry_id"]], ignore_index=True)
    duplicated_ids = all_ids[all_ids.duplicated()]
    if not duplicated_ids.empty:
        logger.error(f"❌ Duplicate IDs found in parsed entries:\n{duplicated_ids.to_string()}")
        any_errors = True
    else:
        logger.info("✓ No duplicate IDs found in parsed entries")

    # Verify no seqentially missing ids in parsed entries
    entry_counts = classified_entries["entryType"].value_counts()
    missing = check_missing_ids(all_ids)
    # UNKNOWNs get IDs but aren't preserved, so expect that number to be missing (when we have it)
    num_missing_expected = entry_counts.get("UNKNOWN", 0) if not START_SPLIT else 0
    if len(missing) != num_missing_expected:
        logger.error(f"❌ Missing {len(missing)} sequential ID numbers (expected {num_missing_expected}): \n\t{missing}")
        any_errors = True
    else:
        logger.info(f"✓ No missing sequential IDs (outside of expected {num_missing_expected})")

    # Verify parsed entries have same number of cities and docs as classified entries
    if entry_counts.get("doc", 0) != len(parsed_docs):
        logger.error(
            f"❌ Number of DOC entries mismatch: "
            f"classified has {entry_counts.get('doc', 0)}, parsed has {len(parsed_docs)}"
        )
        any_errors = True
        doc_in_entries = classified_entries[classified_entries["entryType"].str.upper() == "DOC"]
        doc_in_counts = doc_in_entries.groupby(['publication', 'page_number', 'column'])['x'].count().reset_index(name='count')
        doc_out_counts = parsed_docs.groupby(['publication', 'page_number', 'column'])['x'].count().reset_index(name='count')
        doc_counts = doc_in_counts.merge(doc_out_counts, on = ['publication', 'page_number', 'column'], suffixes=['_in', '_out'], validate='1:1')
        doc_off_counts = doc_counts[doc_counts["count_in"] != doc_counts["count_out"]]
        logger.error("\n" + doc_off_counts.to_string())
    else:
        logger.info("✓ Number of DOC entries matches")

    if entry_counts.get("city", 0) != len(parsed_cities):
        logger.error(
            f"❌ Number of CITY entries mismatch: "
            f"classified has {entry_counts.get('city', 0)}, parsed has {len(parsed_cities)}"
        )
        any_errors = True
    else:
        logger.info("✓ Number of CITY entries matches")

    # Verify all classified entry IDs are in parsed entries
    if 'entry_id' in classified_entries.columns:
        classified_ids = set(classified_entries['entry_id'])
        parsed_doc_ids = set(parsed_docs['entry_id'])
        parsed_city_ids = set(parsed_cities['entry_id'])

        missing_ids = classified_ids - parsed_doc_ids - parsed_city_ids

        if missing_ids:
            logger.error(f"❌ Classify entry IDs ({len(missing_ids)}) not found in parsed docs:\n{missing_ids}")
            any_errors = True
        else:
            logger.info("✓ All classified doc entry IDs are in parsed entries")
    else:
        logger.warning("⚠ 'entry_id' column not found in classified entries, skipping ID consistency check with parsed entries")
        any_warnings = True

    # Verify all city entries referenced by a doc entry
    city_with_doc = parsed_cities.merge(
        parsed_docs[["entry_id", "city_id"]], 
        left_on="entry_id", 
        right_on="city_id", 
        how="left", 
        # validate="1:m"
    )
    missing_doc_refs = city_with_doc[city_with_doc["entry_id_y"].isna() & (city_with_doc["post_reference_type"].str.upper() != "SEE")]
    if not missing_doc_refs.empty:
        missing_doc_refs = missing_doc_refs.rename(columns={"entry_id_x": "city_entry_id"})
        logger.warning(
            f"⚠ City entries ({len(missing_doc_refs)}) with no doc references (this is OK if city a 'See' or small): "
            f"\n{missing_doc_refs[['city_entry_id']].to_string()}\n"
        )
        any_warnings = True
    else:
        logger.info("✓ All city entries have valid doc references")

    # Verify all doc entries reference a valid city entry
    doc_with_city = parsed_docs.merge(
        parsed_cities[["entry_id"]], 
        left_on="city_id", 
        right_on="entry_id", 
        how="left", 
        # validate="m:1"
    )
    missing_city_refs = doc_with_city[doc_with_city["entry_id_y"].isna()]
    if not missing_city_refs.empty:
        missing_city_refs = missing_city_refs.rename(columns={"entry_id_x": "doc_entry_id"})
        logger.error(
            f"❌ Doc entries ({len(missing_city_refs)}) with non-existent city references: "
            f"\n{missing_city_refs[['doc_entry_id', 'city_id']].to_string()}"
        )
        any_errors = True
    else:
        logger.info("✓ All doc entries reference valid city entries")

    # Verify outputs match JSON config schema properties, enums, required fields, and types
    logger.info("=" * 80)
    logger.info("JSON SCHEMA VALIDATION CHECKS")
    logger.info("-" * 80)
    
    entities = schema.get('properties', {})
    for entity, entity_info in entities.items():
        items_ref = entity_info.get('items', {}).get('$ref', '')
        if items_ref:
            def_name = items_ref.split('/')[-1]
            properties_schema = schema.get('definitions', {}).get(def_name, {})
        else:
            properties_schema = entity_info.get('items', {})
            
        if not properties_schema:
            logger.warning(f"⚠ Warning: No definition found for entity '{entity}' in schema.")
            any_warnings = True
            continue
            
        parsed_file_path = data_dir / f"09_{entity.lower()}_parsed.csv"
        split_file_path = data_dir / f"08_{entity.lower()}_entries.csv"
        
        # Check if output is expected
        if not split_file_path.exists():
            continue
            
        if not parsed_file_path.exists():
            logger.error(f"❌ Error: Expected parsed output file {parsed_file_path.name} not found.")
            any_errors = True
            continue
            
        try:
            df_to_check = pd.read_csv(parsed_file_path, encoding="utf-8")
        except Exception as e:
            logger.error(f"❌ Error loading parsed file {parsed_file_path.name}: {e}")
            any_errors = True
            continue
            
        logger.info(f"***Validating {parsed_file_path.name} against schema properties...***")
        
        # Check columns
        expected_cols = set(properties_schema.get('properties', {}).keys())
        actual_cols = set(df_to_check.columns)
        
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols
        
        if missing_cols:
            logger.error(f"❌ Error: {parsed_file_path.name} is missing schema properties: {sorted(list(missing_cols))}")
            any_errors = True
        if extra_cols:
            logger.error(f"❌ Error: {parsed_file_path.name} contains extra columns not in schema: {sorted(list(extra_cols))}")
            any_errors = True
            
        # Check row value types, required fields, and enums
        required_cols = properties_schema.get('required', [])
        for col_name, prop_info in properties_schema.get('properties', {}).items():
            if col_name not in df_to_check.columns:
                continue
                
            is_required = col_name in required_cols
            col_errors = validate_column_values(df_to_check, col_name, prop_info, is_required)
            if col_errors:
                logger.error(f"❌ Error: Schema validation failed for {parsed_file_path.name} column '{col_name}':")
                for err in col_errors[:10]:
                    logger.error(f"  - {err}")
                if len(col_errors) > 10:
                    logger.error(f"  - ... and {len(col_errors) - 10} more errors.")
                any_errors = True
            else:
                logger.info(f"✓ Column '{col_name}' successfully validated.")

    # Print reviewed paths to stdout upon successful sanity check
    print(parsed_docs_path)
    print(parsed_cities_path)
    parsed_state_path = data_dir / "09_state_parsed.csv"
    if parsed_state_path.exists():
        print(parsed_state_path)
        
    if any_errors or any_warnings:
        return 1
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 9: Sanity Check Parsed Entries")
    parser.add_argument("dataset", help="Name of the dataset")
    parser.add_argument("--config", help="Path to JSON schema config file", required=True)
    args = parser.parse_args()
    
    sys.exit(main(args.dataset, args.config))
