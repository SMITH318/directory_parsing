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


def get_definition(schema: dict, entity_name: str) -> dict:
    entity_info = schema.get('properties', {}).get(entity_name, {})
    items = entity_info.get('items', {})
    if isinstance(items, dict) and '$ref' in items:
        def_name = items['$ref'].split('/')[-1]
        return schema.get('definitions', {}).get(def_name, {})
    return items or {}


def get_inheritance_rules(schema: dict) -> dict[str, list[tuple[str, str]]]:
    rules = {}
    for entity_name in schema.get('properties', {}):
        definition = get_definition(schema, entity_name)
        for prop_name, prop_info in definition.get('properties', {}).items():
            if isinstance(prop_info, dict) and 'x-inherits-id-from' in prop_info:
                parent_entity = prop_info['x-inherits-id-from']
                rules.setdefault(entity_name, []).append((parent_entity, prop_name))
    return rules


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
            elif t == 'object':
                if not 'properties' in prop_info:
                    break
                n_props = len(prop_info['properties'])
                if (
                    isinstance(val, dict) 
                    or isinstance(val, list) 
                    or (isinstance(val, str) and val.startswith('{') and val.endswith('}'))
                    or (isinstance(val, str) and val.startswith('[') and val.endswith(']'))
                    or (isinstance(val, str) and val.count(',') >= n_props - 1)
                ):
                    type_matched = True
                    break
            elif t == 'array':
                if (
                    isinstance(val, list) 
                    or (isinstance(val, str) and val.startswith('[') and val.endswith(']'))
                    or (isinstance(val, str) and val.count(',') >= 1)
                ):
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

    classified_entries_path = data_dir / "07_entries_segmented_man_cleaned.csv"
    schema_entities = list(schema.get('properties', {}).keys())
    if not schema_entities:
        logger.error("❌ Error: No entities found in schema properties")
        sys.exit(1)

    split_paths = {
        entity: data_dir / f"08_{entity.lower()}_entries.csv"
        for entity in schema_entities
    }
    parsed_paths = {
        entity: data_dir / f"09_{entity.lower()}_parsed.csv"
        for entity in schema_entities
    }

    if START_SPLIT:
        missing_splits = [entity for entity, path in split_paths.items() if not path.exists()]
        if missing_splits:
            logger.error(
                f"❌ Error: missing classified split input files for schema entities: {missing_splits}"
            )
            sys.exit(1)

    missing_parsed_entities = [entity for entity, path in parsed_paths.items() if not path.exists()]
    if missing_parsed_entities:
        logger.error(
            f"❌ Error: missing parsed output files for schema entities: {missing_parsed_entities}"
        )
        sys.exit(1)

    try:
        if not START_SPLIT:
            classified_entries = pd.read_csv(classified_entries_path, encoding="utf-8")
            if 'entryType' not in classified_entries.columns:
                raise ValueError(f"Classified entries input is missing required 'entryType' column")
        else:
            split_dfs = []
            for entity, path in split_paths.items():
                df = pd.read_csv(path, encoding="utf-8")
                if 'entry_id' not in df.columns:
                    raise ValueError(f"Classified split input for entity '{entity}' is missing required 'entry_id' column")
                if 'entryType' not in df.columns:
                    raise ValueError(f"Classified split input for entity '{entity}' is missing required 'entryType' column")
                split_dfs.append(df)
            classified_entries = pd.concat(split_dfs, ignore_index=True)
        classified_entries['entryType'] = classified_entries['entryType'].str.lower()
    except Exception as e:
        logger.error(f"Error loading classified entries input: {str(e)}")
        sys.exit(1)

    parsed_dfs = {}
    try:
        for entity, path in parsed_paths.items():
            parsed_dfs[entity] = pd.read_csv(path, encoding="utf-8")
            if 'entry_id' not in parsed_dfs[entity].columns:
                raise ValueError(f"Parsed output for entity '{entity}' is missing required 'entry_id' column")
    except Exception as e:
        logger.error(f"Error loading parsed entries output: {str(e)}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("SANITY CHECK REPORT - ENTRY PARSING CHECKS")
    logger.info("-" * 80 + "\n")

    # Verify no repeated ids in parsed entries
    parsed_id_series = []
    for entity, df in parsed_dfs.items():
        parsed_id_series.append(df['entry_id'])
    all_ids = pd.concat(parsed_id_series, ignore_index=True)
    duplicated_ids = all_ids[all_ids.duplicated()]
    if not duplicated_ids.empty:
        logger.error(f"❌ Duplicate IDs found in parsed entries:\n{duplicated_ids.to_string()}")
        any_errors = True
    else:
        logger.info("✓ No duplicate IDs found in parsed entries")

    # Verify no sequentially missing ids in parsed entries
    entry_counts = classified_entries['entryType'].value_counts()
    missing = check_missing_ids(all_ids)
    num_missing_expected = entry_counts.get("unknown", 0) if not START_SPLIT else 0
    if len(missing) != num_missing_expected:
        logger.error(f"❌ Missing {len(missing)} sequential ID numbers (expected {num_missing_expected}): \n\t{missing}")
        any_errors = True
    else:
        logger.info(f"✓ No missing sequential IDs (outside of expected {num_missing_expected})")

    # Verify parsed entity counts against classified entries
    for entity, df in parsed_dfs.items():
        expected_count = int(entry_counts.get(entity, 0))
        parsed_count = len(df)
        if expected_count != parsed_count:
            logger.error(
                f"❌ Number of {entity.upper()} entries mismatch: classified has {expected_count}, parsed has {parsed_count}:"
            )
            any_errors = True
            # report which pages/columns counts are off
            in_entries = classified_entries[classified_entries["entryType"] == entity]
            in_counts = in_entries.groupby(['publication', 'page_number', 'column'])['x'].count().reset_index(name='count')
            out_counts = df.groupby(['publication', 'page_number', 'column'])[df.columns[0]].count().reset_index(name='count')
            joined_counts = in_counts.merge(out_counts, on = ['publication', 'page_number', 'column'], suffixes=['_in', '_out'], validate='1:1')
            off_counts = joined_counts[joined_counts["count_in"] != joined_counts["count_out"]]
            logger.error("\n" + off_counts.to_string())
        else:
            logger.info(f"✓ Number of {entity.upper()} entries matches")


    # Verify all classified entry IDs are in parsed outputs
    if 'entry_id' in classified_entries.columns:
        classified_ids = set(classified_entries['entry_id'])
        parsed_ids_union = set(all_ids)
        missing_ids = classified_ids - parsed_ids_union
        if missing_ids:
            logger.error(f"❌ Classified entry IDs ({len(missing_ids)}) not found in parsed outputs:\n{sorted(missing_ids)}")
            any_errors = True
        else:
            logger.info("✓ All classified entry IDs appear in parsed outputs")
    else:
        logger.warning("⚠ 'entry_id' column not found in classified entries, skipping ID consistency check with parsed outputs")
        any_warnings = True

    # Verify parent references based on schema inheritance rules
    inheritance_rules = get_inheritance_rules(schema)
    for child_entity, rules in inheritance_rules.items():
        child_df = parsed_dfs.get(child_entity)
        if child_df is None:
            logger.error(f"❌ Child entity '{child_entity}' not present in parsed outputs for relationship validation")
            any_errors = True
            continue

        for parent_entity, child_field in rules:
            parent_df = parsed_dfs.get(parent_entity)
            if parent_df is None:
                logger.error(f"❌ Parent entity '{parent_entity}' not present in parsed outputs for relationship validation")
                any_errors = True
                continue
            if child_field not in child_df.columns:
                logger.error(f"❌ Child field '{child_field}' not found in '{child_entity}' output; skipping reference validation")
                any_errors = True
                continue
            parent_ids = set(parent_df['entry_id'].dropna().astype(str))

            # Verify all parent entries are referenced by at least one child entry
            unreferenced_parents = parent_ids - set(child_df[child_field].dropna().astype(str))
            if unreferenced_parents:
                sample = list(unreferenced_parents)[:20]
                logger.warning(
                    f"⚠ {len(unreferenced_parents)} '{parent_entity}' with no '{child_entity}' references in field '{child_field}':\n{sample}"
                )
                any_warnings = True
            else:
                logger.info(f"✓ All '{parent_entity}' entries are referenced by at least one '{child_entity}' in field '{child_field}'")

            # Verify all child entries reference a valid parent entry
            # child_field values that are not null/empty and not in parent_ids are invalid references
            invalid_refs = child_df[
                child_df[child_field].notna() &
                child_df[child_field].astype(str).str.strip().ne("") &
                ~child_df[child_field].astype(str).isin(parent_ids)
            ]
            if not invalid_refs.empty:
                sample = invalid_refs[[child_field]].head(20).to_string(index=False)
                logger.error(
                    f"❌ {len(invalid_refs)} '{child_entity}.{child_field}' with non-existent '{parent_entity}' references:\n{sample}"
                )
                any_errors = True
            else:
                logger.info(f"✓ All '{child_entity}' values in '{child_field}' reference valid '{parent_entity}' entry IDs")

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
            logger.error(f"❌ Error: No definition found for entity '{entity}' in schema.")
            any_errors = True
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
    for entity, path in parsed_paths.items():
        if path.exists():
            print(path)
        
    if any_errors or any_warnings:
        if any_warnings:
            logger.warning("⚠ Sanity check completed with warnings.")
        if any_errors:
            logger.error("❌ Sanity check errored.")
        return 1
    logger.info("✓ Sanity check completed successfully.")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 9: Sanity Check Parsed Entries")
    parser.add_argument("dataset", help="Name of the dataset")
    parser.add_argument("--config", help="Path to JSON schema config file", required=True)
    args = parser.parse_args()
    
    sys.exit(main(args.dataset, args.config))
