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

sys.stdout.reconfigure(encoding='utf-8')

# expects either both split out docs and cities (with IDs) or classified entries with no IDs
START_SPLIT = True#False
if not START_SPLIT:
    classified_entries_path = Path("data/dataset_2026.03.18/07_entries_segmented_2026.03.18_man_cleaned.csv")
else:
    split_docs_path = Path("data/dataset_2026.03.18/08_doc_entries_2026.03.18.csv")
    split_cities_path = Path("data/dataset_2026.03.18/08_city_entries_2026.03.18.csv")
    if not (split_docs_path.exists() and split_cities_path.exists()):
        print(
            f"\n❌ Error: only one input classified split provided:",
            f"{"docs" if split_cities_path else "cities"} missing"
        )
        sys.exit(1)

parsed_docs_path = Path("data/dataset_2026.03.18/10_amd_1918_doc_entries_sorted_deduped.csv")
parsed_cities_path = Path("data/dataset_2026.03.18/10_amd_1918_city_entries_sorted_deduped.csv")
if not (parsed_docs_path.exists() and parsed_cities_path.exists()):
    print(
        f"\n❌ Error: only one step 9 output provided:",
        f"{"docs" if parsed_cities_path else "cities"} missing"
    )
    sys.exit(1)

try:
    if not START_SPLIT:
        classified_entries = pd.read_csv(classified_entries_path, encoding="utf-8")
    else:
        classified_docs = pd.read_csv(split_docs_path, encoding="utf-8")
        classified_cities = pd.read_csv(split_cities_path, encoding="utf-8")
        classified_entries = pd.concat([classified_docs, classified_cities], ignore_index=True)
except Exception as e:
    print(f"Error loading classified entries input: {str(e)}")
    exit(1)
try:
    parsed_docs = pd.read_csv(parsed_docs_path, encoding="utf-8")
    parsed_cities = pd.read_csv(parsed_cities_path, encoding="utf-8")
except Exception as e:
    print(f"Error loading parsed entries output: {str(e)}")
    exit(1)

  
def check_missing_ids(df:pd.DataFrame, id_col: str) -> List[str]:
    """
    Check for missing IDs (gaps in numeric sequences).
    Only works for numeric IDs or numeric suffixes.
    """
    ids = df[id_col]# [id for id in df[id_col] if id]  # Filter empty IDs
    
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

print("\n" + "=" * 80)
print("SANITY CHECK REPORT - ENTRY PARSING CHECKS")
print("-" * 80 + "\n")

# Verify no repeated ids in parsed entries
all_ids = pd.concat([parsed_docs["entry_id"], parsed_cities["entry_id"]], ignore_index=True)
duplicated_ids = all_ids[all_ids.duplicated()]
if not duplicated_ids.empty:
    print(f"\n❌ Duplicate IDs found in parsed entries:\n{duplicated_ids.to_string()}\n")
else:
    print("✓ No duplicate IDs found in parsed entries")

# Verify no seqentially missing ids in parsed entries
entry_counts = classified_entries["entryType"].value_counts()
missing = check_missing_ids(pd.concat([parsed_docs[['entry_id']], parsed_cities[['entry_id']]]), 'entry_id')
# STATE get IDs but aren't preserved, so expect that numnber to be missing
num_missing_expected = entry_counts.get("STATE", 0) if not START_SPLIT else classified_entries['state_name'].nunique()
if missing and num_missing_expected - 1!= len(missing):  # -1 for starting state with no ID
    print(f"\n❌ Missing {len(missing)} sequential ID numbers (expected {num_missing_expected - 1}): \n\t{missing}\n")
    # print("⚠ Skipping deeper ID verification")
    # return
else:
    print(f"✓ No missing sequential IDs (outside of expected {num_missing_expected - 1})")

# Verify parsed entries have same number of cities and docs as classified entries
if entry_counts.get("DOC", 0) != len(parsed_docs):
    print(
        f"\n❌ Number of DOC entries mismatch:",
        f"classified has {entry_counts.get('DOC', 0)}, parsed has {len(parsed_docs)}"
    )
    doc_in_entries = classified_entries[classified_entries["entryType"] == "DOC"]
    doc_in_counts = doc_in_entries.groupby(['publication', 'page_number', 'column'])['x'].count().reset_index(name='count')
    doc_out_counts = parsed_docs.groupby(['publication', 'page_number', 'column'])['x'].count().reset_index(name='count')
    doc_counts = doc_in_counts.merge(doc_out_counts, on = ['publication', 'page_number', 'column'], suffixes=['_in', '_out'], validate='1:1')
    doc_off_counts = doc_counts[doc_counts["count_in"] != doc_counts["count_out"]]
    print(doc_off_counts)
else:
    print("✓ Number of DOC entries matches")

if entry_counts.get("CITY", 0) != len(parsed_cities):
    print(
        f"\n❌ Number of CITY entries mismatch:",
        f"classified has {entry_counts.get('CITY', 0)}, parsed has {len(parsed_cities)}"
    )
else:
    print("✓ Number of CITY entries matches")

# Verify all classified entry IDs are in parsed entries
if 'entry_id' in classified_entries.columns:
    classified_ids = set(classified_entries['entry_id'])
    parsed_doc_ids = set(parsed_docs['entry_id'])
    parsed_city_ids = set(parsed_cities['entry_id'])

    missing_ids = classified_ids - parsed_doc_ids - parsed_city_ids

    if missing_ids:
        print(f"\n❌ Classify entry IDs ({len(missing_ids)}) not found in parsed docs:\n{missing_ids.to_string()}\n")
    else:
        print("✓ All classified doc entry IDs are in parsed entries")
else:
    print("⚠ 'entry_id' column not found in classified entries, skipping ID consistency check with parsed entries")

# Verify all city entries referenced by a doc entry
city_with_doc = parsed_cities.merge(
    parsed_docs[["entry_id", "city_id"]], 
    left_on="entry_id", 
    right_on="city_id", 
    how="left", 
    # validate="1:m"
)
missing_doc_refs = city_with_doc[city_with_doc["entry_id_y"].isna()]
if not missing_doc_refs.empty:
    missing_doc_refs = missing_doc_refs.rename(columns={"entry_id_x": "city_entry_id"})
    print(
        f"\n⚠ City entries ({len(missing_doc_refs)}) with no doc references (this is OK if city a 'See' or small):",
        f"\n{missing_doc_refs[['city_entry_id']].to_string()}\n"
    )
else:
    print("✓ All city entries have valid doc references")

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
    print(
        f"\n❌ Doc entries ({len(missing_city_refs)}) with non-existent city references:",
        f"\n{missing_city_refs[['doc_entry_id', 'city_id']].to_string()}\n"
    )
else:
    print("✓ All doc entries reference valid city entries")        
