"""
Post-line grouping and classification (step 5) sanity checker
Checks for consistency between raw OCR text lines and grouped and classified entries via:
  - Entries exist for all pages, rows, and columns
  - All text has been preserved in order
  - Bounding boxes match grouped lines - NOT IMPLEMENTED
  - Each document has only one, starting state
  - The relative proportion of doc to city entries matches across publications
  - There are only a few, correct UNKNOWNs.
"""

import pandas as pd
from pathlib import Path
import sys
import difflib

sys.stdout.reconfigure(encoding='utf-8')

ocr_path = Path("data/dataset_2026.03.18/04_ocr_output_reviewed_2026.03.18.jsonl")
classified_entries_path = Path("data/dataset_2026.03.18/05_entries_segmented_2026.03.18.csv")
OCR_COLS = ["pub", "page", "col"]
CLASSIFIED_COLS = ["publication", "page_number", "column"]

def compare_texts(text1, text2):
    """Compare two texts and print out the differences with context."""
    matcher = difflib.SequenceMatcher(None, text1, text2)

    for change_type, i1, i2, j1, j2 in matcher.get_opcodes():
        global TEXT_CHANGES
        TEXT_CHANGES += 1
        if change_type == 'equal':
            continue

        affected_text_in = text1[i1:i2]
        affected_text_out = text2[j1:j2]

        # Get 10 characters of surrounding context
        context_before_in = text1[max(0, i1 - 10):i1]
        context_after_in = text1[i2:min(len(text1), i2 + 10)]
        context_before_out = text2[max(0, j1 - 10):j1]
        context_after_out = text2[j2:min(len(text2), j2 + 10)]

        # Calculate max length for padding
        len_in = len(affected_text_in)
        len_out = len(affected_text_out)
        max_len = max(len_in, len_out)

        # Pad the strings for consistent display
        if change_type != "delete":
            affected_text_in = f"{affected_text_in:<{max_len}}"
            affected_text_out = f"{affected_text_out:<{max_len}}"

        print(f"    Change Type: {change_type}")
        # Highlight changes using brackets and padded text
        print(f"    text_in:  '{context_before_in}{{{{{affected_text_in}}}}}{context_after_in}'")
        print(f"    text_out: '{context_before_out}{{{{{affected_text_out}}}}}{context_after_out}'")
        print("--------------------------------------------------")
 
 #Load OCR output JSONL file
try:
    ocr_data = pd.read_json(ocr_path, lines=True)
except Exception as e:
    print(f"Error loading OCR output {ocr_path.name}: {str(e)}")
    exit(1)

# Load classified entries CSV file
try:
    classified_entries = pd.read_csv(classified_entries_path, encoding="utf-8")
except Exception as e:
    print(f"Error loading classified entries output {classified_entries_path.name}: {str(e)}")
    exit(1)

print("\n" + "=" * 80)
print("SANITY CHECK REPORT - LINE GROUPING & CLASSIFICATION CONSISTENCY CHECKS")
print("-" * 80 + "\n")
any_errors = False
any_warnings = False

# Check entries exist for all pages, rows, and columns
ocr_cols = ocr_data[OCR_COLS].drop_duplicates()
classified_cols = classified_entries[CLASSIFIED_COLS].drop_duplicates()
if len(ocr_cols) != len(classified_cols):
    print(
        f"\n❌  Inconsistent publication, column counts:",
        f"OCR has {len(ocr_cols)}, classified has {len(classified_cols)}"
    )
    any_errors = True

unmatched_ocr_cols = ocr_cols.merge(
    classified_cols, 
    left_on=OCR_COLS, 
    right_on=CLASSIFIED_COLS, 
    how="outer", 
    indicator=True
).query("_merge != 'both'")
if not unmatched_ocr_cols.empty:
    unmatched_ocr_cols.loc[unmatched_ocr_cols['_merge'] == "left_only", 'reason'] = "missing from classified"
    unmatched_ocr_cols.loc[unmatched_ocr_cols['_merge'] == "right_only", 'reason'] = "added in classified"
    print(f"\n❌  OCR columns with no classified entries:\n{unmatched_ocr_cols.drop('_merge', axis=1)}\n")
    any_errors = True
else:
    print("✓ All OCR columns have classified entries")

# Check all text has been preserved in order (after removing whitespace and hyphens to deal with lines being collapsed)
ocr_data = ocr_data.copy()
ocr_data['text'] = ocr_data['text'].str.replace(r"[\s-]", "", regex=True).str.strip()
ocr_cols_text = ocr_data.groupby(OCR_COLS)['text'].agg("".join).reset_index()
classified_entries = classified_entries.copy()
classified_entries['full_text'] = classified_entries['full_text'].str.replace(r"[\s-]", "", regex=True).str.strip()
classified_cols_text = classified_entries.groupby(CLASSIFIED_COLS)['full_text'].agg("".join).reset_index()
matched_cols_text = ocr_cols_text.merge(
    classified_cols_text, 
    left_on=OCR_COLS, 
    right_on=CLASSIFIED_COLS, 
    how="inner"
).rename(columns={"text": "ocr_text", "full_text": "classified_text"})
TEXT_CHANGES = 0 # incremented in the compare_texts function for each change detected
matched_cols_text.apply(
    lambda row: (
        print(
            f"⚠ warning: text changed in {row['pub']}.{row['page']}.{row['col']}!",
            f"(length in {len(row['ocr_text'])} vs. out {len(row['classified_text'])})"
        ),
        compare_texts(row["ocr_text"], row["classified_text"])
    ) if row["ocr_text"] != row["classified_text"] else None, 
    axis=1
)
if TEXT_CHANGES == 0:
    print("✓ All OCR text preserved in classified entries")
else:
    print(f"⚠ {TEXT_CHANGES} text changes detected between OCR and classified entries")
    any_warnings = True

# TODO: Check bounding boxes match grouped lines

# Check each document has only one, starting state
drop_num_phys = classified_entries[
    (classified_entries["entryType"] != "UNKNOWN") |
    (~classified_entries["full_text"].str.startswith("NUMBER OF"))
]
type_counts_by_pub = drop_num_phys.groupby("publication", as_index=False)["entryType"].value_counts()
bad_state_counts = type_counts_by_pub[
    (type_counts_by_pub["entryType"] == "STATE") & (type_counts_by_pub["count"] != 1)
]
if len(bad_state_counts) > 0:
    print(f"\n❌ Unexpected number of state entries found:\n{bad_state_counts}\n")
    any_errors = True
else:
    print("✓ All publications have exactly one STATE entry")

# Check that there are only a few, correct UNKNOWNs. 
bad_unknown_counts = type_counts_by_pub[
    (type_counts_by_pub["entryType"] == "UNKNOWN") & (type_counts_by_pub["count"] > 0)
]
if len(bad_unknown_counts) > 0:
    print(f"\n❌ UNKNOWN entries found:\n{bad_unknown_counts}\n")
    any_errors = True
else:
    print("✓ No UNKNOWN entries found")

# Check the relative proportion of doc to city entries across publications
city_doc_proportions_by_pub = classified_entries[
    classified_entries["entryType"].isin(["DOC", "CITY"])
].groupby("publication", as_index=False)["entryType"].value_counts(normalize=True)
bad_doc_proportions_by_pub = city_doc_proportions_by_pub[
    (city_doc_proportions_by_pub["entryType"] == "DOC") & 
    ((city_doc_proportions_by_pub["proportion"] > 0.94) | 
        (city_doc_proportions_by_pub["proportion"] < 0.69))
]
if len(bad_doc_proportions_by_pub) > 0:
    print(f"⚠ Unusually high or low proportions of docs to cities:\n{bad_doc_proportions_by_pub}\n")
    any_warnings = True
else:
    print("✓ Proportions of doc to city entries look consistent across publications")


if not any_errors and not any_warnings:
    print("✓ All checks passed!")

print("=" * 80 + "\n")




    