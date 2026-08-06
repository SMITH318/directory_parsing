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

import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    handlers=[
        logging.FileHandler('05_sanity_check.log', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stderr)
    ],
    level=logging.WARNING
)

TEXT_CHANGES = 0 # incremented in the compare_texts function for each change detected
OCR_COLS = ["pub", "page", "col"]
CLASSIFIED_COLS = ["publication", "page_number", "column"]

def main(dataset: str) -> int:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / dataset
    
    ocr_path = data_dir / "04_ocr_output_cleaned.jsonl"
    classified_entries_path = data_dir / "05_entries_segmented.csv"
    auto_agged_ocr_file = data_dir / "04.09_ocr_output_auto_agged.csv"

    def strip_linebreak_diffs(text_in:list[str], text_out:list[str]) -> list[str]:
        """Remove line(entry)breaks from text_in if they don't appear or are spaces in text_out"""
        text1 = "\n".join(text_in)
        text2 = "\n".join(text_out)
        chars_to_ignore_at_EOL = ".,;: -\n\t"
        matcher = difflib.SequenceMatcher(None, text1, text2)
        text_so_far = ""
        for change_type, i1, i2, j1, j2 in matcher.get_opcodes():
            affected_text_in = text1[i1:i2]
            if change_type == 'delete':
                if len(affected_text_in) > 1:
                    if affected_text_in.startswith("-\n"):
                        affected_text_in = affected_text_in[2:]
                affected_text_in = affected_text_in.strip()
            if change_type == 'replace':
                if len(affected_text_in) > 1:
                    if affected_text_in[0] in chars_to_ignore_at_EOL and affected_text_in[1] == "\n":
                        affected_text_in = affected_text_in[2:]
                if len(affected_text_in) > 1:
                    if affected_text_in[-2] in chars_to_ignore_at_EOL and affected_text_in[-1] == "\n":
                        affected_text_in = affected_text_in[:-2]
                    elif affected_text_in[-1] in chars_to_ignore_at_EOL:
                        affected_text_in = affected_text_in[:-1]
                changed_to = text2[j1:j2]
                if changed_to.replace(":",";").strip(chars_to_ignore_at_EOL) == affected_text_in.strip(chars_to_ignore_at_EOL):
                    text_so_far += changed_to
                    continue                
            text_so_far += affected_text_in
            
        return text_so_far.splitlines()

    def compare_texts(row)->bool:
        """Compare two texts and print out the differences with context."""
        text1 = row['ocr_text']
        text2 = row['classified_text']
        logger.warning(
                f"⚠ warning: text changed in {row['pub']}.{row['page']}.{row['col']}!"
                f"(length in {len(text1)} vs. out {len(text2)})"
            )
        ignore = lambda x: x in " \n"
        matcher = difflib.SequenceMatcher(ignore, text1, text2.replace(":",";"))
        texts_differ = False

        for change_type, i1, i2, j1, j2 in matcher.get_opcodes():
            if change_type == 'equal':
                continue
            
            global TEXT_CHANGES
            TEXT_CHANGES += 1
            texts_differ = True

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
                affected_text_in = affected_text_in#f"{affected_text_in:<{max_len}}"
                affected_text_out = affected_text_out#f"{affected_text_out:<{max_len}}"

            logger.warning(f"  Change Type: {change_type}")
            # Highlight changes using brackets and padded text
            logger.warning(f"    text from OCR:             '{context_before_in}{{{{{affected_text_in}}}}}{context_after_in}'")
            logger.warning(f"    text after classification: '{context_before_out}{{{{{affected_text_out}}}}}{context_after_out}'")
            logger.warning("--------------------------------------------------")
        return texts_differ
    
    #Load OCR output JSONL file
    try:
        ocr_data = pd.read_json(ocr_path, lines=True)
    except Exception as e:
        logger.error(f"Error loading OCR output {ocr_path.name}: {str(e)}")
        return 1

    # Load classified entries CSV file
    try:
        classified_entries = pd.read_csv(classified_entries_path, encoding="utf-8")
        classified_entries["entryType"] = classified_entries["entryType"].str.upper()
    except Exception as e:
        logger.error(f"Error loading classified entries output {classified_entries_path.name}: {str(e)}")
        return 1

    logger.info("=" * 80)
    logger.info("SANITY CHECK REPORT - LINE GROUPING & CLASSIFICATION CONSISTENCY CHECKS")
    logger.info("-" * 80)
    any_errors = False
    any_warnings = False

    # Check entries exist for all pages, rows, and columns
    ocr_cols = ocr_data[OCR_COLS].drop_duplicates()
    classified_cols = classified_entries[CLASSIFIED_COLS].drop_duplicates()
    if len(ocr_cols) != len(classified_cols):
        logger.error(
            f"❌  Inconsistent publication, column counts:"
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
        logger.error(f"❌  OCR columns with no classified entries:\n{unmatched_ocr_cols.drop('_merge', axis=1)}\n")
        any_errors = True
    else:
        logger.info("✓ All OCR columns have classified entries")

    # Check all text has been preserved in order (after removing whitespace and hyphens to deal with lines being collapsed)
    ocr_data['text'] = ocr_data['text'].str.strip()
    ocr_cols_text = ocr_data.groupby(OCR_COLS)['text'].agg(list).reset_index()
    classified_entries['full_text'] = classified_entries['full_text'].str.strip()
    classified_cols_text = classified_entries.groupby(CLASSIFIED_COLS)['full_text'].agg(list).reset_index()
    matched_cols_text = ocr_cols_text.merge(
        classified_cols_text, 
        left_on=OCR_COLS, 
        right_on=CLASSIFIED_COLS, 
        how="inner"
    ).rename(columns={"text": "ocr_text", "full_text": "classified_text"})

    matched_cols_text['ocr_text'] =  matched_cols_text.apply(
        lambda row: strip_linebreak_diffs(row['ocr_text'], row['classified_text']),
        axis = 1
    )

    matched_cols_text[OCR_COLS + ["ocr_text"]].to_csv(auto_agged_ocr_file, index=False, encoding='utf-8')

    matched_cols_text['ocr_text'] = matched_cols_text['ocr_text'].str.join("\n")
    matched_cols_text['classified_text'] = matched_cols_text['classified_text'].str.join("\n")

    matched_cols_text["has_diffs"] = matched_cols_text.apply(
        lambda row: (
            row["ocr_text"] != row["classified_text"]
            and compare_texts(row)
        ),
        axis=1
    )
    if TEXT_CHANGES == 0:
        logger.info(f"✓ All OCR text preserved in classified entries")
    else:
        logger.warning(f"⚠ {TEXT_CHANGES} text changes detected between OCR and classified entries")
        any_warnings = True

    # TODO: Check bounding boxes match grouped lines

    # check there are between 50 and 60 states (b/c of territories, etc.)
    # type_counts = classified_entries["entryType"].value_counts()
    # state_count = type_counts.loc["STATE"]
    # print(state_count)
    # if state_count < 50 or state_count > 60:
    #     logger.error(f"❌ Unexpected number of total state entries found: {state_count} (expected between 50 and 60)")
    #     any_errors = True
    # else:
    #     logger.info(f"✓ Total number of state entries between 50 and 60: {state_count}")


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
        logger.error(f"❌ Unexpected number of state entries found per publication:\n{bad_state_counts}\n")
        any_errors = True
    else:
        logger.info("✓ All publications have exactly one STATE entry")

    # Check that there are only a few, correct UNKNOWNs. 
    bad_unknown_counts = type_counts_by_pub[
        (type_counts_by_pub["entryType"] == "UNKNOWN") & (type_counts_by_pub["count"] > 0)
    ]
    if len(bad_unknown_counts) > 0:
        logger.error(f"❌ UNKNOWN entries found:\n{bad_unknown_counts}\n")
        any_errors = True
    else:
        logger.info("✓ No UNKNOWN entries found")

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
        logger.warning(f"⚠ Unusually high or low proportions of docs to cities:\n{bad_doc_proportions_by_pub}\n")
        any_warnings = True
    else:
        logger.info("✓ Proportions of doc to city entries look consistent across publications")

    print(classified_entries_path)
    if not any_errors and not any_warnings:
        logger.warning(f"✓ All checks passed! ({classified_entries_path})")
    logger.info("=" * 80)
    if not any_errors: # let manual review decide if warnings should stop process
        if any_warnings:
            logger.warning(f"Warnings during sanity check. Manually review them before continuing with step 8.")
        return 0
    return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Sanity Check Classified Entries")
    parser.add_argument("dataset", help="Name of the dataset")
    args = parser.parse_args()
    
    sys.exit(main(args.dataset))





    