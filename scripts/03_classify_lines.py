#!/usr/bin/env python3
"""
Step 3: Article Segmentation
- Reads streaming .jsonl OCR output.
- Groups text blocks into articles based on headlines and vertical gaps.
- Optimized for Tesseract output and 1880s newspaper layouts.
"""

from operator import index
from AMD_1918_util_lib import *
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import csv

def prompt_lines_to_agg(agged_line, assume_no_agg):
    """Prompt the user for the number of lines to aggregate or assume no aggregation."""
    if assume_no_agg:
        return 1
    print("\nCurrent aggregated lines:\n\t", "\n\t".join(agged_line))
    user_agg_key = input("How many lines to keep aggregated? (Enter digit)\n")
    if not user_agg_key or len(user_agg_key) == 0 or not user_agg_key[0].isdigit():
        return 1
    return int(user_agg_key[0])

def prompt_line_type(agged_line):
    """Prompt the user for the line type of the current entry."""
    print("\nCurrent line:\n\t", " ".join(agged_line))
    user_line_type = input("What is the line type? ([U]NKNOWN, [S]TATE, [C]ITY, [D]OC)\n")
    if not user_line_type or len(user_line_type) == 0:
        return LineType.UNKNOWN
    match user_line_type[0]:
        case 'S' | 's':
            return LineType.STATE
        case 'C' | 'c':
            return LineType.CITY
        case 'D' | 'd':
            return LineType.DOC_FULL
        case _:
            return LineType.UNKNOWN

def group_into_entries(blocks, assume_no_agg=False) -> list[dict]: # [{"linetype": LineType, "blocks": [block, ...]}, ...]
    if not blocks:
        return []

    # Sort by column, then by Y position
    sorted_blocks = blocks #sorted(blocks, key=lambda b: (b["bbox"]["y"]))

    entries = [] # [{"linetype": LineType, "blocks": [block, ...]}, ...]

    index = 0
    while index < len(sorted_blocks):
        #agg up to 3 lines until a line fits a pattern
        sub_index = index
        agged_line = []
        current_entry = {"linetype": LineType.UNKNOWN, "blocks": []}
        entry_done = False
        while sub_index-index < 3 and sub_index+1 < len(sorted_blocks):
            block = sorted_blocks[sub_index]
            agged_line.append(block["text"])
            current_entry["blocks"].append(block)

            line_type = get_line_type(' '.join(agged_line))
            print(f'Checking aggregated lines: {" ".join(agged_line)} => {line_type}')
            if line_type in [LineType.STATE, LineType.CITY, LineType.DOC_FULL]:
                # done with entry
                current_entry["linetype"] = line_type
                index = sub_index
                entry_done = True
                break
            else:
                # continue agging
                sub_index += 1
        
                # check next line ???
                # next_line=sorted_blocks[sub_index]["text"]
                # next_line_type = get_line_type(next_line)
                # #print('next_line_type: ', next_line_type)
                # if next_line_type in [LineType.STATE,
                #                             LineType.CITY,
                #                             LineType.DOC_START]:
                #     #print('matched something else')
                #     to_agg = prompt_lines_to_agg(agged_line, assume_no_agg)
                #     index += to_agg - 1
                #     current_entry = agged_blocks[:to_agg]
                #     entry_done = True
                #     break
                # else: #next_line doesn't fit pattern, keep agging
                    
        if not entry_done and len(current_entry["blocks"]) > 0:
            # prompt how many lines to keep aggregated
            to_agg = prompt_lines_to_agg(agged_line, assume_no_agg)
            agged_line = agged_line[:to_agg]
            current_entry["blocks"] = current_entry["blocks"][:to_agg]
            index += to_agg - 1
            # prompt line type? <=========================================================
            current_entry["linetype"] = LineType.UNKNOWN#prompt_line_type(agged_line) # i thinmk unknown is better to mark need to fix later
            entry_done = True

        if entry_done:
            entries.append(current_entry)
        index += 1
    return entries

def main():
    RETAIN_LINES = True  # Set to True to preserve line breaks in full_text

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Segment OCR output into articles')
    parser.add_argument('--input', type=str, help='Input OCR file name (default: ocr_output_tesseract.jsonl)')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Use command line argument or default
    input_filename = args.input if args.input else "ocr_output_reviewed.jsonl"
    input_file = project_root / "data" / "02_raw" / input_filename
    output_file = project_root / "data" / "03_processed" / "entries_segmented.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    # 1. Load streaming data and group by Page
    print("Loading and grouping OCR data...")
    data_by_col = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            # blocks[bbox] is [x, y, w, h], normalize as dict
            if isinstance(entry["bbox"], list):
                bbox = entry["bbox"]
                entry["bbox"] = {"x": bbox[0], "y": bbox[1], "width": bbox[2], "height": bbox[3]}
            # Create a unique key for each column of each PDF
            page_key = (entry['pub'], entry['page'], entry['col'])
            data_by_col[page_key].append(entry)

    all_entries = []

    # 2. Process each column
    for (pub_name, page_num, col_num), blocks in data_by_col.items():
        print(f"Segmenting: {pub_name} - Page {page_num} Column {col_num}")

        entries = group_into_entries(blocks)

        for idx, entry in enumerate(entries, 1):
            if RETAIN_LINES:
                full_text = "\n".join(b["text"] for b in entry["blocks"]).strip()
            else:
                full_text = " ".join(b["text"] for b in entry["blocks"]).strip()
            # Calculate aggregate bounding box for the whole article
            all_x = [b["bbox"]["x"] for b in entry["blocks"]]
            all_y = [b["bbox"]["y"] for b in entry["blocks"]]
            all_x_end = [b["bbox"]["x"] + b["bbox"]["width"] for b in entry["blocks"]]
            all_y_end = [b["bbox"]["y"] + b["bbox"]["height"] for b in entry["blocks"]]

            entry = {
                "entry_id": f"{pub_name}_p{page_num:03d}_c{col_num:03d}_e{idx:03d}",
                "source_pdf": pub_name,
                "page_number": page_num,
                "column": col_num,
                "lineType": entry["linetype"],
                "full_text": full_text,
                "x": min(all_x),
                "y": min(all_y),
                "width": max(all_x_end) - min(all_x),
                "height": max(all_y_end) - min(all_y)
            }
            all_entries.append(entry)

    # 3. Save to CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_entries[0].keys())
        writer.writeheader()

        # Iterate through each snippet (column)
        for e in all_entries:
            writer.writerow(e)

    print(f"\n✓ Success! Extracted {len(all_entries)} entries.")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()