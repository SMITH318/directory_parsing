#!/usr/bin/env python3
"""
Step 5: Group and classify OCR lines into entries
- Reads .jsonl OCR output with line-level text and bounding boxes.
- Groups text blocks into entries based on content, using heuristics and optional user prompts for ambiguous cases.
- Saves segmented entries to CSV with metadata and aggregate bounding boxes.
"""

#from operator import index
from AMD_1918_util_lib import *
import json
from pathlib import Path
from collections import defaultdict
#from datetime import datetime
import csv


################################################## constants ##################################################
MAX_LINES_TO_AGG = 4
RETAIN_LINES = False  # Whether to  preserve line breaks (\n) in full_text or convert them to spaces
ASSUME_NO_AGG = True # with non-test runs, whether to assume not to aggregate, otherwise prompts
RUN_TESTS = False # Whether to run tests or process data via main()
################################################## constants ##################################################

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
        #agg up to MAX_LINES_TO_AGG lines until a line fits a pattern
        sub_index = index
        agged_line = []
        current_entry = {"linetype": LineType.UNKNOWN, "blocks": []}
        entry_done = False
        line_type = None
        while sub_index-index < MAX_LINES_TO_AGG and sub_index < len(sorted_blocks):
            block = sorted_blocks[sub_index]
            agged_line.append(block["text"])
            current_entry["blocks"].append(block)

            prev_line_type = line_type
            line_type = get_line_type(' '.join(agged_line))
            print(f'Checking aggregated lines: {" ".join(agged_line)} => {line_type}')
            if line_type in [LineType.STATE, LineType.CITY, LineType.DOC_FULL]:
                # done with entry
                current_entry["linetype"] = line_type
                index = sub_index
                entry_done = True
                break
            elif prev_line_type in [LineType.DOC_TO_ADDR, LineType.DOC_TO_OFF]:
                # aggregating to prev line looked like it could be full doc, and continuing didn't
                # undo what was added at start of inner, sub_index loop
                agged_line.pop()
                current_entry["blocks"].pop()
                # set appropriate done values
                current_entry["linetype"] = LineType.DOC_FULL
                index = sub_index - 1
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

        # didn't match but check if prev was possible LineType.DOC_FULL                    
        if line_type in [LineType.DOC_TO_ADDR, LineType.DOC_TO_OFF]:
            # treat as LineType.DOC_FULL
            current_entry["linetype"] = LineType.DOC_FULL
            index = sub_index
            entry_done = True

        if not entry_done and len(current_entry["blocks"]) > 0:
            # print(line_type, sub_index-index, "; len:", len(sorted_blocks))
            # prompt how many lines to keep aggregated
            to_agg = prompt_lines_to_agg(agged_line, assume_no_agg)
            agged_line = agged_line[:to_agg]
            current_entry["blocks"] = current_entry["blocks"][:to_agg]
            index += to_agg - 1
            # prompt line type? <=========================================================
            current_entry["linetype"] = LineType.UNKNOWN#prompt_line_type(agged_line) # i think unknown is better to mark need to fix later
            entry_done = True

        if entry_done:
            entries.append(current_entry)
        index += 1
    return entries

def main():
    # Setup project paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    input_filename = "ocr_output_auto_cleaned.jsonl"
    input_file = project_root / "data" / "02_raw_batch" / input_filename
    output_file = project_root / "data" / "03_processed_batch" / "entries_segmented.csv"
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
    num_unknown = 0
    for (pub_name, page_num, col_num), blocks in data_by_col.items():
        print(f"Segmenting: {pub_name} - Page {page_num} Column {col_num}")

        entries = group_into_entries(blocks, ASSUME_NO_AGG)

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

            if entry["linetype"] is LineType.UNKNOWN:
                num_unknown += 1

            entry_out = {
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
            all_entries.append(entry_out)

    # 3. Save to CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_entries[0].keys())
        writer.writeheader()

        # Iterate through each snippet (column)
        for e in all_entries:
            writer.writerow(e)

    print(f"\n✓ Success! Extracted {len(all_entries)} entries.")
    print(f"\tUnknown entries {num_unknown} ({100 * num_unknown / len(all_entries)}%)")
    print(f"Saved to: {output_file}")

if RUN_TESTS:
    def blockify_lines(lines):
        return [{"text": l} for l in lines]

    TESTS = [
        (["OWENS, SEABORN WESLEY-◊; (l 87)", "SCARBROUGH, BEMON CREIGHTON (b'85)-", "Tenn.19,'11; (l 11)"], [1,2]),
        (["Williams, David Calhoun (b'86)-Tenn.8,", "'11; not in practice; R.D.1; ▼", "ASPEL, 26, JACKSON"], [2,1]),
        (["Gattis, Henry Franklin-◊; (l 82)", "ATHENS, 1,715, LIMESTONE"], [1,1]),
        (["DARBY, HENRY ALONZO (b'75)-Ala.4,", "'01; (l 01); R.D"], [2]),
        (["Edwards, James A. (b'64)-Tenn.6,'88; not", "in practice"], [2]),
        (["GLAZE, ANDREW LEWIS, JR. (b'88)⊕-", "Tenn.5,'12; (l 13); D; ▼"], [2]),
        (["DRAKE, JOHN HODGES, SR. (b'45)⊕-", "Ga.5,'67; (l 81)"], [2]),
        (["LEVI, IRWIN PALMER (b'87)⊕-Pa.1,'09;", "(l 09); 1329 Quintard Ave.; office, Hill", "Bldg.; 10-12, 2;30-5; R"], [3]), # greedy addresses
        (["Jones, Lee G. (b'73)-Ga.1,'96, Tenn.11,'98;", "(l t)"], [2]), # 2 schools
        (["GOGGANS, JAMES ADRIAN (b'54)-N.Y.5,", "'77; (l 82); (A628); S"], [2]), # associations
        (["Kyle, Wm. Bailey-Ala.2,'89; (l 89); R.D", "Milhouse, Wm. A.-Tenn.1,'68; (♁); R.D", "Moore, Elisha B.-◊; (l 78); not in practice"], [1,1,1]),
        (["KIMBELL, ISHAM (b'84)-Ala.2,'09; (l 09);", "Ob; ▼G", "Kirven, Thos. C.-Ky.4,'93; (l 93)"], [2,1]),
        (["LACEY, EDWARD PARISH (b'56)⊕-", "Tenn.5,'83; (l 83); 1802, 8th Ave.; office,", "Realty Bldg"], [3]),
        (["WALLER, GEO. DE ILOACH (b'70)-Tenn.5,", "'99; (l 99); 1710, 4th Ave.; office, 210½", "19th St.; 10-12, 2-4"], [3]),
    ]

if __name__ == "__main__":
    if RUN_TESTS:
        any_bad=0
        for test, group_lens in TESTS:
            grouped = group_into_entries(blockify_lines(test), assume_no_agg=True)
            err = False
            if len(grouped) != len(group_lens):
                print("\n**bad test: number of groups", len(grouped),"didn't match for '", test, "'")
                print("\tExpected:", group_lens)
                err = True
            else:
                for i, group in enumerate(grouped):
                    if len(group["blocks"]) != group_lens[i]:
                        print("bad test: number in group", len(group["blocks"]), "didn't match for '", test, "'")
                        print("\tExpected:", group_lens[i])
                        err = True
            if err:
                any_bad+=1
                print("\tFound blocks:")
                for group in grouped:
                    print("\t", group["blocks"])
                print('\n')

        print('****', any_bad, 'bad tests out of', len(TESTS), '****')

    else:
        main()