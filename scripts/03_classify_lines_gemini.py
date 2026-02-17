import json
import numpy as np
from pathlib import Path
from google import genai
from google.genai import types
import pandas as pd
from  pydantic import BaseModel
import os
import time
import csv
from collections import defaultdict
import datetime
from typing import Literal

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
  filename='03_classify_lines_gemini.log', 
  filemode='a', 
  encoding='utf-8', 
  level=logging.INFO) ## <=================== Change logging level here

# --- Gemini API Configuration --- 
API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY')
model_name ='gemini-flash-latest'

if API_KEY == 'YOUR_API_KEY' or not API_KEY:
    print("ERROR: Gemini API key is not set.")
    logger.error("ERROR: Gemini API key is not set.")
    exit(1)

# Initialize Gemini
print("Initializing Gemini for OCR...")
#print(f"Key: {API_KEY}")
logger.info(f"Initializing Gemini for OCR... with {model_name}")
client = genai.Client(api_key=API_KEY)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.float32, np.float64)):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

class ClassifiedLine(BaseModel):
    publication: str
    page_number: int
    column: int
    entryType: Literal["STATE", "CITY", "DOC", "UNKNOWN"]
    full_text: str
    x: int
    y: int
    width: int
    height: int

class ClassifiedEntries(BaseModel):
    entries: list[ClassifiedLine]

MODEL_PROMPT = (
    "Combine these ordered lines of text from a directory into entries. "
    "Each line may be a complete entry or part of a multi-line entry. " 
    "Entries are contained within a single column of a page and publication. "
    "Entries can be: (1) names of states or provinces; " 
    "(2) cities with their county and often with their population; or "
    "(3) doctors including a wide range of informatiom about them "
    "and their careers, including their birth year, education, address, " 
    "office address, any specialty, among others. "
    "For each line of text, I'm providing the name of the publication, "
    "the page number, the column number, "
    "the text, a confidence value (0 to 1), and a bounding box " 
    "represented by an x and y coordinate, a height, and a width. "
    "Return ONLY a JSON array of the combined entries, including the source "
    "information (publication, page number, and column number), " 
    "what kind of entry it is (STATE, CITY, DOC, or UNKNOWN), "
    "the combined lines of text, and the combined bounding box for the lines "
    "specified by an x and y coordinate, a height, and a width. " 
    "When combining lines, remove hyphens or dashes at end of lines for words broken "
    "across lines, and combine the full word."
    "Maintain the UTF-8 character encoding, retaining all characters as they are, "
    "with the exception of hyphens or dashes for words broken across lines. "
)

def group_into_entries(temp_file, col_blocks, max_retries=1) -> list[dict]: # [{"linetype": LineType, "blocks": [block, ...]}, ...]
    # if not blocks:
    #     return []

    # # Sort by column, then by Y position
    # sorted_blocks = blocks #sorted(blocks, key=lambda b: (b["bbox"]["y"]))
    # print(col_blocks)

    with open(temp_file, 'w', encoding='utf-8') as temp_f:
        temp_f.write(json.dumps(col_blocks, cls=NumpyEncoder, ensure_ascii=False) + "\n")

    # upload blocks file
    file = client.files.upload(
        file=temp_file,
        config=types.UploadFileConfig(mime_type='application/json')
    )
    
    # send aggregation prompt 
    logger.info(f"prompting for {len(col_blocks)} blocks at {datetime.datetime.now()}")
    print(f"\tprompting for {len(col_blocks)} blocks")
    #print(f"\tsending {len(blocks[0])}")
    for attempt in range(max_retries):
        entries = [] # [{"linetype": LineType, "blocks": [block, ...]}, ...]
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[file],
                config=types.GenerateContentConfig(
                    system_instruction=MODEL_PROMPT,
                    temperature=0.0,
                    response_mime_type="application/json", 
                    response_json_schema=ClassifiedEntries.model_json_schema(),
                )
            )
            response_text = response.text.strip()
            logging.info("="*20)
            logging.info(response_text)
            logging.info("="*20)

            # Clean markdown formatting if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
                response_text = response_text.replace('```json', '').replace('```', '').strip()

            entries = json.loads(response_text)["entries"]

            if not isinstance(entries, list):
                raise ValueError("Response is not a JSON array")

        except json.JSONDecodeError as e:
            print(f"\n    Warning: JSON parse error (attempt {attempt + 1}/{max_retries})")
            logger.warning(f"JSON parse error (attempt {attempt + 1}/{max_retries})")
            time.sleep(1)
        except Exception as e:
            print(f"\n    Warning: Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
            logger.warning(f"Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)
    

    # delete local and remote blocks files
    temp_file.unlink(missing_ok=True)
    client.files.delete(name=file.name)

    logger.info(f"received {len(entries)} entries at {datetime.datetime.now()}")
    print(f"\treceived {len(entries)}")

    # verify all text is present, in the right order
    text_in = " ".join([block["text"] for block in col_blocks])
    text_out = " ".join([entry["full_text"] for entry in entries])
    if text_in != text_out:
        print(f"warning: entry text dropped!")
        logger.error(f"entry text dropped:\ntext_in:  {text_in}\ntext_out: {text_out}")

    return entries


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Use command line argument or default
    input_filename = "ocr_output_auto_cleaned.jsonl"
    input_file = project_root / "data" / "02_raw_batch" / input_filename
    output_file = project_root / "data" / "03_processed_batch" / "entries_segmented_gemini.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.parent / "temp_col.json"

    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    # 1. Load streaming data and group by Page/Column
    print("Loading and grouping OCR data...")
    data_by_col = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            
            # flatten bbox into entry
            if entry.get("bbox") is not None:
                # blocks[bbox] is [x, y, w, h], normalize as dict
                if isinstance(entry["bbox"], list):
                    bbox = entry["bbox"]
                    entry["bbox"] = {"x": bbox[0], "y": bbox[1], "width": bbox[2], "height": bbox[3]}
                # do flattening
                entry["x"] = int(entry["bbox"]["x"])
                entry["y"] = int(entry["bbox"]["y"])
                entry["width"] = int(entry["bbox"]["width"])
                entry["height"] = int(entry["bbox"]["height"])
                del entry["bbox"]

            # Create a unique key for each column of each PDF
            page_key = (entry['pub'], entry['page'], entry['col'])
            data_by_col[page_key].append(entry)


    # 2. Process each column
    num_unknown = 0
    all_entries = []
    for (pub_name, page_num, col_num), blocks in data_by_col.items():
        print(f"Clumping: {pub_name} - Page {page_num} Column {col_num}")

        entries = group_into_entries(temp_file, blocks)

        for idx, entry in enumerate(entries):
            if entry["entryType"] == "UNKNOWN":
                num_unknown += 1
            all_entries.append(entry)

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
    logger.info(f"\tUnknown entries {num_unknown} ({100 * num_unknown / len(all_entries)}%)")
    logger.error(f"Saved {len(all_entries)} entries to: {output_file}")

if __name__ == "__main__":
    main()


client.close()