"""
Step 3: Combine and Sort OCR Output
Combines multiple OCR output files into one, sorts by pub, page, col, and removes duplicates.
Drops rows with text that matches SKIP_TEXT, which is a placeholder for rows that kept failing 
and were skipped in the OCR process.
"""
import pandas as pd
from pathlib import Path
import json

SKIP_TEXT = "******* KEEPS FAILING! SKIPPING FOR NOW *******"

# Setup project paths
script_dir = Path(__file__).parent if '__file__' in dir() else Path.cwd()
project_root = script_dir if (script_dir / "data").exists() else script_dir.parent

target_dir = project_root / "data" / "02_raw_batch_mass" 
files_to_combine = ["ocr_output.jsonl"]#, "ocr_output_fill_in.jsonl"]
paths_to_combine = [target_dir/f for f in files_to_combine]
output_file = target_dir / "ocr_output_combined_sorted.jsonl"

# load add lines from all files into a list of dicts
ocr_data = []
for path in paths_to_combine:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                ocr_data.append(json.loads(line))

# create a DataFrame, drop rows with SKIP_TEXT, sort by pub, page, col, and drop duplicates
df_ocr = pd.DataFrame(ocr_data)
df_no_skipped = df_ocr.loc[df_ocr["text"] != SKIP_TEXT]
print(f"dropped {len(df_ocr)- len(df_no_skipped)} skipped rows, {len(df_no_skipped)} left")
df_sorted = df_no_skipped.sort_values(by=["pub", "page", "col"])
df_deduped = df_sorted.drop_duplicates()
print(f"dropped {len(df_sorted)- len(df_deduped)} exact duplicate rows, {len(df_deduped)} left")

# save the combined, sorted, deduped DataFrame to a new JSONL file
df_deduped.to_json(output_file, orient="records", force_ascii=False, lines=True)
