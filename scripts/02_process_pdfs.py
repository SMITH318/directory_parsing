#!/usr/bin/env python3
"""
Step 2: Robust Streaming OCR
- Fixes 'ValueError: not enough values to unpack'
- Streaming output to JSONL (safe for Google Colab/M1)
- Handles large snippets without OOM crashes
"""

import json
import gc
import cv2
import numpy as np
import os
from pathlib import Path
from paddleocr import PaddleOCR

# Initialization
print("Initializing PaddleOCR...")
# Note: det_limit_side_len=2000 is the secret for large newspaper scans
ocr = PaddleOCR(lang='en', use_angle_cls=False, det_limit_side_len=2000)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.float32, np.float64)):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def main():
    # Detect Environment and Setup Paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    preprocessed_dir = project_root / "data" / "01_preprocessed"
    output_dir = project_root / "data" / "02_raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = preprocessed_dir / "all_metadata.json"
    if not metadata_path.exists():
        print(f"Error: Could not find metadata at {metadata_path}")
        return

    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)

    output_file = output_dir / "ocr_output.jsonl"
    print(f"Streaming results to: {output_file}")
    
    # Open in 'a' (append) mode so you can resume if it crashes
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for pdf_meta in all_metadata:
            pdf_name = pdf_meta.get('source_pdf', 'unknown.pdf')
            
            for page_meta in pdf_meta.get('pages', []):
                page_num = page_meta.get('page_num', 0)
                print(f"Processing {pdf_name} - Page {page_num}")
                
                for snip in page_meta.get('snippets', []):
                    img_path = snip['path']
                    if not os.path.exists(img_path): continue
                    
                    img = cv2.imread(img_path)
                    if img is None: continue
                    
                    try:
                        # Standard call for PaddleOCR
                        result = ocr.predict(img)
                    except Exception as e:
                        print(f"OCR Error: {e}")
                        continue
                    
                    if not result or not result[0]:
                        del img
                        continue

                    # Each line is: [ [bbox], (text, confidence) ]
                    for line in result[0]:
                        try:
                            # --- FIX FOR UNPACKING ERROR ---
                            # Ensure 'line' has 2 elements and line[1] is the (text, conf) tuple
                            if len(line) < 2 or not isinstance(line[1], (tuple, list)):
                                continue
                            
                            # Ensure line[1] has exactly (text, confidence)
                            if len(line[1]) < 2:
                                continue

                            coords = line[0]
                            text, conf = line[1] # <--- This is where it was crashing

                            x_off = float(snip.get('x_offset', 0))
                            y_off = float(snip.get('y_offset', 0))

                            # Global mapping
                            real_x = min(p[0] for p in coords) + x_off
                            real_y = min(p[1] for p in coords) + y_off

                            entry = {
                                "pub": pdf_name,
                                "page": page_num,
                                "col": snip.get('column', 0),
                                "text": text.strip(),
                                "conf": round(float(conf), 4),
                                "bbox": [int(real_x), int(real_y)]
                            }
                            
                            f_out.write(json.dumps(entry, cls=NumpyEncoder, ensure_ascii=False) + "\n")
                            
                        except Exception:
                            continue # Skip malformed lines

                    # Force Cleanup
                    f_out.flush()
                    del img
                    del result
                    gc.collect()

    print(f"Done! Output is in {output_file}")

if __name__ == "__main__":
    main()