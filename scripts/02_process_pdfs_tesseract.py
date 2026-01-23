#!/usr/bin/env python3
"""
Step 2: Pytesseract OCR - Pythonic Streaming Edition
- Much lower RAM footprint than PaddleOCR.
- Uses system Tesseract binary.
- Streams results to JSONL line-by-line.
"""

import json
import pytesseract
from PIL import Image
from pathlib import Path
import os
import gc

def main():
    # Setup Paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    preprocessed_dir = project_root / "data" / "preprocessed"
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = preprocessed_dir / "all_metadata.json"
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found.")
        return

    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)

    output_file = output_dir / "ocr_output_tesseract.jsonl"
    print(f"Starting Tesseract OCR... Streaming to: {output_file}")

    # CONFIG: --psm 6 assumes a single uniform block of text (ideal for snippets)
    # --oem 1 uses the LSTM engine
    tess_config = r'--oem 1 --psm 6'

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for pdf_meta in all_metadata:
            pub_name = pdf_meta.get('source_pdf')
            
            for page in pdf_meta.get('pages', []):
                page_num = page.get('page_num')
                print(f"Processing {pub_name} - Page {page_num}...")

                for snip in page.get('snippets', []):
                    img_path = snip['path']
                    if not os.path.exists(img_path): continue

                    try:
                        # Pytesseract works directly with PIL Images
                        img = Image.open(img_path)
                        
                        # image_to_data returns a dictionary with text and coordinates
                        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=tess_config)
                        
                        # Data contains lists for 'text', 'left', 'top', 'width', 'height', 'conf'
                        for i in range(len(data['text'])):
                            text = data['text'][i].strip()
                            conf = float(data['conf'][i])

                            # Filter: Tesseract returns empty strings for layout blocks; ignore them.
                            # Also ignore low confidence "noise"
                            if text and conf > 10:
                                x_off = snip['x_offset']
                                y_off = snip['y_offset']

                                # Construct global coordinates
                                entry = {
                                    "pub": pub_name,
                                    "page": page_num,
                                    "col": snip.get('column', 0),
                                    "text": text,
                                    "conf": round(conf / 100, 3), # Tesseract uses 0-100 scale
                                    "bbox": [
                                        int(data['left'][i] + x_off),
                                        int(data['top'][i] + y_off),
                                        int(data['width'][i]),
                                        int(data['height'][i])
                                    ]
                                }
                                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        
                        # Flush after every snippet for streaming
                        f_out.flush()

                    except Exception as e:
                        print(f"Error on snippet {img_path}: {e}")
                    
                    finally:
                        # Clean up memory
                        if 'img' in locals(): del img
                        gc.collect()

    print(f"\n✓ Tesseract OCR Complete. Results in {output_file}")

if __name__ == "__main__":
    main()