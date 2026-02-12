#!/usr/bin/env python3
"""
OCR Processing Script - Gemini Vision Edition
Modified for JSONL output and memory efficiency.
"""

import json
import gc
#import cv2
import numpy as np
from pathlib import Path
import google.generativeai as genai
from PIL import Image
import os
import time
import pandas as pd

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
  filename='02_gemini.log', 
  filemode='a', 
  encoding='utf-8', 
  level=logging.WARNING) ## <=================== Change logging level here

# --- Gemini API Configuration --- 
API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY')
model_name ='gemini-flash-latest'

if API_KEY == 'YOUR_API_KEY' or not API_KEY:
    print("ERROR: Gemini API key is not set.")
    logger.error("ERROR: Gemini API key is not set.")
    exit(1)

# Initialize Gemini
print("Initializing Gemini for OCR...")
logger.info(f"Initializing Gemini for OCR... with {model_name}")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.float32, np.float64)):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def gemini_ocr(image_path, max_retries=3):
    """Extract text and bounding boxes from an image using Gemini Vision API."""
    try:
        img_pil = Image.open(image_path).convert("RGB")
        w, h = img_pil.size
    except Exception as e:
        print(f"    Error opening image {image_path}: {e}")
        logger.error(f"Error opening image {image_path}: {e}")
        return []

    # prompt = (
    #     "Extract all text from this newspaper image. "
    #     "For each text block, provide the text content and its bounding box coordinates. "
    #     "Return ONLY a JSON array with this exact format:\n"
    #     '[\n'
    #     '  {"text": "example text", "x": 10, "y": 20, "width": 100, "height": 15},\n'
    #     '  {"text": "more text", "x": 10, "y": 40, "width": 95, "height": 15}\n'
    #     ']\n'
    #     "Coordinates should be in pixels. Include all text, even small fragments."
    # )

    prompt = (
        "Extract all text from this image of a column from a directory. "
        "For each line of text, provide the text content, its bounding box coordinates, and a confidence score (between 0 and 1). "
        "Lines of text extend horizontally across the whole column. "
        "Return ONLY a JSON array with this exact format:\n"
        '[\n'
        '  {"text": "example text", "x": 10, "y": 20, "width": 100, "height": 15, "confidence": 0.95},\n'
        '  {"text": "more text", "x": 10, "y": 40, "width": 95, "height": 15, "confidence": 0.90}\n'
        ']\n'
        "Coordinates should be in pixels. Include all text, even small fragments. "
        "Include all symbols, punctuation, and line breaks, even if they look like noise. "
        "Encode special characters properly in JSON as UTF-8 characters. "
        "For punctuation like dashes, quotes, and apostrophes, use standard ASCII equivalents."
    )

    for attempt in range(max_retries):
        try:
            response = model.generate_content([prompt, img_pil])
            response_text = response.text.strip()

            # Clean markdown formatting if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
                response_text = response_text.replace('```json', '').replace('```', '').strip()

            text_blocks = json.loads(response_text)

            if not isinstance(text_blocks, list):
                raise ValueError("Response is not a JSON array")

            for block in text_blocks:
            #     block['confidence'] = 0.90 
                if block['confidence'] < 0.90:
                    logger.warning(f"low confidence ({block['confidence']}) produced `{block['text']}` in {image_path}")
            return text_blocks

        except json.JSONDecodeError as e:
            print(f"\n    Warning: JSON parse error (attempt {attempt + 1}/{max_retries})")
            logger.warning(f"JSON parse error (attempt {attempt + 1}/{max_retries}) with {image_path}")
            if attempt == max_retries - 1:
                return fallback_text_extraction(image_path, img_pil, w, h)
            time.sleep(1)
        except Exception as e:
            print(f"\n    Warning: Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
            logger.warning(f"Gemini API error (attempt {attempt + 1}/{max_retries}) with {image_path}: {e}")
            if attempt == max_retries - 1:
                return fallback_text_extraction(image_path, img_pil, w, h)
            time.sleep(2 ** attempt)
    return []

def fallback_text_extraction(image_path, img_pil, width, height):
    """Fallback method when structured extraction fails."""
    logger.warning(f"Using fallback extraction for {image_path}")
    try:
        simple_prompt = "Extract all text from this image. Return only the text, nothing else."
        response = model.generate_content([simple_prompt, img_pil])
        extracted_text = response.text.strip()

        if extracted_text:
            return [{
                "text": extracted_text,
                "x": 0, "y": 0, "width": width, "height": height,
                "confidence": 0.65
            }]
    except:
        pass
    return []

def main():
    # 1. Setup Project Paths
    script_dir = Path(__file__).parent
    project_root = script_dir if (script_dir / "data").exists() else script_dir.parent
    preprocessed_dir = project_root / "data" / "01_preprocessed"
    output_dir = project_root / "data" / "02_raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = preprocessed_dir / "all_metadata.json"
    output_file = output_dir / "ocr_output.jsonl"

    if not metadata_path.exists():
        print(f"Error: Run preprocess.py first. Missing: {metadata_path}")
        logger.error(f"Error: Run preprocess.py first. Missing: {metadata_path}")
        return

    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)

    print(f"Starting OCR. Output will be saved to: {output_file}")

    # 2. Process each PDF
    df_done = pd.read_json(output_file, lines=True)
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for pdf_meta in all_metadata:
            pdf_name = pdf_meta['source_pdf']
            print(f"\nOCRing Snippets for: {pdf_name}")
            logger.info(f"\nOCRing Snippets for: {pdf_name}")

            for page_meta in pdf_meta['pages']:
                page_num = page_meta['page_num']
                print(f"  Page {page_num}...", end="", flush=True)
                logger.info(f"  Page {page_num}...")

                page_blocks = []

                # Process each snippet (article)
                for snip in page_meta['snippets']:
                    col_idx = snip.get('col_idx', snip.get('column', 0))
                    print(f"{col_idx}.", end="", flush=True)
                    if ((df_done['pub'] == pdf_name) & 
                          (df_done['page'] == page_num) &
                          (df_done['col'] == col_idx)).any():
                        logger.info(f"Previously processed column {col_idx}")
                        print(f"Previously processed column {col_idx}", end=" ", flush=True)
                        continue
                    snippet_image_path = snip['path']

                    if not Path(snippet_image_path).exists():
                        logger.error(f"Image file missing: {snippet_image_path}")
                        continue

                    gemini_output = gemini_ocr(snippet_image_path)
                    if not gemini_output or len(gemini_output) == 0:
                        logger.error(f"Failed to parse {snippet_image_path}")

                    for block in gemini_output:
                        try:
                            ## change to write each block as its own line
                            entry = {
                                    "pub": pdf_name,
                                    "page": page_num,
                                    "col": col_idx,
                                    "text": block['text'].strip(),
                                    "conf": round(block.get('confidence', 0.90), 4),
                                    "bbox": {
                                        "x": float(block['x']) + snip['x_offset'],
                                        "y": float(block['y']) + snip['y_offset'],
                                        "width": float(block['width']),
                                        "height": float(block['height'])
                                    }
                                }
                            f_out.write(json.dumps(entry, cls=NumpyEncoder, ensure_ascii=False) + "\n")
                            # Map relative coords to broadsheet coords
                            # page_blocks.append({
                            #     "text": block['text'],
                            #     "confidence": block.get('confidence', 0.90),
                            #     "bbox": {
                            #         "x": float(block['x']) + snip['x_offset'],
                            #         "y": float(block['y']) + snip['y_offset'],
                            #         "width": float(block['width']),
                            #         "height": float(block['height'])
                            #     },
                            #     "column": snip.get('col_idx', snip.get('column', 0))
                            # })
                        except:
                            continue

                    # Rate limiting
                    time.sleep(0.4)

            # 3. Write Page Result to JSONL immediately (Memory Efficiency & Crash Safety)
            # page_data = {
            #     "source_pdf": pdf_name,
            #     "page_number": page_num,
            #     "text_blocks": page_blocks,
            #     "total_blocks": len(page_blocks),
            #     "timestamp": time.time()
            # }

            # with open(output_file, 'a', encoding='utf-8') as f_out:
            #     line = json.dumps(page_data, cls=NumpyEncoder, ensure_ascii=False)
            #     f_out.write(line + '\n')

            # print(f" Saved ({len(page_blocks)} blocks)")
            
            # Explicit cleanup
            del page_blocks
            gc.collect()

    print(f"\n✓ OCR Process Complete. Data in: {output_file}")
    logger.error(f"\n✓ OCR Process Complete. Data in: {output_file}")

if __name__ == "__main__":
    main()
