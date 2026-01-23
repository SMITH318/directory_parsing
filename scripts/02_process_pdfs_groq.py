#!/usr/bin/env python3
"""
OCR Processing Script - Groq Llama 4 Maverick Edition
Optimized for Groq Cloud API with robust JSON handling.
"""

import json
import gc
import cv2
import numpy as np
from pathlib import Path
import os
import time
import base64
from PIL import Image
from groq import Groq

# --- Groq API Configuration ---
API_KEY = os.getenv('GROQ_API_KEY', 'YOUR_API_KEY')

if API_KEY == 'YOUR_API_KEY' or not API_KEY:
    print("ERROR: Groq API key is not set.")
    exit(1)

client = Groq(api_key=API_KEY)
# Using the specific Maverick model requested
MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.float32, np.float64)):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def groq_ocr(image_path, max_retries=3):
    """
    Extract text and bounding boxes from an image using Groq Maverick.
    """
    img_pil = Image.open(image_path)
    w, h = img_pil.size
    base64_image = encode_image(image_path)

    # Prompt updated to strictly request a JSON OBJECT to satisfy Groq's JSON mode
    prompt = (
        "Extract all text from this newspaper image. "
        "Return a JSON object with a key 'blocks' containing an array of text segments. "
        "Each segment must have: 'text', 'x', 'y', 'width', and 'height'. "
        "Coordinates are in pixels. Format:\n"
        '{"blocks": [{"text": "sample", "x": 0, "y": 0, "width": 50, "height": 20}]}'
    )

    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                model=MODEL_NAME,
                response_format={"type": "json_object"} # Forces valid JSON structure
            )

            response_text = chat_completion.choices[0].message.content.strip()
            data = json.loads(response_text)

            # Robust parsing: ensure we get a list of dicts regardless of how model nested them
            text_blocks = []
            if isinstance(data, dict):
                # Look for common keys models use
                if 'blocks' in data: text_blocks = data['blocks']
                elif 'text_blocks' in data: text_blocks = data['text_blocks']
                elif 'data' in data: text_blocks = data['data']
                else: 
                    # If it's a flat dict, it's a single block; wrap it in a list
                    if 'text' in data: text_blocks = [data]
            elif isinstance(data, list):
                text_blocks = data

            # Validate and clean blocks to prevent 'list indices' errors in main()
            cleaned_blocks = []
            for block in text_blocks:
                # Ensure block is a dictionary and has required keys
                if isinstance(block, dict) and 'text' in block:
                    block['confidence'] = block.get('confidence', 0.95)
                    # Ensure numeric values
                    for key in ['x', 'y', 'width', 'height']:
                        block[key] = float(block.get(key, 0))
                    cleaned_blocks.append(block)
            
            return cleaned_blocks

        except Exception as e:
            print(f"\n    Warning: Groq API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return fallback_text_extraction(base64_image, w, h)
            time.sleep(2)

    return []

def fallback_text_extraction(base64_image, width, height):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this image as raw text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            model=MODEL_NAME,
        )
        txt = chat_completion.choices[0].message.content.strip()
        return [{"text": txt, "x": 0, "y": 0, "width": width, "height": height, "confidence": 0.80}]
    except:
        return []

def main():
    # 1. Setup Project Paths
    script_dir = Path(__file__).parent
    project_root = script_dir if (script_dir / "data").exists() else script_dir.parent
    preprocessed_dir = project_root / "data" / "preprocessed"
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Change extension to .jsonl to reflect the format
    output_file = output_dir / "ocr_output.jsonl"

    metadata_path = preprocessed_dir / "all_metadata.json"
    if not metadata_path.exists():
        print(f"Error: Missing metadata: {metadata_path}")
        return

    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)

    # We no longer need the big 'all_results' list in memory
    print(f"Starting OCR. Writing results to: {output_file}")

    # 2. Process each PDF found in metadata
    for pdf_meta in all_metadata:
        pdf_name = pdf_meta['source_pdf']
        print(f"\nOCRing Snippets for: {pdf_name}")

        for page_meta in pdf_meta['pages']:
            print(f"  Page {page_meta['page_num']}...", end="", flush=True)
            page_blocks = []

            for snip in page_meta['snippets']:
                snippet_path = snip['path']
                if not Path(snippet_path).exists(): continue

                results = groq_ocr(snippet_path)

                for block in results:
                    page_blocks.append({
                        "text": block['text'],
                        "confidence": block['confidence'],
                        "bbox": {
                            "x": block['x'] + snip['x_offset'],
                            "y": block['y'] + snip['y_offset'],
                            "width": block['width'],
                            "height": block['height']
                        },
                        "column": snip.get('col_idx', 0)
                    })
                
                gc.collect()
                time.sleep(0.1)

            # Create the data object for this specific page
            page_data = {
                "filename": pdf_name,
                "page_number": page_meta['page_num'],
                "text_blocks": page_blocks,
                "total_blocks": len(page_blocks)
            }

            # --- JSONL CHANGE: Append this page as a single line immediately ---
            with open(output_file, 'a', encoding='utf-8') as f:
                # Use json.dumps to create a single line string
                line = json.dumps(page_data, ensure_ascii=False, cls=NumpyEncoder)
                f.write(line + '\n')
            
            print(f" Done and Appended ({len(page_blocks)} blocks)")

    print(f"\n✓ OCR Process Complete. Output saved as JSONL.")

if __name__ == "__main__":
    main()