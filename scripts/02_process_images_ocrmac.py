#!/usr/bin/env python3
import json
import os
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ocrmac import ocrmac

# --- CONFIGURATION ---
# M1/M2/M3 chips usually have 8-12 cores. 
# Vision framework is very efficient; 4-6 threads is usually the "sweet spot" 
# to saturate the Neural Engine without choking I/O.
MAX_WORKERS = 6 

def process_snippet(snip, pub_name, page_num):
    """
    Worker function to process a single snippet.
    Returns a list of result dictionaries.
    """
    img_path = snip['path']
    if not os.path.exists(img_path):
        return []

    results = []
    try:
        # The Vision framework manages its own memory efficiently.
        # Calling ocrmac inside the thread allows the OS to parallelize the ANE calls.
        annotations = ocrmac.OCR(img_path).recognize()
        
        x_off = snip.get('x_offset', 0)
        y_off = snip.get('y_offset', 0)
        col = snip.get('column', 0)

        for text, conf, bbox in annotations:
            text = text.strip()
            if text:
                b_x, b_y, b_w, b_h = bbox
                results.append({
                    "pub": pub_name,
                    "page": page_num,
                    "col": col,
                    "text": text,
                    "conf": round(conf, 3),
                    "bbox": [
                        int(b_x + x_off),
                        int(b_y + y_off),
                        int(b_w),
                        int(b_h)
                    ]
                })
    except Exception as e:
        print(f"Error on {img_path}: {e}")
    
    return results

def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    preprocessed_dir = project_root / "data" / "preprocessed"
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = preprocessed_dir / "all_metadata.json"
    if not metadata_path.exists():
        return

    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)

    output_file = output_dir / "ocr_output_vision.jsonl"
    print(f"Starting Optimized Vision OCR with {MAX_WORKERS} threads...")

    with open(output_file, 'w', encoding='utf-8') as f_out:
        # We use a ThreadPoolExecutor for Mac Vision. 
        # Unlike ProcessPool, ThreadPool has lower memory overhead 
        # and plays nicer with Apple's system-level frameworks.
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            
            for pdf_meta in all_metadata:
                pub_name = pdf_meta.get('source_pdf')
                
                for page in pdf_meta.get('pages', []):
                    page_num = page.get('page_num')
                    print(f"Queueing {pub_name} - Page {page_num}...")

                    # Submit all snippets for this page to the thread pool
                    snippets = page.get('snippets', [])
                    future_to_snippet = {
                        executor.submit(process_snippet, snip, pub_name, page_num): snip 
                        for snip in snippets
                    }

                    # Collect results as they finish
                    for future in future_to_snippet:
                        snippet_results = future.result()
                        for entry in snippet_results:
                            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    
                    # Periodic cleanup
                    f_out.flush()
                    gc.collect()

    print(f"\n✓ OCR Complete. Results: {output_file}")

if __name__ == "__main__":
    main()