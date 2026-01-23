#!/usr/bin/env python3
"""
Step 1: Preprocess PDFs (300 DPI) - Memory Efficient Version
- Processes ONE PAGE AT A TIME to avoid OOM in Colab
- Hardened against 'Empty Source' OpenCV errors.
- Bypasses PIL pixel limits for large broadsheets.
- Discards snippets < 250KB (white space filter).
"""

import os
import json
import gc
import cv2
import numpy as np
import re
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image

# 1. BYPASS PILLOW LIMIT
Image.MAX_IMAGE_PIXELS = None 


def detect_vertical_columns(img_gray):
    if img_gray is None or img_gray.size == 0: return [0]
    h, w = img_gray.shape
    if w < 50: return [0, w]

    binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    roi = binary[int(h*0.1):int(h*0.9), :]
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150))
    v_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, v_kernel)
    v_proj = np.sum(v_lines, axis=0)
    v_norm = v_proj / (np.max(v_proj) + 1e-6)
    
    peaks = [x for x in range(1, len(v_norm)-1) if v_norm[x] > 0.1 and v_norm[x] > v_norm[x-1] and v_norm[x] > v_norm[x+1]]
    clean_peaks = []
    min_col_w = w // 12
    if peaks:
        clean_peaks.append(peaks[0])
        for p in peaks[1:]:
            if p > clean_peaks[-1] + min_col_w: clean_peaks.append(p)
    return sorted(list(set([0] + clean_peaks + [w])))
    # After clean_peaks is built:
    #clean_peaks = [p + 10 for p in clean_peaks]  # Nudge boundaries into gutter center

def detect_horizontal_rules(column_gray):
    """Refined skew-tolerant horizontal detection."""
    if column_gray is None or column_gray.size == 0: return [0]
    h, w = column_gray.shape
    if h < 100 or w < 10: return [0, h]
    
    shave = max(1, int(w * 0.05))
    inner = column_gray[:, shave:w-shave] if w > 10 else column_gray
    binary = cv2.adaptiveThreshold(inner, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Use the 'Narrow Beam' (10%) and 'Bridge' (Morph Close) logic
    kernel_w = max(10, int(inner.shape[1] * 0.08))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    
    detected = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    detected = cv2.morphologyEx(detected, cv2.MORPH_CLOSE, h_kernel, iterations=2)
    
    y_proj = np.sum(detected, axis=1)
    max_p = np.max(y_proj)
    if max_p == 0: return [0, h]
    
    peak_thresh = max_p * 0.1 # Lowered threshold for tilted rules
    dividers = []
    y = 0
    while y < h:
        if y_proj[y] > peak_thresh:
            dividers.append(y)
            y += 60 # Skip ahead
        y += 1
    return sorted(list(set([0] + dividers + [h])))


def get_pdf_page_count(pdf_path):
    """Get number of pages without loading the whole PDF."""
    try:
        from pdf2image.pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(str(pdf_path))
        return info.get('Pages', 0)
    except Exception:
        # Fallback: try loading first page to check
        try:
            convert_from_path(str(pdf_path), dpi=72, first_page=1, last_page=1)
            # If that works, try to get count another way
            return 10  # Default guess, will stop when no more pages
        except:
            return 0


def process_single_page(pdf_path, page_num, dpi=300):
    """Convert a single page from PDF to image."""
    try:
        images = convert_from_path(
            str(pdf_path), 
            dpi=dpi,
            first_page=page_num,
            last_page=page_num
        )
        if images:
            return images[0]
    except Exception as e:
        print(f"    Error loading page {page_num}: {e}")
    return None


def main():
    # Detect environment
    try:
        from google.colab import drive
        IN_COLAB = True
        project_root = Path("/content/jan11_exp")
        print("Running in Google Colab")
    except ImportError:
        IN_COLAB = False
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        print("Running locally")

    pdf_dir = project_root / "pdfs"
    output_base = project_root / "data" / "01_preprocessed"
    output_base.mkdir(parents=True, exist_ok=True)

    MIN_KB = 250
    DPI = 300
    MAX_SNIP_HEIGHT = 3400#2000

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs to process")
    
    all_metadata = []

    for pdf_idx, pdf_path in enumerate(pdf_files):
        stem = pdf_path.stem
        match = re.match(r"(\d+)_(\d{4}-\d{2}-\d{2})", stem)
        pub_id, pub_date = match.groups() if match else ("000", "0000-00-00")

        print(f"\n[{pdf_idx+1}/{len(pdf_files)}] Processing: {stem}")
        
        # Get page count
        page_count = get_pdf_page_count(pdf_path)
        print(f"  Detected {page_count} pages")

        pdf_out_dir = output_base / stem
        pdf_out_dir.mkdir(parents=True, exist_ok=True)
        pdf_entry = {"source_pdf": stem, "pub_id": pub_id, "date": pub_date, "pages": []}

        # Process ONE PAGE AT A TIME
        page_num = 1
        consecutive_failures = 0
        
        while consecutive_failures < 3:  # Stop after 3 consecutive failures (end of PDF)
            print(f"  Processing page {page_num}...", end=" ", flush=True)
            
            # Load single page
            page_img = process_single_page(pdf_path, page_num, DPI)
            
            if page_img is None:
                consecutive_failures += 1
                print("(no page)")
                page_num += 1
                continue
            
            consecutive_failures = 0  # Reset on success
            
            # Convert to grayscale
            img_gray = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2GRAY)
            
            ## save full page for reference (optional)
            # page_fn = f"p{page_num:02d}.jpg"
            # cv2.imwrite(str(pdf_out_dir / page_fn), img_gray, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            # Free the PIL image immediately
            del page_img
            gc.collect()
            
            v_bounds = detect_vertical_columns(img_gray)
            print(v_bounds)
            page_snippets = []

            for c_idx in range(len(v_bounds)-1):
                x1, x2 = v_bounds[c_idx], v_bounds[c_idx+1]
                if (x2 - x1) < 15: continue 

                margin = 15  # Tunable
                x1_safe = max(0, x1 - margin)
                x2_safe = min(img_gray.shape[1], x2 + margin)
                column_strip = img_gray[:, x1_safe:x2_safe]
                h_bounds = detect_horizontal_rules(column_strip)
                
                for s_idx in range(len(h_bounds)-1):
                    y1, y2 = h_bounds[s_idx], h_bounds[s_idx+1]
                    raw_snippet = column_strip[y1:y2, :]
                    
                    snip_h = raw_snippet.shape[0]
                    
                    # Split logic for long columns
                    num_parts = (snip_h // MAX_SNIP_HEIGHT) + 1
                    
                    for part_idx in range(num_parts):
                        start_y = part_idx * MAX_SNIP_HEIGHT
                        end_y = min((part_idx + 1) * MAX_SNIP_HEIGHT, snip_h)
                        
                        if start_y >= end_y: continue
                        
                        snippet_part = raw_snippet[start_y:end_y, :]
                        
                        snip_fn = f"{pub_id}_{pub_date}_p{page_num:02d}_c{c_idx:02d}_s{s_idx:03d}_pt{part_idx}.jpg"
                        snip_path = pdf_out_dir / snip_fn
                        
                        cv2.imwrite(str(snip_path), snippet_part, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

                        # 250KB Check
                        if os.path.getsize(snip_path) / 1024 < MIN_KB:
                            os.remove(snip_path)
                            continue

                        page_snippets.append({
                            "path": str(snip_path),
                            "x_offset": int(x1),
                            "y_offset": int(y1 + start_y),
                            "column": int(c_idx)
                        })
            
            pdf_entry["pages"].append({"page_num": page_num, "snippets": page_snippets})
            print(f"{len(page_snippets)} snippets")
            
            # Aggressive cleanup after each page
            del img_gray
            gc.collect()
            
            page_num += 1
            
            # Safety check - don't process more than 100 pages
            if page_num > 100:
                print("  (reached 100 page limit)")
                break

        all_metadata.append(pdf_entry)
        
        # Save metadata incrementally (in case of crash)
        with open(output_base / "all_metadata.json", "w") as f:
            json.dump(all_metadata, f, indent=2)
        print(f"  Saved metadata ({len(all_metadata)} PDFs processed)")

    print(f"\nDone! Processed {len(all_metadata)} PDFs")
    print(f"📁 Output: {output_base / 'all_metadata.json'}")


if __name__ == "__main__":
    main()
