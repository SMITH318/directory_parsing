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
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
  filename='01_preprocessor.log', 
  filemode='w', 
  encoding='utf-8', 
  level=logging.WARNING) ## <=================== Change logging level here

# 1. BYPASS PILLOW LIMIT
Image.MAX_IMAGE_PIXELS = None 

def detect_skew_bounds(img_gray):
    """Detect skew center, direction, angle using minAreaRect method."""
    if img_gray is None or img_gray.size == 0: return [0]
    thresh = cv2.threshold(img_gray, 0, 1, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
    coords, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not coords: return [0]
    rect = cv2.minAreaRect(np.vstack(coords))
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = 90 - angle
    return (rect[0], rect[1], angle)

def deskew_image(img_gray, angle):
    """Rotate image to deskew based on detected angle."""
    (h, w) = img_gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def clip_image_to_text(img_gray, skew_rect, padding=10):
    if img_gray is None or img_gray.size == 0: return [0]
    (h, w) = img_gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, skew_rect[2], 1.0)
    box = cv2.boxPoints(skew_rect)
    pts = np.intp(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0
    ys = pts[:, 1]
    xs = pts[:, 0]
    y_min = min(ys) - padding
    y_max = max(ys) + padding
    x_min = min(xs) - padding
    x_max = max(xs) + padding
    return img_gray[y_min:y_max, x_min:x_max]

def detect_horizontal_rows(img_gray):
    if img_gray is None or img_gray.size == 0: return [0]
    h, w = img_gray.shape
    if h < 50: return [h, 0]

    binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    roi = binary[:, int(w*0.05):int(w*0.95)] 
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
    h_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, h_kernel)
    h_proj = np.sum(h_lines, axis=0)
    h_norm = h_proj / (np.max(h_proj) + 1e-6)
    
    peaks = [y for y in range(1, len(h_norm)-1) if h_norm[y] > 0.1 and h_norm[y] > h_norm[y-1] and h_norm[y] > h_norm[y+1]]
    print(f"\n\t\tDetected horizontal peaks at: {peaks}", end=" ")
    clean_peaks = []
    min_col_h = h // 12
    if peaks:
        clean_peaks.append(peaks[0])
        for p in peaks[1:]:
            if p > clean_peaks[-1] + min_col_h: clean_peaks.append(p)
    return sorted(list(set([0] + clean_peaks + [h])))

def detect_vertical_columns(img_gray, outer_pixels_to_ignore, dest_path):
    if img_gray is None or img_gray.size == 0: return [0]
    h, w = img_gray.shape
    if w < 50: return [0, w]

    binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    #cv2.imwrite(str(dest_path)[:-4]+"_01_adaptive.jpg", binary, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    roi = binary[int(h*0.1):int(h*0.9), :]
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150))
    v_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, v_kernel) # iterations dont help here
    #cv2.imwrite(str(dest_path)[:-4]+"_02_morph_open.jpg", v_lines, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    v_proj = np.sum(v_lines, axis=0)
    v_norm = v_proj / (np.max(v_proj) + 1e-6)
    
    #peaks = [x for x in range(1, len(v_norm)-1) if v_norm[x] > 0.1 and v_norm[x] > v_norm[x-1] and v_norm[x] > v_norm[x+1]]
    peaks = [x for x in range(len(v_norm)) if v_norm[x] > np.percentile(v_norm, 95)]
    peaks = [p for p in peaks if p > outer_pixels_to_ignore and p < (w - outer_pixels_to_ignore)]
    #logger.debug(f"    Vertical projection peaks before cleaning: {peaks}")
    clean_peaks = []
    min_col_w = w // 12
    if peaks:
        clean_peaks.append(peaks[0])
        for p in peaks[1:]:
            if p > clean_peaks[-1] + min_col_w: clean_peaks.append(p)
    return sorted(list(set([0] + clean_peaks + [w])))
    # After clean_peaks is built:
    #clean_peaks = [p + 10 for p in clean_peaks]  # Nudge boundaries into gutter center

def detect_horizontal_rules(column_gray, outer_pixels_to_ignore):
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
    
    peak_thresh = max_p * 0.1 # Lowered threshold for tilted rules - TODO: remove? make more like vertical?
    dividers = []
    y = outer_pixels_to_ignore
    while y < h - outer_pixels_to_ignore:
        if y_proj[y] > peak_thresh:
            dividers.append(y)
            y += 60 # Skip ahead
        y += 1
    return sorted(dividers)


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
    # try:
    #     from google.colab import drive
    #     IN_COLAB = True
    #     project_root = Path("/content/jan11_exp")
    #     print("Running in Google Colab")
    # except ImportError:
    IN_COLAB = False
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    print("Running locally")

    ################### Constant/paths start here ##########################
    pdf_dir = project_root / "pdfs" #/ "problems"
    output_base = project_root / "data" / "01_preprocessed"
    output_base.mkdir(parents=True, exist_ok=True)

    SAVE_DIAGNOSTIC_IMAGES = False
    MIN_KB = 0.8
    DPI = 300
    DESCEW_PAGES = False
    CLIP_PAGES = False # clipping screws up most pages
    CLIP_PADDING = 5
    OUTER_PIXELS_TO_IGNORE = 40
    EXPECTED_ROWS = 1
    EXPECTED_COLS = 3

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs to process")
    logger.info(f"Found {len(pdf_files)} PDFs to process")
    
    all_metadata = []

    for pdf_idx, pdf_path in enumerate(pdf_files):
        stem = pdf_path.stem
        match = re.match(r"(\d+)_(\d{4}-\d{2}-\d{2})", stem)
        pub_id, pub_date = match.groups() if match else (stem, "0000-00-00")

        print(f"\n[{pdf_idx+1}/{len(pdf_files)}] Processing: {stem}")
        logger.info(f"[{pdf_idx+1}/{len(pdf_files)}] Processing PDF: {stem}")
        
        # Get page count
        page_count = get_pdf_page_count(pdf_path)
        print(f"  Detected {page_count} pages")
        logger.debug(f"  Detected {page_count} pages")

        pdf_out_dir = output_base / stem
        pdf_out_dir.mkdir(parents=True, exist_ok=True)
        pdf_entry = {"source_pdf": stem, "pub_id": pub_id, "date": pub_date, "pages": []}

        # Process ONE PAGE AT A TIME
        page_num = 1
        consecutive_failures = 0
        
        while consecutive_failures < 3:  # Stop after 3 consecutive failures (end of PDF)
            print(f"  Processing page {page_num}...", end=" ", flush=True)
            logger.debug(f"  Processing page {page_num}")
            
            # Load single page
            page_img = process_single_page(pdf_path, page_num, DPI)
            
            if page_img is None:
                consecutive_failures += 1
                print("(no page)")
                if page_num > page_count:
                    logger.debug(f"    Reached end of PDF at page {page_num}.")
                else:
                    logger.warning(f"    Failed to load page {page_num} in {stem}")
                page_num += 1
                continue
            
            consecutive_failures = 0  # Reset on success
                        
            # Convert to grayscale
            img_gray = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2GRAY)
            print(f"({img_gray.shape[1]} by {img_gray.shape[0]})", end=" ")
            logger.debug(f"    Page {page_num} size: {img_gray.shape[1]}x{img_gray.shape[0]}")

            # Free the PIL image immediately
            del page_img
            gc.collect()
       
            # save page for reference (optional)
            if SAVE_DIAGNOSTIC_IMAGES:
                page_fn = f"{pub_id}_p{page_num:02d}.jpg"
                page_path = pdf_out_dir / page_fn
                cv2.imwrite(str(page_path), img_gray, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            # try descewing
            if DESCEW_PAGES or CLIP_PAGES:
                skew_rect = detect_skew_bounds(img_gray)
                angle = skew_rect[2]
                if DESCEW_PAGES and abs(angle) > 0.1: # only deskew if significant - do we want this??
                    img_gray = deskew_image(img_gray, angle)
                    print(f"(deskewed {angle:.2f}°)", end=" ")
                    logger.debug(f"    Deskewed page {page_num} by {angle:.2f} degrees")
                if CLIP_PAGES:
                    img_gray = clip_image_to_text(img_gray, skew_rect, padding=CLIP_PADDING)
                    print("(clipped)", end=" ")
                    logger.debug(f"    Clipped page {page_num} to text area")
            
            # Detect rows
            #h_bounds = detect_horizontal_rows(img_gray)
            h_bounds = detect_horizontal_rules(img_gray, OUTER_PIXELS_TO_IGNORE)
            # print(f"\n\tFound horizontal rules at: {h_bounds}")
            logger.debug(f"    Detected horizontal bounds at: {h_bounds}")

            # TODO: consider picking rows based on page number and location
            if page_num == 1 and len(h_bounds) > EXPECTED_ROWS:
                # Pick last row on first page
                h_bounds = h_bounds[-2:]
                logger.info(f"    Removed all rows but last on page {page_num} of {stem}")
            
            # check expected rows
            if len(h_bounds) - 1 != EXPECTED_ROWS:
                logger.warning(f"    Expected {EXPECTED_ROWS} rows, but found {len(h_bounds) - 1} in page {page_num} of {stem}")

            # TODO: consider looking for columns across entire page
            # v_page_bounds = detect_vertical_columns(img_gray)
            # print(f"\tFound vertical rules at: {v_page_bounds}")
            # logger.debug(f"    Detected vertical bounds at page level: {v_page_bounds}")

            # Detect columns within each row and extract snippets
            page_snippets = []
            for r_idx in range(len(h_bounds)-1):
                y1, y2 = h_bounds[r_idx], h_bounds[r_idx+1]
                if (y2 - y1) < 15: continue 

                margin = 0  # Tunable
                y1_safe = max(0, y1 - margin)
                y2_safe = min(img_gray.shape[0], y2 + margin)
                row_strip = img_gray[y1_safe:y2_safe, :]

                # save row strip for reference (optional)
                row_fn = f"{pub_id}_p{page_num:02d}_r{r_idx:02d}.jpg"
                row_path = pdf_out_dir / row_fn
                if SAVE_DIAGNOSTIC_IMAGES:
                    cv2.imwrite(str(row_path), row_strip, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

                v_bounds = detect_vertical_columns(row_strip, OUTER_PIXELS_TO_IGNORE, row_path)
                # print(f"\tIn row {r_idx}, found vertical rules at: {v_bounds}")
                logger.debug(f"    In row {r_idx}, detected vertical bounds at: {v_bounds}")
                if len(v_bounds) - 1 != EXPECTED_COLS:
                    logger.warning(f"    Expected {EXPECTED_COLS} columns, but found {len(v_bounds) - 1} in row {r_idx} of page {page_num} in {stem}")
                
                for c_idx in range(len(v_bounds)-1):
                    x1, x2 = v_bounds[c_idx], v_bounds[c_idx+1]
                    row_snippet = row_strip[:, x1:x2]
                                            
                    snip_fn = f"{pub_id}_p{page_num:02d}_r{r_idx:02d}_c{c_idx:03d}.jpg"
                    snip_path = pdf_out_dir / snip_fn

                    cv2.imwrite(str(snip_path), row_snippet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

                    # Size Check to remove tiny snippets
                    if os.path.getsize(snip_path) / 1024 < MIN_KB:                        
                        logger.warning(f"    Removed tiny snippet: {snip_fn} ({os.path.getsize(snip_path) / 1024:.2f} KB)")
                        os.remove(snip_path)
                        continue

                    page_snippets.append({
                        "path": str(snip_path),
                        "x_offset": int(x1),
                        "y_offset": int(y1_safe),
                        "column": int(c_idx)
                    })
            
            pdf_entry["pages"].append({"page_num": page_num, "snippets": page_snippets})
            print(f"\t{len(page_snippets)} snippets")
            logger.info(f"    Page {page_num} produced {len(page_snippets)} snippets")
            
            # Aggressive cleanup after each page
            del img_gray
            gc.collect()
            
            page_num += 1
            
            # Safety check - don't process more than 100 pages
            if page_num > 150:
                print("  (reached 150 page limit)")
                logger.error(f"    Reached 150 page limit for {stem}, stopping further processing.")
                break

        all_metadata.append(pdf_entry)
        
        # Save metadata incrementally (in case of crash)
        with open(output_base / "all_metadata.json", "w") as f:
            json.dump(all_metadata, f, indent=2)
        print(f"  Saved metadata ({len(all_metadata)} PDFs processed)")
        logger.info(f"  Saved metadata after processing {stem} ({len(all_metadata)} PDFs processed)")

    print(f"\nDone! Processed {len(all_metadata)} PDFs")
    logger.info(f"Finished processing {len(all_metadata)} PDFs.")
    print(f"📁 Output: {output_base / 'all_metadata.json'}")
    logger.info(f"Output metadata at {output_base / 'all_metadata.json'}")


if __name__ == "__main__":
    main()
