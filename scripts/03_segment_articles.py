#!/usr/bin/env python3
"""
Step 3: Article Segmentation
- Reads streaming .jsonl OCR output.
- Groups text blocks into articles based on headlines and vertical gaps.
- Optimized for Tesseract output and 1880s newspaper layouts.
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def extract_date_from_filename(filename):
    """
    Extract date from PDF filename (e.g., 'equity_1888-10-25.pdf' -> '1888-10-25')
    """
    match = re.search(r'(\d{4}[-_]\d{2}[-_]\d{2})', filename)
    if match:
        return match.group(1).replace('_', '-')
    return None

def is_likely_headline(block, avg_height):
    """
    1880s Headline Heuristics.
    """
    text = block["text"].strip()
    if len(text) < 3 or len(text) > 100: 
        return False

    # Heuristic 1: All Caps is the #1 indicator for 19th-century headers
    # (e.g., "LOCAL NEWS", "THE MARKETS", "DIED")
    is_caps = text.isupper()
    
    # Heuristic 2: Large font (if Tesseract detected it correctly)
    is_tall = block["bbox"]["height"] > (avg_height * 1.3)

    # Heuristic 3: Common 1880s sections
    is_common_section = any(word in text.upper() for word in ["NOTICE", "WANTED", "FOR SALE", "BIRTHS", "MARRIED"])

    score = 0
    if is_caps: score += 2
    if is_tall: score += 1
    if is_common_section: score += 1
    
    return score >= 2

def group_into_articles(blocks):
    if not blocks:
        return []

    # Sort by column, then by Y position
    sorted_blocks = sorted(blocks, key=lambda b: (b.get("col", 0), b["bbox"]["y"]))

    # Calculate average line height for the page to use in heuristics
    avg_h = sum(b["bbox"]["height"] for b in sorted_blocks) / len(sorted_blocks)

    articles = []
    current_article = None
    
    for i, block in enumerate(sorted_blocks):
        is_headline = False #is_likely_headline(block, avg_h) # try to get whole column
        
        start_new = False
        if current_article is None:
            start_new = True
        else:
            prev_block = sorted_blocks[i - 1]
            
            # RULE 1: Headline detection
            if is_headline:
                start_new = True
            
            # RULE 2: Column change
            elif block.get("col") != prev_block.get("col"):
                start_new = True
                
            # RULE 3: Vertical Gap (approx 2.5x line height)
            else:
                # y_gap = current_y - (prev_y + prev_height)
                gap = block["bbox"]["y"] - (prev_block["bbox"]["y"] + prev_block["bbox"]["height"])
                if gap > (avg_h * 2.5):
                    start_new = True

        if start_new:
            if current_article:
                articles.append(current_article)

            # Extract first 3 words from snippet text as headline
            text = block["text"].strip()
            words = text.split()
            headline = " ".join(words[:3]) if len(words) >= 3 else text
            # If headline is empty or too short, use the detected headline or fallback
            if not headline or len(headline) < 3:
                headline = block["text"] if is_headline else "Untitled Snippet"

            current_article = {
                "headline": headline,
                "blocks": [block],
                "column": block.get("col"),
                "y_start": block["bbox"]["y"]
            }
        else:
            current_article["blocks"].append(block)

    if current_article:
        articles.append(current_article)
    return articles

def main():
    RETAIN_LINES = True  # Set to True to preserve line breaks in full_text

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Segment OCR output into articles')
    parser.add_argument('--input', type=str, help='Input OCR file name (default: ocr_output_tesseract.jsonl)')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Use command line argument or default
    input_filename = args.input if args.input else "ocr_output.jsonl"
    input_file = project_root / "data" / "02_raw" / input_filename
    output_file = project_root / "data" / "03_processed" / "articles.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    # 1. Load streaming data and group by Page
    print("Loading and grouping OCR data...")
    data_by_page = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            # blocks[bbox] is [x, y, w, h]
            if isinstance(entry["bbox"], list):
                bbox = entry["bbox"]
                entry["bbox"] = {"x": bbox[0], "y": bbox[1], "width": bbox[2], "height": bbox[3]}
            # Create a unique key for each page of each PDF
            page_key = (entry['pub'], entry['page'])
            data_by_page[page_key].append(entry)

    all_articles = []

    # 2. Process each page
    for (pub_name, page_num), blocks in data_by_page.items():
        print(f"Segmenting: {pub_name} - Page {page_num}")

        # Extract date from filename
        extracted_date = extract_date_from_filename(pub_name)

        articles = group_into_articles(blocks)

        for art_idx, art in enumerate(articles, 1):
            if RETAIN_LINES:
                full_text = "\n".join(b["text"] for b in art["blocks"]).strip()
            else:
                full_text = " ".join(b["text"] for b in art["blocks"]).strip()

            # Calculate aggregate bounding box for the whole article
            all_x = [b["bbox"]["x"] for b in art["blocks"]]
            all_y = [b["bbox"]["y"] for b in art["blocks"]]
            all_x_end = [b["bbox"]["x"] + b["bbox"]["width"] for b in art["blocks"]]
            all_y_end = [b["bbox"]["y"] + b["bbox"]["height"] for b in art["blocks"]]

            article_entry = {
                "article_id": f"{pub_name}_p{page_num}_a{art_idx:03d}",
                "source_pdf": pub_name,
                "page_number": page_num,
                "column": art["column"],
                "headline": art["headline"],
                "full_text": full_text,
                "word_count": len(full_text.split()),
                "extracted_date": extracted_date,
                "date": extracted_date,  # For timeline compatibility
                "bbox": {
                    "x": min(all_x),
                    "y": min(all_y),
                    "width": max(all_x_end) - min(all_x),
                    "height": max(all_y_end) - min(all_y)
                }
            }
            all_articles.append(article_entry)

    # 3. Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_articles": len(all_articles),
            "articles": all_articles
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Success! Extracted {len(all_articles)} articles.")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()