"""
Post-OCR (step 2) sanity checker
Checks for consistency between images and OCR text lines via:
  - OCR data exists for all images listed in the metadata
  - Each page has COLUMNS_PER_PAGE columns
  - The number of lines for each column on a page are similar
  - The pixels per line are similar across for all columns.

Options:
    SKIP_IMAGE_LOADING: If True, skip loading PIL images (faster, but doesn't check image dimensions)
  
"""

import pandas as pd
import json
from pathlib import Path
from PIL import Image
import sys

sys.stdout.reconfigure(encoding='utf-8')

COLUMNS_PER_PAGE = 3
SKIP_IMAGE_LOADING = False
metadata_path=Path("data/01_preprocessed/all_metadata.json")
ocr_path=Path("data/dataset_2026.03.18/04_ocr_output_reviewed_2026.03.18.jsonl")

# Load metadata JSON file with image snippet information
try:
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
except Exception as e:
    print(f"Error loading metadata {metadata_path.name}: {str(e)}")
    exit(1)
    
# Load OCR output JSONL file
try:
    ocr_data = pd.read_json(ocr_path, lines=True)
except Exception as e:
    print(f"Error loading OCR output {ocr_path.name}: {str(e)}")
    exit(1)


print("=" * 80)
print("SANITY CHECK REPORT - IMAGE & OCR CONSISTENCY CHECKS")
print("-" * 80 + "\n")
any_errors = False
any_warnings = False

# Get base directory - use current working directory
base_dir = Path.cwd()

# OCR data exists for all images listed in the metadata
all_pubs_pages_cols_OCRed = True
if len(metadata) != len(ocr_data["pub"].unique()):
    all_pubs_pages_cols_OCRed = False
    print(
        f"\n❌ number of metadata pubs ({len(metadata)})",
        f"doesn't match number of unique OCRed pubs ({len(ocr_data["pub"].unique())})"
    )
    any_errors = True

# Examine all input publications consistency with OCR data
# Store the pixels per line for each page/column to check for consistency later
pixels_per_line_by_page_col = {} # (doc, page, col) -> int
for pub_data in metadata:
    pub_id = pub_data["pub_id"]
    pub_ocr = ocr_data[ocr_data["pub"] == pub_id] # get matching ouptput for this publication

    # check if number of pages is consistent
    if len(pub_data["pages"]) != len(pub_ocr["page"].unique()):
        all_pubs_pages_cols_OCRed = False
        print(
            f"\n❌ in {pub_id}, number of metadata pages {len(pub_data["pages"])},"
            f"doesn't match number of unique OCRed pages {len(pub_ocr["page"].unique())}"
        )
        any_errors = True
    
    # Examine all input pages for consistency with OCR data
    for page_data in pub_data["pages"]:
        page_num = page_data["page_num"]
        snippets = page_data["snippets"]
        
        # Get OCR lines for this page
        page_ocr = pub_ocr[pub_ocr["page"] == page_num]
        
        # Check expected columns
        if len(snippets) != COLUMNS_PER_PAGE:
            print(
                f"\n❌ {pub_id} page {page_num}: unexpected number of columns,",
                f"found {len(snippets)}"
            )
            any_errors = True

        # Check number of columns match
        if len(snippets) != len(page_ocr["col"].unique()):
            all_pubs_pages_cols_OCRed = False
            print(
                f"\n❌ in {pub_id}.{page_data["page_num"]}, number of metadata columns {len(snippets)},"
                f"doesn't match number of unique OCRed pages {len(page_ocr["col"].unique())}"
            )
            any_errors = True
    
        # Examine all columns (snippets) for consistency with OCR data
        # store line counts for each column to check for consistency later
        column_line_counts = {}
        for snippet in snippets:
            col = snippet["column"]
            col_ocr = page_ocr[page_ocr["col"] == col]
            column_line_counts[col] = len(col_ocr)
        
        # Check for column line count consistency across the page
        if column_line_counts and len(column_line_counts) > 1:
            line_counts = list(column_line_counts.values())
            avg_lines = sum(line_counts) / len(line_counts)
            max_deviation = max(abs(count - avg_lines) for count in line_counts)
            
            # Flag if any column differs by more than 8 from average
            if max_deviation > 8:
                print(
                    f"\n❌ {pub_id} page {page_num}: Column lines differ by more than 8: {max_deviation:.1f}",
                    f"(cols: {column_line_counts})"
                )
                any_errors = True
        
        # Analyze image snippets (load images only if requested)
        if not SKIP_IMAGE_LOADING:
            for snippet in snippets:
                img_path = Path(snippet["path"])
                full_path = base_dir / img_path
                col = snippet["column"]
                
                try:
                    # Try to load image to get dimensions, calculate pixels per line
                    if full_path.exists():
                        with Image.open(full_path) as img:
                            width, height = img.size
                            px_per_lines = height // column_line_counts[col]
                            pixels_per_line_by_page_col[(pub_id, page_num, col)] = px_per_lines
                    else:
                        print(f"\n❌ Image file not found: {full_path}")
                        any_errors = True
                except Exception as e:
                    print(f"\n❌ Could not read image: {str(e)}")
                    any_errors = True

# Calculate overall average pixels per line across images and flag any columns that deviate by more than .3x from average
pix_per_line = list(pixels_per_line_by_page_col.values())
avg_pix_per_lines = sum(pix_per_line) / len(pix_per_line)
deviations = {id: pixels for id, pixels in pixels_per_line_by_page_col.items() if abs(pixels - avg_pix_per_lines) > (avg_pix_per_lines*.3)}
if deviations:
    print(
        f"\n⚠ {len(deviations)} columns' pixels per line vary more than .3x from average ({avg_pix_per_lines}):"
        f"\n\t{"\n\t".join(f"{k}={v}" for k, v in deviations.items())}"
    )
    any_warnings = True

if all_pubs_pages_cols_OCRed:
    print("✓ All publications, pages, and columns exist in OCR")

if not any_errors and not any_warnings:
    print("✓ All checks passed!")

print("=" * 80 + "\n")

