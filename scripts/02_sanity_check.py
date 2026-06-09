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

import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    handlers=[
        logging.FileHandler('02_sanity_check.log', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stderr)
    ],
    level=logging.INFO
)

COLUMNS_PER_PAGE = 3
SKIP_IMAGE_LOADING = False

def main(dataset: str, preprocessed_dir: str = None) -> int:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / dataset
    
    preprocessed_dir_path = Path(preprocessed_dir) if preprocessed_dir else data_dir / "01_preprocessed"
    metadata_path = preprocessed_dir_path / "all_metadata.json"
    ocr_path = data_dir / "02_ocr_output.jsonl"
    
    # Load metadata JSON file with image snippet information
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Error loading metadata {metadata_path.name}: {str(e)}")
        sys.exit(1)
        
    # Load OCR output JSONL file
    try:
        ocr_data = pd.read_json(ocr_path, lines=True)
    except Exception as e:
        logger.error(f"Error loading OCR output {ocr_path.name}: {str(e)}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("SANITY CHECK REPORT - IMAGE & OCR CONSISTENCY CHECKS")
    logger.info("-" * 80)
    any_errors = False
    any_warnings = False

    # Get base directory - use current working directory
    base_dir = Path.cwd()

    # OCR data exists for all images listed in the metadata
    all_pubs_pages_cols_OCRed = True
    if len(metadata) != len(ocr_data["pub"].unique()):
        all_pubs_pages_cols_OCRed = False
        logger.error((
            f"❌ number of metadata pubs ({len(metadata)})",
            f"doesn't match number of unique OCRed pubs ({len(ocr_data["pub"].unique())})"
        ))
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
            logger.error(
                f"❌ in {pub_id}, number of metadata pages {len(pub_data["pages"])},"
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
                logger.error((
                    f"❌ {pub_id} page {page_num}: unexpected number of columns,",
                    f"found {len(snippets)}"
                ))
                any_errors = True

            # Check number of columns match
            if len(snippets) != len(page_ocr["col"].unique()):
                all_pubs_pages_cols_OCRed = False
                logger.error((
                    f"❌ in {pub_id}.{page_data["page_num"]}, number of metadata columns {len(snippets)},"
                    f"doesn't match number of unique OCRed pages {len(page_ocr["col"].unique())}"
                ))
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
                    logger.error(
                        f"❌ {pub_id} page {page_num}: Column lines differ by more than 8: {max_deviation:.1f}",
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
                            logger.error(f"❌ Image file not found: {full_path}")
                            any_errors = True
                    except Exception as e:
                        logger.error(f"❌ Could not read image: {str(e)}")
                        any_errors = True

    # Calculate overall average pixels per line across images and flag any columns that deviate by more than .3x from average
    pix_per_line = list(pixels_per_line_by_page_col.values())
    avg_pix_per_lines = sum(pix_per_line) / len(pix_per_line)
    deviations = {id: pixels for id, pixels in pixels_per_line_by_page_col.items() if abs(pixels - avg_pix_per_lines) > (avg_pix_per_lines*.3)}
    if deviations:
        logger.warning((
            f"⚠ {len(deviations)} columns' pixels per line vary more than .3x from average ({avg_pix_per_lines}):"
            f"\n\t{"\n\t".join(f"{k}={v}" for k, v in deviations.items())}"
        ))
        any_warnings = True

    if all_pubs_pages_cols_OCRed:
        logger.info("✓ All publications, pages, and columns exist in OCR")

    if not any_errors and not any_warnings:
        logger.info("✓ All checks passed!")

    logger.info("=" * 80)
    print(ocr_path)

    if not any_errors and not any_warnings:
        return 0
    return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Sanity Check OCR Output")
    parser.add_argument("dataset", help="Name of the dataset")
    parser.add_argument("--preprocessed", help="Directory for preprocessed images", default=None)
    args = parser.parse_args()
    
    sys.exit(main(args.dataset, args.preprocessed))


