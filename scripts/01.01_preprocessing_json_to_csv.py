"""
Converts the JSON metadata from 01_preprocess.py into a CSV format for easier analysis and integration with other tools.
"""
import json
import csv
import sys
from pathlib import Path
import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    handlers=[
        logging.FileHandler('01.01_preprocessing_json_to_csv.log', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stderr)
    ],
    level=logging.WARNING
)

def main(dataset: str, preprocessed_dir: str = None):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    preprocessed_dir_path = Path(preprocessed_dir) if preprocessed_dir else project_root / "data" / dataset / "01_preprocessed"
    json_path = preprocessed_dir_path / 'all_metadata.json'
    csv_path = preprocessed_dir_path / 'all_metadata.csv'

    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Open CSV file for writing
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['pub_id', 'page_num', 'column', 'path', 'x_offset', 'y_offset']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # Iterate through each publication
        for pub in data:
            pub_id = pub['pub_id']
            
            # Iterate through each page
            for page in pub['pages']:
                page_num = page['page_num']
                
                # Iterate through each snippet (column)
                for snippet in page['snippets']:
                    writer.writerow({
                        'pub_id': pub_id,
                        'page_num': page_num,
                        'column': snippet['column'],
                        'path': snippet['path'],
                        'x_offset': snippet['x_offset'],
                        'y_offset': snippet['y_offset']
                    })

    print(csv_path)
    logger.info("✓ Step completed successfully ({csv_path})")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert metadata JSON to CSV")
    parser.add_argument("dataset", help="Name of the dataset")
    parser.add_argument("--preprocessed", help="Directory for preprocessed images", default=None)
    args = parser.parse_args()
    
    exit_code = main(args.dataset, args.preprocessed)
    sys.exit(exit_code)
