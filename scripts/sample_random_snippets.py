#!/usr/bin/env python3
"""
Script to pick random snippets from all_metadata.json and optionally extract entries from CSVs
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict


def load_metadata(metadata_path):
    """Load metadata from JSON file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def extract_all_snippets(metadata: list[dict]) -> pd.DataFrame:
    """
    Extract all snippets from metadata, preserving context.
    Returns a list of dicts with snippet info and context.
    """
    snippets = []
    
    for publication in metadata:
        source_pdf = publication.get('source_pdf', 'Unknown')
        pub_id = publication.get('pub_id', 'Unknown')
        
        for page in publication.get('pages', []):
            page_num = page.get('page_num', 0)
            
            for snippet in page.get('snippets', []):
                snippet_data = {
                    'publication': pub_id,
                    'page_number': page_num,
                    'path': snippet.get('path'),
                    'x_offset': snippet.get('x_offset'),
                    'y_offset': snippet.get('y_offset'),
                    'column': snippet.get('column'),
                }
                snippets.append(snippet_data)
    
    return pd.DataFrame(snippets)


def sample_snippets(snippets, num_samples):
    """Randomly sample snippets without replacement."""
    if num_samples > len(snippets):
        print(f"Warning: Requested {num_samples} samples but only {len(snippets)} snippets available.")
        num_samples = len(snippets)
    
    return snippets.sample(num_samples)


def print_snippets(snippets):
    """Pretty print snippets."""
    for i, snippet in snippets.iterrows():
        print(f"\n{'='*80}")
        print(f"Sample {i} -- {snippet['publication']} Page: {snippet['page_number']}, Column: {snippet['column']}")
        print(f"Path: {snippet['path']}")


def extract_entries_for_snippet(
        df: pd.DataFrame, 
        snippet: dict[str, any]
    ) -> pd.DataFrame:
    """Extract entries from dfs matching the snippet's publication, page, and column."""
    pub_id = snippet.publication
    page_num = snippet.page_number
    column = snippet.column
    
    # Filter by publication, page_number, and column
    return df[
        (df['publication'] == pub_id) &
        (df['page_number'] == page_num) &
        (df['column'] == column)
    ]

def extract_and_save_entries(snippets, csv_in_out:list[tuple[Path,Path]]):
    """Extract entries from CSVs for all snippets and save to combined CSV."""
    
    for csv_in, csv_out in csv_in_out:
        print(f"\nExtracting from {csv_in.name}")
        df = pd.read_csv(csv_in)
    
        all_extracted = []
        for snippet in snippets.itertuples():
            filtered_df = extract_entries_for_snippet(df, snippet)
            
            if not filtered_df.empty:
                print(
                    f"  Found {len(filtered_df)} entries in for",
                    f"{snippet.publication} page {snippet.page_number} column {snippet.column}"
                )
                all_extracted.append(filtered_df)
        
        sort_cols = ['publication', 'page_number', 'column']
        df = pd.concat(all_extracted)
        df = df.sort_values(by=sort_cols)
                
        # Save to CSV
        df.to_csv(csv_out, index=False, mode='a')
        print(f"  Total entries: {len(df)}, in {csv_out}")


def main(num_samples=5):
    
    data_dir = Path(__file__).parent.parent / 'data'
    metadata_path = data_dir / '01_preprocessed' / 'all_metadata.json'
    csv_dir = Path(data_dir) / '04_extracted_entries_gemini_2026.03.18'
    doc_csv = csv_dir / 'sampled_entries_docs_output.csv'
    city_csv = csv_dir / 'sampled_entries_cities_output.csv'

    # Load and process
    metadata = load_metadata(metadata_path)
    
    print(f"Extracting snippets...")
    all_snippets = extract_all_snippets(metadata)
    print(f"Total snippets found: {len(all_snippets)}")

    # remove already sampled columns
    if doc_csv.exists():
        cols_to_match = ['publication', 'page_number', 'column']
        doc_df = pd.read_csv(doc_csv)[cols_to_match].drop_duplicates()
        merged = all_snippets.merge(doc_df, on=cols_to_match, how='left', indicator=True)
        # Keep only rows from the 'left_only' source and drop the helper column
        all_snippets = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')
        print(f"Snippets not previously selected: {len(all_snippets)}")

    
    print(f"\nSampling {num_samples} random snippets...")
    sampled = sample_snippets(all_snippets, num_samples)
    
    # Print results
    print_snippets(sampled)

    # If extraction mode, extract entries and save combined CSV
    extract_and_save_entries(
        sampled, 
        [
            (csv_dir / "amd_1918_city_entries_sorted_deduped.csv", city_csv),
            (csv_dir / "amd_1918_doc_entries_sorted_deduped.csv", doc_csv)
        ]
    )

if __name__ == '__main__':
    main()
