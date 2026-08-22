#!/usr/bin/env python3
"""
Script to pick random snippets from all_metadata.json and optionally extract entries from CSVs
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse
import sys

def load_metadata(metadata_path):
    """Load metadata from JSON file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def get_all_snippets(metadata: list[dict]) -> pd.DataFrame:
    """
    Get all snippets from metadata, preserving context.
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

def extract_and_save_entries(snippets: pd.DataFrame, csv_in: Path, csv_out: Path) -> int:
    """Extract entries from CSVs for all snippets and save to combined CSV."""
    
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
    
    if not all_extracted:
        print(f"  No entries found for any of the sampled snippets")
        return 0

    # sort_cols = ['publication', 'page_number', 'column']
    df = pd.concat(all_extracted)
    # df = df.sort_values(by=sort_cols)
            
    # Save to CSV
    df.to_csv(csv_out, index=False, mode='a', header=not csv_out.exists())
    print(f"  Entries added: {len(df)}, in {csv_out}")
    return len(df)


def main(count_entity: str, dataset: str, config_path: str, target_num=500):
    # Define the data directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / dataset
    metadata_path = data_dir / '01_preprocessed' / 'all_metadata.json'

    # Load JSON schema config
    config_file = Path(config_path)
    with open(config_file, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    
    schema_entities = list(schema.get('properties', {}).keys())
    paths = {
        entity: (
            data_dir / f"09_{entity.lower()}_parsed.csv", 
            data_dir / f"sampled_entries_{entity.lower()}_output.csv"
        )
        for entity in schema_entities
    }

    num_samples = 5 # starting value
    num_so_far = 0
    
    # Load and process
    metadata = load_metadata(metadata_path)
    
    print(f"Extracting snippets...")
    all_snippets = get_all_snippets(metadata)
    print(f"Total snippets found: {len(all_snippets)}")
    
    while num_so_far < target_num:
        # remove already sampled columns
        remaining_snippets = all_snippets.copy()
        if paths[count_entity][1].exists():
            cols_to_match = ['publication', 'page_number', 'column']
            doc_df = pd.read_csv(paths[count_entity][1])[cols_to_match].drop_duplicates()
            merged = remaining_snippets.merge(doc_df, on=cols_to_match, how='left', indicator=True)
            # Keep only rows from the 'left_only' source and drop the helper column
            remaining_snippets = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')
            # print(f"Snippets not previously selected: {len(remaining_snippets)}")

        
        print(f"\nSampling {num_samples} random snippets...")
        sampled = sample_snippets(remaining_snippets, num_samples)
        
        # Print results
        print_snippets(sampled)

        # extract entries and save
        for entity, (entity_csv, entity_output_csv) in paths.items():
            n_extracted = extract_and_save_entries(sampled, entity_csv, entity_output_csv)
            if entity == count_entity:
                num_so_far += n_extracted
        if num_so_far >= target_num//2:
            num_samples = 1
        print(f"Total {count_entity}s extracted so far: {num_so_far}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pick random columns from all_metadata.json and their extract entries from CSVs.")
    parser.add_argument("entity", help="Name of the entity to count extraction on (e.g., 'doc' or 'city')")
    parser.add_argument("dataset", help="Name of the dataset")
    parser.add_argument("--config", help="Path to JSON schema config file", required=True)
    args = parser.parse_args()
    
    exit_code = main(args.entity, args.dataset, args.config)
    sys.exit(exit_code)