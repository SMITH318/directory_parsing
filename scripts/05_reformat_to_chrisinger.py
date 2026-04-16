#!/usr/bin/env python3
"""
Reformat AMD 1918 doc entries to match Louisiana extract format.

Source format: amd_1918_doc_entries.csv (OCR extracted entries with coordinates)
Target format: louisiana extract - physician_dataset_1906_AMD.csv (processed/linked dataset)
"""

import pandas as pd
import os
from pathlib import Path

def convert_year_to_4_digit(year_str):
    """Convert 2-digit year to 4-digit year, assuming 1900s."""
    if pd.isna(year_str) or year_str == '':
        return None
    
    year_str = str(year_str).strip().strip("'\"")  # Remove whitespace and quotes
    if len(year_str) == 2 and year_str.isdigit():
        year_val = int(year_str)
        if year_val < 0:
            return year_val
        if year_val <= 18:  
            return 1900 + year_val
        return 1800 + year_val
    elif len(year_str) == 4 and year_str.isdigit():
        return int(year_str)
    else:
        return None

def parse_school_code(school_str):
    """Extract state code, school number, and graduation year from school field like 'Ala.4,\'11'"""
    if pd.isna(school_str) or school_str == '':
        return None, None, None
    
    school_str = str(school_str).strip()
    # Format is "STATE.NUM,'YY" - split on comma first to separate school code from year
    parts = school_str.split(',')
    state_code = None
    school_num = None
    grad_year = None
    
    if len(parts) >= 1:
        state_part = parts[0].strip()
        # Extract state abbreviation and number
        # Format: "Ala.4", "La.1", etc.
        if '.' in state_part:
            # Use rsplit with maxsplit=1 to handle states with multiple dots
            state_code, school_num = state_part.rsplit('.', 1)
            state_code = state_code.upper()
    
    # Extract year from second part
    if len(parts) >= 2:
        year_part = parts[1].strip().strip("'\"")  # Remove quotes
        if year_part:
            grad_year = convert_year_to_4_digit(year_part)
    
    return state_code, school_num, grad_year


def reformat_amd_1918(input_csv, output_csv, city_csv=None):
    """
    Reformat AMD 1918 doc entries.
    
    Parameters:
    -----------
    input_csv : str
        Path to amd_1918_doc_entries.csv
    output_csv : str
        Path to output reformatted CSV
    city_csv : str, optional
        Path to amd_1918_city_entries.csv for city/county lookup
    """
    
    # Read the AMD 1918 data
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv, low_memory=False)
    
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    # print(df.info())
    
    # Load city entries for county/city information
    city_df = None
    if city_csv and Path(city_csv).exists():
        print(f"Reading city entries from {city_csv}...")
        city_df = pd.read_csv(city_csv)
        print(f"City entries shape: {city_df.shape}")
        # Create a mapping from entry_id to city info
        city_map = city_df[['entry_id', 'name', 'county_name']].set_index('entry_id').to_dict('index')
    else:
        city_map = {}
    
    # Create new dataframe with Louisiana extract format
    output_df = pd.DataFrame()
    
    # Map/create columns from AMD 1918 to Louisiana extract format
    output_df['unique.id.normed'] = df['publication'].str.upper() + '_' + df['entry_id'].astype(str).str[5:]
    output_df['unique.id'] = output_df['unique.id.normed']
    output_df['physician.name'] = df['name']
    
    # Race columns - use 'col' field which indicates if flagged as colored/Black
    output_df['race.black'] = (df['col'] == "True").astype(int)
    output_df['race.black.prob'] = output_df['race.black']
    
    # Location columns
    # Join with city entries using city_id -> entry_id mapping
    city_names = []
    county_names = []
    for city_id in df['city_id']:
        if city_id in city_map:
            city_names.append(city_map[city_id]['name'])
            county_names.append(city_map[city_id]['county_name'])
        else:
            city_names.append('')
            county_names.append('')
    
    output_df['orig.state'] = df['publication'].str.upper()
    output_df['corr.place.state'] = df['publication'].str.upper()
    output_df['orig.place.state'] = df['publication'].str.upper()
    output_df['orig.city'] = city_names
    output_df['orig.county'] = county_names
    output_df['linked.county.name'] = ''
    output_df['linked.county.GISJOIN2'] = ''
    output_df['linked.county.ICPSRST'] = ''
    output_df['linked.county.ICPSRCTY'] = ''
    
    # Parse school information from 'schools' column
    school_state_list = []
    school_number_list = []
    grad_year_list = []
    for school_str in df['schools']:
        state, num, year = parse_school_code(school_str)
        school_state_list.append(state if state else 'NA')
        school_number_list.append(num if num else 'NA')
        grad_year_list.append(year if year else '')
    
    output_df['school.state.edit'] = school_state_list
    output_df['school.number'] = school_number_list
    output_df['graduation.date'] = grad_year_list
    
    # AMA and license information
    output_df['ama.member'] = df['AMA_member'].astype(int)
    output_df['license.date'] = df['license_year'].astype(str).apply(lambda x: convert_year_to_4_digit(x) if x != 'NA' else None)
    output_df['not.in.practice'] = (df['not_in_practice'] == "True").astype(int)
    
    # Location coordinates - not available in source
    output_df['doc.lat'] = ''
    output_df['doc.lon'] = ''
    
    # Match information
    output_df['match.source'] = ''
    output_df['match.type'] = 'direct_extraction'
    
    # School information - parsed from school code
    school_code_list = []
    for state, num in zip(school_state_list, school_number_list):
        if state != 'NA' and num != 'NA':
            school_code_list.append(f"{state}.{num}")
        else:
            school_code_list.append('NA')
    
    output_df['school.code'] = school_code_list
    output_df['school.name'] = ''  # Would need lookup table
    output_df['school.city'] = ''
    output_df['school.state'] = school_state_list
    output_df['school.black'] = None
    output_df['school.fraudulent'] = None
    output_df['school.lat'] = ''
    output_df['school.lon'] = ''
    output_df['school.dist.km'] = ''

    # columns not included in Chrisinger :
    # birth_year,AMA_fellow,address,office,hours,societies,specialty,military,other_info
    birth_list = []
    for year in df['birth_year'].astype(str).apply(lambda x: convert_year_to_4_digit(x) if x and x != 'NA' else None):
        birth_list.append(year if year else '')
    output_df['birth.date'] = birth_list
    output_df['ama.fellow'] = (df['AMA_fellow'] == "True").astype(int)
    output_df['address'] = df['address']
    output_df['office.address'] = df['office']
    output_df['hours'] = '="' + df['hours'].str.replace(';', ':') + '"'
    output_df['society.memberships'] = df['societies']
    output_df['specialties'] = df['specialty'].str.replace('★', '*')
    output_df['medical.corps'] = df['military'].replace({'▼':'Army Reserve', '▼G': 'National Guard', '▼N': 'Navy Reserve'})
    output_df['misc.info'] = df['other_info']
    
    # Save output
    print(f"Writing to {output_csv}...")
    output_df.to_csv(output_csv, index=False)
    
    print(f"Output shape: {output_df.shape}")
    print(f"Output columns: {output_df.columns.tolist()}")
    print(f"✓ Reformatting complete!")
    print(f"Output file: {output_csv}")


if __name__ == "__main__":
    # Define paths
    project_root = Path(__file__).parent
    project_root = project_root if (project_root / "data").exists() else project_root.parent
    DATA_SET = "2026.03.18"
    input_file = project_root / "data" / f"04_extracted_entries_gemini_{DATA_SET}" / "amd_1918_doc_entries.csv"
    city_file = project_root / "data" / f"04_extracted_entries_gemini_{DATA_SET}" / "amd_1918_city_entries.csv"
    output_file = project_root / "data" / f"05_reformatted_entries_{DATA_SET}" / "amd_1918_reformatted.csv"
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        exit(1)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    
    # Run reformatting
    reformat_amd_1918(str(input_file), str(output_file), str(city_file) if city_file.exists() else None)
