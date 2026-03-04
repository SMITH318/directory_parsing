import json
import numpy as np
from pathlib import Path
from google import genai
from google.genai import types
import pandas as pd
from  pydantic import BaseModel
import os
import time
import csv
from collections import defaultdict
import datetime
from typing import Literal

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
  filename='04_parse_entries_gemini.log', 
  filemode='a', 
  encoding='utf-8', 
  level=logging.INFO) ## <=================== Change logging level here


# type definitions for JSON schemas
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.float32, np.float64)):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

class Entry(BaseModel):
    publication: str
    page_number: int
    column: int
    x: int
    y: int
    width: int
    height: int
    entry_id: str
    name: str

class DocEntry(Entry):
    AMA_member: Literal["True", "False"]
    col: Literal["True", "False"]
    birth_year: int = None
    AMA_fellow: Literal["True", "False"]
    schools: str = None #(state.school_number,'grad_year)
    license_year: int = None # form (l XX)
    not_in_practice: Literal["True", "False"]
    address: str = None
    office: str = None
    city_id: str # the city's entry_id
    hours: str = None
    societies: str = None
    specialty: str = None
    military: str = None
    other_info: str = None

class CityEntry(Entry):
    population: int = None # if present
    county_name: str
    state_name: str
    post_reference: str = None
    post_reference_type: str = None # RD, PO, etc.

class ClassifiedEntries(BaseModel):
    doc_entries: list[DocEntry]
    city_entries: list[CityEntry]

# --- Gemini API Configuration --- 
API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY')
model_name ='gemini-3.1-flash-lite-preview'#'gemini-2.5-flash'#'gemini-3.1-pro-preview'#'gemini-flash-latest'

if API_KEY == 'YOUR_API_KEY' or not API_KEY:
    print("ERROR: Gemini API key is not set.")
    logger.error("ERROR: Gemini API key is not set.")
    exit(1)

# Initialize Gemini
print("Initializing Gemini for OCR...")
#print(f"Key: {API_KEY}")
logger.info(f"Initializing Gemini for OCR... with {model_name}")
client = genai.Client(api_key=API_KEY)

MAX_ENTRIES_SENT = 10
MODEL_PROMPT = (
    "Parse these ordered entries from a medical dirctory, each line is a complete entry. "
    "Entries are contained within a single column of a page and publication. "
    "Entries can be: (1) names of states or provinces; " 
    "(2) cities with their county and often with their population; or "
    "(3) doctors including a wide range of informatiom about them "
    "and their careers, including their birth year, education, address, " 
    "office address, any specialty, among others. "
    "Entries following a state entry are part of that state until another state "
    "entry is reached. Similarly, entries following a city entry are part of that "
    "city until another city entry is reached."
    "For each entry, I'm providing the name of the publication, "
    "the page number, the column number, the entry type, its text, an id, and "
    "a bounding box represented by an x and y coordinate, a height, and a width. "
    "Return ONLY one JSON object that is made up of two JSON arrays, "
    "one of the doctor entries and one of the city entries. "
    "For every entry, keep existing information (including publication, page number, " 
    "column number, id, and its bounding box) and leave it unchanged. "
    "These values have no impact on the other, following fields. "
    "City entries consist of the name of the city, sometimes followed by postal information "
    "in parentheses consiting of the type of postal reference (R.D., P.O., etc.) and its name. "
    "Most cities then are followed by a population number, and all end with the name of the county."
    "Provide the state name for each city based on the last state entry encountered. "
    "Doctor entries can include a number of different elements, but most have only "
    "the doctor's name, their medical school information (represented as a state abbreviation, " 
    "an ID number, and a 2-digit graduation year or as ◊ if the information is missing), "
    "and a 2-digit license year in parentheses after an l or I. "
    "If the license year is unknown, a t appears in place of a year; "
    "encode it as -2. A ♁ can appear in parentheses, instead of licensing "
    "information; encode it as -1 for the license year. "
    "For most doctors, a 2-digit birth year in the format (b'YY) appears. "
    "Leave all years as 2 digits. "
    "Provide the city ID for each doctor based on the last city entry encountered. "
    "⊕ indicates the doctor is an AMA fellow, and their name in capitals indicates AMA membership. "
    "Doctors' entries that include (col), (col.), or near variations - and only those entries - "
    "should have their col value set as True. All others should be false. "
    "Addresses can be street adresses, building names, P.O. boxes, or R.D. routes. "
    "Some doctors are members of societies, which are listed in parentheses and "
    "encoded as capital letters A-G followed by one- and two-digit numbers. "
    "Specialties are short abbreviations that can be follow by either * or ★. "
    "Specialty abbreviations are S, Ob, G, ObG, Or, Pr, Op, A, LR, ALR, OALR, U, D, Pd, N, "
    "P, NP, I (sometimes appears as l), T, Anes, CP, R, Path, Bact, and PH. "
    "▼ and any following N or G indicates a military commission. Membership in federal"
    "organizations like the U.S.P.H.S. should also be recorded in the military field. "
    "Put any extra information that does not fit in another field into the other_info field. "
    "For string/text fields, maintain the UTF-8 character encoding, "
    "retaining all text characters as they are. "
)

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Set up file paths
input_file = project_root / "data" / "03_processed_batch" / "entries_segmented_gemini_2pgs.csv"
output_dir = project_root / "data" / f"04_extracted_entries_gemini_2pgs_{model_name}"
temp_file = output_dir / "temp.csv"
output_file = output_dir / "amd_1918_doc_entries.csv" 
cities_file = output_dir / "amd_1918_city_entries.csv"
prompts_file = output_dir / "extracted_entries_prompts.jsonl"
responses_file = output_dir / "extracted_entries_responses.jsonl"
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Load the data, group by column
if not input_file.exists():
    print(f"Error: {input_file} not found.")
    exit(1)

def make_int_formatter(format_str):
    return lambda str_in : format(str_in, format_str)

all_inputs = pd.read_csv(input_file, encoding="utf-8")
all_inputs["entry_id"] = all_inputs.apply(
    lambda row: 
        f'{row["publication"][:4]}_{row["page_number"]:03d}_{row["column"]:02d}_{row.name:05d}',
    axis = 1
)
grouped_inputs = all_inputs.groupby(["publication", "page_number", "column"])

# create file, write headers for CSVs
with open(output_file, 'w', encoding='utf-8', newline='') as doc_csv:
    doc_writer = csv.DictWriter(doc_csv, DocEntry.model_fields.keys(), restval="")
    doc_writer.writeheader()
    
with open(cities_file, 'w', encoding='utf-8', newline='') as city_csv:
    city_writer = csv.DictWriter(city_csv, CityEntry.model_fields.keys(), restval="")
    city_writer.writeheader()

prev_city = None
for group_name, group_df in grouped_inputs:
    # limit to MAX_ENTRIES_SENT
    next_entry = 0
    while next_entry < len(group_df):
        
        # 2. Save df and upload it
        last_entry = min(next_entry + MAX_ENTRIES_SENT, len(group_df))
        current_df = group_df.iloc[next_entry:last_entry]
        current_df.to_csv(temp_file, index=False, encoding="utf-8")# trying index=False 3/3
        # time.sleep(1) # make sure file has time to be written
        
        # upload blocks file
        file = client.files.upload(
            file=temp_file,
            config=types.UploadFileConfig(mime_type='text/csv')#(mime_type='text/csv; charset=UTF-8')
        )

        # 3. Send parsing prompt
        logger.info(f"prompting for {group_name} ({next_entry}:{last_entry}) at {datetime.datetime.now()}")
        print(f"prompting for {group_name} ({next_entry}:{last_entry}) at {datetime.datetime.now()}")

        try:
            # save prompt for tuning
            with open(temp_file, 'r', encoding='utf-8') as f:
                file_text = f.read()
            with open(prompts_file, 'a') as f:
                f.write(json.dumps(
                    {
                        "systemInstruction" : types.Content(role="system", parts = [types.Part(text=MODEL_PROMPT)]).model_dump(exclude_none=True),
                        "contents": types.UserContent([
                            types.Part(text=f'id of last city encountered was {prev_city["entry_id"]}'), 
                            types.Part(text=f'name of last state encountered was {prev_city["state_name"]}'), 
                            types.Part(text=file_text)
                        ] if prev_city else [file_text]).model_dump(exclude_none=True)
                    }
                )+ '\n')
            #response_stream = client.models.generate_content_stream(
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    f'id of last city encountered was {prev_city["entry_id"]}', 
                    f'name of last state encountered was {prev_city["state_name"]}', 
                    file
                ] if prev_city else [file],
                config=types.GenerateContentConfig(
                    system_instruction=MODEL_PROMPT,
                    temperature=0.0,
                    #thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"), # minimal thinking hurt results 3/3
                    response_mime_type="application/json", 
                    response_json_schema=ClassifiedEntries.model_json_schema(),
                )
            )
            # 3. collect streaming responses
            # response = ""
            # logger.debug("="*20)
            # for chunk in response_stream:
            #     chunk_text = chunk.candidates[0].content.parts[0].text
            #     print(chunk_text)
            #     logger.debug(chunk_text)
            #     response += chunk_text
            # logger.debug("="*20)
            # response_text = response.strip()
            response_text = response.text.strip()
            logger.debug(response_text)

            # Clean markdown formatting if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
                response_text = response_text.replace('```json', '').replace('```', '').strip()

            json_entries = json.loads(response_text)
            with open(responses_file, 'a') as f:
                f.write(json.dumps(json_entries) + '\n')
            doc_entries = json_entries["doc_entries"]
            city_entries = json_entries["city_entries"]

            logger.info(f"received {len(doc_entries)} doc entries and {len(city_entries)} city entries at {datetime.datetime.now()}")
            print(f"\treceived {len(doc_entries)} doc entries and {len(city_entries)} city entries")

            # Do some verification??? 
            citiesExpected = current_df.loc[current_df["entryType"] == "CITY"]
            docsExpected = current_df.loc[current_df["entryType"] == "DOC"]
            if len(city_entries) != len(citiesExpected):
                print(f"\tunexpected number of cities, expected {len(citiesExpected)} but got {len(city_entries)}")
                logger.error(f"\tunexpected number of cities, expected {len(citiesExpected)} but got {len(city_entries)}")
            if len(doc_entries) != len(docsExpected):
                print(f"\tunexpected number of docs, expected {len(docsExpected)} but got {len(doc_entries)}")
                logger.error(f"\tunexpected number of docs, expected {len(docsExpected)} but got {len(doc_entries)}")

            # 4. Save to CSVs
            with open(output_file, 'a', encoding='utf-8', newline='') as doc_csv:
                doc_writer = csv.DictWriter(doc_csv, DocEntry.model_fields.keys(), restval="")
                for e in doc_entries:
                    doc_writer.writerow(e)

            if city_entries:
                with open(cities_file, 'a', encoding='utf-8', newline='') as city_csv:
                    city_writer = csv.DictWriter(city_csv, CityEntry.model_fields.keys(), restval="")
                    for e in city_entries:
                        city_writer.writerow(e)
                prev_city = city_entries[-1]

        #  (attempt {attempt + 1}/{max_retries})
        except json.JSONDecodeError as e:
            print(f"\n    JSON parse error: {e}")
            logger.error(f"JSON parse error: {e}")
            # time.sleep(1)
        except Exception as e:
            print(f"\n    Gemini API error: {e}")
            logger.error(f"Gemini API error: {e}")
            # time.sleep(2 ** attempt)


        # 5. Clean up, prepare for next iteration
        client.files.delete(name=file.name) # Delete remote file
        temp_file.unlink(missing_ok=True) # delete local file
        next_entry += MAX_ENTRIES_SENT
    # end grouping while
# end for column loop
    
client.close()
print(f"\n✓ Success!")# Extracted {len(all_entries)} entries.")
#print(f"\tUnknown entries {num_unknown} ({100 * num_unknown / len(all_entries)}%)")
print(f"Docs saved to: {output_file}")
print(f"Cities saved to: {cities_file}")
# logger.info(f"\tUnknown entries {num_unknown} ({100 * num_unknown / len(all_entries)}%)")
logger.error(f"Saved Docs entries to: {output_file}")
logger.error(f"Saved Cities entries to: {cities_file}")