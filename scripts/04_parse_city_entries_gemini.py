import json
import numpy as np
from pathlib import Path
from google import genai
from google.genai import types
import pandas as pd
from pydantic import BaseModel
import os
import time
import csv
import datetime

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
    
class CityEntry(BaseModel):
    publication: str
    page_number: int
    column: int
    x: int
    y: int
    width: int
    height: int
    entry_id: str
    state_name: str
    name: str
    population: int = None # if present
    county_name: str
    post_reference: str = None
    post_reference_type: str = None # RD, PO, etc.

class CityEntries(BaseModel):
    city_entries: list[CityEntry]

# --- Gemini API Configuration --- 
API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY')
model_name ='gemini-3-flash-preview'#'gemini-3.1-flash-lite-preview'#'gemini-2.5-flash'#'gemini-3.1-pro-preview'#'gemini-flash-latest'
# model_name = 'projects/670765358210/locations/us-central1/endpoints/2458473365090861056' # my tuned model - may have to run in Vertex
# model_name = 'tunedModels/2458473365090861056'

if API_KEY == 'YOUR_API_KEY' or not API_KEY:
    print("ERROR: Gemini API key is not set.")
    logger.error("ERROR: Gemini API key is not set.")
    exit(1)

# Initialize Gemini
print("Initializing Gemini for OCR...")
#print(f"Key: {API_KEY}")
logger.info(f"Initializing Gemini for OCR... with {model_name}")
client = genai.Client(api_key=API_KEY)
# PROJECT_ID = 'digitizing-directories-mrsmith'
# REGION = 'us-central1'
# client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)

MAX_ATTEMPTS = 3
MAX_ENTRIES_SENT = 40
MODEL_PROMPT = (
    "Parse these ordered entries from a medical directory, each line is a complete entry. "
    "Entries are contained within a single column of a page and publication. "
    "Entries are cities with their county and often with their population. "
    "For each entry, I'm providing the name of the publication, "
    "the page number, the column number, the entry type, its text, an id, the name of the "
    "city's state, and a bounding box represented by an x and y coordinate, a height, and a width. "
    "Return ONLY one JSON array of city entries. "
    "For every entry, keep existing information (including publication, page number, " 
    "column number, id, state, and its bounding box) and leave it unchanged. "
    "Ignore the entry type. "
    "Only use the full_text field to determine the values of the following fields. "
    "City entries consist of the name of the city, sometimes followed by postal information "
    "in parentheses consisting of the type of postal reference (R.D., P.O., etc.) "
    "and its name (the postal reference value). "
    "Most cities then are followed by a population number, and all end with the name of the county. "
    "You are critical to the data entry pipeline. "
    "Your goal is to make no mistakes and accurately extract "
    "individual pieces of information from the text input. "
)

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Set up file paths
input_file = project_root / "data" / "03_processed_batch" / "city_entries_4states.csv"
output_dir = project_root / "data" / f"04_extracted_entries_gemini_4states"
# temp_file = output_dir / "temp.csv"
# output_file = output_dir / "amd_1918_doc_entries.csv" 
cities_file = output_dir / "amd_1918_city_entries.csv"
# prompts_file = output_dir / "extracted_entries_prompts.jsonl"
# responses_file = output_dir / "extracted_entries_responses.jsonl"
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Load the data, group by column
if not input_file.exists():
    print(f"Error: {input_file} not found.")
    exit(1)

# def make_int_formatter(format_str):
#     return lambda str_in : format(str_in, format_str)

all_inputs = pd.read_csv(input_file, encoding="utf-8")

# create file, write headers for CSVs
with open(cities_file, 'w', encoding='utf-8', newline='') as doc_csv:
    doc_writer = csv.DictWriter(doc_csv, CityEntry.model_fields.keys(), restval="")
    doc_writer.writeheader()

# cache system prompt
# cache = client.caches.create(
#     model=model_name,
#     config=types.CreateCachedContentConfig(
#         system_instruction=MODEL_PROMPT
#     )
# )

# limit to MAX_ENTRIES_SENT
next_entry = 0
while next_entry < len(all_inputs):
    
    # 2. Save df and upload it
    last_entry = min(next_entry + MAX_ENTRIES_SENT, len(all_inputs))
    current_df = all_inputs.iloc[next_entry:last_entry]
    file = current_df.to_csv(index=False, encoding="utf-8") # produces string if not given file
    
    # upload blocks file
    # file = client.files.upload(
    #     file=temp_file,
    #     config=types.UploadFileConfig(mime_type='text/csv')#(mime_type='text/csv; charset=UTF-8')
    # )

    # 3. Send parsing prompt
    logger.info(f"prompting for cities ({next_entry}:{last_entry}) out of {len(all_inputs)} at {datetime.datetime.now()}")
    print(f"prompting for cities ({next_entry}:{last_entry}) out of {len(all_inputs)} at {datetime.datetime.now()}")

    attempt = 0
    while attempt < MAX_ATTEMPTS:
        print(f"attempt {attempt + 1}/{MAX_ATTEMPTS}")
        logger.info(f"attempt {attempt + 1}/{MAX_ATTEMPTS}")
        try:
            # save prompt for tuning
            # with open(temp_file, 'r', encoding='utf-8') as f:
            #     file_text = f.read()
            # file_text=file
            # with open(prompts_file, 'a') as f:
            #     f.write(json.dumps(
            #         {
            #             "systemInstruction" : types.Content(role="system", parts = [types.Part(text=MODEL_PROMPT)]).model_dump(exclude_none=True),
            #             "contents": types.UserContent([
            #                 types.Part(text=f'id of last city encountered was {prev_city["entry_id"]}'), 
            #                 types.Part(text=f'name of last state encountered was {prev_city["state_name"]}'), 
            #                 types.Part(text=file_text)
            #             ] if prev_city else [file_text]).model_dump(exclude_none=True)
            #         }
            #     )+ '\n')
            response = client.models.generate_content(
                model=model_name,
                contents=[file],
                config=types.GenerateContentConfig(
                    # cached_content=cache.name,
                    systemInstruction = types.Content(role="system", parts = [types.Part(text=MODEL_PROMPT)]),
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"), # minimal thinking hurt results 3/3
                    response_mime_type="application/json", 
                    response_json_schema=CityEntries.model_json_schema(),
                )
            )
            response_text = response.text.strip()
            logger.debug(response_text)

            # Clean markdown formatting if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
                response_text = response_text.replace('```json', '').replace('```', '').strip()

            json_entries = json.loads(response_text)
            # with open(responses_file, 'a') as f:
            #     f.write(json.dumps(json_entries) + '\n')
            city_entries = json_entries["city_entries"]

            logger.info(f"received {len(city_entries)} city entries at {datetime.datetime.now()}")
            print(f"\treceived {len(city_entries)} city entries")

            # Do some verification
            if len(city_entries) != len(current_df):
                print(f"\tunexpected number of cities, expected {len(current_df)} but got {len(city_entries)}")
                logger.error(f"\tunexpected number of cities, expected {len(current_df)} but got {len(city_entries)}")

            # 4. Save to CSV
            with open(cities_file, 'a', encoding='utf-8', newline='') as city_csv:
                city_writer = csv.DictWriter(city_csv, CityEntry.model_fields.keys(), restval="")
                for e in city_entries:
                    city_writer.writerow(e)

        except json.JSONDecodeError as e:
            print(f"\n    JSON parse error: {e}")
            logger.error(f"JSON parse error: {e}")
            time.sleep(1)
            attempt += 1
        except Exception as e:
            print(f"\n    Gemini API error: {e}")
            logger.error(f"Gemini API error: {e}")
            time.sleep(2 ** attempt)
            attempt += 1
        else:
            break # if not errors, break out of attempts loop
    if attempt == MAX_ATTEMPTS:
        logger.error(f"Failed after {attempt} attempts")
        print(f"Failed after {attempt} attempts")

    # 5. Clean up, prepare for next iteration
    # client.files.delete(name=file.name) # Delete remote file
    # temp_file.unlink(missing_ok=True) # delete local file
    next_entry += MAX_ENTRIES_SENT
# end grouping while
    
# client.caches.delete(name=cache.name)
client.close()
print(f"\n✓ Success! Cities saved to: {cities_file}")
logger.error(f"Saved Cities entries to: {cities_file}")