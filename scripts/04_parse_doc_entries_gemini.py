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

class DocEntry(BaseModel):
    publication: str
    page_number: int
    column: int
    x: int
    y: int
    width: int
    height: int
    entry_id: str
    city_id: str # the city's entry_id
    name: str
    AMA_member: Literal["True", "False"]
    col: Literal["True", "False"]
    birth_year: int = None
    AMA_fellow: Literal["True", "False"]
    schools: str = None #(state.school_number,'grad_year)
    license_year: int = None # form (l XX)
    not_in_practice: Literal["True", "False"]
    address: str = None
    office: str = None
    hours: str = None
    societies: str = None
    specialty: str = None
    military: str = None
    other_info: str = None

class DocEntries(BaseModel):
    doc_entries: list[DocEntry]

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
MAX_ENTRIES_SENT = 20
MODEL_PROMPT = (
    "Parse these ordered entries from a medical directory, each line is a complete entry. "
    "Entries are contained within a single column of a page and publication. "
    "Entries are doctors including a wide range of information about them "
    "and their careers, including their birth year, education, address, " 
    "office address, any specialty, among others. "
    "For each entry, I'm providing the name of the publication, "
    "the page number, the column number, the entry type, its text, an id for the doctor, "
    "an id for the city where they live, and "
    "a bounding box represented by an x and y coordinate, a height, and a width. "
    "Return ONLY one JSON array of the doctor entries. "
    "For every entry, keep existing information (including publication, page number, " 
    "column number, id, city id, and its bounding box) and leave it unchanged. "
    "Ignore the entry type. "
    "Only use the full_text field to determine the values of the following fields. "
    "Doctor entries can include a number of different elements, but most have only "
    "the doctor's name, their medical school information (represented as a state abbreviation, " 
    "an ID number, and a 2-digit graduation year or as ◊ if the information is missing), "
    "and a 2-digit license year in parentheses after an l or I. "
    "If the license year is unknown, a t appears in place of a year; "
    "encode it as -2. A ♁ or ‡ can appear in parentheses, instead of licensing "
    "information; encode ♁ as -1 for the license year and ‡ as -3. "
    "Doctors listed as 'not in practice' usually have no license year, leave it blank. "
    "For most doctors, a 2-digit birth year in the format (b'YY) appears. "
    "Leave all years - birth years, license years, etc. - as 2 digits. "
    "⊕ indicates the doctor is an AMA fellow, and their name in upper case (with the exception of "
    "punctuation and at most one lower-case letter) indicates AMA membership. "
    "Doctors' entries that include (col), (col.), or near variations - and only those entries - "
    "have a col value of True. The col values for all other doctors is False. "
    "Addresses can be street addresses, building names, P.O. boxes, or R.D. routes. "
    "P.O. and R.D. always indicate a type of address and "
    "are often followed by a name or number that is part of the address. "
    "Some doctors are members of societies, which are listed in parentheses and "
    "encoded as capital letters A-G followed by one- or two-digit numbers. "
    "Specialties are short abbreviations that can be followed by either * or ★. "
    "Specialty abbreviations are S, Ob, G, ObG, Or, Pr, Op, A, LR, ALR, OALR, U, D, Pd, N, "
    "P, NP, I (sometimes appears as l), T, Anes, CP, R, Path, Bact, and PH. "
    "▼ and any following N or G indicates a military commission. "
    "Hours appear as time ranges separated by commas and with semicolons as hour-minute separators. "
    "Put any extra information that does not fit in another field into the other_info field. "
    "Most individual pieces of information in a doctor entry are separated by semicolons, " 
    "and dashes appear after birth years and before professional information like medical "
    "school and license year. "
    "For string or text fields, maintain the UTF-8 character encoding, "
    "retaining all text characters as they are. "
    "You are critical to the data entry pipeline. "
    "Your primary goal is to make no mistakes and accurately extract "
    "individual pieces of information from the text input. "
    "Here is a typical short doctor entry's full_text, followed by the correct interpretation: "
    "\nRAY, THOS. QUINCY-Ga.5,'95; (l 95)\n"
    "The name is RAY, THOS. QUINCY; AMA_member is True; his schools value is Ga.5,'95; and his license_year is 95. "
    "Here is an example of a long doctor entry's full_text, followed by the correct interpretation: "
    "\nBROTHERS, THOMAS JEFFERSON (b'79) ⊕-Md.3,'03; (l 02); 1809 Christine Ave.; office, 1009 1/2 Noble St.; 11-12, 2;30-4; S\n"
    "The name is BROTHERS, THOMAS JEFFERSON; AMA_member is True; col is False; birth_year is 79; AMA_fellow is True; "
    "schools is Md.3,'03; license_year is 2; not_in_practice is False; address is 1809 Christine Ave.; "
    "office is 1009 1/2 Noble St.; hours is '11-12, 2;30-4'; and specialty is S. "
    "Here is another example doctor entry's full_text, followed by the correct interpretation: "
    "\nWilborn, Daniel W. (col.) (b'80)-N.C.3, '09; (l 10) ; 1701 Mulberry St\n"
    "The name is Wilborn, Daniel W.; AMA_member is False; col is True; birth_year is 80; "
    "schools is N.C.3, '09; license_year is 10; and address is 1701 Mulberry St. "
)

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Set up file paths
input_file = project_root / "data" / "03_processed_batch" / "doc_entries_4states.csv"
output_dir = project_root / "data" / f"04_extracted_entries_gemini_4states"
output_file = output_dir / "amd_1918_doc_entries.csv" 
prompts_file = output_dir / "extracted_entries_prompts.jsonl"
responses_file = output_dir / "extracted_entries_responses.jsonl"
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Load the data, group by column
if not input_file.exists():
    print(f"Error: {input_file} not found.")
    exit(1)

all_inputs = pd.read_csv(input_file, encoding="utf-8")

# create file, write headers for CSVs
with open(output_file, 'w', encoding='utf-8', newline='') as doc_csv:
    doc_writer = csv.DictWriter(doc_csv, DocEntry.model_fields.keys(), restval="")
    doc_writer.writeheader()

# cache system prompt
cache = client.caches.create(
    model=model_name,
    config=types.CreateCachedContentConfig(
        system_instruction=MODEL_PROMPT
    )
)

# limit to MAX_ENTRIES_SENT
next_entry = 0
while next_entry < len(all_inputs):
    
    # 2. Save df and upload it
    last_entry = min(next_entry + MAX_ENTRIES_SENT, len(all_inputs))
    current_df = all_inputs.iloc[next_entry:last_entry]
    file = current_df.to_csv(index=False, encoding="utf-8") # produces string if not given filetten
    
    # upload blocks file
    # file = client.files.upload(
    #     file=temp_file,
    #     config=types.UploadFileConfig(mime_type='text/csv')#(mime_type='text/csv; charset=UTF-8')
    # )

    # 3. Send parsing prompt
    logger.info(f"prompting for docs ({next_entry}:{last_entry}) out of {len(all_inputs)} at {datetime.datetime.now()}")
    print(f"prompting for docs ({next_entry}:{last_entry}) out of {len(all_inputs)} at {datetime.datetime.now()}")

    attempt = 0
    while attempt < MAX_ATTEMPTS:
        print(f"attempt {attempt + 1}/{MAX_ATTEMPTS}")
        logger.info(f"attempt {attempt + 1}/{MAX_ATTEMPTS}")
        try:
            # save prompt for tuning
            with open(prompts_file, 'a') as f:
                f.write(json.dumps(
                    {
                        "systemInstruction" : types.Content(role="system", parts = [types.Part(text=MODEL_PROMPT)]).model_dump(exclude_none=True),
                        "contents": types.UserContent([file]).model_dump(exclude_none=True)
                    }
                )+ '\n')
            #response_stream = client.models.generate_content_stream(
            response = client.models.generate_content(
                model=model_name,
                contents=[file],
                config=types.GenerateContentConfig(
                     cached_content=cache.name,
                    # systemInstruction = types.Content(role="system", parts = [types.Part(text=MODEL_PROMPT)]),
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
                    response_mime_type="application/json", 
                    response_json_schema=DocEntries.model_json_schema(),
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
            with open(responses_file, 'a') as f:
                f.write(json.dumps(json_entries) + '\n')
            doc_entries = json_entries["doc_entries"]

            logger.info(f"received {len(doc_entries)} doc entries at {datetime.datetime.now()}")
            print(f"\treceived {len(doc_entries)} doc entries at {datetime.datetime.now()}")

            # Do some verification
            if len(doc_entries) != len(current_df):
                print(f"\tunexpected number of docs, expected {len(current_df)} but got {len(doc_entries)}")
                logger.error(f"\tunexpected number of docs, expected {len(current_df)} but got {len(doc_entries)}")

            # 4. Save to CSVs
            with open(output_file, 'a', encoding='utf-8', newline='') as doc_csv:
                doc_writer = csv.DictWriter(doc_csv, DocEntry.model_fields.keys(), restval="")
                for e in doc_entries:
                    doc_writer.writerow(e)

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
    
client.caches.delete(name=cache.name)
client.close()
print(f"\n✓ Success!")
print(f"Docs saved to: {output_file}")
logger.error(f"Saved Docs entries to: {output_file}")