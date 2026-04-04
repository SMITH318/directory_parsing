from pydantic import BaseModel
from _ExtractEntriesBatchProcessor import *

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
  filename='04_parse_entries_gemini.log', 
  filemode='a', 
  encoding='utf-8', 
  level=logging.INFO) ## <=================== Change logging level here


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

MODEL_NAME ='gemini-3-flash-preview'
MODEL_PROMPT = (
    "Parse these ordered entries from a medical directory, each line is a complete entry. "
    "Entries are contained within a single column of a page and publication. "
    "Entries are cities with their county and often with their population. "
    "For each entry, I'm providing the name of the publication, "
    "the page number, the column number, the entry type, its text, an id, the name of the "
    "city's state, and a bounding box represented by an x and y coordinate, a height, and a width. "
    "Return ONLY one JSON array of the city entries with this exact format:\n"
    '[\n'
    '  {"publication": "Alabama", "page_number": 1, "column": 0, "x": 297, "y": 1622, "width": 686, "height": 41, '
    '"entry_id": "Alab_001_00_00002", "state_name": "Alabama", "name": "City Name", "population": 10000, '
    '"county_name": "Dallas", "post_reference": "Oakville", "post_reference_type": "RD"},\n'
    '  {"publication": "Texas", "page_number": 4, "column": 0, "x": 215, "y": 1263, "width": 760, "height": 63, '
    '"entry_id": "Texa_004_00_00268", "state_name": "Texas", "name": "Dallas", "population": 15600, '
    '"county_name": "Harris", "post_reference": "", "post_reference_type": ""}\n'
    ']\n'
    "You are critical to the data entry pipeline. "
    "Your goal is to make no mistakes and accurately extract "
    "individual pieces of information from the text input. "
)

batch_processor = ExtractEntriesBatchProcessor(
    logger, 
    MODEL_NAME, 
    MODEL_PROMPT, 
    "city", 
    CityEntry, 
    CityEntries, 
    only_count_tokens=False,#True,
    max_batches_at_once=1,#80,
    max_entries_per_batch=40, 
    initial_wait_seconds=60 * 8, # 8 minutes
    followup_wait_seconds= 60 * 1, # 1 minute
)
input_file, output_dir, output_file_name = gen_extract_entries_paths("city", "2026.03.18")
batch_processor.batch_prompt(input_file, output_dir, output_file_name)