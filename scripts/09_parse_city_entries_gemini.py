"""
Step 9: Parse City Entries with Gemini
This script processes the city entries CSV generated in Step 8 and uses the Gemini model to 
parse detailed information from the city entries.
It uses the Batch API to efficiently handle the parsing of a large number of entries, and 
includes error handling to manage API rate limits and other exceptions.
The parsed city entries are saved in a new CSV file for use in later stages of the data processing pipeline.
"""

from pydantic import BaseModel
from google.genai import errors
from _ExtractEntriesStep import *
from _BatchProcessor import *

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
  filename='04_parse_entries_gemini.log', 
  filemode='a', 
  encoding='utf-8', 
  level=logging.WARNING) ## <=================== Change logging level here


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

INITIAL_WAIT_SECONDS = 60 * 8 # 8 minutes
FOLLOWUP_WAIT_SECONDS = 60 * 1 # 1 minute
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

def create_batch_processor():
    step_config = ExtractEntriesStep(
        MODEL_NAME, 
        MODEL_PROMPT, 
        "city", 
        CityEntry, 
        CityEntries
    )
    return BatchProcessor(
        step_config,
        logger,
        only_count_tokens=False,#True,
        max_batches_at_once=100, # the Batch API max
        max_entries_per_batch=40, 
        initial_wait_seconds=INITIAL_WAIT_SECONDS, # 8 minutes
        followup_wait_seconds=FOLLOWUP_WAIT_SECONDS, # 1 minute
    )

if __name__ == "__main__":
    # 1. Setup Project Paths
    proj_paths = gen_extract_entries_paths("city", "2026.03.18")

    batch_processor = None

    for i in range(100):
        try:
            print("*** Iteration", i, "***")
            if not batch_processor:
                batch_processor = create_batch_processor()
            batch_processor.batch_prompt(
                *proj_paths,
                # record_prompts_responses=True
            )
        except Exception as e:
            if isinstance(e, errors.APIError) and (e.code == 429 or e.code == 503):
                exception = "RESOURCE_EXHAUSTED" if e.code == 429 else "SERVICE UNAVAILABLE"
                print(f"*** main loop {exception} exception, pausing for {INITIAL_WAIT_SECONDS/60} at {datetime.datetime.now()}... ***")
                logger.error(f"*** main loop {exception} exception, pausing for {INITIAL_WAIT_SECONDS/60} at {datetime.datetime.now()}... ***")
                time.sleep(INITIAL_WAIT_SECONDS)
            else:
                print("*** main loop exception, clearing batches, pressing on ***")
                print(type(e).__name__, "-", e)
                # something went very wrong, scrub any ongoing batch jobs and processor
                for job in batch_processor.client.batches.list():
                    try:
                        batch_processor.client.batches.delete(name=job.name)
                    except:
                        pass
                batch_processor = None

