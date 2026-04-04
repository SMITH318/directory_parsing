from google import genai
from pydantic import BaseModel
import os
from _batch_utilities import *

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

# --- Gemini API Configuration --- 
API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY')
MODEL_NAME ='gemini-3-flash-preview'

if API_KEY == 'YOUR_API_KEY' or not API_KEY:
    print("ERROR: Gemini API key is not set.")
    logger.error("ERROR: Gemini API key is not set.")
    exit(1)

# Initialize Gemini
print("Initializing Gemini for OCR...")
#print(f"Key: {API_KEY}")
logger.info(f"Initializing Gemini for OCR... with {MODEL_NAME}")
client = genai.Client(api_key=API_KEY)
# PROJECT_ID = 'digitizing-directories-mrsmith'
# REGION = 'us-central1'
# client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)

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

batch_prompt_dataframe(
    client,
    logger, 
    MODEL_NAME, 
    MODEL_PROMPT, 
    "cities", 
    CityEntry, 
    CityEntries, 
    data_set = "2026.03.18",
    only_count_tokens=False,#True,
    max_batches_at_once=80,
    max_entries_per_batch=40, 
    initial_wait_seconds=60 * 8, # 8 minutes
    followup_wait_seconds= 60 * 1, # 1 minute
    record_prompts_responses=True
)
client.close()