"""
Step 9: Parse Doctor Entries with Gemini
This script processes the doctor entries CSV generated in Step 8 and uses the Gemini model to 
parse detailed information from the doctor entries.
It uses the Batch API to efficiently handle the parsing of a large number of entries, and 
includes error handling to manage API rate limits and other exceptions.
The parsed doctor entries are saved in a new CSV file for use in later stages of the data processing pipeline.
"""
from pydantic import BaseModel
from typing import Literal
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

INITIAL_WAIT_SECONDS = 60 * 8 # 8 minutes
FOLLOWUP_WAIT_SECONDS = 60 * 1 # 1 minute
MODEL_NAME ='gemini-3-flash-preview'
MODEL_PROMPT = (
    "Parse these ordered entries from a medical directory, each line is a complete entry. "
    "Entries are contained within a single column of a page and publication. "
    "Entries are doctors including a wide range of information about them "
    "and their careers, including their birth year, education, address, " 
    "office address, any specialty, among others. "
    "For each entry, I'm providing the name of the publication, "
    "the page number, the column number, the entry type, its text, an entry id for the doctor, "
    "an id for the city where they live, and "
    "a bounding box represented by an x and y coordinate, a height, and a width. "
    "Return ONLY one JSON array of the doctor entries with this exact format:\n"
    '[\n'
    '  {"publication": "Alabama", "page_number": 1, "column": 0, "x": 297, "y": 1622, "width": 686, "height": 41, '
    '"entry_id": "Alab_001_00_00002", "city_id": "Alab_001_00_00001", "name": "CLARK, JAMES THOMAS", '
    '"AMA_member": True, "col": False, "birth_year": "76", "AMA_fellow": False, "schools": "Ala.4,\'11", '
    '"license_year": "11", "not_in_practice": False, "address": "", "office": "", "hours": "", '
    '"societies": "", "specialty": "", "military": "", "other_info": ""},\n'
    '  {"publication": "Alabama", "page_number": 4, "column": 0, "x": 215, "y": 1263, "width": 760, "height": 63, '
    '"entry_id": "Alab_004_00_00268", "city_id": "Alab_003_00_00190", "name": "DAVIS, JOHN DANL. SINKLER", '
    '"AMA_member": True, "col": False, "birth_year": "59", "AMA_fellow": True, "schools": "Ga.1,\'79", '
    '"license_year": "11", "not_in_practice": False, "address": "2031 Ave. G", "office": "", "hours": "1-3", '
    '"societies": "(A1,28)", "specialty": "S★", "military": "", '
    '"other_info": "Prof. Prin. and Prac. of Surg. and Clin. Surg., Ala. G1"},\n'
    '  {"publication": "Alabama", "page_number": 4, "column": 1, "x": 837, "y": 1702, "width": 900, "height": 52, '
    '"entry_id": "Alab_004_01_00312", "city_id": "Alab_003_00_00190", "name": "HANNA, HENRY PIERCE", '
    '"AMA_member": True, "col": False, "birth_year": "86", "AMA_fellow": False, "schools": "Ala.4, \'12", '
    '"license_year": "13", "not_in_practice": False, "address": "1518 N. Allen St.", "office": "First Natl. Bank Bldg.", '
    '"hours": ""11-1, 3-5"", "societies": "", "specialty": "Pd", "military": "▼", "other_info": ""},\n'
    '  {"publication": "Alabama", "page_number": 4, "column": 2, "x": 1477, "y": 324, "width": 574, "height": 44, '
    '"entry_id": "Alab_004_02_00317", "city_id": "Alab_003_00_00190", "name": "Harris, Hardy Fleming", '
    '"AMA_member": False, "col": True, "birth_year": "", "AMA_fellow": False, "schools": "Tenn.7,\'05", '
    '"license_year": "5", "not_in_practice": False, "address": "808 S. 16th St.", "office": "2709 29th Ave", "hours": "", '
    '"societies": "", "specialty": "", "military": "", "other_info": ""}\n'
    ']\n'
    "For every entry, keep existing information (including publication, page number, " 
    "column number, entry id, city id, and its bounding box) and leave it unchanged. "
    "Ignore the entry type. "
    "Only use the full_text field to determine the values of the following fields. "
    "Doctor entries can include a number of different elements, but most have only "
    "the doctor's name, their medical school information (represented as a state abbreviation, " 
    "an ID number, and a 2-digit graduation year or as ◊ or △ if the information is missing), "
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

def create_batch_processor():
    step_config = ExtractEntriesStep(
        MODEL_NAME, 
        MODEL_PROMPT, 
        "doc", 
        DocEntry, 
        DocEntries
    )
    return BatchProcessor(
        step_config,
        logger,
        only_count_tokens=False,#True,
        max_batches_at_once=100, # Batch API max # 40,#80,
        max_entries_per_batch=20, #50, prompts sized <=5400, but never left pending (same with 40); 20 had prompts sized <= 2200
        initial_wait_seconds=INITIAL_WAIT_SECONDS, # 8 minutes
        followup_wait_seconds=FOLLOWUP_WAIT_SECONDS, # 1 minute
    )

if __name__ == "__main__":
    # 1. Setup Project Paths
    proj_paths = gen_extract_entries_paths("doc", "2026.03.18")

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
