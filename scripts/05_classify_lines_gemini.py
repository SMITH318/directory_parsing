"""
Step 5: Group and classify OCR lines into entries
- Reads .jsonl OCR output with line-level text and bounding boxes.
- Groups text blocks into entries based on content, using heuristic Gemini API prompt.
- Uses batching to process many columns/snippets at once, with error handling and retry logic for API rate limits and transient errors.
- Saves segmented entries to CSV with metadata and aggregate bounding boxes.
"""

from google.genai import errors
from typing import Literal
from _AStepConfiguration import *
from _BatchProcessor import *
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='03_classify_lines_gemini.log', 
    filemode='a', 
    encoding='utf-8', 
    level=logging.WARNING) ## <=================== Change logging level here

# ***************************** constants *****************************
INITIAL_WAIT_SECONDS = 60 * 5 # 5 minutes
FOLLOWUP_WAIT_SECONDS = 60 * 1 # 1 minute
MODEL_NAME ='gemini-flash-latest'
MODEL_PROMPT = (
    "Combine these ordered lines of text from a directory into entries, "
    "leaving ALL of the characters in their UTF-8 format. "
    "PRESERVE ALL of these characters: ▼, ⊕, ◊, ★, †, ♁, and ‡. "
    "Each line may be a complete entry or part of a multi-line entry. " 
    "Entries are contained within a single column of a page and publication. "
    "Entries can be: (1) names of states or provinces; " 
    "(2) cities with their county and often with their population; or "
    "(3) doctors including a wide range of informatiom about them "
    "and their careers, including their birth year, education, address, " 
    "office address, any specialty, among others. "
    "For each line of text, I'm providing the name of the publication, "
    "the page number, the column number, "
    "the text, a confidence value (0 to 1), and a bounding box " 
    "represented by an x and y coordinate, a height, and a width. "
    "Return ONLY a JSON array of the combined entries, including the source "
    "information (publication, page number, and column number), " 
    "what kind of entry it is (STATE, CITY, DOC, or UNKNOWN), "
    "the combined lines of text, and the combined bounding box for the lines "
    "specified by an x and y coordinate, a height, and a width. " 
    "When combining lines, remove hyphens or dashes at end of lines for words broken "
    "across lines, and combine the full word."
    "Maintain the UTF-8 character encoding, retaining ALL characters as they are. "
)

class ClassifiedLine(BaseModel):
    publication: str
    page_number: int
    column: int
    entryType: Literal["STATE", "CITY", "DOC", "UNKNOWN"]
    full_text: str
    x: int
    y: int
    width: int
    height: int

class ClassifiedEntries(BaseModel):
    entries: list[ClassifiedLine]

class ClassifyLinesStep(AStepConfiguration):
    # abstract
    def drop_some_finished(self, finished_df: pd.DataFrame) -> pd.DataFrame:
        return finished_df

    # abstract
    def load_input(self, file_in: Path) -> pd.DataFrame:
        # Read JSONL file and load it into a list of dictionaries
        line_dicts = []
        with open(file_in, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                
                # flatten bbox into entry
                if entry.get("bbox") is not None:
                    # blocks[bbox] is [x, y, w, h], normalize as dict
                    if isinstance(entry["bbox"], list):
                        bbox = entry["bbox"]
                        entry["bbox"] = {"x": bbox[0], "y": bbox[1], "width": bbox[2], "height": bbox[3]}
                    # do flattening
                    entry["x"] = int(entry["bbox"]["x"])
                    entry["y"] = int(entry["bbox"]["y"])
                    entry["width"] = int(entry["bbox"]["width"])
                    entry["height"] = int(entry["bbox"]["height"])
                    del entry["bbox"]
                line_dicts.append(entry)

        # Turn list of dictionaries into DataFrame
        df = pd.DataFrame(line_dicts)

        # create a new data frame where each row is a data frame of lines from the same page (grouped by pub, page, col)
        # first create a list of dicts, each with name: key and group: DataFrame
        dfs = [{"name": name, "group": group} for name, group in df.groupby(['pub', 'page', 'col'])]
        return pd.DataFrame(dfs)

    # abstract
    def load_finished(self, file_done: Path) -> pd.DataFrame:
        return pd.DataFrame()
    
    # abstract
    def df_columns_to_check_finished(self) -> list[tuple[str,str]]:
        raise NotImplementedError(""""
            ClassifyLinesBatchProcessor does not check finished entries by key columns, 
            since it is designed to be run after all lines have been extracted and saved to a single output file.
        """)

    # abstract
    def prepare_for_request(self, request_df: pd.DataFrame) -> tuple[str, types.UserContent]: # request key, content
        file = request_df["group"].to_csv(index=False, encoding="utf-8") # produces string if not given file
        content = types.UserContent([file])
        return f"{self.entry_type_name}_{request_df['name']}", content
    
    # abstract
    def save_job_output_content(self, logger: logging.Logger, display_name:str, response_text:str, output_file:Path, responses_file:Path|None = None) -> bool:
        successful = True
        
        # get entries and verify it's a list
        entries = json.loads(response_text)["entries"]
        if not isinstance(entries, list):
            raise ValueError(f"Response to {display_name} is not a JSON array")

        
        lines_received_txt = f"received {len(entries)} entries at {datetime.datetime.now()}"
        logger.info(lines_received_txt)
        print("\t", lines_received_txt)

        # 4. Save 
        with open(output_file, 'a', encoding='utf-8', newline='') as f_out:
            writer = csv.DictWriter(f_out, self.entry_type.model_fields.keys())
            for entry in entries: 
                if entry["entryType"] == "UNKNOWN":
                    num_unknown += 1
                writer.writerow(entry)
        return successful
    
    # abstract
    def prep_output_file(self, output_file:Path):
        with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, self.entry_type.model_fields.keys(), restval="")
            writer.writeheader()


def create_batch_processor():
    step_config = ClassifyLinesStep(
        MODEL_NAME, 
        MODEL_PROMPT, 
        "entry", 
        ClassifiedLine, 
        ClassifiedEntries, 
    )
    return BatchProcessor(
        step_config,
        logger, 
        only_count_tokens=False,#True,
        max_batches_at_once=100, # Batch API MAX
        max_entries_per_batch=1,
        initial_wait_seconds=INITIAL_WAIT_SECONDS, # 8 minutes
        followup_wait_seconds=FOLLOWUP_WAIT_SECONDS, # 1 minute
    )

if __name__ == "__main__":
    # 1. Setup Project Paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    input_filename = "ocr_output_reviewed_batch_mass_2026.03.jsonl"
    input_file = project_root / "data" / "02_raw_batch_mass" / input_filename
    output_dir = project_root / "data" / "03_processed_batch_mass"
    output_file_name = "entries_segmented_batch_class_test.csv"

    batch_processor = None

    for i in range(100):
        try:
            print("*** Iteration", i, "***")
            if not batch_processor:
                batch_processor = create_batch_processor()
            batch_processor.batch_prompt(
                input_file,
                output_dir,
                output_file_name
                # record_prompts_responses=True
            )
     
        except Exception as e:
            if isinstance(e, errors.APIError) and (e.code == 429 or e.code == 503):
                exception = "RESOURCE_EXHAUSTED" if e.code == 429 else "SERVICE UNAVAILABLE"
                print(f"*** main loop {exception} exception, pausing for {INITIAL_WAIT_SECONDS/60} at {datetime.datetime.now()}... ***")
                logger.error(f"*** main loop {exception} exception, pausing for {INITIAL_WAIT_SECONDS/60} at {datetime.datetime.now()}... ***")
                time.sleep(INITIAL_WAIT_SECONDS)
            else:
                print("*** main loop exception, pressing on ***")
                print(type(e).__name__, "-", e)
                # something went very wrong, scrub any ongoing batch jobs and processor
                for job in batch_processor.client.batches.list():
                    try:
                        batch_processor.client.batches.delete(name=job.name)
                    except:
                        pass
                batch_processor = None


