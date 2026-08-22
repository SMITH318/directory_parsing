"""
Step 5: Group and classify OCR lines into entries
- Reads .jsonl OCR output with line-level text and bounding boxes.
- Groups text blocks into entries based on content, using heuristic Gemini API prompt.
- Uses batching to process many columns/snippets at once, with error handling and retry logic for API rate limits and transient errors.
- Saves segmented entries to CSV with metadata and aggregate bounding boxes.
"""

from typing import Literal
import sys
from _AStepConfiguration import *
from _BatchProcessor import *
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    handlers=[
        logging.FileHandler('05_classify_lines_gemini.log', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stderr)
    ],
    level=logging.WARNING) ## <=================== Change logging level here

# ***************************** constants *****************************
INITIAL_WAIT_SECONDS = 60 * 5 # 5 minutes
FOLLOWUP_WAIT_SECONDS = 60 * 1 # 1 minute
MODEL_NAME ="gemini-3-flash-preview"#'gemini-flash-latest'

def get_classified_models(schema_path: Path):
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_data = json.load(f)
    types = list(schema_data.get('properties', {}).keys()) + ["UNKNOWN"]
    prompt = schema_data['x-classify-prompt']

    EntryTypeEnum = Literal[tuple(types)]

    ClassifiedLineDynamic = create_model(
        'ClassifiedLine',
        publication=(str, ...),
        page_number=(int, ...),
        column=(int, ...),
        entryType=(EntryTypeEnum, ...),
        full_text=(str, ...),
        x=(int, ...),
        y=(int, ...),
        width=(int, ...),
        height=(int, ...)
    )

    ClassifiedEntriesDynamic = create_model(
        'ClassifiedEntries',
        entries=(list[ClassifiedLineDynamic], ...)
    )

    return prompt, ClassifiedLineDynamic, ClassifiedEntriesDynamic

class ClassifyLinesStep(AStepConfiguration):
    # abstract
    def drop_some_finished(self, finished_df: pd.DataFrame) -> pd.DataFrame:
        return finished_df

    def _group_df_by_column(self, df: pd.DataFrame) -> pd.DataFrame:
        # create a new data frame where each row is a data frame of lines from the same page (grouped by pub, page, col)
        # first create a list of dicts, each with name: key and group: DataFrame
        cols = ['pub', 'page', 'col']
        dfs = [{"name": name, "group": group} for name, group in df.groupby(cols)]
        return pd.DataFrame(dfs)
    
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

        return self._group_df_by_column(df)

    # abstract
    def load_finished(self, file_done: Path) -> pd.DataFrame:
        df = pd.read_csv(file_done, encoding='utf-8')
        df = df.rename(columns={'publication':'pub', 'page_number':'page', 'column':'col'})
        dfs = self._group_df_by_column(df)
        return dfs
    
    # abstract
    def df_columns_to_check_finished(self) -> list[tuple[str,str]]:
        return [('name', 'name')] 

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

        logger.info(f"received {len(entries)} entries at {datetime.datetime.now()}")
        num_unknown = 0
        # 4. Save 
        with open(output_file, 'a', encoding='utf-8', newline='') as f_out:
            writer = csv.DictWriter(f_out, self.entry_type.model_fields.keys())
            for entry in entries: 
                if entry["entryType"] == "UNKNOWN":
                    num_unknown += 1
                entry["full_text"] = entry["full_text"].replace("\n", " ")
                writer.writerow(entry)
        logger.info(f"{num_unknown} line entries had type UNKNOWN")
        return successful
    
    # abstract
    def prep_output_file(self, output_file:Path):
        with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, self.entry_type.model_fields.keys(), restval="")
            writer.writeheader()


def create_batch_processor(schema_path: Path):
    prompt, ClassifiedLineDynamic, ClassifiedEntriesDynamic = get_classified_models(schema_path)
    step_config = ClassifyLinesStep(
        MODEL_NAME, 
        prompt, 
        "entry", 
        ClassifiedLineDynamic, 
        ClassifiedEntriesDynamic, 
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

def main(dataset: str, config: Path) -> int:
    # 1. Setup Project Paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    input_file = project_root / "data" / dataset / "04_ocr_output_cleaned.jsonl"
    output_dir = project_root / "data" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_name = "05_entries_segmented.csv"

    batch_processor = None
    all_processed = False

    for i in range(100):
        try:
            logger.warning(f"*** Iteration {i} ***")
            if not batch_processor:
                batch_processor = create_batch_processor(config)
            if batch_processor.batch_prompt(
                input_file,
                output_dir,
                output_file_name
                # record_prompts_responses=True
            ):
                all_processed = True
                break
     
        except Exception as e:
            if isinstance(e, errors.APIError) and (e.code == 429 or e.code == 503):
                exception = "RESOURCE_EXHAUSTED" if e.code == 429 else "SERVICE UNAVAILABLE"
                logger.error(f"*** main loop {exception} exception, pausing for {INITIAL_WAIT_SECONDS/60} at {datetime.datetime.now()}... ***")
                time.sleep(INITIAL_WAIT_SECONDS)
            else:
                logger.error("*** main loop exception, pressing on ***")
                logger.error(f"{type(e).__name__} - {e}")
                # something went very wrong, scrub any ongoing batch jobs and processor
                for job in batch_processor.client.batches.list():
                    try:
                        batch_processor.client.batches.delete(name=job.name)
                    except:
                        pass
                batch_processor = None
    
    file_out = output_dir / output_file_name
    print(file_out)
    if all_processed:
        df = pd.read_csv(file_out, encoding='utf-8')
        df_sorted = df.sort_values(by=["publication", "page_number", "column"])
        df_sorted.to_csv(file_out, index=False, encoding='utf-8')
        logger.info(f"✓ Step completed successfully ({file_out})")
        return 0
    else:
        logger.error("✗ Step did not complete all inputs")
        return 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 5: Group and classify OCR lines into entries")
    parser.add_argument("dataset", help="Name of the dataset")
    parser.add_argument("--config", help="Path to JSON schema config with embedded prompts", required=True)
    args = parser.parse_args()
    
    exit_code = main(args.dataset, args.config)
    sys.exit(exit_code)



