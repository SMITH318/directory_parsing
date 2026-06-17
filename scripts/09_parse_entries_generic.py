"""
Step 9: Parse Entries with Gemini
This script processes entry CSVs generated in Step 8 and uses the Gemini model to 
parse detailed information for specific entry types
based on a dynamically loaded JSON schema and prompt.
It uses the Batch API to efficiently handle the parsing of a large number of entries, and 
includes error handling to manage API rate limits and other exceptions.
The parsed entries are saved in a new CSV file for use in later stages of the data processing pipeline.
"""
from google.genai import errors
import sys
from _ExtractEntriesStep import *
from _BatchProcessor import *

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
  handlers=[
      logging.FileHandler('09_parse_entries_gemini.log', mode='w', encoding='utf-8'),
      logging.StreamHandler(sys.stderr)
  ],
  level=logging.WARNING) ## <=================== Change logging level here

INITIAL_WAIT_SECONDS = 60 * 8 # 8 minutes
FOLLOWUP_WAIT_SECONDS = 60 * 1 # 1 minute
MODEL_NAME ='gemini-3-flash-preview'

def generate_model_for_entity(entity_name: str, schema_data: dict) -> ():
    prop_def = schema_data.get('properties', {}).get(entity_name, {})
    if not prop_def:
        return None, None, None
    items_ref = prop_def.get('items', {}).get('$ref', '')
    if items_ref:
        def_name = items_ref.split('/')[-1]
        definition = schema_data.get('definitions', {}).get(def_name, {})
    else:
        definition = prop_def.get('items', {})
        
    if not definition:
        return None, None, None

    prompt = definition.get('x-parse-prompt', "Parse the entry.")
    
    fields = {}
    for prop_name, prop_info in definition.get('properties', {}).items():
        type_info = prop_info.get('type')
        if isinstance(type_info, list):
            if 'integer' in type_info:
                fields[prop_name] = (int | None, None)
            else:
                fields[prop_name] = (str | None, None)
        elif type_info == 'integer':
            fields[prop_name] = (int, ...)
        else:
            fields[prop_name] = (str, ...)

    EntryDynamic = create_model(f'{entity_name.capitalize()}Entry', **fields)
    EntriesDynamic = create_model(f'{entity_name.capitalize()}Entries', **{f"{entity_name}_entries": (list[EntryDynamic], ...)})
    
    return prompt, EntryDynamic, EntriesDynamic

def create_batch_processor(entity_name: str, prompt: str, EntryDynamic: type[BaseModel], EntriesDynamic: type[BaseModel]) -> BatchProcessor:
    step_config = ExtractEntriesStep(
        MODEL_NAME, 
        prompt, 
        entity_name, 
        EntryDynamic, 
        EntriesDynamic
    )
    return BatchProcessor(
        step_config,
        logger,
        only_count_tokens=False,#True,
        max_batches_at_once=100, # Batch API max
        max_entries_per_batch=20, #50, prompts sized <=5400, but never left pending (same with 40); 20 had prompts sized <= 2200
        initial_wait_seconds=INITIAL_WAIT_SECONDS,
        followup_wait_seconds=FOLLOWUP_WAIT_SECONDS,
    )

def main(dataset: str, config: Path):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / dataset

    completion_file = data_dir / "09_parse_complete.txt"
    completion_file.unlink(missing_ok=True)
    
    if not config:
        logger.error("No config provided. Required for generic parsing.")
        return 1

    with open(config, 'r', encoding='utf-8') as f:
        schema_data = json.load(f)

    all_processed = True
    entities = list(schema_data.get('properties', {}).keys())

    for entity_name in entities:
        input_file = data_dir / f"08_{entity_name.lower()}_entries.csv"
        if not input_file.exists():
            logger.error(f"Skipping {entity_name}, no input file {input_file} found.")
            all_processed = False
            continue

        logger.warning(f"=== Processing entity type: {entity_name} ===")
        output_file_name = f"09_{entity_name.lower()}_parsed.csv"
        output_file = data_dir / output_file_name

        prompt, EntryDynamic, EntriesDynamic = generate_model_for_entity(entity_name, schema_data)
        if not prompt:
            logger.error(f"Could not generate model for {entity_name}")
            all_processed = False
            continue

        batch_processor = None
        entity_processed = False

        for i in range(100):
            try:
                logger.warning(f"*** Iteration {i} ***")
                if not batch_processor:
                    batch_processor = create_batch_processor(entity_name, prompt, EntryDynamic, EntriesDynamic)
                if batch_processor.batch_prompt(
                    input_file,
                    data_dir,
                    output_file_name,
                    # record_prompts_responses=True
                ):
                    entity_processed = True
                    break
            except Exception as e:
                if isinstance(e, errors.APIError) and (e.code == 429 or e.code == 503):
                    exception = "RESOURCE_EXHAUSTED" if e.code == 429 else "SERVICE UNAVAILABLE"
                    logger.error(f"*** main loop {exception} exception processing {entity_name}, pausing for {INITIAL_WAIT_SECONDS/60} at {datetime.datetime.now()}... ***")
                    time.sleep(INITIAL_WAIT_SECONDS)
                else:
                    logger.error(f"*** main loop exception processing {entity_name}, clearing batches, pressing on ***")
                    logger.error(f"{type(e).__name__} - {e}")
                    # something went very wrong, scrub any ongoing batch jobs and processor
                    for job in batch_processor.client.batches.list():
                        try:
                            batch_processor.client.batches.delete(name=job.name)
                        except:
                            pass
                    batch_processor = None
        
        if entity_processed:
            logger.info(f"✓ {entity_name} entries successfully parsed ({output_file})")
        else:
            logger.error(f"✗ {entity_name} entries failed to parse")
            all_processed = False

    # create marker
    with open(completion_file, "w") as f:
        f.write("complete\n")

    if all_processed:
        logger.info(f"✓ Parsing step completed successfully parsed")
        return 0
    logger.error("✗ Parsing step did not complete all inputs")
    return 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 9: Parse Entries Generic")
    parser.add_argument("dataset", help="Name of the dataset")
    parser.add_argument("--config", help="Path to JSON schema config", required=True)
    args = parser.parse_args()
    
    exit_code = main(args.dataset, args.config)
    sys.exit(exit_code)
