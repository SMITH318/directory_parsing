#!/usr/bin/env python3
"""
OCR Processing Script - Gemini Vision Edition
Modified for new API and batching.
Processes preprocessed PDF snippets using Gemini Vision API to extract text and bounding boxes.
"""
# TODO: Change to access files from Cloud Bucket

import datetime
import json
import gc
#import cv2
import numpy as np
from pathlib import Path
from google import genai
from google.genai import types
from google.genai import errors
from collections.abc import Callable
from PIL import Image
import os
import time
import pandas as pd
from  pydantic import BaseModel


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
  filename='02_gemini_batch_mass.log', 
  filemode='a', 
  encoding='utf-8', 
  level=logging.WARNING) ## <=================== Change logging level here

# ***************************** constants *****************************
SKIP_TEXT = "******* KEEPS FAILING! SKIPPING FOR NOW *******"
INITIAL_WAIT_SECONDS = 60 * 8 # 8 minutes
FOLLOWUP_WAIT_SECONDS = 60 * 1 # 1 minute
MAX_BATCHES_AT_ONCE = 80 
MODEL_PROMPT = (
    "Your role is to perform OCR on the images you are prompted with. "
    "Extract all text from these images of columns from a medical directory. "
    "For each line of text, provide the text content, its bounding box coordinates "
    "in pixels, and a confidence score (between 0 and 1). "
    "Lines of text extend horizontally across the whole column. "
    "Return ONLY a JSON array with this exact format:\n"
    '[\n'
    '  {"text": "example text", "x": 10, "y": 20, "width": 100, "height": 15, "confidence": 0.95},\n'
    '  {"text": "wierd text; ṽ", "x": 10, "y": 30, "width": 90, "height": 19, "confidence": 0.50},\n'
    '  {"text": "more text", "x": 10, "y": 40, "width": 95, "height": 15, "confidence": 0.90}\n'
    ']\n'
    "Include all text, even small fragments. "
    "Coordinates should be exact, in pixels. "
    "Encode the bounding box in each line's x, y, width, and height fields. "
    "High confidence values close to 1 represent that that the text has been rendered correctly "
    "with little likelihood that the image contained different text. "
    "Lower confidence vales, as low as 0, represent uncertainty about what text the image contained. "
    "Include all symbols, punctuation, and line breaks, even if they look like noise. "
    "Encode special characters properly in JSON as UTF-8 characters, standardizing them across all images. "
    "For instance, there is a small dark cadeuceus at the end of some lines, occasionally followed by a G or N, encode that symbol as ▼. "
    "Places to expect ▼ include \"(l'08) ; ▼\", \"Prowell, James W. (b'73)-Mo.7,'96; (♁) ; ▼\", "
    "\"▼G\", \"'03; (l'03); 1444 N. 31st St.; U*; ▼G\", and \"Marx Bldg.; (A28); S*; ▼N\". "
    "Encode a sun cross or wheel cross that can appear between some right parentheses and dashes as ⊕. "
    "Places to expect ⊕ incude \"McCOLLUM, HERMAN E. (b'77)⊕-Mo.7,\", \"OLSON, EVALD (b'66)⊕-Kan.3,'07; (l'16)\", "
    "\"⊕-Mass.7,'06; Member Mass. Med. Soc.;\", and \"⊕-Pa.1,'00; (l'00); 491 E. State St.;\". "
    "Also, a diamond can appear after dashes and should be encoded as ◊. "
    "Places to expect ◊ include \"Bailey, Alexander Henry-◊; (l'89)\", "
    "\"Johnson, Granville Roswell (b 47) E-◊;\", and \"FREEMAN, JOHN F.-◊; 370 W. 10th St.;\". "
    "A triangle can appear after dashes and should be encoded as △, as in \"Veon, John E. (b'71)-△; (l 17); 2310 B.\" "
    "Stars can appear after one- or two-letter abbreviations and should be encoded as ★. "
    "Places to expect ★ include \"Bldg.; 2-4, 7-8; S★\", \"10-12, 3-5; I★\", "
    "\"Ridotto; 2-4; OALR★\", \"Op★\", and \"office, 314 W. 4th St.; (G1,3); R★\". "
    "† can appear in parentheses after an l, 1, or I, as in "
    "\"(l†)\", \"Harris, J. Monroe-◊; (l'†); not in practice\", "
    "\"Lindner, Carl W. (b'45)-O.9,'75; (l†); re-\", or \"'77, N.Y.5,'77; (1†) ; (A28) ; S\". "
    "♁ or ‡ can appear by themselves in parentheses, "
    "as in \"'89; (♁); S\", \"Stone, Robt. E.-Ga.5,'91; (♁); 648 Wood-\", "
    "\"Md.1,'82; (‡); 2927 St. Paul St.; 4-6\". or \"Murdock, Jos. L. (b'62)-Ga.10,'93; (‡)\". "
    "For punctuation like dashes, quotes, and apostrophes, use standard ASCII equivalents. "
    "Avoid encoding anything as non-ASCII characters beyond ▼, ⊕, ◊, ★, †, ♁, and ‡. "
    "Non-ASCII characters besides ▼, ⊕, ◊, ★, †, ♁, and ‡ decrease the confidence score. "
)

# --- Gemini API Configuration --- 
API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY')
MODEL_NAME ='gemini-flash-latest' # gemini-3-flash-preview <- is what this has been in 2/2026

if API_KEY == 'YOUR_API_KEY' or not API_KEY:
    print("ERROR: Gemini API key is not set.")
    logger.error("ERROR: Gemini API key is not set.")
    exit(1)

# Initialize Gemini
print("Initializing Gemini for OCR...")
#print(f"Key: {API_KEY}")
logger.info(f"Initializing Gemini for OCR... with {MODEL_NAME}")
client = genai.Client(api_key=API_KEY)

def cache_system_prompt():
    return client.caches.create(
        model=MODEL_NAME,
        config=types.CreateCachedContentConfig(
            system_instruction=MODEL_PROMPT,
            # ttl=
        )
)
cache = cache_system_prompt()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.float32, np.float64)):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
    
class OCRLine(BaseModel):
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float

class OCRResult(BaseModel):
    lines: list[OCRLine]

def drop_some_finished(self, finished_df: pd.DataFrame) -> pd.DataFrame:
    finished_df = finished_df[finished_df['text'] != SKIP_TEXT]
    return finished_df

def read_input_drop_completed(
        file_in: Path, 
        finished_files: list[Path], 
        file_in_reader: Callable[[Path], pd.DataFrame], 
        file_done_reader: Callable[[Path], pd.DataFrame]
    ) -> pd.DataFrame:
    data_in = file_in_reader(file_in)
    finished_data = [file_done_reader(file) for file in finished_files]
    finished_df = pd.concat(finished_data, ignore_index=True)
    finished_df = drop_some_finished(finished_df)
    finished_keys = finished_df[['pub', 'page', 'col']].agg().drop_duplicates()
    return data_in[
        ~data_in[['pub', 'page', 'col']].agg().isin(finished_keys)
    ]

def prepare_for_request(request_df: pd.DataFrame) -> types.UserContent:
    """Upload snippet image and create request to extract text and bounding boxes it using Gemini Vision API."""
    if not Path(request_df['path']).exists():
        logger.error(f"Image file missing: {request_df['path']}")
        raise FileNotFoundError
    # upload image
    file = client.files.upload(file=request_df['path'])
    content = types.UserContent([
        types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type)
    ])
    return content

def create_batch_request(
        client:genai.Client, 
        model_name:str, 
        model_prompt:str,
        req_name:str, 
        contents:list[types.Content], 
        entries_type:type, 
        cache:types.CachedContent
    ) -> types.BatchJob:

    return client.batches.create(
        model=model_name,
        src=types.BatchJobSource( 
            inlined_requests=
            [
                types.InlinedRequest(
                    contents = contents,
                    config = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=entries_type.model_json_schema(),
                        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"), 
                        temperature=0.0,
                        cached_content=cache.name,
                        # systemInstruction= types.ModelContent(model_prompt)
                    )
                )
            ]
        ),
        config=types.CreateBatchJobConfig(display_name = req_name)
    )

def prepare_batch_requests(
        client: genai.Client, 
        logger, 
        model_name: str, 
        model_prompt: str, 
        entry_type_name: str, 
        entries_type: type,
        cache: types.CachedContent,
        all_inputs: pd.DataFrame, 
        only_count_tokens: bool, 
        max_batches_at_once: int, 
        initial_wait_seconds: int, 
        max_attempts: int = 2, 
        prompts_file: Path = None,
    ) -> list[types.BatchJob]:
    """Prepare batch requests and upload images for Gemini Vision OCR processing."""
    """Store offsets for bounding box adjustment."""
    """Returns list of request dicts for batch processing."""

    next_entry = 0
    jobs = []
    num_batches = 0
    while next_entry < len(all_inputs) and num_batches < max_batches_at_once:
        
        # 2. Save df and upload it
        current_df = all_inputs.iloc[next_entry]

        # 3. Send parsing prompts
        logger.info(f"OCRing Snippet for: {current_df['pdf_name']}, Page {current_df['page_num']}.{current_df['col_idx']}")
        print(f"OCRing Snippet for: {current_df['pdf_name']}, Page {current_df['page_num']}.{current_df['col_idx']}...", end=" ", flush=True)
        request_key = f"gemini_ocr_{os.path.basename(current_df['path'])}"

        attempt = 0
        while attempt < max_attempts:
            try:
                prep_content = prepare_for_request(current_df)
                if only_count_tokens:
                    print(
                        "\tprompt tokens:", 
                        client.models.count_tokens(model=model_name, contents=prep_content).total_tokens
                    )
                else:
                    if cache:
                        client.caches.update( # extend cache time
                            name = cache.name,
                            config  = types.UpdateCachedContentConfig(
                                ttl=f'{initial_wait_seconds * 2}s'
                            )
                        )
                    job = create_batch_request(client, model_name, model_prompt, request_key, prep_content, entries_type, cache)
                    if job:
                        jobs.append(job)
                    print(" Prepared.")
            except Exception as e:
                print(f"\n    Error preparing snippet {request_key}, attempt {attempt + 1}/{max_attempts}: {e}")
                logger.error(f"Error preparing snippet {request_key}, attempt {attempt + 1}/{max_attempts}: {e}")
                attempt += 1
            else:
                next_entry += 1
                num_batches += 1
                break # attempt while
        # end attempt while
    # end batches while

    gc.collect()
    if len(jobs) == 0:
        print(f"No snippets prepared, all finished(?). Exiting.")
        logger.error(f"No snippets prepared, all finished(?). Exiting.")
        exit(0)
    print(f"Total snippets prepared for OCR: {len(jobs)}")
    logger.info(f"Total snippets prepared for OCR: {len(jobs)}")
    
    return jobs

def split_key(key:str) -> dict:
    # parse example key: gemini_ocr_Alabama_p01_r00_c000.jpg
    key_parts = key.split('.')[0].split('_')
    if len(key_parts) < 6:
        logger.error(f"Invalid key format: {key}")
        return ({}, [])
    state = key_parts[2]
    page_str = key_parts[3][1:]  # remove 'p' prefix
    row_str = key_parts[4][1:]  # remove 'r' prefix
    col_str = key_parts[5][1:]  # remove 'c' prefix 
    return {
        "state": state,
        "page": int(page_str),
        "row": int(row_str),
        "column": int(col_str)
    }

def gemini_extract_snippet(key: str, json_text: str) -> tuple[dict[str, int, int, int], list[OCRLine]]:
    """Extract text blocks from Gemini Vision API response."""

    info = split_key(key)
    # print("************ gemini_extract_snippet")
    # print(json_text)
    try:
        text_blocks = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response for key {key}")
        return (info, [])

    if not isinstance(text_blocks, list):
        logger.error(f"Response is not a JSON array for key {key}")
        return (info, [])

    for block in text_blocks:
        if block.get("confidence", 0) < 0.90:
            logger.warning(f"low confidence ({block.get('confidence', 0)}) produced `{block.get('text', '')}` in {key}")
    return (info, text_blocks)

def process_batch_ocr_output(
        batch_job: types.BatchJob, 
        output_file: Path
    ) -> bool:
    """Process the output of a Gemini Vision batch OCR job. Return True if successful (no errors), False otherwise."""
    success = False
    if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
        print(f"Job did not succeed, unexpected final state: {batch_job.state.name}")
        logger.error(f"Job did not succeed, unexpected final state: {batch_job.state.name}")
        success = False
    else:
        # print(batch_job)
        batch_job = client.batches.get(name=batch_job.name)
        # print(batch_job.dest)
        # print(batch_job.dest.inlined_responses)
        responses = batch_job.dest.inlined_responses
        
        success = process_batch_ocr_output_content(batch_job.display_name, responses, output_file)
    client.batches.delete(name=batch_job.name)
    # exit(0)
    return success

# def process_batch_ocr_output_file(content_file: Path, offsets_file: Path, output_file: Path) -> tuple[bool, dict]:
#     with open(content_file, 'r', encoding='utf-8') as f_in:
#         return process_batch_ocr_output_content(f_in.read(), offsets_file, output_file)

def process_batch_ocr_output_content(
        logger: logging.Logger, 
        entry_type_name: str, 
        entry_type: str, 
        display_name: str, 
        responses: list[types.InlinedResponse], 
        output_file: Path
    ) -> bool:
    successful = True

    # parse each inlined response

    for response in responses:
        content_response = response.response
        logger.debug(content_response)

        finish_reason = content_response.candidates[0].finish_reason
        if finish_reason != "STOP":
            successful = False
            print(f"Unexpected finishReason: {finish_reason} in {display_name}")
            logger.error(f"Unexpected finishReason ({content_response.model_version}): {finish_reason} in {display_name}")
            # logger.info(json.dumps(content_response, indent=2))
            if finish_reason == "MAX_TOKENS":
                logger.warning(f"Total tokens used: {content_response.usage_metadata.total_token_count}")
                create_batch_request(
                    display_name,
                    [
                        content_response.candidates[0].content,
                        types.UserContent(["Finish extracting the text"])#.model_dump(exclude_none=True)
                    ]
                )
                continue
                ## rebatch, continue
        else:
            if not save_job_output_content(logger, display_name, content_response.parts[0].text, output_file, entry_type, responses_file, entry_type_name):
                successful = False
    return successful

def save_job_output_content(logger, display_name, response_text:str, output_file, entry_type: type, responses_file:Path|None = None, entry_type_name:str) -> bool:
    info, gemini_output = gemini_extract_snippet(display_name, response_text)
    # print("gemini_extract_snippet output:", info)
    # print(len(gemini_output))

    # find offsets for this snippet
    # offset_row = offsets_file_df[offsets_file_df['key'] == display_name].iloc[0]

    logger.info(f"received {len(gemini_output)} {entry_type_name} entries at {datetime.datetime.now()}")
    print(f"\treceived {len(gemini_output)} {entry_type_name} entries at {datetime.datetime.now()}")

    # 4. Save 
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for block in gemini_output:    
            try:
                # print(block["text"].strip())
                entry = {
                        "pub": info["state"],
                        "page": info["page"],
                        "col": info["column"],
                        "text": block["text"].strip(),
                        "conf": round(block["confidence"], 4),
                        "x": block["x"],# + offset_row['x_offset'],
                        "y": block["y"],# + offset_row['y_offset'],
                        "width": block["width"],
                        "height": block["height"]
                    }
                f_out.write(json.dumps(entry, cls=NumpyEncoder, ensure_ascii=False) + "\n")
                
            except Exception as e:
                logger.error(f"Error processing block: {block} in snippet {info} ({display_name}), error: {e}")
                successful = False
                continue
    return successful

def main(
        initial_wait_seconds=-1, 
        followup_wait_seconds=-1, 
        max_batches_at_once = MAX_BATCHES_AT_ONCE
    ):

    # 1. Setup Project Paths
    script_dir = Path(__file__).parent
    project_root = script_dir if (script_dir / "data").exists() else script_dir.parent
    preprocessed_dir = project_root / "data" / "01_preprocessed"
    metadata_path = preprocessed_dir / "all_metadata.json"
    output_dir = project_root / "data" / "02_raw_batch_mass"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ocr_output.jsonl"
    done_file = output_file
    
    # Check if any batch jobs are in progress
    if client.batches.list() and client.batches.list()[0]:
        print(f"Batch jobs already in progress.")
        logger.warning(f"Batch jobs already in progress.")
        
    else:
        print("No current batch job found. Preparing new batch job...")
        logger.info("No current batch job found. Preparing new batch job...")

        #1. Load the data
        if not metadata_path.exists():
            print(f"Error: Run preprocess.py first. Missing: {metadata_path}")
            logger.error(f"Error: Run preprocess.py first. Missing: {metadata_path}")
            exit(1)
        
        inputs = read_input_drop_completed(
            metadata_path, 
            done_file if done_file != output_file else [done_file, output_file],
            lambda f: pd.read_json(f, encoding='utf-8', lines=False),
            lambda f: pd.read_json(f, encoding='utf-8', lines=True)
        )

        # 2. Initiate batch requests
        print(f"Initiating Batch OCR requests (n={max_batches_at_once})")
        logger.info(f"Initiating Batch OCR requests (n={max_batches_at_once})")
        jobs = prepare_batch_requests(
            client, 
            logger, 
            MODEL_NAME, 
            MODEL_PROMPT, 
            "columns", 
            OCRResult, 
            cache, 
            inputs, 
            only_count_tokens=False, 
            max_batches_at_once=max_batches_at_once, 
            initial_wait_seconds=initial_wait_seconds
        )

        # 3. Wait for batch jobs to complete
        if initial_wait_seconds > 0:
            print(f"Waiting {initial_wait_seconds/60} minutes for {len(jobs)} batch jobs to complete at {datetime.datetime.now()}...")
            logger.error(f"Waiting {initial_wait_seconds/60} minutes for {len(jobs)} batch jobs to complete at {datetime.datetime.now()}...")
            time.sleep(initial_wait_seconds)
        else:
            print(f"No initial wait time specified, leaving {len(jobs)} jobs pending.")
            logger.error(f"No initial wait time specified, leaving {len(jobs)} jobs pending.")
            exit(0)
    
    # 4. While any batch jobs are pending, check if any are done
    wait_and_process_jobs(
        client, 
        logger, 
        followup_wait_seconds,
        lambda job : process_batch_ocr_output(job, output_file)
    )


    print(f"\n✓ Success! OCR Process Complete. Data in: {output_file}")
    logger.error(f"OCR Process Complete. Saved data in: {output_file}")

if __name__ == "__main__":
    for i in range(1000):
        try:
            print("*** Iteration", i, "***")
            main(initial_wait_seconds=INITIAL_WAIT_SECONDS, followup_wait_seconds=FOLLOWUP_WAIT_SECONDS)
        except errors.APIError as e:
            if e.code == 429:
                print(f"*** main loop RESOURCE_EXHAUSTED exception, pausing for {INITIAL_WAIT_SECONDS/60} at {datetime.datetime.now()}... ***")
                logger.error(f"*** main loop RESOURCE_EXHAUSTED exception, pausing for {INITIAL_WAIT_SECONDS/60} at {datetime.datetime.now()}... ***")
                time.sleep(INITIAL_WAIT_SECONDS)
        except Exception as e:
            print("*** main loop exception, pressing on ***")
            print(type(e).__name__, "–", e)
            # something went very wrong, consider scrub any ongoing batch jobs and the cache
            for job in client.batches.list():
                try:
                    client.batches.delete(name=job.name)
                except:
                    pass
            try:
                client.caches.delete(name=cache.name)
            except:
                pass
            cache = cache_system_prompt()
            pass

client.caches.delete(name=cache.name)
client.close()