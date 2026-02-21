#!/usr/bin/env python3
"""
OCR Processing Script - Gemini Vision Edition
Modified for new API and batching.
Processes preprocessed PDF snippets using Gemini Vision API to extract text and bounding boxes.
"""
# TODO: Change to access files from Cloud Bucket
# TODO: Cache system instruction/model prompt for reuse, more throughput

import datetime
import json
import gc
#import cv2
import numpy as np
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image
import os
import time
import pandas as pd
from  pydantic import BaseModel

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
  filename='02_gemini_batch.log', 
  filemode='a', 
  encoding='utf-8', 
  level=logging.WARNING) ## <=================== Change logging level here

# --- Gemini API Configuration --- 
API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY')
model_name ='gemini-flash-latest' # gemini-3-flash-preview <- is what this has been in 2/2026

if API_KEY == 'YOUR_API_KEY' or not API_KEY:
    print("ERROR: Gemini API key is not set.")
    logger.error("ERROR: Gemini API key is not set.")
    exit(1)

# Initialize Gemini
print("Initializing Gemini for OCR...")
#print(f"Key: {API_KEY}")
logger.info(f"Initializing Gemini for OCR... with {model_name}")
client = genai.Client(api_key=API_KEY)

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

def initiate_batch(requests, batch_file_path, current_batch_job_file, initial_wait_seconds):
    ## 3.1 Write batch requests to JSONL
    logger.info(f"Writing batch requests to {batch_file_path}")
    with open(batch_file_path, 'w') as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
    
    ## 3.2 Upload batch file to Gemini
    logger.debug(f"Uploading file: {batch_file_path}")
    uploaded_batch_requests = client.files.upload(
        file=batch_file_path,
        config=types.UploadFileConfig(mime_type='jsonl', display_name='batch-input-file')
    )
    print(f"uploaded file state: {uploaded_batch_requests.state.name}")
    print(f"uploaded file uri: {uploaded_batch_requests.uri}")
    print(f"Uploaded request name: {uploaded_batch_requests.name}")
    logger.debug(f"Uploaded request name: {uploaded_batch_requests.name}")
    
    ## 3.3 Create batch processing job
    batch_job_from_file = client.batches.create(
        model=model_name,
        src=uploaded_batch_requests.name,
        config=types.CreateBatchJobConfig(display_name='Directory OCR Batch Job')
    )
    job_name = batch_job_from_file.name
    print(f"Created batch job at {datetime.datetime.now()} from file: {job_name}")
    logger.warning(f"Created batch job at {datetime.datetime.now()} from file: {job_name}")
    with open(current_batch_job_file, 'w') as f:
        f.write(job_name)
    if initial_wait_seconds > 0:
        print(f"Waiting {initial_wait_seconds/60} minutes for batch job to complete...")
        logger.error(f"Waiting {initial_wait_seconds/60} minutes for batch job to complete at {datetime.datetime.now()}...")
        time.sleep(initial_wait_seconds)
        return job_name
    else:
        print("No initial wait time specified, leaving job pending.")
        logger.error("No initial wait time specified, leaving job pending.")
        exit(0)

def gemini_finish_thinking(key_contents: dict[str:types.Content], batch_file, current_batch_job_file, initial_wait_seconds, followup_wait_seconds):
    reqs = []
    print(f"** Try finishing thinking on {len(key_contents)} jobs")
    logger.warning(f"Try finishing thinking on {len(key_contents)} jobs")
    with open(batch_file, 'r', encoding='utf-8') as batch_in:
        for line in batch_in.readlines():
            prompt = json.loads(line)
            if prompt["key"] in key_contents:
                # print(key_contents[prompt["key"]])
                prompt["request"]["contents"] = [ # contents starts a dictionary, replace it with a list starting with its old contents
                    prompt["request"]["contents"],
                    key_contents[prompt["key"]],
                    types.UserContent(["Finish extracting the text"]).model_dump(exclude_none=True)
                ]
                reqs.append(prompt)

    job_name = initiate_batch(reqs, batch_file, current_batch_job_file, initial_wait_seconds)
    
    # 4. Wait for completion of batch job - Poll the job status until it's completed.
    return wait_for_job(job_name, followup_wait_seconds)
 
                

def gemini_prepare_snippet(image_path: Path) -> dict:
    """Upload snippet image and create request to extract text and bounding boxes it using Gemini Vision API."""
    request_key = f"gemini_ocr_{os.path.basename(image_path)}"

    try:
        # upload image
        file = client.files.upload(file=image_path)

        user_prompt = "Extract all text from this image of a column from a directory. "
        model_prompt = (
            "Extract all text from these images of columns from a directory. "
            "For each line of text, provide the text content, its bounding box coordinates, and a confidence score (between 0 and 1). "
            "Lines of text extend horizontally across the whole column. "
            "Return ONLY a JSON array. Coordinates should be in pixels. "
            "Include all text, even small fragments. "
            "Include all symbols, punctuation, and line breaks, even if they look like noise. "
            "Encode special characters properly in JSON as UTF-8 characters, standardizing them across all images. "
            "For instance, there is a small dark cadeuceus at the end of some lines, occasionally followed by a G or N, encode that symbol as ▼. "
            "Encode a sun cross or wheel cross that can appear between some right parentheses and dashes as ⊕. "
            "Also, a diamond can appear after dashes and should be encoded as ◊. "
            "For punctuation like dashes, quotes, and apostrophes, use standard ASCII equivalents."
        )
                
        return {
            "key": request_key,
            "request": {
                "contents": types.UserContent(
                    [
                        types.Part.from_text(text=user_prompt),
                        types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type)
                    ]
                ).model_dump(exclude_none=True),
                "generationConfig": types.GenerateContentConfig(
                    response_mime_type="application/json", 
                    response_json_schema=OCRResult.model_json_schema(),
                    temperature=0.0,
                    max_output_tokens=100000 # default is 65,536; expanding because some columns cut off midway
                ).model_dump(exclude_none=True),
                "systemInstruction": types.ModelContent(model_prompt).model_dump(exclude_none=True)
            }
        }

        # return {
        #     "key": request_key,
        #     "request": {
        #         "contents": [{ 
        #             "parts": [ 
        #                 { "text": prompt },
        #                 { "fileData": {
        #                     "mimeType": file.mime_type,
        #                     "fileUri": file.uri
        #                     } 
        #                 }
        #             ]
        #         }],
        #         "generationConfig": { 
        #             "responseMimeType": "application/json",
        #             "responseJsonSchema":  OCRResult.model_json_schema() # works
        #         }
        #     }
        # }

    except Exception as e:
        print(f"\n    Error preparing snippet {request_key}: {e}")
        logger.error(f"Error preparing snippet {request_key}: {e}")
        raise e
    return None

def prepare_batch_requests(all_metadata: dict, df_done: pd.DataFrame, offsets_file: Path, max_to_prep: int = -1) -> list[dict]:
    """Prepare batch requests and upload images for Gemini Vision OCR processing."""
    """Store offsets for bounding box adjustment."""
    """Returns list of request dicts for batch processing."""
    offsets = []
    requests_data = []
    try:
        for pdf_meta in all_metadata:
            pdf_name = pdf_meta['source_pdf']
            print(f"\nPackaging Snippets for: {pdf_name}")
            logger.info(f"\nPackaging Snippets for: {pdf_name}")

            for page_meta in pdf_meta['pages']:
                page_num = page_meta['page_num']
                print(f"  Page {page_num}...", end="", flush=True)
                logger.info(f"  Page {page_num}...")

                # Prepare each snippet (article)
                for snip in page_meta['snippets']:
                    col_idx = snip.get('col_idx', snip.get('column', 0))
                    print(f"{col_idx}.", end="", flush=True)

                    # Skip already processed snippets
                    if ((df_done['pub'] == pdf_name) & 
                            (df_done['page'] == page_num) &
                            (df_done['col'] == col_idx)).any():
                        logger.info(f"Previously processed column {col_idx}")
                        print(f"Previously processed", end=" ", flush=True)
                        continue
                    
                    snippet_image_path = snip['path']
                    if not Path(snippet_image_path).exists():
                        logger.error(f"Image file missing: {snippet_image_path}")
                        continue

                    request = gemini_prepare_snippet(snippet_image_path)
                    if request:
                        requests_data.append(request)

                        # store offsets for response processing
                        offsets.append(
                            {
                                "key": request["key"],
                                "x_offset": snip['x_offset'],
                                "y_offset": snip['y_offset']
                            }
                        )
                        if max_to_prep > 0 and len(requests_data) >= max_to_prep:
                            print(f"\nReached max to prepare limit of {max_to_prep}. Stopping preparation.")
                            logger.info(f"Reached max to prepare limit of {max_to_prep}. Stopping preparation.")
                            raise StopIteration
    

                print(" Prepared.")
    except StopIteration:
        pass
    offsets_df = pd.DataFrame(offsets)
    offsets_df.to_csv(offsets_file, index=False)
    gc.collect()
    print(f"Total snippets prepared for OCR: {len(requests_data)}")
    logger.info(f"Total snippets prepared for OCR: {len(requests_data)}")
    return requests_data

def gemini_extract_snippet(key: str, json_text: str) -> tuple[dict[str, int, int, int], list[OCRLine]]:
    """Extract text blocks from Gemini Vision API response."""

    # parse example key: gemini_ocr_Alabama_p01_r00_c000.jpg
    key_parts = key.split('.')[0].split('_')
    if len(key_parts) < 6:
        logger.error(f"Invalid key format: {key}")
        return ({}, [])
    state = key_parts[2]
    page_str = key_parts[3][1:]  # remove 'p' prefix
    row_str = key_parts[4][1:]  # remove 'r' prefix
    col_str = key_parts[5][1:]  # remove 'c' prefix 
    info = {
        "state": state,
        "page": int(page_str),
        "row": int(row_str),
        "column": int(col_str)
    }

    try:
        json_response = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response for key {key}")
        return (info, [])

    text_blocks = json_response.get("lines", [])

    if not isinstance(text_blocks, list):
        logger.error(f"Response is not a JSON array for key {key}")
        return (info, [])

    for block in text_blocks:
        if block.get("confidence", 0) < 0.90:
            logger.warning(f"low confidence ({block.get('confidence', 0)}) produced `{block.get('text', '')}` in {key}")
    return (info, text_blocks)

def process_batch_ocr_output(batch_job: types.BatchJob, offsets_file: Path, result_file_local: Path, output_file: Path) -> tuple[bool, dict]:
    """Process the output of a Gemini Vision batch OCR job. Return True if successful (no errors), False otherwise."""
    if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
        print(f"Job did not succeed, unexpected final state: {batch_job.state.name}")
        logger.error(f"Job did not succeed, unexpected final state: {batch_job.state.name}")
        return False
    else:
        # The output is in another file. Download and parse it.
        result_file_name = batch_job.dest.file_name
        print(f"Results are in file: {result_file_name}")
        print(f"Downloading and parsing result file content to {result_file_local}...")
        logger.info(f"Downloading results file: {result_file_name}")

        file_content_bytes = client.files.download(file=result_file_name)
        file_content = file_content_bytes.decode('utf-8')    

        # save the results file locally
        with open(result_file_local, 'w', encoding = 'utf-8') as f:
            f.write(file_content)
            logger.error(f"Downloaded results file {result_file_local}")
        return process_batch_ocr_output_content(file_content, offsets_file, output_file)

def process_batch_ocr_output_file(content_file: Path, offsets_file: Path, output_file: Path) -> tuple[bool, dict]:
    with open(content_file, 'r', encoding='utf-8') as f_in:
        return process_batch_ocr_output_content(f_in.read(), offsets_file, output_file)

def process_batch_ocr_output_content(content: str, offsets_file: Path, output_file: Path) -> tuple[bool, dict]:
    successful = True
    offsets_file_df = pd.read_csv(offsets_file)
    needs_more_thinking = {} #{key: content}

    # The result file is also a JSONL file. Parse each line.
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for line in content.splitlines():
            if line:
                parsed_response = json.loads(line)
                # Pretty-print the JSON for readability
                logger.debug(json.dumps(parsed_response, indent=2))
                logger.debug("-" * 20)
                content_response = types.GenerateContentResponse.model_validate(parsed_response["response"])

                finish_reason = content_response.candidates[0].finish_reason#parsed_response["response"]["candidates"][0]["finishReason"]
                if finish_reason != "STOP":
                    print(f"Unexpected finishReason: {finish_reason} in {parsed_response['key']}")
                    logger.warning(f"Unexpected finishReason ({content_response.model_version}): {finish_reason} in {parsed_response['key']}")
                    logger.info(json.dumps(parsed_response, indent=2))
                    if finish_reason == "MAX_TOKENS":
                        logger.warning(f"Total tokens used: {content_response.usage_metadata.total_token_count}")
                        needs_more_thinking[parsed_response["key"]] = {
                            "role": "user",
                            "parts": parsed_response["response"]["candidates"][0]["content"]["parts"]
                        }
                        continue
                        ## rebatch, continue

                info, gemini_output = gemini_extract_snippet(
                    parsed_response["key"], 
                    content_response.parts[0].text # parts of first content
                )

                # find offsets for this snippet
                offset_row = offsets_file_df[offsets_file_df['key'] == parsed_response["key"]].iloc[0]

                for block in gemini_output:
                    try:
                        entry = {
                                "pub": info["state"],
                                "page": info["page"],
                                "col": info["column"],
                                "text": block["text"].strip(),
                                "conf": round(block["confidence"], 4),
                                "x": block["x"] + offset_row['x_offset'],
                                "y": block["y"] + offset_row['y_offset'],
                                "width": block["width"],
                                "height": block["height"]
                            }
                        f_out.write(json.dumps(entry, cls=NumpyEncoder, ensure_ascii=False) + "\n")
                        
                    except Exception as e:
                        logger.error(f"Error processing block: {block} in snippet {info} ({parsed_response['key']}), error: {e}")
                        successful = False
                        continue
    return successful, needs_more_thinking

def wait_for_job(job_name, followup_wait_seconds):
    while True:
        batch_job = client.batches.get(name=job_name)
        if batch_job.state.name in ('JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'):
            break
        if followup_wait_seconds < 0:
            print(f"Negative followup wait seconds detected: {followup_wait_seconds}. Not waiting, leaving job pending.")
            logger.error(f"Negative followup wait seconds detected: {followup_wait_seconds}. Not waiting, leaving job pending.")
            exit(0)
        print(f"Job not finished at {datetime.datetime.now()}. Current state: {batch_job.state.name}. Waiting {followup_wait_seconds/60} minutes...")
        time.sleep(followup_wait_seconds)

    print(f"Job finished with state: {batch_job.state.name}")
    logger.warning(f"Job finished with state: {batch_job.state.name}")
    if batch_job.state.name == 'JOB_STATE_FAILED':
        print(f"Batch job error: {batch_job.error}")
        logger.error(f"Batch job error: {batch_job.error}")
        exit(1)
    return batch_job

def main(initial_wait_seconds=-1, followup_wait_seconds=-1, max_to_batch = 4):
      
    # 1. Setup Project Paths
    script_dir = Path(__file__).parent
    project_root = script_dir if (script_dir / "data").exists() else script_dir.parent
    preprocessed_dir = project_root / "data" / "01_preprocessed"
    output_dir = project_root / "data" / "02_raw_batch"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = preprocessed_dir / "all_metadata.json"
    batch_file_path = output_dir / 'batch_requests.jsonl'
    current_batch_job_file = output_dir / "current_batch_job.txt"
    offsets_file = output_dir / "snippet_offsets.csv"
    result_file_path = output_dir / 'batch_results.jsonl'
    output_file = output_dir / "ocr_output.jsonl"

    # Check if batch job is already in progress, based on file
    if current_batch_job_file.exists():
        print(f"A batch job is already in progress, checking {current_batch_job_file} for job name.")
        logger.error(f"A batch job is already in progress, checking {current_batch_job_file} for job name.")
        with open(current_batch_job_file, 'r') as f:
            job_name = f.read().strip()
            if not job_name:
                print(f"Error: current batch job file is empty: {current_batch_job_file}")
                logger.error(f"Error: current batch job file is empty: {current_batch_job_file}")
                exit(1)
    else:
        print("No current batch job found. Preparing new batch job...")
        logger.info("No current batch job found. Preparing new batch job...")

        # Check if metadata file exists
        if not metadata_path.exists():
            print(f"Error: Run preprocess.py first. Missing: {metadata_path}")
            logger.error(f"Error: Run preprocess.py first. Missing: {metadata_path}")
            return

        with open(metadata_path, 'r') as f:
            all_metadata = json.load(f)

        # 2. Prepare batch requests
        print(f"Packaging Snippets for Batch OCR. Batch requests will be saved to: {batch_file_path}")
        logger.info(f"Packaging Snippets for Batch OCR. Batch requests will be saved to: {batch_file_path}")
        try:
            df_done = pd.read_json(output_file, lines=True)[['pub', 'page', 'col']].drop_duplicates()
        except:
            df_done = pd.DataFrame(columns=['pub', 'page', 'col'])

        requests_data = prepare_batch_requests(all_metadata, df_done, offsets_file, max_to_prep = max_to_batch)

        # 3. initiate batch ocr processing
        job_name = initiate_batch(requests_data, batch_file_path, current_batch_job_file, initial_wait_seconds)
    
    # 4. Wait for completion of batch job - Poll the job status until it's completed.
    batch_job = wait_for_job(job_name, followup_wait_seconds)
 

    # 5. Retrieve, process and save results
    success, needs_more_thinking = process_batch_ocr_output(batch_job, offsets_file, result_file_path, output_file)
    if success:
        # delete current batch job file
        current_batch_job_file.unlink(missing_ok=True)
        if needs_more_thinking:
            rethinking_batch_job = gemini_finish_thinking(
                needs_more_thinking, 
                batch_file_path, 
                current_batch_job_file, 
                initial_wait_seconds, 
                followup_wait_seconds
            )
            success, needs_more_thinking = process_batch_ocr_output(rethinking_batch_job, offsets_file, result_file_path, output_file)
            current_batch_job_file.unlink(missing_ok=True)
            if not success or needs_more_thinking:
                print(f"Rethinking didn't help")
                logger.warning(f"Rethinking didn't help")


    print(f"\n✓ OCR Process Complete. Data in: {output_file}")
    logger.error(f"\n✓ OCR Process Complete. Data in: {output_file}")

if __name__ == "__main__":
    INITIAL_WAIT_SECONDS = 60 * 10 # 10 minutes
    FOLLOWUP_WAIT_SECONDS = 60 * 5 # 5 minutes
    for i in range(1000):
        main(initial_wait_seconds=INITIAL_WAIT_SECONDS, followup_wait_seconds=FOLLOWUP_WAIT_SECONDS)

    ##### setup for testing handling downloaded responses ####
    # script_dir = Path(__file__).parent
    # project_root = script_dir if (script_dir / "data").exists() else script_dir.parent
    # output_dir = project_root / "data" / "02_raw_batch"

    # offsets_file = output_dir / "snippet_offsets.csv"
    # result_file_path = output_dir / 'batch_results.jsonl'
    # output_file = output_dir / "ocr_output.jsonl"

    # process_batch_ocr_output_file(result_file_path, offsets_file, output_file)


client.close()