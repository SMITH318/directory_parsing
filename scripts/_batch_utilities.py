
import datetime
import json
import gc
from pathlib import Path
from google import genai
from google.genai import types
from collections.abc import Callable
import time
import pandas as pd
import csv
import logging

def read_input_drop_completed(
        file_in: Path, 
        finished_files: list[Path], 
        file_in_reader: Callable[[Path], pd.DataFrame], 
        file_done_reader: Callable[[Path], pd.DataFrame]
    ) -> pd.DataFrame:
    data_in = file_in_reader(file_in)
    finished_data = [file_done_reader(file) for file in finished_files]
    finished_df = pd.concat(finished_data, ignore_index=True)
    finished_keys = finished_df[['pub', 'page', 'col']].agg().drop_duplicates()
    return data_in[
        ~data_in[['pub', 'page', 'col']].agg().isin(finished_keys)
    ]

def prepare_for_request(request_df: pd.DataFrame) -> types.UserContent:
    file = request_df.to_csv(index=False, encoding="utf-8") # produces string if not given file
    content = types.UserContent([file])#[types.Part.from_text(text=file)]
        
    # upload blocks file
    # file = client.files.upload(
    #     file=temp_file,
    #     config=types.UploadFileConfig(mime_type='text/csv')#(mime_type='text/csv; charset=UTF-8')
    # )
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
                        cached_content=cache.name if cache else None,
                        systemInstruction=None if cache else types.Content(role="system", parts = [types.Part(text=model_prompt)]),
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
        max_entries_per_batch: int, 
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
        last_entry = min(next_entry + max_entries_per_batch, len(all_inputs))
        current_df = all_inputs.iloc[next_entry:last_entry]

        # 3. Send prompts
        logger.info(f"prompting for {entry_type_name} ({next_entry}:{last_entry}) out of {len(all_inputs)} at {datetime.datetime.now()}")
        print(f"prompting for {entry_type_name} ({next_entry}:{last_entry}) out of {len(all_inputs)} at {datetime.datetime.now()}")
        request_key = f"{entry_type_name}_{next_entry}_{last_entry}"

        attempt = 0
        while attempt < max_attempts:
            try:
                prep_content = prepare_for_request(current_df)
                if prompts_file:
                    # save prompt for tuning
                    with open(prompts_file, 'a') as f:
                        f.write(json.dumps(
                            {
                                "systemInstruction" : types.ModelContent(model_prompt).model_dump(exclude_none=True),
                                "contents": prep_content.model_dump(exclude_none=True)
                            }
                        )+ '\n')
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
                
            except Exception as e:
                print(f"\n    Error preparing job {request_key}, attempt {attempt + 1}/{max_attempts}: {e}")
                logger.error(f"Error preparing job {request_key}, attempt {attempt + 1}/{max_attempts}: {e}")
                attempt += 1
            else:
                next_entry += max_entries_per_batch
                num_batches += 1
                break # attempt while
        # end attempt while
    # end batches while

    gc.collect()
    if len(jobs) == 0:
        print(f"No jobs prepared, all finished(?). Exiting.")
        logger.error(f"No jobs prepared, all finished(?). Exiting.")
        exit(0)
    print(f"Total jobs prepared: {len(jobs)}")
    logger.info(f"Total jobs prepared: {len(jobs)}")
    
    return jobs

def get_finished_jobs(client, logger)-> list[types.BatchJob]:
    finished_jobs = []
    print(f"Checking for finished jobs at {datetime.datetime.now()}")
    for batch_job in client.batches.list():
        print(batch_job.display_name, batch_job.state.name, "created at", batch_job.create_time)
        if batch_job.state.name == 'JOB_STATE_FAILED':
            print(f"Batch job error in {batch_job.display_name}: {batch_job.error}")
            logger.error(f"Batch job error in {batch_job.display_name}: {batch_job.error}")
        elif batch_job.state.name in ('JOB_STATE_SUCCEEDED', 'JOB_STATE_CANCELLED'):
            finished_jobs.append(batch_job)
        
    print(f"{len(finished_jobs)} jobs finished")
    logger.warning(f"{len(finished_jobs)} jobs finished")
    
    return finished_jobs

def wait_and_process_jobs(client, logger, wait_seconds, process_func):
    while client.batches.list() and client.batches.list()[0]:
        print(f">={len(client.batches.list())} batch jobs pending")
        finished_jobs = get_finished_jobs(client, logger)
        if finished_jobs:
            print(f"Processing {len(finished_jobs)} finished jobs at {datetime.datetime.now()}")
            logger.info(f"Processing {len(finished_jobs)} finished jobs at {datetime.datetime.now()}")
            for job in finished_jobs:
                # 5. Retrieve, process and save results
                success = process_func(job)
        else:
            print(f"No finished jobs found at {datetime.datetime.now()}. Waiting {wait_seconds/60} minutes...")
            logger.warning(f"No finished jobs found at {datetime.datetime.now()}. Waiting {wait_seconds/60} minutes...")
            time.sleep(wait_seconds)


def process_job_output(
        client: genai.Client,
        logger: logging.Logger, 
        batch_job: types.BatchJob,
        entry_type_name: str, 
        entry_type: str,
        output_file: Path, 
        responses_file: Path | None  = None
    ) -> bool:
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
        
        success = process_job_output_content(logger, entry_type_name, entry_type, batch_job.display_name, responses, output_file, responses_file)
    client.batches.delete(name=batch_job.name)
    # exit(0)
    return success

def process_job_output_content(
        logger: logging.Logger, 
        entry_type_name: str, 
        entry_type: str, 
        display_name: str, 
        responses: list[types.InlinedResponse], 
        output_file: Path, 
        responses_file: Path | None  = None
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
    entries = json.loads(response_text)
    if responses_file:
        with open(responses_file, 'a') as f:
            f.write(json.dumps(entries) + '\n')
    # entries = json_entries[f"{entry_type_name}_entries"]

    logger.info(f"received {len(entries)} {entry_type_name} entries at {datetime.datetime.now()}")
    print(f"\treceived {len(entries)} {entry_type_name} entries at {datetime.datetime.now()}")

    # 4. Save to CSV
    with open(output_file, 'a', encoding='utf-8', newline='') as f_out:
        entry_writer = csv.DictWriter(f_out, entry_type.model_fields.keys(), restval="")
        for e in entries:
            print(e)
            entry_writer.writerow(e)
    return True

def batch_prompt_dataframe(
        client: genai.Client, 
        logger, 
        model_name: str, 
        model_prompt: str, 
        entry_type_name: str, 
        entry_type: type, 
        entries_type: type, 
        data_set: str,
        only_count_tokens: bool, 
        max_batches_at_once: int, 
        max_entries_per_batch: int, 
        initial_wait_seconds: int, 
        followup_wait_seconds: int,
        max_attempts: int = 2, 
        record_prompts_responses: bool = False
    ):

    # 1. Setup file paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir if (script_dir / "data").exists() else script_dir.parent
    input_file = project_root / "data" / "03_processed_batch" / f"{entry_type_name}_entries_{data_set}.csv"
    output_dir = project_root / "data" / f"04_extracted_entries_gemini_{data_set}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"amd_1918_{entry_type_name}_entries.csv" 
    if record_prompts_responses:
        prompts_file = output_dir / "extracted_entries_prompts.jsonl"
        responses_file = output_dir / "extracted_entries_responses.jsonl"
    
    # Check if any batch jobs are in progress
    if client.batches.list() and client.batches.list()[0]:
        print(f"Batch jobs already in progress.")
        logger.warning(f"Batch jobs already in progress.")
        
    else:
        print("No current batch job found. Preparing new batch job...")
        logger.info("No current batch job found. Preparing new batch job...")

        #1. Load the data
        if not input_file.exists():
            print(f"Error: {input_file} not found.")
            exit(1)

        # create output file, write header for CSV
        if not output_file.exists():
            with open(output_file, 'w', encoding='utf-8', newline='') as csv_out:
                doc_writer = csv.DictWriter(csv_out, entry_type.model_fields.keys(), restval="")
                doc_writer.writeheader()

        inputs = read_input_drop_completed(
            input_file, 
            output_file,
            lambda f: pd.read_csv(f, encoding="utf-8"),
            lambda f: pd.read_csv(f, encoding='utf-8')
        )

        # 2. Initiate batch requests
        print(f"Initiating Batch OCR requests (n={max_batches_at_once})")
        logger.info(f"Initiating Batch OCR requests (n={max_batches_at_once})")
        # cache system prompt if large enough
        model_prompt_tokens = client.models.count_tokens(model=model_name, contents=model_prompt).total_tokens
        cache_model_prompt = model_prompt_tokens >= 1024
        if only_count_tokens:
            print("cached model prompt tokens:", model_prompt_tokens)
        elif cache_model_prompt:
            logger.info("Caching model prompt")
            print("Caching model prompt")
            cache = client.caches.create(
                model=model_name,
                config=types.CreateCachedContentConfig(
                    system_instruction=model_prompt
                )
            )
        else:
            logger.info("Model prompt too small to cache")
            print("Model prompt too small to cache")
        jobs = prepare_batch_requests(
            client, 
            logger, 
            model_name, 
            model_prompt, 
            entry_type_name, 
            entries_type, 
            cache, 
            inputs, 
            only_count_tokens, 
            max_batches_at_once, 
            max_entries_per_batch, 
            initial_wait_seconds
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
        lambda job : process_job_output(client, logger, job, entry_type_name, entry_type, output_file, responses_file)
    )


    print(f"\n✓ Success! {entry_type_name} saved in: {output_file}")
    logger.error(f"Saved {entry_type_name} entries in: {output_file}")

    if cache:
        client.caches.delete(name=cache.name)
