from collections.abc import Generator
from abc import ABC, abstractmethod
from pydantic import ValidationError
import datetime
import json
import gc
import os
from pathlib import Path
from google import genai
from google.genai import types
from google.genai import errors
import time
import pandas as pd
import csv
import logging
from _clean_gemini import *

NUKE_ALL_ON_EXIT = False

class ABatchProcessor:
    def __init__(
            self, 
            logger: logging.Logger, 
            model_name: str, 
            model_prompt: str, 
            entry_type_name: str, 
            entry_type: type, 
            entries_type: type, 
            only_count_tokens: bool, 
            max_batches_at_once: int, 
            max_entries_per_batch: int, 
            initial_wait_seconds: int, 
            followup_wait_seconds: int,
            max_attempts_per_prompt: int = 2, 
        ):
        self.cache = None
        self.logger = logger
        self.model_name = model_name
        self.model_prompt = model_prompt
        self.entry_type_name = entry_type_name
        self.entry_type = entry_type
        self.entries_type = entries_type
        self.only_count_tokens = only_count_tokens
        self.max_batches_at_once = max_batches_at_once
        self.max_entries_per_batch = max_entries_per_batch
        self.initial_wait_seconds = initial_wait_seconds
        self.followup_wait_seconds = followup_wait_seconds
        self.max_attempts_per_prompt = max_attempts_per_prompt
        self.make_next_request_gen = None

        # Initialize Gemini
        api_key = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY')
        if api_key == 'YOUR_API_KEY' or not api_key:
            print("ERROR: Gemini API key is not set.")
            logger.error("ERROR: Gemini API key is not set.")
            exit(1)

        print("Initializing Gemini...")
        logger.info(f"Initializing Gemini for OCR...")
        self.client = genai.Client(api_key=api_key)

    def __del__(self):
        try:
            if self.cache:
                self.client.caches.delete(name=self.cache.name)
        except:
            pass
        if NUKE_ALL_ON_EXIT:
            remove_all_uploaded_files(self.client)
            remove_all_batches(self.client)
            remove_all_caches(self.client)
        try:
            self.client.close()
        except:
            pass
        
    @abstractmethod
    def drop_some_finished(self, finished_df: pd.DataFrame) -> pd.DataFrame:
        """Must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def load_input(self, file_in: Path) -> pd.DataFrame:
        """Must be implemented by subclasses"""
        pass

    @abstractmethod
    def load_finished(self, file_done: Path) -> pd.DataFrame:
        """Must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def df_columns_to_check_finished(self) -> list[str]:
        """Must be implemented by subclasses"""
        pass
    
    def read_input_drop_completed(
            self,
            file_in: Path, 
            finished_files: list[Path]
        ) -> pd.DataFrame:
        data_in = self.load_input(file_in)
        finished_data = [self.load_finished(file) for file in finished_files]
        finished_df = pd.concat(finished_data, ignore_index=True)
        if len(finished_df):
            finished_df = self.drop_some_finished(finished_df)
        if not len(finished_df):
            return data_in
        # print(finished_df.columns)
        cols_to_check = self.df_columns_to_check_finished()
        keys_lambda = lambda row: '-'.join(row.values.astype(str))
        finished_keys = finished_df[
            [col_tup[1] for col_tup in cols_to_check]
        ].apply(keys_lambda, axis=1).drop_duplicates()
        # print(data_in.columns)
        return data_in[
            ~data_in[
                [col_tup[0] for col_tup in cols_to_check]
            ].apply(keys_lambda, axis=1).isin(finished_keys)
        ]

    @abstractmethod
    def prepare_for_request(self, request_df: pd.DataFrame) -> types.UserContent:
        """Must be implemented by subclasses"""
        pass

    def create_batch_request(
            self,
            req_name:str, 
            contents:list[types.Content]
        ) -> types.BatchJob:

        return self.client.batches.create(
            model=self.model_name,
            src=types.BatchJobSource( 
                inlined_requests=
                [
                    types.InlinedRequest(
                        contents = contents,
                        config = types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_json_schema=self.entries_type.model_json_schema(),
                            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"), 
                            temperature=0.0,
                            cached_content=self.cache.name if self.cache else None,
                            systemInstruction=None if self.cache else types.Content(role="system", parts = [types.Part(text=self.model_prompt)]),
                        )
                    )
                ]
            ),
            config=types.CreateBatchJobConfig(display_name = req_name)
        )

    def create_extend_model_prompt_cache(self):
        # if cache already exists, try to extend it
        if self.cache:
            try:
                self.client.caches.update(
                    name = self.cache.name,
                    config  = types.UpdateCachedContentConfig(
                        ttl=f'86400s' # 1 day
                    )
                )
                return
            except ValidationError as e:
                self.logger.info(f"error updating cache '{self.cache.name}', assuming it has expired, create new cache")

        # cache system prompt if large enough
        model_prompt_tokens = self.client.models.count_tokens(model=self.model_name, contents=self.model_prompt).total_tokens
        cache_model_prompt = model_prompt_tokens >= 1024
        if self.only_count_tokens:
            print("cached model prompt tokens:", model_prompt_tokens)
        elif cache_model_prompt:
            self.logger.info("Caching model prompt")
            print("Caching model prompt")
            self.cache = self.client.caches.create(
                model=self.model_name,
                config=types.CreateCachedContentConfig(
                    system_instruction=self.model_prompt
                )
            )
        else:
            self.logger.info("Model prompt too small to cache")
            print("Model prompt too small to cache")

    def prepare_batch_requests(
            self,
            num_jobs: int
        ) -> list[types.BatchJob]:
        """Prepare batch requests and upload images for Gemini Vision OCR processing."""
        """Store offsets for bounding box adjustment."""
        """Returns list of request dicts for batch processing."""

        if not self.make_next_request_gen:
            self.logger.info("No make_next_request_gen, will create in next iteration")
            return []

        # create or extend cache time
        self.create_extend_model_prompt_cache()
        jobs = []
        try:
            for _ in range(num_jobs):
                job = next(self.make_next_request_gen)
                if job:
                    jobs.append(job)
        except StopIteration as e:
            print(f"All inputs processed: {e}")
            self.logger.info(f"All inputs processed: {e}")

        gc.collect()
        if len(jobs) == 0:
            print(f"No jobs prepared, all finished(?). Exiting.")
            self.logger.error(f"No jobs prepared, all finished(?). Exiting.")
            exit(0)
        print(f"Total jobs prepared: {len(jobs)}")
        self.logger.info(f"Total jobs prepared: {len(jobs)}")
        
        return jobs
    
    def make_next_request(
            self,
            all_inputs: pd.DataFrame, 
            prompts_file: Path = None,
        ) -> Generator[types.BatchJob, None, None]:
        
        next_entry = 0
        while next_entry < len(all_inputs):
            
            # 2. Save df and upload it
            if self.max_entries_per_batch == 1:
                last_entry = next_entry
                current_df = all_inputs.iloc[next_entry]
            else:
                last_entry = min(next_entry + self.max_entries_per_batch, len(all_inputs))
                current_df = all_inputs.iloc[next_entry:last_entry]

            # 3. Send prompts
            log_text = f"prompting for {self.entry_type_name} ({next_entry}:{last_entry}) out of {len(all_inputs)} at {datetime.datetime.now()}"
            self.logger.info(log_text)
            print(log_text)

            attempt = 0
            while attempt < self.max_attempts_per_prompt:
                try:
                    request_key, prep_content = self.prepare_for_request(current_df)
                    if prompts_file:
                        # save prompt for tuning
                        with open(prompts_file, 'a') as f:
                            f.write(json.dumps(
                                {
                                    "systemInstruction" : types.ModelContent(self.model_prompt).model_dump(exclude_none=True),
                                    "contents": prep_content.model_dump(exclude_none=True)
                                }
                            )+ '\n')
                    if self.only_count_tokens:
                        print(
                            "\tprompt tokens:", 
                            self.client.models.count_tokens(model=self.model_name, contents=prep_content).total_tokens
                        )
                        yield None
                        break # attempt while
                    else:
                        job = self.create_batch_request(request_key, prep_content)
                        if job:
                            yield job
                            break # attempt while
                   
                except Exception as e:
                    if isinstance(e, errors.APIError) and e.code == 429:
                        raise
                    print(f"\n    Error preparing job {request_key}, attempt {attempt + 1}/{self.max_attempts_per_prompt}: {e}")
                    self.logger.error(f"Error preparing job {request_key}, attempt {attempt + 1}/{self.max_attempts_per_prompt}: {e}")
                    attempt += 1
            # end attempt while
            next_entry += self.max_entries_per_batch                    
            

    def get_finished_jobs(self)-> tuple[bool, list[types.BatchJob]]:
        '''returns whether all existing jobs are still pending and list of finished jobs'''
        finished_jobs = []
        print(f"Checking for finished jobs at {datetime.datetime.now()}")
        all_pending = True
        for batch_job in self.client.batches.list():
            # print(batch_job.display_name, batch_job.state.name, "created at", batch_job.create_time)
            if batch_job.state.name != 'JOB_STATE_PENDING':
                all_pending = False
            if batch_job.state.name == 'JOB_STATE_FAILED':
                print(f"Batch job error in {batch_job.display_name}: {batch_job.error}")
                self.logger.error(f"Batch job error in {batch_job.display_name}: {batch_job.error}")
            elif batch_job.state.name in ('JOB_STATE_SUCCEEDED', 'JOB_STATE_CANCELLED'):
                finished_jobs.append(batch_job)
            
        # print(f"{len(finished_jobs)} jobs finished")
        # self.logger.warning(f"{len(finished_jobs)} jobs finished")
        
        return all_pending, finished_jobs

    def wait_and_process_jobs(self, output_file: Path, responses_file: Path | None = None):
        all_pending_count = 0
        all_pending_wait_sec = self.initial_wait_seconds
        while self.client.batches.list() and self.client.batches.list()[0]:
            print(f">={len(self.client.batches.list())} batch jobs pending")
            all_pending, finished_jobs = self.get_finished_jobs()
            if finished_jobs:
                all_pending_count = 0
                all_pending_wait_sec = self.initial_wait_seconds
                print(f"Processing {len(finished_jobs)} finished jobs at {datetime.datetime.now()}")
                self.logger.warning(f"Processing {len(finished_jobs)} finished jobs at {datetime.datetime.now()}")
                for job in finished_jobs:
                    # 5. Retrieve, process and save results
                    self.process_job_output(job, output_file, responses_file)
                self.prepare_batch_requests(len(finished_jobs)) # start a number of jobs equal to what just finished
            else:
                if all_pending:
                    # every other all_pending, double wait time as long as it's under an hour
                    if all_pending_count > 0 and all_pending_count % 2 == 0 and all_pending_wait_sec < 60 * 60:
                        all_pending_wait_sec *= 2
                    all_pending_count += 1
                else:
                    all_pending_count = 0
                    all_pending_wait_sec = self.initial_wait_seconds
                wait_seconds = all_pending_wait_sec if all_pending else self.followup_wait_seconds
                wait_message = f"No finished jobs found at {datetime.datetime.now()}. {"All pending. " if all_pending else ""}Waiting {wait_seconds/60} minutes..."
                print(wait_message)
                self.logger.warning(wait_message)
                time.sleep(wait_seconds)


    def process_job_output(
            self,
            batch_job: types.BatchJob,
            output_file: Path, 
            responses_file: Path | None  = None
        ) -> bool:
        success = False
        if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
            print(f"Job did not succeed, unexpected final state: {batch_job.state.name}")
            self.logger.error(f"Job did not succeed, unexpected final state: {batch_job.state.name}")
            success = False
        else:
            try:
                self.logger.debug(f"batch_job {batch_job}")
                batch_job = self.client.batches.get(name=batch_job.name)
                self.logger.debug(f"batch_job.dest {batch_job.dest}")
                self.logger.debug(f"inlined_responses {batch_job.dest.inlined_responses}")
                responses = batch_job.dest.inlined_responses
                success = self.process_job_output_content(batch_job.display_name, responses, output_file, responses_file)
            except Exception as e:
                self.logger.warning(f"Exception processing {batch_job.name} output: {e}")
                print(f"Exception processing {batch_job.name} output: {e}")
        self.client.batches.delete(name=batch_job.name)
        # exit(0)
        return success

    def process_job_output_content(
            self,
            display_name: str, 
            responses: list[types.InlinedResponse], 
            output_file: Path, 
            responses_file: Path | None  = None
        ) -> bool:
        successful = True

        # parse each inlined response
        self.logger.debug(f"number of responses {len(responses)}")
        for response in responses:
            self.logger.debug(f"response {response}")
            content_response = response.response
            self.logger.debug(f"response.response {content_response}")

            if not content_response and response.error:
                self.logger.warning(f"{display_name} errored: {response.error.message}")
                successful = False
                continue

            finish_reason = content_response.candidates[0].finish_reason
            if finish_reason != "STOP":
                successful = False
                print(f"Unexpected finishReason: {finish_reason} in {display_name}")
                self.logger.error(f"Unexpected finishReason ({content_response.model_version}): {finish_reason} in {display_name}")
                # logger.info(json.dumps(content_response, indent=2))
                if finish_reason == "MAX_TOKENS":
                    self.logger.warning(f"Total tokens used: {content_response.usage_metadata.total_token_count}")
                    self.create_batch_request(
                        display_name,
                        [
                            content_response.candidates[0].content,
                            types.UserContent(["Finish extracting the text"])#.model_dump(exclude_none=True)
                        ]
                    )
                    ## rebatch, continue
            else:
                if not self.save_job_output_content(display_name, content_response.parts[0].text, output_file, responses_file):
                    successful = False
        return successful
    
    @abstractmethod
    def save_job_output_content(self, display_name:str, response_text:str, output_file:Path, responses_file:Path|None = None) -> bool:
        """Must be implemented by subclasses"""
        pass

    @abstractmethod
    def prep_output_file(self, output_file:Path):
        """Must be implemented by subclasses"""
        pass

    def batch_prompt(
            self,
            input_file: Path,
            output_dir: Path,
            output_file_name: str,
            other_done_files: list[Path]|None = None,
            record_prompts_responses: bool = False
        ):

        # 1. Setup file paths
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / output_file_name
        prompts_file = None
        responses_file = None
        if record_prompts_responses:
            prompts_file = output_dir / "extracted_entries_prompts.jsonl"
            responses_file = output_dir / "extracted_entries_responses.jsonl"
        
        # Check if any batch jobs are in progress
        if ((not self.only_count_tokens) and 
            self.client.batches.list() and self.client.batches.list()[0]):
            print(f"Batch jobs already in progress.")
            self.logger.warning(f"Batch jobs already in progress.")
            
        else:
            print("No current batch job found. Preparing new batch job...")
            self.logger.info("No current batch job found. Preparing new batch job...")

            #1. Load the data
            if not input_file.exists():
                print(f"Error: {input_file} not found.")
                exit(1)

            # create output file, write header for CSV
            if not output_file.exists():
                self.prep_output_file(output_file)

            all_finished_files = other_done_files + [output_file] if other_done_files else [output_file]
            inputs = self.read_input_drop_completed(input_file, all_finished_files)

            # 2. Initiate batch requests
            print(f"Initiating Batch {self.entry_type_name} requests (n={self.max_batches_at_once})")
            self.logger.info(f"Initiating Batch {self.entry_type_name} requests (n={self.max_batches_at_once})")
            
            self.make_next_request_gen = self.make_next_request(inputs, prompts_file)
            jobs = self.prepare_batch_requests(self.max_batches_at_once)

            # 3. Wait for batch jobs to complete
            if not self.only_count_tokens and self.initial_wait_seconds > 0:
                print(f"Waiting {self.initial_wait_seconds/60} minutes for {len(jobs)} batch jobs to complete at {datetime.datetime.now()}...")
                self.logger.error(f"Waiting {self.initial_wait_seconds/60} minutes for {len(jobs)} batch jobs to complete at {datetime.datetime.now()}...")
                time.sleep(self.initial_wait_seconds)
            else:
                print(f"No initial wait time specified, leaving {len(jobs)} jobs pending.")
                self.logger.error(f"No initial wait time specified, leaving {len(jobs)} jobs pending.")
                exit(0)
        
        # 4. While any batch jobs are pending, check if any are done
        self.wait_and_process_jobs(output_file, responses_file)

        print(f"\n✓ Success! {self.entry_type_name} saved in: {output_file}")
        self.logger.error(f"Saved {self.entry_type_name} entries in: {output_file}")

