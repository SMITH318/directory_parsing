"""
Unsuccessful attempt to tune Gemini-2.5-flash on the extracted entries parsing task (step 9).
Must use Vertex for training as of sping 2026.
The tuning process didn't provide a useful result. 
The script includes the setup for Gemini API interaction, dataset preparation, and 
the tuning job configuration. It also includes logging for monitoring the tuning process.
The script is retained for reference and potential future use.
"""
import json
from pathlib import Path
from google import genai
from google.genai import types
import time
import datetime

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
  filename='04.00_tune_gemini_to_parse_entries.log', 
  filemode='a', 
  encoding='utf-8', 
  level=logging.INFO) ## <=================== Change logging level here

# --- Gemini API Configuration --- 
# API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY')
model_name ='gemini-2.5-flash'#'gemini-flash-latest'

# if API_KEY == 'YOUR_API_KEY' or not API_KEY:
#     print("ERROR: Gemini API key is not set.")
#     logger.error("ERROR: Gemini API key is not set.")
#     exit(1)

# Initialize Gemini
print("Initializing Gemini for OCR...")
#print(f"Key: {API_KEY}")
logger.info(f"Initializing Gemini for OCR... with {model_name}")
PROJECT_ID = 'digitizing-directories-mrsmith'
REGION = 'us-central1'
client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)#api_key=API_KEY)
# client = genai.Client(api_key=API_KEY)

# setup file paths
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

training_dataset = types.TuningDataset( 
    gcs_uri= "gs://amd_training/extracted_entries_training.jsonl", # required source for Vertex
    # examples= []  # supported in Gemini??? List[TuningExample] # tuning not supported in gemini ATM
)

if training_dataset.examples is not None:
    print("loading dataset")
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / f"04_extracted_entries_gemini_2pgs_cache_{model_name}"
    prompts_file = data_dir / "extracted_entries_prompts.jsonl"
    responses_file = data_dir / "extracted_entries_responses_cleaned.jsonl"

    with open(prompts_file, 'r', encoding='utf-8') as prompts:
        with open(responses_file, 'r', encoding='utf-8') as responses:
            response_list = responses.read().splitlines()
            for i, prompt in enumerate(prompts.read().splitlines()):
                parsed_prompt = json.loads(prompt)
                parts = parsed_prompt["contents"]["parts"]
                parts_text = "\n".join([p["text"] for p in parts])
                training_dataset.examples.append(types.TuningExample(
                    text_input= parts_text,
                    output=response_list[i]
                ))
        

print(f"Start tuning")
sft_tuning_job = client.tunings.tune(
    base_model = model_name,
    training_dataset = training_dataset, 
    config=types.CreateTuningJobConfig(
        tuned_model_display_name="entry_extractor_AMD_1918"
    )
)
print(f"Tuning started at {datetime.datetime.now()}")
tuning_job = client.tunings.get(name=sft_tuning_job.name)

while tuning_job.state.name in ["JOB_STATE_PENDING", "JOB_STATE_RUNNING"]:
    print(".", end="", flush=True)
    tuning_job = client.tunings.get(name=tuning_job.name)
    time.sleep(600) # wait 10 minutes between checks to avoid hitting rate limits, tuning can take several hours
print(f"\nTuning job done with status {tuning_job.state.name}")
if tuning_job.state.name != 'JOB_STATE_SUCCEEDED':
    print(tuning_job)

tuned_model = tuning_job.tuned_model.endpoint
experiment_name = tuning_job.experiment

print("Tuned model experiment", experiment_name)
print("Tuned model endpoint resource name:", tuned_model)

client.close()