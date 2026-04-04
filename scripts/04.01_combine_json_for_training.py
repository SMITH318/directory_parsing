from pathlib import Path
import json

MAX_ENTRIES_SENT = 10

# setup file paths
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

output_dir = project_root / "data" / f"04_extracted_entries_gemini_2pgs_seeded_986234"
prompt_json = output_dir / "extracted_entries_prompts.jsonl" 
reponse_json = output_dir / "extracted_entries_responses_cleaned.jsonl"
combined_json = output_dir / "extracted_entries_training.jsonl"

with open(prompt_json, 'r', encoding='utf-8') as f:
    prompt_lines = f.readlines()
with open(reponse_json, 'r', encoding='utf-8') as f:
    response_lines = f.readlines()

if len(prompt_lines) != len(response_lines):
    print(f"Error: number of prompts ({len(prompt_lines)}) doesn't match number of responses ({len(response_lines)})!")

combined = []
for i in range(len(prompt_lines)):
    prompt = json.loads(prompt_lines[i])
    # print(response_lines[i])
    response = json.loads(response_lines[i])
    prompt["contents"] = [
        prompt["contents"],
        {
            "role": "model",
            "parts": [
                { "text": json.dumps(response) }
            ]
        }
    ]
    combined.append(prompt)

with open(combined_json, 'w') as combined_f:
    for c in combined:
        combined_f.write(json.dumps(c)+'\n')