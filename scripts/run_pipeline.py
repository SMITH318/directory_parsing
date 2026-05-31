#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

def run_script(script_path: Path, args: list[str]) -> bool:
    print(f"\n{'='*80}")
    print(f"Running: {script_path.name} {' '.join(args)}")
    print(f"{'='*80}")
    
    result = subprocess.run([sys.executable, str(script_path)] + args)
    if result.returncode != 0:
        print(f"\n[ERROR] {script_path.name} failed with return code {result.returncode}")
        return False
    return True

def prompt_manual_step(step_name: str, expected_file: Path):
    print(f"\n{'-'*80}")
    print(f"MANUAL STEP REQUIRED: {step_name}")
    print(f"Expected output file not found: {expected_file}")
    print("Please complete this manual step and then restart the pipeline.")
    print(f"{'-'*80}")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Directory Parsing Pipeline")
    parser.add_argument("dataset", help="Name of the dataset")
    parser.add_argument("--pdf-dir", help="Directory containing the input PDFs", default=None)
    parser.add_argument("--preprocessed", help="Directory for preprocessed images", default=None)
    
    args = parser.parse_args()
    dataset = args.dataset
    
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / dataset
    
    # Setup directories
    pdf_dir_arg = args.pdf_dir if args.pdf_dir else str(project_root / "pdfs")
    preprocessed_arg = args.preprocessed if args.preprocessed else str(data_dir / "01_preprocessed")

    all_args = [dataset, "--pdf-dir", pdf_dir_arg, "--preprocessed", preprocessed_arg]
    preproc_args = [dataset, "--preprocessed", preprocessed_arg]

    steps = [
        # (script_name, arguments, expected_output_file, manual_step_name)
        ("01_preprocess.py", all_args, Path(preprocessed_arg) / "all_metadata.json", None),
        ("01.01_preprocessing_json_to_csv.py", preproc_args, Path(preprocessed_arg) / "all_metadata.csv", None),
        ("02_process_pdfs_gemini_batch_mass.py", preproc_args, data_dir / "02_ocr_output.jsonl", None),
        ("02_sanity_check.py", preproc_args, None, None), # sanity checks just print stuff usually, no specific output file to gate on, or we let them run.
        ("03_combine_sort_OCR.py", [dataset], data_dir / "03_ocr_output_combined_sorted.jsonl", None),
        (None, None, data_dir / "04_ocr_output_cleaned.jsonl", "04_clean_OCR_manual.ipynb"),
        ("05_classify_lines_gemini.py", [dataset], data_dir / "05_entries_segmented.csv", None),
        ("05_sanity_check.py", [dataset], None, None),
        (None, None, data_dir / "07_entries_segmented_man_cleaned.csv", "Prepare for Review & Review Changes (Manual)"),
        ("08_split_parsed_entries.py", [dataset], data_dir / "08_doc_entries.csv", None), # should also output 08_city_entries.csv
        ("09_parse_doc_entries_gemini.py", [dataset], data_dir / "09_doc_entries_parsed.csv", None),
        ("09_parse_city_entries_gemini.py", [dataset], data_dir / "09_city_entries_parsed.csv", None),
        ("09_sanity_check.py", [dataset], None, None),
        ("10_sort_entries.py", [dataset], data_dir / "10_doc_entries_sorted.csv", None),
        ("11_reformat_to_chrisinger.py", [dataset], data_dir / "amd_1918_reformatted.csv", None),
    ]
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for script_name, script_args, expected_output, manual_step_name in steps:
        if manual_step_name:
            if not expected_output.exists():
                prompt_manual_step(manual_step_name, expected_output)
            else:
                print(f"Skipping {manual_step_name}, output {expected_output.name} already exists.")
            continue
            
        script_path = script_dir / script_name
        if not script_path.exists():
            print(f"[ERROR] Script not found: {script_path}. Stopping.")
            sys.exit(1)
            
        # Check if we should skip because output exists
        if expected_output and expected_output.exists():
            print(f"Skipping {script_name}, output {expected_output.name} already exists.")
            continue
            
        # Run the script
        success = run_script(script_path, script_args)
        if not success:
            sys.exit(1)

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
