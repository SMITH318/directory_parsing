#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
  handlers=[
      logging.FileHandler('run_pipeline.log', mode='w', encoding='utf-8'),
      logging.StreamHandler(sys.stderr)
  ],
  level=logging.INFO) ## <=================== Change logging level here

def run_script(script_path: Path, args: list[str]) -> bool:
    logger.info(f"\n{'='*80}")
    logger.info(f"Running: {script_path.name} {' '.join(args)}")
    logger.info(f"{'-'*80}")
    
    result = subprocess.run([sys.executable, str(script_path)] + args)
    logger.info(f"{'-'*80}")
    if result.returncode != 0:
        logger.error(f"{script_path.name} failed with return code {result.returncode}")
        return False
    logger.info("Success!")
    return True

def prompt_manual_step(step_name: str, expected_file: Path):
    logger.warning(f"\n{'='*80}")
    logger.warning(f"MANUAL STEP REQUIRED: {step_name}")
    logger.warning(f"Expected output file not found: {expected_file}")
    logger.warning("Please complete this manual step and then restart the pipeline.")
    logger.warning(f"{'='*80}")
    sys.exit(0)

def main():
    # TODO: have file input/outputs be passed to scripts

    parser = argparse.ArgumentParser(description="Directory Parsing Pipeline")
    parser.add_argument("dataset", help="Name of the dataset")
    parser.add_argument("--pdf-dir", help="Directory containing the input PDFs", default=None)
    parser.add_argument("--preprocessed", help="Directory for preprocessed images", default=None)
    parser.add_argument("--config", help="Path to JSON schema config with embedded prompts", default=None)
    
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
    config_args = [dataset, "--config", args.config] if args.config else [dataset]
    
    steps = [
        # (script_name, arguments, expected_output_file, manual_step_name)
        ("01_preprocess.py", all_args, Path(preprocessed_arg) / "all_metadata.json", None),
        ("01.01_preprocessing_json_to_csv.py", preproc_args, Path(preprocessed_arg) / "all_metadata.csv", None),
        ("02_process_pdfs_gemini_batch_mass.py", preproc_args, data_dir / "02_ocr_output.jsonl", None),
        ("02_sanity_check.py", preproc_args, None, None), 
        ("03_combine_sort_OCR.py", [dataset], data_dir / "03_ocr_output_combined_sorted.jsonl", None),
        (None, None, data_dir / "04_ocr_output_cleaned.jsonl", "04_clean_OCR_manual.ipynb"),
        ("05_classify_lines_gemini.py", config_args, data_dir / "05_entries_segmented.csv", None),
        ("05_sanity_check.py", [dataset], None, None),
        (None, None, data_dir / "07_entries_segmented_man_cleaned.csv", "Review Segmented Entries Manual"),
        ("08_split_parsed_entries.py", config_args, data_dir / "08_split_complete.txt", None), 
        ("09_parse_entries_generic.py", config_args, data_dir / "09_parse_complete.txt", None),
        ("09_sanity_check.py", config_args, None, None),
        ("10_sort_entries.py", config_args, data_dir / "10_entries_sorted.csv", None),
    ]
    if not args.config:
        steps.append(
            ("11_reformat_to_chrisinger.py", [dataset], data_dir / "amd_1918_reformatted.csv", None)
        )
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for script_name, script_args, expected_output, manual_step_name in steps:
        if manual_step_name:
            if not expected_output.exists():
                prompt_manual_step(manual_step_name, expected_output)
            else:
                logger.info(f"\n{'='*80}")
                logger.info(f"Skipping: {manual_step_name}, output {expected_output.name} already exists.")
            continue

        script_path = script_dir / script_name
        if not script_path.exists():
            logger.error("Script not found: {script_path}. Stopping.")
            sys.exit(1)

        # Always run the script so it can report success via its exit code
        success = run_script(script_path, script_args)
        if not success:
            logger.error(f"Pipeline stopping due to failure in {script_name}")
            sys.exit(1)

    logger.error("\nPipeline completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
