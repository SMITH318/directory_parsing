from _AStepConfiguration import *


def gen_extract_entries_paths(entry_type_name: str, data_set: str) -> tuple[Path, Path, str]:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir if (script_dir / "data").exists() else script_dir.parent
    input_file = project_root / "data" / "03_processed_batch" / f"{entry_type_name}_entries_{data_set}.csv"
    output_dir = project_root / "data" / f"04_extracted_entries_gemini_{data_set}"
    output_file_name =  f"amd_1918_{entry_type_name}_entries.csv" 
    return input_file, output_dir, output_file_name

class ExtractEntriesStep(AStepConfiguration):
    # abstract
    def drop_some_finished(self, finished_df: pd.DataFrame) -> pd.DataFrame:
        return finished_df

    # abstract
    def load_input(self, file_in: Path) -> pd.DataFrame:
        return pd.read_csv(file_in, encoding='utf-8')

    # abstract
    def load_finished(self, file_done: Path) -> pd.DataFrame:
        return pd.read_csv(file_done, encoding='utf-8')
    
    # abstract
    def df_columns_to_check_finished(self) -> list[str]:
        # return [('publication', 'publication'), ('page_number', 'page_number'), ('column', 'column')] 
        return [('entry_id', 'entry_id')]

    # abstract
    def prepare_for_request(self, request_df: pd.DataFrame) -> tuple[str, types.UserContent]: # request key, content
        file = request_df.to_csv(index=False, encoding="utf-8") # produces string if not given file
        content = types.UserContent([file])#[types.Part.from_text(text=file)]
            
        # upload blocks file
        # file = client.files.upload(
        #     file=temp_file,
        #     config=types.UploadFileConfig(mime_type='text/csv')#(mime_type='text/csv; charset=UTF-8')
        # )
        return f"{self.entry_type_name}_{request_df.index[0]}_{request_df.index[-1]}", content
   
    # abstract
    def save_job_output_content(self, logger: logging.Logger, display_name:str, response_text:str, output_file:Path, responses_file:Path|None = None) -> bool:
        entries = json.loads(response_text)
        if responses_file:
            with open(responses_file, 'a') as f:
                f.write(json.dumps(entries) + '\n')
        # entries = json_entries[f"{entry_type_name}_entries"]

        logger.info(f"received {len(entries)} {self.entry_type_name} entries at {datetime.datetime.now()}")
        print(f"\treceived {len(entries)} {self.entry_type_name} entries at {datetime.datetime.now()}")

        # 4. Save to CSV
        with open(output_file, 'a', encoding='utf-8', newline='') as f_out:
            entry_writer = csv.DictWriter(f_out, self.entry_type.model_fields.keys(), restval="")
            for e in entries:
                # print(e)
                entry_writer.writerow(e)
        return True
    
    # abstract
    def prep_output_file(self, output_file:Path):
        with open(output_file, 'w', encoding='utf-8', newline='') as csv_out:
            doc_writer = csv.DictWriter(csv_out, self.entry_type.model_fields.keys(), restval="")
            doc_writer.writeheader()

