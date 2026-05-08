from _AStepConfiguration import *

# ***************************** constants *****************************
SKIP_TEXT = "******* KEEPS FAILING! SKIPPING FOR NOW *******"

class OCRStep(AStepConfiguration):
    # abstract
    def drop_some_finished(self, finished_df: pd.DataFrame) -> pd.DataFrame:
        finished_df = finished_df[finished_df['text'] != SKIP_TEXT]
        return finished_df

    # abstract
    def load_input(self, file_in: Path) -> pd.DataFrame:
        return pd.read_csv(file_in, encoding='utf-8')

    # abstract
    def load_finished(self, file_done: Path) -> pd.DataFrame:
        return pd.read_json(file_done, encoding='utf-8', lines=True)
    
    # abstract
    def df_columns_to_check_finished(self) -> list[tuple[str,str]]:
        return [('pub_id', 'pub'), ('page_num', 'page'), ('column', 'col')] 

    # abstract
    def prepare_for_request(self, request_df: pd.DataFrame) -> tuple[str, types.UserContent]: # request key, content
        """Upload snippet image and create request to extract text and bounding boxes it using Gemini Vision API."""
        if not Path(request_df['path']).exists():
            # self.logger.error(f"Image file missing: {request_df['path']}")
            raise FileNotFoundError

        # inline the file
        with open(request_df['path'], 'rb') as f:
            image_bytes = f.read()
        content = types.UserContent([
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
                #media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH, # add? or MEDIA_RESOLUTION_ULTRA_HIGH??
            )
        ])
        return f"gemini_ocr_{os.path.basename(request_df['path'])}", content
   
    def split_key(self, logger: logging.Logger, key:str) -> dict:
        # parse example key: gemini_ocr_Alabama_p01_r00_c000.jpg
        key_parts = key.split('.')[0].split('_')
        if len(key_parts) < 6:
            logger.error(f"Invalid key format: {key}")
            return {}
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

    def gemini_extract_snippet(self, logger: logging.Logger, key: str, json_text: str) -> tuple[dict[str, int, int, int], list]:
        """Extract text blocks from Gemini Vision API response."""

        info = self.split_key(logger, key)
        # print("************ gemini_extract_snippet")
        # print(json_text)
        try:
            text_blocks = json.loads(json_text)["lines"]
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
    
    # abstract
    def save_job_output_content(self, logger: logging.Logger, display_name:str, response_text:str, output_file:Path, responses_file:Path|None = None) -> bool:
        successful = True
        info, gemini_output = self.gemini_extract_snippet(logger, display_name, response_text)
        # find offsets for this snippet
        # offset_row = offsets_file_df[offsets_file_df['key'] == display_name].iloc[0]
        
        lines_received_txt = f"received {len(gemini_output)} {self.entry_type_name}s at {datetime.datetime.now()}"
        logger.info(lines_received_txt)
        print("\t", lines_received_txt)

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
                    f_out.write(json.dumps(entry, ensure_ascii=False) + "\n") # need cls=NumpyEncoder??
                    
                except Exception as e:
                    logger.error(f"Error processing block: {block} in snippet {info} ({display_name}), error: {e}")
                    print(f"\t**Error processing block: {block} in snippet {info} ({display_name}), error: {e}")
                    successful = False
                    continue
        return successful
    
    # abstract
    def prep_output_file(self, output_file:Path):
        with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
            f_out.write("")

