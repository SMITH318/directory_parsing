# from collections.abc import Generator
from abc import ABC, abstractmethod
# from pydantic import ValidationError
import datetime
import json
# import gc
# import os
from pathlib import Path
# from google import genai
from google.genai import types
# from google.genai import errors
# import time
import pandas as pd
from pydantic import BaseModel
# import csv
import logging
from _clean_gemini import *

class AStepConfiguration(ABC):
    def __init__(
            self, 
            model_name: str, 
            model_prompt: str, 
            entry_type_name: str, 
            entry_type: type[BaseModel], 
            entries_type: type[BaseModel], 
        ):
        self.cache = None
        self.model_name = model_name
        self.model_prompt = model_prompt
        self.entry_type_name = entry_type_name
        self.entry_type = entry_type
        self.entries_type = entries_type

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

    @abstractmethod
    def prepare_for_request(self, request_df: pd.DataFrame) -> tuple[str, types.UserContent]:
        """Must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def save_job_output_content(self, logger: logging.Logger, display_name:str, response_text:str, output_file:Path, responses_file:Path|None = None) -> bool:
        """Must be implemented by subclasses"""
        pass

    @abstractmethod
    def prep_output_file(self, output_file:Path):
        """Must be implemented by subclasses"""
        pass
