from typing import List
from streamlit.runtime.uploaded_file_manager import UploadedFile
from fedot_llm.data.data import Split, Dataset
import pandas as pd
class StreamlitDatasetLoader:
    @staticmethod
    def load(files: List[UploadedFile]):
        splits: List[Split] = []
        for file in files:
            if file.name.split('.')[-1] == 'csv':
                dataframe = pd.read_csv(file)
                split_name = file.name
                split = Split(name=split_name, data=dataframe)
                splits.append(split)
        if len(splits) > 0:
            return Dataset(splits=splits)
        else: 
            raise ValueError('No csv files found!')