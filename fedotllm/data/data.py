from pathlib import Path
from typing import List

import pandas as pd
from scipy.io.arff import loadarff


class Split:
    """
    Split within dataset object
    """

    def __init__(self, name: str, data: pd.DataFrame):
        self.name = name
        self.data = data


class Dataset:
    def __init__(self, splits: List[Split], path: Path):
        self.splits = splits
        self.path = path

    @classmethod
    def from_path(cls, path: Path):
        """
        Load Dataset a folder with dataset objects

        Args:
            path: Path to folder with Dataset data
        """

        # Loading all splits in folder
        splits = []
        if path.is_dir():
            files = [x for x in path.glob("**/*") if x.is_file()]
        else:
            files = [path]
        for file in files:
            file_path = file.absolute()
            split_name = file.name
            if file.name.split(".")[-1] == "csv":
                raw_data = pd.read_csv(file_path)
                split = Split(data=raw_data, name=split_name)
                splits.append(split)
            if file.name.split(".")[-1] == "arff":
                raw_data = loadarff(file_path)
                split = Split(data=pd.DataFrame(raw_data[0]), name=split_name)
                splits.append(split)
            if file.name.split(".")[-1] in [
                "xls",
                "xlsx",
                "xlsm",
                "xlsb",
                "odf",
                "ods",
                "odt",
            ]:
                raw_data = pd.read_excel(file_path)
                split = Split(data=raw_data, name=split_name)
                splits.append(split)

        return Dataset(splits=splits, path=path)
