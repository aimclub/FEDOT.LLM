from pathlib import Path
from typing import List

import pandas as pd
from scipy.io.arff import loadarff

from fedotllm.constants import (
    ARFF_SUFFIXES,
    CSV_SUFFIXES,
    DATASET_EXTENSIONS,
    EXCEL_SUFFIXES,
    PARQUET_SUFFIXES,
)


def load_pd(data):
    if isinstance(data, (Path, str)):
        path = data
        if isinstance(path, str):
            path = Path(path)
        format = None
        if path.suffix in EXCEL_SUFFIXES:
            format = "excel"
        elif path.suffix in PARQUET_SUFFIXES:
            format = "parquet"
        elif path.suffix in CSV_SUFFIXES:
            format = "csv"
        elif path.suffix in ARFF_SUFFIXES:
            format = "arff"
        else:
            if format is None:
                raise Exception(f"file format for {path.suffix} not supported!")
            else:
                raise Exception("file format " + format + " not supported!")

        match format:
            case "excel":
                return pd.read_excel(path, engine="calamine")
            case "parquet":
                try:
                    return pd.read_parquet(path, engine="fastparquet")
                except Exception:
                    return pd.read_parquet(path, engine="pyarrow")
            case "arff":
                return pd.DataFrame(loadarff(path)[0])
            case "csv":
                return pd.read_csv(path)
    else:
        return pd.DataFrame(data)


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
            if file_path.suffix in DATASET_EXTENSIONS:
                file_dataframe = load_pd(file_path)
                split = Split(data=file_dataframe, name=file.name)
                splits.append(split)

        return Dataset(splits=splits, path=path)

    def dataset_preview(self, sample_size: int = 11):
        preview = ""
        for split in self.splits:
            preview += f"File: {split.name}\n"
            preview += split.data.sample(sample_size).to_markdown()
            preview += "\n\n"
        return preview

    def __str__(self):
        return self.dataset_preview()
