import io
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


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_pct = (missing / len(df)).round(3) * 100
    missing_df = pd.DataFrame({"Missing": missing, "Percent": missing_pct})
    return missing_df


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

    def get_train_split(self):
        # heuristics to find train split
        for split in self.splits:
            if "train" in split.name.lower():
                train_split = split
                break
        else:
            # Find splits with max column count
            max_cols = max(split.data.shape[1] for split in self.splits)
            max_col_splits = [
                split for split in self.splits if split.data.shape[1] == max_cols
            ]

            # If multiple splits have the same column count, take the one with more rows
            if len(max_col_splits) > 1:
                train_split = max(max_col_splits, key=lambda split: split.data.shape[0])
            else:
                train_split = max_col_splits[0]
        return train_split

    def dataset_eda(self):
        """Generate exploratory data analysis summary only for the split with maximum columns."""
        if not self.splits:
            return "No data splits available"
        # heuristics to find train split
        train_split = self.get_train_split()
        df = train_split.data
        eda = ""
        if df.shape[1] <= 10:
            eda += "\n===== 1. BASIC INFO =====\n"

            buf = io.StringIO()
            df.info(buf=buf)

            info_str = buf.getvalue()
            eda += info_str

            eda += "\n===== 2. MISSING VALUES =====\n"
            eda += missing_values(df).to_markdown()
        return eda

    def dataset_preview(self, sample_size: int = 11):
        preview = ""
        train_split = self.get_train_split()
        if train_split.data.shape[1] > 10:
            preview += f"File: {train_split.name}\n"
            preview += train_split.data.sample(sample_size).to_markdown()
            preview += "\n\n"
            for split in self.splits:
                preview += f"File: {split.name}\n"
                preview += f"Columns: {split.data.columns.tolist()}\n"
                preview += "\n\n"
        else:
            for split in self.splits:
                preview += f"File: {split.name}\n"
                preview += split.data.sample(sample_size).to_markdown()
            preview += "\n\n"
        return preview

    def __str__(self):
        return self.dataset_preview()
