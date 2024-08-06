import json
import os
import random
from functools import lru_cache
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, InitVar, field

import arff
import pandas as pd

@dataclass
class Split:
    """
    Split within dataset object
    """
    
    name: str
    """ The name of the split """
    data: pd.DataFrame = field(repr=False)
    """ Data that is stored in the split """
    path: str
    """ Path to the file with split data """
    description: Optional[str] = None
    """ Description of the split data """
    init_columns_meta: InitVar[Optional[Dict]] = None
    """ Init metadata for columns in the split """
    columns_meta: Dict[str, Dict] = field(default_factory=dict, init=False)
    """ Metadata for columns in the split """
  
    def __post_init__(self, init_columns_meta):
        self.columns_meta = {col: {} for col in self.data.columns}
        if init_columns_meta is not None:
            for col, metainfo in init_columns_meta.items():
                if col not in self.columns_meta:
                    raise RuntimeError(
                        "Failed to find the column {col} defined in the metadata.json file"
                    )
                if "hint" in metainfo:
                    self.columns_meta[col]["hint"] = metainfo["hint"]
                if "description" in metainfo:
                    self.columns_meta[col]["description"] = metainfo["description"]
                    
    def __getitem__(self, key: str) -> pd.Series:
        return self.data[key]

    @property
    def detailed_description(self):
        if self.name is not None:
            description = f"The {self.name} split"
        else:
            description = "The split"

        if self.path is not None:
            fname = os.path.split(self.path)[-1]

            description += f' stored in file "{fname}"'

        description += f" contains following columns: {list(self.data.columns)}."

        if self.name is not None:
            description += f" It is described as {self.description}"

        return description

    @property
    def metadata_description(self) -> str:
        return (
            f"name: {self.name} \npath: {self.path} \ndescription: {self.description}"
        )

    @property
    def text_columns(self):
        return list(self.data.select_dtypes(include=["object"]).columns)

    @property
    def numeric_columns(self):
        return list(self.data.select_dtypes(include=["number"]).columns)

    @property
    def unique_counts(self):
        return self.data.apply(lambda col: col.nunique())

    @property
    def unique_ratios(self):
        return self.data.apply(lambda col: round(col.nunique() / len(col.dropna()), 2))
    
    @lru_cache(maxsize=128)
    def get_column_unique_ratio(self, column_name: str, ndigits: int = 2) -> float:
        return round(self[column_name].nunique() / len(self[column_name].dropna()), ndigits)

    @lru_cache(maxsize=128)
    def get_unique_values(self, column_name: str, max_number: int = -1) -> pd.Series:
        """
        Get unique values from the data attribute up to a specified maximum number.

        Args:
            column_name (str): The name of the column in split.
            max_number (int): Maximum number of unique values to return. Defaults to -1, which means return all unique values.

        Returns:
            pd.Series: A pandas Series containing unique values from the data attribute.
        """
        column_uniq_vals = self.data[column_name].unique().tolist()
        if max_number != -1:
            column_uniq_vals = (
                column_uniq_vals
                if len(column_uniq_vals) < max_number
                else random.sample(column_uniq_vals, k=max_number)
            )
        return pd.Series(column_uniq_vals, name=column_name)

    @property
    def column_types(self):
        return self.data.apply(
            lambda col: "string" if col.name in self.text_columns else "numeric"
        )

    @lru_cache
    def get_head_by_column(self, column_name: str, count: int = 10) -> pd.Series:
        return self[column_name].head(count)

    @property
    def column_descriptions(self):
        return dict(
            (key, value["description"]) for key, value in self.columns_meta.items() if "description" in value
        )

    def __str__(self):
        return self.metadata_description

@dataclass
class Dataset:
    """
    Dataset object that represents an ML task and may contain multiple splits
    """
    
    splits: List[Split]
    """ List of splits in the dataset """
    name: Optional[str] = None
    """ Name of the dataset """
    description: Optional[str] = None
    """ Description of the dataset """
    goal: Optional[str] = None
    """ Goal of the dataset """
    __test_split: Optional[Split] = field(init=False, default=None)
    """ Test split of the dataset """
    __train_split: Optional[Split] = field(init=False, default=None)
    """ Train split of the dataset """
    __target_name: Optional[str] = field(init=False, default=None)
    """ Target column name of the dataset """
    __task_type: Optional[str] = field(init=False, default=None)
    """ Task type of the dataset """
    
    @classmethod
    def load_from_path(cls, path: str, with_metadata: bool = False):
        """
        Load Dataset a folder with dataset objects

        Args:
            path: Path to folder with Dataset data
            with_metadata: Whether Dataset should be loading metadata.json file contained in folder. Defaults to false.

        """

        if with_metadata:
            with open(os.sep.join([path, "metadata.json"]), "r") as json_file:
                dataset_metadata = json.load(json_file)

            # load each split file
            splits = []
            for split in dataset_metadata["splits"]:
                split_name = split["name"]
                split_path = os.sep.join([path, split["path"]])
                split_description = split.get("description", "")
                split_columns = split.get("columns", None)
                if split_path.split(".")[-1] == "csv":
                    data = pd.read_csv(split_path)
                    split = Split(
                        data=data,
                        name=split_name,
                        path=split_path,
                        description=split_description,
                        init_columns_meta=split_columns,
                    )
                    splits.append(split)
                elif split_path.split(".")[-1] == "arff":
                    data = pd.DataFrame(list(arff.load(split_path)))
                    split = Split(
                        data=data,
                        name=split_name,
                        path=split_path,
                        description=split_description,
                        init_columns_meta=split_columns,
                    )
                    splits.append(split)
                else:
                    print(f"split {split_path}: unsupported format")

            return cls(
                name=dataset_metadata["name"],
                description=dataset_metadata["description"],
                goal=dataset_metadata["goal"],
                splits=splits,
            )

        else:
            # Loading all splits in folder
            splits = []
            for fpath in os.listdir(path):
                file_path = os.path.join(path, fpath)
                split_name = os.path.split(file_path)[-1].split(".")[0]
                if file_path.split(".")[-1] == "csv":
                    data = pd.read_csv(file_path)
                    split = Split(data=data, path=file_path, name=split_name)
                    splits.append(split)
                if file_path.split(".")[-1] == "arff":
                    data = arff.load(file_path)
                    split = Split(
                        data=pd.DataFrame(list(data)), path=file_path, name=split_name
                    )
                    splits.append(split)

            return cls(splits=splits)

    def is_train(self) -> bool:
        return self.__train_split is not None

    def is_test(self) -> bool:
        return self.__test_split is not None

    @property
    def train_split(self) -> Split:
        if self.__train_split is None:
            raise ValueError("Train split is not set")
        return self.__train_split

    @train_split.setter
    def train_split(self, train_name: str) -> None:
        train_list = list(filter(lambda split: split.name == train_name, self.splits))
        if not train_list:
            raise ValueError(f"No split found with name '{train_name}'")
        self.__train_split = train_list[0]

    @property
    def test_split(self) -> Split:
        if self.__test_split is None:
            raise ValueError("Test split is not set")
        return self.__test_split

    @test_split.setter
    def test_split(self, test_name: str) -> None:
        test_list = list(filter(lambda split: split.name == test_name, self.splits))
        if not test_list:
            raise ValueError(f"No split found with name '{test_name}'")
        self.__test_split = test_list[0]

    @property
    def target_name(self) -> str:
        if self.__target_name is None:
            raise ValueError("Target name is not set")
        return self.__target_name

    @target_name.setter
    def target_name(self, target: str) -> None:
        self.__target_name = target

    @property
    def task_type(self) -> str:
        if self.__task_type is None:
            raise ValueError("Task type is not set")
        return self.__task_type

    @task_type.setter
    def task_type(self, type: str) -> None:
        if type not in ["regression", "classification"]:
            raise ValueError(
                "Task type must be either 'regression' or 'classification'"
            )
        self.__task_type = type

    @property
    def detailed_description(self) -> str:
        split_description_lines = [split.detailed_description for split in self.splits]

        first_line = [
            "Assume we have a dataset",
            "{name}".format(name=(f" called {self.name}\n" if self.name else "")),
            "{description}".format(
                description=(
                    f"It could be described as following: {self.description}\n"
                    if self.description
                    else ""
                )
            ),
            "{goal}".format(goal=(f"The goal is: {self.goal}\n" if self.goal else "")),
            "The dataset contains the following splits:\n",
        ]
        first_line = "".join(first_line)

        introduction_lines = [
            first_line,
        ] + split_description_lines

        if self.is_train():
            column_descriptions = self.train_split.column_descriptions
            if column_descriptions:
                introduction_lines = [
                    "Below is the type (numeric or string), unique value count and ratio for each column, and few examples of values:",
                    "\n".join(
                        [f"{key}: {value}" for key, value in column_descriptions.items()]
                    ),
                ]

        return "\n".join(introduction_lines)

    @property
    def metadata_description(self) -> str:
        splits_metadatas = [split.metadata_description for split in self.splits]
        descriptions = [
            "{name}".format(
                name=(f"name: {self.name}\n" if self.name is not None else "")
            ),
            "{description}".format(
                description=(
                    f"description: {self.description}\n"
                    if self.description is not None
                    else ""
                )
            ),
            "{goal}".format(
                goal=(f"goal: {self.goal}\n" if self.goal is not None else "")
            ),
            "{train_split}".format(
                train_split=(
                    f"train split: {self.train_split.name}\n" if self.is_train() else ""
                )
            ),
            "{test_split}".format(
                test_split=(
                    f"test split: {self.test_split.name}\n" if self.is_test() else ""
                )
            ),
            "splits:\n\n",
        ]
        description = "".join(descriptions) + "\n".join(
            [f"{split}\n" for split in splits_metadatas]
        )
        return description

    def __str__(self):
        return self.metadata_description
