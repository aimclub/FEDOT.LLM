import json
import os
from typing import List

import arff
import pandas as pd

from fedot_llm.data.data import Dataset, Split


class PathDatasetLoader:
    def load(self, path: str, with_metadata: bool = False):
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
            splits: List[Split] = []
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

            return Dataset(
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

            return Dataset(splits=splits)
