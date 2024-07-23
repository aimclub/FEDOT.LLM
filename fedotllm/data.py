import json
import os
import arff
import pandas as pd

from typing import Any, Dict



class Split:
    """
    Split within dataset object
    """

    def __init__(self, name: str, data: pd.DataFrame, path: str, description: str, columns: dict | None) -> None:
        """
        Initialize an instanse of a Split.
        """
        self.name = name
        self.data = data
        self.path = path
        self.description = description
        
        # init columns metadata info
        self.columns_meta = {col: {} for col in data.columns}
        if columns is not None:
            for col, metainfo in columns.items():
                if col not in self.columns_meta:
                    raise RuntimeError("Failed to find the column {col} defined in the metadata.json file")
                if 'hint' in metainfo:
                    self.columns_meta[col]['hint'] = metainfo['hint']
                if 'description' in metainfo:
                    self.columns_meta[col]['description'] = metainfo['description']
        

    def get_description(self):
        return f"The {self.name} split contains following columns: {self.data.columns}. It is described as {self.description}"

    def get_text_columns(self):
        return list(self.data.select_dtypes(include=['object']).columns)

    def get_numeric_columns(self):
        return list(self.data.select_dtypes(include=['number']).columns)

    def get_unique_counts(self):
        return self.data.apply(lambda col: col.nunique())

    def get_unique_ratios(self):
        return self.data.apply(lambda col: round(col.nunique() / len(col.dropna()), 2))

    def get_column_types(self):
        return self.data.apply(lambda col: "string" if col.name in self.get_text_columns() else "numeric")

    def get_head_by_column(self, column_name, count=10):
        return list(self.data[column_name].head(count))

    def get_column_descriptions(self):
        return dict((key, value['description']) for key, value in self.columns_meta.items())
    
    def get_column_hint(self, column_name: str) -> None | str:
        """ 
        Get the hint associated with a specific column.

        Args:
            column_name (str): The name of the column.

        Returns:
            str or None: The hint associated with the column, or None if not found.
        """
        return self.columns_meta.get(column_name, None).get('hint', None)

    def set_column_descriptions(self, column_description: Dict[str, str]) -> None:
        """
        Set descriptions for columns in the metadata.

        Args:
            column_description (Dict[str, str]): A dictionary where keys are column names and values are descriptions.

        Returns:
            None
        """
        for key, value in column_description.items():
            if key in self.columns_meta:
                self.columns_meta[key]['description'] = value



class Dataset:
    """
    Dataset object that represents an ML task and may contain multiple splits
    """

    def __init__(self, name, description, goal, splits) -> None:
        """
        Initialize an instance of a Dataset.
        """
        self.name = name
        self.description = description
        self.goal = goal
        self.splits = splits
        

    @classmethod
    def load_from_path(cls, path):
        """
        Load Dataset a folder with dataset objects

        WIP: add support for incomplete descriptions?
        """
        with open(os.sep.join([path, 'metadata.json']), 'r') as json_file:
            dataset_metadata = json.load(json_file)

        # load each split file
        splits = []
        for split in dataset_metadata['splits']:
            split_name = split['name']
            split_path = os.sep.join([path, split["path"]])
            split_description = split.get("description", '')
            split_columns = split.get('columns', None)
            if split_path.split(".")[-1] == "csv":
                data = pd.read_csv(split_path)
                split = Split(data=data,
                              name=split_name,
                              path=split_path,
                              description=split_description,
                              columns=split_columns)
                splits.append(split)
            elif split_path.split(".")[-1] == "arff":
                data = pd.DataFrame(arff.loadarff(split_path)[0])
                split = Split(data=data,
                              name=split_name,
                              path=split_path,
                              description=split_description,
                              columns=split_columns)
                splits.append(split)
            else:
                print(f"split {split_path}: unsupported format")

        return cls(
            name=dataset_metadata["name"],
            description=dataset_metadata["description"],
            goal=dataset_metadata["goal"],
            splits=splits,
        )

    def get_description(self):
        train_split = next(split for split in self.splits if split.name == self.train_split_name)
        split_description_lines = [split.get_description() for split in self.splits]
        column_descriptions = train_split.get_column_descriptions()

        introduction_lines = [
                                 f"Assume we have a dataset called '{self.name}', which describes {self.description}, and the task is to {self.goal}.",
                                 f""
                             ] + split_description_lines + [
                                 f"Below is the type (numeric or string), unique value count and ratio for each "
                                 f"column, and few examples of values:",
                                 f""
                             ] + column_descriptions + [
                                 f"",
                             ]

        return "\n".join(introduction_lines)

    def get_metadata_description(self):
        splits_names = [split.name for split in self.splits]
        description = f"name: {self.name} \ndescription: {self.description} \ntrain_split_name: {self.train_split_name} \nsplits: {splits_names}"
        return description
