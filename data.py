import json
import os
import requests
import arff
import pandas as pd

from typing import Any


class Split:
    """
    Split within dataset object
    """
    def __init__(self, name, data, path, description) -> None:
        """
        Initialize an instanse of a Split.
        """
        self.name = name
        self.data = data
        self.path = path
        self.description = description

    def get_description(self):
        return f"The {self.name} split contains following columns: {self.data.columns}. It is described as {self.description}"
    
    def get_text_columns(self):
        return list(self.data.select_dtypes(include=['object']).columns)
    
    def get_numeric_columns(self):
        return list(self.data.select_dtypes(include=['number']).columns)

    def get_unique_counts(self):
        return self.data.apply(lambda col: col.nunique())
    
    def get_unique_ratios(self):
        return self.data.apply(lambda col: col.nunique() / len(col))
    
    def get_column_types(self):
        return self.data.apply(lambda col: "string" if col.name in self.get_text_columns() else "numeric")
    
    def get_head_by_column(self, column_name, count = 10):
        return list(self.data[column_name].head(count))
                    
    def get_column_descriptions(self):
        unique_counts = self.get_unique_counts()
        unique_ratios = self.get_unique_ratios()
        column_types = self.get_column_types()
        
        column_descriptions = [f"{column_name}: {column_types[column_name]}"
                            f"{100 * unique_ratios[column_name]:.2f}% unique values, examples: {self.get_head_by_column(column_name)}"
                            for column_name in self.data.columns]
        return column_descriptions


class Dataset:
    """
    Dataset object that represents an ML task and may contain multiple splits
    """
    def __init__(self, name, description, goal, splits, train_split_name) -> None:
        """
        Initialize an instance of a Dataset.
        """
        self.name = name
        self.description = description
        self.goal = goal
        self.splits = splits
        self.train_split_name = train_split_name

    @classmethod
    def load_from_path(cls, path):
        """
        Load Dataset a folder with dataset objects

        WIP: add support for incomplete descriptions?
        """
        with open(os.sep.join([path, 'metadata.json']), 'r') as json_file:
            dataset_metadata = json.load(json_file)
        
        #load each split file
        splits = []
        for split_name in dataset_metadata['split_names']:
            split_path = os.sep.join([path, dataset_metadata["split_paths"][split_name]]) 
            split_description = dataset_metadata["split_descriptions"][split_name] 
            if split_path.split(".")[-1] == "csv":
                data = pd.read_csv(split_path)
                split = Split(data = data,
                              name = split_name,
                              path = split_path,
                              description = split_description)
                splits.append(split)
            elif split_path.split(".")[-1] == "arff":
                data = arff.loadarff(split_path)
                split = Split(data = pd.DataFrame(data[0]),
                              name = split_name,
                              path = split_path,
                              description = split_description)
                splits.append(split)
            else:
                print(f"split {split_path}: unsupported format")

        #if we have model responses saved already = obasolete for now
        # if os.path.exists(os.sep.join([path,'model_responses.json'])):
        #     with open(os.sep.join([path, 'model_responses.json']), 'r') as json_file:
        #         model_responses = json.load(json_file)
        #         dataset_metadata.update(model_responses)
        
        return cls(
            name = dataset_metadata["name"],
            description = dataset_metadata["description"],
            goal = dataset_metadata["goal"],
            splits = splits,
            train_split_name = dataset_metadata["train_split_name"],
            )

    def get_description(self):
        train_split = next(split for split in self.splits if split.name == self.train_split_name)
        split_description_lines = [split.get_description() for split in self.splits]
        column_descriptions = train_split.get_column_descriptions()

        introduction_lines = [
            f"Assume we have a dataset called '{self.name}', which describes {self.description}, and the task is to {self.goal}.",
            f""
        ] + split_description_lines + [
            f"Below is the type (numeric or string), unique value count and ratio for each column, and few examples of values:",
            f""
        ] + column_descriptions + [
            f"",
        ]

        return "\n".join(introduction_lines)
    
    def get_metadata_description(self):
        splits_names = [split.name for split in self.splits ]
        description = f"name: {self.name} \ndescription: {self.description} \ntrain_split_name: {self.train_split_name} \nsplits: {splits_names}"
        return description