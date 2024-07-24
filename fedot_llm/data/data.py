import json
import os
import arff
import pandas as pd

from typing import Any, Dict



class Split:
    """
    Split within dataset object
    """

    def __init__(self, name: str, data: pd.DataFrame, path: str, description: str | None = None, columns: dict | None = None) -> None:
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
        if self.name is not None:
            description = f"The {self.name} split"
        else:
            description = f"The split"

        if self.path is not None:

            fname = os.path.split(self.path)[-1]

            description += f' stored in file "{fname}"'

        description += f" contains following columns: {list(self.data.columns)}."

        if self.name is not None:
            description += f" It is described as {self.description}"

        return description
    
    def get_metadata_description(self):
        return f"name: {self.name} \npath: {self.path} \ndescription: {self.description}"
    
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

    def __init__(self, splits: Split, name: str = None, description: str = None, goal: str = None) -> None:
        """
        Initialize an instance of a Dataset.
        """
        self.name = name
        self.description = description
        self.goal = goal
        self.splits = splits
        self.train_split_name = train_split_name

    @classmethod
    def load_from_path(cls, path, with_metadata = False):
        """
        Load Dataset a folder with dataset objects
        
        Args:
            path: Path to folder with Dataset data
            with_metadata: Whether Dataset should be loading metadata.json file contained in folder. Defaults to false. 

        """

        if with_metadata:
        
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
                splits=splits
            )
        
        else:
            #Loading all splits in folder
            splits = [] 
            for fpath in os.listdir(path):
                file_path = os.path.join(path, fpath)
                split_name = os.path.split(file_path)[-1].split(".")[0]
                if file_path.split(".")[-1] == "csv":
                    data = pd.read_csv(file_path)
                    split = Split(data = data,
                                path = file_path,
                                name = split_name)
                    splits.append(split)
                if file_path.split(".")[-1] == "arff":
                    data = arff.loadarff(file_path)
                    split = Split(data = pd.DataFrame(data[0]),
                                path = file_path,
                                name = split_name)
                    splits.append(split)
            
            return cls(splits = splits)

    def get_description(self):
        split_description_lines = [split.get_description() for split in self.splits]

        first_line = "Assume we have a dataset"

        if self.name is not None:
            first_line += f" called '{self.name}.'"

        if self.description is not None:
            first_line += f"\n It could be described as following: {self.description}"

        if self.goal is not None:
            first_line += f"\n The goal is: {self.goal}"

        introduction_lines = [
            first_line,
            f"The dataset contains the following splits:",
            f" ",
        ] + split_description_lines 
        
        train_split = next((split for split in self.splits if split.name == self.train_split_name), None)
        if train_split is not None:
            column_descriptions = train_split.get_column_descriptions()
            introduction_lines += [
                f"Below is the type (numeric or string), unique value count and ratio for each column, and few examples of values:",
                f""
            ] + column_descriptions + [
                f"",
            ]

        return "\n".join(introduction_lines)
    
    def get_metadata_description(self):
        splits_metadatas = [split.get_metadata_description() for split in self.splits]
        description = f"name: {self.name} \ndescription: {self.description} \ngoal: {self.goal} \ntrain_split_name: {self.train_split_name} \nsplits:\n\n"
        description += '\n\n'.join(splits_metadatas)
        return description
