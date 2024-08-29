import os

from pathlib import Path
from typing import Union, List, Dict, Literal
from pandas import Series, DataFrame

from tqdm import tqdm
from ucimlrepo import fetch_ucirepo 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import pandas as pd

def check_dataset_loaded(name: str, datasets_store_path: Union[Path, str]):
    """
    Check if a benchmark dataset is loaded

    Args:
        name: name of the dataset that coincides with folder name
        datasets_store_path: Path to folder where datasets are stored
    """
    full_path = os.path.join(datasets_store_path, name)
    return os.path.exists(full_path) and os.listdir(full_path)
    
def fetch_uci_dataset(id: int) -> tuple[DataFrame, DataFrame]:
    """
    Fetch a dataset from UCI

    Args:
        id: id of the dataset
        
    Returns:
        tuple[DataFrame, DataFrame]: A pandas dataframe containing dataset and exra data like column roles, types, etc.
    """
    dataset = fetch_ucirepo(id = id)
    df: DataFrame = dataset.data.original
    extra_data: DataFrame = dataset.variables
    return df, extra_data

def translate_openml_types(type):
    type_translator = {
        "float64": "Numeric",
        "category": "Categorical",
        "string": "String"
    }
    if type in type_translator:
        return type_translator[type]
    else:
        return type

def fetch_openml_dataset(id: int) -> tuple[DataFrame, DataFrame]:
    """
    Fetch a dataset from OpenML

    Args:
        id: id of the dataset

    Returns:
        tuple[DataFrame, DataFrame]: A pandas dataframe containing dataset and extra data like column roles, types, etc.
    """
    dataset = fetch_openml(data_id = id)
    df = dataset.frame
    cols = dataset.frame.columns 
    targets = [dataset.target.name] if isinstance(dataset.target, Series) else dataset.target.columns
    roles = cols.map(lambda x: "Target" if x in targets else "Feature")
    types = dataset.frame.dtypes.map(translate_openml_types)
    extra_data = DataFrame.from_dict({
        "name": cols,
        "role": roles,
        "type": types
    })
    return df, extra_data

def fetch_dataset(id: int, source: Literal['uci', 'openml']):
    """
    Fetch a dataset from one of the available sources

    Args:
        id: id of the dataset
        source: the source to download the dataset from (currently uci and openml are supported)
    
    Returns:
        tuple[DataFrame, DataFrame]: A pandas dataframe containing dataset and exra data like column roles, types, etc.
    """
    match source:
        case "uci":
            return fetch_uci_dataset(id)
        case "openml":
            return fetch_openml_dataset(id)
        
        
@staticmethod            
def fetch_and_save_dataset(metadata: Dict[str, any], datasets_store_path: Union[Path, str]):
    """
    Fetch and save dataset from metadata, from one of the available sources

    Args:
        metadata: a dict with dataset metadata
        datasets_store_path: path to folder where dataset should be stored
    """
    id = int(metadata['id'])
    source = metadata['source']
    df, extra_data = fetch_dataset(id, source)
    full_path = os.path.join(datasets_store_path, metadata['name'])
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train.to_csv(os.path.join(full_path, 'train.csv'))
    test.to_csv(os.path.join(full_path, 'test.csv'))
    extra_data.to_csv(os.path.join(full_path, 'extra.csv'))

@staticmethod
def fetch_and_save_all_datasets(dataset_metadatas: List[Dict[str, any]], 
                                datasets_store_path: Union[Path, str],
                                force_reload: bool = False):
    """
    Fetch and save datasets from list of metadatas, one of the available sources

    Args:
        dataset_metadatas: a list of dicts with dataset metadata
        datasets_store_path: path to folder where datasets should be stored
    """
    print("Fetching datasets")
    for metadata in tqdm(dataset_metadatas):
        if force_reload or not check_dataset_loaded(metadata['name'], datasets_store_path):
            fetch_and_save_dataset(metadata, datasets_store_path)

