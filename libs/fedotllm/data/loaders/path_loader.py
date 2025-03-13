from pathlib import Path
from typing import Union, List

import pandas as pd
import numpy as np
import json
from scipy.io.arff import loadarff

from fedotllm.data import Dataset, MultimodalSplit, Split

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

class PathDatasetLoader:
    @staticmethod
    def load(path: Union[Path, str]):
        """
        Load Dataset a folder with dataset objects

        Args:
            path: Path to folder with Dataset data
        """

        if isinstance(path, str):
            path = Path(path)

        # Loading all data in folder
        splits = []
        if path.is_dir():
            files = [x for x in path.glob('**/*') if x.is_file()]
        else:
            files = [path]

        # Checking for multimodality (image-text for now)
        image_extensions = ["jpg", "jpeg", "png"]
        metadata_extensions = ["json"]

        image_files = [file for file in files if file.name.split(".")[-1] in image_extensions]
        metadata_files = [file for file in files if file.name.split(".")[-1] in metadata_extensions]

        image_file_names = [file.name.split(".")[0] for file in image_files]
        metadata_file_names = [file.name.split(".")[0] for file in metadata_files]

        is_multimodal = image_file_names and image_file_names == metadata_file_names

        # Load all data (as a single split for now, will get 
        # train/test splitted before calling framework)
        if is_multimodal:
            imgs = [cv2.imread(img_path) for img_path in image_files]
            metadatas = [json.load(open(metadata_path, 'r')) for metadata_path in metadata_files]

            split = MultimodalSplit(image_data=np.asarray(imgs), 
                                    data = pd.DataFrame.from_records(metadatas), 
                                    name="all")
            splits.append(split)
            return Dataset(splits=splits, path=path)

        for file in files:
            file_path = file.absolute()
            split_name = file.name
            if file.name.split(".")[-1] == "csv":
                raw_data = pd.read_csv(file_path)
                split = Split(data=raw_data, name=split_name)
                splits.append(split)
            if file.name.split(".")[-1] == "arff":
                raw_data = loadarff(file_path)
                split = Split(
                    data=pd.DataFrame(raw_data[0]), name=split_name
                )
                splits.append(split)
            if file.name.split(".")[-1] in ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt']:
                raw_data = pd.read_excel(file_path)
                split = Split(data=raw_data, name=split_name)
                splits.append(split)

        return Dataset(splits=splits, path=path)
