from pathlib import Path
from typing import Union

from scipy.io.arff import loadarff
import pandas as pd

from fedot_llm.agents.automl.data.data import Dataset, Split


class PathDatasetLoader:
    @staticmethod
    def load(path: Union[Path, str]):
        """
        Load Dataset a folder with dataset objects

        Args:
            path: Path to folder with Dataset data
            with_metadata: Whether Dataset should be loading metadata.json file contained in folder. Defaults to false.

        """

        if isinstance(path, str):
            path = Path(path)

        # Loading all splits in folder
        splits = []
        if path.is_dir():
            files = [x for x in path.glob('**/*') if x.is_file()]
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
                split = Split(
                    data=pd.DataFrame(raw_data[0]), name=split_name
                )
                splits.append(split)
            if file.name.split(".")[-1] in ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt']:
                raw_data = pd.read_excel(file_path)
                split = Split(data=raw_data, name=split_name)
                splits.append(split)

        return Dataset(splits=splits, path=path)
