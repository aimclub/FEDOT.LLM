from pydantic import BaseModel, Field, ConfigDict
from typing import List
from pathlib import Path

import pandas as pd

class Split(BaseModel):
    """
    Split within dataset object
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    """ The name of the split """
    data: pd.DataFrame = Field(repr=False)
    """ Data that is stored in the split """

class Dataset(BaseModel):
    splits: List[Split]
    """ List of splits in the dataset """
    path: Path = Field(default=None)
    """ Path to the dataset """