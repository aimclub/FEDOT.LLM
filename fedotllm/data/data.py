from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict


class Split(BaseModel):
    """
    Split within dataset object
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    """ The name of the split """
    data: pd.DataFrame = Field(repr=False)
    """ Data that is stored in the split """

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data": self.data.to_dict(orient="records")
        }


class Dataset(BaseModel):
    splits: List[Split] = Field(default_factory=list)
    """ List of splits in the dataset """
    path: Path = Field(default=None)
    """ Path to the dataset """

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        return {
            "splits": [split.model_dump() for split in self.splits],
            "path": str(self.path) if self.path else None
        }
