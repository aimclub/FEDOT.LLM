from typing_extensions import Literal
from fedot_llm.data.data import Dataset


def set_split(split_name: str, split_type: Literal["train", "test"], dataset: Dataset):
    split_name = split_name.split(".")[0]
    if split_type == "train":
        dataset.train_split = split_name
    elif split_type == "test":
        dataset.test_split = split_name
    else:
        raise ValueError(f"Not supported split type: {split_type}")
