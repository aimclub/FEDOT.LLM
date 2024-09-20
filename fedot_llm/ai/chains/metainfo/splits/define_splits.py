from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableParallel

from fedot_llm.ai.chains.base import BaseRunnableChain
from fedot_llm.ai.chains.metainfo.splits.test_split import DatasetTestSplitChain
from fedot_llm.ai.chains.metainfo.splits.train_split import DatasetTrainSplitChain
from fedot_llm.data import Dataset


class DefineSplitsChain(BaseRunnableChain):
    """Define dataset splits
    
    Args
    ----
        model: BaseChatModel
            The chat model to use.
        dataset: Dataset
            The dataset object.
    Parameters
    ----------
        dataset_detailed_description: str
            The formated description of the dataset.
    
    Returns
    -------
    dict
        The train and test split of the dataset
    
    Examples
    --------
    >>> DefineSplitsChain(model, dataset).invoke({"dataset_detailed_description": dataset.detailed_description})
    {'train': 'train', 'test': 'test'}
    """

    def __init__(self, model: BaseChatModel, dataset: Dataset):
        self.chain = RunnableParallel({
            "train": DatasetTrainSplitChain(model, dataset),
            "test": DatasetTestSplitChain(model, dataset)
        })
