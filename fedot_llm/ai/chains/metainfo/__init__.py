from langchain_core.language_models.chat_models import BaseChatModel

from fedot_llm.ai.chains.base import (
    BaseRunnableChain, ChainPassthrough,
    ChainMapToClassName, ChainAddKey)
from fedot_llm.ai.chains.metainfo.dataset.define_dataset import \
    DefineDatasetChain
from fedot_llm.ai.chains.metainfo.splits.define_splits import DefineSplitsChain
from fedot_llm.ai.chains.metainfo.task import DefineTaskChain
from fedot_llm.data import Dataset
from langchain_core.runnables import RunnablePick, RunnableLambda


class DefineMetaInfo(BaseRunnableChain):
    """Define meta information of the task and dataset
    
    Args
    ----
        model: BaseChatModel
            The chat model to use.
        dataset: Dataset
            The dataset object.
        
    Parameters
    ----------
        dataset_description: str
            The long description of the dataset.

    Returns
    -------
    dict
        The name, description, goal, train and test split of the dataset, target column, and task type.
    
    Examples
    --------
    >>> DefineMetaInfo(model, dataset).invoke({"dataset_description": "This is a long dataset description"})
    {
        'DefineDatasetChain': {
                                'name': 'dataset_name',
                                'description': 'This is a short dataset description',
                                'goal': 'This is a dataset goal'
                              },
        'DefineSplitsChain':  {
                                'train': 'train',
                                'test': 'test'
                              },
        'DefineTaskChain': {
                            'target_column': 'target_column',
                            'task_type': 'regression'
                          }
    }
    """

    def __init__(self, model: BaseChatModel, dataset: Dataset):
        self.chain = (
                ChainMapToClassName(DefineDatasetChain(model, dataset))
                | ChainAddKey("dataset_detailed_description", RunnableLambda(lambda _: dataset.detailed_description))
                | ChainPassthrough(DefineSplitsChain(model, dataset))
                | ChainPassthrough(DefineTaskChain(model, dataset))
                | RunnablePick(['DefineDatasetChain', 'DefineSplitsChain', 'DefineTaskChain'])
        )
