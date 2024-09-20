from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableParallel

from fedot_llm.ai.chains.base import BaseRunnableChain
from fedot_llm.ai.chains.metainfo.dataset.dataset_description import DatasetDescriptionChain
from fedot_llm.ai.chains.metainfo.dataset.dataset_goal import DatasetGoalChain
from fedot_llm.ai.chains.metainfo.dataset.dataset_name import DatasetNameChain
from fedot_llm.data import Dataset


class DefineDatasetChain(BaseRunnableChain):
    """Define dataset meta information
    
    This chain takes long dataset description and defines 
    the name, short description, and goal of the dataset.
    
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
        The name, description, and goal of the dataset
    
    Examples
    --------
    >>> DefineDatasetChain(model, dataset).invoke({"dataset_description": "This is a long dataset description"})
    {'name': 'dataset_name', 'description': 'This is a short dataset description', 'goal': 'This is a dataset goal'}
    """

    def __init__(self, model: BaseChatModel, dataset: Dataset, **kwargs):
        self.chain = RunnableParallel({
            "name": DatasetNameChain(model, dataset),
            "description": DatasetDescriptionChain(model, dataset),
            "goal": DatasetGoalChain(model, dataset)
        })
