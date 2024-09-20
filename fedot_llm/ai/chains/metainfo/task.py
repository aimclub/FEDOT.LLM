from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

from fedot_llm.ai.chains.base import BaseRunnableChain
from fedot_llm.data import Dataset

DEFINE_TARGET_TEMPLATE = ChatPromptTemplate([
    ('system', 'Your task is to return the target column of the dataset. '
               'Only answer with a column name.'),
    ('human', '{dataset_detailed_description}'),
    ('ai', 'Target column:\n')
])
"""INPUT:
- dataset_detailed_description: property of the dataset object"""

DEFINE_TASK_TYPE_TEMPLATE = ChatPromptTemplate([
    ('system', 'Your task is to define whether the task is regression or classification.'
               'Only answer with a task type'),
    ('human', '{dataset_detailed_description}')
])
"""INPUT:
- dataset_detailed_description: property of the dataset object"""


class DefineTargetColumnChain(BaseRunnableChain):
    """Define target column
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
        str
            The target column of the dataset.
    Examples
    --------
    >>> DefineTargetColumnChain(model, dataset).invoke({"dataset_detailed_description": dataset.detailed_description})
    'target_column'
    """

    def __init__(self, model: BaseChatModel, dataset: Dataset):
        self.chain = (
                DEFINE_TARGET_TEMPLATE
                | model
                | StrOutputParser()
                | (lambda x: x.strip().strip(r",.\'\"“”‘’`´"))
                | (lambda x: setattr(dataset, "target_name", x) or x)
        )


class DefineTaskTypeChain(BaseRunnableChain):
    """Define target column
    
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
        str
            The task type of the dataset.
    
    Examples
    --------
    >>> DefineTaskTypeChain(model, dataset).invoke({"dataset_detailed_description": dataset.detailed_description})
    'classification'
    """

    def __init__(self, model: BaseChatModel, dataset: Dataset):
        self.chain = (
                DEFINE_TASK_TYPE_TEMPLATE
                | model
                | StrOutputParser()
                | (lambda x: x.lower().strip().strip(r",.\'\"“”‘’`´"))
                | (lambda x: setattr(dataset, "task_type", x) or x)
        )


class DefineTaskChain(BaseRunnableChain):
    """Define target column and task type
    
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
            The target column and task type of the dataset.
    
    Examples
    --------
    >>> DefineTaskMetadescription(model, dataset).invoke({"dataset_detailed_description": dataset.detailed_description})
    {'target_column': 'target_column', 'task_type': 'classification'}
    """

    def __init__(self, model: BaseChatModel, dataset: Dataset):
        self.chain = RunnableParallel({
            "target_column": DefineTargetColumnChain(model, dataset),
            "task_type": DefineTaskTypeChain(model, dataset)
        })
