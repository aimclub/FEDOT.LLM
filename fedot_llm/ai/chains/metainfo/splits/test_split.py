from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from fedot_llm.ai.chains.base import BaseRunnableChain
from fedot_llm.ai.chains.metainfo.splits.utils import set_split
from fedot_llm.data import Dataset

TEST_SPLIT_TEMPLATE = ChatPromptTemplate([
    ('system', 'Define the test split of this dataset. '
               'Answer only with the name of the file with test split. '
               'Mind the register.'),
    ('human', '{dataset_detailed_description}')
])
"""INPUT:
- dataset_detailed_description: property of the dataset object"""


class DatasetTestSplitChain(BaseRunnableChain):
    """Define dataset test split
    
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
            The name of the test split file.
    
    Examples
    -------
    >>> DatasetTestSplitChain(model, dataset).invoke({"dataset_detailed_description": dataset.detailed_description})
    'test'
    """

    def __init__(self, model: BaseChatModel, dataset: Dataset):
        self.chain = (
                TEST_SPLIT_TEMPLATE
                | model
                | StrOutputParser()
                | (lambda name: name.split(".")[0].strip().strip(r",.\'\"“”‘’`´"))
                | (lambda name: set_split(str(name), "test", dataset) or name)
        )
