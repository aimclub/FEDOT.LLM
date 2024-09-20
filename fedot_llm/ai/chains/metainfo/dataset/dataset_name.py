from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from fedot_llm.ai.chains.base import BaseRunnableChain
from fedot_llm.data import Dataset

DATASET_NAME_TEMPLATE = ChatPromptTemplate([
    ('system', 'Define a concisethe name of this dataset. Answer only with the name.'),
    ('human', '{dataset_description}')
])
"""INPUT: 
- dataset_description -- user input big description of dataset"""


class DatasetNameChain(BaseRunnableChain):
    """Define dataset name
    
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
        str
            The name of the dataset
            
    Examples
    --------
    >>> DatasetNameChain(model, dataset).invoke({"dataset_description": "This is a dataset description"})
    'dataset_name'
    """

    def __init__(self, model: BaseChatModel, dataset: Dataset):
        self.chain = (DATASET_NAME_TEMPLATE
                      | model
                      | StrOutputParser()
                      | (lambda x: setattr(dataset, "name", x) or x))
