from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from fedot_llm.ai.chains.base import BaseRunnableChain
from fedot_llm.data import Dataset

DATASET_DESCRIPTION_TEMPLATE = ChatPromptTemplate([
    ('system', ('Formulate a short description this dataset.'
                'It should be no longer than a paragraph. ')),
    ('human', '{dataset_description}'),
    ('ai', 'Here is a short description of the dataset:\n\n')
])
"""INPUT: 
- dataset_description -- user input big description of dataset"""

class DatasetDescriptionChain(BaseRunnableChain):
    """Make a short description of the dataset
    
    Args
    -------
    model :  BaseChatModel
            The chat model to use.
    dataset : Dataset
            The dataset object.
    
    Parameters
    --------
    dataset_description : str
            The long description of the dataset.
    
    Returns
    --------
    str
            The description of the dataset.
            
    Examples
    --------
    >>> DatasetDescriptionChain(model, dataset).invoke({"dataset_description": "This is a long dataset description"})
    'This is a sort dataset description'
    """
    
    def __init__(self, model: BaseChatModel, dataset: Dataset):
        self.chain = ( DATASET_DESCRIPTION_TEMPLATE
                        | model
                        | StrOutputParser()
                        | (lambda x: setattr(dataset, "description", x) or x))