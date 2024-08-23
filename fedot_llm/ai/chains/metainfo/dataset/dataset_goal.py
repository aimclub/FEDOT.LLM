from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from fedot_llm.ai.chains.base import BaseRunnableChain
from fedot_llm.data import Dataset

DATASET_GOAL_TEMPLATE = ChatPromptTemplate([
    ('system', ('Formulate the goal associated with this dataset. Write a concisethe goal description.'
                'It should be 1 sentences long.')),
    ('human', '{dataset_description}'),
    ('ai', 'The goal is\n')
])
"""INPUT: 
- dataset_description -- user input big description of dataset"""

class DatasetGoalChain(BaseRunnableChain):
    """Define dataset goal
    
    Args
    -----
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
            The goal of the dataset.
            
    Examples
    --------
    >>> DatasetGoalChain(model, dataset).invoke({"dataset_description": "This is a dataset description"})
    'This is a dataset goal'
    """
    def __init__(self, model: BaseChatModel, dataset: Dataset, **kwargs):
        self.chain = ((DATASET_GOAL_TEMPLATE
                       | model
                       | StrOutputParser()
                       | (lambda x: setattr(dataset, "goal", x) or x)))