from fedot_llm.ai.chains.base import BaseRunnableChain
from fedot_llm.data.data import Dataset
from langchain_core.runnables import RunnableLambda

class LoadSplitDataChain(BaseRunnableChain):
    """Load split data chain
    
    Load the split data from the input.
    
    Args
    ----
        dataset: Dataset
            The dataset object.
    
    Parameters
    ----------
        split: str
            Dataset's split name
    
    Returns
    -------
    pd.DataFrame
        The split data.
    
    Examples
    --------
    >>> LoadSplitDataChain().invoke({"split": 'train'})
    pd.DataFrame
    """
    
    def process(self, input):
        split = list(filter(lambda split: split.name == input['split'], self.dataset.splits))[0]
        if split:
            return split.data
        else:
            raise ValueError(f"Split {input['split']} not found in the dataset.")
        
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.chain = (
            RunnableLambda(self.process)
        )