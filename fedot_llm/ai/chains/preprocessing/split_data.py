
from langchain_core.runnables import RunnableLambda
from sklearn.model_selection import train_test_split

from fedot_llm.ai.chains.base import BaseRunnableChain


class SplitDataChain(BaseRunnableChain):
    """Split data chain
    
    Splits the data into train and test sets.
    
    Args
    ----
        train_size: float
            The size of the training set.
        random_state: int
            The random state for the split.
    
    Parameters
    ----------
        data: pd.DataFrame
            The data to split.
    
    Returns
    -------
    dict
        The training and testing data.
    
    Examples
    --------
    >>> SplitDataChain().invoke({"data": pd.DataFrame('data.csv')})
    {'train': '1,2,3,4,5', 'test': '6,7,8,9,10'}
    """
    def process(self, input):
        new_train, new_test = train_test_split(input['data'], train_size=self.train_size, random_state=self.random_state)
        return {'train': new_train, 'test': new_test}
    def __init__(self, train_size:float = 0.8, random_state:int = 42):
        self.train_size = train_size
        self.random_state = random_state
        self.chain = (
            RunnableLambda(self.process)
        )