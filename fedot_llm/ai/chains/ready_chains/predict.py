from fedot_llm.ai.chains.base import BaseRunnableChain, ChainAddKey
from langchain_core.language_models.chat_models import BaseChatModel
from fedot_llm.data import Dataset
from fedot_llm.ai.chains.metainfo import DefineMetaInfo
from fedot_llm.ai.chains.fedot import FedotPredictChain
from fedot_llm.ai.chains.analyze import AnalyzeFedotResultChain
from fedot_llm.ai.chains.preprocessing.split_data import SplitDataChain
from fedot_llm.ai.chains.dataset_utils import LoadSplitDataChain
from langchain_core.runnables import RunnablePick
from golem.core.dag.graph_utils import graph_structure

class PredictChain(BaseRunnableChain):
    """Predict chain
    
    Chain that takes dataset and create a model that predicts the target.
    
    Args
    ----
        model: BaseChatModel
            The model that will be used to predict.
        dataset: Dataset
            The dataset that will be used to predict.
    
    Parameters
    ----------
        dataset_description : str
            The long description of the dataset.
    
    Returns
    -------
    str
        The analysis of the built model pipeline in Markdown.
        
    Examples
    --------
    >>> PredictChain().invoke({"dataset_description": 'This is a dataset description'})
    Here is the pipeline of the model I built:...
    """
    def __init__(self, model: BaseChatModel, dataset: Dataset):
        self.chain = (
            DefineMetaInfo(model=model, dataset=dataset)
            #-------split data-------------------
            | ChainAddKey('data_splits',
                    ChainAddKey('data',
                            ChainAddKey('split', RunnablePick('DefineSplitsChain') | RunnablePick('train'))  
                            | LoadSplitDataChain(dataset=dataset))
                    | SplitDataChain())
           #------------------------------------
            | { "train": RunnablePick('data_splits') | RunnablePick('train'),
                "test": RunnablePick('data_splits') | RunnablePick('test'),
                "task_type": RunnablePick('DefineTaskChain') | RunnablePick('task_type'),
                "target_column": RunnablePick('DefineTaskChain') | RunnablePick('target_column')
            } 
            | FedotPredictChain()
            #------------------------------------
            | {
                "parameters": RunnablePick('best_pipeline') | (lambda input: graph_structure(input)),
                "metrics": RunnablePick('auto_model') | (lambda input: input.get_metrics())
            }
            | AnalyzeFedotResultChain(model)
        )