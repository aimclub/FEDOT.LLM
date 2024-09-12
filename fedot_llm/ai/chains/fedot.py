from fedot_llm.ai.chains.base import BaseRunnableChain
from fedot.api.main import Fedot
from langchain_core.runnables import RunnableLambda

class FedotPredictChain(BaseRunnableChain):
    """Run a Fedot AutoML on the input data
    
    Args
    ----
        seed: int
            The seed for the model.
        timeout: int
            The timeout for the model.
        cv_folds: int
            The number of cross-validation folds.
        with_tuning: bool
            Whether to tune the model.
        metric: list
            The metrics to use.
        n_jobs: int
            The number of jobs to

    Parameters
    ----
        train: pd.DataFrame
            The training data.
        test: pd.DataFrame
            The test data.
        target: str
            Name of the target column.
        task_type: str
            The type of the task.
    
    Returns
    ----
        dict
            The predictions and the model, model, and best pipeline.
    
    Examples
    ----
    >>> import pandas as pd
    >>> FedotPredictChain().invoke({"train": pd.DataFrame('train.csv'),
                                    "test": pd.DataFrame('test.csv'),
                                    "target": 'target',
                                    "task_type": 'classification'})
    {'predictions': ndarray, 'auto_model': Fedot, 'best_pipeline': Pipeline}
    """
    
    @staticmethod
    def predict(input):
        auto_model = Fedot(problem=input['task_type'],
                            timeout=1,
                            cv_folds=10,
                            with_tuning=True,
                            metric=['roc_auc', 'accuracy'],
                            n_jobs=-1)
        best_pipeline = auto_model.fit(
            features=input['train'], target=input['target_column'])
        predictions = auto_model.predict(features=input['test'])
        return {'predictions': predictions, 'auto_model': auto_model, 'best_pipeline': best_pipeline}
        
    def __init__(self, seed=42, timeout=1, cv_folds=10, with_tuning=True, metric=['roc_auc', 'accuracy'], n_jobs=-1):
        self.seed = seed
        self.timeout = timeout
        self.cv_folds = cv_folds
        self.with_tuning = with_tuning
        self.metric = metric
        self.n_jobs = n_jobs
         
        self.chain = (
            RunnableLambda(self.predict)
        )