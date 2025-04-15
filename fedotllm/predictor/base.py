from typing import Any, Optional, Union

import pandas as pd

from .task import PredictionTask
from fedotllm.constants import NO_ID_COLUMN_IDENTIFIED

class Predictor:
    def fit(
        self, task: PredictionTask, time_limit: Optional[float] = None
    ) -> "Predictor":
        return self

    def predict(self, task: PredictionTask) -> Any:
        raise NotImplementedError

    def fit_predict(self, task: PredictionTask) -> Any:
        return self.fit(task).predict(task)

    def save_artifacts(self, path: str) -> None:
        raise NotImplementedError
    
    def make_prediction_outputs(
        self,
        task: PredictionTask,
        predictions: Union[pd.DataFrame, pd.Series]
    ) -> pd.DataFrame:
        if isinstance(predictions, pd.Series):
            outputs = predictions.to_frame()
        else:
            outputs = predictions.copy()
        
        assert task.label_column is not None, "Label column is not set"
        # Ensure we only keep required output columns from predictions
        common_cols = [col for col in task.output_columns if col in outputs.columns]
        outputs = outputs[common_cols]
        
        # Handle specific test ID column if providded and detected
        if task.test_id_column is not None and task.test_id_column != NO_ID_COLUMN_IDENTIFIED:
            test_ids = task.test_data[task.test_id_column]
            output_ids = task.sample_submission_data[task.output_id_column]
            
            if not test_ids.equals(output_ids):
                print("Warming: Test IDs and output IDs do not match!")
                
            # Ensure test ID column is included
            if task.test_id_column not in outputs.columns:
                outputs = pd.concat([task.test_data[task.test_id_column], outputs], axis="columns")
        
        # Handle undetected ID columns
        missing_columns = [col for col in task.output_columns if col not in outputs.columns]
        if missing_columns:
            print(
                "Warming: The following columns are not in predictions and will be treated as ID columns:"
                f"{missing_columns}"
            )
            
            for col in missing_columns:
                if task.test_data is not None and col in task.test_data.columns:
                    # Copy from test data if available
                    outputs[col] = task.test_data[col]
                    print(f"Warming: Copied from test data for column '{col}'")
                else:
                    # Generate unique integer values
                    outputs[col] = range(len(outputs))
                    print(f"Warming: Generated unique integer values for column '{col}'"
                        "as it was not found in test data")
        
        # Ensure columns are in the correct order
        outputs = outputs[task.output_columns]
        
        return outputs
