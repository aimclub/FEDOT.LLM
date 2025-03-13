import numpy as np

from pandas import DataFrame
from typing import List

from fedotllm.data import Dataset
from fedotllm.agents.automl_multimodal.state import AutoMLMultimodalAgentState
from fedotllm.agents.automl_multimodal.structured import ProblemReflection
from fedotllm.llm.inference import AIInference
from fedotllm.log import get_logger
from fedotllm.settings.config_loader import get_settings

logger = get_logger()

def _get_numeric_columns(df: DataFrame) -> List[str]:
    return df.select_dtypes(include=np.number).columns.tolist()

def _is_string_col(col: str, 
                   df: DataFrame,
                   not_strict: bool = False #True allows for single nested string values, e.g. ['Example']
                   ) -> bool:
    item = df.iloc[0][col]
    is_nested_string = isinstance(item, List) and len(item) == 1 and isinstance(item[0], str)
    if isinstance(item, str) or not_strict and is_nested_string:
        return True
    return False

def _get_string_columns(df: DataFrame) -> List[str]:
    object_cols = df.select_dtypes(include=object).columns
    return [col for col in object_cols if _is_string_col(col, df)]

def run_problem_reflection(state: AutoMLMultimodalAgentState, inference: AIInference, 
                           dataset: Dataset):
    logger.info("Running problem reflection")

    dataset_description = []
    for split in dataset.splits:
        
        #For now we drop any columns with missing values
        split_data = split.data
        split_data = split_data[split_data.columns[~split_data.isna().any()]]

        numeric_columns = _get_numeric_columns(split_data)
        string_columns = _get_string_columns(split_data)
        other_columns = [col for col in split_data.columns if col not in string_columns + numeric_columns]

        dataset_description.append(
            "<dataset-split>\n" +
            f"{split.name}\n" +
            "<numeric-features>\n" +
            '\n'.join([f'- {col}' for col in numeric_columns]) +
            "\n</numeric-features>\n" +
            "<text-features>\n" +
            '\n'.join([f'- {col}' for col in string_columns]) +
            "\n</text-features>\n" +
            "<other-features>\n" +
            '\n'.join([f'- {col}' for col in other_columns]) +
            "\n</other-features>\n" +
            "</dataset-split>"
        )
    dataset_description = "\n".join(dataset_description)

    state['dataset_splits_description'] = dataset_description
    reflection = inference.chat_completion(
        get_settings().prompts.automl.run_problem_reflection.user.format(
            description=state['description'],
            dataset_description=dataset_description
        ),
        structured=ProblemReflection
    )
    state['reflection'] = reflection
    return state


