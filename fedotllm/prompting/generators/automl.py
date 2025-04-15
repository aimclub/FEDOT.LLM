from typing import List

import fedotllm.prompting.prompts.automl as automl_prompts
from fedotllm.prompting.generators.base import JsonFieldPromptGenerator
from fedotllm.utils.parsers import get_outer_columns


class TaskTypePromptGenerator(JsonFieldPromptGenerator):
    fields = ["task_type"]

    def __init__(self, data_description: str):
        self.prompt = automl_prompts.task_type_prompt(data_description, self.fields)


class DataFilePromptGenerator(JsonFieldPromptGenerator):
    fields = ["train_data", "test_data", "sample_submission_data"]

    def __init__(self, data_description: str, filenames: List[str]):
        filenames_prompt = ""
        for filename in filenames:
            filenames_prompt += f"File:\n\n{filename}"
        self.prompt = automl_prompts.data_file_name_prompt(
            data_description, filenames_prompt, self.fields
        )


class LabelColumnPromptGenerator(JsonFieldPromptGenerator):
    fields = ["label_column"]

    def __init__(self, data_description: str, column_names: List[str]):
        self.prompt = automl_prompts.label_column_prompt(
            data_description, get_outer_columns(column_names), self.fields
        )


class ProblemTypePromptGenerator(JsonFieldPromptGenerator):
    fields = ["problem_type"]

    def __init__(self, data_description: str):
        self.prompt = automl_prompts.problem_type_prompt(data_description, self.fields)


class IDColumnPromptGenerator(JsonFieldPromptGenerator):
    fields = ["id_column"]

    def __init__(
        self, data_description: str, column_names: List[str], label_column: str
    ):
        self.prompt = automl_prompts.id_column_prompt(
            data_description, get_outer_columns(column_names), label_column, self.fields
        )


class TestIDColumnPromptGenerator(IDColumnPromptGenerator):
    fields = ["test_id_column"]


class TrainIDColumnPromptGenerator(IDColumnPromptGenerator):
    fields = ["train_id_column"]


class OutputIDColumnPromptGenerator(IDColumnPromptGenerator):
    fields = ["output_id_column"]


class EvalMetricPromptGenerator(JsonFieldPromptGenerator):
    fields = ["eval_metric"]

    def __init__(self, data_description: str, metrics: List[str]):
        self.prompt = automl_prompts.evaluation_metrics_prompt(
            data_description, metrics, self.fields
        )
