import os
import tempfile

import pandas as pd
import pytest

from fedot_llm.data import Dataset, Split
from fedot_llm.language_models.actions import ModelAction
from fedot_llm.language_models.llms import OllamaLLM


class TestModelAction:
    SYSTEM_PROMPT = "system prompt"
    CONTEXT_PROMPT = "context prompt"
    TASK_PROMPT = "task prompt"

    @pytest.fixture
    def model_action(self):
        model = OllamaLLM(model="llama3")
        return ModelAction(model)

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame(
            {
                "Room": [1, 2, 3, 1, 2, 4],
                "Age": [11, 22, 33, 44, 55, 66],
                "Sex": ['Male', 'Female', 'Male', 'Male', 'Female', 'Male'],
                "Country": ['RU', None, 'CN', 'USA', None, None],
            }
        )

    @pytest.fixture
    def sample_split(self, sample_data):
        return Split(data=sample_data, name="test_split", path="path")

    @pytest.fixture
    def sample_dataset(self, sample_split):
        return Dataset(splits=sample_split, name="dataset", description="description")

    def test_run_model_call(self, model_action):
        system = self.SYSTEM_PROMPT
        context = self.CONTEXT_PROMPT
        task = self.TASK_PROMPT
        response = model_action.run_model_call(system, context, task)
        assert isinstance(response, str)

    def test_run_model_multicall(self, model_action):
        prompts = {
            "task1": {
                "system": f"{self.SYSTEM_PROMPT} 1",
                "context": f"{self.CONTEXT_PROMPT} 1",
                "task": f"{self.TASK_PROMPT} 1",
            },
            "task2": {
                "system": f"{self.SYSTEM_PROMPT} 2",
                "context": f"{self.CONTEXT_PROMPT} 2",
                "task": f"{self.TASK_PROMPT} 2",
            },
        }
        responses = model_action.run_model_multicall(prompts)
        assert isinstance(responses, dict)
        assert len(responses) == len(prompts)

    def test_process_model_responses(self, model_action):
        responses = {"task1": "response 1", "task2": "response 2"}
        operations = {"task1": lambda x: x.upper(), "task2": lambda x: x.lower()}
        processed_responses = model_action.process_model_responses(
            responses, operations
        )
        assert isinstance(processed_responses, dict)
        assert len(processed_responses) == len(responses)
        assert processed_responses["task1"] == "RESPONSE 1"
        assert processed_responses["task2"] == "response 2"

    def test_process_model_responses_for_v1(self, model_action):
        responses = {
            "categorical_columns": "column1\ncolumn2",
            "task_type": "ClaSSifiCATion",
        }
        processed_responses = model_action.process_model_responses_for_v1(responses)
        assert isinstance(processed_responses, dict)
        assert len(processed_responses) == len(responses)
        assert processed_responses["categorical_columns"] == ["column1", "column2"]
        assert processed_responses["task_type"] == "classification"

    def test_save_model_responses(self, model_action):
        responses = {"task1": "response 1", "task2": "response 2"}
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_action.save_model_responses(responses, tmp_dir)
            assert os.path.exists(os.path.join(tmp_dir, "model_responses.json"))

    def assert_column_description(self, column_name, descriptions):
        assert column_name in descriptions
        assert isinstance(descriptions[column_name], str)

    def test_generate_all_column_description(
        self, model_action, sample_split, sample_dataset
    ):
        split = sample_split
        dataset = sample_dataset
        descriptions = model_action.generate_all_column_description(split, dataset)
        assert isinstance(descriptions, dict)
        assert len(descriptions) == len(split.data.columns)
        for column in split.data.columns:
            self.assert_column_description(column, descriptions)

    def test_get_categorical_features(self, model_action, sample_split, sample_dataset):
        split = sample_split
        dataset = sample_dataset
        descriptions = model_action.generate_all_column_description(split, dataset)
        split.set_column_descriptions(descriptions)
        categorical_features = model_action.get_categorical_features(split, dataset)
        assert isinstance(categorical_features, list)
    
