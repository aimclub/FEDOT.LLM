import pandas as pd
import pytest

from fedot_llm.data import Split
from fedot_llm.data import Dataset

import os


def assert_unique_values(unique_values, column_name, expected_max_length):
    assert isinstance(unique_values, pd.Series)
    assert len(unique_values) <= expected_max_length
    assert unique_values.name == column_name


class TestSplit:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame(
            {
                "col1": [1, 2, 3, 1, 2, 4],
                "col2": [1, 2, 3, 4, 5, 6],
            }
        )

    @pytest.fixture
    def split_object(self, sample_data) -> Split:
        return Split(name="test_split", data=sample_data, path="path")

    @pytest.mark.parametrize("column_name", ["col1", "col2"])
    def test_get_unique_values_return_all_values(self, split_object, column_name):
        unique_values = split_object.get_unique_values(column_name=column_name)
        expected_length = 4 if column_name == "col1" else 6
        assert_unique_values(unique_values, column_name, expected_length)

    @pytest.mark.parametrize("column_name", ["col1", "col2"])
    @pytest.mark.parametrize("max_number", [2, 3, 4, 5, 6, 7])
    def test_get_unique_values_return_limited_values(
        self, split_object, column_name, max_number
    ):
        unique_values = split_object.get_unique_values(
            column_name=column_name, max_number=max_number
        )
        assert_unique_values(unique_values, column_name, max_number)


class TestDataset:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame(
            {
                "col1": [1, 2, 3, 1, 2, 4],
                "col2": [1, 2, 3, 4, 5, 6],
            }
        )

    @pytest.fixture
    def split_object(self, sample_data) -> Split:
        return Split(name="test_split", data=sample_data, path="path")

    @pytest.fixture
    def dataset_object(self, split_object) -> Dataset:
        return Dataset(splits=[split_object])
    
    def test_load_from_path_csv(self):
        dataset_path = os.sep.join(['..', '..', '..', 'datasets', 'health_insurance'])
        dataset = Dataset.load_from_path(dataset_path)
        assert len(dataset.splits) == 2
        assert dataset.splits[0].name in ["test", "train"]
        assert dataset.splits[1].name in ["test", "train"]
        
    @pytest.mark.parametrize("split_type, is_method", [
        ("train_split", "is_train"),
        ("test_split", "is_test")
    ])
    def test_split(self, dataset_object, split_object, split_type, is_method):
        assert not getattr(dataset_object, is_method)()
        setattr(dataset_object, split_type, "test_split")
        assert getattr(dataset_object, is_method)()
        assert getattr(dataset_object, split_type) == split_object

    def test_target_name(self, dataset_object):
        dataset_object.train_split = "test_split"
        dataset_object.target_name = "col1"
        assert dataset_object.target_name == "col1"

    def test_target_name_not_in_train(self, dataset_object):
        dataset_object.train_split = "test_split"
        with pytest.raises(ValueError):
            dataset_object.target_name = "col3"

    def test_set_target_name_train_is_none(self, dataset_object):
        with pytest.raises(ValueError):
            dataset_object.target_name = "col1"

    def test_target_name_not_set(self, dataset_object):
        dataset_object.train_split = "test_split"
        with pytest.raises(ValueError):
            dataset_object.target_name

    def test_task_type(self, dataset_object):
        dataset_object.task_type = "regression"
        assert dataset_object.task_type == "regression"
    
    def test_task_type_incorrect(self, dataset_object):
        with pytest.raises(ValueError):
            dataset_object.task_type = "incorrect"
    def test_task_type_not_set(self, dataset_object):
        with pytest.raises(ValueError):
            dataset_object.task_type