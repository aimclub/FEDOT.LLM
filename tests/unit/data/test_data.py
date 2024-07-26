import pandas as pd
import pytest

from fedot_llm.data import Split


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
