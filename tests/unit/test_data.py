import io
import pytest
from unittest.mock import patch, MagicMock, mock_open, call
from pathlib import Path
import pandas as pd
import numpy as np

# Assuming constants are accessible for checking suffixes if needed,
# but primarily we'll mock Path.suffix directly.
from fedotllm.constants import (
    ARFF_SUFFIXES,
    CSV_SUFFIXES,
    EXCEL_SUFFIXES,
    PARQUET_SUFFIXES,
    DATASET_EXTENSIONS
)
from fedotllm.data import load_pd, missing_values, Split, Dataset


# --- Fixtures ---
@pytest.fixture
def mock_df():
    return pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})

@pytest.fixture
def mock_path():
    mock = MagicMock(spec=Path)
    mock.name = "test_file.csv" # Default name
    mock.suffix = ".csv"        # Default suffix
    # Make it behave as an absolute path for internal calls if any
    mock.absolute.return_value = mock 
    return mock

# --- Tests for load_pd ---

@patch('pandas.read_excel')
def test_load_pd_excel(mock_read_excel, mock_path, mock_df):
    mock_path.suffix = EXCEL_SUFFIXES[0]
    mock_read_excel.return_value = mock_df
    df = load_pd(mock_path)
    mock_read_excel.assert_called_once_with(mock_path, engine="calamine")
    pd.testing.assert_frame_equal(df, mock_df)

@patch('pandas.read_parquet')
def test_load_pd_parquet_fastparquet(mock_read_parquet, mock_path, mock_df):
    mock_path.suffix = PARQUET_SUFFIXES[0]
    mock_read_parquet.return_value = mock_df
    df = load_pd(mock_path)
    mock_read_parquet.assert_called_once_with(mock_path, engine="fastparquet")
    pd.testing.assert_frame_equal(df, mock_df)

@patch('pandas.read_parquet')
def test_load_pd_parquet_pyarrow_fallback(mock_read_parquet, mock_path, mock_df):
    mock_path.suffix = PARQUET_SUFFIXES[0]
    # First call (fastparquet) raises an error, second call (pyarrow) succeeds
    mock_read_parquet.side_effect = [Exception("fastparquet failed"), mock_df]
    df = load_pd(mock_path)
    assert mock_read_parquet.call_count == 2
    mock_read_parquet.assert_any_call(mock_path, engine="fastparquet")
    mock_read_parquet.assert_any_call(mock_path, engine="pyarrow")
    pd.testing.assert_frame_equal(df, mock_df)

@patch('pandas.read_csv')
def test_load_pd_csv(mock_read_csv, mock_path, mock_df):
    mock_path.suffix = CSV_SUFFIXES[0]
    mock_read_csv.return_value = mock_df
    df = load_pd(mock_path)
    mock_read_csv.assert_called_once_with(mock_path)
    pd.testing.assert_frame_equal(df, mock_df)

@patch('fedotllm.data.loadarff') # Patch where it's imported and used
@patch('pandas.DataFrame') 
def test_load_pd_arff(mock_pd_dataframe_constructor, mock_fedot_loadarff, mock_path, mock_df):
    mock_path.suffix = ARFF_SUFFIXES[0]
    arff_data_payload = [('val1', 1), ('val2', 2)] # Just the data part
    arff_metadata_mock = MagicMock()
    # mock_fedot_loadarff is the mock for the loadarff function *within fedotllm.data module*
    mock_fedot_loadarff.return_value = (arff_data_payload, arff_metadata_mock) 
    mock_pd_dataframe_constructor.return_value = mock_df

    df = load_pd(mock_path)
    mock_fedot_loadarff.assert_called_once_with(mock_path)
    mock_pd_dataframe_constructor.assert_called_once_with(arff_data_payload)
    pd.testing.assert_frame_equal(df, mock_df)


def test_load_pd_unsupported_suffix(mock_path):
    mock_path.suffix = ".unsupported"
    with pytest.raises(Exception, match="file format for .unsupported not supported!"):
        load_pd(mock_path)

def test_load_pd_from_memory_list_of_lists(mock_df):
    data = [[1, 'a'], [2, 'b']]
    # We can't easily mock pd.DataFrame(data) if 'data' itself is the input.
    # Instead, we'll compare the result.
    expected_df = pd.DataFrame(data) 
    df = load_pd(data)
    pd.testing.assert_frame_equal(df, expected_df)

def test_load_pd_from_memory_numpy_array(mock_df):
    data = np.array([[1, 'a'], [2, 'b']])
    expected_df = pd.DataFrame(data)
    df = load_pd(data)
    pd.testing.assert_frame_equal(df, expected_df)

# --- Tests for missing_values ---

def test_missing_values_with_missing():
    data = {'col1': [1, 2, None, 4], 'col2': ['a', None, 'c', None], 'col3': [1,2,3,4]}
    df = pd.DataFrame(data)
    missing_df = missing_values(df)
    
    assert not missing_df.empty
    assert 'col2' in missing_df.index
    assert 'col1' in missing_df.index
    assert missing_df.loc['col2', 'Missing'] == 2
    assert missing_df.loc['col1', 'Missing'] == 1
    assert missing_df.loc['col2', 'Percent'] == 50.0 # (2/4)*100
    assert missing_df.loc['col1', 'Percent'] == 25.0 # (1/4)*100
    assert missing_df.index.tolist() == ['col2', 'col1'] # Sorted by missing count

def test_missing_values_none_missing():
    data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}
    df = pd.DataFrame(data)
    missing_df = missing_values(df)
    assert missing_df.empty

# --- Tests for Split class ---

def test_split_instantiation(mock_df):
    split = Split(name="train", data=mock_df)
    assert split.name == "train"
    pd.testing.assert_frame_equal(split.data, mock_df)

# --- Tests for Dataset class ---

class TestDataset:

    @patch('fedotllm.data.load_pd')
    def test_from_path_single_file(self, mock_load_pd_func, mock_path, mock_df):
        mock_path.is_dir.return_value = False
        mock_path.is_file.return_value = True
        mock_path.suffix = ".csv" # A supported extension
        mock_path.name = "my_data.csv"
        mock_load_pd_func.return_value = mock_df

        dataset = Dataset.from_path(mock_path)
        
        mock_load_pd_func.assert_called_once_with(mock_path.absolute())
        assert len(dataset.splits) == 1
        assert dataset.splits[0].name == "my_data.csv"
        pd.testing.assert_frame_equal(dataset.splits[0].data, mock_df)
        assert dataset.path == mock_path

    @patch('fedotllm.data.load_pd')
    def test_from_path_directory(self, mock_load_pd_func, mock_path, mock_df):
        mock_path.is_dir.return_value = True
        
        file1_mock = MagicMock(spec=Path)
        file1_mock.name = "train.csv"
        file1_mock.suffix = ".csv"
        file1_mock.is_file.return_value = True
        file1_mock.absolute.return_value = file1_mock

        file2_mock = MagicMock(spec=Path)
        file2_mock.name = "test.parquet"
        file2_mock.suffix = ".parquet"
        file2_mock.is_file.return_value = True
        file2_mock.absolute.return_value = file2_mock
        
        unsupported_file_mock = MagicMock(spec=Path)
        unsupported_file_mock.name = "notes.txt"
        unsupported_file_mock.suffix = ".txt" # Not in DATASET_EXTENSIONS
        unsupported_file_mock.is_file.return_value = True
        unsupported_file_mock.absolute.return_value = unsupported_file_mock

        mock_path.glob.return_value = [file1_mock, file2_mock, unsupported_file_mock]
        
        df1 = pd.DataFrame({'f1': [1]})
        df2 = pd.DataFrame({'f2': [2]})
        mock_load_pd_func.side_effect = [df1, df2] # load_pd called for supported files

        dataset = Dataset.from_path(mock_path)

        assert mock_load_pd_func.call_count == 2
        mock_load_pd_func.assert_any_call(file1_mock.absolute())
        mock_load_pd_func.assert_any_call(file2_mock.absolute())
        
        assert len(dataset.splits) == 2
        assert dataset.splits[0].name == "train.csv"
        pd.testing.assert_frame_equal(dataset.splits[0].data, df1)
        assert dataset.splits[1].name == "test.parquet"
        pd.testing.assert_frame_equal(dataset.splits[1].data, df2)

    def test_from_path_empty_directory(self, mock_path):
        mock_path.is_dir.return_value = True
        mock_path.glob.return_value = []
        dataset = Dataset.from_path(mock_path)
        assert len(dataset.splits) == 0
    
    def test_get_train_split_by_name(self):
        s1 = Split("test.csv", pd.DataFrame())
        s_train = Split("my_train_data.csv", pd.DataFrame({'a': [1]}))
        s3 = Split("other.csv", pd.DataFrame())
        dataset = Dataset(splits=[s1, s_train, s3], path=MagicMock(spec=Path))
        assert dataset.get_train_split() is s_train

        s_train_upper = Split("TRAIN.CSV", pd.DataFrame({'b': [2]}))
        dataset_upper = Dataset(splits=[s1, s_train_upper], path=MagicMock(spec=Path))
        assert dataset_upper.get_train_split() is s_train_upper

    def test_get_train_split_max_cols(self):
        s1 = Split("s1.csv", pd.DataFrame({'a': [1], 'b': [2]})) # 2 cols
        s2 = Split("s2.csv", pd.DataFrame({'x': [1], 'y': [2], 'z': [3]})) # 3 cols (train)
        s3 = Split("s3.csv", pd.DataFrame({'c': [1]})) # 1 col
        dataset = Dataset(splits=[s1, s2, s3], path=MagicMock(spec=Path))
        assert dataset.get_train_split() is s2

    def test_get_train_split_max_rows_tie_break(self):
        s1 = Split("s1.csv", pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})) # 3 rows, 2 cols
        s2 = Split("s2.csv", pd.DataFrame({'x': [1], 'y': [2]}))        # 1 row, 2 cols
        s_train = Split("s_train.csv", pd.DataFrame({'c': [1,2,3,4], 'd': [5,6,7,8]})) # 4 rows, 2 cols (train)
        dataset = Dataset(splits=[s1, s2, s_train], path=MagicMock(spec=Path))
        assert dataset.get_train_split() is s_train
        
    def test_get_train_split_no_splits(self):
        dataset = Dataset(splits=[], path=MagicMock(spec=Path))
        with pytest.raises(ValueError): # max() arg is an empty sequence
             dataset.get_train_split()

    @patch('fedotllm.data.Dataset.get_train_split')
    @patch('fedotllm.data.missing_values')
    def test_dataset_eda_less_than_10_cols(self, mock_missing_vals_func, mock_get_train, mock_df):
        train_split_mock = MagicMock(spec=Split)
        # Mock DataFrame with less than 10 columns
        df_small = pd.DataFrame({f'col{i}': [1,2] for i in range(5)}) 
        train_split_mock.data = df_small
        mock_get_train.return_value = train_split_mock
        
        mock_missing_vals_func.return_value = pd.DataFrame({'Missing': [1], 'Percent': [50.0]}, index=['col1'])

        # Mock df.info()
        with patch.object(df_small, 'info') as mock_info:
            dataset = Dataset(splits=[train_split_mock], path=MagicMock())
            eda_summary = dataset.dataset_eda()

            mock_info.assert_called_once()
            mock_missing_vals_func.assert_called_once_with(df_small)
            assert "===== 1. BASIC INFO =====" in eda_summary
            assert "===== 2. MISSING VALUES =====" in eda_summary
            assert "Missing" in eda_summary # From missing_values markdown

    @patch('fedotllm.data.Dataset.get_train_split')
    def test_dataset_eda_more_than_10_cols(self, mock_get_train):
        train_split_mock = MagicMock(spec=Split)
        # Mock DataFrame with more than 10 columns
        df_large = pd.DataFrame({f'col{i}': [1,2] for i in range(15)})
        train_split_mock.data = df_large
        mock_get_train.return_value = train_split_mock
        
        dataset = Dataset(splits=[train_split_mock], path=MagicMock())
        with patch.object(df_large, 'info') as mock_info: # Ensure info is not called
            eda_summary = dataset.dataset_eda()
            mock_info.assert_not_called()
            assert eda_summary == "" # Current logic returns empty string

    def test_dataset_eda_no_splits(self):
        dataset = Dataset(splits=[], path=MagicMock())
        assert dataset.dataset_eda() == "No data splits available"

    @patch('fedotllm.data.Dataset.get_train_split')
    def test_dataset_preview_train_gt_10_cols(self, mock_get_train):
        train_df = pd.DataFrame({f'col{i}': range(5) for i in range(15)}) # 15 cols
        train_split_mock = Split(name="train_large.csv", data=train_df)
        
        other_df = pd.DataFrame({'a': [1], 'b': [2]})
        other_split_mock = Split(name="other.csv", data=other_df)
        
        mock_get_train.return_value = train_split_mock
        
        dataset = Dataset(splits=[train_split_mock, other_split_mock], path=MagicMock())

        with patch.object(train_df, 'sample') as mock_train_sample, \
             patch.object(other_df, 'sample') as mock_other_sample:
            
            mock_train_sample.return_value.to_markdown.return_value = "train_markdown_sample"
            # other_sample should not be called for its data if train_df > 10 cols
            
            preview = dataset.dataset_preview(sample_size=3)

            mock_train_sample.assert_called_once_with(3)
            mock_other_sample.assert_not_called() # Only column list for other splits
            
            assert "File: train_large.csv" in preview
            assert "train_markdown_sample" in preview
            assert "File: other.csv" in preview
            assert f"Columns: {other_df.columns.tolist()}" in preview


    @patch('fedotllm.data.Dataset.get_train_split')
    def test_dataset_preview_train_lte_10_cols(self, mock_get_train):
        train_df = pd.DataFrame({f'col{i}': range(5) for i in range(5)}) # 5 cols
        train_split_mock = Split(name="train_small.csv", data=train_df)
        
        other_df = pd.DataFrame({'a': [10], 'b': [20]})
        other_split_mock = Split(name="other_small.csv", data=other_df)
        
        mock_get_train.return_value = train_split_mock
        dataset = Dataset(splits=[train_split_mock, other_split_mock], path=MagicMock())

        with patch.object(train_df, 'sample') as mock_train_sample, \
             patch.object(other_df, 'sample') as mock_other_sample:
            
            mock_train_sample.return_value.to_markdown.return_value = "train_markdown_sample_small"
            mock_other_sample.return_value.to_markdown.return_value = "other_markdown_sample_small"
            
            preview = dataset.dataset_preview(sample_size=2)

            mock_train_sample.assert_called_once_with(2)
            mock_other_sample.assert_called_once_with(2)
            
            assert "File: train_small.csv" in preview
            assert "train_markdown_sample_small" in preview
            assert "File: other_small.csv" in preview
            assert "other_markdown_sample_small" in preview
            assert "Columns:" not in preview 

    def test_dataset_preview_no_train_split(self): 
        dataset_no_splits = Dataset(splits=[], path=MagicMock(spec=Path)) 
        with pytest.raises(ValueError):
            dataset_no_splits.dataset_preview()

    @patch.object(Dataset, 'dataset_preview')
    def test_dataset_str(self, mock_dataset_preview):
        mock_dataset_preview.return_value = "preview_string"
        dataset = Dataset(splits=[], path=MagicMock())
        s = str(dataset)
        mock_dataset_preview.assert_called_once()
        assert s == "preview_string"
