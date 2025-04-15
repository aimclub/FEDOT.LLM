"""A task encapsulates the data for a data science task or project. It contains descriptions, data, metadata."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from fedotllm.constants import (
    BINARY,
    DEFAULT_FORECAST_HORIZON,
    DESCRIPTION,
    MULTIMODAL,
    OUTPUT,
    PREFERED_METRIC_BY_PROBLEM_TYPE,
    REGRESSION,
    STATIC_FEATURES,
    TABULAR,
    TEST,
    TIME_SERIES,
    TRAIN,
)
from fedotllm.tabular import load_pd

logger = logging.getLogger(__name__)


class PredictionTask:
    """A task contains data and metadata for a tabular machine learning task, including datasets, metadata such as
    problem type, test_id_column, etc.
    """

    def __init__(
        self,
        metadata: Dict[str, Any],
        filepaths: List[Path] = [],
        name: Optional[str] = "",
        description: Optional[str] = "",
        files_mapping: Optional[Dict[str, Union[Path, None]]] = {},
    ):
        self.metadata: Dict[str, Any] = {
            "name": name,
            "description": description,
            "data_description_file": None,
            "task_type": None,
            "label_column": None,
            "problem_type": None,
            "eval_metric": None,
            "train_id_column": None,
            "test_id_column": None,
            "output_id_column": None,
            "images_column": None,
            "forecast_horizon": None,
            "timestamp_column": None,
        }
        if not filepaths and files_mapping:
            filepaths = [Path(v) for v in files_mapping.values() if v is not None]

        self.metadata.update(metadata)

        self.filepaths = filepaths

        self.files_mapping: Dict[str, Union[Path, None]] = {
            DESCRIPTION: None,
            TRAIN: None,
            TEST: None,
            OUTPUT: None,
            STATIC_FEATURES: None,
        }

        if files_mapping is not None:
            self.files_mapping.update(files_mapping)

        # TODO: each data split can have multiple files
        self.dataset_mapping: Dict[str, Union[Path, pd.DataFrame, None]] = {
            TRAIN: None,
            TEST: None,
            OUTPUT: None,
            STATIC_FEATURES: None,
        }
        for k, v in self.dataset_mapping.items():
            if v is None:
                self.dataset_mapping[k] = self.files_mapping[k]
            else:
                self.dataset_mapping[k] = v

    def __repr__(self) -> str:
        return f"TabularPredictionTask(name={self.metadata['name']}, description={self.metadata['description'][:100]}, {len(self.dataset_mapping)} datasets)"

    @classmethod
    def from_path(
        cls, task_root_dir: Path, description: Optional[str] = None
    ) -> "PredictionTask":
        # Get all filenames under task_root_dir
        task_data_filenames = []
        for entry in task_root_dir.iterdir():
            if entry.is_file():
                # Get just the filename (no subdir paths)
                task_data_filenames.append(entry.name)

        return cls(
            filepaths=[task_root_dir / fn for fn in task_data_filenames],
            metadata=dict(
                name=task_root_dir.name,
                description=description,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the PredictionTask to a dictionary for serialization.

        Returns:
            Dictionary representation of the PredictionTask with serializable values.
        """
        # Convert Path objects to strings
        files_mapping_serializable = {}
        for k, v in self.files_mapping.items():
            if isinstance(v, Path) or v is None:
                files_mapping_serializable[k] = str(v)
            else:
                files_mapping_serializable[k] = v

        return {
            "filepaths": [str(fp) for fp in self.filepaths],
            "metadata": self.metadata,
            "files_mapping": files_mapping_serializable,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionTask":
        """Create a PredictionTask from a dictionary.

        Args:
            data: Dictionary containing the PredictionTask data.

        Returns:
            A new PredictionTask instance.
        """
        # Convert string paths back to Path objects
        filepaths = [Path(fp) for fp in data.get("filepaths", [])]

        # Convert string paths in files_mapping back to Path objects
        files_mapping: Dict[str, Union[Path, None]] = {}
        for k, v in data["files_mapping"].items():
            if v is not None and isinstance(v, str):
                files_mapping[k] = Path(v)
            else:
                files_mapping[k] = None

        return cls(
            filepaths=filepaths,
            metadata=data["metadata"],
            name=data["metadata"]["name"],
            description=data["metadata"]["description"],
            files_mapping=files_mapping,
        )

    @staticmethod
    def save_artifacts(full_save_path, predictor):
        predictor.save_artifacts(full_save_path)

    def save_to_yaml(
        self,
        yaml_path: Path,
    ) -> None:
        """Save the task to a YAML file, with optional support for saving DataFrames.

        Args:
            yaml_path: Path where to save the YAML file
        """
        import yaml

        task_dict = self.to_dict()
        with open(yaml_path, "w") as f:
            yaml.dump(task_dict, f)

    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "PredictionTask":
        """Load a task from a YAML file.

        Args:
            yaml_path: Path to the YAML file
        Returns:
            A PredictionTask object
        """
        import yaml

        with open(yaml_path, "r") as f:
            task_dict = yaml.safe_load(f)

        task = cls.from_dict(task_dict)

        return task

    def load_task_data(self, dataset_key: str) -> pd.DataFrame | None:
        """Load the competition file for the task."""
        if dataset_key not in self.dataset_mapping:
            raise ValueError(
                f"Dataset type {dataset_key} not found for task {self.name}"
            )

        dataset = self.dataset_mapping[dataset_key]
        if dataset is None:
            return None
        if isinstance(dataset, pd.DataFrame):
            return load_pd(dataset)
        elif isinstance(dataset, pd.DataFrame):
            return dataset
        else:
            if isinstance(dataset, Path):
                filepath = dataset
            else:
                filepath = Path(dataset)

            if filepath.suffix == ".json":
                raise TypeError(f"File {filepath.name} has unsupported type: json")
            else:
                return load_pd(filepath)

    def _set_task_files(
        self,
        dataset_name_mapping: Dict[str, Union[str, Path, pd.DataFrame]],
    ):
        """Set the task files for the task."""
        for k, v in dataset_name_mapping.items():
            if v is None:
                self.dataset_mapping[k] = None
            elif isinstance(v, pd.DataFrame):
                self.dataset_mapping[k] = v
            elif isinstance(v, Path):
                self.dataset_mapping[k] = v
            elif isinstance(v, str):
                filepath = next(
                    iter([path for path in self.filepaths if path.name == v]),
                    self.filepaths[0].parent / v,
                )
                if not filepath.is_file():
                    raise ValueError(
                        f"File {v} not found in task {self.metadata['name']}"
                    )
                else:
                    self.dataset_mapping[k] = filepath
            else:
                raise TypeError(f"Unsupported type for dataset_mapping: {type(v)}")

    @property
    def name(self) -> str:
        return self.metadata["name"]

    @property
    def description(self) -> str:
        return self.metadata["description"]

    @description.setter
    def description(self, description: str) -> None:
        self.metadata["description"] = description

    @property
    def task_type(self) -> Optional[str]:
        return self.metadata["task_type"] or self._find_task_type_in_description()

    @task_type.setter
    def task_type(self, task_type: str) -> None:
        self.metadata["task_type"] = task_type

    @property
    def train_data(self) -> pd.DataFrame | None:
        return self.load_task_data(TRAIN)

    @train_data.setter
    def train_data(self, data: Union[str, Path, pd.DataFrame]) -> None:
        if isinstance(data, (str, Path)):
            self.files_mapping[TRAIN] = Path(data)
        self._set_task_files({TRAIN: data})

    @property
    def test_data(self) -> pd.DataFrame | None:
        return self.load_task_data(TEST)

    @test_data.setter
    def test_data(self, data: Union[str, Path, pd.DataFrame]) -> None:
        if isinstance(data, (str, Path)):
            self.files_mapping[TEST] = Path(data)
        self._set_task_files({TEST: data})

    @property
    def sample_submission_data(self) -> pd.DataFrame | None:
        return self.load_task_data(OUTPUT)

    @sample_submission_data.setter
    def sample_submission_data(self, data: Union[str, Path, pd.DataFrame]) -> None:
        if isinstance(data, (str, Path)):
            self.files_mapping[OUTPUT] = Path(data)
        if self.sample_submission_data is not None:
            raise ValueError("Output data already set for task")
        self._set_task_files({OUTPUT: data})

    @property
    def static_features_data(self) -> pd.DataFrame | None:
        return self.load_task_data(STATIC_FEATURES)

    @static_features_data.setter
    def static_features_data(self, data: Union[str, Path, pd.DataFrame]) -> None:
        if isinstance(data, (str, Path)):
            self.files_mapping[STATIC_FEATURES] = Path(data)
        self._set_task_files({STATIC_FEATURES: data})

    @property
    def data_description_file(self) -> Optional[str]:
        return self.metadata["data_description_file"]

    @data_description_file.setter
    def data_description_file(self, data: str) -> None:
        if isinstance(data, (str, Path)):
            self.files_mapping[DESCRIPTION] = Path(data)
        self.metadata["data_description_file"] = data

    @property
    def forecast_horizon(self) -> int:
        horizon = None
        data = self.metadata["forecast_horizon"]
        if isinstance(data, str):
            horizon = _safe_int_conversion(data)
        elif isinstance(data, int):
            horizon = data
        if horizon is None:
            horizon = DEFAULT_FORECAST_HORIZON
        return horizon

    @forecast_horizon.setter
    def forecast_horizon(self, data: int | str):
        self.metadata["forecast_horizon"] = data

    @property
    def output_columns(self) -> List[str] | None:
        if self.sample_submission_data is None:
            if self.label_column:
                return [self.label_column]
            else:
                return None
        else:
            return self.sample_submission_data.columns.to_list()

    @property
    def label_column(self) -> Optional[str]:
        """Return the label column for the task."""
        if "label_column" in self.metadata and self.metadata["label_column"]:
            return self.metadata["label_column"]
        else:
            # should ideally never be called after LabelColumnInference has run
            return self._infer_label_column_from_sample_submission_data()

    @label_column.setter
    def label_column(self, label_column: str) -> None:
        self.metadata["label_column"] = label_column

    @property
    def timestamp_column(self) -> Optional[str]:
        """Return the timestamp column for the task."""
        if "timestamp_column" in self.metadata and self.metadata["timestamp_column"]:
            return self.metadata["timestamp_column"]
        else:
            # should ideally never be called after TimestampColumnInference has run
            return self._find_timestamp_column_in_train()

    @timestamp_column.setter
    def timestamp_column(self, data: str) -> None:
        self.metadata["timestamp_column"] = data

    @property
    def columns_in_train_but_not_test(self) -> List[str]:
        assert self.train_data is not None, "Train data is not set yet"
        assert self.test_data is not None, "Test data is not set yet"
        return list(set(self.train_data.columns) - set(self.test_data.columns))

    @property
    def data_description(self) -> str:
        return self.metadata.get("data_description", self.description)

    @property
    def evaluation_description(self) -> str:
        return self.metadata.get("evaluation_description", self.description)

    @property
    def test_id_column(self) -> Optional[str]:
        return self.metadata.get("test_id_column", None)

    @test_id_column.setter
    def test_id_column(self, test_id_column: str) -> None:
        self.metadata["test_id_column"] = test_id_column

    @property
    def train_id_column(self) -> Optional[str]:
        return self.metadata.get("train_id_column", None)

    @train_id_column.setter
    def train_id_column(self, train_id_column: str) -> None:
        self.metadata["train_id_column"] = train_id_column

    @property
    def output_id_column(self) -> Optional[str]:
        return self.metadata.get(
            "output_id_column",
            self.sample_submission_data.columns[0]
            if self.sample_submission_data is not None
            else None,
        )

    @output_id_column.setter
    def output_id_column(self, output_id_column: str) -> None:
        self.metadata["output_id_column"] = output_id_column

    @property
    def problem_type(self) -> Optional[str]:
        return self.metadata["problem_type"] or self._find_problem_type_in_description()

    @problem_type.setter
    def problem_type(self, problem_type: str) -> None:
        self.metadata["problem_type"] = problem_type

    @property
    def eval_metric(self) -> Optional[str]:
        return self.metadata["eval_metric"] or (
            PREFERED_METRIC_BY_PROBLEM_TYPE[self.problem_type]
            if self.problem_type
            else None
        )

    @eval_metric.setter
    def eval_metric(self, eval_metric: str) -> None:
        self.metadata["eval_metric"] = eval_metric

    @property
    def images_column(self) -> Optional[str]:
        return self.metadata["images_column"] or (
            self._find_path_column_in_train() if self.train_data is not None else None
        )

    @property
    def text_columns(self) -> List[str]:
        return self._find_text_columns_in_train()

    def _infer_label_column_from_sample_submission_data(self) -> Optional[str]:
        if self.output_columns is None:
            return None

        # Assume the first output column is the ID column and ignore it
        relevant_output_cols = self.output_columns[1:]

        # Check if any of the output columns exists in the train data
        assert self.train_data is not None, "Train data is not set yet"
        existing_output_cols = [
            col for col in relevant_output_cols if col in self.train_data.columns
        ]

        # Case 1: If there's only one output column in the train data, use it
        if len(existing_output_cols) == 1:
            return existing_output_cols[0]

        # Case 2: For example in some multiclass problems, look for a column
        #         whose unique values match or contain the output columns
        output_set = set(col.lower() for col in relevant_output_cols)
        for col in self.train_data.columns:
            unique_values = set(
                str(val).lower()
                for val in self.train_data[col].unique()
                if pd.notna(val)
            )
            if output_set == unique_values or output_set.issubset(unique_values):
                return col

        # If no suitable column is found, raise an exception
        raise ValueError(
            "Unable to infer the label column. Please specify it manually."
        )

    def _find_text_columns_in_train(self) -> List[str]:
        if self.train_data is not None:
            return [
                col
                for col in self.train_data.columns
                if _column_contains_text(self.train_data[col])
                and col != self.images_column
            ]
        return []

    def _find_timestamp_column_in_train(self) -> Optional[str]:
        """Find column that contain timestamp"""
        if self.train_data is not None:
            datetime_cols = [
                col
                for col in self.train_data.columns
                if self.train_data[col].dtype.kind == "M"
            ]
            if len(datetime_cols) > 0:
                return datetime_cols[0]
        return None

    def _find_path_column_in_train(self) -> Optional[str]:
        """Find column that contain paths"""
        assert self.train_data is not None, "Train data is not set yet"
        path_columns = []
        for col in self.train_data.columns:
            try:
                path = Path(str(self.train_data[col][0]))
                if path.exists():
                    path_columns.append(col)
            except Exception:
                continue
        if len(path_columns) > 0:
            return path_columns[0]
        return None

    def _find_task_type_in_description(self) -> Optional[str]:
        desc = self.description.lower()
        # Check in priority order
        if any(kw in desc for kw in ["tabular"]):
            return TABULAR
        if any(kw in desc for kw in ["multimodal", "image"]):
            return MULTIMODAL
        if any(kw in desc for kw in ["timeseries", "time series"]):
            return TIME_SERIES
        return None

    def _find_problem_type_in_description(self) -> Optional[str]:
        if "regression" in self.description.lower():
            return REGRESSION
        elif "classification" in self.description.lower():
            return BINARY
        else:
            return None


def _safe_int_conversion(string_value: str):
    try:
        return int(string_value)
    except ValueError:
        logger.warning(f"Could not covert '{string_value}' to an integer")
        return None


def _is_float_compatible(column: pd.Series) -> bool:
    """
    :param column: pandas series with data
    :return: True if column contains only float or nan values
    """
    nans_number = column.isna().sum()
    converted_column = pd.to_numeric(column, errors="coerce")
    result_nans_number = converted_column.isna().sum()
    failed_objects_number = result_nans_number - nans_number
    non_nan_all_objects_number = len(column) - nans_number
    failed_ratio = failed_objects_number / non_nan_all_objects_number
    return failed_ratio < 0.5


def _column_contains_text(column: pd.Series) -> bool:
    """
    Column contains text if:
    1. it's not float or float compatible
    (e.g. ['1.2', '2.3', '3.4', ...] is float too)
    2. fraction of unique values (except nans) is more than 0.95

    :param column: pandas series with data
    :return: True if column contains text
    """
    FRACTION_OF_UNIQUE_VALUES = 0.95
    if column.dtype == object and not _is_float_compatible(column):
        unique_num = len(column.unique())
        nan_num = pd.isna(column).sum()
        return (
            unique_num / len(column) > FRACTION_OF_UNIQUE_VALUES
            if nan_num == 0
            else (unique_num - 1) / (len(column) - nan_num) > FRACTION_OF_UNIQUE_VALUES
        )
    return False
