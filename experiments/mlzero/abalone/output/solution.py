### UNMODIFIABLE IMPORT BEGIN ###
import random
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import (
    Task,
    TaskTypesEnum,
)  # classification, regression, ts_forecasting.
def train_model(train_features: np.ndarray | pd.DataFrame, train_target: np.ndarray | pd.DataFrame | pd.Series):
    if isinstance(train_features, pd.DataFrame) and isinstance(train_target, (pd.DataFrame, pd.Series)):
        input_data = InputData.from_dataframe(train_features, train_target, task=Task(TaskTypesEnum.regression))
    elif isinstance(train_features, np.ndarray) and isinstance(train_target, np.ndarray):
        input_data = InputData.from_numpy(train_features, train_target, task=Task(TaskTypesEnum.regression))
    else:
        raise ValueError("Unsupported data types for train_features and train_target. "
                         "Expected pandas DataFrame and (DataFrame or Series), or numpy ndarray and numpy ndarray."
                         f"Got: {type(train_features)} and {type(train_target)}")
        
    model = Fedot(problem=TaskTypesEnum.regression.value,
            timeout=10,
            seed=42,
            cv_folds=5,
            preset='auto',
            metric='rmse',
            n_jobs=1,
            with_tuning=True,
            show_progress=True)

    try:
        model.fit(features=input_data) # this is the training step, after this step variable 'model' will be a trained model
    except Exception as e:
        raise RuntimeError(
            f"Model training failed. Please check your data preprocessing carefully. "
            f"Common issues include: missing values, incorrect data types, feature scaling problems, "
            f"or incompatible target variable format. Original error: {str(e)}"
        ) from e

    # Save the pipeline
    pipeline = model.current_pipeline
    pipeline.save(path=PIPELINE_PATH, create_subdir=False, is_datetime_in_path=False)

    return model
def evaluate_model(model: Fedot, test_features: np.ndarray | pd.DataFrame | pd.Series, test_target: np.ndarray | pd.DataFrame | pd.Series):
    if isinstance(test_features, pd.DataFrame) and isinstance(test_target, (pd.DataFrame, pd.Series)):
        input_data = InputData.from_dataframe(test_features, test_target, task=Task(TaskTypesEnum.regression))
    elif isinstance(test_features, np.ndarray) and isinstance(test_target, np.ndarray):
        input_data = InputData.from_numpy(test_features, test_target, task=Task(TaskTypesEnum.regression))
    else:
        raise ValueError("Unsupported data types for test_features and test_target. "
                         "Expected pandas DataFrame and (DataFrame or Series), or numpy ndarray and numpy ndarray."
                         f"Got: {type(test_features)} and {type(test_target)}")
    y_pred = model.predict(features=input_data)
    print("Model metrics: ", model.get_metrics())
    return model.get_metrics()
def automl_predict(model: Fedot, features: np.ndarray | pd.DataFrame | pd.Series) -> np.ndarray:
    if isinstance(features, (pd.DataFrame, pd.Series)):
        features = features.to_numpy()
    input_data = InputData.from_numpy(features, None, task=Task(TaskTypesEnum.regression))
    predictions = model.predict(features=input_data)
    print(f"Predictions shape: {predictions.shape}")
    return predictions

### UNMODIFIABLE IMPORT END ###
# USER CODE BEGIN IMPORTS #
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# USER CODE END IMPORTS #

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

### UNMODIFIABLE CODE BEGIN ###
DATASET_PATH = Path("/home/stas/Documents/GitHub/FEDOT.LLM/examples/mlzero/abalone/competition")  # path for saving and loading dataset(s)
WORKSPACE_PATH = Path("/home/stas/Documents/GitHub/FEDOT.LLM/examples/mlzero/abalone/output")
PIPELINE_PATH = WORKSPACE_PATH / "pipeline"  # path for saving and loading pipelines
SUBMISSION_PATH = WORKSPACE_PATH / "submission.csv"  # path for saving submission file
EVAL_SET_SIZE = 0.2  # 20% of the data for evaluation
### UNMODIFIABLE CODE END ###

# --- TODO: Update these paths for your specific competition ---
TRAIN_FILE = DATASET_PATH / "train.csv"  # TODO: Replace with your actual filename
TEST_FILE = DATASET_PATH / "test.csv"  # TODO: Replace with your actual filename
SAMPLE_SUBMISSION_FILE = DATASET_PATH / "sample_submission.csv"  # TODO: Replace with your actual filename or None


# USER CODE BEGIN LOAD_DATA #
def load_data():
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    return train, test


# USER CODE END LOAD_DATA #


def transform_data(dataset: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to transform data into a format that can be used for training the model.
    Used on both Train and Test data. Test data may initially not contain target columns.
    """

    target_columns = ['rings']  # Target column for regression task

    # Separating features and target if present
    data = dataset.copy(deep=True)
    has_target = any(col in data.columns for col in target_columns)
    if has_target:
        features = data.drop(columns=target_columns)
        target = data[target_columns].values
    else:
        features = data
        target = None

    # Imputing missing values - 'mean' strategy for numeric columns, 'most_frequent' otherwise
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    categorical_cols = features.select_dtypes(exclude=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_imputer = SimpleImputer(strategy="mean")
        features[numeric_cols] = numeric_imputer.fit_transform(features[numeric_cols])
    if len(categorical_cols) > 0:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        features[categorical_cols] = categorical_imputer.fit_transform(
            features[categorical_cols]
        )

    # Encoding categorical features
    if categorical_cols.any():
        encoder = OneHotEncoder()
        features = pd.get_dummies(features, columns=categorical_cols, drop_first=True)

    return features.values, target


# The main function to orchestrate the data loading, feature engineering, model training and model evaluation
def create_model():
    """
    Function to execute the ML pipeline.
    """
    # USER CODE BEGIN CREATE MODEL #
    # Step 1. Retrieve or load a dataset from user's local storage
    train, X_test = load_data()

    # Step 2. Create a train-test split of the data
    train_data, eval_test_data = train_test_split(
        train, test_size=EVAL_SET_SIZE, random_state=SEED
    )  # corresponding to 80%, 20% of dataset

    train_features, train_target = transform_data(train_data)
    eval_test_features, eval_test_target = transform_data(eval_test_data)
    test_features, _ = transform_data(X_test)

    # Step 3. Train AutoML model
    model = train_model(train_features, train_target)

    # Step 4. Evaluate the trained model
    model_performance = evaluate_model(model, eval_test_features, eval_test_target)

    # Step 5. Evaluate predictions for the test dataset using AutoML Framework
    predictions: np.ndarray = automl_predict(model, test_features)  # returns 2D array
    output = pd.DataFrame(predictions, columns=['rings'])  # Naming the column as 'rings'
    output['id'] = X_test['id'].astype(int).values  # Adding the ID column

    output = output[['id', 'rings']]  # Reordering columns to match submission format
    output.to_csv(SUBMISSION_PATH, index=False)
    
    # USER CODE END CREATE MODEL #
    return model_performance


### UNMODIFIABLE CODE BEGIN ###
def main():
    """
    Main function to execute the ML pipeline.
    """
    print("Files and directories:")
    paths = {
        "Dataset Path": DATASET_PATH,
        "Workspace Path": WORKSPACE_PATH,
        "Pipeline Path": PIPELINE_PATH,
        "Submission Path": SUBMISSION_PATH,
        "Train File": TRAIN_FILE,
        "Test File": TEST_FILE,
        "Sample Submission File": SAMPLE_SUBMISSION_FILE,
    }
    for name, path in paths.items():
        print(f"{name}: {path}")

    model_performance = create_model()
    print("Model Performance on Test Set:", model_performance)


if __name__ == "__main__":
    main()
### UNMODIFIABLE CODE END ###