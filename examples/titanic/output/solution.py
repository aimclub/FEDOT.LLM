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
        input_data = InputData.from_dataframe(train_features, train_target, task=Task(TaskTypesEnum.classification))
    elif isinstance(train_features, np.ndarray) and isinstance(train_target, np.ndarray):
        input_data = InputData.from_numpy(train_features, train_target, task=Task(TaskTypesEnum.classification))
    else:
        raise ValueError("Unsupported data types for train_features and train_target. "
                         "Expected pandas DataFrame and (DataFrame or Series), or numpy ndarray and numpy ndarray."
                         f"Got: {type(train_features)} and {type(train_target)}")
        
    model = Fedot(problem=TaskTypesEnum.classification.value,
            timeout=10,
            seed=42,
            cv_folds=3,
            preset='best_quality',
            metric='accuracy',
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
        input_data = InputData.from_dataframe(test_features, test_target, task=Task(TaskTypesEnum.classification))
    elif isinstance(test_features, np.ndarray) and isinstance(test_target, np.ndarray):
        input_data = InputData.from_numpy(test_features, test_target, task=Task(TaskTypesEnum.classification))
    else:
        raise ValueError("Unsupported data types for test_features and test_target. "
                         "Expected pandas DataFrame and (DataFrame or Series), or numpy ndarray and numpy ndarray."
                         f"Got: {type(test_features)} and {type(test_target)}")
    y_pred = model.predict_proba(features=input_data)
    print("Model metrics: ", model.get_metrics())
    return model.get_metrics()
def automl_predict(model: Fedot, features: np.ndarray | pd.DataFrame | pd.Series) -> np.ndarray:
    if isinstance(features, (pd.DataFrame, pd.Series)):
        features = features.to_numpy()
    input_data = InputData.from_numpy(features, None, task=Task(TaskTypesEnum.classification))
    predictions = model.predict_proba(features=input_data)
    print(f"Predictions shape: {predictions.shape}")
    return predictions

### UNMODIFIABLE IMPORT END ###
# USER CODE BEGIN IMPORTS #
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
# USER CODE END IMPORTS #

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

### UNMODIFIABLE CODE BEGIN ###
DATASET_PATH = Path("/Users/aleksejlapin/Work/STABLE-FedotLLM/examples/titanic/competition")  # path for saving and loading dataset(s)
WORKSPACE_PATH = Path("/Users/aleksejlapin/Work/STABLE-FedotLLM/examples/titanic/output")
PIPELINE_PATH = WORKSPACE_PATH / "pipeline"  # path for saving and loading pipelines
SUBMISSION_PATH = WORKSPACE_PATH / "submission.csv"  # path for saving submission file
EVAL_SET_SIZE = 0.2  # 20% of the data for evaluation
### UNMODIFIABLE CODE END ###

# --- TODO: Update these paths for your specific competition ---
TRAIN_FILE = DATASET_PATH / "train.csv"  # TODO: Replace with your actual filename
TEST_FILE = DATASET_PATH / "test.csv"  # TODO: Replace with your actual filename
SAMPLE_SUBMISSION_FILE = DATASET_PATH / "gender_submission.csv"  # TODO: Replace with your actual filename or None


# USER CODE BEGIN LOAD_DATA #
def load_data():
    # TODO: this function is for loading a dataset from user’s local storage
    train = pd.read_csv(TRAIN_FILE)
    X_test = pd.read_csv(TEST_FILE)
    return train, X_test


# USER CODE END LOAD_DATA #


def transform_data(dataset: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to transform data into a format that can be used for training the model.
    Used on both Train and Test data. Test data may initially not contain target columns.
    """

    target_columns = ['Survived'] # TODO: Replace with ACTUAL target columns

    # Separating features and target if present
    data = dataset.copy(deep=True)
    has_target = any(col in data.columns for col in target_columns)
    if has_target:
        features = data.drop(columns=target_columns)
        target = data[target_columns].values.ravel()  # Ensure target is 1D
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

    # TODO: Drop all columns from features that are not important for prdictions. All other dataset transformations are STRICTLY FORBIDDEN.
    # TODO: Before any operations, make sure to check whether columns you operate on are present in data. Do not raise exceptions.
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    existing_columns_to_drop = [col for col in columns_to_drop if col in features.columns]
    features = features.drop(columns=existing_columns_to_drop)

    return features.values, target


# The main function to orchestrate the data loading, feature engineering, model training and model evaluation
def create_model():
    """
    Function to execute the ML pipeline.
    """
    # USER CODE BEGIN CREATE MODEL #
    # TODO: Step 1. Retrieve or load a dataset from hub (if available) or user’s local storage, start path from the DATASET_PATH
    train, X_test = load_data()

    # TODO: Step 2. Create a train-test split of the data by splitting the ‘dataset‘ into train_data and test_data.
    # Create a train-validation split
    # Note: EVAL_SET_SIZE is a constant defined above, corresponding to 20% of the data for evaluation
    # Note: You may need to use stratified sampling if the target is categorical
    train_data, eval_test_data = train_test_split(
        train, test_size=EVAL_SET_SIZE, random_state=SEED, stratify=train['Survived']
    )  # corresponding to 80%, 20% of ‘dataset‘

    train_features, train_target = transform_data(train_data)
    eval_test_features, eval_test_target = transform_data(eval_test_data)
    test_features, _ = transform_data(X_test)

    # TODO: Step 3. Train AutoML model. AutoML performs feature engineering and model training.
    model = train_model(train_features, train_target)

    # TODO: Step 4. evaluate the trained model using the defined "evaluate_model" function model_performance, model_complexity = evaluate_model()
    model_performance = evaluate_model(model, eval_test_features, eval_test_target)

    # TODO: Step 5.  Evaluate predictions for the test datase using AutoML Framework
    # **YOU MUST USE automl_predict()**
    # Prediction result will not have an ID column, only a column for target (or columns for multiple targets)
    # If output submission should have an ID column, add it to the prediction.
    # If ID column has numeric type, convert it to integer
    predictions: np.ndarray = automl_predict(model, test_features)  # returns 2D array
    test_passenger_ids = pd.read_csv(TEST_FILE)['PassengerId']
    output = pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': predictions.flatten().astype(int)})

    # USER CODE END CREATE MODEL #
    # If target submission format is not numeric, convert predictions to expected format
    # For example: convert probabilities to class labels, apply inverse transformations,
    # or map numeric predictions back to categorical labels if needed
    output.to_csv(SUBMISSION_PATH, index=False)
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