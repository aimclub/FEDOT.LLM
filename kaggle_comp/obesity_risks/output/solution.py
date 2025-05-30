### UNMODIFIABLE IMPORT BEGIN ###
import random
from pathlib import Path
import numpy as np
from typing import Tuple
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task
from fedot.core.repository.tasks import TaskTypesEnum  # classification, regression, ts_forecasting.
def train_model(train_features: np.ndarray, train_target: np.ndarray):
    input_data = InputData.from_numpy(train_features, train_target, task=Task(TaskTypesEnum.classification))
    model = Fedot(problem=TaskTypesEnum.classification.value,
            timeout=60,
            seed=42,
            cv_folds=5,
            preset='best_quality',
            metric='accuracy',
            n_jobs=1,
            with_tuning=True,
            show_progress=True)

    model.fit(features=input_data) # this is the training step, after this step variable ‘model‘ will be a trained model

    # Save the pipeline
    pipeline = model.current_pipeline
    pipeline.save(path=PIPELINE_PATH, create_subdir=False, is_datetime_in_path=False)

    return model
def evaluate_model(model, test_features: np.ndarray, test_target: np.ndarray):
    input_data = InputData.from_numpy(test_features, test_target, task=Task(TaskTypesEnum.classification))
    y_pred = model.predict(features=input_data)
    print("Model metrics: ", model.get_metrics())
    return model.get_metrics()
def automl_predict(model, features: np.ndarray) -> np.ndarray:
    input_data = InputData.from_numpy(features, None, task=Task(TaskTypesEnum.classification))
    predictions = model.predict(features=input_data)
    print(f"Predictions shape: {predictions.shape}")
    return predictions
### UNMODIFIABLE IMPORT END ###
# USER CODE BEGIN IMPORTS #
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
# USER CODE END IMPORTS #

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

### UNMODIFIABLE CODE BEGIN ###
DATASET_PATH = Path("/Users/aleksejlapin/Work/STABLE-FedotLLM/kaggle_comp/obesity_risks/competition")  # path for saving and loading dataset(s)
WORKSPACE_PATH = Path("/Users/aleksejlapin/Work/STABLE-FedotLLM/kaggle_comp/obesity_risks/output")
PIPELINE_PATH = WORKSPACE_PATH / "pipeline"  # path for saving and loading pipelines
SUBMISSION_PATH = WORKSPACE_PATH / "submission.csv"  # path for saving submission file
EVAL_SET_SIZE = 0.2  # 20% of the data for evaluation
### UNMODIFIABLE CODE END ###
# --- TODO: Update these paths for your specific competition ---
TRAIN_FILE = DATASET_PATH / "train.csv"  # Replace with your actual filename
TEST_FILE = DATASET_PATH / "test.csv"  # Replace with your actual filename
SAMPLE_SUBMISSION_FILE = DATASET_PATH / "sample_submission.csv"  # Replace with your actual filename or None

# USER CODE BEGIN LOAD_DATA #
def load_data():
    # Load training and test datasets
    train = pd.read_csv(TRAIN_FILE)
    X_test = pd.read_csv(TEST_FILE)
    return train, X_test
# USER CODE END LOAD_DATA #

def transform_data(dataset: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to transform data into a format that can be used for training the model.
    Used on both Train and Test data. Test data may initially not contain target columns.
    """

    # Specify target column
    target_column = 'NObeyesdad'

    # Separating features and target if present
    data = dataset.copy(deep=True)
    has_target = target_column in data.columns
    if has_target:
        features = data.drop(columns=[target_column, 'id'])
        target = data[target_column].values
    else:
        features = data.drop(columns=['id'])
        target = None

    # Imputing missing values - 'mean' strategy for numeric columns, 'most_frequent' otherwise
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    categorical_cols = features.select_dtypes(exclude=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_imputer = SimpleImputer(strategy='mean')
        features[numeric_cols] = numeric_imputer.fit_transform(features[numeric_cols])
    if len(categorical_cols) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        features[categorical_cols] = categorical_imputer.fit_transform(features[categorical_cols])

    # Converting categorical columns to numerical using one-hot encoding
    features = pd.get_dummies(features, columns=categorical_cols, drop_first=True)

    return features.values, target


def create_model():
    """
    Function to execute the ML pipeline.
    """
    # USER CODE BEGIN CREATE MODEL #
    # Step 1. Retrieve or load datasets from local storage
    train, X_test = load_data()

    # Step 2. Create a train-test split of the data
    train_data, eval_test_data = train_test_split(train, test_size=EVAL_SET_SIZE, random_state=SEED, stratify=train['NObeyesdad'])

    # Transform datasets
    train_features, train_target = transform_data(train_data)
    eval_test_features, eval_test_target = transform_data(eval_test_data)
    test_features, _ = transform_data(X_test)

    # Step 3. Map target labels to integer encoding for model training
    target_classes = sorted(train_data['NObeyesdad'].unique())
    class_to_label = {cls: idx for idx, cls in enumerate(target_classes)}
    label_to_class = {idx: cls for cls, idx in class_to_label.items()}
    
    # Encode training and evaluation targets
    train_target_encoded = np.array([class_to_label[target] for target in train_target])
    eval_test_target_encoded = np.array([class_to_label[target] for target in eval_test_target])

    # Step 4. Train AutoML model
    model = train_model(train_features, train_target_encoded)

    # Step 5. Evaluate the trained model
    model_performance = evaluate_model(model, eval_test_features, eval_test_target_encoded)

    # Step 6. Generate predictions for the test dataset using AutoML framework
    predictions: np.ndarray = automl_predict(model, test_features)  # returns 2D array
    predictions = predictions.flatten()
    predictions_decoded = [label_to_class[int(prediction)] for prediction in predictions]

    # Prepare submission file
    output = pd.DataFrame({
        'id': X_test['id'],
        'NObeyesdad': predictions_decoded
    })

    # Correcting discrepancies - Ensure exact format match with expected submission file
    expected_format = pd.read_csv(SAMPLE_SUBMISSION_FILE)
    if not output['NObeyesdad'].isin(expected_format['NObeyesdad'].unique()).all():
        raise ValueError("Prediction labels in submission do not match expected labels in sample submission.")

    # Save submission file
    output.to_csv(SUBMISSION_PATH, index=False)
    return model_performance
    # USER CODE END CREATE MODEL #


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
        "Sample Submission File": SAMPLE_SUBMISSION_FILE
    }
    for name, path in paths.items():
        print(f"{name}: {path}")

    model_performance = create_model()
    print("Model Performance on Test Set:", model_performance)


if __name__ == "__main__":
    main()
### UNMODIFIABLE CODE END ###