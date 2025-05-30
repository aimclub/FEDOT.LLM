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
            cv_folds=None,
            preset='auto',
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
DATASET_PATH = Path("/Users/aleksejlapin/Work/STABLE-FedotLLM/examples/titanic/competition")  # path for saving and loading dataset(s)
WORKSPACE_PATH = Path("/Users/aleksejlapin/Work/STABLE-FedotLLM/examples/titanic/output")
PIPELINE_PATH = WORKSPACE_PATH / "pipeline"  # path for saving and loading pipelines
SUBMISSION_PATH = WORKSPACE_PATH / "submission.csv"  # path for saving submission file
EVAL_SET_SIZE = 0.2  # 20% of the data for evaluation
### UNMODIFIABLE CODE END ###

# USER CODE BEGIN LOAD_DATA #
def load_data():
    train = pd.read_csv(DATASET_PATH / "train.csv")
    X_test = pd.read_csv(DATASET_PATH / "test.csv")
    return train, X_test
# USER CODE END LOAD_DATA #

def transform_data(dataset: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to transform data into a format that can be used for training the model.
    Used on both Train and Test data. Test data may initially not contain target columns.
    """

    # TODO: Specify target columns 
    target_columns = ['Survived']

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
        numeric_imputer = SimpleImputer(strategy='mean')
        features[numeric_cols] = numeric_imputer.fit_transform(features[numeric_cols])
    if len(categorical_cols) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        features[categorical_cols] = categorical_imputer.fit_transform(features[categorical_cols])

    # Drop all columns from features that are not important for predictions. All other dataset transformations are STRICTLY FORBIDDEN.
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    features = features.drop(columns=[col for col in columns_to_drop if col in features.columns])

    return features.values, target

# The main function to orchestrate the data loading, feature engineering, model training and model evaluation
def create_model():
    """
    Function to execute the ML pipeline.
    """
    # USER CODE BEGIN CREATE MODEL #
    train, X_test = load_data()
    
    # Create a train-test split of the data
    train_data, eval_test_data = train_test_split(train, test_size=EVAL_SET_SIZE, random_state=SEED, stratify=train['Survived'])

    train_features, train_target = transform_data(train_data)
    eval_test_features, eval_test_target = transform_data(eval_test_data)
    test_features, _ = transform_data(X_test)

    # Train AutoML model
    model = train_model(train_features, train_target)

    # Evaluate the trained model
    model_performance = evaluate_model(model, eval_test_features, eval_test_target)

    # Evaluate predictions for the test dataset using AutoML Framework
    predictions = automl_predict(model, test_features)
    
    # Create a DataFrame for output submission
    output = pd.DataFrame({
        'PassengerId': X_test['PassengerId'],
        'Survived': predictions.flatten().astype(int)
    })
    
    # USER CODE END CREATE MODEL #

    output.to_csv(SUBMISSION_PATH, index=False)
    return model_performance

### UNMODIFIABLE CODE BEGIN ###
def main():
    """ 
    Main function to execute the ML pipeline.
    """
    model_performance = create_model()
    print("Model Performance on Test Set:", model_performance)
        
if __name__ == "__main__":
    main()
### UNMODIFIABLE CODE END ###