### UNMODIFIABLE IMPORT BEGIN ###
import random
from pathlib import Path
import numpy as np
from typing import Tuple
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task
from fedot.core.repository.tasks import TaskTypesEnum # classification, regression, ts_forecasting.
def train_model(train_features: np.ndarray, train_target: np.ndarray):
    input_data = InputData.from_numpy(train_features, train_target, task=Task(TaskTypesEnum.classification))
    model = Fedot(problem=TaskTypesEnum.classification.value,
            timeout=5.0,
            seed=42,
            cv_folds=5,
            preset='auto',
            metric='roc_auc',
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
    y_pred = model.predict_proba(features=input_data)
    print("Model metrics: ", model.get_metrics())
    return model.get_metrics()
def automl_predict(model, features: np.ndarray) -> np.ndarray:
    input_data = InputData.from_numpy(features, None, task=Task(TaskTypesEnum.classification))
    predictions = model.predict_proba(features=input_data)
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
DATASET_PATH = Path("/Users/aleksejlapin/Work/STABLE-FedotLLM/examples/playground-series-s4e7/competition") # path for saving and loading dataset(s)
PIPELINE_PATH = Path("/Users/aleksejlapin/Work/STABLE-FedotLLM/examples/playground-series-s4e7/output") / "pipeline" # path for saving and loading pipelines
### UNMODIFIABLE CODE END ###

# USER CODE BEGIN LOAD_DATA #
def load_data():
    train = pd.read_csv(DATASET_PATH / 'train.csv')
    X_test = pd.read_csv(DATASET_PATH / 'test.csv')
    return train, X_test
# USER CODE END LOAD_DATA #

def transform_data(dataset: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to transform data into a format that can be used for training the model.
    Used on both Train and Test data. Test data may initially not contain target columns.
    """

    # TODO: Specify target columns 
    target_columns = ['Response']

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

    # Dropping columns that do not contribute to predictions
    feature_columns = ['Age', 'Annual_Premium', 'Vintage', 'Gender', 
                       'Driving_License', 'Region_Code', 'Previously_Insured',
                       'Vehicle_Age', 'Vehicle_Damage', 'Policy_Sales_Channel']
    
    features = features[feature_columns] if all(col in features.columns for col in feature_columns) else features

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
    # Here, the train_data contains 80% of the ‘dataset‘ and the test_data contains 20% of the ‘dataset‘.
    train_data, eval_test_data = train_test_split(train, test_size=0.2, random_state=SEED) # corresponding to 80%, 20% of ‘dataset‘
    
    train_features, train_target = transform_data(train_data)
    eval_test_features, eval_test_target = transform_data(eval_test_data)
    test_features, _ = transform_data(X_test)
    
    
    # TODO: Step 3. Train AutoML model. AutoML performs feature engineering and model training.
    model = train_model(train_features, train_target)
    
    # TODO: Step 4. evaluate the trained model using the defined "evaluate_model" function model_performance, model_complexity = evaluate_model()
    model_performance = evaluate_model(model, eval_test_features, eval_test_target)

    # TODO: Step 5.  Evaluate predictions for the test dataset using AutoML Framework
    # **YOU MUST USE automl_predict()**
    predictions: np.ndarray = automl_predict(model, test_features) # returns 2D array
    output = pd.DataFrame(predictions, columns=['Response'])
    
    output['id'] = X_test['id'].values.astype(int)  # Add ID column
    
    # USER CODE END CREATE MODEL # 

    output.to_csv(Path("/Users/aleksejlapin/Work/STABLE-FedotLLM/examples/playground-series-s4e7/output") / 'submission.csv', index=False)
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