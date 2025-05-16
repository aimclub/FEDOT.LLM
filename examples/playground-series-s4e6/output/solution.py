### UNMODIFIABLE IMPORT BEGIN ###
import random
from pathlib import Path
import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task
from fedot.core.repository.tasks import TaskTypesEnum  # classification, regression, ts_forecasting.
def train_model(train_features: pd.DataFrame | pd.Series, train_target: pd.DataFrame | pd.Series):
    input_data = InputData.from_dataframe(train_features, train_target, task='classification')
    model = Fedot(problem=TaskTypesEnum.classification.value,
            timeout=60.0,
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
def evaluate_model(model, test_features: pd.DataFrame | pd.Series, test_target: pd.DataFrame | pd.Series):
    input_data = InputData.from_dataframe(test_features, test_target, task=Task(TaskTypesEnum.classification))
    y_pred = model.predict(features=input_data)
    print("Model metrics: ", model.get_metrics())
    return model.get_metrics()
def automl_predict(model, features: pd.DataFrame | pd.Series) -> np.ndarray:
    input_data = InputData.from_numpy(features.to_numpy(), None, task=Task(TaskTypesEnum.classification))
    predictions = model.predict(features=input_data)
    print(f"Predictions shape: {predictions.shape}")
    return predictions
### UNMODIFIABLE IMPORT END ###
# USER CODE BEGIN IMPORTS #
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.compose import ColumnTransformer
# USER CODE END IMPORTS #

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

### UNMODIFIABLE CODE BEGIN ###
DATASET_PATH = Path("/Users/aleksejlapin/Work/STABLE-FedotLLM/competition")  # path for saving and loading dataset(s)
WORKSPACE_PATH = Path("/Users/aleksejlapin/Work/STABLE-FedotLLM/examples/playground-series-s4e6/output")
PIPELINE_PATH = WORKSPACE_PATH / "pipeline"  # path for saving and loading pipelines
EVAL_SET_SIZE = 0.2  # 20% of the data for evaluation
### UNMODIFIABLE CODE END ###

TRAIN_FILE = "train.csv"  # Replace with your actual filename
TEST_FILE = "test.csv"  # Replace with your actual filename
SAMPLE_SUBMISSION_FILE = "sample_submission.csv"  # Replace with your actual filename
ID_COLUMN = "id"  # Replace with your actual ID column name, if any
TARGET_COLUMN = "Target"  # Replace with your actual target column name


def load_data(data_dir: Path, train_file: str, test_file: str, sample_submission_file: str = None):
    """Loads train, test, and optionally sample submission files."""
    try:
        train_df = pd.read_csv(data_dir / train_file)
        test_df = pd.read_csv(data_dir / test_file)
        sample_sub_df = None
        if sample_submission_file and (data_dir / sample_submission_file).exists():
            sample_sub_df = pd.read_csv(data_dir / sample_submission_file)
        else:
            print("Sample submission file not found or not specified.")
        print("Data loaded successfully.")
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df, sample_sub_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please check filenames.")
        return None, None, None


def cleaning_data(df_train: pd.DataFrame, df_test: pd.DataFrame, target_col: str):
    """Cleans the data by handling target and ID columns."""
    print("\nCleaning data...")
    # Make copies to avoid modifying original dataframes
    X_train = df_train.copy()
    X_test = df_test.copy()

    # Separate target if present
    if target_col in X_train.columns:
        y_train = X_train.pop(target_col).copy()
    else:
        y_train = None
        print(f"Warning: Target column '{target_col}' not in training data.")

    # Remove target from test if present
    if target_col in X_test.columns:
        print(f"Warning: Target '{target_col}' found in test data and removed.")
        X_test.drop(columns=[target_col], inplace=True)

    # Remove ID column from feature lists if it's present and not a feature
    if ID_COLUMN in X_train: 
        X_train.drop(columns=[ID_COLUMN], inplace=True)
    if ID_COLUMN in X_test: 
        X_test.drop(columns=[ID_COLUMN], inplace=True)

    return X_train, X_test, y_train


def preprocess_data(train_features: pd.DataFrame, test_features: pd.DataFrame):
    """Preprocesses the data using imputation, encoding, and scaling."""
    print("\nPreprocessing data...")
    X_train = train_features.copy()
    X_test = test_features.copy()

    # Identify feature types
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    # Preprocessing Steps
    numerical_transformer = SklearnPipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = SklearnPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Fit on train, transform train and test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert processed arrays back into DataFrame
    processed_feature_names = preprocessor.get_feature_names_out()
    X_train_processed = pd.DataFrame(X_train_processed, columns=processed_feature_names)
    X_test_processed = pd.DataFrame(X_test_processed, columns=processed_feature_names)

    print("Preprocessing complete.")
    print(f"Processed Train shape: {X_train_processed.shape}, Processed Test shape: {X_test_processed.shape}")
    return X_train_processed, X_test_processed


def create_submission_file(test_ids, test_predictions, target_column_name, sample_sub_df, submission_filename="submission.csv"):
    """Creates the submission file in the required format."""
    print(f"\nCreating submission file: {submission_filename}")
    # Flatten predictions if they are multi-dimensional
    if isinstance(test_predictions, np.ndarray) and test_predictions.ndim > 1:
        test_predictions = np.argmax(test_predictions, axis=1)

    # Retrieve IDs from sample submission or fallback
    if test_ids is None:
        if sample_sub_df is not None and ID_COLUMN in sample_sub_df.columns:
            test_ids = sample_sub_df[ID_COLUMN]
        else:
            test_ids = np.arange(len(test_predictions))

    submission_df = pd.DataFrame({ID_COLUMN: test_ids, target_column_name: test_predictions})
    submission_file_path = WORKSPACE_PATH / submission_filename
    submission_df.to_csv(submission_file_path, index=False)

    print(f"Submission file created at: {submission_file_path}")
    print("Submission file head:")
    print(submission_df.head())
    return submission_df


def main():
    """Main function to orchestrate the ML pipeline."""
    print("Starting the ML Workflow...")
    current_problem_type = 'classification'  # Example

    # Load Data
    train_df, test_df, sample_sub_df = load_data(DATASET_PATH, TRAIN_FILE, TEST_FILE, SAMPLE_SUBMISSION_FILE)
    if train_df is None or test_df is None:
        print("Exiting due to data loading failure.")
        return
    original_test_ids = test_df[ID_COLUMN] if ID_COLUMN in test_df.columns else None

    # Cleaning
    X_train_cleaned, X_test_cleaned, y_train_full = cleaning_data(train_df, test_df, TARGET_COLUMN)

    # Preprocessing
    X_train_processed, X_test_processed = preprocess_data(X_train_cleaned, X_test_cleaned)

    # Train-validation split
    print(f"\nCreating train-evaluation split (Eval size: {EVAL_SET_SIZE * 100}%) ...")
    stratify_target = y_train_full if current_problem_type.startswith('classification') else None
    X_train_main, X_eval_holdout, y_train_main, y_eval_holdout = train_test_split(
        X_train_processed, y_train_full, test_size=EVAL_SET_SIZE, random_state=SEED, stratify=stratify_target
    )

    # Train AutoML model
    model = train_model(X_train_main, y_train_main)

    # Evaluate model
    evaluate_model(model, X_eval_holdout, y_eval_holdout)

    # Generate predictions for the test dataset
    predictions = automl_predict(model, X_test_processed)

    # Create and save submission file
    if predictions is not None:
        target_col_in_submission = TARGET_COLUMN if sample_sub_df is None else sample_sub_df.columns[1]
        create_submission_file(original_test_ids, predictions, target_col_in_submission, sample_sub_df, "submission.csv")
    else:
        print("No predictions generated; skipping submission file creation.")
    print("\nML Workflow completed.")


if __name__ == "__main__":
    main()