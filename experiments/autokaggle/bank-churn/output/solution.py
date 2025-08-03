### UNMODIFIABLE IMPORT BEGIN ###
import random
from pathlib import Path
import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task
from fedot.core.repository.tasks import TaskTypesEnum # classification, regression, ts_forecasting.
def train_model(train_features: pd.DataFrame | pd.Series, train_target: pd.DataFrame | pd.Series):
    input_data = InputData.from_dataframe(train_features, train_target, task='classification')
    model = Fedot(problem=TaskTypesEnum.classification.value,
            timeout=10.0,
            seed=42,
            cv_folds=None,
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
def evaluate_model(model, test_features: pd.DataFrame | pd.Series, test_target: pd.DataFrame | pd.Series):
    input_data = InputData.from_dataframe(test_features, test_target, task=Task(TaskTypesEnum.classification))
    y_pred = model.predict_proba(features=input_data)
    print("Model metrics: ", model.get_metrics())
    return model.get_metrics()
def automl_predict(model, features: pd.DataFrame | pd.Series) -> np.ndarray:
    input_data = InputData.from_numpy(features.to_numpy(), None, task=Task(TaskTypesEnum.classification))
    predictions = model.predict_proba(features=input_data)
    print(f"Predictions shape: {predictions.shape}")
    return predictions
### UNMODIFIABLE IMPORT END ###
# USER CODE BEGIN IMPORTS #
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# USER CODE END IMPORTS #

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

### UNMODIFIABLE CODE BEGIN ###
DATASET_PATH = Path("/Users/aleksejlapin/Work/STABLE-FedotLLM/examples/bank-churn/competition") # path for saving and loading dataset(s)
WORKSPACE_PATH = Path("/Users/aleksejlapin/Work/STABLE-FedotLLM/examples/bank-churn/output")
PIPELINE_PATH = WORKSPACE_PATH / "pipeline" # path for saving and loading pipelines
EVAL_SET_SIZE = 0.2 # 20% of the data for evaluation
### UNMODIFIABLE CODE END ###

# --- TODO: Update these paths for your specific competition ---
TRAIN_FILE = "train.csv" # Replace with your actual filename
TEST_FILE = "test.csv" # Replace with your actual filename
SAMPLE_SUBMISSION_FILE = "sample_submission.csv" # Replace with your actual filename
ID_COLUMN = "id" # Replace with your actual ID column name, if any
TARGET_COLUMN = "Exited" # Updated to use the actual target column name


def load_data(data_dir: Path, train_file: str, test_file: str, sample_submission_file: str = None):
    """Loads train, test, and optionally sample submission files."""
    try:
        train_df = pd.read_csv(data_dir / train_file) # TODO: Adjust pandas loader if needed
        test_df = pd.read_csv(data_dir / test_file) # TODO: Adjust pandas loader if needed
        sample_sub_df = None
        if sample_submission_file and (data_dir / sample_submission_file).exists():
           sample_sub_df = pd.read_csv(data_dir / sample_submission_file) # TODO: Adjust pandas loader if needed
        else:
            print("Sample submission file not found or not specified.")
        print("Data loaded successfully.")
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df, sample_sub_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please check filenames.")
        return None, None, None
    
# --------------------------------------------------------------------------- #
# Section 2: Data Cleaning
# --------------------------------------------------------------------------- #

def cleaning_data(df_train: pd.DataFrame, df_test: pd.DataFrame, target_col: str):
    print("\nCleaning data...")
    # Make copies to avoid modifying original dataframes
    X_train = df_train.copy()
    X_test = df_test.copy()

    # --- Step 0: Handle specific data cleaning tasks (e.g., inconsistent values, typos) ---
    # This section remains for custom, early-stage cleaning if necessary.
    
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
    if ID_COLUMN in X_train: X_train.drop(columns=[ID_COLUMN], inplace=True)
    if ID_COLUMN in X_test: X_test.drop(columns=[ID_COLUMN], inplace=True)
    
    return X_train, X_test, y_train

# --------------------------------------------------------------------------- #
# Section 3: Data Preprocessing
# --------------------------------------------------------------------------- #
def preprocess_data(train_features: pd.DataFrame, test_features: pd.DataFrame):
    """Cleans and transforms data into a suitable format for modeling."""
    
    print("\nPreprocessing data...")
    # Make copies to avoid modifying original dataframes
    X_train = train_features.copy()
    X_test = test_features.copy()
    
    # --- Step 1: Identify feature types (based on training data) ---
    categorical_cols = ['Geography', 'Gender']  # Example categorical columns
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']  # Example numerical columns
    
    # --- Step 2: Handle Missing Values (Fit on Train, Transform Train & Test) ---
    # Numerical Imputation
    if numerical_cols:
        num_imputer = SimpleImputer(strategy='mean')
        X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

    # Categorical Imputation
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

    # --- Step 3: Encode Categorical Features (Fit on Train, Transform Train & Test) ---
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # Utilize OneHotEncoder
    X_train_encoded = ohe.fit_transform(X_train[categorical_cols])
    X_test_encoded = ohe.transform(X_test[categorical_cols])
    
    # Create dataframes from the encoded features
    train_encoded_df = pd.DataFrame(X_train_encoded, columns=ohe.get_feature_names_out(categorical_cols), index=X_train.index)
    test_encoded_df = pd.DataFrame(X_test_encoded, columns=ohe.get_feature_names_out(categorical_cols), index=X_test.index)
    
    # Drop original categorical columns and concatenate encoded columns
    X_train = pd.concat([X_train.drop(columns=categorical_cols), train_encoded_df], axis=1)
    X_test = pd.concat([X_test.drop(columns=categorical_cols), test_encoded_df], axis=1)

    # --- Step 4: Feature Scaling (Fit on Train, Transform Train & Test) ---
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    print("Preprocessing complete.")
    print(f"Train processed shape: {X_train.shape}, Test processed shape: {X_test.shape}")
    return X_train, X_test

# --------------------------------------------------------------------------- #
# Section 6: Final Model Training & Submission
# --------------------------------------------------------------------------- #
def create_submission_file(test_ids, test_predictions, target_column_name, sample_sub_df, submission_filename="submission.csv"):
    """Creates the submission file in the required format."""
    print(f"\nCreating submission file: {submission_filename}")
    if isinstance(test_predictions, np.ndarray) and test_predictions.ndim > 1:
        test_predictions = test_predictions.flatten()
    
    if not isinstance(test_ids, (pd.Series, np.ndarray)):
        try:
            if sample_sub_df is not None and ID_COLUMN in sample_sub_df.columns:
                test_ids = sample_sub_df[ID_COLUMN]
            else:
                test_ids = np.arange(len(test_predictions))
                print(f"Warning: Using default range for test_ids as {ID_COLUMN} not found in sample submission.")
        except Exception as e:
            print(f"Warning: Could not load sample submission to get test_ids: {e}. Using default range.")
            test_ids = np.arange(len(test_predictions))

    submission_df = pd.DataFrame({ID_COLUMN: test_ids, target_column_name: test_predictions})

    submission_file_path = DATASET_PATH / submission_filename
    submission_df.to_csv(submission_file_path, index=False)
    print(f"Submission file created at: {submission_file_path}")
    print("Submission file head:")
    print(submission_df.head())
    return submission_df

# --------------------------------------------------------------------------- #
# Main Orchestration Logic
# --------------------------------------------------------------------------- #
def main():
    """Main function to orchestrate the ML pipeline."""
    print("Starting Kaggle ML Workflow...")
    current_problem_type = 'classification' # Example

    train_df, test_df, sample_sub_df = load_data(DATASET_PATH, TRAIN_FILE, TEST_FILE, SAMPLE_SUBMISSION_FILE)
    if train_df is None or test_df is None:
        print("Exiting due to data loading failure.")
        return
    
    original_test_ids = test_df[ID_COLUMN] if ID_COLUMN in test_df.columns else test_df.index
    
    X_train_cleaned, X_test_cleaned, y_train_full = cleaning_data(train_df, test_df, TARGET_COLUMN)

    X_train_processed, X_test_processed  = preprocess_data(X_train_cleaned, X_test_cleaned)
    
    print(f"\nCreating a fixed train-evaluation split (Eval size: {EVAL_SET_SIZE*100}%) ...")
    stratify_target = y_train_full if current_problem_type.startswith('classification') else None
    X_train_main, X_eval_holdout, y_train_main, y_eval_holdout = train_test_split(
        X_train_processed, y_train_full,
        test_size=EVAL_SET_SIZE,
        random_state=SEED,
        stratify=stratify_target
    )
    
    model = train_model(X_train_main, y_train_main)
    
    evaluate_model(model, X_eval_holdout, y_eval_holdout)

    predictions:np.ndarray = automl_predict(model, X_test_processed) # returns 2D array
  
    if predictions is not None:
        target_col_in_submission = TARGET_COLUMN
        if sample_sub_df is not None and len(sample_sub_df.columns) == 2:
            target_col_in_submission = sample_sub_df.columns.difference([ID_COLUMN])[0]

        create_submission_file(original_test_ids, predictions,
                               target_col_in_submission, sample_sub_df, "submission.csv")
    else:
        print("No test predictions generated, skipping submission file creation.")
    print("\nKaggle ML Workflow finished.")

if __name__ == "__main__":
    main()