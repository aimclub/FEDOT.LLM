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
DATASET_PATH = Path("/Users/aleksejlapin/Work/STABLE-FedotLLM/examples/playground-series-s4e3/competition") # path for saving and loading dataset(s)
WORKSPACE_PATH = Path("/Users/aleksejlapin/Work/STABLE-FedotLLM/examples/playground-series-s4e3/output")
PIPELINE_PATH = WORKSPACE_PATH / "pipeline" # path for saving and loading pipelines
EVAL_SET_SIZE = 0.2 # 20% of the data for evaluation
### UNMODIFIABLE CODE END ###

# --- TODO: Update these paths for your specific competition ---
TRAIN_FILE = "train.csv" # Replace with your actual filename
TEST_FILE = "test.csv" # Replace with your actual filename
SAMPLE_SUBMISSION_FILE = "sample_submission.csv" # Replace with your actual filename
ID_COLUMN = "id" # Replace with your actual ID column name, if any
TARGET_COLUMNS = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"] # Replace with your actual target column names


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

def cleaning_data(df_train: pd.DataFrame, df_test: pd.DataFrame, target_cols: list):
    print("\nCleaning data...")
    # Make copies to avoid modifying original dataframes
    X_train = df_train.copy()
    X_test = df_test.copy()

    # --- Step 0: Handle specific data cleaning tasks (e.g., inconsistent values, typos) ---
    # This section remains for custom, early-stage cleaning if necessary.

    y_train = None
    # Check if all target columns are in X_train
    if all(col in X_train.columns for col in target_cols):
        y_train = X_train[target_cols].copy()
        X_train = X_train.drop(columns=target_cols)
    else:
        missing_cols = [col for col in target_cols if col not in X_train.columns]
        print(f"Warning: Target column(s) {missing_cols} not in training data.")
        # Optionally, remove existing target columns if some are missing
        present_target_cols = [col for col in target_cols if col in X_train.columns]
        if present_target_cols:
            y_train = X_train[present_target_cols].copy() # or handle as error
            X_train = X_train.drop(columns=present_target_cols)
            print(f"Using available target columns: {present_target_cols}")

    # Remove target columns from test if present
    test_cols_to_drop = [col for col in target_cols if col in X_test.columns]
    if test_cols_to_drop:
        print(f"Warning: Target column(s) {test_cols_to_drop} found in test data and removed.")
        X_test.drop(columns=test_cols_to_drop, inplace=True)

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

    # --- Step 0: Column Dropping ---
    # Define important features based on domain knowledge or data exploration
    important_features = list(X_train.columns)  # Keeping all for now, you can refine this
    X_train = X_train[important_features]
    X_test = X_test[important_features]

    # --- Step 1: Identify feature types (based on training data) ---
    categorical_cols = ['TypeOfSteel_A300', 'TypeOfSteel_A400']  
    # Validate that categorical columns are in X_train
    categorical_cols = [col for col in categorical_cols if col in X_train.columns]
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # --- Step 2: Handle Missing Values (Fit on Train, Transform Train & Test) ---
    # Numerical Imputation
    num_imputer = SimpleImputer(strategy='median')
    X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

    # Categorical Imputation
    cat_imputer = SimpleImputer(strategy='most_frequent')
    if categorical_cols:  # Only apply if there are categorical columns left
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

    # --- Step 3: Encode Categorical Features (Fit on Train, Transform Train & Test) ---
    if categorical_cols:  # Only execute encoding if there are categorical columns
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_train_encoded = ohe.fit_transform(X_train[categorical_cols])
        X_test_encoded = ohe.transform(X_test[categorical_cols])
        X_train = pd.concat([X_train.drop(categorical_cols, axis=1).reset_index(drop=True), 
                             pd.DataFrame(X_train_encoded)], axis=1)
        X_test = pd.concat([X_test.drop(categorical_cols, axis=1).reset_index(drop=True), 
                            pd.DataFrame(X_test_encoded)], axis=1)

    # --- Step 4: Feature Scaling (Fit on Train, Transform Train & Test) ---
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Log transformation for skewed numerical features
    for col in numerical_cols:
        if X_train[col].skew() > 1.0:
            print(f"Log transforming skewed column: {col}")
            X_train[col] = np.log1p(X_train[col] - X_train[col].min())
            X_test[col] = np.log1p(X_test[col] - X_test[col].min()) # Apply same shift as train

    print("Preprocessing complete.")
    print(f"Train processed shape: {X_train.shape}, Test processed shape: {X_test.shape}")
    return X_train, X_test

# --------------------------------------------------------------------------- #
# Section 6: Final Model Training & Submission
# --------------------------------------------------------------------------- #
def create_submission_file(test_ids, test_predictions, target_column_names, sample_sub_df, submission_filename="submission.csv"):
    """Creates the submission file in the required format."""
    print(f"\nCreating submission file: {submission_filename}")

    # Prepare the submission DataFrame
    submission_df = pd.DataFrame(test_predictions, columns=target_column_names)
    submission_df.insert(0, ID_COLUMN, test_ids) 

    submission_file_path = WORKSPACE_PATH / submission_filename
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
    current_problem_type = 'multi-label classification'  # Set to multi-label classification

    # Step 0: Load Data
    train_df, test_df, sample_sub_df = load_data(DATASET_PATH, TRAIN_FILE, TEST_FILE, SAMPLE_SUBMISSION_FILE)
    if train_df is None or test_df is None:
        print("Exiting due to data loading failure.")
        return

    # Store original test IDs for submission
    original_test_ids = test_df[ID_COLUMN] if ID_COLUMN in test_df.columns else test_df.index

    # Step 1: Cleaning
    X_train_cleaned, X_test_cleaned, y_train_full = cleaning_data(train_df, test_df, TARGET_COLUMNS)

    # Step 2: Preprocessing
    X_train_processed, X_test_processed  = preprocess_data(X_train_cleaned, X_test_cleaned)

    # Step 3: Create fixed validation split from processed training data for evaluation
    print(f"\nCreating a fixed train-evaluation split (Eval size: {EVAL_SET_SIZE*100}%) ...")
    X_train_main, X_eval_holdout, y_train_main, y_eval_holdout = train_test_split(
        X_train_processed, y_train_full,
        test_size=EVAL_SET_SIZE,
        random_state=SEED,
        stratify=y_train_full
    )

    # Step 4: Train AutoML model
    model = train_model(X_train_main, y_train_main)

    # Step 5: Evaluate the trained model
    evaluate_model(model, X_eval_holdout, y_eval_holdout)

    # Step 6: Predict on the test dataset using AutoML Framework
    predictions = automl_predict(model, X_test_processed)

    # Step 7: Create submission file
    if predictions is not None:
        target_cols_in_submission = TARGET_COLUMNS
        create_submission_file(original_test_ids, predictions, target_cols_in_submission, sample_sub_df, "submission.csv")
    else:
        print("No test predictions generated, skipping submission file creation.")

    print("\nKaggle ML Workflow finished.")

if __name__ == "__main__":
    main()