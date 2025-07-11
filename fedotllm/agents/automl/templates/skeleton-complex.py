### UNMODIFIABLE IMPORT BEGIN ###
import random
from pathlib import Path
import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum  # classification, regression, ts_forecasting.
from automl import train_model, evaluate_model, automl_predict
### UNMODIFIABLE IMPORT END ###
# USER CODE BEGIN IMPORTS #
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# USER CODE END IMPORTS #

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

### UNMODIFIABLE CODE BEGIN ###
DATASET_PATH = Path("{%dataset_path%}") # path for saving and loading dataset(s)
WORKSPACE_PATH = Path("{%work_dir_path%}")
PIPELINE_PATH = WORKSPACE_PATH / "pipeline" # path for saving and loading pipelines
SUBMISSION_PATH = WORKSPACE_PATH / "submission.csv"
EVAL_SET_SIZE = 0.2 # 20% of the data for evaluation
### UNMODIFIABLE CODE END ###

# --- TODO: Update these paths for your specific competition ---
TRAIN_FILE = DATASET_PATH / "train.csv" # Replace with your actual filename
TEST_FILE = DATASET_PATH / "test.csv" # Replace with your actual filename
SAMPLE_SUBMISSION_FILE = DATASET_PATH / "sample_submission.csv" # Replace with your actual filename or None
ID_COLUMN = "id" # Replace with your actual ID column name, if any
TARGET_COLUMNS = ["target"] # Replace with your actual target column name(s)


def load_data():
    """Loads train, test, and optionally sample submission files."""
    try:
        train_df = pd.read_csv(TRAIN_FILE) # TODO: Adjust pandas loader if needed
        test_df = pd.read_csv(TEST_FILE) # TODO: Adjust pandas loader if needed
        sample_sub_df = None
        if SAMPLE_SUBMISSION_FILE and (DATASET_PATH / SAMPLE_SUBMISSION_FILE).exists():
           sample_sub_df = pd.read_csv(DATASET_PATH / SAMPLE_SUBMISSION_FILE) # TODO: Adjust pandas loader if needed
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
    """Cleans and transforms data into a suitable format for modeling
    1.  Feature Engineering & Preprocessing:
        - Column Dropping: Drop irrelevant columns (e.g., IDs, high cardinality text not used, columns with too many NaNs if not imputed).
        - Missing Value Imputation:
            - Use `SimpleImputer(strategy='mean')` for numerical or `SimpleImputer(strategy='most_frequent')` for categorical.
            - IMPORTANT: Fit imputers ONLY on training data. Then, use the FITTED imputer to `.transform()` train, validation, AND test sets.
              This template's structure implies `transform_data` is called multiple times. You'll need to manage fitted transformers
              (e.g., define them globally or pass them if they are stateful and fitted on the first call with training data).
              A simpler approach for one-shot filling might be to perform stateless transformations or ensure any fitting
              logic is conditional (e.g. `if not hasattr(self, 'imputer'): self.imputer.fit(X_train)`).
        - Categorical Encoding:
            - For nominal: `OneHotEncoder(handle_unknown='ignore')`.
            - For ordinal: `OrdinalEncoder()`.
            - Fit encoders ONLY on training data and transform all sets.
        - Numerical Scaling (Optional but often beneficial for some models):
            - `StandardScaler()`. Fit ONLY on training data.
        - Feature Creation (Domain-specific): Create new features from existing ones if it makes sense for the problem.
        - Data Type Conversion: Ensure all features are numeric (int/float) before returning.

    2.  Consistency:
        - Transformations applied to training data MUST be applied identically to test data.
        - Ensure the order of columns is consistent if not inherently handled by transformers.
        - DO NOT drop rows from the test set (for submission) unless absolutely necessary and accounted for, as it will affect submission alignment.
    """
    
    print("\nPreprocessing data...")
    # Make copies to avoid modifying original dataframes
    X_train = train_features.copy()
    X_test = test_features.copy()
    
    # --- Step 0: Column Dropping ---
    # Leave only important features
    important_features = [ ]
    X_train = X_train[important_features]
    X_test = X_test[important_features]
    
    # --- Step 1: Identify feature types (based on training data) ---
    numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    # --- Step 2: Handle Missing Values (Fit on Train, Transform Train & Test) ---
    # Numerical Imputation
    # if numerical_cols:
    #     num_imputer = SimpleImputer(strategy='median')
    #     X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
    #     X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

    # Categorical Imputation
    # if categorical_cols:
    #     cat_imputer = SimpleImputer(strategy='most_frequent') # Or 'constant' with fill_value='Missing'
    #     X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
    #     X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])
    

    # --- Step 3: Encode Categorical Features (Fit on Train, Transform Train & Test) ---
    # Using OneHotEncoder for columns with few unique values and LabelEncoder for others.
    # Store encoders if inverse_transform or consistent application to new data is needed.
    # For robust OHE, ensure all categories seen during fit are handled during transform.
    # Important Note:
    # ---------
    # OneHotEncoder sparse attribute is DEPRECATED! **Must use sparse_output instead.**
    # ----------
    # Example:
    # ---------
    # ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse_output=False for DataFrame compatibility
    # ohe.fit(X_train[[col]])
    # train_encoded_names = ohe.get_feature_names_out([col])
    # # Transform training data
    # train_encoded_df = pd.DataFrame(ohe.transform(X_train[[col]]), columns=train_encoded_names, index=X_train.index)
    # X_train = pd.concat([X_train.drop(columns=[col]), train_encoded_df], axis=1)
    # # Transform test data
    # test_encoded_df = pd.DataFrame(ohe.transform(X_test[[col]]), columns=train_encoded_names, index=X_test.index)
    # X_test = pd.concat([X_test.drop(columns=[col]), test_encoded_df], axis=1)
    # ---------
    # le = LabelEncoder()
    # X_train[col] = le.fit_transform(X_train[col].astype(str))
    # X_test[col] = X_test[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1) # -1 for unknown
   
    # --- Step 4: Feature Scaling (Fit on Train, Transform Train & Test) ---
    # Scale numerical features (original numerical + label encoded if they are treated as numerical)
    # Exclude OHE columns if you prefer not to scale them, though scaling them is often fine.
    # Example:
    # ---------
    # scaler = StandardScaler()
    # X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    # X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # --- Step 5: Other transformations (e.g., log transform skewed features) ---
    # Apply these transformations consistently after fitting on training data if applicable.
    # Example: Log transform (fit on train, transform train & test)
    # for col in numerical_cols: # Use original numerical_cols or an updated list
    #    if col in X_train.columns and X_train[col].skew() > 1.0:
    #        print(f"Log transforming skewed column: {col}")
    #        # Add 1 to prevent log(0) issues, ensure positive values
    #        X_train[col] = np.log1p(X_train[col] - X_train[col].min())
    #        X_test[col] = np.log1p(X_test[col] - X_test[col].min()) # Apply same shift as train
    
    # --- Step 6: Feature Engineering ---
    # Create new features from existing ones if it makes sense for the problem.
    # Use domain knowledge to create new features if it makes sense.
    # Example:
    # ---------
    # X_train['feature_name'] = X_train['feature_name'].map({})
    # X_train['feature_name'] = X_train['feature_name'] + X_train['feature_name_2']

    print("Preprocessing complete.")
    print(f"Train processed shape: {X_train.shape}, Test processed shape: {X_test.shape}")
    return X_train, X_test

# --------------------------------------------------------------------------- #
# Section 6: Final Model Training & Submission
# --------------------------------------------------------------------------- #
def create_submission_file(test_ids, test_predictions, sample_sub_df, submission_filename="submission.csv"):
    """Creates the submission file in the required format."""
    print(f"\nCreating submission file: {submission_filename}")
    # Flatten the predictions if they're 2D
    if isinstance(test_predictions, np.ndarray) and test_predictions.ndim > 1:
        test_predictions = test_predictions.flatten()
    if not isinstance(test_ids, (pd.Series, np.ndarray)):
        try:
            # Attempt to get ID from sample submission if test_ids is not directly usable
            if sample_sub_df is not None and ID_COLUMN in sample_sub_df.columns:
                test_ids = sample_sub_df[ID_COLUMN]
            else: # Fallback if ID_COLUMN not in sample submission
                test_ids = np.arange(len(test_predictions)) # Simple range if no IDs
                print(f"Warning: Using default range for test_ids as {ID_COLUMN} not found in sample submission.")
        except Exception as e:
            print(f"Warning: Could not load sample submission to get test_ids: {e}. Using default range.")
            test_ids = np.arange(len(test_predictions))


    submission_df = pd.DataFrame(test_predictions, columns=[TARGET_COLUMNS])
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
    # --- TODO: Define problem_type ('classification', 'regression', 'multi-label classification', 'ts_forecasting') ---
    current_problem_type = 'classification' # Example

    # --- Step 0: Load Data ---
    train_df, test_df, sample_sub_df = load_data()
    if train_df is None or test_df is None:
        print("Exiting due to data loading failure.")
        return
    
    # Store original test IDs for submission
    original_test_ids = test_df[ID_COLUMN] if ID_COLUMN in test_df.columns else test_df.index
    
    # --- Step 1: Cleaning ---
    X_train_cleaned, X_test_cleaned, y_train_full = cleaning_data(train_df, test_df, TARGET_COLUMNS)

    # --- Step 2: Preprocessing ---
    # The target column name (TARGET_COLUMN) is passed to preprocess_data
    X_train_processed, X_test_processed  = preprocess_data(X_train_cleaned, X_test_cleaned)
    
    # --- Create a fixed validation split from the processed full training data ---
    print(f"\nCreating a fixed train-evaluation split (Eval size: {EVAL_SET_SIZE*100}%) ...")
    # For multi-target, set stratify=None
    stratify_target = None
    if current_problem_type.startswith('classification') and len(TARGET_COLUMNS) == 1:
        stratify_target = y_train_full  # Only stratify for single-target classification
    X_train_main, X_eval_holdout, y_train_main, y_eval_holdout = train_test_split(
        X_train_processed, y_train_full,
        test_size=EVAL_SET_SIZE,
        random_state=SEED,
        stratify=stratify_target
    )
    
    # --- Step 3: Train AutoML model ---
    # You must use pre-defined train_model function to train the model.
    model = train_model(X_train_main, y_train_main)
    
    # --- Step 4: Evaluate the trained model ---
    # You must use pre-defined evaluate_model function to evaluate the model.
    model_performance = evaluate_model(model, X_eval_holdout, y_eval_holdout)

    # --- Step 5: Evaluate predictions for the test dataset using AutoML Framework ---
    # You must use pre-defined automl_predict function to predict the model.
    predictions:np.ndarray = automl_predict(model, X_test_processed) # returns 2D array
  
    # --- Step 6: Create submission file ---
    if predictions is not None:
        create_submission_file(original_test_ids, predictions, sample_sub_df, "submission.csv")
    else:
        print("No test predictions generated, skipping submission file creation.")
    print("\nKaggle ML Workflow finished.")

if __name__ == "__main__":
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
    main()