# Task Inference
NO_FILE_IDENTIFIED = "NO_FILE_IDENTIFIED"
NO_ID_COLUMN_IDENTIFIED = "NO_ID_COLUMN_IDENTIFIED"

# Data types and files
TRAIN = "train"
TEST = "test"
OUTPUT = "output"
STATIC_FEATURES = "static_features"
DESCRIPTION = "description"

# Task types
TABULAR = "tabular"
MULTIMODAL = "multimodal"
TIME_SERIES = "time_series"
TASK_TYPES = [TABULAR, MULTIMODAL, TIME_SERIES]

# Problem types
BINARY = "binary"
MULTICLASS = "multiclass"
REGRESSION = "regression"
TIME_SERIES = "time_series"
PROBLEM_TYPES = [BINARY, MULTICLASS, REGRESSION, TIME_SERIES]
CLASSIFICATION_PROBLEM_TYPES = [BINARY, MULTICLASS]

# Presets/Configs
CONFIGS = "configs"
MEDIUM_QUALITY = "medium_quality"
HIGH_QUALITY = "high_quality"
BEST_QUALITY = "best_quality"
DEFAULT_QUALITY = BEST_QUALITY
PRESETS = [MEDIUM_QUALITY, HIGH_QUALITY, BEST_QUALITY]

# Metrics
ROC_AUC = "roc_auc"
LOG_LOSS = "log_loss"
ACCURACY = "accuracy"
F1 = "f1"
QUADRARIC_KAPPA = "quadratic_kappa"
BALANCED_ACCURACY = "balanced_accuracy"
ROOT_MEAN_SQUARED_ERROR = "root_mean_squared_error"
MEAN_SQUARED_ERROR = "mean_squared_error"
MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
R2 = "r2"
ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR = "root_mean_squared_logarithmic_error"

CLASSIFICATION_PROBA_EVAL_METRIC = [ROC_AUC, LOG_LOSS, F1]

METRICS_DESCRIPTION = {
    ROC_AUC: "Area under the receiver operating characteristics (ROC) curve",
    LOG_LOSS: "Log loss, also known as logarithmic loss",
    ACCURACY: "Accuracy",
    F1: "F1 score",
    QUADRARIC_KAPPA: "Quadratic kappa, i.e., the Cohen kappa metric",
    BALANCED_ACCURACY: "Balanced accuracy, i.e., the arithmetic mean of sensitivity and specificity",
    ROOT_MEAN_SQUARED_ERROR: "Root mean squared error (RMSE)",
    MEAN_SQUARED_ERROR: "Mean squared error (MSE)",
    MEAN_ABSOLUTE_ERROR: "Mean absolute_error (MAE)",
    R2: "R-squared",
    ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR: "Root mean squared logarithmic error (RMSLE)",
}

METRICS_BY_PROBLEM_TYPE = {
    BINARY: [ROC_AUC, LOG_LOSS, ACCURACY, F1, QUADRARIC_KAPPA, BALANCED_ACCURACY],
    MULTICLASS: [ROC_AUC, LOG_LOSS, ACCURACY, F1, QUADRARIC_KAPPA, BALANCED_ACCURACY],
    REGRESSION: [
        ROOT_MEAN_SQUARED_ERROR,
        MEAN_SQUARED_ERROR,
        MEAN_ABSOLUTE_ERROR,
        R2,
        ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR,
    ],
    TIME_SERIES: [
        ROOT_MEAN_SQUARED_ERROR,
        MEAN_SQUARED_ERROR,
        MEAN_ABSOLUTE_ERROR,
        ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR,
    ],
}

PREFERED_METRIC_BY_PROBLEM_TYPE = {
    BINARY: ROC_AUC,
    MULTICLASS: ROC_AUC,
    REGRESSION: ROOT_MEAN_SQUARED_ERROR,
}

DEFAULT_FORECAST_HORIZON = 1

# File formats
CSV_SUFFIXES = [".csv"]
PARQUET_SUFFIXES = [".parquet", ".pq"]
EXCEL_SUFFIXES = [".xls", ".xlsx", ".xlsm", ".xlsb", ".odf", ".ods", ".odt"]
ARFF_SUFFIXES = [".arff"]
DATASET_EXTENSIONS = [*CSV_SUFFIXES, *PARQUET_SUFFIXES, *EXCEL_SUFFIXES, *ARFF_SUFFIXES]
# Initial Session state
DEFAULT_SESSION_VALUES = {
    "llm": {},
    "uploaded_files": {},
    "lang": "en",
    "messages": [
        {
            "role": "assistant",
            "content": "Hello! Pick a model, upload the dataset files and send me the task description.",
        }
    ],
    "prev_graph": None,
    "output_filename": None,
    "output_file": None,
    "task_description": None,
    "task_running": False,
    "config_overrides": None,
}
