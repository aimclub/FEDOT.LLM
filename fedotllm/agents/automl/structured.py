from enum import Enum
from typing import Literal, Optional, Union, List

from fedot.core.repository.tasks import TaskTypesEnum
from pydantic import BaseModel, Field, ConfigDict


class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TS_FORECASTING = "ts_forecasting"


class PresetType(str, Enum):
    BEST_QUALITY = "best_quality"
    FAST_TRAIN = "fast_train"
    STABLE = "stable"
    AUTO = "auto"
    GPU = "gpu"
    TS = "ts"
    AUTOML = "automl"


class IndustrialStrategyType(str, Enum):
    ANOMALY_DETECTION = "anomaly_detection"
    FEDERATED_AUTOML = "federated_automl"
    FORECASTING_ASSUMPTIONS = "forecasting_assumptions"
    FORECASTING_EXOGENOUS = "forecasting_exogenous"
    KERNEL_AUTOML = "kernel_automl"
    LORA_STRATEGY = "lora_strategy"
    SAMPLING_STRATEGY = "sampling_strategy"


class ClassificationMetricsEnum(str, Enum):
    ROCAUC = "roc_auc"
    precision = "precision"
    # f1 = 'f1'
    # logloss = 'neg_log_loss'
    # ROCAUC_penalty = 'roc_auc_pen'
    accuracy = "accuracy"


class RegressionMetricsEnum(str, Enum):
    RMSE = "rmse"
    MSE = "mse"
    MSLE = "neg_mean_squared_log_error"
    MAPE = "mape"
    SMAPE = "smape"
    MAE = "mae"
    R2 = "r2"
    RMSE_penalty = "rmse_pen"


class TimeSeriesForecastingMetricsEnum(str, Enum):
    MASE = "mase"
    RMSE = "rmse"
    MSE = "mse"
    MSLE = "neg_mean_squared_log_error"
    MAPE = "mape"
    SMAPE = "smape"
    MAE = "mae"
    R2 = "r2"
    RMSE_penalty = "rmse_pen"


class FedotConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    problem: TaskTypesEnum = Field(
        ..., description="Name of the modelling problem to solve: classification - predicting a distinct class, regression - predicting numeric value, ts_forecasting - forecasting a time series. Prioritize clear objective description over target data type if conflicting."
    )
    timeout: float = Field(
        ..., description="Time for model design (in minutes): Default: 1.0"
    )
    cv_folds: Optional[int] = Field(
        ..., description="Number of folds for cross-validation: Default: None"
    )
    preset: PresetType = Field(
        ...,
        description=(
            "Name of the preset for model building. Possible options:\n"
            "best_quality -> All models that are available for this data type and task are used\n"
            "fast_train -> Models that learn quickly. This includes preprocessing operations (data operations) that only reduce the dimensionality of the data, but cannot increase it. For example, there are no polynomial features and one-hot encoding operations\n"
            "stable -> The most reliable preset in which the most stable operations are included\n"
            "auto -> Automatically determine which preset should be used\n"
            "gpu -> Models that use GPU resources for computation\n"
            "ts -> A special preset with models for time series forecasting task\n"
            "automl -> A special preset with only AutoML libraries such as TPOT and H2O as operations"
            "Default: auto"
        ),
    )
    metric: Union[
        ClassificationMetricsEnum,
        RegressionMetricsEnum,
        TimeSeriesForecastingMetricsEnum,
    ] = Field(
        ...,
        description="Choose relevant to problem metric of model quality assessment.",
    )
    predict_method: Literal["predict", "predict_proba", "forecast"] = Field(
        ...,
        description="Method for prediction: predict - for classification and regression, especially if target is categorical, predict_proba - for classification, forecast - for time series forecasting",
    )


class FedotIndustrialConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    problem: ProblemType = Field(
        ..., description="Name of the modelling problem to solve"
    )
    timeout: float = Field(
        ..., description="Time for model design (in minutes): Default: 1.0"
    )
    cv_folds: Optional[int] = Field(
        ..., description="Number of folds for cross-validation: Default: None"
    )
    metrics: List[
        Union[
            ClassificationMetricsEnum,
            RegressionMetricsEnum,
            TimeSeriesForecastingMetricsEnum,
        ]
    ] = Field(
        ...,
        description="Choose all relevant to problem metrics of model quality assessment",
    )
    predict_method: Literal["predict", "predict_proba"] = Field(
        ...,
        description="Method for prediction: predict - for classification, regression, time series forecasting, predict_proba - for classification",
    )
    industrial_strategy: Optional[IndustrialStrategyType] = Field(
        None, description="Industrial strategy for model building"
    )


class ProblemReflection(BaseModel):
    reflection: str = Field(
        ...,
        description="Reflect on the problem, and describe it in your own words, in bullet points."
        "Pay attention to small details, nuances, notes and examples in the problem description.",
    )
    target: str = Field(
        ...,
        description="Name of probem target feature. This feature we want to predict.",
    )
    train_features: str = Field(
        ..., description="Name of dataset split with train data"
    )
    test_features: str = Field(..., description="Name of dataset split with test data")
