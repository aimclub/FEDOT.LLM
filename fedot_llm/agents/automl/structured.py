from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional, Union, List
from enum import Enum


class ProblemType(str, Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    TS_FORECASTING = 'ts_forecasting'


class PresetType(str, Enum):
    BEST_QUALITY = 'best_quality'
    FAST_TRAIN = 'fast_train'
    STABLE = 'stable'
    AUTO = 'auto'
    GPU = 'gpu'
    TS = 'ts'
    AUTOML = 'automl'


class ClassificationMetricsEnum(str, Enum):
    ROCAUC = 'roc_auc'
    precision = 'precision'
    f1 = 'f1'
    logloss = 'neg_log_loss'
    ROCAUC_penalty = 'roc_auc_pen'
    accuracy = 'accuracy'


class RegressionMetricsEnum(str, Enum):
    RMSE = 'rmse'
    MSE = 'mse'
    MSLE = 'neg_mean_squared_log_error'
    MAPE = 'mape'
    SMAPE = 'smape'
    MAE = 'mae'
    R2 = 'r2'
    RMSE_penalty = 'rmse_pen'


class TimeSeriesForecastingMetricsEnum(str, Enum):
    MASE = 'mase'
    RMSE = 'rmse'
    MSE = 'mse'
    MSLE = 'neg_mean_squared_log_error'
    MAPE = 'mape'
    SMAPE = 'smape'
    MAE = 'mae'
    R2 = 'r2'
    RMSE_penalty = 'rmse_pen'


class FedotConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    problem: ProblemType = Field(...,
                                 description='Name of the modelling problem to solve')
    timeout: float = Field(
        1, description="Time for model design (in minutes): None or -1 means infinite time.")
    seed: Optional[int] = Field(None, description="Seed for random generation")
    cv_folds: Optional[int] = Field(
        None, description="Number of folds for cross-validation")
    preset: PresetType = Field(
        PresetType.AUTO,
        description=(
            "Name of the preset for model building. Possible options:\n"
            "best_quality -> All models that are available for this data type and task are used\n"
            "fast_train -> Models that learn quickly. This includes preprocessing operations (data operations) that only reduce the dimensionality of the data, but cannot increase it. For example, there are no polynomial features and one-hot encoding operations\n"
            "stable -> The most reliable preset in which the most stable operations are included\n"
            "auto -> Automatically determine which preset should be used\n"
            "gpu -> Models that use GPU resources for computation\n"
            "ts -> A special preset with models for time series forecasting task\n"
            "automl -> A special preset with only AutoML libraries such as TPOT and H2O as operations"
        )
    )
    metrics: List[Union[ClassificationMetricsEnum, RegressionMetricsEnum, TimeSeriesForecastingMetricsEnum]
                  ] = Field(..., description="Choose all relevant to problem metrics of model quality assessment")
    predict_method: Literal['predict', 'predict_proba', 'forecast'] = Field(...,
                                                                            description="Method for prediction: predict - for classification and regression, predict_proba - for classification, forecast - for time series forecasting")


class ProblemReflection(BaseModel):
    reflection: str = Field(..., description="Reflect on the problem, and describe it in your own words, in bullet points."
                            "Pay attention to small details, nuances, notes and examples in the problem description.")
    target: str = Field(
        ..., description="Name of probem target feature. This feature we want to predict.")
