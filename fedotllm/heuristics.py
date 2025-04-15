import numpy as np
from pandas import Series

from fedotllm.constants import BINARY, MULTICLASS, REGRESSION
from fedotllm.log import logger

MULTICLASS_UPPER_LIMIT = 1000  # assume regression if dtype is numeric and unique label count is above this limit
LARGE_DATA_THRESHOLD = 1000
REGRESS_THRESHOLD_LARGE_DATA = 0.05
REGRESS_THRESHOLD_SMALL_DATA = 0.1


# Adapted from AutoGluon. Licensed under the Apache License 2.0.
# https://github.com/autogluon/autogluon/blob/master/core/src/autogluon/core/utils/utils.py
def infer_problem_type(y: Series, silent=False) -> str:
    """Identifies which type of prediction problem we are interested in (if user has not specified).
    Ie. binary classification, multi-class classification, or regression.
    """
    # treat None, NaN, INF, NINF as NA
    y = y.replace([np.inf, -np.inf], np.nan, inplace=False)
    y = y.dropna()
    num_rows = len(y)

    if num_rows == 0:
        raise ValueError("Label column cannot have 0 valid values")

    unique_values = y.unique()

    if num_rows > LARGE_DATA_THRESHOLD:
        regression_threshold = REGRESS_THRESHOLD_LARGE_DATA  # if the unique-ratio is less than this, we assume multiclass classification, even when labels are integers
    else:
        regression_threshold = REGRESS_THRESHOLD_SMALL_DATA

    unique_count = len(unique_values)
    if unique_count == 2:
        problem_type = BINARY
        reason = "only two unique label-values observed"
    elif y.dtype.name in ["object", "category", "string"]:
        problem_type = MULTICLASS
        reason = f"dtype of label-column == {y.dtype.name}"
    elif np.issubdtype(y.dtype, np.floating):
        unique_ratio = unique_count / float(num_rows)
        if (unique_ratio <= regression_threshold) and (
            unique_count <= MULTICLASS_UPPER_LIMIT
        ):
            try:
                can_convert_to_int = np.array_equal(y, y.astype(int))
                if can_convert_to_int:
                    problem_type = MULTICLASS
                    reason = "dtype of label-column == float, but few unique label-values observed and label-values can be converted to int"
                else:
                    problem_type = REGRESSION
                    reason = "dtype of label-column == float and label-values can't be converted to int"
            except Exception:
                problem_type = REGRESSION
                reason = "dtype of label-column == float and label-values can't be converted to int"
        else:
            problem_type = REGRESSION
            reason = (
                "dtype of label-column == float and many unique label-values observed"
            )
    elif np.issubdtype(y.dtype, np.integer):
        unique_ratio = unique_count / float(num_rows)
        if (unique_ratio <= regression_threshold) and (
            unique_count <= MULTICLASS_UPPER_LIMIT
        ):
            problem_type = MULTICLASS  # TODO: Check if integers are from 0 to n-1 for n unique values, if they have a wide spread, it could still be regression
            reason = (
                "dtype of label-column == int, but few unique label-values observed"
            )
        else:
            problem_type = REGRESSION
            reason = (
                "dtype of label-column == int and many unique label-values observed"
            )
    else:
        raise NotImplementedError(f"label dtype {y.dtype} not supported!")
    if not silent:
        logger.log(
            25,
            f"AutoGluon infers your prediction problem is: '{problem_type}' (because {reason}).",
        )

        # TODO: Move this outside of this function so it is visible even if problem type was not inferred.
        if problem_type in [BINARY, MULTICLASS]:
            if unique_count > 10:
                logger.log(
                    20,
                    f"\tFirst 10 (of {unique_count}) unique label values:  {list(unique_values[:10])}",
                )
            else:
                logger.log(
                    20, f"\t{unique_count} unique label values:  {list(unique_values)}"
                )
        elif problem_type == REGRESSION:
            y_max = y.max()
            y_min = y.min()
            y_mean = y.mean()
            y_stddev = y.std()
            logger.log(
                20,
                f"\tLabel info (max, min, mean, stddev): ({y_max}, {y_min}, {round(y_mean, 5)}, {round(y_stddev, 5)})",
            )

        logger.log(
            25,
            f"\tIf '{problem_type}' is not the correct problem_type, please manually specify the problem_type parameter during Predictor init "
            f"(You may specify problem_type as one of: {[BINARY, MULTICLASS, REGRESSION]})",
        )
    return problem_type
