from agents.automl.state import AutoMLAgentState
from log import get_logger
from agents.automl.templates.load_template import load_template, render_template

logger = get_logger()


def run_select_skeleton(state: AutoMLAgentState) -> AutoMLAgentState:
    logger.info("Running select skeleton")
    fedot_config = state['fedot_config']

    # Get prediction method
    predict_method = {
        'predict': "predict(features=input_data)",
        'forecast': "forecast(pre_history=input_data)",
        'predict_proba': "predict_proba(features=input_data)"
    }.get(fedot_config.predict_method)

    if predict_method is None:
        raise ValueError(
            f"Unknown predict method: {fedot_config.predict_method}")

    skeleton = load_template("skeleton")
    skeleton = render_template(
        template=skeleton,
        # Fedot config
        problem=f"{fedot_config.problem}",
        timeout=fedot_config.timeout,
        seed=fedot_config.seed,
        cv_folds=fedot_config.cv_folds,
        preset=f"'{fedot_config.preset.value}'",
        metric=fedot_config.metric.value,
        # Prediction method
        predict_method=predict_method
    )

    state['skeleton'] = skeleton
    return state
