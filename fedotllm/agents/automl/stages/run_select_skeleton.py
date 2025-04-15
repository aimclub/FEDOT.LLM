from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.agents.automl.templates.load_template import (
    load_template,
    render_template,
)
from fedotllm.log import get_logger
from fedotllm.tabular import Dataset

logger = get_logger()


def run_select_skeleton(state: AutoMLAgentState, dataset: Dataset) -> AutoMLAgentState:
    logger.info("Running select skeleton")
    fedot_config = state["fedot_config"]

    # Get prediction method
    predict_method = {
        "predict": "predict(features=input_data)",
        "forecast": "forecast(pre_history=input_data)",
        "predict_proba": "predict_proba(features=input_data)",
    }.get(fedot_config.predict_method)

    if predict_method is None:
        raise ValueError(f"Unknown predict method: {fedot_config.predict_method}")

    skeleton = load_template("skeleton")
    skeleton = render_template(
        template=skeleton,
        dataset_path=dataset.path,
        workspace=state["workspace"].resolve(),
    )

    state["skeleton"] = skeleton
    return state
