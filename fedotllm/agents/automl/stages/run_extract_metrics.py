import re
from pathlib import Path

from fedot.api.main import Fedot
from golem.core.dag.graph_utils import graph_structure

from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.log import get_logger

logger = get_logger()


def _extract_metrics(raw_output: str):
    pattern = r"Model metrics:\s*(\{.*?\})"
    match = re.search(pattern, raw_output)
    if match:
        return match.group(1).strip()
    return None


def run_extract_metrics(state: AutoMLAgentState):
    solution = state["solutions"][-1]
    workspace = state["workspace"]

    logger.info("Running extract_metrics")
    state["metrics"] = _extract_metrics(solution["exec_result"].stdout)
    logger.info(f"Metrics: {state['metrics']}")

    if Path(workspace / "pipeline").exists():
        model = Fedot(problem="classification")
        model.load(workspace / "pipeline")
        state["pipeline"] = graph_structure(model.current_pipeline)
        logger.info(f"Pipeline: {state['pipeline']}")
    else:
        logger.error("Pipeline not found")
    return state
