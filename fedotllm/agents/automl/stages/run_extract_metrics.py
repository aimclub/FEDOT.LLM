import re

from fedot.api.main import Fedot
from golem.core.dag.graph_utils import graph_structure

from pathlib import Path
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.log import get_logger

logger = get_logger()


def _extract_metrics(raw_output: str):
    pattern = "Model metrics:(.*?)"
    matches = re.findall(pattern, raw_output)
    if matches:
        return matches[0].strip()
    return None


def run_extract_metrics(state: AutoMLAgentState):
    solution = state['solutions'][-1]
    work_dir = state['work_dir']

    logger.info("Running extract_metrics")
    state['metrics'] = _extract_metrics(solution['exec_result'].stdout)
    logger.info(f"Metrics: {state['metrics']}")

    if Path(work_dir / 'pipeline').exists():
        model = Fedot(problem='classification')
        model.load(work_dir / 'pipeline')
        state['pipeline'] = graph_structure(model.current_pipeline)
        logger.info(f"Pipeline: {state['pipeline']}")
    else:
        logger.error("Pipeline not found")
    return state
