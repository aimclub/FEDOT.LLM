from pathlib import Path

from fedot.api.main import Fedot
from golem.core.dag.graph_utils import graph_structure

from agents.automl.automl import AutoMLAgentState
from log import get_logger
from settings.config_loader import get_settings

logger = get_logger()


def run_extract_metrics(state: AutoMLAgentState):
    logger.info("Running extract_metrics")
    solution = state['solutions'][-1]
    state['metrics'] = solution['exec_result'].global_vars.get(
        'MODEL_PERFORMANCE')
    logger.info(f"Metrics: {state['metrics']}")
    model = Fedot(problem='classification')
    model.load(Path(get_settings().config.output_dir) / 'pipeline')
    state['pipeline'] = graph_structure(model.current_pipeline)
    logger.info(f"Pipeline: {state['pipeline']}")
    return state
