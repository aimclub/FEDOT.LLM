from agents.automl.automl import AutoMLAgentState
from settings.config_loader import get_settings
from pathlib import Path
import shutil
from log import get_logger

logger = get_logger()


def run_save_results(state: AutoMLAgentState):
    solution = state['solutions'][-1]
    if solution['code'] is not None:
        result_dir = Path(get_settings()['config']['result_dir'])
        result_dir.mkdir(parents=True, exist_ok=True)
        with open(result_dir / "solution.py", "w") as f:
            f.write(solution['code'])
        logger.info(f"Saved solution to {result_dir / 'solution.py'}")

    pipeline_path = Path(get_settings()['config']['output_dir']) / "pipeline"
    if pipeline_path.exists():
        shutil.copytree(pipeline_path, result_dir /
                        "pipeline", dirs_exist_ok=True)
        logger.info(f"Saved pipelines to {result_dir / 'pipeline'}")
