import shutil
from pathlib import Path

from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.log import get_logger
from fedotllm.settings.config_loader import get_settings

logger = get_logger()


def run_save_results(state: AutoMLAgentState):
    solution = state['solutions'][-1]
    result_dir = Path(get_settings()['config']['result_dir'])
    result_dir.mkdir(parents=True, exist_ok=True)
    if solution['code'] is not None:
        with open(result_dir / "solution.py", "w") as f:
            f.write(solution['code'])
        logger.info(f"Saved solution to {result_dir / 'solution.py'}")

    pipeline_path = Path(get_settings()['config']['output_dir']) / "pipeline"
    if pipeline_path.exists():
        shutil.copytree(pipeline_path, result_dir /
                        "pipeline", dirs_exist_ok=True)
        logger.info(f"Saved pipelines to {result_dir / 'pipeline'}")
    
    submission_path  = Path(get_settings()['config']['output_dir']) / "submission.csv"
    if submission_path.exists():
        shutil.copy(submission_path, result_dir / "submission.csv")
        logger.info(f"Saved submission to {result_dir / 'pipeline'}")