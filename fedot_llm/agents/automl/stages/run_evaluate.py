from fedot_llm.agents.automl.state import AutoMLAgentState
from fedot_llm.agents.automl.eval.local_exec import execute_code, ProgramStatus, ExecutionResult
from settings.config_loader import get_settings
from pathlib import Path
from typing import Literal
from fedot_llm.log import get_logger

logger = get_logger()


def run_evaluate(state: AutoMLAgentState, stage: Literal['main', 'deploy']):
    logger.info(f"Running evaluate {stage}")
    solution = state['solutions'][-1]
    result: ExecutionResult
    if stage == 'main':
        result = execute_code(solution['code'],
                              timeout=20*60,
                              sandbox=True,
                              output_dir=Path(
                                  get_settings()['config']['output_dir']),
                              vaults=[Path(get_settings()['config']['dataset_dir'])])
    elif stage == 'deploy':
        result = execute_code(solution['code'],
                              timeout=20*60,
                              sandbox=True,
                              argv=["--deploy", "--test"],
                              vaults=[Path(get_settings()['config']['dataset_dir']),
                                      Path(get_settings()['config']['output_dir']) / "_pipelines"])

    if result:
        if result.program_status == ProgramStatus.kFailed:
            logger.error(result.sandbox_result)
        logger.debug(
            f"Evaluate result\nStatus: {result.program_status}\nStdout: {result.stdout}\nStderr: {result.stderr}\nSandbox result: {result.sandbox_result}")

    solution['exec_result'] = result
    return state
