from fedot_llm.agents.automl.state import AutoMLAgentState
from fedot_llm.agents.automl.eval.local_exec import execute_code, ProgramStatus, ExecutionResult
from settings.config_loader import get_settings
from pathlib import Path
from fedot_llm.log import get_logger

logger = get_logger()


def run_evaluate(state: AutoMLAgentState):
    solution = state['solutions'][-1]
    logger.info("Running evaluate")
    logger.debug(f"{solution['code']}")
    result: ExecutionResult
    result = execute_code(solution['code'],
                          timeout=20*60,
                          sandbox=True,
                          output_dir=Path(
        get_settings()['config']['output_dir']),
        vaults=[Path(get_settings()['config']['dataset_dir'])])

    if result:
        if result.program_status == ProgramStatus.kFailed:
            logger.error(result.sandbox_result)
        logger.debug(
            f"Evaluate result\nStatus: {result.program_status}\nStdout: {result.stdout}\nStderr: {result.stderr}\nSandbox result: {result.sandbox_result}")

    solution['exec_result'] = result
    return state
