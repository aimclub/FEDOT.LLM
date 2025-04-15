from pathlib import Path

from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.log import get_logger

from ..eval.simple_eval import execute_code
from ..eval.types import ExecutionResult, ProgramStatus

logger = get_logger()


def _generate_code_file(code: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "solution.py", "w") as f:
        f.write(code)
    return output_dir / "solution.py"


def run_evaluate(state: AutoMLAgentState):
    assert state["solutions"] is not None, "Solutions are not found"
    solution = state["solutions"][-1]
    assert solution["code"] is not None, "Code is not found"
    workspace = state["workspace"]
    assert workspace is not None, "Workspace is not found"
    logger.info("Running evaluate")
    logger.debug(f"{solution['code']}")
    code_path = _generate_code_file(solution["code"], workspace)
    result: ExecutionResult
    result = execute_code(path_to_run_code=code_path, timeout=None)
    if result:
        if result.program_status == ProgramStatus.kFailed:
            logger.error(result.sandbox_result)
        logger.debug(
            f"Evaluate result\nStatus: {result.program_status}\nStdout: {result.stdout}\nStderr: {result.stderr}\nSandbox result: {result.sandbox_result}"
        )

    solution["exec_result"] = result
    return state
