import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from .types import ExecutionResult, ProgramStatus

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def execute_code(path_to_run_code: Path, timeout: Optional[float] = None, cwd: Optional[Path] = None) -> ExecutionResult:
    execution_result = ExecutionResult()
    try:
        result = subprocess.run(['python3', '-W', 'ignore', path_to_run_code],
                                capture_output=True, text=True, timeout=timeout,
                                cwd=cwd,
                                preexec_fn=os.setsid)
    except subprocess.TimeoutExpired:
        logger.info("Code execution timed out.")
        execution_result.program_status = ProgramStatus.kTimeout
    except subprocess.CalledProcessError as e:
        if e.returncode < 0:
            # Negative return codes usually indicate termination by a signal
            logger.info(f"Process was killed by signal {-e.returncode}")
            execution_result.stderr = f"Process was terminated by the operating system (signal {-e.returncode})"
            execution_result.program_status = ProgramStatus.kFailed
        else:
            logger.info(f"Process exited with non-zero status: {e.returncode}")
            execution_result.stderr = f"Process exited with status {e.returncode}: {e.stderr}"
    else:
        if result.returncode != 0:
            logger.info(
                f"Process exited with non-zero status: {result.returncode}")
            execution_result.stderr = f"Process exited with status {result.returncode}: {result.stderr}"
            execution_result.program_status = ProgramStatus.kFailed
        else:
            logger.info("Code executed successfully without errors.")
            execution_result.program_status = ProgramStatus.kSuccess

    if result and hasattr(result, 'stdout'):
        execution_result.stdout = result.stdout

    return execution_result
