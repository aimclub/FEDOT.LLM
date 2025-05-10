import logging
import os
import subprocess
from pathlib import Path

from .types import CodeObservation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def execute_code(path_to_run_code: Path) -> CodeObservation:
    try:
        result = subprocess.run(
            ["python3", "-W", "ignore", path_to_run_code],
            capture_output=True,
            text=True,
            preexec_fn=os.setsid,
        )
        return CodeObservation(
            error=result.returncode != 0, stdout=result.stdout, stderr=result.stderr
        )
    except Exception as e:
        stderr = f"An unexpected error occurred in the execution harness: {type(e).__name__}: {e}"
        logger.error(
            f"Unexpected error executing {path_to_run_code}: {e}", exc_info=True
        )
        return CodeObservation(error=True, stdout="", stderr=stderr)
