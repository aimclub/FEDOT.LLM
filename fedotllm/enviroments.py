import os
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from fedotllm.log import logger


class Observation(BaseModel):
    error: bool = Field(default=False)
    msg: str = Field(default="")
    stdout: str = Field(default="")
    stderr: str = Field(default="")


def execute_code(path_to_run_code: Path) -> Observation:
    try:
        result = subprocess.run(
            ["python3", "-W", "ignore", path_to_run_code],
            capture_output=True,
            text=True,
            preexec_fn=os.setsid,
        )
        logger.debug(f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        return Observation(
            error=result.returncode != 0, stdout=result.stdout, stderr=result.stderr
        )
    except Exception as e:
        stderr = f"An unexpected error occurred in the execution harness: {type(e).__name__}: {e}"
        logger.error(
            f"Unexpected error executing {path_to_run_code}: {e}", exc_info=True
        )
        return Observation(error=True, stdout="", stderr=stderr)
