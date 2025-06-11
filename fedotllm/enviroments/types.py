from dataclasses import dataclass
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ProgramStatus(Enum):
    kUnknown = 0
    kSuccess = 1
    kFailed = 2
    kTimeout = 3


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    program_status: ProgramStatus = ProgramStatus.kUnknown
    sandbox_result: str = ""
    trace: Optional[str] = None
    global_vars: Optional[dict] = None


class Observation(BaseModel):
    error: bool = Field(default=False)
    msg: str = Field(default="")
    stdout: str = Field(default="")
    stderr: str = Field(default="")
