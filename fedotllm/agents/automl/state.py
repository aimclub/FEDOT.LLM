from typing import List
from typing import TypedDict, Optional

from fedotllm.agents.automl.eval.local_exec import ExecutionResult
from fedotllm.agents.automl.structured import ProblemReflection, FedotConfig
from fedotllm.agents.state import FedotLLMAgentState


class Solution(TypedDict):
    code: Optional[str]
    exec_result: Optional[ExecutionResult]
    fix_tries: int


class AutoMLAgentState(FedotLLMAgentState):
    description: str
    reflection: Optional[ProblemReflection]
    fedot_config: Optional[FedotConfig]
    skeleton: Optional[str]
    codegen_sol: Solution
    solutions: List[Solution]
    exec_result: Optional[ExecutionResult]
    metrics: Optional[str]
    pipeline: Optional[str]
