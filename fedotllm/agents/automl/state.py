from typing import List, Optional, TypedDict

from fedotllm.agents.automl.eval.local_exec import ExecutionResult
from fedotllm.agents.automl.structured import FedotConfig
from fedotllm.agents.state import FedotLLMAgentState
from fedotllm.predictor.task import PredictionTask


class Solution(TypedDict):
    code: Optional[str]
    exec_result: Optional[ExecutionResult]
    fix_tries: int


class AutoMLAgentState(FedotLLMAgentState):
    description: str
    reflection: str
    fedot_config: Optional[FedotConfig]
    skeleton: Optional[str]
    codegen_sol: Solution
    solutions: List[Solution]
    exec_result: Optional[ExecutionResult]
    metrics: Optional[str]
    pipeline: Optional[str]
    report: Optional[str]
    prediction_task: Optional[PredictionTask]
