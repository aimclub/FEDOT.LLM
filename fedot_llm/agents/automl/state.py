from typing import TypedDict, Optional
from fedot_llm.agents.state import FedotLLMAgentState
from fedot_llm.agents.automl.structured import ProblemReflection, FedotConfig
from fedot_llm.data.data import Dataset
from fedot_llm.agents.automl.eval.local_exec import ExecutionResult
from typing import List


class Solution(TypedDict):
    code: str
    exec_result: ExecutionResult
    fix_tries: int = 0


class AutoMLAgentState(FedotLLMAgentState):
    description: str
    dataset: Dataset
    reflection: Optional[ProblemReflection] = None
    fedot_config: Optional[FedotConfig] = None
    skeleton: Optional[str] = None
    solutions: List[Solution] = []
    exec_result: Optional[ExecutionResult] = None
