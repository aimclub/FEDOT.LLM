from fedotllm.enviroments.types import CodeObservation
from fedotllm.agents.automl.structured import ProblemReflection, FedotConfig
from fedotllm.agents.base import FedotLLMAgentState


class AutoMLAgentState(FedotLLMAgentState):
    description: str
    reflection: ProblemReflection | None
    fedot_config: FedotConfig | None
    skeleton: str | None
    raw_code: str | None
    code: str | None
    code_observation: CodeObservation | None
    fix_attempts: int
    metrics: str
    pipeline: str
    report: str
