from fedotllm.enviroments.types import CodeObservation
from fedotllm.agents.automl.structured import FedotConfig
from fedotllm.agents.base import FedotLLMAgentState


class AutoMLAgentState(FedotLLMAgentState):
    description: str
    reflection: str
    fedot_config: FedotConfig
    skeleton: str
    raw_code: str | None
    code: str | None
    code_observation: CodeObservation | None
    fix_attempts: int
    metrics: str
    pipeline: str
    report: str
