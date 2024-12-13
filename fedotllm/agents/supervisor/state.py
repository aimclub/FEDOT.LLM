from enum import Enum

from fedotllm.agents.state import FedotLLMAgentState


class NextAgent(str, Enum):
    FINISH = "finish"
    RESEARCHER = "researcher"
    AUTOML = "automl"


class SupervisorState(FedotLLMAgentState):
    next: NextAgent
