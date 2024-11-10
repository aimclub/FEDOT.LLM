from fedot_llm.agents.state import FedotLLMAgentState
from enum import Enum
class NextAgent(str, Enum):
    FINISH = "finish"
    RESEARCHER = "researcher"
    AUTOML = "automl"

class SupervisorState(FedotLLMAgentState):
    next: NextAgent