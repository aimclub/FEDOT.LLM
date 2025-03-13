from typing import Optional

from fedotllm.agents.automl_multimodal.structured import ProblemReflection, FedotConfig
from fedotllm.agents.state import FedotLLMAgentState



class AutoMLMultimodalAgentState(FedotLLMAgentState):
    description: str
    dataset_splits_description: str
    reflection: Optional[ProblemReflection]
    fedot_config: Optional[FedotConfig]
    metrics: Optional[dict]
    pipeline: Optional[str]
