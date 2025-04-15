from typing import List

from litellm.types.utils import ChatCompletionMessageToolCall

from fedotllm.agents.state import FedotLLMAgentState


class DataAnalystAgentState(FedotLLMAgentState):
    tool_calls: List[ChatCompletionMessageToolCall]
    problem_description: str
