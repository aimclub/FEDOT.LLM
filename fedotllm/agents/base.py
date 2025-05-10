from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph


class Agent:
    def create_graph(self) -> CompiledStateGraph:
        raise NotImplementedError


class FedotLLMAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
