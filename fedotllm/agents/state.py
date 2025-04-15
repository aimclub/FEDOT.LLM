from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class FedotLLMAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    workspace: Annotated[Path, "workspace"]
    task_path: Annotated[Path, "task_path"]
