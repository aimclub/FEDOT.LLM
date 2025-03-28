from pathlib import Path
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class FedotLLMAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    work_dir: Optional[Path]
