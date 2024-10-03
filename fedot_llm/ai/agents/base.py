import yaml
from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState


def extract_calls(state: MessagesState):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage):
        if isinstance(last_message.content, str):
            if last_message.content.startswith("calls:"):
                return yaml.safe_load(last_message.content).get("calls", None)
