from langchain_core.messages import HumanMessage

from fedotllm.agents.automl.state import AutoMLAgentState


def run_send_message(state: AutoMLAgentState):
    if state["solutions"][-1]["code"] is None:
        state["messages"] = [
            HumanMessage(
                content="Solution not found. Please try again.", name="AutoMLAgent"
            )
        ]
    else:
        state["messages"] = [HumanMessage(content=state["report"], name="AutoMLAgent")]
    return state
