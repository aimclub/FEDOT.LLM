from fedotllm.agents.automl.automl import AutoMLAgent
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.agents.base import Agent, FedotLLMAgentState
from fedotllm.agents.researcher.researcher import ResearcherAgent
from fedotllm.agents.researcher.state import ResearcherAgentState


def run_entry_state_connector(state: FedotLLMAgentState, agent: Agent):
    if isinstance(agent, ResearcherAgent):
        state: ResearcherAgentState
        state["question"] = state["messages"][-1].content
    elif isinstance(agent, AutoMLAgent):
        state: AutoMLAgentState
        state["description"] = state["messages"][-1].content
    return state
