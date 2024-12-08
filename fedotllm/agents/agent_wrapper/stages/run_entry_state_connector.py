from agents.state import FedotLLMAgentState
from agents.researcher.researcher import ResearcherAgent
from agents.automl.automl import AutoMLAgent
from agents.base import Agent


def run_entry_state_connector(state: FedotLLMAgentState, agent: Agent):
    if isinstance(agent, ResearcherAgent):
        state['question'] = state['messages'][-1].content
    elif isinstance(agent, AutoMLAgent):
        state['description'] = state['messages'][-1].content
    return state
