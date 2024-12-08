from fedot_llm.agents.state import FedotLLMAgentState
from fedot_llm.agents.researcher.researcher import ResearcherAgent
from fedot_llm.agents.automl.automl import AutoMLAgent
from fedot_llm.agents.base import Agent

def run_entry_state_connector(state: FedotLLMAgentState, agent: Agent):
    if isinstance(agent, ResearcherAgent):
        state['question'] = state['messages'][-1].content
    elif isinstance(agent, AutoMLAgent):
        state['description'] = state['messages'][-1].content
    return state
