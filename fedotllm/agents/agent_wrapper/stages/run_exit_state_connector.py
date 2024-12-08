from langchain_core.messages import HumanMessage
from agents.state import FedotLLMAgentState
from agents.researcher.researcher import ResearcherAgent
from agents.automl.automl import AutoMLAgent
from agents.base import Agent


def run_exit_state_connector(state: FedotLLMAgentState, agent: Agent):
    if isinstance(agent, ResearcherAgent):
        return state | {'messages': [HumanMessage(content=state['answer'], name="ResearcherAgent")]}
    elif isinstance(agent, AutoMLAgent):
        return state | {'messages': [HumanMessage(content=state['solutions'][-1]['code'], name="AutoMLAgent")]}
