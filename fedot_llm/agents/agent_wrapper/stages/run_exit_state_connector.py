from fedot_llm.agents.state import FedotLLMAgentState
from fedot_llm.agents.researcher.researcher import ResearcherAgent
from fedot_llm.agents.automl.automl import AutoMLAgent
from langchain_core.messages import HumanMessage
from fedot_llm.agents.base import Agent

def run_exit_state_connector(state: FedotLLMAgentState, agent: Agent):
    if isinstance(agent, ResearcherAgent):
        return state | {'messages': [HumanMessage(content=state['answer'], name="ResearcherAgent")]}
    elif isinstance(agent, AutoMLAgent):
        return state | {'messages': [HumanMessage(content=state['solutions'][-1]['code'], name="AutoMLAgent")]}
