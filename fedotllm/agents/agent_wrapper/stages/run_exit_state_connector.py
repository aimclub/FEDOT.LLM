from langchain_core.messages import HumanMessage

from fedotllm.agents.automl.automl import AutoMLAgent, AutoMLAgentState
from fedotllm.agents.base import Agent, FedotLLMAgentState
from fedotllm.agents.researcher import ResearcherAgent, ResearcherAgentState


def run_exit_state_connector(state: FedotLLMAgentState, agent: Agent):
    if isinstance(agent, ResearcherAgent):
        state: ResearcherAgentState
        return state | {
            "messages": [HumanMessage(content=state["answer"], name="ResearcherAgent")]
        }
    elif isinstance(agent, AutoMLAgent):
        state: AutoMLAgentState
        return state | {
            "messages": [
                HumanMessage(content=state["solutions"][-1]["code"], name="AutoMLAgent")
            ]
        }
