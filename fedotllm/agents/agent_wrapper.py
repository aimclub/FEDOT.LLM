from functools import partial

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from fedotllm.agents.automl.automl import AutoMLAgent
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.agents.base import Agent, FedotLLMAgentState
from fedotllm.agents.researcher.researcher import ResearcherAgent
from fedotllm.agents.researcher.state import ResearcherAgentState


class AgentWrapper(Agent):
    def __init__(self, agent: Agent):
        self.agent = agent

    def create_graph(self):
        if isinstance(self.agent, ResearcherAgent):
            workflow = StateGraph(ResearcherAgentState)
        elif isinstance(self.agent, AutoMLAgent):
            workflow = StateGraph(AutoMLAgentState)
        else:
            raise ValueError("Not supported agent in AgentWrapper.")
        workflow.add_node(
            "run_entry_state_connector",
            partial(run_entry_state_connector, agent=self.agent),
        )
        workflow.add_node("agent", self.agent.create_graph())
        workflow.add_node(
            "run_exit_state_connector",
            partial(run_exit_state_connector, agent=self.agent),
        )

        workflow.add_edge(START, "run_entry_state_connector")
        workflow.add_edge("run_entry_state_connector", "agent")
        workflow.add_edge("agent", "run_exit_state_connector")
        workflow.add_edge("run_exit_state_connector", END)

        return workflow.compile()


def run_entry_state_connector(state: FedotLLMAgentState, agent: Agent):
    if isinstance(agent, ResearcherAgent):
        state: ResearcherAgentState
        state["question"] = state["messages"][-1].content
    elif isinstance(agent, AutoMLAgent):
        state: AutoMLAgentState
        state["description"] = state["messages"][-1].content
    return state


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
