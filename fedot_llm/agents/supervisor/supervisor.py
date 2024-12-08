from functools import partial

from langgraph.graph import END, START, StateGraph

from fedot_llm.agents.agent_wrapper.agent_wrapper import AgentWrapper
from fedot_llm.agents.automl.automl import AutoMLAgent
from fedot_llm.agents.base import Agent
from fedot_llm.agents.memory import LongTermMemory
from fedot_llm.agents.researcher.researcher import ResearcherAgent
from fedot_llm.agents.supervisor.stages.run_choose_next import run_choose_next
from fedot_llm.agents.supervisor.state import SupervisorState
from fedot_llm.llm.inference import AIInference


class SupervisorAgent(Agent):
    def __init__(self, memory: LongTermMemory, inference: AIInference):
        self.memory = memory
        self.inference = inference
        self.researcher_agent = AgentWrapper(ResearcherAgent(
            inference=self.inference, memory=self.memory))
        self.automl_agent = AgentWrapper(AutoMLAgent(inference=self.inference))

    def create_graph(self):
        workflow = StateGraph(SupervisorState)
        workflow.add_node("choose_next", partial(run_choose_next, inference=self.inference))
        workflow.add_node("researcher", self.researcher_agent.create_graph())
        workflow.add_node("automl", self.automl_agent.create_graph())

        workflow.add_edge(START, "choose_next")
        workflow.add_conditional_edges(
            "choose_next",
            lambda state: state["next"].value,
            {
                "finish": END,
                "researcher": "researcher",
                "automl": "automl"
            }
        )
        workflow.add_edge("researcher", "choose_next")
        workflow.add_edge("automl", "choose_next")
        return workflow.compile()
