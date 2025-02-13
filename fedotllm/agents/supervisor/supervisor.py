from functools import partial
from typing import Optional

from langgraph.graph import END, START, StateGraph

from fedotllm.agents.agent_wrapper.agent_wrapper import AgentWrapper
from fedotllm.agents.automl.automl_chat import AutoMLAgentChat
from fedotllm.agents.base import Agent
from fedotllm.agents.researcher.researcher import ResearcherAgent
from fedotllm.agents.supervisor.stages.run_choose_next import run_choose_next
from fedotllm.agents.supervisor.state import SupervisorState
from fedotllm.data import Dataset
from fedotllm.llm.inference import AIInference, OpenaiEmbeddings


class SupervisorAgent(Agent):
    def __init__(self, embeddings: OpenaiEmbeddings, inference: AIInference, dataset: Optional[Dataset]):
        self.embeddings = embeddings
        self.inference = inference
        self.dataset = dataset
        self.researcher_agent = AgentWrapper(ResearcherAgent(
            inference=self.inference, embeddings=self.embeddings)).create_graph()
        if dataset is not None:
            self.automl_agent = AutoMLAgentChat(
                inference=self.inference, dataset=self.dataset).create_graph()
        else:
            def _automl_agent_error(*args, **kwargs):
                raise ValueError("Dataset not provided for AutoML agent.")

            self.automl_agent = _automl_agent_error

    def create_graph(self):
        workflow = StateGraph(SupervisorState)
        workflow.add_node("choose_next", partial(
            run_choose_next, inference=self.inference))
        workflow.add_node("Researcher", self.researcher_agent)
        workflow.add_node("AutoMLChat", self.automl_agent)

        workflow.add_edge(START, "choose_next")
        workflow.add_conditional_edges(
            "choose_next",
            lambda state: state["next"].value,
            {
                "finish": END,
                "researcher": "Researcher",
                "automl": "AutoMLChat"
            }
        )
        workflow.add_edge("Researcher", "choose_next")
        workflow.add_edge("AutoMLChat", "choose_next")
        return workflow.compile().with_config(config={"run_name": "SupervisorAgent"})
