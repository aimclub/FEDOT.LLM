from functools import partial

from langgraph.graph import END, START, StateGraph
from omegaconf import DictConfig

from fedotllm.agents.automl.automl_chat import AutoMLAgentChat
from fedotllm.agents.base import Agent
from fedotllm.agents.researcher.researcher import ResearcherAgent
from fedotllm.agents.supervisor.stages.run_choose_next import run_choose_next
from fedotllm.agents.supervisor.state import SupervisorState
from fedotllm.llm import LiteLLMModel, OpenaiEmbeddings
from fedotllm.utils import unpack_omega_config


class SupervisorAgent(Agent):
    def __init__(
        self,
        config: DictConfig,
        embeddings: OpenaiEmbeddings,
        session_id: str,
        researcher_agent: ResearcherAgent,
        automl_agent: AutoMLAgentChat,
    ):
        self.config = config
        self.embeddings = embeddings
        self.session_id = session_id
        self.llm = LiteLLMModel(
            **unpack_omega_config(config.llm), session_id=session_id
        )
        self.researcher_agent = researcher_agent
        self.automl_agent = automl_agent

    def create_graph(self):
        workflow = StateGraph(SupervisorState)
        workflow.add_node("choose_next", partial(run_choose_next, llm=self.llm))
        workflow.add_node("Researcher", self.researcher_agent)
        workflow.add_node("AutoMLChat", self.automl_agent)

        workflow.add_edge(START, "choose_next")
        workflow.add_conditional_edges(
            "choose_next",
            lambda state: state["next"].value,
            {"finish": END, "researcher": "Researcher", "automl": "AutoMLChat"},
        )
        workflow.add_edge("Researcher", "choose_next")
        workflow.add_edge(START, "AutoMLChat")
        workflow.add_edge("AutoMLChat", END)
        return workflow.compile().with_config(config={"run_name": "SupervisorAgent"})
