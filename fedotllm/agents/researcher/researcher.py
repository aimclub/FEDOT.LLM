from functools import partial

from langgraph.graph import END, START, StateGraph
from omegaconf import DictConfig

from fedotllm.agents.base import Agent
from fedotllm.agents.researcher.stages import (
    is_continue,
    is_grounded,
    is_useful,
    run_generate,
    run_render_answer,
    run_retrieve,
    run_retrieve_grader,
    run_rewrite_question,
)
from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.llm import LiteLLMModel, OpenaiEmbeddings
from fedotllm.utils import unpack_omega_config


class ResearcherAgent(Agent):
    def __init__(
        self, config: DictConfig, embeddings: OpenaiEmbeddings, session_id: str
    ):
        self.config = config
        self.llm = LiteLLMModel(
            **unpack_omega_config(config.llm), session_id=session_id
        )
        self.embeddings = embeddings

    def create_graph(self):
        workflow = StateGraph(ResearcherAgentState)
        workflow.add_node("retrieve", partial(run_retrieve, embeddings=self.embeddings))
        workflow.add_node("retrieve_grader", partial(run_retrieve_grader, llm=self.llm))
        workflow.add_node("generate", partial(run_generate, llm=self.llm))
        workflow.add_node("render_answer", run_render_answer)
        workflow.add_node(
            "rewrite_question", partial(run_rewrite_question, llm=self.llm)
        )

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "retrieve_grader")
        workflow.add_conditional_edges(
            "retrieve_grader",
            lambda state: not (
                len(state["retrieved"]["documents"]) == 0 and is_continue(state)
            ),
            {
                True: "generate",
                False: "rewrite_question",
            },
        )
        workflow.add_conditional_edges(
            "generate",
            lambda state: not (
                not (is_grounded(state, self.llm) and is_useful(state, self.llm))
                and is_continue(state)
            ),
            {
                True: "render_answer",
                False: "generate",
            },
        )
        workflow.add_edge("rewrite_question", "retrieve")
        workflow.add_edge("render_answer", END)
        return workflow.compile().with_config(config={"run_name": "ResearcherAgent"})
