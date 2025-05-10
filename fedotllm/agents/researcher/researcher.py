from functools import partial

from langgraph.graph import END, START, StateGraph

from fedotllm.agents.base import Agent
from fedotllm.agents.researcher.nodes import (
    generate_response,
    grade_retrieve,
    is_continue,
    is_grounded,
    is_useful,
    render_answer,
    retrieve_documents,
    rewrite_question,
)
from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.llm import AIInference, OpenaiEmbeddings


class ResearcherAgent(Agent):
    def __init__(self, inference: AIInference, embeddings: OpenaiEmbeddings):
        self.inference = inference
        self.embeddings = embeddings

    def create_graph(self):
        workflow = StateGraph(ResearcherAgentState)
        workflow.add_node(
            "retrieve", partial(retrieve_documents, embeddings=self.embeddings)
        )
        workflow.add_node(
            "retrieve_grader", partial(grade_retrieve, inference=self.inference)
        )
        workflow.add_node(
            "generate", partial(generate_response, inference=self.inference)
        )
        workflow.add_node("render_answer", render_answer)
        workflow.add_node(
            "rewrite_question", partial(rewrite_question, inference=self.inference)
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
                not (
                    is_grounded(state, self.inference)
                    and is_useful(state, self.inference)
                )
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
