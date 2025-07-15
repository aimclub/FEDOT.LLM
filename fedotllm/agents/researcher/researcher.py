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
from fedotllm.configs.schema import AppConfig
from fedotllm.llm import AIInference, LiteLLMEmbeddings

RETRIEVE = "retrieve"
RETRIEVE_GRADER = "retrieve_grader"
GENERATE = "generate"
RENDER_ANSWER = "render_answer"
REWRITE_QUESTION = "rewrite_question"


class ResearcherAgent(Agent):
    def __init__(self, config: AppConfig):
        self.inference = AIInference(config.llm, config.session_id)
        self.embeddings = LiteLLMEmbeddings(config.embeddings)

    def create_graph(self):
        workflow = StateGraph(ResearcherAgentState)
        workflow.add_node(
            RETRIEVE, partial(retrieve_documents, embeddings=self.embeddings)
        )
        workflow.add_node(
            RETRIEVE_GRADER, partial(grade_retrieve, inference=self.inference)
        )
        workflow.add_node(
            GENERATE, partial(generate_response, inference=self.inference)
        )
        workflow.add_node(RENDER_ANSWER, render_answer)
        workflow.add_node(
            REWRITE_QUESTION, partial(rewrite_question, inference=self.inference)
        )

        workflow.add_edge(START, RETRIEVE)
        workflow.add_edge(RETRIEVE, RETRIEVE_GRADER)
        workflow.add_conditional_edges(
            RETRIEVE_GRADER,
            lambda state: not (
                len(state["retrieved"]["documents"]) == 0 and is_continue(state)
            ),
            {
                True: GENERATE,
                False: REWRITE_QUESTION,
            },
        )
        workflow.add_conditional_edges(
            GENERATE,
            lambda state: not (
                not (
                    is_grounded(state, self.inference)
                    and is_useful(state, self.inference)
                )
                and is_continue(state)
            ),
            {
                True: RENDER_ANSWER,
                False: GENERATE,
            },
        )
        workflow.add_edge(REWRITE_QUESTION, RETRIEVE)
        workflow.add_edge(RENDER_ANSWER, END)
        return workflow.compile().with_config(run_name="ResearcherAgent")
