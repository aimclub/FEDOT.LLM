from typing import Annotated, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnablePick
from langchain_core.tools import BaseTool
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from fedot_llm.ai.agents.load import load_fedot_docs
from fedot_llm.ai.agents.researcher.edges import is_docs_relevant
from fedot_llm.ai.agents.researcher.nodes.answer_grader import AnswerGraderCondNode
from fedot_llm.ai.agents.researcher.nodes.generate import GenerateNode
from fedot_llm.ai.agents.researcher.nodes.hallucination_grader import (
    HallucinationGraderCondNode,
)
from fedot_llm.ai.agents.researcher.nodes.render_answer import RenderAnswerNode
from fedot_llm.ai.agents.researcher.nodes.retrieve import RetrieveNode
from fedot_llm.ai.agents.researcher.nodes.retrieve_grader import RetrievalGraderNode
from fedot_llm.ai.agents.researcher.nodes.rewrite_question import RewriteQuestionNode
from fedot_llm.ai.agents.researcher.nodes.should_continue import ShouldContinueCondNode
from fedot_llm.ai.agents.researcher.state import GraphState
from fedot_llm.ai.memory import LongTermMemory


class ResearcherAgent:
    def __init__(self, llm: BaseChatModel, memory: LongTermMemory):
        self.llm = llm
        self.memory = memory
        self.as_graph = self.create_graph()

    def run(self, question: str):
        return (self.as_graph | RunnablePick("answer")).invoke({"question": question})

    @property
    def as_tool(self):
        """Research answer to the question about Fedot AutoML Framework
        """

        @tool("ResearcherAgent")
        def research(question: Annotated[str, "Question to research"]):
            """Research answer to the question about Fedot AutoML Framework"""
            return self.run(question)

        return research

    def create_graph(self):
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("retrieve",
                          RetrieveNode(retriever=self.memory.get_collection("docs").get_retriever()))
        workflow.add_node("retrieve_grader", RetrievalGraderNode(llm=self.llm))
        workflow.add_node("generate", GenerateNode(llm=self.llm))
        answer_grader = AnswerGraderCondNode(llm=self.llm)
        workflow.add_node("answer_grader", answer_grader)
        rewrite_question = RewriteQuestionNode(llm=self.llm)
        workflow.add_node("rewrite_question", rewrite_question)
        should_continue = ShouldContinueCondNode()
        workflow.add_node("should_continue", should_continue)
        workflow.add_node("render_answer", RenderAnswerNode(llm=self.llm))

        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "retrieve_grader")
        workflow.add_conditional_edges(
            "retrieve_grader",
            is_docs_relevant,
            {
                "rewrite_question": "should_continue",
                "generate": "generate",
            },
        )
        workflow.add_conditional_edges(
            "generate",
            HallucinationGraderCondNode(llm=self.llm).condition,
            {
                "incorrect": "generate",
                "correct": "answer_grader",
            },
        )
        workflow.add_conditional_edges(
            "answer_grader",
            answer_grader.condition,
            {
                "useful": "render_answer",
                "not useful": "should_continue",
            },
        )
        workflow.add_edge("render_answer", END)
        workflow.add_conditional_edges(
            "should_continue",
            should_continue.condition,
            {
                "yes": "rewrite_question",
                "no": END,
            },
        )
        workflow.add_edge("rewrite_question", "retrieve")

        app = workflow.compile()
        return app
