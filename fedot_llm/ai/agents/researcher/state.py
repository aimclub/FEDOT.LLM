from typing import TypedDict, List, Union

from langchain_core.documents import Document

from fedot_llm.ai.agents.researcher.models import GenerateWithCitations


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: GenerateWithCitations
    documents: List[Union[str, Document]]
    answer: str
