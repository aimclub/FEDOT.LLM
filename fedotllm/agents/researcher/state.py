from typing import List, Union

from langchain_core.documents import Document

from fedotllm.agents.researcher.structured import GenerateWithCitations
from fedotllm.agents.state import FedotLLMAgentState


class ResearcherAgentState(FedotLLMAgentState):
    """
    Represents the state of our graph.

    Attributes:
        messages: list of messages
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question: str
    generation: GenerateWithCitations
    documents: List[Union[str, Document]]
    answer: str
    attempt: int
