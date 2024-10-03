from typing import List, Union

from langchain_core.documents import Document
from langgraph.graph import MessagesState

from fedot_llm.ai.agents.researcher.models import GenerateWithCitations


class ResearcherAgentState(MessagesState):
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
