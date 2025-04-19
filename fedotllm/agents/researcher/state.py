from chromadb.api.types import QueryResult

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
    retrieved: QueryResult
    answer: str
    attempt: int
