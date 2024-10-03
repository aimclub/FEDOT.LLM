from typing import Any, Callable

from langchain.vectorstores.base import VectorStoreRetriever

from fedot_llm.ai.agents.base import extract_calls
from fedot_llm.ai.agents.prebuild.nodes import AgentNode
from fedot_llm.ai.agents.researcher.state import ResearcherAgentState


class RetrieveNode(AgentNode):
    def __init__(self, retriever: VectorStoreRetriever):
        self.chain = retriever
        super().__init__(chain=self.chain, name="RetrieveNode")

    def _process(self, state: ResearcherAgentState, chain_invoke: Callable) -> Any:
        """
        Retrieve documents

        Args:
            state (dict): The current graph state
            
        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        # logger.debug("Retrieve documents")
        if calls := extract_calls(state):
            if (question := calls.get("ResearcherAgent", {}).get("args", {}).get("question")) is None:
                raise ValueError("Question is not provided in the RetrieverAgent")
        else:
            raise ValueError("This agent wasn't called.")

        # Retrieval
        documents = chain_invoke(question)
        return {"documents": documents, "question": question}
