from langchain.vectorstores.base import VectorStoreRetriever

from fedotllm.agents.researcher.state import ResearcherAgentState


def run_retrieve(state: ResearcherAgentState, retriever: VectorStoreRetriever) -> ResearcherAgentState:
    question = state["messages"][-1].content
    documents = retriever.invoke(question)
    state["documents"] = documents
    return state
