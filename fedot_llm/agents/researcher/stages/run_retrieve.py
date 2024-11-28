from fedot_llm.agents.researcher.state import ResearcherAgentState
from langchain.vectorstores.base import VectorStoreRetriever


def run_retrieve(state: ResearcherAgentState, retriever: VectorStoreRetriever) -> ResearcherAgentState:
    question = state["messages"][-1].content
    documents = retriever.invoke(question)
    state["documents"] = documents
    return state
