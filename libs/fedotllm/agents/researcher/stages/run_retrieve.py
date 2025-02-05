from fedotllm.llm.inference import OpenaiEmbeddings
from fedotllm.agents.retrieve import RetrieveTool

from fedotllm.agents.researcher.state import ResearcherAgentState


def run_retrieve(state: ResearcherAgentState, embeddings: OpenaiEmbeddings) -> ResearcherAgentState:
    question = state["messages"][-1].content
    retriever = RetrieveTool(embeddings=embeddings)
    if retriever.count == 0:
        retriever.create_db_docs()
    state['retrieved'] = retriever.query_docs(question)
    return state
