from fedotllm.llm.inference import OpenaiEmbeddings
from fedotllm.agents.retrieve import RetrieveTool

from fedotllm.agents.researcher.state import ResearcherAgentState


def run_retrieve(state: ResearcherAgentState, embeddings: OpenaiEmbeddings) -> ResearcherAgentState:
    retriever = RetrieveTool(embeddings=embeddings)
    if retriever.count() == 0:
        retriever.create_db_docs()
    state['retrieved'] = retriever.query_docs(state["question"])
    return state
