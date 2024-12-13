from typing import List

from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.agents.researcher.structured import Citation


def run_render_answer(state: ResearcherAgentState):
    generation = state["generation"]
    answer = generation.answer
    citations: List[Citation] = generation.citations
    for citation in citations:
        answer = answer.replace(
            f"[{citation.number}]", f"[\\[{citation.number}\\]]({citation.url})")

    state['answer'] = answer
    return state
