from typing import Any, Optional, Callable
from typing import List

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import HumanMessage

from fedot_llm.ai.agents.prebuild.nodes import AgentNode
from fedot_llm.ai.agents.researcher.models import Citation
from fedot_llm.ai.agents.researcher.state import ResearcherAgentState


class RenderAnswerNode(AgentNode):
    def __init__(self, llm: BaseChatModel, name: str = "RenderAnswer", tags: Optional[list[str]] = None, ):
        self.chain = llm.bind(temperature=0)
        super().__init__(chain=self.chain, name=name, tags=tags)

    def _process(self, state: ResearcherAgentState, chain_invoke: Callable) -> Any:
        """
        Render the answer.
        """
        generation = state["generation"]
        answer = generation.answer
        citations: List[Citation] = generation.citations
        for citation in citations:
            answer = answer.replace(f"[{citation.number}]", f"[\\[{citation.number}\\]]({citation.url})")

        return state | {'messages': [HumanMessage(content=answer, name="ResearcherAgent")]}
