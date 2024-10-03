from typing import Optional, Any

from langchain_core.messages import HumanMessage

from fedot_llm.ai.agents.prebuild.nodes import ConditionalNode
from fedot_llm.ai.agents.researcher.state import ResearcherAgentState


class ShouldContinueCondNode(ConditionalNode):
    def __init__(self, max_iterations: int = 3, name: str = "ShouldContinue", tags: Optional[list[str]] = None, ):
        self.max_iterations = max_iterations
        self.current_iteration = 0
        super().__init__(name=name, tags=tags)

    def condition(self, state: ResearcherAgentState) -> Any:
        """
        Determine whether to continue iteration.

        Args:
            state (dict): The current graph state
        Returns:
            state (str): "yes" or "no"
        """
        if self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            return "yes"
        else:
            answer = "Uh, I'm sorry. I don't know the answer to your question. "
            state["messages"].append(HumanMessage(content=answer, name="ResearcherAgent"))
            return "no"
