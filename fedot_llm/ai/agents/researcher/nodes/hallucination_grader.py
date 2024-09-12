import logging
from typing import Any, Optional

from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from fedot_llm.ai.agents.prebuild.nodes import ConditionalNode
from fedot_llm.ai.agents.researcher.state import GraphState

logger = logging.getLogger(__name__)


class GradeHallucination(BaseModel):
    """Binary score for hallucination check on generated answer."""

    score: str = Field(
        description="Answer is grounded in facts, 'yes' or 'no'."
    )


HALLUCINATION_PROMPT = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n Here 
    are the facts: \n ------- \n {documents} \n ------- \n Here is the answer: {generation} Give a binary score 'yes' 
    or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n Provide the binary 
    score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "documents"],
)


class HallucinationGraderCondNode(ConditionalNode):
    def __init__(self, llm: BaseChatModel, name: str = "HallucinationGrader", tags: Optional[list[str]] = None, ):
        self.structured_llm = llm.with_structured_output(GradeHallucination)
        self.is_hallucination_chain = HALLUCINATION_PROMPT | self.structured_llm.bind(temperature=0)
        super().__init__(name=name, tags=tags)

    def condition(self, state: GraphState) -> Any:
        """
        Determine whether the generated answer is a hallucination.

        Args:
            state (dict): The current graph state
        Returns:
            state (str): "correct" if the answer is grounded in facts, "incorrect" otherwise
        """
        documents = state["documents"]
        generation = state["generation"]

        score = self.is_hallucination_chain.invoke(
            {"documents": documents, "generation": generation.answer}
        )

        score = GradeHallucination.parse_obj(score)
        grade = score.score
        if grade == "yes":
            return "correct"
        else:
            return "incorrect"
