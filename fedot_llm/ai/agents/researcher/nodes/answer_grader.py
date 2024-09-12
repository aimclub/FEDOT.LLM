import logging
from typing import Any, Optional

from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from fedot_llm.ai.agents.prebuild.nodes import ConditionalNode
from fedot_llm.ai.agents.researcher.state import GraphState

logger = logging.getLogger(__name__)


class GradeAnswer(BaseModel):
    """Binary score for hallucination check on generated answer."""

    score: str = Field(
        description="Answer is useful to resolve a question, 'yes' or 'no'."
    )


ANSWER_PROMPT = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
        Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question}
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "question"],
)


class AnswerGraderCondNode(ConditionalNode):
    def __init__(self, llm: BaseChatModel, name: str = "AnswerGrader", tags: Optional[list[str]] = None, ):
        super().__init__(name=name, tags=tags)
        self.structured_llm = llm.with_structured_output(GradeAnswer)
        self.chain = ANSWER_PROMPT | self.structured_llm.bind(temperature=0)

    def condition(self, state: GraphState) -> Any:
        """
        Determine whether the generated answer is useful to resolve a question.

        Args:
            state (dict): The current graph state
        Returns:
            state (str): "useful" if the answer is useful to resolve a question, "not useful" otherwise
        """
        question = state["question"]
        generation = state["generation"]

        score = GradeAnswer.parse_obj(self.chain.invoke(
            {"generation": generation.answer, "question": question}
        ))
        grade = score.score

        if grade == "yes":
            return "useful"
        else:
            return "not useful"
