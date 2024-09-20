from __future__ import annotations

from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from fedot_llm.ai.agents.researcher.state import ResearcherAgentState


def is_docs_relevant(state: ResearcherAgentState):
    """
    Determine if the retrieved documents are relevant to the given question.

    Args:
        state (ResearcherAgentState): The current state of the graph containing the question and documents.

    Returns:
        str: Decision on whether to rewrite_question or generate an answer.
    """
    # logger.info("Assess graded documents")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # logger.info("Decision: all documents are not relevant to question, transform query")
        return "rewrite_question"
    else:
        # We have relevant documents, so generate answer
        # logger.info("Decision: generate")
        return "generate"


class CheckHallucinationAndAnswerEdge:
    """
    Check if the generated answer is a hallucination.

    This class provides functionality to assess whether a generated answer
    is grounded in the provided documents or if it's a hallucination.

    Attributes:
        HALLUCINATION_PROMPT (PromptTemplate): A template for prompting the LLM
            to check for hallucinations.


    Methods:
        __init__(self, llm: BaseChatModel): Initialize the edge with a language model.
        is_hallucination(self, state: GraphState) -> str: Check if the generated
            answer is a hallucination based on the provided documents.
    """

    HALLUCINATION_PROMPT = PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}
        Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "documents"],
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

    class GradeHallucination(BaseModel):
        """Binary score for hallucination check on generated answer."""

        score: str = Field(
            description="Answer is grounded in facts, 'yes' or 'no'."
        )

    class GradeAnswer(BaseModel):
        """Binary score for hallucination check on generated answer."""

        score: str = Field(
            description="Answer is useful to resolve a question, 'yes' or 'no'."
        )

    @property
    def hallucination_chain(self):
        structured_llm = self.llm.with_structured_output(self.GradeHallucination)
        return self.HALLUCINATION_PROMPT | structured_llm.bind(temperature=0)

    @property
    def answer_grader_chain(self):
        structured_llm = self.llm.with_structured_output(self.GradeAnswer)
        return self.ANSWER_PROMPT | structured_llm.bind(temperature=0)

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def is_hallucination_and_answer(self, state: ResearcherAgentState):
        """
        Determine if the generated answer is a hallucination and if it answers the question.

        Args:
            state (ResearcherAgentState): The current state of the graph containing the question and documents.

        Returns:
            str: Decision on whether to rewrite_question or generate an answer.
        """
        # logger.debug("Check hallucinations")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.GradeHallucination.model_validate(self.hallucination_chain.invoke(
            {"documents": documents, "generation": generation.answer}
        ))
        grade = score.score

        # Check hallucination
        if grade == "yes":
            # logger.debug("Decision: generation is grounded in documents")
            # Check question-answering
            # logger.debug("Grade generation vs question")
            score = self.GradeAnswer.model_validate(
                self.answer_grader_chain.invoke({"question": question, "generation": generation.answer}))
            grade = score.score
            if grade == "yes":
                # logger.debug("Decision: generation addresses question")
                return "useful"
            else:
                # logger.debug("Decision: generation does not address question")
                return "not useful"
        else:
            # logger.debug("Decision: generation is not grounded in documents, re-try")
            return "not supported"
