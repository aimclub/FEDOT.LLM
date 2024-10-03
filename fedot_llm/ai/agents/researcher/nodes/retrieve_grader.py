from typing import Any, Optional, Callable

from langchain.chat_models.base import BaseChatModel
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate

from fedot_llm.ai.agents.prebuild.nodes import AgentNode
from fedot_llm.ai.agents.researcher.models import GradeDocuments
from fedot_llm.ai.agents.researcher.state import ResearcherAgentState

RETRIEVAL_GRADER_PROMPT = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["question", "document"],
)


class RetrievalGraderNode(AgentNode):
    def __init__(self, llm: BaseChatModel, name: str = "RetrievalGrader", tags: Optional[list[str]] = None, ):
        self.structured_llm = llm.with_structured_output(GradeDocuments)
        self.chain = RETRIEVAL_GRADER_PROMPT | self.structured_llm.bind(temperature=0)
        super().__init__(chain=self.chain, name=name, tags=tags)

    def _process(self, state: ResearcherAgentState, chain_invoke: Callable) -> Any:
        """
        Determine whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state
            
        Returns:
            state (dict): Updated documents key with only filtered relevant documents
        """
        # logger.info("Check document relevance to question")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            if not isinstance(d, Document):
                raise ValueError("Document must be an instance of Document")
            score = GradeDocuments.model_validate(chain_invoke(
                {"question": question, "document": d.page_content}
            ))
            grade = score.score
            if grade == 'yes':
                # logger.info("Grade: document relevant")
                filtered_docs.append(d)
            else:
                # logger.info("Grade: document not relevant")
                continue
        return {"documents": filtered_docs, "question": question}
