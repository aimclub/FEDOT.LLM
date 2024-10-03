from typing import Any, Optional, Callable

from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import PromptTemplate

from fedot_llm.ai.agents.prebuild.nodes import AgentNode
from fedot_llm.ai.agents.researcher.models import RewriteQuestion
from fedot_llm.ai.agents.researcher.state import ResearcherAgentState

RE_WRITE_PROMPT = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized \n 
    for vectorstore retrieval. Look at the initial and formulate an improved question. \n
    Here is the initial question: \n\n {question}. Provide improved question as a JSON with a single key
    'question' and no preamble or explanation.: \n """,
    input_variables=["question"],
)


class RewriteQuestionNode(AgentNode):
    def __init__(self, llm: BaseChatModel, name: str = "RewriteQuestion", tags: Optional[list[str]] = None, ):
        self.structured_llm = llm.with_structured_output(RewriteQuestion).bind(temperature=0).with_retry()
        self.chain = RE_WRITE_PROMPT | self.structured_llm
        super().__init__(chain=self.chain, name=name, tags=tags)

    def _process(self, state: ResearcherAgentState, chain_invoke: Callable) -> Any:
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state
        Returns:
            state (dict): Updated question key with a re-phrased question
        """
        # logger.info("Transform query")
        question = state["question"]
        documents = state['documents']

        # Re-write question
        better_question = RewriteQuestion.model_validate(chain_invoke({"question": question})).question
        return {"documents": documents, "question": better_question}
