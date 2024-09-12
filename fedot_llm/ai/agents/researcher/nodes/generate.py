import logging
from typing import Any, Optional
from typing import Callable

from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from fedot_llm.ai.agents.prebuild.nodes import AgentNode
from fedot_llm.ai.agents.researcher.models import GenerateWithCitations
from fedot_llm.ai.agents.researcher.state import GraphState

logger = logging.getLogger(__name__)

GENERATE_PROMPT = ChatPromptTemplate([
    ('system', """You are DocBot, a helpful assistant that is an expert at helping users with the documentation. \n
    Here is the relevant documentation: \n
    <documentation>
    {context}
    </documentation>
    If you don't know the answer, just say that you don't know. Keep the answer concise. \n
    When a user asks a question, perform the following tasks:
    1. Find the quotes from the documentation that are the most relevant to answering the question. These quotes can be quite long if necessary (even multiple paragraphs). You may need to use many quotes to answer a single question, including code snippits and other examples.
    2. Assign numbers to these quotes in the order they were found. Each page of the documentation should only be assigned a number once.
    3. Based on the document and quotes, answer the question. Directly quote the documentation when possible, including examples. When relevant, code examples are preferred.
    4. When answering the question provide citations references in square brackets containing the number generated in step 2 (the number the citation was found)
    5. Structure the output
    Example output:
    {{
        "citations": [
                {{
                    "page_title": "FEDOT 0.7.4 documentation",
                    "url": "https://fedot.readthedocs.io/en/latest",
                    "number": 1,
                    "relevant_passages": [
                            "This example explains how to solve regression task using Fedot.",
                        ]
                }}
            ],
        "answer": "The answer to the question."
    }}
    """),
    ('human', "{question}"),
],
    template_format='f-string'
)


class GenerateNode(AgentNode):
    """
    A node in the graph that generates answers based on input questions and documents.

    This class extends RunnableCallable and is responsible for generating responses
    using a language model, based on the provided question and relevant documents.

    Attributes:
        structured_llm (BaseLanguageModel): A language model with structured output.
        chain (Chain): A chain of operations including the prompt and the language model.

    Methods:
        _func: Synchronously generate an answer.
        _afunc: Asynchronously generate an answer.
    """

    def __init__(self, llm: BaseChatModel, name: str = "Generate", tags: Optional[list[str]] = None, ):
        """
        Initialize the GenerateNode with a language model and set up the chain.

        Args:
            llm (BaseLanguageModel): The language model to be used for generating responses.
            name (str): The name of the node.
            tags (Optional[list[str]]): A list of tags for the node.
        """
        self.structured_llm = llm.with_structured_output(GenerateWithCitations)
        self.chain = GENERATE_PROMPT | self.structured_llm.bind(
            temperature=0).with_retry()
        super().__init__(chain=self.chain, name=name, tags=tags)

    def _process(self, state: GraphState, chain_invoke: Callable) -> Any:
        logger.info("Generate answer")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = chain_invoke({"context": documents, "question": question})
        generation = GenerateWithCitations.parse_obj(generation)
        return {"documents": documents, "question": question, "generation": generation}
