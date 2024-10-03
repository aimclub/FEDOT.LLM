from abc import ABC, abstractmethod
from typing import Optional, Callable, Any

from langchain_core.runnables import Runnable
from langgraph.utils.runnable import RunnableCallable


class AgentNode(RunnableCallable, ABC):
    def __init__(self, chain: Runnable, name: str, tags: Optional[list[str]] = None, ):
        self.chain = chain
        super().__init__(func=self._func, name=name, tags=tags, trace=False)

    def _func(self, input: Any) -> Any:
        return self._process(input, lambda input: self.chain.invoke(input))

    # async def _async_invoke(self, input: Any) -> Any:
    #     return await self.chain.ainvoke(input)

    # async def _afunc(self, input: Any) -> Any:
    #     return self._process(input, self._async_invoke)

    @abstractmethod
    def _process(self, state: Any, chain_invoke: Callable) -> Any:
        pass


class PassthroughNode(RunnableCallable):
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

    def __init__(self, name: str = "Passthrough", tags: Optional[list[str]] = None, ):
        """
        Initialize the GenerateNode with a language model and set up the chain.

        Args:
            llm (BaseLanguageModel): The language model to be used for generating responses.
            name (str): The name of the node.
            tags (Optional[list[str]]): A list of tags for the node.
        """
        super().__init__(func=lambda state: state, afunc=lambda state: state, name=name, tags=tags, trace=False)


class ConditionalNode(AgentNode, ABC):
    def __init__(self, chain: Runnable = lambda state: state, name: str = "Conditional",
                 tags: Optional[list[str]] = None, ):
        super().__init__(chain=chain, name=name, tags=tags)

    def _process(self, state: Any, chain_invoke: Callable) -> Any:
        return state

    @abstractmethod
    def condition(self, state: Any) -> Any:
        pass
