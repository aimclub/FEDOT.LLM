from typing import Any, AsyncIterator, Iterator

from langchain_core.callbacks import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.runnables import RunnableConfig, patch_config
from langchain_core.runnables.utils import Input, Output
from typing_extensions import Optional, cast


class BaseRunnableChain(Runnable[Input, Output]):
    """Base class for runnable chains."""

    chain: Runnable
    '''The chain to run.'''

    def _invoke(
            self,
            input: Input,
            run_manager: CallbackManagerForChainRun,
            config: RunnableConfig,
    ) -> Output:

        recursion_limit = config["recursion_limit"]
        if recursion_limit <= 0:
            raise RecursionError(
                f"Recursion limit reached when invoking {self} with input {input}."
            )
        output = self.chain.invoke(
            input,
            patch_config(
                config,
                callbacks=run_manager.get_child(),
                recursion_limit=recursion_limit - 1,
            ),
        )
        return cast(Output, output)

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        return self._call_with_config(
            self._invoke,
            input,
            config,
        )

    async def _ainvoke(
            self,
            input: Input,
            run_manager: AsyncCallbackManagerForChainRun,
            config: RunnableConfig,
            **kwargs: Any,
    ) -> Output:
        recursion_limit = config["recursion_limit"]
        if recursion_limit <= 0:
            raise RecursionError(
                f"Recursion limit reached when invoking {self} with input {input}."
            )
        output = await self.chain.ainvoke(
            input,
            patch_config(
                config,
                callbacks=run_manager.get_child(),
                recursion_limit=recursion_limit - 1,
            ),
        )
        return cast(Output, output)

    async def ainvoke(
            self,
            input: Input,
            config: Optional[RunnableConfig] = None,
            **kwargs: Optional[Any],
    ) -> Output:
        return await self._acall_with_config(
            self._ainvoke,
            input,
            config,
            **kwargs,
        )

    def _transform(
            self,
            input: Iterator[Input],
            run_manager: CallbackManagerForChainRun,
            config: RunnableConfig,
            **kwargs: Any,
    ) -> Iterator[Output]:
        final: Input
        got_first_val = False
        for ichunk in input:
            # By definitions, RunnableLambdas consume all input before emitting output.
            # If the input is not addable, then we'll assume that we can
            # only operate on the last chunk.
            # So we'll iterate until we get to the last chunk!
            if not got_first_val:
                final = ichunk
                got_first_val = True
            else:
                try:
                    final = final + ichunk  # type: ignore[operator]
                except TypeError:
                    final = ichunk
        recursion_limit = config["recursion_limit"]
        if recursion_limit <= 0:
            raise RecursionError(
                f"Recursion limit reached when invoking "
                f"{self} with input {final}."
            )
        for chunk in self.chain.stream(
                final,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(),
                    recursion_limit=recursion_limit - 1,
                ),
        ):
            yield chunk

    def transform(
            self,
            input: Iterator[Input],
            config: Optional[RunnableConfig] = None,
            **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        for output in self._transform_stream_with_config(
                input,
                self._transform,
                config,
                **kwargs,
        ):
            yield output

    def stream(
            self,
            input: Input,
            config: Optional[RunnableConfig] = None,
            **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        return self.transform(iter([input]), config, **kwargs)

    async def _atransform(
            self,
            input: AsyncIterator[Input],
            run_manager: AsyncCallbackManagerForChainRun,
            config: RunnableConfig,
            **kwargs: Any,
    ) -> AsyncIterator[Output]:
        final: Input
        got_first_val = False
        async for ichunk in input:
            # By definitions, RunnableLambdas consume all input
            # before emitting output.
            # If the input is not addable, then we'll assume that we can
            # only operate on the last chunk.
            # So we'll iterate until we get to the last chunk!
            if not got_first_val:
                final = ichunk
                got_first_val = True
            else:
                try:
                    final = final + ichunk  # type: ignore[operator]
                except TypeError:
                    final = ichunk
        recursion_limit = config["recursion_limit"]
        if recursion_limit <= 0:
            raise RecursionError(
                f"Recursion limit reached when invoking "
                f"{self} with input {final}."
            )
        async for chunk in self.chain.astream(
                final,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(),
                    recursion_limit=recursion_limit - 1,
                ),
        ):
            yield chunk

    async def atransform(
            self,
            input: AsyncIterator[Input],
            config: Optional[RunnableConfig] = None,
            **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        async for output in self._atransform_stream_with_config(
                input,
                self._atransform,
                config,
                **kwargs,
        ):
            yield output

    async def astream(
            self,
            input: Input,
            config: Optional[RunnableConfig] = None,
            **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        async def input_aiter() -> AsyncIterator[Input]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk


class ChainPassthrough(BaseRunnableChain):
    """Chain to passthrough inputs with additional keys.

    Args
    ----
        chain: Runnable
            The chain to run.

    Examples
    --------
    >>> ChainPassthrough(chain).invoke({"key": "value"})
    {"key": "value", "ChainClassName": "chain_output"}
    """

    def __init__(self, chain: Runnable):
        self.chain = RunnablePassthrough().assign(
            **{chain.__class__.__name__: chain})


class ChainMapToClassName(BaseRunnableChain):
    """Chain to map the output to the class name.

    Args
    ----
        chain: Runnable
            The chain to run.

    Examples
    --------
    >>> ChainMapToClassName(chain).invoke({"key": "value"})
    {"ChainClassName": "chain_output"}
    """

    def __init__(self, chain: Runnable):
        self.chain = RunnableParallel({chain.__class__.__name__: chain})


class ChainAddKey(BaseRunnableChain):
    """Chain to add a key to the input.

    Args
    ----
        key: str
            The key to add.
        chain: Runnable
            The chain to run.

    Examples
    --------
    >>> ChainAddKey("new_key", chain).invoke({"key": "value"})
    {"key": "value", "new_key": "chain_output"}
    """

    def __init__(self, key: str, chain: Runnable):
        self.chain = RunnablePassthrough().assign(**{key: chain})


class ChainAddStrKey(ChainAddKey):
    """Chain to add a string key to the input.

    Args
    ----
        key: str
            The key to add.
        value: str
            The value to add.

    Examples
    --------
    >>> ChainAddStrKey("new_key", "value").invoke({"key": "value"})
    {"key": "value", "new_key": "value"}
    """

    def __init__(self, key: str, value: str):
        super().__init__(key, RunnableLambda(lambda _: value))
