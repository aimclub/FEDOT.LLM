from __future__ import annotations

import copy
import json
import logging
import os
import pprint
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar, Union, overload

import instructor
import litellm
import litellm.types.utils
from instructor.dsl.partial import Partial
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from fedotllm.exceptions import ContextWindowExceededError

T = TypeVar("T", bound=Union[BaseModel, "Iterable[Any]", "Partial[Any]"])

logger = logging.getLogger(__name__)
_MAX_RETRUES = 0
_MAX_INPUT_TOKENS_DEFAULT = 8000
_MAX_OUTPUT_TOKENS_DEFAULT = 4000

# Disable litellm's verbose logging
os.environ["LITELLM_LOG"] = os.getenv("LITELLM_LOG", "WARNING")
litellm.set_verbose = False

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")

if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]


def _get_non_retryable_exceptions() -> tuple:
    """Get the tuple of exceptions that should not be retried.

    Returns:
        tuple: A tuple of exception classes that should not trigger a retry.
    """
    base_exceptions = [
        RuntimeError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.APIError,
    ]

    # Conditionally add UnsupportedParamsError if available
    if hasattr(litellm.exceptions, "UnsupportedParamsError"):
        base_exceptions.append(litellm.exceptions.UnsupportedParamsError)

    return tuple(base_exceptions)


def trim_messages(
    messages: list[dict],
    model: str = "openai/gpt-4o",
    max_tokens: Optional[int] = None,
    trim_ratio: float = 0.75,
) -> list[dict]:
    messages = copy.deepcopy(messages)
    if max_tokens is None:
        if model in litellm.model_cost:
            max_tokens_for_model = litellm.model_cost[model].get(
                "max_input_tokens", litellm.model_cost[model]["max_tokens"]
            )
            max_tokens = int(max_tokens_for_model * trim_ratio)
        else:
            # if user did not specify max (input) tokens
            # or passed an llm litellm does not know
            # do nothing, just return messages
            return messages
    max_tokens = int(trim_ratio * max_tokens)
    system_messages = []
    for message in messages:
        if message["role"] == "system":
            system_messages.append(message)
        else:
            break
    messages = messages[len(system_messages) :]
    first_user_messages = []
    for message in messages:
        if message["role"] == "user":
            first_user_messages.append(message)
        else:
            break
    messages = messages[len(first_user_messages) :]
    start_messages = system_messages + first_user_messages
    start_tokens = litellm.utils.token_counter(model=model, messages=start_messages)
    if start_tokens > max_tokens:
        logger.warning(
            f"System and user messages tokens {start_tokens} are greater than max tokens {max_tokens}"
        )
        start_messages = system_messages
        start_tokens = litellm.utils.token_counter(model=model, messages=start_messages)
        if start_tokens > max_tokens:
            logger.warning(
                f"System messages tokens {start_tokens} are greater than max tokens {max_tokens}"
            )
            return messages
    last_tool_call: dict[str, Any] = {}
    current_tokens = start_tokens + litellm.utils.token_counter(
        model=model, messages=messages
    )
    while current_tokens > max_tokens:
        removing_messages = []
        removing_messages.append(messages.pop(0))
        if removing_messages[0]["role"] != "tool":
            removing_messages.append(last_tool_call)
            last_tool_call = {}
        if removing_messages[0].get("tool_calls"):
            removing_messages.pop(0)
            last_tool_call = removing_messages[0]
        else:
            current_tokens -= litellm.utils.token_counter(
                model=model, messages=removing_messages
            )
    if messages and messages[0]["role"] == "tool":
        messages = start_messages + messages + [last_tool_call]
    else:
        messages = start_messages + messages
    logger.info(f"Current tokens {current_tokens}")
    return messages


class APIStats(BaseModel):
    model: str
    base_url: str | None = None
    total_cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    history: list = Field(default_factory=list)

    def __add__(self, other: APIStats) -> APIStats:
        if not isinstance(other, APIStats):
            raise TypeError(
                f"Can only add APIStats with APIStats, got type {type(other)}"
            )
        return APIStats(
            **{
                field_name: getattr(self, field_name) + getattr(other, field_name)
                for field_name in self.model_fields.keys()
            }
        )

    def __str__(self):
        return (
            f"Model: {self.model}\n"
            f"Base URL: {self.base_url}\n"
            f"Total cost: {self.total_cost:.2f}\n"
            f"Input tokens: {self.input_tokens:,}\n"
            f"Output tokens: {self.output_tokens:,}\n"
            f"API calls: {self.api_calls:,}\n"
        )


class LiteLLMModel:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        register_model: Optional[Dict[str, Any]] = None,
        session_id: str | None = None,
        **kwargs,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.top_p = top_p

        if register_model is not None:
            litellm.utils.register_model(register_model)

        if api_key:
            self.api_key = api_key
        elif "OPENAI_API_KEY" in os.environ:
            self.api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise Exception("OpenAI API env variable OPENAI_API_KEY not set")

        self.stats = APIStats(model=self.model, base_url=self.base_url)

        self.model_max_input_tokens = litellm.model_cost.get(self.model, {}).get(
            "max_input_tokens", _MAX_INPUT_TOKENS_DEFAULT
        )
        self.model_max_output_tokens = litellm.model_cost.get(self.model, {}).get(
            "max_output_tokens", _MAX_OUTPUT_TOKENS_DEFAULT
        )
        self.provider = litellm.model_cost.get(self.model, {}).get("litellm_provider")
        if self.provider is None and self.base_url is not None:
            logger.warning(
                f"Using a custom API base: {self.base_url}. "
                "Cost managment and context length error checking will not work"
            )
        if session_id is None:
            session_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4()}"

        self.completion_kwargs = {
            "model": self.model,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "drop_params": True,
            "metadata": {
                "session_id": session_id,
            },
            **kwargs,
        }

        instructor_mode = (
            instructor.Mode.TOOLS
            if litellm.supports_function_calling(self.model)
            else instructor.Mode.MD_JSON
        )
        self.instructor_client = instructor.from_litellm(
            litellm.completion, mode=instructor_mode
        )

    def __repr__(self):
        return f"LiteLLMModel(model={self.model}, base_url={self.base_url})"

    @overload
    def create(
        self,
        response_model: type[T],
        messages: List[Dict[str, str]] | str,
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> T: ...

    @overload
    def create(
        self,
        response_model: None,
        messages: List[Dict[str, str]] | str,
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> Any: ...

    def create(
        self,
        response_model: type[T] | None,
        messages: List[Dict[str, str]] | str,
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> T | Any:
        """
        Calls the instructor client's create method with pre-configured arguments.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        merged_kwargs = {
            **{k: v for k, v in self.completion_kwargs.items() if v is not None},
            **kwargs,
        }
        model, raw_response = self.instructor_client.create_with_completion(
            response_model=response_model,
            messages=messages,
            max_retries=max_retries,
            validation_context=validation_context,
            context=context,
            strict=strict,
            **merged_kwargs,
        )
        # May be incorrect in counting input tokens
        #self.update_stats(messages, raw_response)
        return model

    def update_stats(
        self,
        input: List[Dict[str, str]],
        output: litellm.types.utils.ModelResponse,
        tools: Optional[List] = None,
    ) -> float:
        cost = litellm.cost_calculator.completion_cost(
            model=self.model, completion_response=output
        )
        self.stats.total_cost += cost
        if output.usage is not None:  # type: ignore
            try:
                self.stats.input_tokens += getattr(output.usage, "prompt_tokens", 0)
                self.stats.output_tokens += getattr(
                    output.usage, "completion_tokens", 0
                )
            except (AttributeError, TypeError):
                # Fallback to token counting if usage object doesn't have expected attributes
                self.stats.input_tokens += litellm.utils.token_counter(
                    model=self.model, messages=input, tools=tools
                )
                assert isinstance(output.choices[0], litellm.types.utils.Choices), (
                    f"LiteLLMModel not support {type(output.choices[0])}."
                )
                self.stats.output_tokens += litellm.utils.token_counter(
                    model=self.model, text=output.choices[0].message.content
                )
        else:
            self.stats.input_tokens += litellm.utils.token_counter(
                model=self.model, messages=input, tools=tools
            )
            assert isinstance(output.choices[0], litellm.types.utils.Choices), (
                f"LiteLLMModel not support {type(output.choices[0])}."
            )
            self.stats.output_tokens += litellm.utils.token_counter(
                model=self.model, text=output.choices[0].message.content
            )
        self.stats.api_calls += 1

        try:
            input_tokens = getattr(output.usage, "prompt_tokens", 0)
            output_tokens = getattr(output.usage, "completion_tokens", 0)
        except (AttributeError, TypeError):
            input_tokens = litellm.utils.token_counter(
                model=self.model, messages=input, tools=tools
            )
            output_tokens = litellm.utils.token_counter(
                model=self.model, text=output.choices[0].message.content
            )

        self.stats.history.append(
            {
                "cost": cost,
                "input": input,
                "output": pprint.pformat(output.model_dump_json(), indent=4),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )
        return cost

    def _generate_prompt_with_tools(
        self, tools: List, tool_choice: Optional[str] = None
    ) -> str:
        tool_description = []
        match tool_choice:
            case "any" | True | "required":
                tool_choice_mode = "required"
            case "auto" | None:
                tool_choice_mode = "auto"
            case _:
                tool_choice_mode = f"<<{tool_choice}>>"
        for tool in tools:
            match tool:
                case dict():
                    tool = tool["function"]
                    tool_description.append(
                        f"Funtion name: {tool['name']}\n"
                        f"Descripiton: {tool['description']}\n"
                        f"Parameters: {json.dumps(tool['parameters'], ensure_ascii=False)}"
                    )
                case _:
                    raise ValueError(
                        "Unsupported tool type. Try using a dictionary as tool"
                    )
        tool_prefix = "\n\nYou have access to the following functions:\n\n"
        tool_instruction = (
            "There are the following 4 function call options:\n"
            "- str of the form <<tool_name>>: call <<tool_name>> tool.\n"
            "- 'auto': automatically select a tool (including no tool).\n"
            "- 'required': at least one tool have to be called.\n\n"
            f"User-selected option - {tool_choice_mode}\n\n"
            "If you choose to call a function ONLY reply in the following format with no prefix or suffix:\n"
            '<function=example_function_name>{"example_parameter_name": "example_parameter_value"}</function>'
        )
        return tool_prefix + "\n\n".join(tool_description) + "\n\n" + tool_instruction

    def _parser_function_calls(
        self, content: str
    ) -> List[litellm.types.utils.ChatCompletionMessageToolCall]:
        tool_calls: List[litellm.types.utils.ChatCompletionMessageToolCall] = []
        pattern = r"<function=(.*?)>(.*?)</function>"
        matches = re.findall(pattern, content, re.DOTALL)

        for match in matches:
            function_name, function_args = match
            try:
                arguments = json.loads(function_args)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error when decoding function arguments: {e}")

            tool_call = litellm.types.utils.ChatCompletionMessageToolCall(
                function=litellm.types.utils.Function(
                    arguments=arguments, name=function_name
                ),
                id=f"call_{len(tool_calls) + 1}",
            )
            tool_calls.append(tool_call)
        return tool_calls

    def _setup_query(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List] = None,
        tool_choice: Optional[str] = None,
        trim: bool = True,
        **kwargs,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        if trim:
            # Find the first tool messages and trim them
            messages = trim_messages(
                messages,
                model=self.model,
                max_tokens=self.model_max_input_tokens,
                trim_ratio=0.75,
            )
            tool_message = []
            for message in messages:
                if message["role"] == "tool":
                    tool_message.append(message)
                else:
                    break
            messages = messages[len(tool_message) :]
            print(
                f"DEBUG: Trimming messages for model {self.model}, tool_message: {tool_message}"
            )

        input_tokens: int = litellm.utils.token_counter(
            model=self.model, messages=messages
        )
        print(
            f"DEBUG: Input tokens: {input_tokens}, trim_threshold: {self.model_max_input_tokens * 0.75}"
        )
        if self.model_max_input_tokens is None:
            logger.warning(f"No max input tokens found for model {self.model}")
        elif input_tokens > self.model_max_input_tokens:
            raise ContextWindowExceededError(
                f"Input tokens {input_tokens} exceed max tokens {self.model_max_input_tokens}"
            )

        completion_kwargs = {
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            **self.completion_kwargs,
            **kwargs,
        }

        if tools is not None and not litellm.supports_function_calling(self.model):
            messages[-1]["content"] += self._generate_prompt_with_tools(
                tools=tools, tool_choice=tool_choice
            )
            completion_kwargs.update({"tools": None, "tool_choice": None})

        return messages, completion_kwargs

    def _process_response(
        self,
        input: List[Dict[str, str]],
        output: litellm.types.utils.ModelResponse,
        tools: Optional[List] = None,
    ) -> Union[
        str, Tuple[str, List[litellm.types.utils.ChatCompletionMessageToolCall]]
    ]:
        #self.update_stats(input=input, output=output, tools=tools)

        choices = output.choices
        assert isinstance(choices[0], litellm.types.utils.Choices), (
            f"LiteLLMModel not support {type(output.choices[0])}."
        )
        content = choices[0].message.content or ""
        if tools:
            if not litellm.supports_function_calling(self.model):
                return content, self._parser_function_calls(content)
            else:
                return content, choices[0].message.tool_calls or []

        return content

    @overload
    def query(
        self,
        messages: str,
        tools: None = None,
        tool_choice: None = None,
        trim: bool = True,
        **kwargs,
    ) -> str: ...

    @overload
    def query(
        self,
        messages: List[Dict[str, str]],
        tools: None = None,
        tool_choice: None = None,
        trim: bool = True,
        **kwargs,
    ) -> str: ...

    @overload
    def query(
        self,
        messages: List[Dict[str, str]],
        tools: List,
        tool_choice: Optional[str] = None,
        trim: bool = True,
        **kwargs,
    ) -> Tuple[str, List[litellm.types.utils.ChatCompletionMessageToolCall]]: ...

    @overload
    def query(
        self,
        messages: str,
        tools: List,
        tool_choice: Optional[str] = None,
        trim: bool = True,
        **kwargs,
    ) -> Tuple[str, List[litellm.types.utils.ChatCompletionMessageToolCall]]: ...

    @retry(
        wait=wait_random_exponential(min=60, max=180),
        reraise=True,
        stop=stop_after_attempt(_MAX_RETRUES),
        retry=retry_if_not_exception_type(_get_non_retryable_exceptions()),
    )
    def query(
        self,
        messages: Union[List[Dict[str, str]], str],
        tools: Optional[List] = None,
        tool_choice: Optional[str] = None,
        trim: bool = True,
        **kwargs,
    ) -> Union[
        str, Tuple[str, List[litellm.types.utils.ChatCompletionMessageToolCall]]
    ]:
        """Synchronous query to the LLM model.

        Args:
            messages: Either a list of message dictionaries or a single query string
            tools: Optional list of tools/functions available to the model
            tool_choice: Optional tool choice mode
            trim: Whether to trim the messages to the model's max input tokens
            **kwargs: Additional arguments passed to the completion call

        Returns:
            Either a string response or a list of tool calls depending on the input
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        messages, completion_kwargs = self._setup_query(
            messages, tools, tool_choice, trim, **kwargs
        )
        try:
            response = litellm.completion(
                **{k: v for k, v in completion_kwargs.items() if v is not None}
            )
        except Exception as e:
            logger.exception(f"Error during LLM query: {e}")
            raise e

        return self._process_response(
            messages, response, completion_kwargs.get("tools")
        )

    @overload
    async def aquery(
        self,
        messages: str,
        tools: None = None,
        tool_choice: None = None,
        trim: bool = True,
        **kwargs,
    ) -> str: ...

    @overload
    async def aquery(
        self,
        messages: List[Dict[str, str]],
        tools: None = None,
        tool_choice: None = None,
        trim: bool = True,
        **kwargs,
    ) -> str: ...

    @overload
    async def aquery(
        self,
        messages: List[Dict[str, str]],
        tools: List,
        tool_choice: Optional[str] = None,
        trim: bool = True,
        **kwargs,
    ) -> Tuple[str, List[litellm.types.utils.ChatCompletionMessageToolCall]]: ...

    @overload
    async def aquery(
        self,
        messages: str,
        tools: List,
        tool_choice: Optional[str] = None,
        trim: bool = True,
        **kwargs,
    ) -> Tuple[str, List[litellm.types.utils.ChatCompletionMessageToolCall]]: ...

    @retry(
        wait=wait_random_exponential(min=60, max=180),
        reraise=True,
        stop=stop_after_attempt(_MAX_RETRUES),
        retry=retry_if_not_exception_type(_get_non_retryable_exceptions()),
    )
    async def aquery(
        self,
        messages: Union[List[Dict[str, str]], str],
        tools: Optional[List] = None,
        tool_choice: Optional[str] = None,
        trim: bool = True,
        **kwargs,
    ) -> Union[
        str, Tuple[str, List[litellm.types.utils.ChatCompletionMessageToolCall]]
    ]:
        """Asynchronous query to the LLM model.

        Args:
            messages_or_query: Either a list of message dictionaries or a single query string
            tools: Optional list of tools/functions available to the model
            tool_choice: Optional tool choice mode
            **kwargs: Additional arguments passed to the completion call

        Returns:
            Either a string response or a list of tool calls depending on the input
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        messages, completion_kwargs = self._setup_query(
            messages, tools, tool_choice, trim, **kwargs
        )
        try:
            response = await litellm.acompletion(
                **{k: v for k, v in completion_kwargs.items() if v is not None}
            )
        except Exception as e:
            logger.exception(f"Error during LLM query: {e}")
            raise e

        return self._process_response(
            messages, response, completion_kwargs.get("tools")
        )
