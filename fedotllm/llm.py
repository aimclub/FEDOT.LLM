import os
from typing import Any, Dict, List, Optional, Type, TypeVar

import litellm
import tiktoken
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from fedotllm import prompts
from fedotllm.agents.utils import parse_json
from fedotllm.log import logger
from fedotllm.settings.config_loader import get_settings

T = TypeVar("T", bound=BaseModel)

litellm._logging._disable_debugging()

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")

if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]


class AIInference:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        settings = get_settings()
        self.base_url = base_url or settings.get("config.base_url")
        self.model = model or settings.get("config.model")
        self.api_key = api_key or os.getenv("FEDOTLLM_LLM_API_KEY")

        if not self.api_key:
            raise Exception(
                "API key not provided and FEDOTLLM_LLM_API_KEY environment variable not set"
            )

        self.completion_params = {
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            # "max_completion_tokens": 8000,
            "extra_headers": {"X-Title": "FEDOT.LLM"},
        }

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def create(self, messages: str, response_model: Type[T]) -> T:
        messages = f"{messages}\n{prompts.utils.structured_response(response_model)}"
        response = self.query(messages)
        json_obj = parse_json(response) if response else None
        return response_model.model_validate(json_obj)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def query(self, messages: str | List[Dict[str, Any]]) -> str | None:
        messages = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else messages
        )
        logger.debug("Sending messages to LLM: %s", messages)
        response = litellm.completion(
            messages=messages,
            **self.completion_params,
        )
        logger.debug("Received response from LLM: %s", response.choices[0].message.content)
        return response.choices[0].message.content


class OpenaiEmbeddings:
    MAX_INPUT = 8191

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        base_url = base_url or get_settings().get("config.base_url", None)
        model = model or get_settings().get("config.embeddings", None)

        if api_key:
            self.api_key = api_key
        elif "FEDOTLLM_EMBEDDINGS_API_KEY" in os.environ:
            self.api_key = os.environ["FEDOTLLM_EMBEDDINGS_API_KEY"]
        else:
            raise Exception(
                "OpenAI API env variable FEDOTLLM_EMBEDDINGS_API_KEY not set"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def encode(self, input: str):
        try:
            response = self.client.embeddings.create(
                model=self.model, input=input, encoding_format="float"
            )
        except Exception:
            len_embeddings = num_tokens_from_string(input)
            if len_embeddings > self.MAX_INPUT:
                raise Exception(f"Input exceeds the limit of <{self.model}>!")
            else:
                raise Exception("Embeddings generation failed!")
        return response.data


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """
    Returns the number of tokens in a text string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))

    return num_tokens


if __name__ == "__main__":
    inference = AIInference()
    print(inference.query("Say hello world!"))
