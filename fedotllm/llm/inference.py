import time
from typing import Optional

import tiktoken
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, SecretStr

from fedotllm.log import get_logger
from fedotllm.settings.config_loader import get_settings

logger = get_logger()


class AIInference:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        try:
            headers = None
            chat_base_url = base_url or get_settings().get("config.base_url", None)
            chat_model = str(model or get_settings().config.model or "")
            chat_api_key = SecretStr(api_key or get_settings()["OPENAI_TOKEN"])
            if "vsegpt" in chat_base_url:
                headers = {"X-Title": "FEDOT.LLM"}
            self.model: ChatOpenAI = ChatOpenAI(
                model=chat_model,
                base_url=chat_base_url,
                api_key=chat_api_key,
                default_headers=headers,
            )
        except AttributeError as e:
            raise ValueError("OpenAI key is required") from e

    def chat_completion(
        self,
        user: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        frequency_penalty: float = 0.0,
        *,
        format_vals=None,
        structured: Optional[BaseModel] = None,
        tools: Optional[list] = None,
        max_retries: int = 3,
    ):
        if format_vals is None:
            format_vals = {}
        logger.debug(f"Inferring with model: {self.model}".center(100, "="))
        logger.debug(f"User: {user}")
        system and logger.debug(f"System: {system}")
        model = self.model.bind(
            temperature=temperature, frequency_penalty=frequency_penalty
        )
        if tools:
            model = model.bind_tools(tools)
        if structured:
            model = model.with_structured_output(structured)

        template: ChatPromptTemplate | PromptTemplate
        if system:
            template = ChatPromptTemplate(
                [("system", system), ("user", user)], template_format="jinja2"
            )
        else:
            template = PromptTemplate.from_template(user, template_format="jinja2")

        execution_chain = template | model

        for attempt in range(max_retries):
            try:
                result = execution_chain.invoke(format_vals)
            except Exception as e:
                logger.exception(
                    f"Inference failed with error: {str(e)}.\t Retrying... ({attempt + 1}/{max_retries})\t"
                )
                time.sleep(2**attempt)

        logger.debug(f"Result: {result.__str__()}")
        logger.debug("=" * 100)
        return result


class OpenaiEmbeddings:
    MAX_INPUT = 8191

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        api_key = api_key or get_settings()["OPENAI_TOKEN"]
        base_url = base_url or get_settings().get("config.base_url", None)
        model = model or get_settings().get("config.embeddings", None)

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def encode(self, input: str):
        try:
            response = self.client.embeddings.create(
                model=self.model, input=input, encoding_format="float"
            )
        except Exception:
            len_embeddings = self.num_tokens_from_string(input)
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
