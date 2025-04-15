from typing import Optional

import tiktoken
from openai import OpenAI

from fedotllm.log import get_logger
from fedotllm.settings.config_loader import get_settings

logger = get_logger()


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
