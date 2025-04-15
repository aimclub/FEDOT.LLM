from functools import partial
from typing import Dict, List

from fedotllm.prompting.prompts.utils import field_parsing_prompt
from fedotllm.utils.parsers import parse_and_check_json


class PromptGenerator:
    def __init__(self, prompt: str):
        self.prompt = prompt

    @property
    def chat_prompt(self) -> List[Dict[str, str]]:
        chat_prompt = [{"role": "user", "content": self.prompt}]

        return chat_prompt


class JsonFieldPromptGenerator(PromptGenerator):
    def __init__(self, prompt: str, fields: List[str] = []):
        self.prompt = prompt
        self.fields = fields

    @property
    def field_parsing_prompt(self) -> str:
        return field_parsing_prompt(self.fields)

    def create_parser(self):
        return partial(parse_and_check_json, expected_keys=self.fields)
