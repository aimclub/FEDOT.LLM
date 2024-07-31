import json
from typing import Dict, List, Optional

import requests

from fedot_llm.language_models.base import BaseLLM


class CustomWebLLM(BaseLLM):
    def __init__(self, url: str, model: str, timeout: Optional[int] = 10):
        self.url = url
        self.model = model
        self.timeout = timeout
    
    def _generate(self, formatted_prompt: Dict[str, List[Dict[str, str]]], **kwargs) -> str:
        try:
            response = requests.post(url=self.url, json=formatted_prompt, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise RuntimeError(err)
        return json.loads(response.text)['choices'][0]['message']['content']
    