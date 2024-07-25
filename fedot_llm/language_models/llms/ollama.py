import json
from typing import Dict, List, Optional

import requests

from fedot_llm.language_models.base import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(self, model: str, url: str = 'http://localhost:11434/api/chat', timeout: Optional[int] = 10):
        self.url = url
        self.model = model
        self.timeout = timeout
    
    def _generate(self, formatted_prompt: Dict[str, List[Dict[str, str]]]) -> str:
        payload = {"model": self.model, "format": "json", "stream": False, 'messages': formatted_prompt['messages']}

        try:
            response = requests.post(url=self.url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise RuntimeError(err)
        return json.loads(response.text)['message']['content']