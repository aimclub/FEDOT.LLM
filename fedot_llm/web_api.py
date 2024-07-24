import json
import os
import uuid
from typing import Any

import requests


class WebAssistant:
    """
    Web implementation of LLM assistant for answering urbanistic questions.
    """
    def __init__(self, url, model_type = "8b") -> None:
        """
        Initialize an instanse of LLM assistant.

        Args:
            url (str): Url that requests are being sent to.
            model_type (str, optional): Which model chat template to choose ("8b" or "70b") Defaults to 8b.
        """
        self._system_prompt = None
        self._context = None
        self._url = url
        self._model_type = model_type

    def set_sys_prompt(self, new_prompt: str) -> None:
        """Set model's role and generation instructions.

        Args:
            new_prompt (str): New instructions.
        """
        self._system_prompt = new_prompt

    def set_context(self, context: str) -> None:
        """Set a context to model's prompt

        Args:
            context (str): context related to question.
        """
        self._context = context

    def __call__(self, user_question: str,
                 temperature: float = .015,
                 top_p: float = .5,
                 token_limits: int = 8000,
                 timeout: int = 10,
                 *args: Any,
                 **kwargs: Any) -> str:
        """Get a response from model for given question.

        Args:
            user_question (str): A user's prompt. Question that requires an answer.
            temperature (float, optional): Generation temperature. 
            The higher ,the less stable answers will be. Defaults to 0.015.
            top_p (float, optional): Nuclear sampling. Selects the most likely tokens from a probability distribution,
            timeout (int, optional): Timeout for the request. Defaults to 10 sec. 
            considering the cumulative probability until it reaches a predefined threshold “top_p”. Defaults to 0.5.

        Returns:
            str: Model's answer to user's question. 
        """

        formatted_prompt = {}

        if self._model_type == "8b":

            formatted_prompt = {
                "messages": [
                    {
                        "role": "system",
                        "content": self._system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Question: {user_question} Context: {self._context}"
                    }
                ]
            }
            try:
                response = requests.post(url=self._url, json=formatted_prompt, timeout=timeout)
                response.raise_for_status()
            except requests.exceptions.HTTPError as err:
                raise RuntimeError(err)
            
            if kwargs.get('as_json'):
                try:
                    res = json.loads(response.text)['choices'][0]['message']['content'].split('ANSWER: ')[1]
                except:
                    res = json.loads(response.text)['choices'][0]['message']['content']
                return res
            else:
                return response.text

        elif self._model_type == "70b":

            job_id = str(uuid.uuid4())
            content = ''
            f'<|begin_of_text|><|start_header_id|>system<|end_header_id|> {self._system_prompt}'
            f'<|eot_id|><|start_header_id|>user<|end_header_id|> Question: {user_question}'
            f'Context: {self._context} <|eot_id|><|start_header_id|>assistant<|end_header_id|>'

            formatted_prompt = {
                "job_id": job_id,
                "meta": {
                    "temperature": temperature,
                    "tokens_limit": token_limits,
                    "stop_words": [
                        "string"
                    ]
                },
                "content": content
            }
            try:
                response = requests.post(url=self._url, json=formatted_prompt, timeout=timeout)
                response.raise_for_status()
            except requests.exceptions.HTTPError as err:
                raise RuntimeError(err)
            
            if kwargs.get('as_json'):
                try:
                    res = json.loads(response.text)['content'].split('ОТВЕТ: ')[1]
                except:
                    res = json.loads(response.text)['content']
                return res
            else:
                res = json.loads(response.text)
                return res['content']
        else:
            raise NotImplementedError("Model type not supported")
        