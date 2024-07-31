import json
from typing import Dict, List, Optional, Union, Tuple, Callable, Any

import requests
from dataclasses import dataclass

from fedot_llm.language_models.base import BaseLLM


class OllamaEndpointNotFoundError(Exception):
    """Raised when the Ollama endpoint is not found."""

@dataclass
class OllamaLLM(BaseLLM):
    base_url: str = "http://localhost:11434"
    """Base url the model is hosted under."""

    model: str = "llama3"
    """Model name to use."""

    mirostat: Optional[int] = None
    """Enable Mirostat sampling for controlling perplexity.
    (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)"""

    mirostat_eta: Optional[float] = None
    """Influences how quickly the algorithm responds to feedback
    from the generated text. A lower learning rate will result in
    slower adjustments, while a higher learning rate will make
    the algorithm more responsive. (Default: 0.1)"""

    mirostat_tau: Optional[float] = None
    """Controls the balance between coherence and diversity
    of the output. A lower value will result in more focused and
    coherent text. (Default: 5.0)"""

    num_ctx: Optional[int] = None
    """Sets the size of the context window used to generate the
    next token. (Default: 2048)	"""

    num_gpu: Optional[int] = None
    """The number of GPUs to use. On macOS it defaults to 1 to
    enable metal support, 0 to disable."""

    num_thread: Optional[int] = None
    """Sets the number of threads to use during computation.
    By default, Ollama will detect this for optimal performance.
    It is recommended to set this value to the number of physical
    CPU cores your system has (as opposed to the logical number of cores)."""

    num_predict: Optional[int] = None
    """Maximum number of tokens to predict when generating text.
    (Default: 128, -1 = infinite generation, -2 = fill context)"""

    repeat_last_n: Optional[int] = None
    """Sets how far back for the model to look back to prevent
    repetition. (Default: 64, 0 = disabled, -1 = num_ctx)"""

    repeat_penalty: Optional[float] = None
    """Sets how strongly to penalize repetitions. A higher value (e.g., 1.5)
    will penalize repetitions more strongly, while a lower value (e.g., 0.9)
    will be more lenient. (Default: 1.1)"""

    temperature: Optional[float] = None
    """The temperature of the model. Increasing the temperature will
    make the model answer more creatively. (Default: 0.8)"""

    stop: Optional[List[str]] = None
    """Sets the stop tokens to use."""

    tfs_z: Optional[float] = None
    """Tail free sampling is used to reduce the impact of less probable
    tokens from the output. A higher value (e.g., 2.0) will reduce the
    impact more, while a value of 1.0 disables this setting. (default: 1)"""

    top_k: Optional[int] = None
    """Reduces the probability of generating nonsense. A higher value (e.g. 100)
    will give more diverse answers, while a lower value (e.g. 10)
    will be more conservative. (Default: 40)"""

    top_p: Optional[float] = None
    """Works together with top-k. A higher value (e.g., 0.95) will lead
    to more diverse text, while a lower value (e.g., 0.5) will
    generate more focused and conservative text. (Default: 0.9)"""

    system: Optional[str] = None
    """system prompt (overrides what is defined in the Modelfile)"""

    template: Optional[str] = None
    """full prompt or prompt template (overrides what is defined in the Modelfile)"""

    format: Optional[str] = None
    """Specify the format of the output (e.g., json)"""

    keep_alive: Optional[Union[int, str]] = None
    """How long the model will stay loaded into memory.

    The parameter (Default: 5 minutes) can be set to:
    1. a duration string in Golang (such as "10m" or "24h");
    2. a number in seconds (such as 3600);
    3. any negative number which will keep the model loaded \
        in memory (e.g. -1 or "-1m");
    4. 0 which will unload the model immediately after generating a response;

    See the [Ollama documents](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-keep-a-model-loaded-in-memory-or-make-it-unload-immediately)"""

    raw: Optional[bool] = None
    """raw or not."""

    stream: Optional[bool] = False
    """if false the response will be returned as a single response object, rather than a stream of objects"""

    headers: Optional[dict] = None
    """Additional headers to pass to endpoint (e.g. Authorization, Referer).
    This is useful when Ollama is hosted on cloud services that require
    tokens for authentication.
    """

    auth: Union[Callable, Tuple, None] = None
    """Additional auth tuple or callable to enable Basic/Digest/Custom HTTP Auth.
    Expects the same format, type and values as requests.request auth parameter."""

    timeout: Optional[int] = 10
    """Timeout for the request stream in seconds. (Default: 10)"""

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Ollama."""
        return {
            "model": self.model,
            "format": self.format,
            "options": {
                "mirostat": self.mirostat,
                "mirostat_eta": self.mirostat_eta,
                "mirostat_tau": self.mirostat_tau,
                "num_ctx": self.num_ctx,
                "num_gpu": self.num_gpu,
                "num_thread": self.num_thread,
                "num_predict": self.num_predict,
                "repeat_last_n": self.repeat_last_n,
                "repeat_penalty": self.repeat_penalty,
                "temperature": self.temperature,
                "stop": self.stop,
                "tfs_z": self.tfs_z,
                "top_k": self.top_k,
                "top_p": self.top_p,
            },
            "system": self.system,
            "template": self.template,
            "keep_alive": self.keep_alive,
            "stream": self.stream,
            "raw": self.raw,
        }

    def _generate(
            self,
            payload: Dict[str, List[Dict[str, str]]],
            **kwargs) -> str:

        params = self._default_params

        for key in self._default_params:
            if key in kwargs:
                params[key] = kwargs[key]

        if "options" in kwargs:
            params["options"] = kwargs["options"]
        else:
            params["options"] = {
                **params["options"],
                **{key: value for key, value in kwargs.items() if key not in self._default_params},
            }

        request_payload = {"messages": payload.get("messages", []), **params}
        try:
            response = requests.post(
                url=f"{self.base_url}/api/chat",
                headers={
                    "Content-Type": "application/json",
                    **(self.headers if isinstance(self.headers, dict) else {}),
                },
                auth=self.auth,
                json=request_payload,
                stream=False,
                timeout=self.timeout,
            )

            response.encoding = "utf-8"
            if response.status_code != 200:
                if response.status_code == 404:
                    raise OllamaEndpointNotFoundError(
                        f"Ollama endpoint not found at {self.base_url}."
                        " Maybe the model is not found"
                        f" and you should pull the model with `ollama pull {self.model}`."
                    )
                else:
                    optional_detail = response.text
                    raise RuntimeError(
                        f"Failed to generate response from Ollama: {response.status_code}."
                        f" Details: {optional_detail}"
                    )
        except requests.exceptions.HTTPError as err:
            raise RuntimeError(err)
        return json.loads(response.text)['message']['content']
