from typing import Dict, List, Optional, Any

from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer

from fedot_llm.language_models.base import BaseLLM

"""
Under Development
"""
@dataclass
class HuggingFaceLLM(BaseLLM):
    repo_id: str = field(default="microsoft/Phi-3-mini-4k-instruct")
    """Repo to use."""
    
    api_token: Optional[str] = None
    """huggingface hub api token."""
    
    max_new_tokens: int = 512
    """Maximum number of generated tokens"""
    
    top_k: Optional[int] = None
    """The number of highest probability vocabulary tokens to keep for
    top-k-filtering."""
    
    top_p: Optional[float] = 0.95
    """If set to < 1, only the smallest set of most probable tokens with probabilities
    that add up to `top_p` or higher are kept for generation."""
    
    typical_p: Optional[float] = 0.95
    """Typical Decoding mass. See [Typical Decoding for Natural Language
    Generation](https://arxiv.org/abs/2202.00666) for more information."""
    
    temperature: Optional[float] = 0.8
    """The value used to module the logits distribution."""
    
    repetition_penalty: Optional[float] = None
    """The parameter for repetition penalty. 1.0 means no penalty.
    See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details."""

    
    truncate: Optional[int] = None
    """Truncate inputs tokens to the given size"""

    
    seed: Optional[int] = None
    """Random sampling seed"""
    
    inference_server_url: str = ""
    """text-generation-inference instance base url"""
    
    timeout: int = 120
    """Timeout in seconds"""
    
    streaming: bool = False
    """Whether to generate a stream of tokens asynchronously"""
    
    do_sample: bool = False
    """Activate logits sampling"""
    
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Holds any model parameters valid for `call` not explicitly specified"""
    
    tokenizer: Any = field(init=False)
    """Huggingface tokenizer"""
    
    model: Any = field(init=False)
    """Huggingface model"""
    
    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id, token=self.api_token)
        self.model = AutoModelForCausalLM.from_pretrained(self.repo_id, token=self.api_token)
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling text generation inference API."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "typical_p": self.typical_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "truncate": self.truncate,
            "seed": self.seed,
            "do_sample": self.do_sample,
            **self.model_kwargs,
        }
    
    
    
    def _generate(self, payload: Dict[str, List[Dict[str, str]]], **kwargs) -> str:
        params = self._default_params

        for key in self._default_params:
            if key in kwargs:
                params[key] = kwargs[key]
        
        input_ids = self.tokenizer.apply_chat_template(
                payload.get("messages", []),
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
        outputs = self.model.generate(
                input_ids,
                **params
            )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)
        