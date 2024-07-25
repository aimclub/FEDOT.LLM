from typing import Any, Type
from fedot_llm.language_models.base import BaseLLM

def _import_web_model() -> Type[BaseLLM]:
    from fedot_llm.language_models.llms.web_model import CustomWebLLM
    
    return CustomWebLLM

def _import_hugging_face() -> Type[BaseLLM]:
    from fedot_llm.language_models.llms.huggingface import HuggingFaceLLM
    
    return HuggingFaceLLM

def _import_ollama()-> Type[BaseLLM]:
    from fedot_llm.language_models.llms.ollama import OllamaLLM

    return OllamaLLM
    
def __getattr__(name: str) -> Any:
    if name == 'HuggingFaceLLM':
        return _import_hugging_face()
    elif name == 'CustomWebLLM':
        return _import_web_model()
    elif name == 'OllamaLLM':
        return _import_ollama()
    else:
        raise AttributeError(f"Could not find: {name}")
    
__all__ = [
    "HuggingFaceLLM",
    "CustomWebLLM",
    "OllamaLLM"
]