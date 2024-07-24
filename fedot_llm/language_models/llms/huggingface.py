from fedot_llm.language_models.base import BaseLLM
from typing import Optional, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM

"""
Under Development
"""
class HuggingFaceLLM(BaseLLM):
    def __init__(self, model_id: str, access_token: Optional[str] = None, **generation_kwargs) -> None:
        """
        Initializes a new instance of the HuggingFaceLLM.
        
        Args:
            model_id (str): The identifier of the model to be used.
            access_token (Optional[str]): The access token for authentication (default is None).
        """
        self.model_id = model_id,
        self.access_token = access_token
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, token=access_token)
        self.generation_kwargs = generation_kwargs
    
    def _generate(self, formatted_prompt: Dict[str, List[Dict[str, str]]]) -> str:
        input_ids = self.tokenizer.apply_chat_template(
                formatted_prompt['messages'],
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
        outputs = self.model.generate(
                input_ids,
                **self.generation_kwargs
            )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)
        