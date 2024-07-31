from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseLLM(ABC):
    
    def generate(self, user_prompt: str, sys_prompt: Optional[str] = None, context: Optional[str] = None, **kwargs) -> str:
        """
        Generate a response based on user input, system prompt, and context.

        Args:
            user_prompt (str): The user input prompt.
            sys_prompt (str): The system prompt.
            context (str): Optional context information.
            kwargs (dict): Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            str: Model string response
        """
        
        formatted_prompt = {}
        formatted_prompt["messages"] = []
        if sys_prompt is not None:
            formatted_prompt["messages"].append(
                {
                    "role": "system",
                    "content": sys_prompt
                }
            )
        if user_prompt is not None:
            formatted_prompt["messages"].append(
                {
                    "role": "user",
                    "content": user_prompt + "{context}".format(
                        context=(
                            f"\n\nCONTEXT:\n{context}" if context is not None else ''
                        )
                    )
                }
            )
        else:
            raise RuntimeError("User_promt can't be None!")
        return self._generate(formatted_prompt, **kwargs)
    
    @abstractmethod
    def _generate(self, formatted_prompt: Dict[str, List[Dict[str, str]]], **kwargs) -> str:
        """Generate a response using an LLM-specific interface

        Args:
            formatted_prompt (Dict[str, List[Dict[str, str]]]): The formatted prompt to generate a response
            kwargs (dict): Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            str: Model string response
        """

        
            
            
    