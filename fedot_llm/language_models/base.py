from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseLLM(ABC):
    
    def generate(self, user_prompt: str, sys_prompt: Optional[str] = None, context: Optional[str] = None) -> str:
        """
        Generate a response based on user input, system prompt, and context.

        Args:
            user_prompt (str): The user input prompt.
            sys_prompt (str): The system prompt.
            context (None): Optional context information.

        Returns:
            response (str): Model response
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
        return self._generate(formatted_prompt)
    
    @abstractmethod
    def _generate(self, formatted_prompt: Dict[str, List[Dict[str, str]]]) -> str:
        """Generate a response using an LLM-specific interface

        Args:
            formatted_prompt (Dict[str, List[Dict[str, str]]]): The formatted prompt to generate a response

        Returns:
            str: The generated response
        """

        
            
            
    