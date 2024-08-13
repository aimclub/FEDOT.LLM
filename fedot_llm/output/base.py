from abc import ABC, abstractmethod
from typing import Any, Dict

from langchain_core.runnables import Runnable


class BaseFedotAIOutput(ABC):
    @abstractmethod
    async def _chain_call(self, chain: Runnable, chain_input: Dict[str, Any]):
        """call chain add display results
        """
