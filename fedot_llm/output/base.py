from abc import ABC, abstractmethod
from typing import Any, Dict

from langchain_core.runnables import Runnable


class BaseFedotAIOutput(ABC):
    def __init__(self, finish_event_name: str = 'master'):
        self.finish_event_name = finish_event_name

    @abstractmethod
    async def _chain_call(self, chain: Runnable, chain_input: Dict[str, Any]):
        """call chain add display results
        """
