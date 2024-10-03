from dataclasses import dataclass, field
from typing import Callable
from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.schema import StreamEvent
from typing_extensions import AsyncIterator, Any

from fedot_llm.ai.agents.supervisor import SupervisorAgent
from fedot_llm.ai.memory import LongTermMemory
from fedot_llm.data import Dataset


@dataclass
class FedotAI:
    dataset: Dataset
    model: BaseChatModel
    memory: LongTermMemory = field(default_factory=LongTermMemory)
    entry_point: Runnable = field(init=False)
    handlers: List[Callable[[StreamEvent], None]] = field(default_factory=list)

    def __post_init__(self):
        self.entry_point = SupervisorAgent(llm=self.model, memory=self.memory, dataset=self.dataset).as_graph

    async def ask(self, message: str) -> AsyncIterator[Any]:
        async for event in self.entry_point.astream_events({"messages": [HumanMessage(content=message)]}, version="v2"):
            for handler in self.handlers:
                handler(event)
            yield event
