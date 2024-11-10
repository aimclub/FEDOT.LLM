
from typing import Callable, List

from pydantic import BaseModel, Field, ConfigDict, model_validator
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.schema import StreamEvent
from typing_extensions import Any, AsyncIterator

from fedot_llm.agents.supervisor.supervisor import SupervisorAgent
from fedot_llm.agents.memory import LongTermMemory
from fedot_llm.data import Dataset
from fedot_llm.llm.inference import AIInference

class FedotAI(BaseModel):
    dataset: Dataset
    inference: AIInference = Field(default_factory=AIInference)
    memory: LongTermMemory = Field(default_factory=LongTermMemory)
    entry_point: Runnable = Field(default=None)
    handlers: List[Callable[[StreamEvent], None]] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def set_entry_point(self) -> Any:
        self.entry_point = SupervisorAgent(
            inference=self.inference,
            memory=self.memory
        ).create_graph()
        return self

    async def ask(self, message: str) -> AsyncIterator[Any]:
        async for event in self.entry_point.astream_events(
            {"messages": [HumanMessage(content=message)], "dataset": self.dataset},
            version="v2"
        ):
            for handler in self.handlers:
                handler(event)
            yield event
