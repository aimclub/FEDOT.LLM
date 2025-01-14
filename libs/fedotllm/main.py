from typing import Callable, List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.runnables.schema import StreamEvent
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Any, AsyncIterator

from fedotllm.agents.memory import LongTermMemory
from fedotllm.agents.supervisor.supervisor import SupervisorAgent
from fedotllm.data import Dataset
from fedotllm.llm.inference import AIInference


class FedotAI(BaseModel):
    dataset: Optional[Dataset] = Field(default=None)
    inference: AIInference = Field(default_factory=AIInference)
    memory: LongTermMemory = Field(default_factory=LongTermMemory)
    handlers: List[Callable[[StreamEvent], None]] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def ask(self, message: str) -> AsyncIterator[Any]:
        entry_point = SupervisorAgent(
            inference=self.inference,
            memory=self.memory,
            dataset=self.dataset
        ).create_graph()
        async for event in entry_point.astream_events(
                {"messages": [HumanMessage(content=message)]},
                version="v2"
        ):
            for handler in self.handlers:
                handler(event)
            yield event
