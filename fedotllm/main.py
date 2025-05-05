from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.runnables.schema import StreamEvent
from typing_extensions import Any, AsyncIterator

from fedotllm.agents.supervisor.supervisor import SupervisorAgent
from fedotllm.agents.automl.automl_chat import AutoMLAgentChat
from fedotllm.data import Dataset
from fedotllm.llm.inference import AIInference, OpenaiEmbeddings


class FedotAI:
    def __init__(
        self,
        task_path: Optional[Path] = None,
        inference: Optional[AIInference] = AIInference(),
        embeddings: Optional[OpenaiEmbeddings] = OpenaiEmbeddings(),
        handlers: Optional[List[Callable[[StreamEvent], None]]] = [],
        work_dir: Optional[Path] = None,
        automl_only: Optional[bool] = False,
    ):
        self.task_path = task_path.resolve()
        assert task_path.is_dir(), (
            "Task path does not exist, please provide a valid directory."
        )

        self.inference = inference
        self.embeddings = embeddings
        self.handlers = handlers
        self.work_dir = work_dir
        self.automl_only = automl_only

    async def ask(self, message: str) -> AsyncIterator[Any]:
        if not self.work_dir:
            self.work_dir = Path(
                f"fedotllm-output-{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            )

        dataset = Dataset.from_path(self.task_path)

        if self.automl_only:
            entry_point = AutoMLAgentChat(
                inference=self.inference, dataset=dataset
            ).create_graph()
        else:
            entry_point = SupervisorAgent(
                inference=self.inference, embeddings=self.embeddings, dataset=dataset
            ).create_graph()

        async for event in entry_point.astream_events(
            {
                "messages": [HumanMessage(content=message)],
                "work_dir": self.work_dir,
            },
            version="v2",
        ):
            for handler in self.handlers:
                handler(event)
            yield event
