import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Callable, List, Optional

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.runnables.schema import StreamEvent
from omegaconf import DictConfig
from rich import print as rprint

from fedotllm.agents.agent_wrapper.agent_wrapper import AgentWrapper
from fedotllm.agents.automl.automl import AutoMLAgent
from fedotllm.agents.automl.automl_chat.automl_chat import AutoMLAgentChat
from fedotllm.agents.researcher.researcher import ResearcherAgent
from fedotllm.agents.supervisor.supervisor import SupervisorAgent
from fedotllm.llm import LiteLLMModel, OpenaiEmbeddings
from fedotllm.tabular import Dataset
from fedotllm.utils import unpack_omega_config


class FedotAI:
    def __init__(
        self,
        config: DictConfig,
        task_path: Path,
        handlers: List[Callable[[StreamEvent], None]] = [],
        workspace: Optional[Path] = None,
        session_id: Optional[str] = None,
    ):
        self.task_path = Path(task_path).resolve()
        assert self.task_path.is_dir(), (
            "Task path does not exist, please provide a valid directory."
        )
        rprint(f"Task path: {self.task_path}")

        self.llm = LiteLLMModel(**unpack_omega_config(config.llm))
        self.config = config
        self.embeddings = OpenaiEmbeddings(**unpack_omega_config(config.embeddings))
        self.handlers = handlers
        self.workspace = workspace
        self.dataset = Dataset.from_path(self.task_path)
        if session_id is None:
            self.session_id = str(uuid.uuid4())
        else:
            self.session_id = session_id

    async def ask(self, message: str) -> AsyncIterator[Any]:
        if not self.workspace:
            self.workspace = Path(
                f"fedotllm-output-{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            )

        automl_agent = AutoMLAgentChat(
            AutoMLAgent(self.config, self.dataset, self.session_id)
        ).create_graph()
        research_agent = AgentWrapper(
            ResearcherAgent(self.config, self.embeddings, self.session_id)
        ).create_graph()

        entry_point = SupervisorAgent(
            self.config, self.embeddings, self.session_id, research_agent, automl_agent
        ).create_graph()

        async for event in entry_point.astream_events(
            {
                "messages": [HumanMessage(content=message)],
                "task_path": self.task_path,
                "workspace": self.workspace,
            },
            version="v2",
        ):
            for handler in self.handlers:
                handler(event)
            yield event
