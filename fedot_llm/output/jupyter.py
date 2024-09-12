from dataclasses import dataclass
from typing import List

from fedot_llm.ai.actions import Actions, Action
from fedot_llm.main import FedotAI
from web.common.types import Response
from langchain_core.runnables.schema import StreamEvent

@dataclass
class JupyterOutputBackend:
    actions: Actions = Actions(id='')

    def answer_handler(self) -> None:
        def handler(event: StreamEvent, action: Action) -> None:
            pass

        for action in self.actions.records.values():
            action.on_change.append(handler)