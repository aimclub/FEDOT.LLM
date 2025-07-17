from pathlib import Path
from typing import List, Literal, Optional

from typing_extensions import AsyncIterator

from fedotllm.main import FedotAI
from fedotllm.web.common.types import (
    BaseResponse,
    GraphResponse,
    MessagesHandler,
    Response,
    ResponseState,
)


async def ask(
    msg: str,
    task_path: Path,
    config_overrides: List[str] | None = None,
    workspace: Optional[Path] = None,
    lang: Literal["en", "ru"] = "en",
) -> AsyncIterator[BaseResponse]:
    response = Response()
    message_handler = MessagesHandler(lang=lang).message_handler(response)
    graph_handler = GraphResponse(lang=lang).graph_handler(response)
    fedot_ai = FedotAI(
        task_path=task_path,
        config_overrides=config_overrides,
        handlers=[message_handler, graph_handler],
        workspace=workspace,
    )

    async for _ in fedot_ai.ask(msg):
        if len(response.context) > 0:
            yield response.pack()
        response.clean()
    response.root.state = ResponseState.COMPLETE
    yield response.pack()
