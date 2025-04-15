from pathlib import Path
from typing import List, Literal, Optional

from deep_translator import GoogleTranslator
from typing import AsyncIterator

from fedotllm import run_ui
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
    config_overrides: Optional[List[str]] = None,
    workspace: Optional[Path] = None,
    lang: Literal["en", "ru"] = "en",
) -> AsyncIterator[BaseResponse]:
    response = Response()
    message_handler = MessagesHandler(lang=lang).message_handler(response)
    graph_handler = GraphResponse(lang=lang).graph_handler(response)
    fedot_ai = run_ui(
        task_path=task_path,
        presets="default",
        workspace_path=workspace,
        config_overrides=config_overrides,
    )
    fedot_ai.handlers = [message_handler, graph_handler]

    async for _ in fedot_ai.ask(
        GoogleTranslator(source="auto", target="en").translate(msg)
    ):
        if len(response.context) > 0:
            yield response.pack()
        response.clean()
    response.root.state = ResponseState.COMPLETE
    yield response.pack()
