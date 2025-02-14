from typing import Literal, Optional

from deep_translator import GoogleTranslator
from fedotllm.llm.inference import AIInference, OpenaiEmbeddings
from fedotllm.main import FedotAI
from fedotllm.web.common.types import (BaseResponse, GraphResponse,
                                       MessagesHandler,
                                       Response, ResponseState)
from typing_extensions import AsyncIterator
from pathlib import Path



async def ask(msg: str,
                task_path: Path,
                llm_name: str,
                llm_base_url: Optional[str] = None,
                llm_api_key: Optional[str] = None,
                work_dir: Optional[Path] = None,
                lang: Literal['en', 'ru'] = 'en') -> AsyncIterator[BaseResponse]:

    response = Response()
    message_handler = MessagesHandler(lang=lang).message_handler(response)
    graph_handler = GraphResponse(lang=lang).graph_handler(response)
    fedot_ai = FedotAI(task_path=task_path,
                        inference=AIInference(
                            model=llm_name, base_url=llm_base_url, api_key=llm_api_key),
                        embeddings=OpenaiEmbeddings(
                            api_key=llm_api_key, base_url=llm_base_url),
                        handlers=[message_handler, graph_handler],
                        work_dir=work_dir
                        )

    async for _ in fedot_ai.ask(GoogleTranslator(source='auto', target='en').translate(msg)):
        if len(response.context) > 0:
            yield response.pack()
        response.clean()
    response.root.state = ResponseState.COMPLETE
    yield response.pack()
