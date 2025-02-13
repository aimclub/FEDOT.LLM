from typing import Literal

from deep_translator import GoogleTranslator
from fedotllm.data import Dataset
from fedotllm.llm.inference import AIInference, OpenaiEmbeddings
from fedotllm.main import FedotAI
from fedotllm.web.common.types import (BaseResponse, GraphResponse, InitModel,
                                       MessagesHandler, RequestFedotLLM,
                                       Response, ResponseState)
from typing_extensions import AsyncIterator


class FedotAIBackend:
    def __init__(self) -> None:
        self.fedot_ai = FedotAI()

    def init_model(self, init_model: InitModel) -> None:
        self.fedot_ai.inference = AIInference(
            model=init_model.name,
            base_url=init_model.base_url,
            api_key=init_model.api_key
        )
        self.fedot_ai.embeddings = OpenaiEmbeddings(
            api_key=init_model.api_key,
            base_url=init_model.base_url
        )

    def init_dataset(self, init_dataset: Dataset) -> None:
        self.fedot_ai.dataset = init_dataset

    async def ask(self, request: RequestFedotLLM, lang: Literal['en', 'ru'] = 'en') -> AsyncIterator[BaseResponse]:
        response = Response()
        message_handler = MessagesHandler(lang=lang).message_handler(response)
        graph_handler = GraphResponse(lang=lang).graph_handler(response)
        self.fedot_ai.handlers = [message_handler, graph_handler]

        async for _ in self.fedot_ai.ask(GoogleTranslator(source='auto', target='en').translate(request['msg'])):
            if len(response.context) > 0:
                yield response.pack()
            response.clean()
        response.root.state = ResponseState.COMPLETE
        yield response.pack()
