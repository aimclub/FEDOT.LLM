from typing_extensions import AsyncIterator

from fedotllm.data import Dataset
from fedotllm.llm.inference import AIInference
from fedotllm.main import FedotAI
from fedotllm.web.common.types import (BaseResponse, GraphResponse,
                                       RequestFedotLLM, Response,
                                       ResponseState, InitModel)
from fedotllm.web.common.types import MessagesHandler


class FedotAIBackend:
    def __init__(self) -> None:
        self.fedot_ai = FedotAI()

    def init_model(self, init_model: InitModel) -> None:
        self.fedot_ai.inference = AIInference(
            model=init_model.name,
            base_url=init_model.base_url,
            api_key=init_model.api_key
        )

    def init_dataset(self, init_dataset: Dataset) -> None:
        self.fedot_ai.dataset = init_dataset

    async def ask(self, request: RequestFedotLLM) -> AsyncIterator[BaseResponse]:
        response = Response()
        message_handler = MessagesHandler().message_handler(response)
        graph_handler = GraphResponse().graph_handler(response)
        self.fedot_ai.handlers = [message_handler, graph_handler]

        async for _ in self.fedot_ai.ask(request['msg']):
            if len(response.context) > 0:
                yield response.pack()
            response.clean()
        response.root.state = ResponseState.COMPLETE
        yield response.pack()
