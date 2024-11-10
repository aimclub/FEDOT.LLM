from typing_extensions import AsyncIterator
from fedot_llm.main import FedotAI
from web.common.types import (BaseResponse, GraphResponse,
                              RequestFedotLLM, Response,
                              ResponseState)
from web.common.types import MessagesHandler


class FedotAIBackend:
    def __init__(self, fedot_ai: FedotAI) -> None:
        self.fedot_ai = fedot_ai

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
