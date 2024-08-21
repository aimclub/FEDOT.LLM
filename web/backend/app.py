from fedot_llm.main import FedotAI
from web.common.types import AnalyzeResponse, RequestFedotLLM, BaseResponse, ProgressResponse, PipeLineResponse
from dataclasses import dataclass
from typing_extensions import Iterator, List, AsyncIterator


@dataclass
class FedotAIBackend:
    fedotAI: FedotAI

    def get_response(self, request: RequestFedotLLM) -> Iterator[BaseResponse]:
        for chain in self.fedotAI.chain_builder.assistant.stream(request['msg']):
            yield BaseResponse(state='running', content=str(chain.content), stream=True)
        yield BaseResponse(state='complete', stream=True)

    async def get_predict(self, request: RequestFedotLLM) -> AsyncIterator[BaseResponse]:
        predict_chain = self.fedotAI.chain_builder.predict_chain
        stages: List[BaseResponse] = [ProgressResponse(), PipeLineResponse(), AnalyzeResponse()]
        handlers = [stage.handler for stage in stages]
        response = BaseResponse()
        async for event in predict_chain.astream_events({'big_description': request['msg']}, version='v2'):
            content = []
            is_changed = False
            for handler in handlers:
                if msg := handler(event=event):
                    if msg.content or msg.name or msg.state:
                        is_changed = True
                    content.append(msg)
            if is_changed:
                response.state = 'running'
                response.content = content
                yield response
        response.state = 'complete'
        yield response
