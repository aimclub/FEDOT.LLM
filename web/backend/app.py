from fedot_llm.main import FedotAI
from web.common.types import AnalyzeResponse, RequestFedotLLM, BaseResponse, ProgressResponse
from dataclasses import dataclass
from typing_extensions import Iterator, Dict, Any, List, AsyncIterator
import asyncio
from fedot_llm.output.base import BaseFedotAIOutput
from langchain_core.runnables import Runnable
from langchain_core.runnables.schema import StreamEvent
import logging


@dataclass
class FedotAIBackend:
    fedotAI: FedotAI

    def get_response(self, request: RequestFedotLLM) -> Iterator[BaseResponse]:
        for chain in self.fedotAI.chain_builder.assistant.stream(request['msg']):
            yield BaseResponse(state='running', content=str(chain.content), stream=True)
        yield BaseResponse(state='complete', stream=True)

    async def get_predict(self, request: RequestFedotLLM) -> AsyncIterator[BaseResponse]:
        predict_chain = self.fedotAI.chain_builder.predict_chain
        stages: List[BaseResponse] = [ProgressResponse(), AnalyzeResponse()]
        handlers = [stage.handler for stage in stages]
        async for event in predict_chain.astream_events({'big_description': request['msg']}, version='v2'):
            content = []
            for handler in handlers:
                content.append(handler(event=event))
            yield BaseResponse(state='running', content=content)
        yield BaseResponse(state='complete')
