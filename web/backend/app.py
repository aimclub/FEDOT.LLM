from dataclasses import dataclass

from typing_extensions import AsyncIterator, Iterator, List

from fedot_llm.ai.actions import Action, Actions
from fedot_llm.ai.chains.analyze import AnalyzeFedotResultChain
from fedot_llm.ai.chains.fedot import FedotPredictChain
from fedot_llm.ai.chains.metainfo import (DefineDatasetChain,
                                          DefineSplitsChain, DefineTaskChain)
from fedot_llm.ai.chains.ready_chains.predict import PredictChain
from fedot_llm.main import FedotAI
from web.common.types import (AnalyzeResponse, BaseResponse, PipeLineResponse,
                              ProgressResponse, RequestFedotLLM, Response, UIElement,
                              ResponseState, get_logger_handler)


@dataclass
class FedotAIBackend:
    fedotAI: FedotAI
    actions: Actions = Actions([
        Action.from_chain(DefineDatasetChain),
        Action.from_chain(DefineSplitsChain),
        Action.from_chain(DefineTaskChain),
        Action.from_chain(FedotPredictChain),
        Action.from_chain(AnalyzeFedotResultChain)
    ])

    def get_response(self, request: RequestFedotLLM) -> Iterator[BaseResponse]:
        for chain in self.fedotAI.model.stream(request['msg']):
            yield BaseResponse(state=ResponseState.RUNNING, content=str(chain.content), stream=True)
        yield BaseResponse(state=ResponseState.COMPLETE, stream=True)

    async def get_predict(self, request: RequestFedotLLM) -> AsyncIterator[BaseResponse]:
        response = Response()

        chain = PredictChain(self.fedotAI.model, self.fedotAI.dataset)

        page_elements: List[UIElement] = [ProgressResponse(), AnalyzeResponse(), PipeLineResponse()]
        for page_element in page_elements:
            page_element.register_hooks(response=response, actions=self.actions)
            
        handlers = [get_logger_handler(), self.actions.handler]

        async for event in chain.astream_events({'dataset_description': request['msg']}, version='v2'):
            for handler in handlers:
                handler(event=event)
            if len(response.context) > 0:
                yield response.pack()
            response.clean()
        response.root.state = ResponseState.COMPLETE
        yield response.pack()
