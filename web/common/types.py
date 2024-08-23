from __future__ import annotations

from dataclasses import dataclass, field

from dataclasses_json import dataclass_json
from langchain_core.runnables.schema import StreamEvent
from typing_extensions import (List, Literal, Optional,
                               TypeAlias, TypedDict, Union, Dict)

from fedot_llm.ai.chains.legacy.chains import steps

from web.backend.utils.graph import GraphvizBuilder

from hashlib import sha256
from datetime import datetime

States: TypeAlias = Literal["running", "complete", "error"]


class RequestFedotLLM(TypedDict):
    msg: str


class TypedContentResponse(TypedDict):
    data: ResponseContent
    type: str


@dataclass_json
@dataclass
class BaseResponse:
    id: Optional[str] = None
    state: Optional[States] = None
    name: Optional[str] = None
    content: ResponseContent = None
    stream: bool = False

    def __post_init__(self):
        if not self.id:
            self.id = self.new_id(self.name)

    @staticmethod
    def new_id(name: Optional[str] = ''):
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if name:
            return f"{name}{sha256(f'{date}'.encode()).hexdigest()}"
        else:
            return sha256(f'{date}'.encode()).hexdigest()

    def handler(self, event: StreamEvent) -> Optional[BaseResponse]:
        """Method that return stage handler"""
        return self

    def __iadd__(self, other: Union[BaseResponse, None]) -> BaseResponse:
        if other is None:
            return self
        if not isinstance(other, BaseResponse):
            raise TypeError("Right operand must be of type BaseResponse")

        if self.id != other.id:
            raise ValueError("Cannot add objects with different ids")

        if isinstance(self.content, List) and isinstance(other.content, List):
            self_content_dict = {item.id: item for item in self.content}
            for new_item in other.content:
                if new_item.id in self_content_dict:
                    self_content_dict[new_item.id] += new_item
                else:
                    self.content.append(new_item)
        elif isinstance(self.content, str) and isinstance(other.content, str):
            self.content = self.content + other.content if other.stream else other.content
        elif isinstance(self.content, Dict) and isinstance(other.content, Dict):
            self.content = self.content | other.content
        elif isinstance(self.content, BaseResponse) and isinstance(other.content, BaseResponse):
            self.content += other.content
        else:
            if other.content:
                self.content = other.content

        if other.state:
            self.state = other.state
        if other.name:
            self.name = other.name

        return self

    def __eq__(self, other: Union[BaseResponse, List[BaseResponse], None]) -> bool:
        if other is None or not isinstance(other, BaseResponse):
            return False

        if self.id != other.id or self.state != other.state or self.name != other.name:
            return False

        if isinstance(self.content, List) and isinstance(other.content, List):
            if len(self.content) != len(other.content):
                return False

            self_content_dict = {item.id: item for item in self.content}
            other_content_dict = {item.id: item for item in other.content}

            if self_content_dict.keys() != other_content_dict.keys():
                return False

            return all(self_content_dict[key] == other_content_dict[key] for key in self_content_dict)
        else:
            return self.content == other.content


@dataclass_json
@dataclass
class ProgressResponse(BaseResponse):
    def __post_init__(self):
        self.name = 'progress'
        self.state = None
        self.content = None
        self.stream = False
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def handler(self, event: StreamEvent) -> Optional[BaseResponse]:
        content = ''
        state_changed = False
        for step in steps:
            if step.id in event['name']:
                if event['event'] == 'on_chain_start':
                    step.status = 'Running'
                    state_changed = True
                elif event['event'] == 'on_chain_stream':
                    step.status = 'Streaming'
                    state_changed = True
                elif event['event'] == 'on_chain_end':
                    step.status = 'Сompleted'

        if event['name'] == 'master':
            if event['event'] == 'on_chain_start':
                self.state = 'running'
                state_changed = True
            if event['event'] == 'on_chain_end':
                self.state = 'complete'
                state_changed = True

        if state_changed:
            for step in steps:
                if step.status == 'Waiting':
                    content += f":gray[:material/pending:] {step.name}\n\n"
                if step.status == 'Running' or step.status == 'Streaming':
                    content += f":orange[:material/sprint:] {step}\n\n"
                elif step.status == 'Сompleted':
                    content += f":green[:material/check:] {step.name}\n\n"
            self.content = content
            return BaseResponse(id=self.id,
                                name=self.name,
                                state=self.state,
                                content=self.content,
                                stream=self.stream)
        return BaseResponse(id=self.id)


@dataclass_json
@dataclass
class AnalyzeResponse(BaseResponse):
    def __post_init__(self):
        self.name = None
        self.state = None
        self.content = None
        self.stream = True
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def handler(self, event: StreamEvent) -> Optional[BaseResponse]:
        if 'print' in event['tags']:
            if 'chunk' in event['data']:
                if not self.content:
                    self.state = 'running'
                self.content = event['data']['chunk']
                return self
        if event['name'] == 'fedot_analyze_predictions_chain':
            if event['event'] == 'on_chain_end':
                self.state = 'complete'
                return BaseResponse(id=self.id,
                                    name=self.name,
                                    state=self.state,
                                    content=self.content,
                                    stream=self.stream)
        return BaseResponse(id=self.id)


@dataclass_json
@dataclass
class PipeLineResponse(BaseResponse):
    graph: GraphvizBuilder = field(init=False)

    def __post_init__(self):
        self.name = 'pipeline'
        self.state = None
        self.content = {
            'data': None,
            'type': 'graphviz'
        }
        self.stream = False
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.graph = GraphvizBuilder()

    def handler(self, event: StreamEvent) -> Optional[BaseResponse]:
        for step in steps:
            if step.id in event['name']:
                if event['event'] == 'on_chain_stream':
                    self.graph.add_node(step.name, fillcolor='orange')
                    self.content['data'] = self.graph.get_graph()
                    return BaseResponse(id=self.id,
                                        name=self.name,
                                        state=self.state,
                                        content=self.content,
                                        stream=self.stream)
        return BaseResponse(id=self.id)


ResponseContent: TypeAlias = Union[None, str, List[BaseResponse], BaseResponse, TypedContentResponse]
