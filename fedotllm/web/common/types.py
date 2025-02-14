from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from hashlib import sha256

from deep_translator import GoogleTranslator
from fedotllm.web.backend.utils.graphviz_builder import (Edge, GraphvizBuilder,
                                                         Node)
from fedotllm.web.common.colors import BSColors, STColors
from fedotllm.web.frontend.localization import lclz
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.schema import StreamEvent
from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)
from typing_extensions import (ClassVar, Dict, List, Literal, Optional, Set,
                               TypeAlias, TypedDict, Union)

ResponseContent: TypeAlias = Union[None, str,
                                   List['BaseResponse'], 'BaseResponse', 'TypedContentResponse']

Lang: TypeAlias = Literal['en', 'ru']


class ResponseState(Enum):
    RUNNING = 'running'
    COMPLETE = 'complete'


class TypedContentResponse(TypedDict):
    data: ResponseContent
    type: str


class BaseResponse(BaseModel):
    id: Optional[str] = Field(default=None)
    state: Optional[ResponseState] = None
    name: Optional[str] = None
    content: ResponseContent = None
    stream: bool = False

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode='after')
    def set_id(self) -> BaseResponse:
        if self.id is None:
            self.id = self.new_id(self.name)
        return self

    @classmethod
    def new_id(cls, name: Optional[str] = '') -> str:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        random_part = sha256(
            f'{cls.__name__}{date}{random.random()}'.encode()).hexdigest()
        if name:
            return f"{name}{cls.__name__}{random_part}"
        else:
            return f"{cls.__name__}{random_part}"

    def handler(self, _: StreamEvent) -> Optional[BaseResponse]:
        """Method that return stage handler"""
        return self

    def __iadd__(self, other: Union[BaseResponse, None]) -> BaseResponse:
        if other is None:
            return self
        if not isinstance(other, BaseResponse):
            raise TypeError(
                f"Right operand must be of type BaseResponse: {type(other)}")

        if self.id != other.id:
            raise ValueError("Cannot add objects with different ids")

        if isinstance(self.content, List) and isinstance(other.content, List):
            self_content_dict = {
                item.id: item for item in self.content if isinstance(item, BaseResponse)}
            for new_item in other.content:
                if new_item.id in self_content_dict:
                    self_content_dict[new_item.id] += new_item
                else:
                    self.content.append(new_item)
        elif isinstance(self.content, str) and isinstance(other.content, str):
            self.content = self.content + other.content if other.stream else other.content
        elif isinstance(self.content, Dict) and isinstance(other.content, Dict):
            self.content = {**self.content, **other.content}
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

            self_content_dict = {
                item.id: item for item in self.content if isinstance(item, BaseResponse)}
            other_content_dict = {
                item.id: item for item in other.content if isinstance(item, BaseResponse)}

            if self_content_dict.keys() != other_content_dict.keys():
                return False

            return all(self_content_dict[key] == other_content_dict[key] for key in self_content_dict)
        else:
            return self.content == other.content


@dataclass
class Response:
    root: BaseResponse = field(default_factory=BaseResponse)
    context: List[BaseResponse] = field(default_factory=list)

    def clean(self):
        self.context = []

    def append(self, item: BaseResponse):
        self.context.append(item)

    def pack(self) -> BaseResponse:
        self.root.content = self.context
        return self.root


def trim_string(input_string: str, max_length: int) -> str:
    return input_string if len(input_string) <= max_length else input_string[:max_length - 3] + '...'


class MessagesHandler(BaseResponse):
    SUBSCRIBE_EVENTS: ClassVar[List[str]] = [
        'SupervisorAgent', 'ResearcherAgent', 'AutoMLAgent']
    SEPARATOR: ClassVar[str] = "---\n\n"
    lang: Lang = Field(default='en')

    def message_handler(self, response: Response) -> callable:
        content: List[str] = []
        message_idx: Set[str] = set()

        def process_event(event: StreamEvent) -> None:
            if event['name'] not in self.SUBSCRIBE_EVENTS or not event.get('data'):
                return

            output = event['data'].get('output')
            if not isinstance(output, dict):
                return

            messages = output.get('messages')
            if messages is None:
                return

            new_messages = self._process_messages(messages, message_idx)
            self._update_content(new_messages, content)

            if new_messages:
                self._append_response(response, content)

        return process_event

    @staticmethod
    def _process_messages(messages: Union[List, AIMessage, HumanMessage], message_idx: Set[str]) -> List[
            Union[AIMessage, HumanMessage]]:
        new_messages = []
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, (AIMessage, HumanMessage)) and message.id not in message_idx:
                    message_idx.add(message.id)
                    new_messages.append(message)
        elif isinstance(messages, (AIMessage, HumanMessage)) and messages.id not in message_idx:
            message_idx.add(messages.id)
            new_messages.append(messages)
        return new_messages

    @classmethod
    def _update_content(cls, new_messages: List[Union[AIMessage, HumanMessage]], content: List[str]) -> None:
        for message in new_messages:
            if isinstance(message, AIMessage):
                content.append(f"{cls.SEPARATOR}**Supervisor**")
                content.append(message.content)
            elif isinstance(message, HumanMessage) and message.name:
                content.append(f"{cls.SEPARATOR}**{message.name}**")
                content.append(message.content)

    def _append_response(self, response: Response, content: List[str]) -> None:
        response.append(BaseResponse(
            id=self.id,
            name=self.name,
            state=self.state,
            content=GoogleTranslator(source='en', target=self.lang).translate(
                '\n\n'.join(content)),
            stream=self.stream
        ))


class GraphResponse(BaseResponse):
    name: Optional[str] = Field(default='graph', init=False)
    state: Optional['ResponseState'] = Field(default=None, init=False)
    stream: bool = Field(default=False, init=False)
    lang: Lang = Field(default='en')

    @field_validator('content', mode='before')
    @classmethod
    def set_content(cls, v: Optional[ResponseContent]) -> TypedContentResponse:
        if v is None:
            return {
                'data': None,
                'type': 'graphviz'
            }
        return v

    @model_validator(mode='after')
    def set_name(self) -> GraphResponse:
        self.name = lclz[self.lang]['GRAPH']
        return self

    @staticmethod
    def init_default_graph(name: str = '') -> GraphvizBuilder:
        label = '' if name == '__start__' else name
        return GraphvizBuilder(
            name=name,
            graph_type='digraph',
            graph_attr={
                'bgcolor': 'transparent',
                'label': label,
                'rankdir': 'LR',
            },
            edge_attr={
                'color': BSColors.SECONDARY.value
            },
            node_attr={
                'shape': 'box',
                'color': STColors.SECONDARY.value,
                'fontcolor': STColors.TEXT.value,
                'style': 'filled'
            }
        )

    def graph_handler(self, response: Response):
        content: TypedContentResponse = {
            'data': None,
            'type': 'graphviz'
        }
        nesting_graphs: List[GraphvizBuilder] = []

        def handler(event: StreamEvent):
            nonlocal content
            event_name = event['name']
            event_state = event['event']
            event_metadata = event.get("metadata", None)
            if event_metadata:
                langgraph_node = event_metadata.get("langgraph_node", None)
                if langgraph_node:
                    ns = event_metadata.get("langgraph_checkpoint_ns", None)
                    if ns:
                        ns = str(ns).split(":")[0]
                    if not nesting_graphs:
                        new_graph = self.init_default_graph(ns)
                        nesting_graphs.append(new_graph)
                    elif ns != nesting_graphs[-1].name:
                        if len(nesting_graphs) < 3 or ns != nesting_graphs[-2].name:
                            new_graph = self.init_default_graph(ns)
                            nesting_graphs[-1].subgraphs.append(new_graph)
                            nesting_graphs.append(new_graph)
                        else:
                            nesting_graphs.pop()

                    if langgraph_node == event_name and (event_name != '__start__' or ns == '__start__'):
                        if event_state == "on_chain_start":
                            new_node = Node(name=event_name, attrs={'label': str(event_name),
                                                                    'color': BSColors.PRIMARY.value})
                            if nesting_graphs[-1].edges:
                                prev_node = nesting_graphs[-1].edges[-1].dst
                                nesting_graphs[-1].add_edge(
                                    Edge(src=prev_node, dst=new_node))
                            elif not nesting_graphs[-1].edges and len(nesting_graphs[-1].nodes) == 1:
                                prev_node = next(
                                    iter(nesting_graphs[-1].nodes.values()))
                                nesting_graphs[-1].add_edge(
                                    Edge(src=prev_node, dst=new_node))
                            elif len(nesting_graphs) > 1:
                                if nesting_graphs[-2].edges:
                                    prev_node = nesting_graphs[-2].edges[-1].dst
                                    nesting_graphs[-2].add_edge(
                                        Edge(src=prev_node, dst=new_node))
                                if not nesting_graphs[-2].edges and len(nesting_graphs[-2].nodes) == 1:
                                    prev_node = next(
                                        iter(nesting_graphs[-2].nodes.values()))
                                    nesting_graphs[-2].add_edge(
                                        Edge(src=prev_node, dst=new_node))
                            nesting_graphs[-1].add_node(new_node)
                        elif event_state == "on_chain_end":
                            nesting_graphs[-1].add_node(Node(name=event_name, attrs={'label': str(event_name),
                                                                                     'color': BSColors.SUCCESS.value}))
                        content['data'] = nesting_graphs[0].compile().source

                        response.append(BaseResponse(
                            id=self.id,
                            name=self.name,
                            state=self.state,
                            content=content,
                            stream=self.stream
                        ))

        return handler
