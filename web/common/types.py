from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from hashlib import sha256

from dataclasses_json import dataclass_json
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.schema import StreamEvent
from typing_extensions import (List, Optional,
                               TypeAlias, TypedDict, Union, Dict,
                               Set)

from web.backend.utils.graphviz_builder import GraphvizBuilder, Node, Edge
from web.common.colors import BSColors, STColors


class ResponseState(Enum):
    RUNNING = 'running'
    COMPLETE = 'complete'


class RequestFedotLLM(TypedDict):
    msg: str


class TypedContentResponse(TypedDict):
    data: ResponseContent
    type: str


@dataclass_json
@dataclass
class BaseResponse:
    id: Optional[str] = None
    state: Optional[ResponseState] = None
    name: Optional[str] = None
    content: ResponseContent = None
    stream: bool = False

    def __post_init__(self):
        if not self.id:
            self.id = self.new_id(self.name)

    def new_id(self, name: Optional[str] = ''):
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if name:
            return f"{name}{self.__class__.__name__}{sha256(f'{self.__class__.__name__}{date}{random.random()}'.encode()).hexdigest()}"
        else:
            return f"{self.__class__.__name__}{sha256(f'{self.__class__.__name__}{date}{random.random()}'.encode()).hexdigest()}"

    def handler(self, _: StreamEvent) -> Optional[BaseResponse]:
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


def trim_string(input_string: str, max_length: int):
    if len(input_string) <= max_length:
        return input_string
    else:
        return input_string[:max_length - 3] + '...'


class MessagesHandler(BaseResponse):
    SUBSCRIBE_EVENTS = ['SupervisorAgent', 'ResearcherAgent', 'AutoMLAgent']
    SEPARATOR = "---\n\n"

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
            content='\n\n'.join(content),
            stream=self.stream
        ))


@dataclass
class GraphResponse(BaseResponse):
    name: Optional[str] = field(init=False, default='graph')
    state: Optional[ResponseState] = field(init=False, default=None)
    stream: bool = field(init=False, default=False)

    def __post_init__(self):
        self.content = {
            'data': None,
            'type': 'graphviz'
        }
        super().__post_init__()

    @staticmethod
    def init_default_graph(name: str = ''):
        if name == '__start__':
            label = ''
        else:
            label = name
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
            if event_metadata := event.get("metadata", None):
                if langgraph_node := event_metadata.get("langgraph_node", None):
                    ns = event_metadata["checkpoint_ns"].split(":")[0]
                    if len(nesting_graphs) == 0:
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
                            if len(nesting_graphs[-1].edges) > 0:
                                prev_node = nesting_graphs[-1].edges[-1].dst
                                nesting_graphs[-1].add_edge(Edge(src=prev_node, dst=new_node))
                            elif len(nesting_graphs[-1].edges) == 0 and len(nesting_graphs[-1].nodes) == 1:
                                prev_node = list(nesting_graphs[-1].nodes.values())[0]
                                nesting_graphs[-1].add_edge(Edge(src=prev_node, dst=new_node))
                            elif len(nesting_graphs) > 1:
                                if len(nesting_graphs[-2].edges) > 0:
                                    prev_node = nesting_graphs[-2].edges[-1].dst
                                    nesting_graphs[-2].add_edge(Edge(src=prev_node, dst=new_node))
                                if len(nesting_graphs[-2].edges) == 0 and len(nesting_graphs[-2].nodes) == 1:
                                    prev_node = list(nesting_graphs[-2].nodes.values())[0]
                                    nesting_graphs[-2].add_edge(Edge(src=prev_node, dst=new_node))
                            nesting_graphs[-1].add_node(new_node)
                        elif event_state == "on_chain_end":
                            nesting_graphs[-1].add_node(Node(name=event_name, attrs={'label': str(event_name),
                                                                                     'color': BSColors.SUCCESS.value}))
                        content['data'] = nesting_graphs[0].compile().source

                        response.append(BaseResponse(id=self.id,
                                                     name=self.name,
                                                     state=self.state,
                                                     content=content,
                                                     stream=self.stream))

        return handler


def get_logger_handler():
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(
        filename='events.log', mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    def handler(event: StreamEvent) -> None:
        logger.debug(event)

    return handler


ResponseContent: TypeAlias = Union[None, str, List[BaseResponse], BaseResponse, TypedContentResponse]
