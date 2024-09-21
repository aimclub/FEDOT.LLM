from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from hashlib import sha256

from dataclasses_json import dataclass_json
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.schema import StreamEvent
from typing_extensions import (List, Optional,
                               TypeAlias, TypedDict, Union, Dict)

from fedot_llm.ai.actions import Actions, Action
from fedot_llm.ai.chains.analyze import AnalyzeFedotResultChain
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
    root: BaseResponse = BaseResponse()
    context: List[BaseResponse] = field(default_factory=list)

    def clean(self):
        self.context = []

    def append(self, item: BaseResponse):
        self.context.append(item)

    def pack(self) -> BaseResponse:
        self.root.content = self.context
        return self.root


class UIElement(ABC):

    @abstractmethod
    def register_hooks(self, response: Response, actions: Actions) -> None:
        """Method that register actions hooks"""


def trim_string(input_string: str, max_length: int):
    if len(input_string) <= max_length:
        return input_string
    else:
        return input_string[:max_length - 3] + '...'


class MessagesHandler(BaseResponse):

    def message_handler(self, response: Response):
        subscribe_events = ['SupervisorAgent', 'ResearcherAgent', 'AutoMLAgent']
        content = []
        message_idx = set()

        def handler(event: StreamEvent):
            nonlocal subscribe_events, content
            event_name = event.get('name', '')
            data = event.get('data', {})
            new_messages = []
            if event_name in subscribe_events:
                if data:
                    output = data.get('output', None)
                    if output is not None:
                        if isinstance(output, dict):
                            messages = output.get('messages', None)
                            if messages is not None:
                                if isinstance(messages, list):
                                    for message in messages:
                                        if isinstance(message, AIMessage) or isinstance(message, HumanMessage):
                                            if message.id not in message_idx:
                                                message_idx.add(message.id)
                                                new_messages.append(message)
                                else:
                                    if isinstance(messages, AIMessage) or isinstance(messages, HumanMessage):
                                        if messages.id not in message_idx:
                                            message_idx.add(messages.id)
                                            new_messages.append(messages)
                                for message in new_messages:
                                    if isinstance(message, AIMessage):
                                        content.append("---\n\n**Supervisor**")
                                        content.append(message.content)
                                    if isinstance(message, HumanMessage):
                                        if message.name:
                                            content.append(f"---\n\n**{message.name}**")
                                            content.append(message.content)

                            if len(new_messages) > 0:
                                response.append(BaseResponse(id=self.id,
                                                             name=self.name,
                                                             state=self.state,
                                                             content='\n\n'.join(content),
                                                             stream=self.stream))

        return handler


@dataclass_json
@dataclass
class AnalyzeResponse(BaseResponse, UIElement):
    name: Optional[str] = field(init=False, default=None)
    state: Optional[ResponseState] = field(init=False, default=None)
    content: Optional[str] = field(init=False, default=None)
    stream: bool = field(init=False, default=True)

    def __post_init__(self):
        super().__post_init__()

    def register_hooks(self, response: Response, actions: Actions) -> None:

        def on_stream_hook(event: StreamEvent, action: Action) -> None:
            if action.state == 'Streaming':
                if 'data' in event and 'chunk' in event['data']:
                    response.append(BaseResponse(id=self.id,
                                                 name=self.name,
                                                 state=self.state,
                                                 content=event['data']['chunk'],
                                                 stream=self.stream))

        def on_change_hook(_: StreamEvent, action: Action) -> None:
            if action.state == 'Running':
                self.state = ResponseState.RUNNING
            if action.state == 'Completed':
                self.state = ResponseState.COMPLETE
            response.append(BaseResponse(id=self.id,
                                         name=self.name,
                                         state=self.state,
                                         content=None,
                                         stream=self.stream))

        if AnalyzeFedotResultChain.__name__ in actions.records:
            actions.records[AnalyzeFedotResultChain.__name__].on_stream.append(on_stream_hook)
            actions.records[AnalyzeFedotResultChain.__name__].on_change.append(on_change_hook)


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
                    langgraph_step = event_metadata["langgraph_step"]
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
                            new_node = Node(name=event_name, attrs={'label': event_name,
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
                            nesting_graphs[-1].add_node(Node(name=event_name, attrs={'label': event_name,
                                                                                     'color': BSColors.SUCCESS.value}))
                        content['data'] = nesting_graphs[0].compile().source

                        response.append(BaseResponse(id=self.id,
                                                     name=self.name,
                                                     state=self.state,
                                                     content=content,
                                                     stream=self.stream))

        return handler


@dataclass_json
@dataclass
class PipeLineResponse(BaseResponse, UIElement):
    graph: GraphvizBuilder = field(init=False)
    name: Optional[str] = field(init=False, default='graph')
    state: Optional[ResponseState] = field(init=False, default=None)
    stream: bool = field(init=False, default=False)

    def __post_init__(self):
        self.content = {
            'data': None,
            'type': 'graphviz'
        }
        super().__post_init__()
        self.graph = GraphvizBuilder(
            graph_type='digraph',
            graph_attr={
                'bgcolor': 'transparent',
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

    def register_hooks(self, response: Response, actions: Actions) -> None:
        content: TypedContentResponse = {
            'data': None,
            'type': 'graphviz'
        }

        def on_change_hook(_: StreamEvent, action: Action) -> None:
            nonlocal content
            if action.state in ['Running', 'Completed']:
                if action.state == 'Running':
                    new_node = Node(name=action.id, attrs={'label': action.name,
                                                           'color': BSColors.PRIMARY.value})
                    if len(self.graph.edges) > 0:
                        prev_node = self.graph.edges[-1].dst
                        self.graph.add_edge(Edge(src=prev_node, dst=new_node))
                    elif len(self.graph.edges) == 0 and len(self.graph.nodes) == 1:
                        prev_node = list(self.graph.nodes.values())[0]
                        self.graph.add_edge(Edge(src=prev_node, dst=new_node))
                    self.graph.add_node(new_node)
                if action.state == 'Completed':
                    self.graph.add_node(Node(name=action.id, attrs={'label': action.name,
                                                                    'color': BSColors.SUCCESS.value}))
                content['data'] = self.graph.compile().source.replace('\n\t', ' ')
                response.append(BaseResponse(id=self.id,
                                             name=self.name,
                                             state=self.state,
                                             content=content,
                                             stream=self.stream))

        for action in actions.records.values():
            action.on_change.append(on_change_hook)


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
