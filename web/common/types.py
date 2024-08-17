from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from dataclasses_json import dataclass_json
from langchain_core.runnables.schema import StreamEvent
from typing_extensions import (Callable, List, Literal, NotRequired, Optional,
                               TypeAlias, TypedDict, Union)

from fedot_llm.chains import steps

# ResponseContent: TypeAlias = Union[None, str, List[BaseResponse], BaseResponse]
States: TypeAlias = Literal["running", "complete", "error"]


class RequestFedotLLM(TypedDict):
    msg: str


@dataclass_json
@dataclass
class BaseResponse:
    state: Optional[States] = None
    name: Optional[str] = None
    content: Optional[Union[str, List[BaseResponse]]] = None
    stream: bool = False

    def handler(self, event: StreamEvent) -> BaseResponse:
        """Method that return stage handler"""
        return self

    def __add__(self, other: BaseResponse):
        if not isinstance(other, BaseResponse):
            raise ArithmeticError(
                "Правый операнд должен быть типом BaseResponse")
        if isinstance(self.content, List) and isinstance(other.content, List):
            content = []
            for old, new in zip(self.content, other.content):
                content.append(old + new)
        elif isinstance(self.content, str) and isinstance(other.content, str):
            if other.stream:
                content = self.content + other.content
            else:
                content = other.content
        else:
            if other.content:
                content = other.content
            else:
                content = self.content

        state, name = self.state, self.name
        if other.state:
            state = other.state
        if other.name:
            name = other.name
        if other.content:
            content = content
        return BaseResponse(state=state, name=name, content=content)

    def __iadd__(self, other: BaseResponse):
        if not isinstance(other, BaseResponse):
            raise ArithmeticError(
                "Правый операнд должен быть типом BaseResponse")
        if isinstance(self.content, List) and isinstance(other.content, List):
            for old, new in zip(self.content, other.content):
                old += new
            content = self.content
        elif isinstance(self.content, str) and isinstance(other.content, str):
            if other.stream:
                content = self.content + other.content
            else:
                content = other.content
        else:
            content = other.content
        if other.state:
            self.state = other.state
        if other.name:
            self.name = other.name
        if other.content:
            self.content = content
        return self


@dataclass_json
@dataclass
class ProgressResponse(BaseResponse):
    def __post_init__(self):
        self.name = 'progress'
        self.state = None
        self.content = None
        self.stream = False

    def handler(self, event: StreamEvent) -> BaseResponse:
        content = ''
        for step in steps:
            if step.id in event['name']:
                if event['event'] == 'on_chain_start':
                    step.status = 'Running'
                elif event['event'] == 'on_chain_stream':
                    step.status = 'Streaming'
                elif event['event'] == 'on_chain_end':
                    step.status = 'Сompleted'
        for step in steps:
            if step.status == 'Waiting':
                content += f":gray[:material/pending:] {step.name}\n\n"
            if step.status == 'Running' or step.status == 'Streaming':
                content += f":orange[:material/sprint:] {step}\n\n"
            elif step.status == 'Сompleted':
                content += f":green[:material/check:] {step.name}\n\n"

        if event['name'] == 'master':
            if event['event'] == 'on_chain_start':
                self.state = 'running'
            if event['event'] == 'on_chain_end':
                self.state = 'complete'

        self.content = content  # from one stable state to another
        return self

@dataclass_json
@dataclass
class AnalyzeResponse(BaseResponse):
    def __post_init__(self):
        self.name = None
        self.state = None
        self.content = None
        self.stream = False
    
    def handler(self, event: StreamEvent) -> BaseResponse:
        if 'print' in event['tags']:
            if 'chunk' in event['data']:
                if not self.content:
                    self.state = 'running'
                    self.content = event['data']['chunk']
                else:
                    self.content += event['data']['chunk']
        if event['name'] == 'fedot_analyze_predictions_chain':
            if event['event'] == 'on_chain_end':
                    self.state = 'complete'
        return self