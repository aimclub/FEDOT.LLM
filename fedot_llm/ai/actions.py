from __future__ import annotations

from langchain_core.runnables.schema import StreamEvent
from pydantic import BaseModel, Field
from typing_extensions import Literal, Dict, Any, List, Callable

from fedot_llm.ai.chains.base import BaseRunnableChain

ActionState = Literal['Waiting', 'Running', 'Streaming', 'Completed']

STATE_MAP: Dict[str, ActionState] = {
    'on_chain_start': 'Running',
    'on_chain_stream': 'Streaming',
    'on_chain_end': 'Completed'
}


class Action(BaseModel):
    id: str
    name: str
    state: ActionState = Field(default='Waiting')
    on_change: List[Callable[[StreamEvent, Action], Any]] = Field(default_factory=list)
    on_stream: List[Callable[[StreamEvent, Action], Any]] = Field(default_factory=list)

    @classmethod
    def from_chain(cls, chain: type[BaseRunnableChain]) -> Action:
        if not chain.__doc__:
            raise ValueError(f"{chain.__name__} has no docstring")
        name = chain.__doc__.split('\n')[0]
        return cls(id=chain.__name__, name=name)

    def __str__(self) -> str:
        return self.name

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Action):
            return NotImplemented
        return self.id == other.id


class Actions(BaseModel):
    records: Dict[str, Action] = Field(default_factory=dict, init=False)

    def __init__(self, actions: List[Action], **kwargs):
        super().__init__(**kwargs)
        if actions:
            self.records = {action.id: action for action in actions}

    def append(self, action: Action):
        self.records[action.id] = action

    def extend(self, actions: Actions):
        self.records = self.records | actions.records

    def __call__(self):
        return self.records

    def handler(self, event: StreamEvent) -> None:
        name = event['name']
        if name not in self.records:
            return

        event_type = event['event']
        if event_type not in STATE_MAP:
            return

        new_state = STATE_MAP[event_type]
        current_state = self.records[name].state

        if new_state != current_state:
            self.records[name].state = new_state
            if len(on_change_hooks := self.records[name].on_change) > 0:
                for hook in on_change_hooks:
                    hook(event, self.records[name])

        if len(on_stream_hooks := self.records[name].on_stream) > 0:
            # streaming generates a lot of events, let's first see if there's someone to handle them.
            if new_state == 'Streaming':
                if event['data'] and 'chunk' in event['data']:
                    for hook in on_stream_hooks:
                        hook(event, self.records[name])
