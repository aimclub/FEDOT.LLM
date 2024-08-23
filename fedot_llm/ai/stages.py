from __future__ import annotations
from pydantic import BaseModel, Field
from typing_extensions import Literal, Dict, Any, List, Union, Optional
from fedot_llm.ai.chains.base import BaseRunnableChain


class Stages(BaseModel):
    data: Dict[str, Stage] = Field(default_factory=dict, init=False)

    def __init__(self, stages: List[Stage], **kwargs):
        super().__init__(**kwargs)
        if stages:
            self.data = {stage.id: stage for stage in stages}

    def append(self, stage: Stage):
        self.data[stage.id] = stage

    def extend(self, stages: Stages):
        self.data = self.data | stages.data
        
    def __call__(self):
        return self.data


class Stage(BaseModel):
    id: str
    name: str
    state: Literal['Waiting', 'Running', 'Streaming', 'Completed'] = Field(default='Waiting')
    
    @classmethod
    def from_chain(cls, chain: type[BaseRunnableChain]):
        id = chain.__name__
        if not chain.__doc__:
            raise ValueError(f"{chain.__name__} has not docstring")
        name = chain.__doc__.split('\n')[0]
        return cls(id=id, name=name)
