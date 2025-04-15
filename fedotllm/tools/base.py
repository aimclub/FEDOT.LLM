from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BaseTool(ABC, BaseModel):
    name: str
    description: str
    parameters: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True

    async def __call__(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Observation:
        """Execute the tool with given parameters."""

    def to_param(self) -> Dict:
        """Convert tool to function call format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Observation(BaseModel):
    """Represents the result of a tool execution."""

    is_success: bool = Field(default=True)
    message: str = Field(default="")
    base64_images: Optional[List[str]] = Field(default=None)


class ToolError(Observation):
    is_success: bool = Field(default=False)
    message: str = Field(default="")
    base64_images: Optional[List[str]] = Field(default=None)
