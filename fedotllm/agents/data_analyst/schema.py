from enum import Enum
from typing import List, Literal, Optional

from litellm.types.utils import ChatCompletionMessageToolCall
from pydantic import BaseModel, Field

import litellm
import logging
from fedotllm.llm.litellm import LiteLLMModel

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """Message role options"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]  # type: ignore


class AgentState(str, Enum):
    """Agent execution states"""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class Message(BaseModel):
    """Represents a chat message in the conversation"""

    role: ROLE_TYPE = Field(...)  # type: ignore
    content: Optional[str] = Field(default=None)
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)
    base64_images: Optional[List[str]] = Field(default=None)

    def to_dict(self) -> dict:
        """Convert message to dictionary format"""
        message = {"role": self.role}
        # TODO: seems like litellm fails in {"type": "text", "text": self.content} to count tokens correctly and trim messages
        if self.base64_images:
            message["content"] = []
            if self.content is not None:
                message["content"].append({"type": "text", "text": self.content})
            message["content"].append(
                [
                    {"type": "image_url", "image_url": image_url}
                    for image_url in self.base64_images
                ]
            )
        else:
            message["content"] = self.content if self.content is not None else ""
        if self.tool_calls is not None:
            message["tool_calls"] = [
                tool_call.model_dump() for tool_call in self.tool_calls
            ]
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            message["name"] = self.name
        return message

    @classmethod
    def user_message(
        cls,
        content: str,
        name: Optional[str] = None,
        base64_images: Optional[List[str]] = None,
    ) -> "Message":
        """Create a user message"""
        return cls(
            role=Role.USER, content=content, name=name, base64_images=base64_images
        )

    @classmethod
    def system_message(cls, content: str) -> "Message":
        """Create a system message"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(
        cls,
        content: Optional[str] = None,
        name: Optional[str] = None,
        base64_images: Optional[List[str]] = None,
        tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None,
    ) -> "Message":
        """Create an assistant message"""
        return cls(
            role=Role.ASSISTANT,
            content=content,
            name=name,
            base64_images=base64_images,
            tool_calls=tool_calls,
        )

    @classmethod
    def tool_message(
        cls,
        content: str,
        name,
        tool_call_id: str,
        base64_images: Optional[List[str]] = None,
    ) -> "Message":
        """Create a tool message"""
        return cls(
            role=Role.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            base64_images=base64_images,
        )


class Memory(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    llm: LiteLLMModel
    max_tokens: int = Field(default=8000)  # Default max tokens if not provided by model

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        if self.llm is not None and self.llm.model_max_input_tokens is not None:
            self.max_tokens = self.llm.model_max_input_tokens
            logger.info(
                f"Setting max_tokens to {self.max_tokens} from model configuration"
            )
        else:
            logger.warning(f"Using default max_tokens: {self.max_tokens}")

    def add_message(self, message: Message) -> None:
        """Add a message to memory"""
        self.messages.append(message)

    def add_messages(self, messages: List[Message]) -> None:
        """Add multiple messages to memory"""
        self.messages.extend(messages)

    def get_messages_token_count(self) -> int:
        """Get the token count of the messages"""
        print(self.to_dict_list())
        return litellm.utils.token_counter(
            model=self.llm.model, messages=self.to_dict_list()
        )

    def clear(self) -> None:
        """Clear all messages"""
        self.messages.clear()

    def get_recent_messages(self, n: int) -> List[Message]:
        """Get n most recent messages"""
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        """Convert messages to list of dicts"""
        return [msg.to_dict() for msg in self.messages]
