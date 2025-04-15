from pydantic import BaseModel
from datetime import datetime
from enum import Enum
from typing import ClassVar


class EventSource(str, Enum):
    AGENT = "agent"
    USER = "user"
    ENVIRONMENT = "environment"


class Event(BaseModel):
    INVALID_ID: ClassVar[int] = -1

    @property
    def message(self) -> str | None:
        if hasattr(self, "_message"):
            msg = getattr(self, "_message")
            return str(msg) if msg is not None else None
        return ""

    @property
    def id(self) -> int:
        if hasattr(self, "_id"):
            id_val = getattr(self, "_id")
            return int(id_val) if id_val is not None else Event.INVALID_ID
        return Event.INVALID_ID

    @property
    def timestamp(self) -> str | None:
        if hasattr(self, "_timestamp") and isinstance(self._timestamp, str):
            ts = getattr(self, "_timestamp")
            return str(ts) if ts is not None else None
        return None

    @timestamp.setter
    def timestamp(self, value: datetime) -> None:
        if isinstance(value, datetime):
            self._timestamp = value.isoformat()

    @property
    def source(self) -> EventSource | None:
        if hasattr(self, "_source"):
            src = getattr(self, "_source")
            return EventSource(src) if src is not None else None
        return None

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        return cls(**data)
