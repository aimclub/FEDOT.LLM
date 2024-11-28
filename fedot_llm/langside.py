from dataclasses import dataclass, field
from typing import Any, Generator, Optional, ParamSpec, TypeVar, Callable, TypedDict, Literal, List, DefaultDict
from contextvars import ContextVar
import threading
import queue
from fedot_llm.utils.singleton import SingletonMeta
from functools import wraps
import asyncio
from uuid import uuid4

P = ParamSpec("P")
R = TypeVar("R")

class EventRecord(TypedDict):
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str
    status: Literal["start", "end", "unknown"]
    inputs: dict
    output: Any
    nesting: int = field(default=0)
    metadata: dict = field(default_factory=dict)


_inside_stack_context: ContextVar[
    List[EventRecord]
] = ContextVar("inside_stack_context", default=[])


@dataclass
class Langside(metaclass=SingletonMeta):
    _event_queue: queue.Queue = field(default_factory=queue.Queue)
    _end_event: threading.Event = field(default_factory=threading.Event)

    def add_event(self, event: Any) -> None:
        self._event_queue.put(event)

    def stream(self) -> Generator[Any, None, None]:
        while not self._end_event.is_set():
            try:
                event = self._event_queue.get(timeout=0.1)
                yield event
            except queue.Empty:
                continue

    def end(self) -> None:
        self._end_event.set()

    def inside(self,
               func: Optional[Callable[P, R]] = None,
               *,
               name: Optional[str] = None) -> Callable[[Callable[P, R]], Callable[P, R]]:

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            return (
                self._async_inside(
                    func,
                    name=name,
                )
                if asyncio.iscoroutinefunction(func)
                else self._sync_inside(
                    func,
                    name=name,
                )
            )
        return decorator(func) if func else decorator
    
    def _prepare_call(self, 
                      func: Callable[P, R],
                      name: Optional[str] = None) -> EventRecord:
        try:
            stack = _inside_stack_context.get()
            event = EventRecord(name=name or func.__name__, status="unknown", inputs={}, output=None, nesting=len(stack))
            _inside_stack_context.set(stack + [event])
            return event
        except Exception as e:
            raise e
            
    def _end_call(self, event: EventRecord) -> None:
        try:
            stack = _inside_stack_context.get()
            _inside_stack_context.set(stack[:-1])
        except Exception as e:
            raise e
        
    def update_event(self, 
                     name: Optional[str] = None,
                     status: Optional[Literal["start", "end", "unknown"]] = None,
                     inputs: Optional[dict] = None,
                     output: Optional[Any] = None,
                     metadata: Optional[dict] = None,
                     nesting: Optional[int] = None) -> None:
        try:
            stack = _inside_stack_context.get()
            event = stack[-1] if stack else None
            if not event:
                raise ValueError("No event found in the current context")
            
            update_params = {
                k: v
                for k, v in {
                    "name": name,
                    "status": status,
                    "inputs": inputs,
                    "output": output,
                    "metadata": metadata,
                    "nesting": nesting,
                }.items()
                if v is not None
            }
            event.update(update_params)
        except Exception as e:
            raise e
            

    def _async_inside(self,
                      func: Callable[P, R],
                      name: Optional[str] = None) -> Callable[P, R]:

        @wraps(func)
        async def wrapper(*args, **kwargs):

            event = self._prepare_call(func=func, name=name)
            event["status"] = "start"
            event["inputs"] = {
                "args": args,
                "kwargs": kwargs
            }

            self.add_event(event)

            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                raise e

            stack = _inside_stack_context.get()
            event = stack[-1] if stack else None
            if not event:
                raise ValueError("No event found in the current context")
            
            event["output"] = result
            event["status"] = "end"
            self.add_event(event)
            self._end_call(event)

            return result
        return wrapper

    def _sync_inside(self,
                     func: Callable[P, R],
                     name: Optional[str] = None) -> Callable[P, R]:

        @wraps(func)
        def wrapper(*args, **kwargs):

            event = self._prepare_call(func=func, name=name)
            event["status"] = "start"
            event["inputs"] = {
                    "args": args,
                    "kwargs": kwargs
            }

            self.add_event(event)

            try:
                result = func(*args, **kwargs)
            except Exception as e:
                raise e

            
            event["output"] = result
            event["status"] = "end"


            self.add_event(event)
            self._end_call(event)
            
            return result
        return wrapper


langside = Langside()
inside = langside.inside
