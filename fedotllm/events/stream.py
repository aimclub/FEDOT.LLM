import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fedotllm.events.event import Event, EventSource
from typing import Callable, Any, Iterable, AsyncIterator
import logging
from fedotllm.storage.location import (
    get_conversation_events_dir,
    get_id_from_filename,
    get_filename_for_id,
)
from fedotllm.storage.base import FileStore
from enum import Enum
from functools import partial
from datetime import datetime

logger = logging.getLogger(__name__)


class EventStreamSubscriber(str, Enum):
    AGENT_CONTROLLER = "agent_controller"
    SECURITY_ANALYZER = "security_analyzer"
    RESOLVER = "openhands_resolver"
    SERVER = "server"
    RUNTIME = "runtime"
    MEMORY = "memory"
    MAIN = "main"
    TEST = "test"
    
class AsyncEventStreamWrapper:
    def __init__(self, event_stream: 'EventStream', *args: Any, **kwargs: Any) -> None:
        self.event_stream = event_stream
        self.args = args
        self.kwargs = kwargs

    async def __aiter__(self) -> AsyncIterator[Event]:
        loop = asyncio.get_running_loop()

        # Create an async generator that yields events
        for event in self.event_stream.get_events(*self.args, **self.kwargs):
            # Run the blocking get_events() in a thread pool
            def get_event(e: Event = event) -> Event:
                return e

            yield await loop.run_in_executor(None, get_event)


class EventStream:
    sid: str
    user_id: str | None
    file_store: FileStore
    _subscribers: dict[str, dict[str, Callable]]
    _lock: threading.Lock
    _queue: queue.Queue[Event]
    _queue_thread: threading.Thread
    _queue_loop: asyncio.AbstractEventLoop | None
    _thread_pools: dict[str, dict[str, ThreadPoolExecutor]]
    _thread_loops: dict[str, dict[str, asyncio.AbstractEventLoop]]
    _cur_id: int

    def __init__(self, sid: str, file_store: FileStore, user_id: str | None = None):
        self.sid = sid
        self.user_id = user_id
        self.file_store = file_store
        self._queue = queue.Queue()
        self._stop_flag = threading.Event()
        self._thread_pools = {}
        self._thread_loops = {}
        self._queue_loop = None
        self._queue_thread = threading.Thread(target=self._run_queue_loop)
        self._queue_thread.daemon = True
        self._queue_thread.start()
        self._subscribers = {}
        self._lock = threading.Lock()
        self._cur_id = 0

    def __post_init__(self) -> None:
        events = []
        try:
            events_dir = get_conversation_events_dir(self.sid, self.user_id)
            events += self.file_store.list(events_dir)
        except FileNotFoundError:
            logger.debug(f"No events found for session {self.sid} at {events_dir}")

        if not events:
            self._cur_id = 0
            return

        # if we have events, we need to find the highest id to prepare for new events
        for event_path in events:
            id = get_id_from_filename(event_path)
            if id >= self._cur_id:
                self._cur_id = id + 1

    def close(self) -> None:
        self._stop_flag.set()
        if self._queue_thread.is_alive():
            self._queue_thread.join()

        subscriber_ids = list(self._subscribers.keys())
        for subscriber_id in subscriber_ids:
            callback_ids = list(self._subscribers[subscriber_id].keys())
            for callback_id in callback_ids:
                self._clean_up_subscriber(subscriber_id, callback_id)

        # Clear queue
        while not self._queue.empty():
            self._queue.get()

    def _init_thread_loop(self, subscriber_id: str, callback_id: str) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if subscriber_id not in self._thread_loops:
            self._thread_loops[subscriber_id] = {}
        self._thread_loops[subscriber_id][callback_id] = loop

    def _clean_up_subscriber(self, subscriber_id: str, callback_id: str) -> None:
        if subscriber_id not in self._subscribers:
            logger.warning(f"Subscriber not found during cleanup: {subscriber_id}")
            return
        if callback_id not in self._subscribers[subscriber_id]:
            logger.warning(f"Callback not found during cleanup: {callback_id}")
            return
        if (
            subscriber_id in self._thread_loops
            and callback_id in self._thread_loops[subscriber_id]
        ):
            loop = self._thread_loops[subscriber_id][callback_id]
            try:
                loop.stop()
                loop.close()
            except Exception as e:
                logger.warning(
                    f"Error closing loop for {subscriber_id}/{callback_id}: {e}"
                )
            del self._thread_loops[subscriber_id][callback_id]

        if (
            subscriber_id in self._thread_pools
            and callback_id in self._thread_pools[subscriber_id]
        ):
            pool = self._thread_pools[subscriber_id][callback_id]
            pool.shutdown()
            del self._thread_pools[subscriber_id][callback_id]

        del self._subscribers[subscriber_id][callback_id]

    def subscribe(
        self,
        subscriber_id: EventStreamSubscriber,
        callback: Callable[[Event], None],
        callback_id: str,
    ) -> None:
        initializer = partial(self._init_thread_loop, subscriber_id, callback_id)
        pool = ThreadPoolExecutor(max_workers=1, initializer=initializer)
        if subscriber_id not in self._subscribers:
            self._subscribers[subscriber_id] = {}
            self._thread_pools[subscriber_id] = {}

        if callback_id in self._subscribers[subscriber_id]:
            raise ValueError(
                f"Callback ID on subscriber {subscriber_id} already exists: {callback_id}"
            )

        self._subscribers[subscriber_id][callback_id] = callback
        self._thread_pools[subscriber_id][callback_id] = pool

    def unsubscribe(
        self, subscriber_id: EventStreamSubscriber, callback_id: str
    ) -> None:
        if subscriber_id not in self._subscribers:
            logger.warning(f"Subscriber not found during unsubscribe: {subscriber_id}")
            return

        if callback_id not in self._subscribers[subscriber_id]:
            logger.warning(f"Callback not found during unsubscribe: {callback_id}")
            return
        self._clean_up_subscriber(subscriber_id, callback_id)

    def add_event(self, event: Event, source: EventSource) -> None:
        if event.id != Event.INVALID_ID:
            raise ValueError(
                f"Event already has an ID:{event.id}. It was probably added back to the EventStream from inside a handler, triggering a loop."
            )
        with self._lock:
            event._id = self._cur_id  # type: ignore [attr-defined]
            self._cur_id += 1
        logger.debug(f"Adding {type(event).__name__} id={event.id} from {source.name}")
        event._timestamp = datetime.now().isoformat()
        event._source = source  # type: ignore [attr-defined]
        if event.id is not None:
            self.file_store.write(
                get_filename_for_id(self.sid, event.id, self.user_id),
                event.model_dump_json(),
            )
        self._queue.put(event)

    def _run_queue_loop(self) -> None:
        self._queue_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._queue_loop)
        try:
            self._queue_loop.run_until_complete(self._process_queue())
        finally:
            self._queue_loop.close()

    async def _process_queue(self) -> None:
        while not self._stop_flag.is_set():
            event = None
            try:
                event = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # pass each event to each callback in order
            for key in sorted(self._subscribers.keys()):
                callbacks = self._subscribers[key]
                for callback_id in callbacks:
                    callback = callbacks[callback_id]
                    pool = self._thread_pools[key][callback_id]
                    future = pool.submit(callback, event)
                    future.add_done_callback(self._make_error_handler(callback_id, key))

    def get_events(self, start_id: int, end_id: int) -> Iterable[Event]:
        """Get events from the file store.
         Yields:
            Events from the stream that match the criteria.
        """
        for id in range(start_id, end_id):
            try:
                yield self.get_event(id)
            except FileNotFoundError:
                logger.warning(f"Event with id {id} not found")

    def get_event(self, id: int) -> Event:
        filename = get_filename_for_id(self.sid, id, self.user_id)
        try:
            content = self.file_store.read(filename)
            data = Event.model_validate_json(content)
            return data
        except FileNotFoundError:
            raise ValueError(f"Event with id {id} not found")

    def _make_error_handler(
        self, callback_id: str, subscriber_id: str
    ) -> Callable[[Any], None]:
        def _handle_callback_error(fut: Any) -> None:
            try:
                # This will raise any exception that occurred during callback execution
                fut.result()
            except Exception as e:
                logger.error(
                    f"Error in event callback {callback_id} for subscriber {subscriber_id}: {str(e)}",
                )
                # Re-raise in the main thread so the error is not swallowed
                raise e

        return _handle_callback_error
