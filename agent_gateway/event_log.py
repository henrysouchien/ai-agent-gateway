from __future__ import annotations

import asyncio
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import (
  Any,
  AsyncIterator,
  Callable,
  Deque,
  Dict,
  List,
  Mapping,
  Literal,
  Optional,
  Sequence,
  Set,
  Tuple,
)


TERMINAL_EVENT_TYPES = {"stream_complete", "error"}
OnEvent = Callable[[Dict[str, Any], str], None]
PrepareEvent = Callable[[Dict[str, Any]], Dict[str, Any]]
OnEventErrorPolicy = Literal["ignore", "raise"]


@dataclass
class LogEntry:
  """Single event log entry with sequence number and timestamp."""

  seq: int
  timestamp: float
  event: Dict[str, Any]


def _copy_log_entry(entry: LogEntry) -> LogEntry:
  return LogEntry(
    seq=entry.seq,
    timestamp=entry.timestamp,
    event=deepcopy(entry.event),
  )


class EventLog:
  """Append-only event buffer for gateway streams.

  Runners write structured events here and the HTTP layer consumes them through
  `iter_from()`. The log closes automatically when a terminal event such as
  `stream_complete` or `error` is appended.
  """

  def __init__(
    self,
    *,
    on_event: OnEvent | None = None,
    session_id: str = "",
    defer_terminal_close: bool = False,
    on_event_error: OnEventErrorPolicy = "ignore",
    prepare_event: PrepareEvent | None = None,
  ) -> None:
    if on_event_error not in {"ignore", "raise"}:
      raise ValueError(
        "on_event_error must be either 'ignore' or 'raise'"
      )
    self._entries: List[LogEntry] = []
    self._closed = False
    self._has_terminal = False
    self._defer_terminal_close = bool(defer_terminal_close)
    self._next_seq = 1
    self._version = 0
    self._updated = asyncio.Event()
    self._on_event = on_event
    self._on_event_error = on_event_error
    self._prepare_event = prepare_event
    self._session_id = session_id
    self._raw_tool_inputs: Dict[str, Dict[str, Any]] = {}

  def append(self, event: Dict[str, Any]) -> Optional[LogEntry]:
    if self._closed:
      return None

    prepared_event = event
    if self._prepare_event is not None:
      try:
        prepared_event = self._prepare_event(event)
        if type(prepared_event) is not dict:
          raise TypeError(
            "prepare_event must return an exact event dictionary"
          )
      except BaseException:
        self._closed = True
        self._version += 1
        self._updated.set()
        raise

    owned_event = deepcopy(prepared_event)
    entry = LogEntry(
      seq=self._next_seq,
      timestamp=time.time(),
      event=owned_event,
    )
    self._next_seq += 1
    self._entries.append(entry)

    if self._on_event is not None:
      try:
        self._on_event(deepcopy(owned_event), self._session_id)
      except Exception:
        if self._on_event_error == "raise":
          self._closed = True
          self._version += 1
          self._updated.set()
          raise

    if entry.event.get("type") in TERMINAL_EVENT_TYPES:
      self._has_terminal = True
      if not self._defer_terminal_close:
        self._closed = True

    self._version += 1
    self._updated.set()
    return _copy_log_entry(entry)

  def close(self, error: Optional[str] = None) -> None:
    if self._closed:
      return

    if self._has_terminal:
      self._closed = True
      self._version += 1
      self._updated.set()
      return

    reason = error or "stream closed"
    self.append({"type": "error", "error": reason})
    if self._defer_terminal_close:
      self._closed = True
      self._version += 1
      self._updated.set()

  async def iter_from(self, after_seq: int = 0) -> AsyncIterator[LogEntry]:
    index = max(after_seq, 0)

    while True:
      while index < len(self._entries):
        entry = self._entries[index]
        index += 1
        yield _copy_log_entry(entry)
        if entry.event.get("type") in TERMINAL_EVENT_TYPES and not self._defer_terminal_close:
          return

      if self._closed:
        return

      version = self._version
      self._updated.clear()
      if self._version != version:
        continue
      await self._updated.wait()

  @property
  def entries(self) -> List[LogEntry]:
    return [_copy_log_entry(entry) for entry in self._entries]

  def record_raw_tool_input(
    self,
    *,
    tool_call_id: str,
    tool_name: str,
    tool_input: Mapping[str, Any],
  ) -> None:
    if not tool_call_id:
      return
    self._raw_tool_inputs[str(tool_call_id)] = {
      "tool_name": str(tool_name),
      "tool_input": deepcopy(dict(tool_input)),
    }

  @property
  def raw_tool_inputs(self) -> Dict[str, Dict[str, Any]]:
    return deepcopy(self._raw_tool_inputs)

  @property
  def closed(self) -> bool:
    return self._closed

  @property
  def has_terminal(self) -> bool:
    return self._has_terminal

  @property
  def defer_terminal_close(self) -> bool:
    return self._defer_terminal_close

  @property
  def next_seq(self) -> int:
    return self._next_seq


def log_has_terminal(log: Any) -> bool:
  has_terminal = getattr(log, "has_terminal", None)
  if has_terminal is not None:
    return bool(has_terminal)
  return bool(log.closed)


@dataclass(frozen=True)
class _BusEvent:
  event: Dict[str, Any]
  timestamp: float
  seq: int | None
  control_run_id: str | None
  terminal: bool = False


def _copy_bus_event(entry: _BusEvent) -> _BusEvent:
  return _BusEvent(
    event=deepcopy(entry.event),
    timestamp=entry.timestamp,
    seq=entry.seq,
    control_run_id=entry.control_run_id,
    terminal=entry.terminal,
  )


@dataclass(eq=False)
class _Subscriber:
  user_id: str
  control_run_id: str | None
  max_queue_size: int
  queue: Deque[_BusEvent]
  updated: asyncio.Event
  after_seq_floor: int = 0
  stop_at_terminal: bool = False
  stream_terminated: bool = False
  dropped_count: int = 0
  oldest_dropped_ts: float | None = None
  dropped_through_seq: int | None = None
  closed: bool = False

  def matches(self, user_id: str, control_run_id: str) -> bool:
    if self.user_id != user_id:
      return False
    if self.control_run_id is None:
      return True
    return self.control_run_id == control_run_id


class _SubscriberClosed(Exception):
  pass


class _SubscriberTerminated(Exception):
  pass


class EventSubscription(AsyncIterator[_BusEvent]):
  """Explicit event subscription whose close works before first iteration."""

  def __init__(
    self,
    *,
    bus: "UserEventBus",
    subscriber: _Subscriber,
    replay_events: Sequence[_BusEvent],
    truncation_event: _BusEvent | None = None,
    registered: bool,
    stop_at_terminal: bool = False,
  ) -> None:
    self._bus = bus
    self._subscriber = subscriber
    self._replay_events: Deque[_BusEvent] = deque(
      _copy_bus_event(entry) for entry in replay_events
    )
    self._truncation_event = (
      _copy_bus_event(truncation_event)
      if truncation_event is not None
      else None
    )
    self._registered = registered
    self._closed = not registered
    self._stop_at_terminal = stop_at_terminal
    self._terminal_delivered = False

  def __aiter__(self) -> "EventSubscription":
    return self

  async def __anext__(self) -> _BusEvent:
    if self._closed:
      raise StopAsyncIteration
    if self._truncation_event is not None:
      entry = self._truncation_event
      self._truncation_event = None
      return _copy_bus_event(entry)
    if self._replay_events:
      entry = self._replay_events.popleft()
      if self._stop_at_terminal and entry.terminal:
        self._terminal_delivered = True
      return _copy_bus_event(entry)
    if self._stop_at_terminal and self._terminal_delivered:
      await self.aclose()
      raise StopAsyncIteration
    try:
      entry = await self._bus._next_entry(self._subscriber)
    except _SubscriberTerminated:
      await asyncio.shield(self.aclose())
      raise StopAsyncIteration from None
    except _SubscriberClosed:
      self._closed = True
      self._registered = False
      raise StopAsyncIteration from None
    except asyncio.CancelledError:
      await asyncio.shield(self.aclose())
      raise
    if self._stop_at_terminal and entry.terminal:
      self._terminal_delivered = True
    return entry

  async def aclose(self) -> None:
    if self._closed:
      return
    self._closed = True
    if not self._registered:
      return
    self._registered = False
    await asyncio.shield(self._bus._unsubscribe(self._subscriber))


class UserEventBus:
  """Per-user in-process event bus with bounded per-run replay."""

  def __init__(
    self,
    *,
    subscriber_queue_max: int = 1000,
    replay_buffer_max: int = 5000,
  ):
    if subscriber_queue_max < 1:
      raise ValueError("subscriber_queue_max must be positive")
    if replay_buffer_max < 1:
      raise ValueError("replay_buffer_max must be positive")

    self._subscriber_queue_max = subscriber_queue_max
    self._replay_buffer_max = replay_buffer_max
    self._cleanup_delay_seconds = 60.0
    self._lock = asyncio.Lock()
    self._subscribers: Dict[str, Set[_Subscriber]] = {}
    self._replay_buffers: Dict[Tuple[str, str], Deque[_BusEvent]] = {}
    self._next_seq_by_run: Dict[Tuple[str, str], int] = {}
    self._cleanup_tasks: Dict[Tuple[str, str], asyncio.Task[None]] = {}
    self._terminated_runs: Set[Tuple[str, str]] = set()
    self._shutdown = False

  async def publish(self, user_id: str, control_run_id: str, event: dict) -> None:
    """Publish an event tagged with user_id + control_run_id."""
    await self._publish(
      user_id,
      control_run_id,
      event,
    )

  async def publish_terminal_if_absent(
    self,
    user_id: str,
    control_run_id: str,
    event: dict,
  ) -> bool:
    """Publish and terminalize one run atomically, or report it already terminal."""

    normalized_user_id = str(user_id)
    normalized_run_id = str(control_run_id)
    key = (normalized_user_id, normalized_run_id)
    event_copy = deepcopy(dict(event))

    async with self._lock:
      if self._shutdown:
        raise RuntimeError("required event bus is closed")
      if key in self._terminated_runs:
        return False
      seq = self._next_seq_by_run.get(key, 1)
      self._next_seq_by_run[key] = seq + 1
      entry = _BusEvent(
        event=event_copy,
        timestamp=time.time(),
        seq=seq,
        control_run_id=normalized_run_id,
        terminal=True,
      )
      replay_buffer = self._replay_buffers.get(key)
      if replay_buffer is None:
        replay_buffer = deque(maxlen=self._replay_buffer_max)
        self._replay_buffers[key] = replay_buffer
      replay_buffer.append(entry)
      self._terminated_runs.add(key)
      self._schedule_cleanup_locked(key)

      for subscriber in list(self._subscribers.get(normalized_user_id, ())):
        if subscriber.matches(normalized_user_id, normalized_run_id):
          self._enqueue_for_subscriber(subscriber, entry)
      return True

  async def _publish(
    self,
    user_id: str,
    control_run_id: str,
    event: dict,
  ) -> None:
    normalized_user_id = str(user_id)
    normalized_run_id = str(control_run_id)
    key = (normalized_user_id, normalized_run_id)

    async with self._lock:
      if self._shutdown:
        return
      seq = self._next_seq_by_run.get(key, 1)
      self._next_seq_by_run[key] = seq + 1
      entry = _BusEvent(
        event=deepcopy(dict(event)),
        timestamp=time.time(),
        seq=seq,
        control_run_id=normalized_run_id,
      )

      replay_buffer = self._replay_buffers.get(key)
      if replay_buffer is None:
        replay_buffer = deque(maxlen=self._replay_buffer_max)
        self._replay_buffers[key] = replay_buffer
      replay_buffer.append(entry)

      for subscriber in list(self._subscribers.get(normalized_user_id, ())):
        if not subscriber.matches(normalized_user_id, normalized_run_id):
          continue
        self._enqueue_for_subscriber(subscriber, entry)

  async def seed_replay_buffer(
    self,
    user_id: str,
    control_run_id: str,
    events: list[dict],
    *,
    terminated: bool = False,
  ) -> int:
    """Backfill one run's replay buffer from a durable event source.

    The bus assigns per-run sequence numbers, so durable events are stored with
    synthetic seqs that match their position in the persisted stream. Future
    live publishes then continue at ``len(events) + 1``.
    """
    normalized_user_id = str(user_id)
    normalized_run_id = str(control_run_id)
    key = (normalized_user_id, normalized_run_id)
    seed_events = [deepcopy(event) for event in events if isinstance(event, dict)]

    async with self._lock:
      if self._shutdown:
        return 0
      if key in self._replay_buffers or key in self._next_seq_by_run:
        if terminated:
          replay_buffer = self._replay_buffers.get(key)
          if replay_buffer:
            last = replay_buffer.pop()
            replay_buffer.append(
              _BusEvent(
                event=last.event,
                timestamp=last.timestamp,
                seq=last.seq,
                control_run_id=last.control_run_id,
                terminal=True,
              )
            )
          self._terminated_runs.add(key)
          self._schedule_cleanup_locked(key)
        return 0

      replay_buffer = deque(maxlen=self._replay_buffer_max)
      total_events = len(seed_events)
      tail = seed_events[-self._replay_buffer_max :]
      first_seq = total_events - len(tail) + 1
      now = time.time()
      for index, event in enumerate(tail, start=first_seq):
        replay_buffer.append(
          _BusEvent(
            event=event,
            timestamp=now,
            seq=index,
            control_run_id=normalized_run_id,
            terminal=terminated and index == total_events,
          )
        )

      self._replay_buffers[key] = replay_buffer
      self._next_seq_by_run[key] = total_events + 1
      if terminated:
        self._terminated_runs.add(key)
        self._schedule_cleanup_locked(key)
      return len(replay_buffer)

  def subscribe(
    self,
    user_id: str,
    *,
    control_run_id: str | None = None,
  ) -> AsyncIterator[dict]:
    """Subscribe to this user's events, optionally scoped to one run."""

    async def _iterator() -> AsyncIterator[dict]:
      subscriber = _Subscriber(
        user_id=str(user_id),
        control_run_id=str(control_run_id) if control_run_id is not None else None,
        max_queue_size=self._subscriber_queue_max,
        queue=deque(),
        updated=asyncio.Event(),
      )
      replay_events: List[_BusEvent] = []
      async with self._lock:
        if self._shutdown:
          return
        self._register_subscriber_locked(subscriber)
        if subscriber.control_run_id is None:
          self._cancel_user_cleanup_locked(subscriber.user_id)
        else:
          self._cancel_cleanup_locked((subscriber.user_id, subscriber.control_run_id))
          replay_events = [
            _copy_bus_event(entry)
            for entry in self._replay_buffers.get(
              (subscriber.user_id, subscriber.control_run_id),
              (),
            )
          ]

      try:
        for entry in replay_events:
          yield dict(entry.event)

        while True:
          try:
            event = await self._next_event(subscriber)
          except _SubscriberClosed:
            return
          yield event
      finally:
        await asyncio.shield(self._unsubscribe(subscriber))

    return _iterator()

  async def subscribe_entries(
    self,
    user_id: str,
    *,
    control_run_id: str | None = None,
    after_seq: int = 0,
  ) -> EventSubscription:
    """Atomically register and return an explicitly closeable subscription."""

    subscriber = _Subscriber(
      user_id=str(user_id),
      control_run_id=str(control_run_id) if control_run_id is not None else None,
      max_queue_size=self._subscriber_queue_max,
      queue=deque(),
      updated=asyncio.Event(),
      stop_at_terminal=control_run_id is not None,
    )
    normalized_after_seq = max(int(after_seq), 0)
    if subscriber.control_run_id is not None:
      subscriber.after_seq_floor = normalized_after_seq
    replay_events: List[_BusEvent] = []
    truncation_event: _BusEvent | None = None
    registered = False
    async with self._lock:
      if not self._shutdown:
        self._register_subscriber_locked(subscriber)
        registered = True
        if subscriber.control_run_id is None:
          self._cancel_user_cleanup_locked(subscriber.user_id)
        else:
          key = (subscriber.user_id, subscriber.control_run_id)
          self._cancel_cleanup_locked(key)
          subscriber.stream_terminated = key in self._terminated_runs
          replay_buffer = [
            _copy_bus_event(entry)
            for entry in self._replay_buffers.get(
              (subscriber.user_id, subscriber.control_run_id),
              (),
            )
          ]
          if replay_buffer:
            head_seq = replay_buffer[0].seq
            if head_seq is not None and normalized_after_seq < head_seq - 1:
              truncation_event = _BusEvent(
                event={
                  "type": "replay_truncated",
                  "run_id": subscriber.control_run_id,
                  "control_run_id": subscriber.control_run_id,
                  "dropped_before_seq": head_seq,
                },
                timestamp=time.time(),
                seq=None,
                control_run_id=subscriber.control_run_id,
              )
            replay_events = [
              entry
              for entry in replay_buffer
              if entry.seq is not None and entry.seq > normalized_after_seq
            ]
    return EventSubscription(
      bus=self,
      subscriber=subscriber,
      replay_events=replay_events,
      truncation_event=truncation_event,
      registered=registered,
      stop_at_terminal=subscriber.stop_at_terminal,
    )

  async def cleanup_run(self, user_id: str, control_run_id: str) -> None:
    """Schedule deferred per-run replay-buffer cleanup."""
    key = (str(user_id), str(control_run_id))
    async with self._lock:
      if self._shutdown:
        return
      self._terminated_runs.add(key)
      self._schedule_cleanup_locked(key)

  async def shutdown(self) -> None:
    """Close subscribers and cancel deferred cleanup tasks."""
    async with self._lock:
      if self._shutdown:
        return
      self._shutdown = True
      cleanup_tasks = list(self._cleanup_tasks.values())
      self._cleanup_tasks.clear()
      for task in cleanup_tasks:
        task.cancel()
      for subscribers in self._subscribers.values():
        for subscriber in subscribers:
          subscriber.closed = True
          subscriber.updated.set()
      self._subscribers.clear()
      self._terminated_runs.clear()
      self._replay_buffers.clear()
      self._next_seq_by_run.clear()

    if cleanup_tasks:
      await asyncio.gather(*cleanup_tasks, return_exceptions=True)

  def _register_subscriber_locked(self, subscriber: _Subscriber) -> None:
    subscribers = self._subscribers.setdefault(subscriber.user_id, set())
    subscribers.add(subscriber)

  def _unregister_subscriber_locked(self, subscriber: _Subscriber) -> None:
    subscriber.closed = True
    subscriber.updated.set()
    subscribers = self._subscribers.get(subscriber.user_id)
    if subscribers is None:
      return
    subscribers.discard(subscriber)
    if not subscribers:
      self._subscribers.pop(subscriber.user_id, None)

  async def _unsubscribe(self, subscriber: _Subscriber) -> None:
    async with self._lock:
      self._unregister_subscriber_locked(subscriber)
      if self._shutdown:
        return

      for key in list(self._terminated_runs):
        if key[0] != subscriber.user_id:
          continue
        if subscriber.control_run_id is not None and key[1] != subscriber.control_run_id:
          continue
        self._schedule_cleanup_locked(key)

  async def _next_event(self, subscriber: _Subscriber) -> dict:
    while True:
      async with self._lock:
        if subscriber.dropped_count:
          sentinel = {
            "type": "events_dropped",
            "count": subscriber.dropped_count,
            "oldest_ts": subscriber.oldest_dropped_ts,
          }
          subscriber.dropped_count = 0
          subscriber.oldest_dropped_ts = None
          subscriber.dropped_through_seq = None
          return sentinel
        if subscriber.queue:
          return deepcopy(subscriber.queue.popleft().event)
        if subscriber.closed or self._shutdown:
          raise _SubscriberClosed()
        subscriber.updated.clear()
      await subscriber.updated.wait()

  async def _next_entry(self, subscriber: _Subscriber) -> _BusEvent:
    while True:
      async with self._lock:
        if subscriber.dropped_count:
          event = {
            "type": "events_dropped",
            "count": subscriber.dropped_count,
            "oldest_ts": subscriber.oldest_dropped_ts,
            "dropped_through_seq": subscriber.dropped_through_seq,
          }
          if subscriber.control_run_id is not None:
            event["run_id"] = subscriber.control_run_id
            event["control_run_id"] = subscriber.control_run_id
          subscriber.dropped_count = 0
          subscriber.oldest_dropped_ts = None
          subscriber.dropped_through_seq = None
          return _BusEvent(
            event=event,
            timestamp=time.time(),
            seq=None,
            control_run_id=subscriber.control_run_id,
          )
        if subscriber.queue:
          return _copy_bus_event(subscriber.queue.popleft())
        if subscriber.closed or self._shutdown:
          raise _SubscriberClosed()
        if subscriber.stop_at_terminal and subscriber.stream_terminated:
          raise _SubscriberTerminated()
        subscriber.updated.clear()
      await subscriber.updated.wait()

  def _enqueue_for_subscriber(self, subscriber: _Subscriber, entry: _BusEvent) -> None:
    terminal_transition = subscriber.stop_at_terminal and entry.terminal
    if terminal_transition:
      subscriber.stream_terminated = True
    if (
      entry.seq is not None
      and entry.seq <= subscriber.after_seq_floor
    ):
      if terminal_transition:
        subscriber.updated.set()
      return
    if len(subscriber.queue) >= subscriber.max_queue_size:
      dropped = subscriber.queue.popleft()
      subscriber.dropped_count += 1
      if subscriber.oldest_dropped_ts is None:
        subscriber.oldest_dropped_ts = dropped.timestamp
      if dropped.seq is not None:
        subscriber.dropped_through_seq = dropped.seq
    subscriber.queue.append(_copy_bus_event(entry))
    subscriber.updated.set()

  def _schedule_cleanup_locked(self, key: Tuple[str, str]) -> None:
    if self._has_matching_subscriber_locked(key):
      return
    task = self._cleanup_tasks.get(key)
    if task is not None and not task.done():
      return
    self._cleanup_tasks[key] = asyncio.create_task(self._deferred_cleanup(key))

  async def _deferred_cleanup(self, key: Tuple[str, str]) -> None:
    try:
      await asyncio.sleep(self._cleanup_delay_seconds)
      async with self._lock:
        if self._cleanup_tasks.get(key) is not asyncio.current_task():
          return
        if self._has_matching_subscriber_locked(key):
          return
        self._replay_buffers.pop(key, None)
        self._next_seq_by_run.pop(key, None)
        self._terminated_runs.discard(key)
        self._cleanup_tasks.pop(key, None)
    except asyncio.CancelledError:
      raise

  def _has_matching_subscriber_locked(self, key: Tuple[str, str]) -> bool:
    user_id, control_run_id = key
    return any(
      subscriber.matches(user_id, control_run_id)
      for subscriber in self._subscribers.get(user_id, ())
    )

  def _has_run_state_locked(self, key: Tuple[str, str]) -> bool:
    return (
      key in self._replay_buffers
      or key in self._next_seq_by_run
      or key in self._terminated_runs
      or key in self._cleanup_tasks
    )

  def _cancel_user_cleanup_locked(self, user_id: str) -> None:
    for key in list(self._cleanup_tasks):
      if key[0] == user_id:
        self._cancel_cleanup_locked(key)

  def _cancel_cleanup_locked(self, key: Tuple[str, str]) -> None:
    task = self._cleanup_tasks.pop(key, None)
    if task is not None:
      task.cancel()
