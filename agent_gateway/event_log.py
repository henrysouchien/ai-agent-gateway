from __future__ import annotations

import asyncio
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Deque, Dict, List, Mapping, Optional, Set, Tuple


TERMINAL_EVENT_TYPES = {"stream_complete", "error"}
OnEvent = Callable[[Dict[str, Any], str], None]


@dataclass
class LogEntry:
  """Single event log entry with sequence number and timestamp."""

  seq: int
  timestamp: float
  event: Dict[str, Any]


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
  ) -> None:
    self._entries: List[LogEntry] = []
    self._closed = False
    self._next_seq = 1
    self._version = 0
    self._updated = asyncio.Event()
    self._on_event = on_event
    self._session_id = session_id
    self._raw_tool_inputs: Dict[str, Dict[str, Any]] = {}

  def append(self, event: Dict[str, Any]) -> Optional[LogEntry]:
    if self._closed:
      return None

    entry = LogEntry(seq=self._next_seq, timestamp=time.time(), event=dict(event))
    self._next_seq += 1
    self._entries.append(entry)

    if self._on_event is not None:
      try:
        self._on_event(event, self._session_id)
      except Exception:
        pass

    if entry.event.get("type") in TERMINAL_EVENT_TYPES:
      self._closed = True

    self._version += 1
    self._updated.set()
    return entry

  def close(self, error: Optional[str] = None) -> None:
    if self._closed:
      return

    has_terminal = any(entry.event.get("type") in TERMINAL_EVENT_TYPES for entry in self._entries)
    if has_terminal:
      self._closed = True
      self._version += 1
      self._updated.set()
      return

    reason = error or "stream closed"
    self.append({"type": "error", "error": reason})

  async def iter_from(self, after_seq: int = 0) -> AsyncIterator[LogEntry]:
    index = max(after_seq, 0)

    while True:
      while index < len(self._entries):
        entry = self._entries[index]
        index += 1
        yield entry
        if entry.event.get("type") in TERMINAL_EVENT_TYPES:
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
    return list(self._entries)

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
  def next_seq(self) -> int:
    return self._next_seq


@dataclass(frozen=True)
class _BusEvent:
  event: Dict[str, Any]
  timestamp: float
  seq: int | None
  control_run_id: str | None


@dataclass(eq=False)
class _Subscriber:
  user_id: str
  control_run_id: str | None
  max_queue_size: int
  queue: Deque[_BusEvent]
  updated: asyncio.Event
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
    normalized_user_id = str(user_id)
    normalized_run_id = str(control_run_id)
    key = (normalized_user_id, normalized_run_id)

    async with self._lock:
      if self._shutdown:
        return
      seq = self._next_seq_by_run.get(key, 1)
      self._next_seq_by_run[key] = seq + 1
      entry = _BusEvent(
        event=dict(event),
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
    seed_events = [dict(event) for event in events if isinstance(event, dict)]

    async with self._lock:
      if self._shutdown:
        return 0
      if key in self._replay_buffers or key in self._next_seq_by_run:
        if terminated:
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
          replay_events = list(self._replay_buffers.get((subscriber.user_id, subscriber.control_run_id), ()))

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

  def subscribe_entries(
    self,
    user_id: str,
    *,
    control_run_id: str | None = None,
    after_seq: int = 0,
  ) -> AsyncIterator[_BusEvent]:
    """Subscribe to bus entries with per-run seq metadata for projected streams."""

    async def _iterator() -> AsyncIterator[_BusEvent]:
      subscriber = _Subscriber(
        user_id=str(user_id),
        control_run_id=str(control_run_id) if control_run_id is not None else None,
        max_queue_size=self._subscriber_queue_max,
        queue=deque(),
        updated=asyncio.Event(),
      )
      normalized_after_seq = max(int(after_seq), 0)
      replay_events: List[_BusEvent] = []
      truncation_event: _BusEvent | None = None
      async with self._lock:
        if self._shutdown:
          return
        self._register_subscriber_locked(subscriber)
        if subscriber.control_run_id is None:
          self._cancel_user_cleanup_locked(subscriber.user_id)
        else:
          self._cancel_cleanup_locked((subscriber.user_id, subscriber.control_run_id))
          replay_buffer = list(self._replay_buffers.get((subscriber.user_id, subscriber.control_run_id), ()))
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

      try:
        if truncation_event is not None:
          yield truncation_event
        for entry in replay_events:
          yield entry

        while True:
          try:
            entry = await self._next_entry(subscriber)
          except _SubscriberClosed:
            return
          yield entry
      finally:
        await asyncio.shield(self._unsubscribe(subscriber))

    return _iterator()

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

  async def _unsubscribe(self, subscriber: _Subscriber) -> None:
    async with self._lock:
      subscriber.closed = True
      subscriber.updated.set()
      subscribers = self._subscribers.get(subscriber.user_id)
      if subscribers is not None:
        subscribers.discard(subscriber)
        if not subscribers:
          self._subscribers.pop(subscriber.user_id, None)
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
          return dict(subscriber.queue.popleft().event)
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
          return subscriber.queue.popleft()
        if subscriber.closed or self._shutdown:
          raise _SubscriberClosed()
        subscriber.updated.clear()
      await subscriber.updated.wait()

  def _enqueue_for_subscriber(self, subscriber: _Subscriber, entry: _BusEvent) -> None:
    if len(subscriber.queue) >= subscriber.max_queue_size:
      dropped = subscriber.queue.popleft()
      subscriber.dropped_count += 1
      if subscriber.oldest_dropped_ts is None:
        subscriber.oldest_dropped_ts = dropped.timestamp
      if dropped.seq is not None:
        subscriber.dropped_through_seq = dropped.seq
    subscriber.queue.append(entry)
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

  def _cancel_user_cleanup_locked(self, user_id: str) -> None:
    for key in list(self._cleanup_tasks):
      if key[0] == user_id:
        self._cancel_cleanup_locked(key)

  def _cancel_cleanup_locked(self, key: Tuple[str, str]) -> None:
    task = self._cleanup_tasks.pop(key, None)
    if task is not None:
      task.cancel()
