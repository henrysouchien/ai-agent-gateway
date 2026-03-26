from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, List, Optional


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

  @property
  def closed(self) -> bool:
    return self._closed
