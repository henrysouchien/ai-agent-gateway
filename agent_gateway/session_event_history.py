from __future__ import annotations

from collections import deque
from typing import Deque


class SessionEventHistory:
  """Append-only, bounded event history retained for a gateway session."""

  def __init__(self, *, max_events: int = 5000) -> None:
    if max_events < 1:
      raise ValueError("max_events must be positive")
    self._events: Deque[dict] = deque(maxlen=max_events)

  def append(self, event: dict) -> None:
    """Append an event copy. Terminal stream events do not close this history."""
    self._events.append(dict(event))

  def snapshot(self, *, tail: int | None = None) -> list[dict]:
    """Return retained events, or only the last ``tail`` events when provided."""
    events = list(self._events)
    if tail is None:
      return [dict(event) for event in events]
    if tail <= 0:
      return []
    return [dict(event) for event in events[-tail:]]

  def __len__(self) -> int:
    return len(self._events)


__all__ = ["SessionEventHistory"]
