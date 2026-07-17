from __future__ import annotations

from collections import Counter
from threading import Lock


_COUNTERS: Counter[str] = Counter()
_LOCK = Lock()


def record(name: str) -> None:
  with _LOCK:
    _COUNTERS[str(name)] += 1


def snapshot() -> dict[str, int]:
  with _LOCK:
    return dict(_COUNTERS)


__all__ = ["record", "snapshot"]
