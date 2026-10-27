"""Dependency-free NDJSON codec shared by host and shipped executor sources.

Wire bytes (including the newline) count once in each direction against one
executing cell's aggregate budget. Only an aggregate breach poisons that cell,
not its interpreter. A single frame over its caller-chosen `limit` is a
ProtocolError for that frame alone: it neither charges nor poisons the budget.
"""
import json
import threading
from typing import Any

MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_CELL_BYTES = 64 * 1024 * 1024

_MIB = 1024 * 1024


class ProtocolError(ValueError):
  code = "protocol_error"


def _over_limit(limit: int) -> ProtocolError:
  bound = f"{limit // _MIB} MiB" if limit % _MIB == 0 else f"{limit} byte"
  return ProtocolError(f"frame exceeds the {bound} bound")


class CellBudget:
  """One cell's aggregate wire-byte budget."""

  def __init__(self, aggregate_limit: int = MAX_CELL_BYTES) -> None:
    self.aggregate_limit = aggregate_limit
    self.used_bytes = 0
    self.error: str | None = None
    self._lock = threading.Lock()

  def fail(self, message: str) -> ProtocolError:
    with self._lock:
      if self.error is None:
        self.error = str(message)
    return ProtocolError(self.error)

  def charge(self, size: int) -> None:
    with self._lock:
      if self.error is not None:
        raise ProtocolError(self.error)
      if self.used_bytes + size > self.aggregate_limit:
        self.error = "cell frames exceed the 64 MiB aggregate bound"
        raise ProtocolError(self.error)
      self.used_bytes += size


def encode_frame(frame: Any, budget: CellBudget | None = None, *, limit: int = MAX_CELL_BYTES) -> bytes:
  """Serialize one frame, bounded by `limit`, without allocating an unbounded JSON body."""
  data = bytearray()
  try:
    for part in json.JSONEncoder(separators=(",", ":"), default=str, ensure_ascii=False).iterencode(frame):
      # UTF-8 cannot use fewer bytes than characters. Check before the encode
      # allocation, then check again for multibyte text.
      if len(part) + len(data) + 1 > limit:
        raise _over_limit(limit)
      encoded = part.encode("utf-8")
      if len(encoded) + len(data) + 1 > limit:
        raise _over_limit(limit)
      data.extend(encoded)
  except ProtocolError:
    raise
  except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
    raise ProtocolError("frame cannot be encoded as JSON") from exc
  data.append(10)
  if budget is not None:
    budget.charge(len(data))
  return bytes(data)


def decode_frame(line: bytes, budget: CellBudget | None = None, *, limit: int = MAX_CELL_BYTES) -> dict[str, Any]:
  """Check byte bounds before JSON allocation; accept only an object frame."""
  if len(line) > limit:
    raise _over_limit(limit)
  if budget is not None:
    budget.charge(len(line))
  try:
    frame = json.loads(line)
  except (ValueError, UnicodeError, RecursionError) as exc:
    raise ProtocolError("frame is not valid JSON") from exc
  if not isinstance(frame, dict):
    raise ProtocolError("frame is not an object")
  return frame
