"""Dependency-free NDJSON codec shared by host and shipped executor sources.

Wire bytes (including the newline) count once in each direction against one
executing cell's budget. A bounds failure poisons that cell, not its interpreter.
The terminal protocol_error reply still fits the frame bound but is not data
charged to an already exhausted budget.
"""
import json
import threading
from typing import Any

MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_CELL_BYTES = 64 * 1024 * 1024


class ProtocolError(ValueError):
  code = "protocol_error"


class CellBudget:
  def __init__(self, frame_limit: int = MAX_FRAME_BYTES, aggregate_limit: int = MAX_CELL_BYTES) -> None:
    self.frame_limit = frame_limit
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
      if size > self.frame_limit:
        self.error = "frame exceeds the 2 MiB bound"
      elif self.used_bytes + size > self.aggregate_limit:
        self.error = "cell frames exceed the 64 MiB aggregate bound"
      if self.error is not None:
        raise ProtocolError(self.error)
      self.used_bytes += size


def encode_frame(frame: Any, budget: CellBudget | None = None) -> bytes:
  """Serialize one bounded frame without allocating an unbounded JSON body."""
  limit = budget.frame_limit if budget is not None else MAX_FRAME_BYTES
  data = bytearray()
  try:
    for part in json.JSONEncoder(separators=(",", ":"), default=str, ensure_ascii=False).iterencode(frame):
      # UTF-8 cannot use fewer bytes than characters. Check before the encode
      # allocation, then check again for multibyte text.
      if len(part) + len(data) + 1 > limit:
        message = "frame exceeds the 2 MiB bound"
        raise budget.fail(message) if budget is not None else ProtocolError(message)
      encoded = part.encode("utf-8")
      if len(encoded) + len(data) + 1 > limit:
        message = "frame exceeds the 2 MiB bound"
        raise budget.fail(message) if budget is not None else ProtocolError(message)
      data.extend(encoded)
  except ProtocolError:
    raise
  except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
    raise ProtocolError("frame cannot be encoded as JSON") from exc
  data.append(10)
  if budget is not None:
    budget.charge(len(data))
  return bytes(data)


def decode_frame(line: bytes, budget: CellBudget | None = None) -> dict[str, Any]:
  """Check byte bounds before JSON allocation; accept only an object frame."""
  if budget is not None:
    budget.charge(len(line))
  elif len(line) > MAX_FRAME_BYTES:
    raise ProtocolError("frame exceeds the 2 MiB bound")
  try:
    frame = json.loads(line)
  except (ValueError, UnicodeError, RecursionError) as exc:
    raise ProtocolError("frame is not valid JSON") from exc
  if not isinstance(frame, dict):
    raise ProtocolError("frame is not an object")
  return frame
