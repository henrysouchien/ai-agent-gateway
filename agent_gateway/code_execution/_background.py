from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

from ._backends import ExecutionBackend
from ._frame_reader import CellChannel


class OutputRingBuffer:
  """Bounded in-memory tail buffer for streaming task output."""

  _MAX_LINE_BYTES = 4096

  def __init__(self, max_lines: int = 200) -> None:
    self._lines: deque[str] = deque(maxlen=max_lines)

  def append(self, stream_name: str, text: str) -> None:
    _ = stream_name
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) > self._MAX_LINE_BYTES:
      clipped = encoded[: self._MAX_LINE_BYTES].decode("utf-8", errors="ignore")
      text = f"{clipped}…\n"
    self._lines.append(text)

  def tail(self, n: int = 20) -> str:
    return "".join(list(self._lines)[-n:])


@dataclass
class BackgroundTask:
  """Track an in-flight background code execution task.

  `bridge` is the cell's frame channel. It lives on this session record, not
  on the bundle that started the cell: the product builds a bundle per
  request, and a later request's `code_execute_status` reads the cell's
  `nested_calls` and `sources` from here when it collects.
  """

  task_id: str
  handle: Any
  backend: ExecutionBackend
  stdout_buf: OutputRingBuffer
  stderr_buf: OutputRingBuffer
  tool_call_id: str | None = None
  bridge: CellChannel | None = None
  result: Optional[dict] = None
  started_at: float = 0.0
  _terminated: bool = False
  _in_progress: bool = False

  async def safe_collect(self, backend: ExecutionBackend):
    if self._terminated:
      return self.result
    if self._in_progress:
      return {"status": "running", "message": "Collection in progress, retry next turn"}
    self._in_progress = True
    try:
      self.result = await backend.collect(self.handle)
      await backend.cleanup(self.handle.work_dir, task_id=self.task_id)
      self._terminated = True
    except BaseException:
      self._in_progress = False
      raise
    await self.close_bridge()
    if self.bridge is not None and isinstance(self.result, dict):
      self.result.update(self.bridge.summary())
    return self.result

  async def safe_cancel(self, backend: ExecutionBackend) -> None:
    if self._terminated or self._in_progress:
      return
    self._in_progress = True
    try:
      await backend.cancel(self.handle)
      await backend.cleanup(self.handle.work_dir, task_id=self.task_id)
      self._terminated = True
    except BaseException:
      self._in_progress = False
      raise
    await self.close_bridge()

  async def close_bridge(self) -> None:
    """End registration and await nested dispatch/fd cleanup; idempotent."""
    if self.bridge is not None:
      await self.bridge.aclose()
