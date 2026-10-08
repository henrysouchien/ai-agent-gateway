from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol

from .._frame_codec import CellBudget


OnOutputChunk = Callable[[str, str], None]


class CellBridge(Protocol):
  """One cell's frame channel to the host (`_frame_reader.CellChannel`).

  Subprocess hands `child_fds()` to the child with `child_env()` and runs
  `serve()` on its inherited pipes. Docker initializes its container runner
  over stdin and routes the runner's bridge frames through `handle_frame()`.
  The transport decodes data/request frames against `budget` once; the reader
  charges bridge replies. Both close the registration when the cell ends.
  """

  budget: CellBudget

  def activate(self) -> None: ...

  def deactivate(self) -> None: ...

  def child_env(self) -> Dict[str, str]: ...

  def child_fds(self) -> tuple[int, ...]: ...

  def close_child_ends(self) -> None: ...

  async def serve(self) -> None: ...

  async def answer_frame(self, line: bytes, *, accounted: bool = False) -> Dict[str, Any]: ...

  async def handle_frame(self, line: bytes, *, accounted: bool = False) -> bytes: ...

  def close(self) -> None: ...


@dataclass
class ExecutionHandle:
  backend_name: str
  handle_id: str
  work_dir: str
  _backend_data: Any = None


class ExecutionBackend:
  @property
  def name(self) -> str:
    raise NotImplementedError

  @property
  def sandboxed(self) -> bool:
    raise NotImplementedError

  async def execute(
    self,
    code: str,
    work_dir: str,
    *,
    task_id: str = "",
    timeout_ms: int = 30_000,
    env: Optional[Dict[str, str]] = None,
    on_output: Optional[OnOutputChunk] = None,
  ) -> Dict[str, Any]:
    raise NotImplementedError

  async def start(
    self,
    code: str,
    work_dir: str,
    *,
    task_id: str = "",
    timeout_ms: int = 30_000,
    env: Optional[Dict[str, str]] = None,
    on_output: Optional[OnOutputChunk] = None,
  ) -> ExecutionHandle:
    raise NotImplementedError

  # Backends with frame transport override these entry points. A backend
  # without one closes the registration and runs its ordinary code cell.

  async def execute_bridged(
    self,
    code: str,
    work_dir: str,
    *,
    bridge: CellBridge,
    task_id: str = "",
    timeout_ms: int = 30_000,
    env: Optional[Dict[str, str]] = None,
    on_output: Optional[OnOutputChunk] = None,
  ) -> Dict[str, Any]:
    bridge.close()
    return await self.execute(code, work_dir, task_id=task_id, timeout_ms=timeout_ms, env=env, on_output=on_output)

  async def start_bridged(
    self,
    code: str,
    work_dir: str,
    *,
    bridge: CellBridge,
    task_id: str = "",
    timeout_ms: int = 30_000,
    env: Optional[Dict[str, str]] = None,
    on_output: Optional[OnOutputChunk] = None,
  ) -> ExecutionHandle:
    bridge.close()
    return await self.start(code, work_dir, task_id=task_id, timeout_ms=timeout_ms, env=env, on_output=on_output)

  async def poll(self, handle: ExecutionHandle) -> Dict[str, Any]:
    raise NotImplementedError

  async def cancel(self, handle: ExecutionHandle) -> None:
    raise NotImplementedError

  async def collect(self, handle: ExecutionHandle) -> Dict[str, Any]:
    raise NotImplementedError

  async def cleanup(self, work_dir: str, *, task_id: str | None = None) -> None:
    raise NotImplementedError

  def unavailable_reason(self) -> Optional[str]:
    """Why this backend cannot run code right now, or None when it can.

    The one availability probe: `available()` derives from it, so a backend
    that `host=auto` skips always has a cause to log and to report.
    """
    return None

  def available(self) -> bool:
    return self.unavailable_reason() is None
