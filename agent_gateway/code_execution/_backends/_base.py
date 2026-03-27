from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


OnOutputChunk = Callable[[str, str], None]


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

  async def poll(self, handle: ExecutionHandle) -> Dict[str, Any]:
    raise NotImplementedError

  async def cancel(self, handle: ExecutionHandle) -> None:
    raise NotImplementedError

  async def collect(self, handle: ExecutionHandle) -> Dict[str, Any]:
    raise NotImplementedError

  async def cleanup(self, work_dir: str, *, task_id: str | None = None) -> None:
    raise NotImplementedError

  def available(self) -> bool:
    return True
