"""Reentrant cross-process lock for autonomous run mutations."""

from __future__ import annotations

import asyncio
import contextvars
import fcntl
import os
from pathlib import Path
from typing import Any


class AutonomousRunMutationLock:
  """One stable flock, reentrant for the owning asyncio task.

  Blocking flock work always runs in a worker thread.  The stable descriptor is
  reused for nested acquisition so resume -> start cannot split-lock or deadlock.
  """

  def __init__(self, log_dir: Path):
    self.path = Path(log_dir) / ".run-mutations.lock"
    self._local_lock = asyncio.Lock()
    self._owner: contextvars.ContextVar[object | None] = contextvars.ContextVar(
      f"autonomous_run_lock_owner_{id(self)}", default=None
    )
    self._token = object()
    self._depth: contextvars.ContextVar[int] = contextvars.ContextVar(
      f"autonomous_run_lock_depth_{id(self)}", default=0
    )
    self._fd: int | None = None

  def _acquire_fd(self) -> int:
    self.path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
      flags |= os.O_NOFOLLOW
    fd = os.open(self.path, flags, 0o600)
    try:
      fcntl.flock(fd, fcntl.LOCK_EX)
    except Exception:
      os.close(fd)
      raise
    return fd

  @staticmethod
  def _release_fd(fd: int) -> None:
    try:
      fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
      os.close(fd)

  async def __aenter__(self) -> "AutonomousRunMutationLock":
    if self._owner.get() is self._token:
      self._depth.set(self._depth.get() + 1)
      return self
    await self._local_lock.acquire()
    try:
      self._fd = await asyncio.to_thread(self._acquire_fd)
      self._owner.set(self._token)
      self._depth.set(1)
      return self
    except Exception:
      self._local_lock.release()
      raise

  async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
    depth = self._depth.get()
    if depth > 1:
      self._depth.set(depth - 1)
      return
    fd = self._fd
    self._fd = None
    self._depth.set(0)
    self._owner.set(None)
    try:
      if fd is not None:
        await asyncio.to_thread(self._release_fd, fd)
    finally:
      self._local_lock.release()


__all__ = ["AutonomousRunMutationLock"]
