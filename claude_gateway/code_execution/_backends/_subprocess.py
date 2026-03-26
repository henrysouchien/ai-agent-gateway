from __future__ import annotations

import asyncio
import os
import shutil
import signal
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from .._config import CodeExecutionConfig
from .._helpers import (
  _STREAM_READER_LIMIT,
  _collect_generated_images,
  _prepare_code_execute_env,
  _read_truncated,
  _remove_path,
  _remove_task_artifacts,
  _snapshot_image_mtimes,
  _tail_file,
  _task_stderr_path,
  _task_stdout_path,
  _write_code_execute_script,
)
from ._base import ExecutionBackend, ExecutionHandle, OnOutputChunk


async def _read_stream_to_file(
  stream: asyncio.StreamReader | None,
  filepath: Path,
  stream_name: str,
  on_output: Optional[OnOutputChunk],
) -> None:
  if stream is None:
    return
  with filepath.open("w", encoding="utf-8", buffering=1) as handle:
    while True:
      line = await stream.readline()
      if not line:
        break
      text = line.decode("utf-8", errors="replace")
      handle.write(text)
      if on_output is not None:
        on_output(stream_name, text)


class SubprocessBackend(ExecutionBackend):
  """Local subprocess code execution backend.

  This backend is convenient for development but is intentionally marked
  unsandboxed so callers can require explicit approval before use.
  """

  def __init__(self, config: CodeExecutionConfig | None = None) -> None:
    self._config = config or CodeExecutionConfig()

  @property
  def name(self) -> str:
    return "subprocess"

  @property
  def sandboxed(self) -> bool:
    return False

  def available(self) -> bool:
    return shutil.which("python3") is not None

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
    handle = await self.start(
      code,
      work_dir,
      task_id=task_id,
      timeout_ms=timeout_ms,
      env=env,
      on_output=on_output,
    )
    data = handle._backend_data
    try:
      try:
        await data["reader_task"]
      except asyncio.CancelledError:
        await self.cancel(handle)
        await asyncio.shield(asyncio.gather(data["reader_task"], return_exceptions=True))
        raise
      return await self.collect(handle)
    finally:
      if task_id:
        await self.cleanup(handle.work_dir, task_id=task_id)
      else:
        self._cleanup_foreground_files(handle)

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
    work_dir_path = Path(work_dir)
    work_dir_path.mkdir(parents=True, exist_ok=True)
    process_env = dict(env) if env is not None else _prepare_code_execute_env(self._config)
    handle_id = uuid.uuid4().hex
    script_path = _write_code_execute_script(work_dir_path, code, self._config, task_id=task_id)
    stdout_file = (
      _task_stdout_path(work_dir_path, task_id)
      if task_id
      else work_dir_path / f"_code_execute_{handle_id}_stdout.log"
    )
    stderr_file = (
      _task_stderr_path(work_dir_path, task_id)
      if task_id
      else work_dir_path / f"_code_execute_{handle_id}_stderr.log"
    )
    before_mtimes = None if task_id else _snapshot_image_mtimes(work_dir_path)
    process = await asyncio.create_subprocess_exec(
      "python3",
      "-u",
      str(script_path),
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
      cwd=str(work_dir_path),
      env=process_env,
      preexec_fn=os.setsid,
      limit=_STREAM_READER_LIMIT,
    )

    data: Dict[str, Any] = {
      "task_id": task_id,
      "process": process,
      "script_path": script_path,
      "stdout_file": stdout_file,
      "stderr_file": stderr_file,
      "started_at": time.time(),
      "started_ns": time.time_ns(),
      "before_mtimes": before_mtimes,
      "completed": False,
      "timed_out": False,
      "return_code": None,
    }
    handle = ExecutionHandle(
      backend_name=self.name,
      handle_id=handle_id,
      work_dir=str(work_dir_path),
      _backend_data=data,
    )

    async def _monitor() -> None:
      stdout_task = asyncio.create_task(_read_stream_to_file(process.stdout, stdout_file, "stdout", on_output))
      stderr_task = asyncio.create_task(_read_stream_to_file(process.stderr, stderr_file, "stderr", on_output))
      try:
        try:
          await asyncio.wait_for(process.wait(), timeout=timeout_ms / 1000)
        except asyncio.TimeoutError:
          data["timed_out"] = True
          await self.cancel(handle)
        await asyncio.gather(stdout_task, stderr_task)
      finally:
        data["return_code"] = process.returncode
        data["completed"] = True

    data["reader_task"] = asyncio.create_task(_monitor())
    return handle

  async def poll(self, handle: ExecutionHandle) -> Dict[str, Any]:
    data = handle._backend_data
    if data.get("completed"):
      return {"status": "completed"}
    return {"status": "running", "stdout_tail": _tail_file(data["stdout_file"], n=20)}

  async def cancel(self, handle: ExecutionHandle) -> None:
    data = handle._backend_data
    process = data["process"]
    if process.returncode is not None:
      return
    try:
      os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
      return
    except Exception:
      process.terminate()
    try:
      await asyncio.wait_for(process.wait(), timeout=2)
    except asyncio.TimeoutError:
      try:
        os.killpg(process.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
      except Exception:
        process.kill()
      await process.wait()

  async def collect(self, handle: ExecutionHandle) -> Dict[str, Any]:
    data = handle._backend_data
    await asyncio.gather(data["reader_task"], return_exceptions=False)
    stdout, stdout_truncated = _read_truncated(data["stdout_file"], self._config.max_output_bytes)
    stderr, stderr_truncated = _read_truncated(data["stderr_file"], self._config.max_output_bytes)
    if data.get("timed_out") and not stderr:
      stderr = "Process killed after timeout"
    return {
      "stdout": stdout,
      "stderr": stderr,
      "return_code": data.get("return_code"),
      "images": _collect_generated_images(
        Path(handle.work_dir),
        data["started_ns"],
        self._config,
        before_mtimes=data.get("before_mtimes"),
        task_id=data.get("task_id") or None,
      ),
      "timed_out": bool(data.get("timed_out")),
      "duration_ms": int((time.time() - data["started_at"]) * 1000),
      "truncated": bool(stdout_truncated or stderr_truncated),
    }

  async def cleanup(self, work_dir: str, *, task_id: str | None = None) -> None:
    work_dir_path = Path(work_dir)
    if task_id is None:
      shutil.rmtree(work_dir_path, ignore_errors=True)
      return
    _remove_task_artifacts(work_dir_path, task_id)

  def _cleanup_foreground_files(self, handle: ExecutionHandle) -> None:
    data = handle._backend_data
    _remove_path(data.get("script_path"))
    _remove_path(data.get("stdout_file"))
    _remove_path(data.get("stderr_file"))
