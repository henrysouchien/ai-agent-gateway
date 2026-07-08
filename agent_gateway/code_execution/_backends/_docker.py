from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from ...agent_telemetry import AGENT_TELEMETRY_ENV_TO_HEADER
from .._config import CodeExecutionConfig
from .._helpers import (
  _STREAM_READER_LIMIT,
  _collect_generated_images,
  _read_truncated,
  _remove_path,
  _remove_task_artifacts,
  _snapshot_image_mtimes,
  _tail_file,
  _task_stderr_path,
  _task_stdout_path,
  _write_code_execute_script,
)
from .._provenance import AGENT_CODE_EXECUTE_WORK_DIR_ENV
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


class DockerBackend(ExecutionBackend):
  """Docker-based code execution backend.

  This backend runs Python in an isolated container with networking disabled and
  is considered sandboxed by the approval system.
  """

  def __init__(self, image: Optional[str] = None, config: CodeExecutionConfig | None = None) -> None:
    self._config = config or CodeExecutionConfig()
    self._image = image or os.getenv("CODE_EXECUTE_DOCKER_IMAGE", "ai-excel-addin-code-exec:latest")

  @property
  def name(self) -> str:
    return "docker"

  @property
  def sandboxed(self) -> bool:
    return True

  def available(self) -> bool:
    docker_bin = shutil.which("docker")
    if docker_bin is None:
      return False
    try:
      subprocess.run(
        [docker_bin, "image", "inspect", self._image],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=3,
      )
    except Exception:
      return False
    return True

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
    cache_root = work_dir_path / ".code_execute_cache"
    (cache_root / "mplconfig").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg").mkdir(parents=True, exist_ok=True)
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
    command = [
      "docker",
      "run",
      "-d",
      "--network",
      "none",
      "--memory",
      "512m",
      "--cpus",
      "1.0",
      "--pids-limit",
      "128",
      "-v",
      f"{work_dir_path}:/workspace:rw",
      "-w",
      "/workspace",
      "-e",
      "PYTHONUNBUFFERED=1",
      "-e",
      "MPLCONFIGDIR=/workspace/.code_execute_cache/mplconfig",
      "-e",
      "XDG_CACHE_HOME=/workspace/.code_execute_cache/xdg",
    ]
    command.extend(self._env_args(env))
    command.extend(
      [
        self._image,
        "python3",
        "-u",
        f"/workspace/{script_path.name}",
      ]
    )
    run_proc = await asyncio.create_subprocess_exec(
      *command,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
      limit=_STREAM_READER_LIMIT,
    )
    stdout_bytes, stderr_bytes = await run_proc.communicate()
    if run_proc.returncode != 0:
      stderr = stderr_bytes.decode("utf-8", errors="replace").strip() or "docker run failed"
      raise RuntimeError(stderr)
    container_id = stdout_bytes.decode("utf-8", errors="replace").strip()
    if not container_id:
      raise RuntimeError("docker run did not return a container id")

    data: Dict[str, Any] = {
      "task_id": task_id,
      "container_id": container_id,
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
      logs_proc = await asyncio.create_subprocess_exec(
        "docker",
        "logs",
        "-f",
        container_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=_STREAM_READER_LIMIT,
      )
      data["logs_process"] = logs_proc
      stdout_task = asyncio.create_task(_read_stream_to_file(logs_proc.stdout, stdout_file, "stdout", on_output))
      stderr_task = asyncio.create_task(_read_stream_to_file(logs_proc.stderr, stderr_file, "stderr", on_output))
      wait_proc = await asyncio.create_subprocess_exec(
        "docker",
        "wait",
        container_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=_STREAM_READER_LIMIT,
      )
      data["wait_process"] = wait_proc
      try:
        try:
          wait_stdout, _wait_stderr = await asyncio.wait_for(wait_proc.communicate(), timeout=timeout_ms / 1000)
          wait_text = wait_stdout.decode("utf-8", errors="replace").strip()
          if wait_text:
            try:
              data["return_code"] = int(wait_text.splitlines()[-1])
            except ValueError:
              data["return_code"] = None
        except asyncio.TimeoutError:
          data["timed_out"] = True
          data["return_code"] = 124
          await self.cancel(handle)
      finally:
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        await asyncio.gather(logs_proc.wait(), return_exceptions=True)
        data["completed"] = True

    data["reader_task"] = asyncio.create_task(_monitor())
    return handle

  async def poll(self, handle: ExecutionHandle) -> Dict[str, Any]:
    data = handle._backend_data
    if data.get("completed"):
      return {"status": "completed"}
    await self._inspect_container(data.get("container_id", ""))
    return {"status": "running", "stdout_tail": _tail_file(data["stdout_file"], n=20)}

  async def cancel(self, handle: ExecutionHandle) -> None:
    container_id = str(handle._backend_data.get("container_id") or "")
    if not container_id:
      return
    await self._docker_exec("stop", container_id, "--time", "5", check=False)
    await self._docker_exec("rm", "-f", container_id, check=False)

  async def collect(self, handle: ExecutionHandle) -> Dict[str, Any]:
    data = handle._backend_data
    await asyncio.gather(data["reader_task"], return_exceptions=False)
    stdout, stdout_truncated = _read_truncated(data["stdout_file"], self._config.max_output_bytes)
    stderr, stderr_truncated = _read_truncated(data["stderr_file"], self._config.max_output_bytes)
    if data.get("timed_out") and not stderr:
      stderr = "Process killed after timeout"
    await self._docker_exec("rm", data["container_id"], check=False)
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

  async def _docker_exec(self, *args: str, check: bool = True) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
      "docker",
      *args,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
      limit=_STREAM_READER_LIMIT,
    )
    stdout_bytes, stderr_bytes = await proc.communicate()
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    if check and proc.returncode != 0:
      raise RuntimeError(stderr.strip() or stdout.strip() or f"docker {' '.join(args)} failed")
    return proc.returncode, stdout, stderr

  async def _inspect_container(self, container_id: str) -> None:
    if not container_id:
      return
    await self._docker_exec("inspect", container_id, check=False)

  def _env_args(self, env: Optional[Dict[str, str]]) -> list[str]:
    if not env:
      return []
    args: list[str] = []
    forwarded_keys = (
      "PYTHONPATH",
      AGENT_CODE_EXECUTE_WORK_DIR_ENV,
      *(env_name for env_name, _header_name in AGENT_TELEMETRY_ENV_TO_HEADER),
    )
    for key in forwarded_keys:
      value = env.get(key)
      if value:
        args.extend(["-e", f"{key}={value}"])
    return args

  def _cleanup_foreground_files(self, handle: ExecutionHandle) -> None:
    data = handle._backend_data
    _remove_path(data.get("script_path"))
    _remove_path(data.get("stdout_file"))
    _remove_path(data.get("stderr_file"))
