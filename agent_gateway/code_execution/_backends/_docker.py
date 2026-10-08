"""Network-isolated Docker placement for the shared persistent kernel runner."""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Dict, Optional

from ...agent_telemetry import AGENT_TELEMETRY_ENV_TO_HEADER
from .._config import CodeExecutionConfig
from .._frame_codec import MAX_CELL_BYTES, MAX_FRAME_BYTES
from .._helpers import _STREAM_READER_LIMIT
from .._provenance import AGENT_CODE_EXECUTE_WORK_DIR_ENV
from ._kernel import KernelBackend, KernelState, _KernelProcess, _runner_source


def _project_env(env: Dict[str, str]) -> Dict[str, str]:
  """The existing Docker projection, shared by launch and per-cell refresh."""
  forwarded_keys = (
    AGENT_CODE_EXECUTE_WORK_DIR_ENV,
    *(env_name for env_name, _header_name in AGENT_TELEMETRY_ENV_TO_HEADER),
  )
  return {key: env[key] for key in forwarded_keys if env.get(key)}


async def _close_cli(process: asyncio.subprocess.Process) -> None:
  if process.stdin is not None:
    process.stdin.close()
  if process.returncode is None:
    try:
      process.kill()
    except ProcessLookupError:
      pass
  for stream in (process.stdout, process.stderr):
    transport = getattr(stream, "_transport", None)
    if transport is not None:
      transport.close()
  await process.wait()


async def _docker_exec(*args: str, check: bool = True) -> tuple[int, str, str]:
  spawn = asyncio.create_task(asyncio.create_subprocess_exec(
    "docker", *args,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    limit=_STREAM_READER_LIMIT,
  ))
  process: asyncio.subprocess.Process | None = None
  try:
    created: asyncio.subprocess.Process = await asyncio.shield(spawn)
    process = created
    stdout_bytes, stderr_bytes = await created.communicate()
  except BaseException:
    if process is None:
      process = await spawn
    await asyncio.shield(_close_cli(process))
    raise
  return_code = await created.wait()
  stdout = stdout_bytes.decode("utf-8", errors="replace")
  stderr = stderr_bytes.decode("utf-8", errors="replace")
  if check and return_code != 0:
    raise RuntimeError(stderr.strip() or stdout.strip() or f"docker {' '.join(args)} failed")
  return return_code, stdout, stderr


class _DockerProcess(_KernelProcess):
  def __init__(self, work_dir: str, image: str) -> None:
    super().__init__(work_dir)
    self.image = image
    self.container_id: str | None = None
    self.container_name = f"code-execute-{uuid.uuid4().hex}"
    self._removed = False

  async def start(self, env: Dict[str, str]) -> asyncio.subprocess.Process:
    try:
      # A teardown cannot mark an as-yet uncreated container removed, then
      # leave a late create/attach alive after the session has closed.
      async with self._close_lock:
        return await self._launch(env)
    except BaseException:
      await asyncio.shield(self.aclose())
      raise

  async def _launch(self, env: Dict[str, str]) -> asyncio.subprocess.Process:
    self.spawn_env = dict(env)
    work_dir = Path(self.work_dir)
    cache_root = work_dir / ".code_execute_cache"
    (cache_root / "mplconfig").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg").mkdir(parents=True, exist_ok=True)
    command = [
      "create", "-i", "--init", "--name", self.container_name,
      "--network", "none", "--memory", "512m", "--cpus", "1.0", "--pids-limit", "128",
      "-v", f"{work_dir.resolve()}:/workspace:rw", "-w", "/workspace",
      "-e", "PYTHONUNBUFFERED=1",
      "-e", "MPLCONFIGDIR=/workspace/.code_execute_cache/mplconfig",
      "-e", "XDG_CACHE_HOME=/workspace/.code_execute_cache/xdg",
      *(arg for key, value in env.items() for arg in ("-e", f"{key}={value}")),
      self.image, "python3", "-u", "-c", _runner_source(), str(MAX_FRAME_BYTES), str(MAX_CELL_BYTES),
    ]
    create = asyncio.create_task(_docker_exec(*command))
    try:
      _, created_stdout, _ = await asyncio.shield(create)
    except asyncio.CancelledError:
      _, created_stdout, _ = await create
      self.container_id = created_stdout.strip()
      raise
    self.container_id = created_stdout.strip()
    spawn = asyncio.create_task(asyncio.create_subprocess_exec(
      "docker", "start", "--attach", "--interactive", self.container_id,
      stdin=asyncio.subprocess.PIPE,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
      limit=MAX_FRAME_BYTES,
    ))
    try:
      process = await asyncio.shield(spawn)
    except asyncio.CancelledError:
      self.process = await spawn
      raise
    self.process = process
    self.stderr_task = asyncio.create_task(self._drain_stderr(process))
    return process

  async def interrupt(self) -> None:
    process = self.process
    if self.container_id is not None and process is not None and process.returncode is None:
      await _docker_exec("kill", "--signal", "SIGINT", self.container_id, check=False)

  async def exit_code(self) -> int | None:
    if self.container_id is None:
      return await super().exit_code()
    # Attach EOF is not a terminal-state notification from the daemon.
    # Await Docker's own exit status before disposing the container.
    _code, state, _ = await _docker_exec("wait", self.container_id)
    return int(state.strip())

  async def aclose(self) -> None:
    async with self._close_lock:
      try:
        if not self._removed:
          code, _stdout, stderr = await _docker_exec(
            "rm", "-f", self.container_id or self.container_name, check=False,
          )
          if code != 0 and "No such container" not in stderr:
            raise RuntimeError(stderr.strip() or "Docker container disposal failed")
          self._removed = True
      finally:
        if self.process is not None:
          await _close_cli(self.process)
          await self._close_pipes()


class DockerBackend(KernelBackend):
  """Sandboxed placement; interpreter reuse never widens child credentials."""

  def __init__(
    self, image: Optional[str] = None, config: CodeExecutionConfig | None = None,
    *, state: KernelState | None = None,
  ) -> None:
    super().__init__(config=config, state=state)
    self._image = image or os.getenv("CODE_EXECUTE_DOCKER_IMAGE", "ai-excel-addin-code-exec:latest")

  @property
  def name(self) -> str:
    return "docker"

  @property
  def sandboxed(self) -> bool:
    return True

  def unavailable_reason(self) -> Optional[str]:
    docker_bin = shutil.which("docker")
    if docker_bin is None:
      return "docker is not on PATH"
    try:
      subprocess.run(
        [docker_bin, "image", "inspect", self._image],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=3,
      )
    except subprocess.CalledProcessError as exc:
      detail = (exc.stderr or b"").decode("utf-8", errors="replace").strip()[:500]
      return f"docker image inspect {self._image} exited {exc.returncode}: {detail or 'no stderr'}"
    except Exception as exc:
      return f"docker image inspect {self._image} failed: {type(exc).__name__}: {exc}"
    return None

  def _make_runner(self, work_dir: str) -> _KernelProcess:
    return _DockerProcess(work_dir, self._image)

  def _script_name(self, path: Path) -> str:
    return f"/workspace/{path.name}"

  def _project_env(self, env: Dict[str, str]) -> Dict[str, str]:
    return _project_env(env)

  async def _docker_exec(self, *args: str, check: bool = True) -> tuple[int, str, str]:
    return await _docker_exec(*args, check=check)
