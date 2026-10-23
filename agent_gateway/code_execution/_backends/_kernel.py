"""Persistent local and Docker cells share a session-owned interpreter runtime.

`KernelState` belongs to GatewaySession, not a workdir registry. Request-scoped
backend wrappers reuse it while keeping their own configuration and preamble.
Cell completion releases only the bridge registration; `aclose` awaits the
interpreter and its transport at session/task teardown.
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from ...child_environment import CODE_EXECUTE_ENVIRONMENT, filter_child_environment

from .._config import CodeExecutionConfig
from .._frame_codec import CellBudget, MAX_CELL_BYTES, MAX_FRAME_BYTES, ProtocolError, decode_frame, encode_frame
from .._frame_reader import CELL_TOKEN_ENV, REQUEST_FRAME_TYPES, _error_reply
from .._helpers import (
  _collect_generated_images,
  _prepare_code_execute_env,
  _read_truncated,
  _remove_path,
  _remove_task_artifacts,
  _snapshot_image_mtimes,
  _write_code_execute_script,
)
from ._base import CellBridge, ExecutionBackend, ExecutionHandle, OnOutputChunk

log = logging.getLogger(__name__)
_INTERRUPT_GRACE_S = 2.0
_TERMINATE_GRACE_S = 2.0


def _runner_source() -> str:
  """Ship the shared codec/client and runner bytes, with no gateway imports."""
  root = Path(__file__).resolve().parent.parent
  modules = (
    ("_hank_frame_codec", root / "_frame_codec.py"),
    ("_hank_bridge_client", root / "_bridge_client.py"),
  )
  bootstrap = "import sys as _s, types as _t\n"
  for name, path in modules:
    source = path.read_text(encoding="utf-8")
    bootstrap += (
      f"_m = _t.ModuleType({name!r})\n_s.modules[{name!r}] = _m\n"
      f"exec(compile({source!r}, {path.name!r}, 'exec'), _m.__dict__)\n"
    )
  return bootstrap + (root / "_kernel_runner.py").read_text(encoding="utf-8")


class _KernelLost(Exception):
  """The runner ended before completing its cell."""


class _KernelProcess:
  """Local process transport; Docker overrides launch, signal and disposal."""

  def __init__(self, work_dir: str) -> None:
    self.work_dir = work_dir
    self.process: asyncio.subprocess.Process | None = None
    self.stderr_task: asyncio.Task[None] | None = None
    self.stderr_tail: collections.deque[str] = collections.deque(maxlen=20)
    self.spawn_env: Dict[str, str] = {}
    self._close_lock = asyncio.Lock()
    self._closed = False

  @property
  def pid(self) -> int | None:
    process = self.process
    return process.pid if process is not None and process.returncode is None else None

  async def start(self, env: Dict[str, str]) -> asyncio.subprocess.Process:
    self.spawn_env = dict(env)
    spawn = asyncio.create_task(asyncio.create_subprocess_exec(
      sys.executable, "-u", "-c", _runner_source(), str(MAX_FRAME_BYTES), str(MAX_CELL_BYTES),
      stdin=asyncio.subprocess.PIPE,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
      cwd=self.work_dir,
      env=env,
      start_new_session=True,
      limit=MAX_FRAME_BYTES,
    ))
    try:
      process = await asyncio.shield(spawn)
    except asyncio.CancelledError:
      self.process = await spawn
      await asyncio.shield(self.aclose())
      raise
    self.process = process
    self.stderr_task = asyncio.create_task(self._drain_stderr(process))
    return process

  async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
    assert process.stderr is not None
    while True:
      chunk = await process.stderr.read(8192)
      if not chunk:
        return
      text = chunk.decode("utf-8", errors="replace")
      self.stderr_tail.append(text)
      log.info("code_execute runner %s stderr: %s", self.work_dir, text.rstrip())

  async def interrupt(self) -> None:
    process = self.process
    if process is not None and process.returncode is None:
      try:
        os.kill(process.pid, signal.SIGINT)
      except ProcessLookupError:
        pass

  def _signal_group(self, signum: int) -> None:
    if self.process is not None:
      try:
        os.killpg(self.process.pid, signum)
      except ProcessLookupError:
        pass

  async def exit_code(self) -> int | None:
    if self.process is None:
      return None
    return await self.process.wait()

  async def _close_pipes(self) -> None:
    process = self.process
    if process is None:
      return
    if process.stdin is not None:
      process.stdin.close()
    # A descendant may keep an inherited pipe open after its leader exits.
    for stream in (process.stdout, process.stderr):
      transport = getattr(stream, "_transport", None)
      if transport is not None:
        transport.close()
    await process.wait()
    if process.stdin is not None:
      try:
        await process.stdin.wait_closed()
      except (BrokenPipeError, ConnectionResetError):
        pass
    if self.stderr_task is not None:
      self.stderr_task.cancel()
      await asyncio.gather(self.stderr_task, return_exceptions=True)
      self.stderr_task = None

  async def aclose(self) -> None:
    async with self._close_lock:
      process = self.process
      if process is None or self._closed:
        return
      if process.stdin is not None:
        process.stdin.close()
      self._signal_group(signal.SIGTERM)
      try:
        await asyncio.wait_for(process.wait(), timeout=_TERMINATE_GRACE_S)
      except asyncio.TimeoutError:
        self._signal_group(signal.SIGKILL)
      # Teardown owns descendants too, including children ignoring SIGTERM.
      self._signal_group(signal.SIGKILL)
      await self._close_pipes()
      self._closed = True


class KernelState:
  """One session's interpreter and cells; construction needs no running loop."""

  def __init__(self) -> None:
    self.lock = asyncio.Lock()
    self._close_lock = asyncio.Lock()
    self.runner: _KernelProcess | None = None
    self.cells: Dict[str, Dict[str, Any]] = {}
    self.closed = False

  @property
  def process(self) -> asyncio.subprocess.Process | None:
    return self.runner.process if self.runner is not None else None

  @property
  def pid(self) -> int | None:
    return self.runner.pid if self.runner is not None else None

  @property
  def container_id(self) -> str | None:
    return getattr(self.runner, "container_id", None)

  async def aclose(self) -> None:
    """Revoke running/queued cells and await every process/transport owner."""
    async with self._close_lock:
      self.closed = True
      for cell in self.cells.values():
        cell["interrupt"].set()
      try:
        runner = self.runner
        if runner is not None:
          await runner.aclose()
      finally:
        tasks = [cell["cell_task"] for cell in self.cells.values()]
        if tasks:
          await asyncio.gather(*tasks, return_exceptions=True)
        # A launch already in flight when close began installs then releases
        # its runner before its cell task settles.
        self.runner = None


class _OutputTail:
  def __init__(self, limit: int) -> None:
    self.limit = max(0, limit)
    self.data = bytearray()
    self.total = 0

  def append(self, text: str) -> None:
    chunk = text.encode("utf-8", errors="replace")
    self.total += len(chunk)
    self.data.extend(chunk)
    if len(self.data) > self.limit:
      del self.data[:len(self.data) - self.limit]

  def text(self, lines: int) -> str:
    text = self.data.decode("utf-8", errors="ignore")
    return "".join(text.splitlines(True)[-lines:])


class KernelBackend(ExecutionBackend):
  """Unsandboxed local cells; one interpreter for the supplied session state."""

  def __init__(self, config: CodeExecutionConfig | None = None, *, state: KernelState | None = None) -> None:
    self._config = config or CodeExecutionConfig()
    self._state = state if state is not None else KernelState()

  @property
  def runtime(self) -> KernelState:
    return self._state

  @property
  def name(self) -> str:
    return "kernel"

  @property
  def sandboxed(self) -> bool:
    return False

  def unavailable_reason(self) -> Optional[str]:
    if not sys.executable or not os.path.isfile(sys.executable):
      return "no Python interpreter (sys.executable is not a file)"
    return None

  def _make_runner(self, work_dir: str) -> _KernelProcess:
    return _KernelProcess(work_dir)

  def _script_name(self, path: Path) -> str:
    return str(path)

  def _project_env(self, env: Dict[str, str]) -> Dict[str, str]:
    return filter_child_environment(env, purpose=CODE_EXECUTE_ENVIRONMENT)

  async def execute(
    self, code: str, work_dir: str, *, task_id: str = "", timeout_ms: int = 30_000,
    env: Optional[Dict[str, str]] = None, on_output: Optional[OnOutputChunk] = None,
  ) -> Dict[str, Any]:
    handle = await self.start(code, work_dir, task_id=task_id, timeout_ms=timeout_ms, env=env, on_output=on_output)
    return await self.collect(handle)

  async def execute_bridged(
    self, code: str, work_dir: str, *, bridge: CellBridge, task_id: str = "", timeout_ms: int = 30_000,
    env: Optional[Dict[str, str]] = None, on_output: Optional[OnOutputChunk] = None,
  ) -> Dict[str, Any]:
    handle = await self.start_bridged(
      code, work_dir, bridge=bridge, task_id=task_id, timeout_ms=timeout_ms, env=env, on_output=on_output,
    )
    return await self.collect(handle)

  async def start(
    self, code: str, work_dir: str, *, task_id: str = "", timeout_ms: int = 30_000,
    env: Optional[Dict[str, str]] = None, on_output: Optional[OnOutputChunk] = None,
  ) -> ExecutionHandle:
    return self._start(code, work_dir, task_id=task_id, timeout_ms=timeout_ms, env=env, on_output=on_output)

  async def start_bridged(
    self, code: str, work_dir: str, *, bridge: CellBridge, task_id: str = "", timeout_ms: int = 30_000,
    env: Optional[Dict[str, str]] = None, on_output: Optional[OnOutputChunk] = None,
  ) -> ExecutionHandle:
    return self._start(
      code, work_dir, bridge=bridge, task_id=task_id, timeout_ms=timeout_ms, env=env, on_output=on_output,
    )

  def _start(
    self, code: str, work_dir: str, *, task_id: str, timeout_ms: int,
    env: Optional[Dict[str, str]], on_output: Optional[OnOutputChunk], bridge: Optional[CellBridge] = None,
  ) -> ExecutionHandle:
    handle_id = uuid.uuid4().hex
    data: Dict[str, Any] = {
      "task_id": task_id,
      "stdout": _OutputTail(self._config.max_output_bytes),
      "stderr": _OutputTail(self._config.max_output_bytes),
      "interrupt": asyncio.Event(),
      "completed": False,
    }
    handle = ExecutionHandle(self.name, handle_id, work_dir, data)

    async def run() -> Dict[str, Any]:
      try:
        return await self._cell(handle, code, bridge=bridge, timeout_ms=timeout_ms, env=env, on_output=on_output)
      finally:
        if bridge is not None:
          bridge.close()
        data["completed"] = True
        self._state.cells.pop(handle_id, None)

    data["cell_task"] = asyncio.create_task(run())
    self._state.cells[handle_id] = data
    return handle

  @contextlib.asynccontextmanager
  async def _cell_slot(self, interrupt: asyncio.Event) -> AsyncIterator[None]:
    """A queued cell settles on cancellation without interrupting the lock owner."""
    acquire_task = asyncio.create_task(self._state.lock.acquire())
    interrupt_task = asyncio.create_task(interrupt.wait())
    try:
      await asyncio.wait({acquire_task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED)
      if not acquire_task.done():
        acquire_task.cancel()
        await asyncio.gather(acquire_task, return_exceptions=True)
      yield
    finally:
      acquire_task.cancel()
      interrupt_task.cancel()
      await asyncio.gather(acquire_task, interrupt_task, return_exceptions=True)
      if not acquire_task.cancelled() and acquire_task.exception() is None:
        self._state.lock.release()

  async def _cell(
    self, handle: ExecutionHandle, code: str, *, bridge: Optional[CellBridge],
    timeout_ms: int, env: Optional[Dict[str, str]], on_output: Optional[OnOutputChunk],
  ) -> Dict[str, Any]:
    data = handle._backend_data
    started_at = time.monotonic()
    async with self._cell_slot(data["interrupt"]):
      if self._state.closed or data["interrupt"].is_set():
        return {
          "stdout": "", "stderr": "", "stdout_tail": "", "images": [], "displays": [],
          "return_code": 130, "cancelled": True, "timed_out": False, "truncated": False,
          "host_wait_ms": 0, "compute_ms": 0,
          "duration_ms": int((time.monotonic() - started_at) * 1000),
        }
      work_dir = Path(handle.work_dir)
      work_dir.mkdir(parents=True, exist_ok=True)
      cell_env = self._project_env(dict(env) if env is not None else _prepare_code_execute_env(self._config))
      runner = self._state.runner
      if runner is not None and (runner.process is None or runner.process.returncode is not None):
        await runner.aclose()
        runner = None
      if runner is None:
        runner = self._make_runner(str(work_dir))
        self._state.runner = runner
        try:
          await runner.start(cell_env)
        except BaseException:
          await asyncio.shield(runner.aclose())
          self._state.runner = None
          raise
      process = runner.process
      assert process is not None
      data["process"] = process
      data["container_id"] = getattr(runner, "container_id", None)
      if self._state.closed:
        await runner.aclose()
      script = _write_code_execute_script(work_dir, code, self._config, task_id=data["task_id"])
      data["script_path"] = script
      data["stdout_file"] = work_dir / f"_cell_{handle.handle_id}_stdout.log"
      data["stderr_file"] = work_dir / f"_cell_{handle.handle_id}_stderr.log"
      before_mtimes = _snapshot_image_mtimes(work_dir)
      started_ns = time.time_ns()
      try:
        if bridge is not None:
          bridge_child_env = bridge.child_env()
          token = bridge_child_env.get(CELL_TOKEN_ENV)
          cell_env.update(bridge_child_env)
          bridge.activate()
          bridge.close_child_ends()
        else:
          token = None
        with data["stdout_file"].open("w", encoding="utf-8", buffering=1) as stdout_log, \
            data["stderr_file"].open("w", encoding="utf-8", buffering=1) as stderr_log:
          def capture(stream: str, text: str) -> None:
            (stdout_log if stream == "stdout" else stderr_log).write(text)
            data[stream].append(text)
            if on_output is not None:
              on_output(stream, text)

          # Only the backend's projected environment can cross the frame.
          # A new wrapper refreshes telemetry/native signed claims per cell.
          env_delta = {
            key: cell_env.get(key)
            for key in runner.spawn_env.keys() | cell_env.keys()
            if runner.spawn_env.get(key) != cell_env.get(key)
          }
          result = await self._run_cell(
            handle, runner, {
              "type": "cell", "id": handle.handle_id, "script": self._script_name(script),
              "cell_token": token, "env": env_delta,
            },
            bridge=bridge, timeout_ms=timeout_ms, capture=capture,
          )
        generated = _collect_generated_images(
          work_dir, started_ns, self._config, before_mtimes=before_mtimes, task_id=data["task_id"] or None,
        )
        images = result["images"]
        for image in generated:
          if not image.get("skipped") and sum(not row.get("skipped") for row in images) >= self._config.max_images:
            image = {"filename": image["filename"], "skipped": True, "reason": f"exceeds {self._config.max_images} image limit"}
          images.append(image)
        # Ordinary output aligns with the subprocess backend: the full
        # stdout/stderr up to the byte cap (a measurement, with the spill
        # files below holding the complete text), not a 20-line tail. The
        # tail fields keep their running-status meaning.
        stdout, stdout_truncated = _read_truncated(
          data["stdout_file"], self._config.max_output_bytes,
        )
        stderr, stderr_truncated = _read_truncated(
          data["stderr_file"], self._config.max_output_bytes,
        )
        result.update(
          stdout=stdout, stderr=stderr,
          stdout_tail=data["stdout"].text(20), stderr_tail=data["stderr"].text(20),
          spill={"stdout": data["stdout_file"].name, "stderr": data["stderr_file"].name},
          truncated=bool(stdout_truncated or stderr_truncated),
          duration_ms=int((time.monotonic() - started_at) * 1000),
        )
        return result
      finally:
        if bridge is not None:
          bridge.deactivate()
        _remove_path(script)

  async def _run_cell(
    self, handle: ExecutionHandle, runner: _KernelProcess, cell_frame: Dict[str, Any], *,
    bridge: Optional[CellBridge], timeout_ms: int, capture: OnOutputChunk,
  ) -> Dict[str, Any]:
    process = runner.process
    assert process is not None and process.stdin is not None and process.stdout is not None
    stdin, stdout = process.stdin, process.stdout
    data = handle._backend_data
    budget = bridge.budget if bridge is not None else CellBudget()
    requests: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
    fault = asyncio.Event()
    protocol_error: str | None = None
    displays: list[Dict[str, Any]] = []
    images: list[Dict[str, Any]] = []
    # Two clocks: `timeout_ms` bounds compute only. The runner brackets every
    # bridge request made on the cell's own thread with host_wait_begin/
    # host_wait_end (a spawned thread's call sends none, so it never pauses a
    # computing interpreter); while any is open the compute deadline is paused,
    # and the runner's `blocked_ms` sum is `host_wait_ms`. A host wait never
    # times the cell out or interrupts it.
    host_waits: set[str] = set()
    host_wait_ms = 0
    compute_s = 0.0
    resumed_at: float | None = None
    clock = asyncio.Event()

    def pause_compute() -> None:
      nonlocal compute_s, resumed_at
      if resumed_at is not None:
        compute_s += time.monotonic() - resumed_at
        resumed_at = None
      clock.set()

    def resume_compute() -> None:
      nonlocal resumed_at
      if resumed_at is None and not host_waits:
        resumed_at = time.monotonic()
      clock.set()

    def compute_remaining_s() -> float | None:
      """Seconds of compute left; None while a host wait holds an unspent clock.

      A pause never hides compute already spent: a begin frame read after the
      deadline passed (a stalled loop) still times the cell out.
      """
      spent = compute_s + (time.monotonic() - resumed_at if resumed_at is not None else 0.0)
      remaining = timeout_ms / 1000 - spent
      return None if resumed_at is None and remaining > 0 else remaining

    def fail_protocol(message: str) -> None:
      nonlocal protocol_error
      if protocol_error is None:
        protocol_error = message
        fault.set()

    def add_bundle(bundle: Dict[str, Any]) -> None:
      displays.append(bundle)
      png = bundle.get("image/png")
      if isinstance(png, str):
        name = f"display_{len(displays)}.png"
        if sum(not row.get("skipped") for row in images) >= self._config.max_images:
          images.append({"filename": name, "skipped": True, "reason": f"exceeds {self._config.max_images} image limit"})
        elif len(png) > self._config.max_image_base64_bytes:
          images.append({"filename": name, "skipped": True, "reason": f"exceeds {self._config.max_image_base64_bytes // 1024}KB"})
        else:
          images.append({"filename": name, "media_type": "image/png", "data_base64": png})

    async def dispatch_requests() -> None:
      while True:
        frame = await requests.get()
        try:
          if bridge is not None:
            reply = await bridge.handle_decoded_frame(frame)
          else:
            reply = encode_frame(_error_reply(frame.get("type"), frame.get("call_id"), "bridge_unavailable", "no bridge for this cell"), budget)
          stdin.write(reply)
          await stdin.drain()
        except ProtocolError as exc:
          fail_protocol(str(exc))
        except (BrokenPipeError, ConnectionResetError):
          raise _KernelLost("runner closed stdin") from None

    async def read_frames() -> Dict[str, Any]:
      nonlocal host_wait_ms
      while True:
        try:
          line = await stdout.readline()
        except ValueError as exc:
          fail_protocol(str(exc))
          continue
        if not line:
          if bridge is not None:
            bridge.deactivate()
          raise _KernelLost("runner exited")
        try:
          frame = (
            bridge.decode_frame(line)
            if bridge is not None and protocol_error is None
            else decode_frame(line, budget if protocol_error is None else None)
          )
        except ProtocolError as exc:
          fail_protocol(str(exc))
          # A poisoned budget still needs the bounded terminal control frame;
          # otherwise an interrupt would discard an otherwise healthy kernel.
          try:
            frame = decode_frame(line)
          except ProtocolError:
            continue
        kind = frame.get("type")
        if kind in REQUEST_FRAME_TYPES:
          if protocol_error is None:
            requests.put_nowait(frame)
          continue
        if frame.get("id") != handle.handle_id:
          log.debug("code_execute dropped stale %s frame for %s", kind, frame.get("id"))
          continue
        if kind == "done":
          if bridge is not None:
            bridge.deactivate()
          if frame.get("protocol_error") or budget.error:
            fail_protocol(str(frame.get("protocol_error") or budget.error))
          return frame
        if kind == "host_wait_begin":
          host_waits.add(str(frame.get("call_id")))
          pause_compute()
          continue
        if kind == "host_wait_end":
          host_waits.discard(str(frame.get("call_id")))
          blocked_ms = frame.get("blocked_ms")
          if isinstance(blocked_ms, int):
            host_wait_ms += blocked_ms
          resume_compute()
          continue
        if protocol_error is not None:
          continue
        if kind in ("stdout", "stderr") and isinstance(frame.get("text"), str):
          capture(kind, frame["text"])
        elif kind == "display" and isinstance(frame.get("bundle"), dict):
          add_bundle(frame["bundle"])

    dispatch_task = asyncio.create_task(dispatch_requests())
    reader_task = asyncio.create_task(read_frames())
    interrupt_task = asyncio.create_task(data["interrupt"].wait())
    fault_task = asyncio.create_task(fault.wait())
    data["reader_task"] = reader_task
    timed_out = cancelled = False
    compute_ms = 0
    done: Dict[str, Any] | None = None
    lost: str | None = None
    try:
      stdin.write(encode_frame(cell_frame, budget))
      await stdin.drain()
      # The deadline is rearmed on every host-wait transition: `clock` wakes
      # this wait, and only unpaused elapsed time is spent from `timeout_ms`.
      # An expired deadline still collects tasks that finished at that moment.
      resume_compute()
      while True:
        remaining = compute_remaining_s()
        clock.clear()
        clock_task = asyncio.create_task(clock.wait())
        try:
          finished, _ = await asyncio.wait(
            {reader_task, interrupt_task, fault_task, dispatch_task, clock_task},
            timeout=None if remaining is None else max(remaining, 0.0), return_when=asyncio.FIRST_COMPLETED,
          )
        finally:
          clock_task.cancel()
        finished.discard(clock_task)
        if finished or (remaining is not None and remaining <= 0):
          break
      pause_compute()
      compute_ms = int(compute_s * 1000)
      if dispatch_task in finished:
        dispatch_task.result()
      if reader_task not in finished:
        cancelled = interrupt_task in finished
        timed_out = not finished
        dispatch_task.cancel()
        await asyncio.gather(dispatch_task, return_exceptions=True)
        await runner.interrupt()
        finished, _ = await asyncio.wait({reader_task}, timeout=_INTERRUPT_GRACE_S)
        if reader_task not in finished:
          lost = f"cell did not stop within {_INTERRUPT_GRACE_S:g} s of an interrupt"
      if lost is None:
        done = reader_task.result()
    except asyncio.CancelledError:
      dispatch_task.cancel()
      await asyncio.gather(dispatch_task, return_exceptions=True)
      await runner.interrupt()
      async def settle() -> None:
        finished, _ = await asyncio.wait({reader_task}, timeout=_INTERRUPT_GRACE_S)
        if reader_task not in finished or reader_task.cancelled() or reader_task.exception() is not None:
          await runner.aclose()
      await asyncio.shield(settle())
      raise
    except (BrokenPipeError, ConnectionResetError, _KernelLost) as exc:
      lost = str(exc)
    except ProtocolError as exc:
      fail_protocol(str(exc))
      lost = "could not send the cell control frame"
    finally:
      dispatch_task.cancel()
      interrupt_task.cancel()
      fault_task.cancel()
      reader_task.cancel()
      await asyncio.gather(dispatch_task, interrupt_task, fault_task, reader_task, return_exceptions=True)

    error = done.get("error") if done is not None else None
    if lost is not None:
      exit_code = await runner.exit_code() if lost == "runner exited" or process.returncode is not None else None
      diagnostics = "".join(runner.stderr_tail).strip()
      await runner.aclose()
      if self._state.runner is runner:
        self._state.runner = None
      capture("stderr", f"[kernel lost: {lost}; earlier interpreter state is gone]\n")
      if diagnostics:
        capture("stderr", diagnostics + "\n")
      return_code = 124 if timed_out else 130 if cancelled else exit_code if exit_code not in (None, 0) else 1
    else:
      cancelled = cancelled or bool(done and done.get("cancelled"))
      return_code = 124 if timed_out else 130 if cancelled else 1 if protocol_error or (done and done.get("status") == "error") else 0
      if timed_out:
        capture("stderr", f"[cell interrupted after {timeout_ms} ms timeout; kernel state kept]\n")
      elif isinstance(error, dict):
        lines = error.get("traceback")
        capture("stderr", "".join(str(line) for line in lines) if isinstance(lines, list) else f"{error.get('ename')}: {error.get('evalue')}\n")
    result: Dict[str, Any] = {
      "return_code": return_code, "images": images, "displays": displays,
      "timed_out": timed_out, "cancelled": cancelled,
      "host_wait_ms": host_wait_ms, "compute_ms": compute_ms,
    }
    if error is not None:
      result["error"] = error
    if protocol_error is not None:
      result["protocol_error"] = {"code": "protocol_error", "message": protocol_error}
      if error is None:
        capture("stderr", f"protocol_error: {protocol_error}\n")
    return result

  async def poll(self, handle: ExecutionHandle) -> Dict[str, Any]:
    data = handle._backend_data
    if data["completed"]:
      return {"status": "completed"}
    return {"status": "running", "stdout_tail": data["stdout"].text(20), "stderr_tail": data["stderr"].text(5)}

  async def cancel(self, handle: ExecutionHandle) -> None:
    data = handle._backend_data
    data["interrupt"].set()
    await asyncio.shield(asyncio.gather(data["cell_task"], return_exceptions=True))

  async def collect(self, handle: ExecutionHandle) -> Dict[str, Any]:
    task = handle._backend_data["cell_task"]
    try:
      return await asyncio.shield(task)
    except asyncio.CancelledError:
      await self.cancel(handle)
      raise

  async def cleanup(self, work_dir: str, *, task_id: str | None = None) -> None:
    if task_id is not None:
      _remove_task_artifacts(Path(work_dir), task_id)
    else:
      await self._state.aclose()
