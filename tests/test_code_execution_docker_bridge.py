"""Real Docker cells use the host reader, not a container-side tool registry.

Only an unavailable Docker daemon/CLI or execution image skips these tests.
The MCP fixture is deterministic; every Python cell and container lifecycle is
real, with mounts in the Docker-shared home state directory.
"""
import asyncio
import contextlib
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
# The suite uses importlib mode; explicitly expose the sibling fixture module.
for path in (PKG_DIR, Path(__file__).resolve().parent):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from agent_gateway import SessionStore
from agent_gateway.code_execution import CodeExecutionConfig, build_code_execution, cleanup_code_execution
from agent_gateway.code_execution._backends._docker import DockerBackend
from agent_gateway.code_execution._backends._kernel import KernelState
from agent_gateway.code_execution._frame_codec import MAX_FRAME_BYTES
from agent_gateway.code_execution._background import OutputRingBuffer
from agent_gateway.tool_dispatcher_helpers import ToolResult
from test_code_execution_bridge import (
  FRED_SERIES,
  GET_FILINGS,
  OUTER_ID,
  REFRESH_FILINGS,
  SEARCH_FILINGS,
  _CODE_CALLABLE,
  _CaptureMcp,
  _FILING_HANDLE,
  _bind,
  _fd_closed,
  _filings_extractor,
  _policy,
  _record_open_cells,
)


@pytest.fixture(scope="module")
def docker_image() -> str:
  image = os.environ.get("CODE_EXECUTE_DOCKER_IMAGE", "ai-excel-addin-code-exec:latest")
  reason = DockerBackend(image=image).unavailable_reason()
  if reason is not None:
    pytest.skip(reason)
  return image


@pytest.fixture
def docker_runtime(docker_image: str, monkeypatch: pytest.MonkeyPatch, _policy: Any) -> Any:
  # macOS's OS temp directory is not reliably shared with Docker Desktop.
  state_root = Path.home() / ".local" / "state" / "hank"
  state_root.mkdir(parents=True, exist_ok=True)
  handles: list[Any] = []
  original_start = DockerBackend.start_bridged

  async def recording_start(self: DockerBackend, *args: Any, **kwargs: Any) -> Any:
    handle = await original_start(self, *args, **kwargs)
    handles.append((self, handle))
    return handle

  monkeypatch.setattr(DockerBackend, "start_bridged", recording_start)
  with tempfile.TemporaryDirectory(prefix="docker-bridge-test-", dir=state_root) as root:
    yield CodeExecutionConfig(
      docker_image=docker_image,
      register_subprocess=False,
      work_dir_root=root,
      max_output_bytes=4 * 1024 * 1024,
    ), handles


class _BlockedMcp(_CaptureMcp):
  """Keep a real nested dispatcher call pending until its channel cancels it."""

  def __init__(self, *, frame_reader: Any) -> None:
    super().__init__(frame_reader=frame_reader)
    self.entered = asyncio.Event()
    self.finished = asyncio.Event()

  async def call_tool(self, *args: Any, **kwargs: Any) -> ToolResult:
    result = await super().call_tool(*args, **kwargs)
    self.entered.set()
    try:
      await asyncio.Event().wait()
      return result
    except asyncio.CancelledError:
      self.cancelled = True
      raise
    finally:
      self.finished.set()


def _setup(config: CodeExecutionConfig, *, blocked: bool = False) -> tuple[Any, ...]:
  session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
  bundle = build_code_execution(session, config=config)
  mcp = (_BlockedMcp if blocked else _CaptureMcp)(frame_reader=bundle.frame_reader)
  dispatcher = _bind(bundle, mcp, extract_sources=_filings_extractor)
  return session, bundle, mcp, dispatcher, _record_open_cells(bundle)


async def _run_cell(dispatcher: Any, code: str, **extra: Any) -> tuple[Any, Any]:
  return await dispatcher.dispatch(
    OUTER_ID,
    "code_execute",
    {"code": textwrap.dedent(code), "host": "docker", **extra},
  )


async def _collect_background(dispatcher: Any, task_id: str) -> Any:
  async def collect() -> Any:
    while True:
      result, error = await dispatcher.dispatch("toolu_status", "code_execute_status", {"task_id": task_id})
      assert error is None, error
      if result.get("status") != "running":
        return result
      await asyncio.sleep(0.05)

  return await asyncio.wait_for(collect(), timeout=30)


async def _assert_cell_finished(
  bundle: Any,
  opened: list[tuple[Any, dict[int, os.stat_result]]],
  handles: list[Any],
) -> None:
  assert bundle.frame_reader.active_cells() == ()
  for cell, original_fds in opened:
    assert all(_fd_closed(fd, original) for fd, original in original_fds.items())
    assert cell._raw_fds == set()
    assert cell._pipe_files == []
    assert cell._read_transport is None
    assert cell._write_transport is None
  assert len(handles) == len(opened)
  for _backend, handle in handles:
    assert handle._backend_data["reader_task"].done()
    assert handle._backend_data["cell_task"].done()


async def _assert_runtime_running(handles: list[Any]) -> None:
  assert handles
  container_ids = {handle._backend_data["container_id"] for _backend, handle in handles}
  assert len(container_ids) == 1
  backend, handle = handles[-1]
  container_id = handle._backend_data["container_id"]
  code, stdout, stderr = await backend._docker_exec(
    "container", "inspect", "--format", "{{.State.Running}}", container_id,
  )
  assert code == 0 and stdout.strip() == "true", stderr
  assert handle._backend_data["process"].returncode is None


async def _teardown(session: Any, bundle: Any, opened: list[Any], handles: list[Any]) -> None:
  await cleanup_code_execution(session)
  await _assert_cell_finished(bundle, opened, handles)
  for backend, handle in handles:
    container_id = handle._backend_data["container_id"]
    code, stdout, stderr = await backend._docker_exec("container", "inspect", container_id, check=False)
    assert code != 0, stdout
    assert "No such container" in stderr and container_id in stderr, stderr
    process = handle._backend_data["process"]
    assert process.returncode is not None
    assert process.stdin is not None and process.stdin.is_closing()
    for fd in (0, 1, 2):
      transport = process._transport.get_pipe_transport(fd)
      assert transport is None or transport.is_closing()


@contextlib.asynccontextmanager
async def _backend(config: CodeExecutionConfig, *, state: KernelState | None = None) -> AsyncIterator[DockerBackend]:
  backend = DockerBackend(image=config.docker_image, config=config, state=state)
  try:
    yield backend
  finally:
    container_id = backend.runtime.container_id
    process = backend.runtime.process
    await backend.runtime.aclose()
    if container_id is not None:
      code, _stdout, stderr = await backend._docker_exec("container", "inspect", container_id, check=False)
      assert code != 0 and "No such container" in stderr, stderr
    if process is not None:
      assert process.returncode is not None


def _assert_no_new_tasks(before: set[Any], handles: list[Any]) -> None:
  owned = {
    backend.runtime.runner.stderr_task
    for backend, _handle in handles
    if backend.runtime.runner is not None
  }
  leaked = {task for task in asyncio.all_tasks() if not task.done()} - before - owned
  assert leaked == set(), leaked


@pytest.mark.parametrize("background", [False, True], ids=["foreground", "background"])
def test_docker_nested_results_compose_and_preserve_ledger_sources(docker_runtime: Any, background: bool) -> None:
  async def run() -> None:
    config, handles = docker_runtime
    session, bundle, mcp, dispatcher, opened = _setup(config)
    before = {task for task in asyncio.all_tasks() if not task.done()}
    try:
      result, error = await _run_cell(dispatcher, f"""
        filings = hank.call({GET_FILINGS!r}, cik="0000320193", limit=2)
        series = hank.call({FRED_SERIES!r}, series_id="CPIAUCSL")
        print(round(float(series["observations"][0]["value"]) / len(filings["filings"]), 2))
      """, background=background)
      assert error is None, error
      assert result["backend"] == "docker" and result["sandboxed"] is True
      if background:
        task_id = result["task_id"]
        result = await _collect_background(dispatcher, task_id)
        assert task_id not in session.background_tasks
      assert result["return_code"] == 0, result
      assert result["stdout"] == "160.7\n"
      assert result["stderr"] == ""
      assert mcp.calls == [
        (GET_FILINGS, {"cik": "0000320193", "limit": 2}),
        (FRED_SERIES, {"series_id": "CPIAUCSL"}),
      ]
      assert mcp.active_during_call == [(OUTER_ID,), (OUTER_ID,)]
      nested = result["nested_calls"]
      assert [entry["nested_call_id"] for entry in nested] == [f"{OUTER_ID}.1", f"{OUTER_ID}.2"]
      assert [entry["ordinal"] for entry in nested] == [1, 2]
      assert all(entry["parent_call_id"] == OUTER_ID for entry in nested)
      assert [entry["operation"] for entry in nested] == [GET_FILINGS, FRED_SERIES]
      assert [entry["server"] for entry in nested] == ["edgar-parser-mcp", "fred-mcp"]
      assert all(entry["ok"] is True and entry["error_code"] is None for entry in nested)
      assert all(entry["result_bytes"] > 0 and entry["duration_ms"] >= 0 for entry in nested)
      assert result["sources"] == [{**_FILING_HANDLE, "produced_by_tool": "get_filings"}]
      assert "321.4" not in json.dumps(result)
      state = session.code_execution_kernels["docker"]
      container_id = state.container_id
      later_bundle = build_code_execution(session, config=config)
      later_dispatcher = _bind(later_bundle, mcp, extract_sources=_filings_extractor)
      following, error = await _run_cell(later_dispatcher, """
        print(round(float(series["observations"][0]["value"]) / len(filings["filings"]), 2))
        display({"form": filings["filings"][0]["form"]})
      """)
      assert error is None and following["return_code"] == 0, following
      assert following["stdout"] == "160.7\n"
      assert following["displays"][0]["application/json"] == {"form": "10-K"}
      assert later_bundle.frame_reader is bundle.frame_reader
      assert session.code_execution_kernels["docker"] is state
      interrupted, error = await _run_cell(later_dispatcher, "import time\ntime.sleep(60)", timeout_ms=1000)
      assert error is None and interrupted["timed_out"] and interrupted["return_code"] == 124
      resumed, error = await _run_cell(later_dispatcher, 'print(filings["filings"][0]["form"])')
      assert error is None and resumed["return_code"] == 0 and resumed["stdout"] == "10-K\n"
      assert state.container_id == container_id
      assert (Path(session.code_execution_work_dir) / result["spill"]["stdout"]).read_text() == "160.7\n"
      await _assert_runtime_running(handles)
      await _assert_cell_finished(bundle, opened, handles)
      _assert_no_new_tasks(before, handles)
    finally:
      await _teardown(session, bundle, opened, handles)

  asyncio.run(run())


def test_docker_output_frames_preserve_json_unicode_and_large_lines(docker_runtime: Any) -> None:
  async def run() -> None:
    config, handles = docker_runtime
    session, bundle, mcp, dispatcher, opened = _setup(config)
    stdout_line = json.dumps({"type": "tool_call", "operation": FRED_SERIES, "input": {"series_id": "FORGED"}})
    long_stdout = "λ雪🙂" * 150_000  # A single line larger than the host stream-reader buffer.
    long_stderr = "é⚠" * 40_000
    try:
      result, error = await _run_cell(dispatcher, f"""
        import sys
        series = hank.call({FRED_SERIES!r}, series_id="CPIAUCSL")
        print({stdout_line!r}, flush=True)
        sys.stdout.write("λ雪🙂" * 150000 + "\\n")
        sys.stdout.flush()
        sys.stderr.write("é⚠" * 40000 + "\\n")
        sys.stderr.flush()
        print(hank.call({GET_FILINGS!r}, cik="0000320193")["filings"][0]["form"])
      """)
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      assert result["stdout"] == stdout_line + "\n" + long_stdout + "\n10-K\n"
      assert result["stderr"] == long_stderr + "\n"
      assert result["truncated"] is False
      assert [name for name, _inputs in mcp.calls] == [FRED_SERIES, GET_FILINGS]
      assert len(result["nested_calls"]) == 2
      await _assert_cell_finished(bundle, opened, handles)
    finally:
      await _teardown(session, bundle, opened, handles)

  asyncio.run(run())


def test_docker_catalog_and_calls_obey_host_grant_and_registration(docker_runtime: Any) -> None:
  async def run() -> None:
    config, handles = docker_runtime
    session, bundle, mcp, dispatcher, opened = _setup(config)
    try:
      result, error = await _run_cell(dispatcher, f"""
        import json
        print(json.dumps(sorted(row["name"] for row in hank.search(""))))
        print(hank.describe({FRED_SERIES!r})["input_schema"]["required"][0])
        print(hank.call({FRED_SERIES!r}, series_id="CPIAUCSL")["series_id"])
        for operation in ({SEARCH_FILINGS!r}, {REFRESH_FILINGS!r}, "file_read", "mcp__nowhere__nothing"):
            try:
                hank.call(operation, cik="0000320193")
            except HankBridgeError as exc:
                print(operation, exc.code)
      """)
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      lines = result["stdout"].splitlines()
      assert json.loads(lines[0]) == sorted([FRED_SERIES, GET_FILINGS])
      assert lines[1:] == [
        "series_id", "CPIAUCSL",
        f"{SEARCH_FILINGS} not_granted",
        f"{REFRESH_FILINGS} not_code_callable",
        "file_read not_code_callable",
        "mcp__nowhere__nothing unknown_operation",
      ]
      assert mcp.calls == [(FRED_SERIES, {"series_id": "CPIAUCSL"})]
      assert [(entry["operation"], entry["error_code"]) for entry in result["nested_calls"]] == [
        (FRED_SERIES, None),
        (SEARCH_FILINGS, "not_granted"),
        (REFRESH_FILINGS, "not_code_callable"),
        ("file_read", "not_code_callable"),
        ("mcp__nowhere__nothing", "unknown_operation"),
      ]
      assert result["sources"] == []
      await _assert_cell_finished(bundle, opened, handles)
    finally:
      await _teardown(session, bundle, opened, handles)

  asyncio.run(run())


@pytest.mark.parametrize("background", [False, True], ids=["foreground", "background"])
def test_docker_timeout_cancels_blocked_nested_dispatch(docker_runtime: Any, background: bool) -> None:
  async def run() -> None:
    config, handles = docker_runtime
    session, bundle, mcp, dispatcher, opened = _setup(config, blocked=True)
    before = {task for task in asyncio.all_tasks() if not task.done()}
    running = asyncio.create_task(_run_cell(dispatcher, f"""
      saved = 71
      print("calling", flush=True)
      hank.call({FRED_SERIES!r}, series_id="CPIAUCSL")
      print("unreachable")
    """, background=background, timeout_ms=5000))
    try:
      await asyncio.wait_for(mcp.entered.wait(), timeout=20)
      assert bundle.frame_reader.active_cells() == (OUTER_ID,)
      result, error = await asyncio.wait_for(running, timeout=30)
      assert error is None, error
      if background:
        result = await _collect_background(dispatcher, result["task_id"])
      assert result["timed_out"] is True
      assert result["return_code"] != 0
      assert result["stdout"] == "calling\n"
      assert mcp.cancelled is True and mcp.finished.is_set()
      assert [(entry["ok"], entry["error_code"]) for entry in result["nested_calls"]] == [(False, "cancelled")]
      assert session.background_tasks == {}
      following, error = await _run_cell(dispatcher, "print(saved)")
      assert error is None and following["return_code"] == 0 and following["stdout"] == "71\n"
      await _assert_runtime_running(handles)
      await _assert_cell_finished(bundle, opened, handles)
      _assert_no_new_tasks(before, handles)
    finally:
      running.cancel()
      await asyncio.gather(running, return_exceptions=True)
      await _teardown(session, bundle, opened, handles)

  asyncio.run(run())


@pytest.mark.parametrize("termination", ["foreground_cancel", "background_cancel", "session_cleanup"])
def test_docker_cancel_during_nested_dispatch_releases_every_owner(docker_runtime: Any, termination: str) -> None:
  async def run() -> None:
    config, handles = docker_runtime
    session, bundle, mcp, dispatcher, opened = _setup(config, blocked=True)
    before = {task for task in asyncio.all_tasks() if not task.done()}
    running = asyncio.create_task(_run_cell(
      dispatcher,
      f'saved = 67\nhank.call({FRED_SERIES!r}, series_id="CPIAUCSL")\nprint("unreachable")',
      background=termination != "foreground_cancel",
    ))
    try:
      await asyncio.wait_for(mcp.entered.wait(), timeout=20)
      assert bundle.frame_reader.active_cells() == (OUTER_ID,)
      [(_backend, handle)] = handles
      code, stdout, stderr = await _backend._docker_exec(
        "container", "inspect", "--format", "{{.State.Running}}", handle._backend_data["container_id"],
      )
      assert code == 0 and stdout.strip() == "true", stderr
      if termination == "foreground_cancel":
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
          await asyncio.wait_for(running, timeout=20)
      else:
        started, error = await asyncio.wait_for(running, timeout=20)
        assert error is None, error
        if termination == "background_cancel":
          result, error = await asyncio.wait_for(dispatcher.dispatch(
            "toolu_cancel", "code_execute_status", {"task_id": started["task_id"], "cancel": True},
          ), timeout=20)
          assert error is None, error
          assert result == {"status": "cancelled", "task_id": started["task_id"]}
        else:
          await asyncio.wait_for(_teardown(session, bundle, opened, handles), timeout=20)
      assert mcp.cancelled is True and mcp.finished.is_set()
      assert mcp.calls == [(FRED_SERIES, {"series_id": "CPIAUCSL"})]
      assert session.background_tasks == {}
      [(cell, _original_fds)] = opened
      assert [(entry["ok"], entry["error_code"]) for entry in cell.summary()["nested_calls"]] == [(False, "cancelled")]
      if termination != "session_cleanup":
        following, error = await _run_cell(dispatcher, "print(saved)")
        assert error is None and following["return_code"] == 0 and following["stdout"] == "67\n"
        await _assert_runtime_running(handles)
      await _assert_cell_finished(bundle, opened, handles)
      _assert_no_new_tasks(before, handles)
    finally:
      running.cancel()
      await asyncio.gather(running, return_exceptions=True)
      await _teardown(session, bundle, opened, handles)

  asyncio.run(run())


@pytest.mark.parametrize("background", [False, True], ids=["foreground", "background"])
def test_docker_cell_done_cancels_pending_dispatch_without_blocking_output(docker_runtime: Any, background: bool) -> None:
  async def run() -> None:
    config, handles = docker_runtime
    session, bundle, mcp, dispatcher, opened = _setup(config, blocked=True)
    before = {task for task in asyncio.all_tasks() if not task.done()}
    running = asyncio.create_task(_run_cell(dispatcher, f"""
      import pathlib, threading, time
      def pending(error_type=HankBridgeError):
          try:
              hank.call({FRED_SERIES!r}, series_id="CPIAUCSL")
          except error_type:
              pass
      threading.Thread(target=pending, daemon=True).start()
      while not pathlib.Path("release-exit").exists():
          time.sleep(0.01)
      print("done", flush=True)
    """, background=background, timeout_ms=5000))
    try:
      await asyncio.wait_for(mcp.entered.wait(), timeout=20)
      [(_backend, handle)] = handles
      (Path(handle.work_dir) / "release-exit").touch()
      result, error = await asyncio.wait_for(running, timeout=20)
      assert error is None, error
      if background:
        result = await _collect_background(dispatcher, result["task_id"])
      assert mcp.entered.is_set()
      assert result["stdout"] == "done\n"
      assert result["return_code"] == 0 and result["timed_out"] is False, result
      assert mcp.cancelled and mcp.finished.is_set()
      assert [(entry["ok"], entry["error_code"]) for entry in result["nested_calls"]] == [(False, "cancelled")]
      await _assert_runtime_running(handles)
      await _assert_cell_finished(bundle, opened, handles)
      _assert_no_new_tasks(before, handles)
    finally:
      running.cancel()
      await asyncio.gather(running, return_exceptions=True)
      await _teardown(session, bundle, opened, handles)

  asyncio.run(run())


def test_docker_live_output_is_lines_and_tail_keeps_last_twenty(docker_runtime: Any) -> None:
  async def run() -> None:
    config, _handles = docker_runtime
    lines = [f"{i:03d} " + "x" * 36 + "\n" for i in range(300)] + ["END\n"]
    events: list[str] = []
    ring = OutputRingBuffer()

    def output(stream: str, text: str) -> None:
      assert stream == "stdout"
      events.append(text)
      ring.append(stream, text)

    async with _backend(config) as backend:
      result = await backend.execute(
        'import sys\nsys.stdout.write("".join(f"{i:03d} " + "x" * 36 + "\\n" for i in range(300)))\nprint("END")',
        config.work_dir_root, on_output=output,
      )
      assert result["return_code"] == 0, result
      assert result["stdout"] == result["stdout_tail"] == "".join(lines[-20:])
      assert result["truncated"] is True
      assert (Path(config.work_dir_root) / result["spill"]["stdout"]).read_text() == "".join(lines)
      assert events == lines
      assert ring.tail(20) == "".join(lines[-20:])

  asyncio.run(run())


def test_docker_descendant_pipe_does_not_mask_cell_exit(docker_runtime: Any) -> None:
  async def run() -> None:
    config, handles = docker_runtime
    session, bundle, _mcp, dispatcher, opened = _setup(config)
    try:
      result, error = await _run_cell(dispatcher, """
        import subprocess
        subprocess.Popen(["sleep", "120"])
        print("done")
      """, timeout_ms=5000)
      assert error is None, error
      assert result["stdout"] == "done\n"
      assert result["return_code"] == 0 and result["timed_out"] is False, result
      await _assert_cell_finished(bundle, opened, handles)
    finally:
      await _teardown(session, bundle, opened, handles)

  asyncio.run(run())


@pytest.mark.parametrize("direction", ["request", "reply"])
def test_docker_frame_bounds_fail_only_cell_and_keep_interpreter(docker_runtime: Any, direction: str) -> None:
  async def run() -> None:
    config, handles = docker_runtime
    session, bundle, mcp, dispatcher, opened = _setup(config)

    async def large_result(*args: Any, **kwargs: Any) -> ToolResult:
      await _CaptureMcp.call_tool(mcp, *args, **kwargs)
      return {"payload": "x" * MAX_FRAME_BYTES}, None

    if direction == "reply":
      mcp.call_tool = large_result
    argument = "'x' * 2097152" if direction == "request" else "'CPIAUCSL'"
    try:
      result, error = await _run_cell(dispatcher, f"""
        saved = 88
        try:
            hank.call({FRED_SERIES!r}, series_id={argument})
        except HankBridgeError:
            pass
      """)
      assert error is None, error
      assert result["return_code"] == 1 and result["protocol_error"]["code"] == "protocol_error", result
      assert len(mcp.calls) == (1 if direction == "reply" else 0)
      if direction == "reply":
        assert [(entry["ok"], entry["error_code"]) for entry in result["nested_calls"]] == [(False, "protocol_error")]
      else:
        assert result.get("nested_calls", []) == []
      following, error = await _run_cell(dispatcher, "print(saved)")
      assert error is None and following["return_code"] == 0
      assert following["stdout"] == "88\n"
      await _assert_runtime_running(handles)
      await _assert_cell_finished(bundle, opened, handles)
    finally:
      await _teardown(session, bundle, opened, handles)

  asyncio.run(run())


def test_docker_signal_exit_uses_docker_status(docker_runtime: Any) -> None:
  async def run() -> None:
    config, handles = docker_runtime
    session, bundle, _mcp, dispatcher, opened = _setup(config)
    try:
      result, error = await _run_cell(dispatcher, """
        import os, signal
        print("before signal", flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
      """)
      assert error is None, error
      assert result["stdout"] == "before signal\n"
      assert result["return_code"] == 137 and result["timed_out"] is False, result
      await _assert_cell_finished(bundle, opened, handles)
    finally:
      await _teardown(session, bundle, opened, handles)

  asyncio.run(run())


@pytest.mark.parametrize("bridged", [False, True])
def test_docker_raw_json_like_stdout_preserves_good_output(docker_runtime: Any, bridged: bool) -> None:
  async def run() -> None:
    config, handles = docker_runtime
    session, bundle, _mcp, dispatcher, opened = _setup(config)
    code = """
      import os
      print("before", flush=True)
      os.write(1, b'{"type":"stdout"}\\nnot-json\\n')
      print("after", flush=True)
    """
    try:
      if bridged:
        result, error = await _run_cell(dispatcher, code)
        assert error is None, error
      else:
        async with _backend(config) as backend:
          result = await backend.execute(textwrap.dedent(code), config.work_dir_root)
      assert result["return_code"] == 0, result
      assert sorted(result["stdout"].splitlines()) == sorted(["before", '{"type":"stdout"}', "not-json", "after"])
      assert result["stderr"] == ""
      if bridged:
        await _assert_cell_finished(bundle, opened, handles)
    finally:
      await _teardown(session, bundle, opened, handles)

  asyncio.run(run())


def test_docker_single_output_line_is_split_into_bounded_frames_without_loss(docker_runtime: Any) -> None:
  async def run() -> None:
    config, _handles = docker_runtime
    config.max_output_bytes = 20 * 1024 * 1024
    size = 17 * 1024 * 1024
    events: list[tuple[str, str]] = []
    async with _backend(config) as backend:
      result = await backend.execute(
        f'print("x" * {size})\nprint("END")',
        config.work_dir_root, on_output=lambda stream, text: events.append((stream, text)),
      )
      assert result["return_code"] == 0 and result["timed_out"] is False, result["stderr"]
      assert result["stderr"] == "" and result["truncated"] is False
      assert result["stdout"] == "x" * size + "\nEND\n"
      assert "".join(text for stream, text in events if stream == "stdout") == result["stdout"]
      assert all(len(text.encode("utf-8")) < MAX_FRAME_BYTES for _stream, text in events)

  asyncio.run(run())


def test_docker_session_states_are_isolated_even_with_same_workdir(docker_runtime: Any) -> None:
  async def run() -> None:
    config, _handles = docker_runtime
    async with _backend(config) as first, _backend(config) as second:
      assert (await first.execute("private_value = 7", config.work_dir_root))["return_code"] == 0
      isolated = await second.execute("print('private_value' in globals())", config.work_dir_root)
      assert isolated["stdout"] == "False\n" and isolated["return_code"] == 0
      first_id = first.runtime.container_id
      assert first_id is not None
      assert first_id != second.runtime.container_id
      await first.runtime.aclose()
      code, _stdout, stderr = await first._docker_exec("container", "inspect", first_id, check=False)
      assert code != 0 and "No such container" in stderr, stderr
      following = await second.execute("print(17)", config.work_dir_root)
      assert following["stdout"] == "17\n" and following["return_code"] == 0

  asyncio.run(run())


def test_docker_current_cell_identity_refreshes_without_forwarding_secrets(docker_runtime: Any) -> None:
  async def run() -> None:
    config, _handles = docker_runtime
    key = "AGENT_TELEMETRY_TOOL_CALL_ID"
    forbidden = ["EDGAR_API_KEY", "AGENT_API_CLAIM_SIGNATURE", "UNREGISTERED_CELL_SECRET"]
    code = f"""
import json, os, pathlib
print(os.environ.get({key!r}))
print(json.dumps([name for name in {forbidden!r} if name in os.environ]))
print(os.environ.get("AGENT_CODE_EXECUTE_CELL_TOKEN"))
pathlib.Path(os.environ["AGENT_CODE_EXECUTE_WORK_DIR"], os.environ[{key!r}] + ".identity").write_text(os.environ[{key!r}])
"""
    env = {
      key: "first-cell", "AGENT_CODE_EXECUTE_WORK_DIR": "/workspace",
      "EDGAR_API_KEY": "not-forwarded", "AGENT_API_CLAIM_SIGNATURE": "not-forwarded",
      "UNREGISTERED_CELL_SECRET": "not-forwarded",
    }
    async with _backend(config) as backend:
      first = await backend.execute(code, config.work_dir_root, env=env)
      container_id = backend.runtime.container_id
      assert container_id is not None
      later = DockerBackend(image=config.docker_image, config=config, state=backend.runtime)
      second = await later.execute(code, config.work_dir_root, env={**env, key: "second-cell"})
      assert first["stdout"] == "first-cell\n[]\nNone\n" and first["return_code"] == 0
      assert second["stdout"] == "second-cell\n[]\nNone\n" and second["return_code"] == 0
      assert backend.runtime.container_id == container_id
      root = Path(config.work_dir_root)
      assert (root / "first-cell.identity").read_text() == "first-cell"
      assert (root / "second-cell.identity").read_text() == "second-cell"
      third = await later.execute("import os\nprint(os.environ.get('AGENT_TELEMETRY_TOOL_CALL_ID'))", config.work_dir_root, env={})
      assert third["stdout"] == "None\n" and third["return_code"] == 0
      _status, state_json, _stderr = await backend._docker_exec("inspect", container_id)
      [state] = json.loads(state_json)
      assert state["HostConfig"]["NetworkMode"] == "none"
      assert state["HostConfig"]["Memory"] == 512 * 1024 * 1024
      assert state["HostConfig"]["NanoCpus"] == 1_000_000_000
      assert state["HostConfig"]["PidsLimit"] == 128
      assert state["Config"]["WorkingDir"] == "/workspace"
      assert [mount["Destination"] for mount in state["Mounts"]] == ["/workspace"]
      assert all(name not in {entry.partition("=")[0] for entry in state["Config"]["Env"]} for name in forbidden)

  asyncio.run(run())


def test_docker_catalog_is_refetched_when_next_wrapper_has_narrower_grant(docker_runtime: Any) -> None:
  async def run() -> None:
    config, handles = docker_runtime
    session, bundle, mcp, dispatcher, opened = _setup(config)
    try:
      first, error = await _run_cell(dispatcher, "print(sorted(row['name'] for row in hank.search('')))")
      assert error is None and first["return_code"] == 0
      assert first["stdout"] == repr(sorted([GET_FILINGS, FRED_SERIES])) + "\n"
      later_bundle = build_code_execution(session, config=config)
      later = _bind(later_bundle, mcp)
      later_bundle.bind_dispatcher(
        later,
        tool_definitions=lambda: [row for row in later.get_tool_definitions() if row["name"] != FRED_SERIES],
        code_callable=_CODE_CALLABLE,
      )
      result, error = await _run_cell(later, f"""
        print([row["name"] for row in hank.search("")])
        try:
            hank.describe({FRED_SERIES!r})
        except HankBridgeError as exc:
            print(exc.code)
        try:
            hank.call({FRED_SERIES!r}, series_id="CPIAUCSL")
        except HankBridgeError as exc:
            print(exc.code)
      """)
      assert error is None and result["return_code"] == 0, result
      assert result["stdout"].splitlines() == [repr([GET_FILINGS]), "not_granted", "not_granted"]
      assert mcp.calls == []
      await _assert_runtime_running(handles)
    finally:
      await _teardown(session, bundle, opened, handles)

  asyncio.run(run())


def test_docker_background_thread_cannot_borrow_next_cell_token(docker_runtime: Any) -> None:
  async def run() -> None:
    config, handles = docker_runtime
    session, bundle, mcp, dispatcher, opened = _setup(config)
    try:
      first, error = await _run_cell(dispatcher, f"""
        import pathlib, threading
        release = threading.Event()
        def old_thread():
            release.wait()
            try:
                hank.call({FRED_SERIES!r}, series_id="STALE")
                outcome = "borrowed future token"
            except HankBridgeError as exc:
                outcome = exc.code
            pathlib.Path("thread-result").write_text(outcome)
        worker = threading.Thread(target=old_thread, daemon=True)
        worker.start()
      """)
      assert error is None and first["return_code"] == 0
      second, error = await _run_cell(dispatcher, """
        release.set()
        worker.join(5)
        print(pathlib.Path("thread-result").read_text())
      """)
      assert error is None and second["return_code"] == 0, second
      assert second["stdout"] == "bridge_closed\n"
      assert mcp.calls == []
      legitimate, error = await _run_cell(dispatcher, f'print(hank.call({FRED_SERIES!r}, series_id="CPIAUCSL")["series_id"])')
      assert error is None and legitimate["return_code"] == 0
      assert legitimate["stdout"] == "CPIAUCSL\n"
      assert mcp.calls == [(FRED_SERIES, {"series_id": "CPIAUCSL"})]
      await _assert_runtime_running(handles)
    finally:
      await _teardown(session, bundle, opened, handles)

  asyncio.run(run())


def test_docker_explicit_display_and_preamble_plot_images_remain_consumable(docker_runtime: Any) -> None:
  async def run() -> None:
    config, _handles = docker_runtime
    async with _backend(config) as backend:
      warmup = await backend.execute("prepared = 1", config.work_dir_root)
      assert warmup["return_code"] == 0
      result = await backend.execute(
        "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.plot([1, 2], [3, 5])\ndisplay(fig)\nplt.show()",
        config.work_dir_root,
      )
      assert result["return_code"] == 0, result
      assert "image/png" in result["displays"][0]
      images = [image for image in result["images"] if not image.get("skipped")]
      assert len(images) == 2
      assert all(image["media_type"] == "image/png" and image["data_base64"] for image in images)
      assert any(image["filename"].startswith("_plot_") for image in images)

  asyncio.run(run())


def test_docker_data_frames_obey_per_cell_aggregate_and_next_cell_recovers(docker_runtime: Any) -> None:
  async def run() -> None:
    config, handles = docker_runtime
    session, bundle, _mcp, dispatcher, opened = _setup(config)
    try:
      result, error = await _run_cell(dispatcher, """
        import sys
        saved = 97
        for i in range(70):
            sys.stdout.write("x" * (1024 * 1024))
      """, timeout_ms=30_000)
      assert error is None, error
      assert result["return_code"] == 1 and result["protocol_error"]["code"] == "protocol_error", result
      assert (Path(session.code_execution_work_dir) / result["spill"]["stdout"]).stat().st_size <= 64 * 1024 * 1024
      following, error = await _run_cell(dispatcher, "print(saved)")
      assert error is None and following["return_code"] == 0 and following["stdout"] == "97\n"
      await _assert_runtime_running(handles)
    finally:
      await _teardown(session, bundle, opened, handles)

  asyncio.run(run())
