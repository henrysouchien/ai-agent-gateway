"""Session-owned kernel behavior against real, isolated Python interpreters."""
import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
for path in (PKG_DIR, Path(__file__).resolve().parent):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from agent_gateway import SessionStore
from agent_gateway.code_execution import CodeExecutionConfig, build_code_execution, cleanup_code_execution
from agent_gateway.code_execution._frame_reader import FrameReader
from agent_gateway.code_execution._backends._kernel import KernelBackend, KernelState
from agent_gateway.code_execution._frame_codec import MAX_FRAME_BYTES
from test_code_execution_bridge import FRED_SERIES, GET_FILINGS, _CODE_CALLABLE, _CaptureMcp, _bind, _filings_extractor, _policy


@contextlib.asynccontextmanager
async def _kernel(config: CodeExecutionConfig | None = None) -> AsyncIterator[KernelBackend]:
  backend = KernelBackend(config=config or CodeExecutionConfig(register_docker=False))
  try:
    yield backend
  finally:
    await backend.runtime.aclose()


def test_request_wrappers_reuse_session_namespace_without_freezing_preamble(tmp_path: Path) -> None:
  state = KernelState()  # Session/bundle construction happens outside the loop.

  async def run() -> None:
    first = KernelBackend(config=CodeExecutionConfig(preamble_suffix="configured = 1"), state=state)
    second = KernelBackend(config=CodeExecutionConfig(preamble_suffix="configured = 2"), state=state)
    try:
      assert first.name == "kernel" and first.sandboxed is False
      assert state.pid is None
      initial = await first.execute("answer = 42", str(tmp_path))
      pid = state.pid
      following = await second.execute("print(answer, configured)", str(tmp_path))
      assert initial["return_code"] == following["return_code"] == 0
      assert following["stdout"] == "42 2\n"
      assert state.pid == pid and pid is not None
      assert first.runtime is second.runtime is state
    finally:
      await state.aclose()

  asyncio.run(run())


def test_separate_sessions_do_not_alias_even_when_workdir_matches(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as first, _kernel() as second:
      initial = await first.execute("private_value = 7", str(tmp_path))
      isolated = await second.execute("print('private_value' in globals())", str(tmp_path))
      assert initial["return_code"] == isolated["return_code"] == 0
      assert isolated["stdout"] == "False\n"
      assert first.runtime.pid != second.runtime.pid
      await first.runtime.aclose()
      alive = await second.execute("print(17)", str(tmp_path))
      assert alive["stdout"] == "17\n" and alive["return_code"] == 0

  asyncio.run(run())


def test_session_cleanup_releases_remaining_resources_when_one_close_fails(tmp_path: Path) -> None:
  class _Kernel(KernelState):
    def __init__(self, error: Exception | None = None) -> None:
      super().__init__()
      self.closed = False
      self.error = error

    async def aclose(self) -> None:
      self.closed = True
      if self.error is not None:
        raise self.error
      await super().aclose()

  class _Reader(FrameReader):
    closed = False

    async def aclose(self) -> None:
      self.closed = True
      await super().aclose()

  async def run() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    failed, live = _Kernel(ProcessLookupError("container already gone")), _Kernel()
    reader = _Reader()
    work_dir = tmp_path / "session-work"
    work_dir.mkdir()
    session.code_execution_kernels.update({"docker": failed, "kernel": live})
    session.code_execution_frame_reader = reader
    session.code_execution_work_dir = str(work_dir)

    await cleanup_code_execution(session)

    assert failed.closed and live.closed and reader.closed
    assert session.code_execution_kernels == {}
    assert session.code_execution_frame_reader is None
    assert session.code_execution_work_dir is None and not work_dir.exists()

  asyncio.run(run())


def test_cancel_interrupts_one_cell_and_task_cleanup_keeps_namespace(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      assert (await backend.execute("saved = 19", str(tmp_path)))["return_code"] == 0
      pid = backend.runtime.pid
      ready = asyncio.Event()
      handle = await backend.start(
        "import time\nprint('ready', flush=True)\ntime.sleep(60)", str(tmp_path),
        task_id="interrupt", timeout_ms=10_000,
        on_output=lambda stream, text: ready.set() if stream == "stdout" and "ready" in text else None,
      )
      await asyncio.wait_for(ready.wait(), timeout=5)
      assert (await backend.poll(handle))["status"] == "running"
      await backend.cancel(handle)
      result = await backend.collect(handle)
      assert result["return_code"] == 130 and result["cancelled"] and not result["timed_out"]
      assert (await backend.poll(handle))["status"] == "completed"
      await backend.cleanup(str(tmp_path), task_id="interrupt")
      assert (tmp_path / result["spill"]["stdout"]).read_text() == "ready\n"
      following = await backend.execute("print(saved)", str(tmp_path))
      assert following["stdout"] == "19\n" and following["return_code"] == 0
      assert backend.runtime.pid == pid

  asyncio.run(run())


def test_cancel_queued_cell_does_not_interrupt_or_wait_for_running_cell(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      ready = asyncio.Event()
      release_file = tmp_path / "release-running-cell"
      running = await backend.start(
        "import time\nfrom pathlib import Path\nprint('ready', flush=True)\n"
        f"while not Path({str(release_file)!r}).exists():\n    time.sleep(0.01)\n"
        "saved = 29\nprint('finished')",
        str(tmp_path), timeout_ms=10_000,
        on_output=lambda stream, text: ready.set() if stream == "stdout" and "ready" in text else None,
      )
      await asyncio.wait_for(ready.wait(), timeout=5)
      pid = backend.runtime.pid
      queued = await backend.start("saved = -1\nprint('queued ran')", str(tmp_path))
      await asyncio.sleep(0)  # Let the second cell queue at the interpreter lock.
      try:
        await asyncio.wait_for(backend.cancel(queued), timeout=1)
        result = await backend.collect(queued)
        assert result["return_code"] == 130 and result["cancelled"] and not result["timed_out"]
        assert result["stdout"] == ""
        assert (await backend.poll(queued))["status"] == "completed"
        assert (await backend.poll(running))["status"] == "running"
        assert backend.runtime.pid == pid
      finally:
        release_file.touch()
        running_result = await backend.collect(running)
      assert running_result["return_code"] == 0 and not running_result["cancelled"]
      assert running_result["stdout"] == "ready\nfinished\n"
      following = await backend.execute("print(saved)", str(tmp_path))
      assert following["stdout"] == "29\n" and following["return_code"] == 0
      assert backend.runtime.pid == pid

  asyncio.run(run())


def test_timeout_keeps_locals_and_each_cell_gets_a_fresh_budget(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      assert (await backend.execute("saved = 23", str(tmp_path)))["return_code"] == 0
      pid = backend.runtime.pid
      result = await backend.execute("import time\ntime.sleep(60)", str(tmp_path), timeout_ms=300)
      assert result["timed_out"] and result["return_code"] == 124
      following = await backend.execute("print(saved)", str(tmp_path))
      assert following["stdout"] == "23\n" and following["return_code"] == 0
      assert backend.runtime.pid == pid

  asyncio.run(run())


def test_teardown_awaits_process_and_transport_release(tmp_path: Path) -> None:
  async def run() -> None:
    backend = KernelBackend()
    result = await backend.execute("old_value = 31", str(tmp_path))
    assert result["return_code"] == 0
    process = backend.runtime.process
    assert process is not None
    pid = process.pid
    await backend.runtime.aclose()
    await backend.runtime.aclose()
    assert backend.runtime.process is None and backend.runtime.cells == {}
    assert process.returncode is not None
    assert process.stdin is not None and process.stdin.is_closing()
    assert process.stdout is not None and process.stdout.at_eof()
    assert process.stderr is not None and process.stderr.at_eof()
    with pytest.raises(ProcessLookupError):
      os.kill(pid, 0)
    async with _kernel() as replacement:
      following = await replacement.execute("print('old_value' in globals())", str(tmp_path))
      assert following["stdout"] == "False\n"

  asyncio.run(run())


def test_python_native_and_descendant_output_share_ordered_capture(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      chunks: list[tuple[str, str]] = []
      result = await backend.execute(
        "import os, subprocess\nprint('before', flush=True)\nos.write(1, b'native\\n')\n"
        "subprocess.run(['echo', 'child'])\nprint('after')",
        str(tmp_path), on_output=lambda stream, text: chunks.append((stream, text)),
      )
      assert result["return_code"] == 0
      assert result["stdout"] == "before\nnative\nchild\nafter\n"
      assert "".join(text for stream, text in chunks if stream == "stdout") == result["stdout"]

  asyncio.run(run())


def test_explicit_display_preserves_json_and_plain_text_only(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      result = await backend.execute("display({'a': 1})\ndisplay(6 * 7)\n{'private': 'last expression'}", str(tmp_path))
      assert result["return_code"] == 0
      assert len(result["displays"]) == 2
      assert result["displays"][0]["application/json"] == {"a": 1}
      assert result["displays"][1]["text/plain"] == "42"
      assert "private" not in json.dumps(result)

  asyncio.run(run())


def test_stdout_tail_and_spill_keep_partial_and_multibyte_output(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel(CodeExecutionConfig(max_output_bytes=1024)) as backend:
      result = await backend.execute("import sys\nfor i in range(80):\n    print(i, '雪')\nsys.stdout.write('partial')", str(tmp_path))
      expected = "".join(f"{i} 雪\n" for i in range(80)) + "partial"
      assert result["stdout"] == "".join(expected.splitlines(True)[-20:])
      assert result["stdout_tail"] == result["stdout"] and result["truncated"]
      assert (tmp_path / result["spill"]["stdout"]).read_text() == expected

  asyncio.run(run())


def test_display_over_frame_bound_is_not_success_and_kernel_survives(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      assert (await backend.execute("saved = 37", str(tmp_path)))["return_code"] == 0
      pid = backend.runtime.pid
      result = await backend.execute(f"display('x' * {MAX_FRAME_BYTES})", str(tmp_path))
      assert result["return_code"] == 1
      assert result["protocol_error"]["code"] == "protocol_error"
      assert result["displays"] == []
      following = await backend.execute("print(saved)", str(tmp_path))
      assert following["return_code"] == 0 and following["stdout"] == "37\n"
      assert backend.runtime.pid == pid

  asyncio.run(run())


def test_output_aggregate_bound_poison_is_per_cell_not_per_session(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      assert (await backend.execute("saved = 41", str(tmp_path)))["return_code"] == 0
      pid = backend.runtime.pid
      result = await backend.execute("import sys\nfor i in range(70):\n    sys.stdout.write('x' * (1024 * 1024))", str(tmp_path), timeout_ms=30_000)
      assert result["return_code"] == 1 and result["protocol_error"]["code"] == "protocol_error"
      assert (tmp_path / result["spill"]["stdout"]).stat().st_size <= 64 * 1024 * 1024
      following = await backend.execute("print(saved)", str(tmp_path))
      assert following["stdout"] == "41\n" and following["return_code"] == 0
      assert backend.runtime.pid == pid

  asyncio.run(run())


def test_error_and_unbridged_call_do_not_destroy_kernel(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      assert (await backend.execute("saved = 53", str(tmp_path)))["return_code"] == 0
      pid = backend.runtime.pid
      error = await backend.execute("raise ValueError('cell failure')", str(tmp_path))
      assert error["return_code"] == 1 and error["error"]["ename"] == "ValueError"
      bridge_error = await backend.execute("hank.call('example_operation')", str(tmp_path))
      assert bridge_error["return_code"] == 1 and "bridge_unavailable" in bridge_error["stderr"]
      following = await backend.execute("print(saved)", str(tmp_path))
      assert following["stdout"] == "53\n" and backend.runtime.pid == pid

  asyncio.run(run())


def test_real_nested_dispatch_reuses_locals_but_rebinds_cell_grant(tmp_path: Path, _policy: Any) -> None:
  async def run() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    config = CodeExecutionConfig(register_kernel=True, register_docker=False, register_subprocess=False, work_dir_root=str(tmp_path))
    first = build_code_execution(session, config=config)
    mcp = _CaptureMcp(frame_reader=first.frame_reader)
    dispatcher = _bind(first, mcp, extract_sources=_filings_extractor)
    try:
      result, error = await dispatcher.dispatch("cell-first", "code_execute", {
        "host": "kernel", "code": f"filings = hank.call({GET_FILINGS!r}, cik='0000320193')\nprint(len(hank.search('')))",
      })
      assert error is None and result is not None and result["return_code"] == 0
      assert result["stdout"] == "2\n"
      state = session.code_execution_kernels["kernel"]
      pid = state.pid
      second = build_code_execution(session, config=config)
      later_dispatcher = _bind(second, mcp)
      second.bind_dispatcher(
        later_dispatcher,
        tool_definitions=lambda: [row for row in later_dispatcher.get_tool_definitions() if row["name"] != FRED_SERIES],
        code_callable=_CODE_CALLABLE,
      )
      following, error = await later_dispatcher.dispatch("cell-next", "code_execute", {
        "host": "kernel", "code": f"print(filings['filings'][0]['form'])\nprint([row['name'] for row in hank.search('')])\n"
        f"try:\n    hank.call({FRED_SERIES!r}, series_id='CPIAUCSL')\nexcept HankBridgeError as exc:\n    print(exc.code)",
      })
      assert error is None and following is not None and following["return_code"] == 0
      assert following["stdout"].splitlines() == ["10-K", repr([GET_FILINGS]), "not_granted"]
      assert mcp.calls == [(GET_FILINGS, {"cik": "0000320193"})]
      assert second.frame_reader is first.frame_reader
      assert session.code_execution_kernels["kernel"] is state and state.pid == pid
      assert second.frame_reader.active_cells() == ()
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_new_wrapper_refreshes_call_identity_in_the_same_interpreter(tmp_path: Path) -> None:
  state = KernelState()

  async def run() -> None:
    code = "import os\nprint(os.environ.get('AGENT_TELEMETRY_TOOL_CALL_ID'))"
    first = KernelBackend(config=CodeExecutionConfig(prepare_env=lambda env: {
      **env, "AGENT_TELEMETRY_TOOL_CALL_ID": "call-first",
    }), state=state)
    later = KernelBackend(config=CodeExecutionConfig(prepare_env=lambda env: {
      **env, "AGENT_TELEMETRY_TOOL_CALL_ID": "call-next",
    }), state=state)
    try:
      initial = await first.execute(code, str(tmp_path))
      pid = state.pid
      following = await later.execute(code, str(tmp_path))
      assert initial["stdout"] == "call-first\n" and initial["return_code"] == 0
      assert following["stdout"] == "call-next\n" and following["return_code"] == 0
      assert state.pid == pid
      cleared = await later.execute(code, str(tmp_path), env={})
      assert cleared["stdout"] == "None\n" and cleared["return_code"] == 0
    finally:
      await state.aclose()

  asyncio.run(run())


def test_kernel_token_tracks_slot_and_environment_lifetime(tmp_path: Path) -> None:
  from agent_gateway.code_execution._frame_codec import encode_frame
  from agent_gateway.code_execution._frame_reader import CELL_TOKEN_ENV, lookup_active_cell

  async def run() -> None:
    reader = FrameReader()
    async with _kernel() as backend:
      await backend.runtime.lock.acquire()
      cell = reader.open_cell("queued", abort_event=None, active=False)
      handle = await backend.start_bridged(
        f"import os, time; print(os.environ[{CELL_TOKEN_ENV!r}], flush=True); time.sleep(0.1)",
        str(tmp_path), bridge=cell,
      )
      try:
        await asyncio.sleep(0)
        assert lookup_active_cell(cell.token) is None
        denied = await cell.answer_frame(encode_frame({
          "type": "hello", "call_id": "before", "cell_token": cell.token,
        }))
        assert denied["error"]["code"] == "bad_cell_token"
        backend.runtime.lock.release()
        for _ in range(200):
          if handle._backend_data["stdout"].total:
            break
          await asyncio.sleep(0.005)
        assert lookup_active_cell(cell.token) is not None
        result = await backend.collect(handle)
        assert result["stdout"].strip() == cell.token
        assert lookup_active_cell(cell.token) is None
        later = reader.open_cell("later", abort_event=None, active=False)
        following = await backend.execute_bridged(
          f"print(os.environ[{CELL_TOKEN_ENV!r}])", str(tmp_path), bridge=later,
        )
        assert following["stdout"].strip() == later.token
        assert later.token != cell.token
        unbridged = await backend.execute(f"print({CELL_TOKEN_ENV!r} in os.environ)", str(tmp_path))
        assert unbridged["stdout"] == "False\n"
      finally:
        if backend.runtime.lock.locked():
          backend.runtime.lock.release()
        await reader.aclose()

  asyncio.run(run())


@pytest.mark.parametrize("ending", ["error", "timeout", "cancel"])
def test_kernel_abnormal_end_revokes_active_token(tmp_path: Path, ending: str) -> None:
  from agent_gateway.code_execution._frame_reader import lookup_active_cell

  async def run() -> None:
    reader = FrameReader()
    async with _kernel() as backend:
      cell = reader.open_cell("abnormal", abort_event=None, active=False)
      code = "raise ValueError('cell failure')" if ending == "error" else "import time; print('running', flush=True); time.sleep(30)"
      handle = await backend.start_bridged(
        code, str(tmp_path), bridge=cell, timeout_ms=100 if ending == "timeout" else 30_000,
      )
      try:
        if ending == "cancel":
          for _ in range(200):
            if handle._backend_data["stdout"].total:
              break
            await asyncio.sleep(0.005)
          assert lookup_active_cell(cell.token) is not None
          await backend.cancel(handle)
        result = await backend.collect(handle)
        assert result["return_code"] != 0
        assert lookup_active_cell(cell.token) is None
      finally:
        await reader.aclose()

  asyncio.run(run())
