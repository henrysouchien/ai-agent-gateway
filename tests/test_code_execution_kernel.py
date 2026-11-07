"""Session-owned kernel behavior against real, isolated Python interpreters."""
import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import pytest


from agent_gateway import SessionStore
from agent_gateway.code_execution import CodeExecutionConfig, build_code_execution, cleanup_code_execution
from agent_gateway.code_execution._frame_reader import FrameReader
from agent_gateway.code_execution._backends._kernel import KernelBackend, KernelState
from agent_gateway.code_execution._frame_codec import MAX_FRAME_BYTES, decode_frame
from agent_gateway.code_execution._kernel_runner import _implicit_mime_bundle
from agent_gateway.event_log import EventLog
from agent_gateway.session import SessionStream
from agent_gateway.tool_dispatcher_helpers import ToolResult
from agent_gateway.tool_policy_registry import PreparedToolCall
from test_code_execution_bridge import (
  FRED_SERIES,
  GET_FILINGS,
  _CODE_CALLABLE,
  _FRED_RESULT,
  _CaptureMcp,
  _bind,
  _filings_extractor,
  _job_runner,
  _policy,
)


@contextlib.asynccontextmanager
async def _kernel(config: CodeExecutionConfig | None = None) -> AsyncIterator[KernelBackend]:
  backend = KernelBackend(config=config or CodeExecutionConfig())
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


def test_a_release_landing_while_the_queued_cell_gives_up_never_runs_it_unlocked(tmp_path: Path) -> None:
  """A holder that releases exactly while the waiter's cancelled acquire settles.

  The waiter never owned the kernel, so it is answered kernel_busy and its code
  never reaches the interpreter, even though the holder field is already clear.
  """

  async def run() -> None:
    async with _kernel() as backend:
      state = backend.runtime

      class ReleasingOnCancelLock(asyncio.Lock):
        async def acquire(self) -> Literal[True]:
          try:
            return await super().acquire()
          except asyncio.CancelledError:
            if self.locked():  # the holder finishes in the waiter's cancellation window
              state.holder = None
              self.release()
            raise

      state.lock = ReleasingOnCancelLock()
      await state.lock.acquire()
      state.holder = "ce_holder"
      marker = tmp_path / "waiter-ran"
      queued = await backend.start(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
        str(tmp_path), task_id="ce_waiter", timeout_ms=200,
      )
      result = await asyncio.wait_for(backend.collect(queued), timeout=5)
      assert (result.get("error") or {}).get("code") == "kernel_busy", result
      assert result["error"]["held_by"] == "ce_holder"
      assert result["cancelled"] is True and result["compute_ms"] == 0
      assert not marker.exists() and state.runner is None  # the interpreter never saw the cell
      assert state.holder is None and not state.lock.locked()

  asyncio.run(run())


def test_cell_queued_behind_held_kernel_is_answered_kernel_busy_in_its_own_bound(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      ready = asyncio.Event()
      release_file = tmp_path / "release-holder"
      holder = await backend.start(
        "import time\nfrom pathlib import Path\nprint('ready', flush=True)\n"
        f"while not Path({str(release_file)!r}).exists():\n    time.sleep(0.01)\n"
        "saved = 31\nprint('finished')",
        str(tmp_path), task_id="ce_holder", timeout_ms=30_000,
        on_output=lambda stream, text: ready.set() if stream == "stdout" and "ready" in text else None,
      )
      await asyncio.wait_for(ready.wait(), timeout=5)
      pid = backend.runtime.pid
      try:
        started = asyncio.get_running_loop().time()
        queued = await backend.start("print('queued ran')", str(tmp_path), task_id="ce_waiter", timeout_ms=1000)
        result = await asyncio.wait_for(backend.collect(queued), timeout=5)
        elapsed = asyncio.get_running_loop().time() - started
        assert 0.9 <= elapsed < 3
        assert result["cancelled"] is True and result["timed_out"] is False
        assert result["error"]["code"] == "kernel_busy" and result["error"]["held_by"] == "ce_holder"
        assert "ce_holder" in result["error"]["message"]
        assert result["host_wait_ms"] >= 900 and result["compute_ms"] == 0 and result["duration_ms"] >= 900
        assert result["stdout"] == ""
        assert (await backend.poll(holder))["status"] == "running"
        assert backend.runtime.pid == pid
      finally:
        release_file.touch()
        holder_result = await backend.collect(holder)
      assert holder_result["return_code"] == 0 and not holder_result["cancelled"]
      assert holder_result["stdout"] == "ready\nfinished\n"
      following = await backend.execute("print(saved)", str(tmp_path))
      assert following["stdout"] == "31\n" and "error" not in following
      assert backend.runtime.holder is None and not backend.runtime.lock.locked()

  asyncio.run(run())


def test_cell_queued_behind_kernel_that_frees_within_its_bound_runs_normally(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      ready = asyncio.Event()
      holder = await backend.start(
        "import time\nprint('ready', flush=True)\ntime.sleep(0.3)\nsaved = 37",
        str(tmp_path), task_id="ce_holder", timeout_ms=30_000,
        on_output=lambda stream, text: ready.set() if stream == "stdout" and "ready" in text else None,
      )
      await asyncio.wait_for(ready.wait(), timeout=5)
      queued = await backend.start("print(saved)", str(tmp_path), task_id="ce_waiter", timeout_ms=10_000)
      result = await asyncio.wait_for(backend.collect(queued), timeout=15)
      assert result["return_code"] == 0 and result["stdout"] == "37\n"
      assert not result["cancelled"] and "error" not in result
      assert (await backend.collect(holder))["return_code"] == 0
      assert backend.runtime.holder is None

  asyncio.run(run())


def test_timeout_keeps_locals_and_each_cell_gets_a_fresh_budget(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      assert (await backend.execute("saved = 23", str(tmp_path)))["return_code"] == 0
      pid = backend.runtime.pid
      result = await backend.execute("import time\ntime.sleep(60)", str(tmp_path), timeout_ms=300)
      assert result["timed_out"] and result["return_code"] == 124
      assert "[cell interrupted after 300 ms timeout; kernel state kept]" in result["stderr"]
      assert 290 <= result["compute_ms"] < 5_000 and result["host_wait_ms"] == 0
      following = await backend.execute("print(saved)", str(tmp_path))
      assert following["stdout"] == "23\n" and following["return_code"] == 0
      assert backend.runtime.pid == pid

  asyncio.run(run())


def _slow_boot(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
  """The runner's interpreter is still booting for `seconds` after launch."""
  from agent_gateway.code_execution._backends import _kernel

  runner_source = _kernel._runner_source
  monkeypatch.setattr(_kernel, "_runner_source", lambda: f"import time\ntime.sleep({seconds})\n" + runner_source())


def test_boot_longer_than_timeout_does_not_time_the_first_cell_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  _slow_boot(monkeypatch, 2.5)

  async def run() -> None:
    async with _kernel() as backend:
      result = await backend.execute("print('first')", str(tmp_path), timeout_ms=1000)
      assert result["return_code"] == 0 and result["stdout"] == "first\n", result
      assert result["timed_out"] is False and "error" not in result
      assert result["compute_ms"] < 1000

  asyncio.run(run())


def test_cell_exceeding_timeout_after_a_slow_boot_still_times_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  _slow_boot(monkeypatch, 1.5)

  async def run() -> None:
    async with _kernel() as backend:
      result = await backend.execute("import time\ntime.sleep(60)", str(tmp_path), timeout_ms=1000)
      assert result["timed_out"] is True and result["return_code"] == 124, result
      assert result.get("error", {}).get("code") != "kernel_boot_failed"
      assert "[cell interrupted after 1000 ms timeout; kernel state kept]" in result["stderr"]
      assert 990 <= result["compute_ms"] < 5_000

  asyncio.run(run())


def test_boot_that_never_starts_fails_typed_after_the_boot_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  _slow_boot(monkeypatch, 3600)

  async def run() -> None:
    config = CodeExecutionConfig(kernel_boot_timeout_ms=500)
    async with _kernel(config) as backend:
      begun = asyncio.get_running_loop().time()
      result = await asyncio.wait_for(backend.execute("print('never')", str(tmp_path), timeout_ms=1000), timeout=30)
      elapsed = asyncio.get_running_loop().time() - begun
      assert 0.4 <= elapsed < 10
      process = backend.runtime.process
      assert result["error"]["code"] == "kernel_boot_failed", result
      assert result["timed_out"] is False and result["return_code"] not in (0, 124)
      assert result["compute_ms"] == 0 and result["stdout"] == ""
      assert "[kernel lost: interpreter did not start the cell within 0.5 s" in result["stderr"]
      assert process is None and backend.runtime.runner is None

  asyncio.run(run())




def test_host_wait_longer_than_timeout_does_not_interrupt_the_cell(tmp_path: Path, _policy: Any) -> None:
  async def run() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    config = CodeExecutionConfig(work_dir_root=str(tmp_path))
    bundle = build_code_execution(session, config=config)
    mcp = _CaptureMcp(delay_s=1.6, frame_reader=bundle.frame_reader)
    dispatcher = _bind(bundle, mcp)
    try:
      # A warm interpreter keeps cold boot out of the 1000 ms compute window.
      warm, error = await dispatcher.dispatch("cell-warm", "code_execute", {"code": "pass"})
      assert error is None and warm is not None and warm["return_code"] == 0, error
      result, error = await dispatcher.dispatch("cell-wait", "code_execute", {
        "timeout_ms": 1000, "host_wait_ms": 2000,
        "code": f"saved = 61\nfilings = hank.call({GET_FILINGS!r}, cik='0000320193')\nprint(filings['filings'][0]['form'])",
      })
      assert error is None and result is not None, error
      assert result["timed_out"] is False and result["return_code"] == 0, result["stderr"]
      assert result["stdout"] == "10-K\n"
      assert result["host_wait_ms"] >= 1600
      assert result["compute_ms"] < 1000
      assert result["requested_host_wait_ms"] == 2000
      state = session.code_execution_kernels["kernel"]
      pid = state.pid
      following, error = await dispatcher.dispatch("cell-after", "code_execute", {"code": "print(saved)"})
      assert error is None and following is not None and following["stdout"] == "61\n"
      assert following["host_wait_ms"] == 0 and "requested_host_wait_ms" not in following
      assert state.pid == pid
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_kernel_cell_bridge_operation_is_answered(tmp_path: Path, _policy: Any) -> None:
  # A registered bridge op (here `wait`) reaches the bridge from a kernel cell
  # and its reply comes back; the reader must not drop it as a stale frame.
  async def run() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    runner = _job_runner()
    session.active_turn = SessionStream(event_log=EventLog(), runner_task=None, runtime_owner=runner)
    config = CodeExecutionConfig(work_dir_root=str(tmp_path))
    bundle = build_code_execution(session, config=config, job_runner=lambda: runner)
    dispatcher = _bind(bundle, _CaptureMcp(frame_reader=bundle.frame_reader))
    try:
      result, error = await asyncio.wait_for(dispatcher.dispatch("cell-wait-op", "code_execute", {
        "code": "import json\nprint(json.dumps(hank.wait(['bg_9'], timeout=0)))",
      }), timeout=30)
      assert error is None and result is not None, error
      assert result["return_code"] == 0, result["stderr"]
      assert json.loads(result["stdout"])["unknown"] == ["bg_9"]
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_spawned_thread_host_call_does_not_pause_the_compute_deadline(tmp_path: Path, _policy: Any) -> None:
  async def run() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    config = CodeExecutionConfig(work_dir_root=str(tmp_path))
    bundle = build_code_execution(session, config=config)
    mcp = _CaptureMcp(delay_s=2.0, frame_reader=bundle.frame_reader)
    dispatcher = _bind(bundle, mcp)
    thread_call = (
      "import threading, time\nbox = {}\n"
      f"worker = threading.Thread(target=lambda: box.update(filings=hank.call({GET_FILINGS!r}, cik='0000320193')))\n"
      "worker.start()\n"
    )
    try:
      # A warm interpreter keeps cold boot out of the 1000 ms compute window.
      warm, error = await dispatcher.dispatch("cell-warm", "code_execute", {"code": "pass"})
      assert error is None and warm is not None and warm["return_code"] == 0, error
      result, error = await dispatcher.dispatch("cell-spin", "code_execute", {
        "timeout_ms": 1000,
        "code": thread_call + "end = time.monotonic() + 1.5\nwhile time.monotonic() < end:\n    pass\nworker.join()\nprint('finished')",
      })
      assert error is None and result is not None, error
      assert result["timed_out"] is True and result["return_code"] == 124, result
      assert "finished" not in result["stdout"]
      assert result["compute_ms"] >= 1000
      assert result["host_wait_ms"] == 0
      assert mcp.calls == [(GET_FILINGS, {"cik": "0000320193"})]
      # Reply routing for a spawned thread is unchanged; its wait is compute.
      following, error = await dispatcher.dispatch("cell-join", "code_execute", {
        "timeout_ms": 30_000,
        "code": thread_call + "worker.join()\nprint(box['filings']['filings'][0]['form'])",
      })
      assert error is None and following is not None, error
      assert following["timed_out"] is False and following["return_code"] == 0, following["stderr"]
      assert following["stdout"] == "10-K\n"
      assert following["host_wait_ms"] == 0
      assert following["compute_ms"] >= 2000
    finally:
      await cleanup_code_execution(session)

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


def test_explicit_display_and_trailing_expression_bundles(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      result = await backend.execute("display({'a': 1})\ndisplay(6 * 7)\n{'private': 'last expression'}", str(tmp_path))
      assert result["return_code"] == 0
      # The two explicit displays keep their bundle shapes, and the trailing
      # expression now returns its own bundle (the REPL contract that
      # replaced the dropped last expression).
      assert len(result["displays"]) == 3
      assert result["displays"][0]["application/json"] == {"a": 1}
      assert result["displays"][1]["text/plain"] == "42"
      assert result["displays"][2] == {"text/plain": "{'private': 'last expression'}"}
      assert "application/json" not in result["displays"][2]
      # A trailing display(...) call returns None, so it adds no bundle.
      none_result = await backend.execute("display(1)\nprint('side')", str(tmp_path))
      assert none_result["return_code"] == 0
      assert len(none_result["displays"]) == 1
      assert none_result["stdout"] == "side\n"

  asyncio.run(run())


def test_implicit_result_json_projection_stays_within_text_budget() -> None:
  within = _implicit_mime_bundle({"rows": ["x" * 100] * 100})
  assert "application/json" not in within
  assert within["text/plain"].startswith("{'rows':")
  over = _implicit_mime_bundle({"rows": ["x" * 100] * 200})
  assert len(over["text/plain"].encode("utf-8")) <= 16 * 1024
  assert "truncated" in over["text/plain"]
  assert "application/json" not in over


def test_implicit_result_renderer_errors_stay_in_presentation() -> None:
  class BrokenRepr:
    def __repr__(self) -> str:
      raise RuntimeError("repr failed")

  class BrokenJson(dict):
    def items(self):
      raise RuntimeError("JSON projection failed")

  assert _implicit_mime_bundle(BrokenRepr()) == {
    "text/plain": "<implicit repr unavailable: RuntimeError>",
  }
  assert _implicit_mime_bundle(BrokenJson(answer=42)) == {
    "text/plain": "{'answer': 42}",
  }


def test_large_trailing_expression_is_bounded_and_value_stays_in_kernel(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      size = 3 * 1024 * 1024
      result = await backend.execute(
        f"saved = {{'payload': '雪' * {size}}}\nsaved", str(tmp_path),
      )
      assert result["return_code"] == 0, result
      bundle = result["displays"][0]
      assert len(bundle["text/plain"].encode("utf-8")) <= 16 * 1024
      assert "truncated" in bundle["text/plain"]
      assert bundle["text/plain"].endswith("full value is bound to `_`]")
      assert "application/json" not in bundle
      following = await backend.execute("len(saved['payload'])", str(tmp_path))
      assert following["return_code"] == 0
      assert following["displays"] == [{"text/plain": str(size)}]
      # An unassigned trailing expression is reachable as `_` in the next cell.
      unassigned = await backend.execute(f"{{'payload': 'x' * {size}}}", str(tmp_path))
      assert unassigned["return_code"] == 0
      assert "truncated" in unassigned["displays"][0]["text/plain"]
      recovered = await backend.execute("len(_['payload'])", str(tmp_path))
      assert recovered["displays"] == [{"text/plain": str(size)}]

  asyncio.run(run())


def test_stdout_tail_and_spill_keep_partial_and_multibyte_output(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel(CodeExecutionConfig(max_output_bytes=1024)) as backend:
      result = await backend.execute("import sys\nfor i in range(80):\n    print(i, '雪')\nsys.stdout.write('partial')", str(tmp_path))
      expected = "".join(f"{i} 雪\n" for i in range(80)) + "partial"
      # New ordinary-output contract: `stdout` is the full text up to
      # `max_output_bytes` (here the whole output fits), `stdout_tail`
      # keeps its last-20-lines meaning, and the spill file holds the
      # complete text either way.
      assert result["stdout"] == expected
      assert result["stdout_tail"] == "".join(expected.splitlines(True)[-20:])
      assert not result["truncated"]
      assert (tmp_path / result["spill"]["stdout"]).read_text() == expected

  asyncio.run(run())


def test_display_over_frame_bound_is_not_success_and_kernel_survives(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      assert (await backend.execute("saved = 37", str(tmp_path)))["return_code"] == 0
      pid = backend.runtime.pid
      result = await backend.execute(f"display('x' * {MAX_FRAME_BYTES})", str(tmp_path))
      assert result["return_code"] == 1
      assert "2 MiB" in result["error"]["evalue"]
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


def test_top_level_await_cell_runs_and_its_coroutines_persist_across_cells(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      first = await backend.execute(
        "import asyncio\n"
        "async def double(value):\n"
        "    await asyncio.sleep(0)\n"
        "    return value * 2\n"
        "total = await double(21)\n"
        "print(total)",
        str(tmp_path),
      )
      assert first["return_code"] == 0, first["stderr"]
      assert first["stdout"] == "42\n"
      pid = backend.runtime.pid
      second = await backend.execute("print(total, await double(4))\nawait double(5)", str(tmp_path))
      assert second["return_code"] == 0, second["stderr"]
      assert second["stdout"] == "42 8\n"
      assert second["displays"] == [{"text/plain": "10"}]
      failed = await backend.execute("await double('x')\nawait asyncio.sleep(0)\nraise_later = undefined_name", str(tmp_path))
      assert failed["return_code"] == 1 and failed["error"]["ename"] == "NameError"
      following = await backend.execute("print(total, await double(1))", str(tmp_path))
      assert following["stdout"] == "42 2\n" and following["return_code"] == 0
      assert backend.runtime.pid == pid

  asyncio.run(run())


def test_create_task_in_body_with_await_only_in_the_trailing_expression_runs_on_the_loop(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      gathered = await backend.execute(
        "import asyncio\n"
        "async def work(n):\n"
        "    await asyncio.sleep(0)\n"
        "    return n * 2\n"
        "tasks = [asyncio.create_task(work(n)) for n in range(3)]\n"
        "await asyncio.gather(*tasks)",
        str(tmp_path),
      )
      assert gathered["return_code"] == 0, gathered["stderr"]
      assert [bundle["text/plain"] for bundle in gathered["displays"]] == ["[0, 2, 4]"]
      awaited = await backend.execute(
        "single = asyncio.create_task(work(21))\nawait single", str(tmp_path),
      )
      assert awaited["return_code"] == 0, awaited["stderr"]
      assert [bundle["text/plain"] for bundle in awaited["displays"]] == ["42"]
      # A cell without top-level await still runs with no loop driving it.
      plain = await backend.execute(
        "try:\n    asyncio.get_running_loop()\nexcept RuntimeError:\n    print('no loop')\n", str(tmp_path),
      )
      assert plain["return_code"] == 0 and plain["stdout"] == "no loop\n"

  asyncio.run(run())


def test_task_that_swallows_cancellation_cannot_hang_the_cell_or_lose_the_namespace(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      assert (await backend.execute("pass", str(tmp_path)))["return_code"] == 0
      pid = backend.runtime.pid
      # The task ignores cancellation for 8 s (and for a bounded iteration
      # count, so it ends by itself either way). Cell cleanup runs with SIGINT
      # ignored, so an unbounded wait for it outlives the 3 s timeout and its
      # grace, and the interpreter is lost.
      stubborn = await backend.execute(
        "import asyncio, time\n"
        "saved = 83\n"
        "async def stubborn():\n"
        "    end = time.monotonic() + 8\n"
        "    for _ in range(5_000_000):\n"
        "        if time.monotonic() > end:\n"
        "            return\n"
        "        try:\n"
        "            await asyncio.sleep(0)\n"
        "        except asyncio.CancelledError:\n"
        "            pass\n"
        "stubborn_task = asyncio.create_task(stubborn())\n"
        "await asyncio.sleep(0)",
        str(tmp_path), timeout_ms=3_000,
      )
      assert stubborn["return_code"] == 0 and not stubborn["timed_out"], stubborn["stderr"]
      following = await backend.execute(
        "import asyncio\nawait asyncio.sleep(0)\nprint(saved, stubborn_task.done())", str(tmp_path),
      )
      assert following["return_code"] == 0, following["stderr"]
      assert following["stdout"] == "83 False\n"
      assert backend.runtime.pid == pid

  asyncio.run(run())


def test_acall_wait_pauses_the_compute_deadline(tmp_path: Path, _policy: Any) -> None:
  async def run() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    config = CodeExecutionConfig(work_dir_root=str(tmp_path))
    bundle = build_code_execution(session, config=config)
    mcp = _CaptureMcp(delay_s=1.4, frame_reader=bundle.frame_reader)
    dispatcher = _bind(bundle, mcp)
    try:
      # A warm interpreter keeps cold boot out of the 1000 ms compute window.
      warm, error = await dispatcher.dispatch("cell-warm", "code_execute", {"code": "pass"})
      assert error is None and warm is not None and warm["return_code"] == 0, error
      result, error = await dispatcher.dispatch("cell-acall", "code_execute", {
        "timeout_ms": 1000,
        "code": (
          f"filings = await hank.acall({GET_FILINGS!r}, cik='0000320193')\n"
          "print(filings['filings'][0]['form'])"
        ),
      })
      assert error is None and result is not None, error
      assert result["timed_out"] is False and result["return_code"] == 0, result["stderr"]
      assert result["stdout"] == "10-K\n"
      assert result["host_wait_ms"] >= 1200
      assert result["compute_ms"] < 1000
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_acall_gather_runs_concurrent_nested_calls_and_plain_hank_ops_stay_sync(tmp_path: Path, _policy: Any) -> None:
  async def run() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    runner = _job_runner()
    session.active_turn = SessionStream(event_log=EventLog(), runner_task=None, runtime_owner=runner)
    config = CodeExecutionConfig(work_dir_root=str(tmp_path))
    bundle = build_code_execution(session, config=config, job_runner=lambda: runner)
    mcp = _CaptureMcp(frame_reader=bundle.frame_reader)
    dispatcher = _bind(bundle, mcp)
    arrived: list[str] = []
    all_in = asyncio.Event()

    async def gated(
      name: str,
      tool_input: dict[str, Any] | PreparedToolCall,
      meta: object | None = None,
      abort_event: asyncio.Event | None = None,
      gateway_session: object | None = None,
      allow_uncertain_replay: bool = True,
      trusted_dispatch_scope: object | None = None,
    ) -> ToolResult:
      _ = (name, meta, abort_event, gateway_session, allow_uncertain_replay, trusted_dispatch_scope)
      assert isinstance(tool_input, dict)
      # All three calls must be in flight together before any is answered; a
      # serialized bridge never reaches the third and this wait times out.
      arrived.append(tool_input["series_id"])
      if len(arrived) == 3:
        all_in.set()
      await asyncio.wait_for(all_in.wait(), timeout=30)
      return {**_FRED_RESULT, "series_id": tool_input["series_id"]}, None

    mcp.call_tool = gated
    try:
      result, error = await asyncio.wait_for(dispatcher.dispatch("cell-gather", "code_execute", {
        "code": (
          "import asyncio\n"
          f"results = await asyncio.gather(*(hank.acall({FRED_SERIES!r}, series_id=name) for name in ('A', 'B', 'C')))\n"
          "print([row['series_id'] for row in results])\n"
          "waited = await hank.acall('wait', ['bg_9'], timeout=0)\n"
          "print(waited['unknown'])"
        ),
      }), timeout=60)
      assert error is None and result is not None, error
      assert result["return_code"] == 0, result["stderr"]
      assert result["stdout"] == "['A', 'B', 'C']\n['bg_9']\n"
      assert sorted(arrived) == ["A", "B", "C"]
      assert [entry["operation"] for entry in result["nested_calls"]].count(FRED_SERIES) == 3
      # An operation called without acall stays synchronous and returns a dict (or dict carrier when source_refs are present).
      del mcp.call_tool
      sync, error = await asyncio.wait_for(dispatcher.dispatch("cell-sync", "code_execute", {
        "code": (
          "import inspect\n"
          "waited = hank.wait(['bg_9'], timeout=0)\n"
          f"row = getattr(hank, {FRED_SERIES!r})(series_id='CPIAUCSL')\n"
          "print(type(waited) is dict, inspect.isawaitable(waited), waited['unknown'])\n"
          "print(isinstance(row, dict), inspect.isawaitable(row), row['series_id'])"
        ),
      }), timeout=60)
      assert error is None and sync is not None, error
      assert sync["return_code"] == 0, sync["stderr"]
      assert sync["stdout"] == "True False ['bg_9']\nTrue False CPIAUCSL\n"
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_real_nested_dispatch_reuses_locals_but_rebinds_cell_grant(tmp_path: Path, _policy: Any) -> None:
  async def run() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    config = CodeExecutionConfig(work_dir_root=str(tmp_path))
    first = build_code_execution(session, config=config)
    mcp = _CaptureMcp(frame_reader=first.frame_reader)
    dispatcher = _bind(first, mcp, extract_sources=_filings_extractor)
    try:
      result, error = await dispatcher.dispatch("cell-first", "code_execute", {
        "code": f"filings = hank.call({GET_FILINGS!r}, cik='0000320193')\nprint(len(hank.search('')))",
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
        "code": f"print(filings['filings'][0]['form'])\nprint([row['name'] for row in hank.search('')])\n"
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


def test_kernel_reuses_one_hank_bridge_instance_across_cells_without_hello(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      first = await backend.execute(
        "hank_id = id(hank)\nprint(callable(hank.carry) and callable(hank.input))", str(tmp_path),
      )
      second = await backend.execute(
        "print(id(hank) == hank_id)\nprint(callable(hank.carry) and callable(hank.input))", str(tmp_path),
      )
      assert first["return_code"] == 0, first["stderr"]
      assert second["return_code"] == 0, second["stderr"]
      assert first["stdout"] == "True\n"
      assert second["stdout"].splitlines() == ["True", "True"]
      assert backend.runtime.user_ns_locals == {"hank_id": "int"}

  asyncio.run(run())


_RISK_STYLE_PREAMBLE = (
  "from abc import ABCMeta\n"
  "from typing import Protocol\n\n"
  "class DomainBridge(Protocol):\n"
  "  def call(self, operation: str) -> object: ...\n\n"
  "class DomainRouteError(LookupError):\n"
  "  pass\n\n"
  "class HankDomainTransport:\n"
  "  def __init__(self, get_bridge):\n"
  "    self._get_bridge = get_bridge\n\n"
  "_risk = HankDomainTransport(lambda: hank)\n"
)


def test_cell_traceback_line_numbers_are_one_indexed_with_a_long_preamble(tmp_path: Path) -> None:
  async def run() -> None:
    config = CodeExecutionConfig(preamble_suffix=_RISK_STYLE_PREAMBLE + "configured = 1\n" * 40)
    async with _kernel(config) as backend:
      result = await backend.execute("x = 1\ny = 2\nraise ValueError('line three')", str(tmp_path))
      assert result["return_code"] == 1
      assert result["error"]["ename"] == "ValueError"
      assert 'File "<cell>", line 3, in <module>' in result["stderr"]
      assert 'File "<cell>", line 1,' not in result["stderr"]
      assert any('File "<cell>", line 3, in <module>' in line for line in result["error"]["traceback"])
      assert result["stderr"].count('File "<cell>"') == 1

  asyncio.run(run())


def test_import_hank_works_and_yields_the_injected_bridge(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      first = await backend.execute(
        "import sys\n_injected = hank\nimport hank\nprint(hank is _injected)\nprint(sys.modules['hank'] is _injected)",
        str(tmp_path),
      )
      assert first["return_code"] == 0, first["stderr"]
      assert first["stdout"] == "True\nTrue\n"
      second = await backend.execute("import hank as other\nprint(other is hank)", str(tmp_path))
      assert second["return_code"] == 0, second["stderr"]
      assert second["stdout"] == "True\n"

  asyncio.run(run())


def test_risk_style_preamble_binds_clients_without_leaking_helper_names(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel(CodeExecutionConfig(preamble_suffix=_RISK_STYLE_PREAMBLE)) as backend:
      result = await backend.execute(
        "leaked = [n for n in ('ABCMeta', 'Protocol', 'DomainBridge', 'DomainRouteError', 'HankDomainTransport')"
        " if n in globals()]\n"
        "print(leaked)\n"
        "print(type(_risk).__name__)\n"
        "print(_risk._get_bridge() is hank)",
        str(tmp_path),
      )
      assert result["return_code"] == 0, result["stderr"]
      assert result["stdout"] == "[]\nHankDomainTransport\nTrue\n"

  asyncio.run(run())


def test_percent_load_executes_helper_into_cell_namespace_and_keeps_line_numbers(tmp_path: Path) -> None:
  helper = tmp_path / "helper_module.py"
  helper.write_text("def double(value):\n  return value * 2\n\nloaded = double(21)\n", encoding="utf-8")

  async def run() -> None:
    async with _kernel() as backend:
      result = await backend.execute(
        f"%load {helper}\nprint(loaded, double(2))\nraise ValueError('after load')", str(tmp_path),
      )
      assert result["return_code"] == 1
      assert result["stdout"] == "42 4\n"
      assert result["error"]["ename"] == "ValueError"
      assert 'File "<cell>", line 3, in <module>' in result["stderr"]
      assert result["stderr"].count('File "<cell>"') == 1
      # The helper's names persist in the session namespace for later cells.
      following = await backend.execute("print(double(loaded))", str(tmp_path))
      assert following["stdout"] == "84\n"

  asyncio.run(run())


async def _dispatch_bridged_cells(tmp_path: Path, cells: list[str]) -> list[dict[str, Any]]:
  session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
  runner = _job_runner()
  session.active_turn = SessionStream(event_log=EventLog(), runner_task=None, runtime_owner=runner)
  config = CodeExecutionConfig(work_dir_root=str(tmp_path))
  bundle = build_code_execution(session, config=config, job_runner=lambda: runner)
  dispatcher = _bind(bundle, _CaptureMcp(frame_reader=bundle.frame_reader))
  results: list[dict[str, Any]] = []
  try:
    for index, code in enumerate(cells):
      result, error = await asyncio.wait_for(
        dispatcher.dispatch(f"cell-footer-{index}", "code_execute", {"code": code}), timeout=30,
      )
      assert error is None and result is not None, error
      assert result["return_code"] == 0, result["stderr"]
      results.append(result)
  finally:
    await cleanup_code_execution(session)
  return results


def test_silent_assignment_after_a_hank_call_reports_what_it_bound(tmp_path: Path, _policy: Any) -> None:
  [result] = asyncio.run(_dispatch_bridged_cells(tmp_path, ["st = hank.wait(['bg_9'], timeout=0)"]))
  assert result["stdout"].startswith("[bound in kernel: st = dict[")
  assert result["stdout"].endswith("]\n")
  assert result["stdout"].count("\n") == 1


def test_silent_assignment_without_a_hank_call_prints_nothing(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      result = await backend.execute("st = {'a': 1, 'b': 2}", str(tmp_path))
      assert result["return_code"] == 0, result["stderr"]
      assert result["stdout"] == ""
      assert backend.runtime.user_ns_locals["st"].startswith("dict[2]")

  asyncio.run(run())


def test_cell_that_prints_gets_no_bound_footer(tmp_path: Path, _policy: Any) -> None:
  [result] = asyncio.run(_dispatch_bridged_cells(
    tmp_path, ["st = hank.wait(['bg_9'], timeout=0)\nprint('done')"],
  ))
  assert result["stdout"] == "done\n"


def test_hank_call_cell_that_displays_or_rebinds_nothing_gets_no_bound_footer(tmp_path: Path, _policy: Any) -> None:
  first, displayed, unchanged = asyncio.run(_dispatch_bridged_cells(tmp_path, [
    "st = hank.wait(['bg_9'], timeout=0)",
    "other = hank.wait(['bg_9'], timeout=0)\ndisplay(1)",
    "hank.wait(['bg_9'], timeout=0)",
  ]))
  assert first["stdout"].startswith("[bound in kernel: st = dict[")
  assert displayed["stdout"] == ""
  assert len(displayed["displays"]) == 1 and displayed["displays"][0]["text/plain"] == "1"
  assert unchanged["stdout"] == ""


def test_kernel_locals_summary_survives_hostile_values_and_clears_deleted_names(tmp_path: Path) -> None:
  async def run() -> None:
    async with _kernel() as backend:
      setup = (
        "import threading\n"
        "hank_id = id(hank)\n"
        "class Bad:\n"
        "    @property\n"
        "    def shape(self):\n"
        "        raise ValueError('shape exploded')\n"
        "bad = Bad()\n"
        "big = {'k' * 100_000: 1}\n"
        "keep = 41\n"
        "_stop = threading.Event()\n"
        "def _churn():\n"
        "    i = 0\n"
        "    while not _stop.is_set():\n"
        "        globals()['churn_' + str(i % 50)] = i\n"
        "        globals().pop('churn_' + str((i + 25) % 50), None)\n"
        "        i += 1\n"
        "    for name in [n for n in list(globals()) if n.startswith('churn_')]:\n"
        "        globals().pop(name, None)\n"
        "_thread = threading.Thread(target=_churn, daemon=True)\n"
        "_thread.start()\n"
      )
      first = await backend.execute(setup, str(tmp_path))
      assert first["return_code"] == 0, first["stderr"]
      pid = backend.runtime.pid
      assert pid is not None
      second = await backend.execute("print(keep)", str(tmp_path))
      assert second["return_code"] == 0, second["stderr"]
      assert second["stdout"] == "41\n"
      assert backend.runtime.pid == pid
      assert backend.runtime.user_ns_locals["keep"] == "int"
      assert backend.runtime.user_ns_locals["bad"] == "Bad"
      assert all(len(name) <= 48 and len(summary) <= 256 for name, summary in backend.runtime.user_ns_locals.items())
      cleared = await backend.execute("_stop.set()\n_thread.join()\ndel hank_id, big, bad, keep", str(tmp_path))
      assert cleared["return_code"] == 0, cleared["stderr"]
      assert backend.runtime.pid == pid
      assert backend.runtime.user_ns_locals == {}

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
        denied = decode_frame(await cell.handle_frame(encode_frame({
          "type": "catalog", "request": "search", "query": "", "call_id": "before", "cell_token": cell.token,
        })))
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
