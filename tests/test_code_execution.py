from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, SessionStore, ToolDispatcher
from agent_gateway.code_execution import (
  CodeExecutionConfig,
  SubprocessBackend,
  build_code_execution,
  cleanup_code_execution,
  clear_docker_availability_cache,
  docker_backend_unavailable_reason,
)
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.event_log import EventLog
from agent_gateway.runner import ToolResultContext
from agent_gateway.secret_boundary import REDACTED_SECRET, SecretBoundary
from agent_gateway.task_registry import TASK_NOTIFICATION_INLINE_PAYLOAD_MAX_BYTES, TaskState
from agent_gateway.tool_dispatcher_helpers import ToolResult
from agent_gateway.tool_result_spill import SpillCapabilities, SpillSink, resolve_spill_producer
from tests.host_timing_test_support import harness_wait_seconds
from gateway_test_support.capability_execution_test_support import stub_runner_capability_execution

# Cells that cold-import matplotlib took 12-32s on the loaded 12-worker gate; the 30s
# product default killed them. The cell's compute bound is a hang guard here.
_COLD_MATPLOTLIB_TIMEOUT_MS = 120_000


def _run(coro):
  return asyncio.run(coro)


class _FakeMcp(McpClientManager):
  def __init__(self) -> None:
    super().__init__(config_path=None)

  def is_mcp_tool(self, name: str) -> bool:
    _ = name
    return False

  async def call_tool(
    self,
    name: str,
    tool_input: object,
    meta: object | None = None,
    abort_event: asyncio.Event | None = None,
    gateway_session: object | None = None,
    allow_uncertain_replay: bool = True,
    trusted_dispatch_scope: object | None = None,
  ) -> ToolResult:
    _ = (
      name,
      tool_input,
      meta,
      abort_event,
      gateway_session,
      allow_uncertain_replay,
      trusted_dispatch_scope,
    )
    raise AssertionError("MCP should not execute in code_execution tests")


async def _dispatch_bundle_tool(
  session,
  bundle,
  tool_name: str,
  tool_input: Dict[str, Any],
  *,
  event_log: EventLog | None = None,
  needs_approval=None,
) -> tuple[dict | None, dict | None]:
  dispatcher = ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=_FakeMcp(),
    local_tool_handlers=bundle.handlers,
    role="owner",
    needs_approval=needs_approval or (lambda _name, _tool_input, _qualifier: False),
    event_log=event_log or EventLog(),
    approval_key_qualifier=bundle.approval_qualifier,
  )
  return await dispatcher.dispatch(f"{tool_name}_call", tool_name, tool_input)


def test_subprocess_large_cell_traceback_keeps_exception_line(tmp_path: Path) -> None:
  rows = ",".join(f"'row-{index:05d}'" for index in range(5000))
  code = f"rows = [{rows}]\nrows[10**6]\n"
  assert len(code) > 60_000
  result = _run(SubprocessBackend().execute(code, str(tmp_path)))
  assert result["return_code"] == 1
  assert "IndexError" in result["stderr"]
  assert len(result["stderr"]) < 4096


@pytest.mark.parametrize(
  ("code", "expected"),
  [
    ('s = "page1\x0cpage2"\nlen(s)\n', "11"),
    ('s = "page1\x85page2\u2028page3"\nlen(s)\n', "17"),
    ('def act():\n  return 42\nlabel = "café"; act()\n', "42"),
  ],
)
def test_subprocess_trailing_expression_preserves_source(
  tmp_path: Path, code: str, expected: str,
) -> None:
  result = _run(SubprocessBackend().execute(code, str(tmp_path)))
  assert result["return_code"] == 0, result["stderr"]
  assert result["displays"] == [{"text/plain": expected}]


def test_subprocess_trailing_expression_calls_exactly_once(tmp_path: Path) -> None:
  result = _run(SubprocessBackend().execute(
    "calls = []\ndef act():\n  calls.append(1)\n  return len(calls)\n"
    'label = "éééééééééé"; act()\n',
    str(tmp_path),
  ))
  assert result["return_code"] == 0, result["stderr"]
  assert result["displays"] == [{"text/plain": "1"}]


def test_subprocess_large_trailing_expression_is_bounded(tmp_path: Path) -> None:
  result = _run(SubprocessBackend().execute(
    "saved = {'payload': '雪' * (3 * 1024 * 1024)}\nsaved",
    str(tmp_path),
  ))
  assert result["return_code"] == 0, result["stderr"]
  bundle = result["displays"][0]
  assert len(bundle["text/plain"].encode("utf-8")) <= 16 * 1024
  assert "truncated" in bundle["text/plain"]
  assert bundle["text/plain"].endswith("full value is bound to `_`]")
  assert "application/json" not in bundle


def test_code_execute_basic_execution_and_work_dir_persistence() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

    first_result, first_error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {"code": 'from pathlib import Path\nPath("artifact.txt").write_text("persisted", encoding="utf-8")'},
    )
    second_result, second_error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {"code": 'print(open("artifact.txt").read())'},
    )

    assert first_error is None
    assert first_result is not None
    assert second_error is None
    assert second_result is not None
    assert second_result["stdout"] == "persisted\n"
    assert second_result["stderr"] == ""
    assert second_result["return_code"] == 0
    assert second_result["images"] == []
    assert second_result["timed_out"] is False
    assert second_result["truncated"] is False
    assert session.code_execution_work_dir is not None
    assert list(Path(session.code_execution_work_dir).glob("_code_execute_*.py")) == []

  asyncio.run(_run_test())


def test_code_execute_output_is_redacted_by_the_run_boundary() -> None:
  # The sandbox prints whatever the model told it to print. A run's active
  # credential that the model found on disk or in a variable must not reach the
  # result the session persists, nor the streamed chunk that precedes it.
  async def _run_test() -> None:
    canary = "CUSTOM-ACTIVE-CREDENTIAL-CODE-EXECUTE"
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))
    event_log = EventLog()
    dispatcher = ToolDispatcher(
      capability_bind=TEST_CAPABILITY_BIND,
      mcp_client=_FakeMcp(),
      local_tool_handlers=bundle.handlers,
      role="owner",
      needs_approval=lambda _name, _tool_input, _qualifier: False,
      event_log=event_log,
      approval_key_qualifier=bundle.approval_qualifier,
    )
    dispatcher.bind_secret_boundary(SecretBoundary((canary,)))

    result, error = await dispatcher.dispatch(
      "code_execute_call",
      "code_execute",
      {
        "code": (
          "import sys\n"
          f"print('leaked {canary}')\n"
          f"print('sk-ant-api03-CODEEXEC-CANARY-DO-NOT-USE-7c11', file=sys.stderr)\n"
        )
      },
    )

    assert error is None
    assert result is not None
    assert canary not in result["stdout"]
    assert "leaked" in result["stdout"]
    assert "sk-ant-api03-CODEEXEC-CANARY-DO-NOT-USE-7c11" not in result["stderr"]
    chunks = [
      entry.event for entry in event_log.entries
      if entry.event.get("type") == "tool_output_chunk"
    ]
    assert chunks
    assert all(canary not in str(chunk.get("text") or "") for chunk in chunks)

  asyncio.run(_run_test())


def test_code_execute_redacts_structured_errors_and_display_keys(tmp_path: Path) -> None:
  async def _run_test() -> None:
    canary = "CUSTOM-ACTIVE-CREDENTIAL-STRUCTURED-CODE-EXECUTE"
    credential_file = tmp_path / "credential.txt"
    credential_file.write_text(canary, encoding="utf-8")
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(
      session,
      config=CodeExecutionConfig(register_kernel=True, register_docker=False, register_subprocess=False),
    )
    dispatcher = ToolDispatcher(
      capability_bind=TEST_CAPABILITY_BIND,
      mcp_client=_FakeMcp(),
      local_tool_handlers=bundle.handlers,
      role="owner",
      needs_approval=lambda _name, _tool_input, _qualifier: False,
      approval_key_qualifier=bundle.approval_qualifier,
    )
    dispatcher.bind_secret_boundary(SecretBoundary((canary,)))
    try:
      result, error = await dispatcher.dispatch(
        "structured-output", "code_execute", {
          "host": "kernel",
          "code": (
            "from pathlib import Path\n"
            f"credential = Path({str(credential_file)!r}).read_text()\n"
            "display({credential: {'nested': [credential]}})\n"
            "raise ValueError(credential)"
          ),
        },
      )
      assert error is None and result is not None
      assert result["return_code"] == 1
      assert result["error"]["ename"] == "ValueError"
      assert result["error"]["evalue"] and result["error"]["traceback"]
      assert result["displays"][0]["application/json"]
      assert canary not in json.dumps(result)
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run_test())


def test_code_execute_runs_any_timeout_and_reports_it() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(
      session,
      config=CodeExecutionConfig(register_docker=False, default_timeout_ms=1000, max_timeout_ms=2000),
    )

    result, error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {"code": "print('ok')", "timeout_ms": 1000.0},
    )

    assert error is None
    assert result is not None
    assert result["stdout"] == "ok\n"
    assert result["timeout_ms"] == 1000

    result, error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {"code": "print('ok')", "timeout_ms": 2001},
    )

    assert error is None
    assert result is not None
    assert result["stdout"] == "ok\n"
    assert result["timeout_ms"] == 2001

  asyncio.run(_run_test())


def test_code_execute_rejects_invalid_timeout_ms() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(
      session,
      config=CodeExecutionConfig(register_docker=False, default_timeout_ms=1000, max_timeout_ms=2000),
    )

    cases = [
      ("1000", "timeout_ms must be an integer"),
      (1.9, "timeout_ms must be an integer"),
      (True, "timeout_ms must be an integer"),
      (None, "timeout_ms must be an integer"),
      (999, "timeout_ms must be >= 1000"),
    ]
    for timeout_ms, expected_message in cases:
      result, error = await _dispatch_bundle_tool(
        session,
        bundle,
        "code_execute",
        {"code": "print('should-not-run')", "timeout_ms": timeout_ms},
      )
      assert result is None
      assert error == {"code": "invalid_input", "message": expected_message}

  asyncio.run(_run_test())


def test_code_execute_rejects_invalid_code_input() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

    for code, expected_message in (
      (123, "code must be a string"),
      (True, "code must be a string"),
      (None, "code must be a string"),
      ("", "code is required"),
    ):
      result, error = await _dispatch_bundle_tool(
        session,
        bundle,
        "code_execute",
        {"code": code},
      )
      assert result is None
      assert error == {"code": "invalid_input", "message": expected_message}

  asyncio.run(_run_test())


def test_code_execute_rejects_invalid_host_input() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

    for host in (123, True, None):
      result, error = await _dispatch_bundle_tool(
        session,
        bundle,
        "code_execute",
        {"code": "print('should-not-run')", "host": host},
      )
      assert result is None
      assert error == {"code": "invalid_input", "message": "host must be a string"}

    result, error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {"code": "print('should-not-run')", "host": ""},
    )

    assert result is None
    assert error == {"code": "invalid_input", "message": "Unknown host: ''"}

  asyncio.run(_run_test())


def test_code_execute_invalid_input_returns_validation_error() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(
      session,
      config=CodeExecutionConfig(register_docker=False, default_timeout_ms=1000, max_timeout_ms=2000),
    )

    cases = [
      ({"code": 123}, "code must be a string"),
      ({"code": "print('should-not-run')", "host": None}, "host must be a string"),
      ({"code": "print('should-not-run')", "host": "missing"}, "Unknown host: 'missing'"),
      ({"code": "print('should-not-run')", "background": "true"}, "background must be a boolean"),
      ({"code": "print('should-not-run')", "timeout_ms": "1000"}, "timeout_ms must be an integer"),
    ]
    for tool_input, expected_message in cases:
      result, error = await _dispatch_bundle_tool(
        session,
        bundle,
        "code_execute",
        tool_input,
      )
      assert result is None
      assert error == {"code": "invalid_input", "message": expected_message}

  asyncio.run(_run_test())


def test_code_execute_rejects_invalid_background_flag() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

    for background in ("true", 1, None):
      result, error = await _dispatch_bundle_tool(
        session,
        bundle,
        "code_execute",
        {"code": "print('should-not-run')", "background": background},
      )
      assert result is None
      assert error == {"code": "invalid_input", "message": "background must be a boolean"}

  asyncio.run(_run_test())


def test_code_execute_background_rejects_invalid_code_input() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

    for code, expected_message in (
      (123, "code must be a string"),
      (None, "code must be a string"),
      ("", "code is required"),
    ):
      result, error = await _dispatch_bundle_tool(
        session,
        bundle,
        "code_execute",
        {"code": code, "background": True},
      )
      assert result is None
      assert error == {"code": "invalid_input", "message": expected_message}

  asyncio.run(_run_test())


def test_code_execute_background_rejects_invalid_timeout_ms() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(
      session,
      config=CodeExecutionConfig(register_docker=False, default_timeout_ms=1000, max_timeout_ms=2000),
    )

    for timeout_ms, expected_message in (
      ("1000", "timeout_ms must be an integer"),
      (1.9, "timeout_ms must be an integer"),
      (True, "timeout_ms must be an integer"),
      (999, "timeout_ms must be >= 1000"),
    ):
      result, error = await _dispatch_bundle_tool(
        session,
        bundle,
        "code_execute",
        {"code": "print('should-not-run')", "background": True, "timeout_ms": timeout_ms},
      )
      assert result is None
      assert error == {"code": "invalid_input", "message": expected_message}

  asyncio.run(_run_test())


def test_code_execute_background_accepts_integral_float_timeout_ms() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    runner = _job_runner()
    bundle = build_code_execution(
      session,
      config=CodeExecutionConfig(register_docker=False, default_timeout_ms=1000, max_timeout_ms=2000),
      job_runner=lambda: runner,
    )

    try:
      result, error = await _dispatch_bundle_tool(
        session,
        bundle,
        "code_execute",
        {
          "code": "import time\nprint('started', flush=True)\ntime.sleep(0.2)",
          "background": True,
          "timeout_ms": 1000.0,
        },
      )

      assert error is None
      assert result is not None
      assert result["status"] == "running"
      entry = runner._task_registry.get(result["task_id"])
      assert entry is not None and entry.asyncio_task is not None
      await asyncio.wait_for(entry.asyncio_task, timeout=10)
      assert entry.state == TaskState.COMPLETED
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run_test())


def test_code_execute_preamble_defers_matplotlib_import() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

    result, error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {"code": 'import sys\nprint("matplotlib" in sys.modules)\nprint("matplotlib.pyplot" in sys.modules)'},
    )

    assert error is None
    assert result is not None
    assert result["stdout"] == "False\nFalse\n"
    assert result["stderr"] == ""
    assert result["return_code"] == 0

  asyncio.run(_run_test())


def test_code_execute_captures_images_and_extra_env(tmp_path: Path) -> None:
  async def _run_test() -> None:
    support_dir = tmp_path / "support"
    support_dir.mkdir()
    (support_dir / "helpermod.py").write_text('VALUE = "ok"\n', encoding="utf-8")

    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(
      session,
      config=CodeExecutionConfig(
        register_docker=False,
        default_timeout_ms=_COLD_MATPLOTLIB_TIMEOUT_MS,
        extra_env={"PYTHONPATH": str(support_dir)},
      ),
    )

    result, error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {
        "code": (
          "import helpermod\n"
          "import matplotlib.pyplot as plt\n"
          "print(helpermod.VALUE)\n"
          "plt.plot([1, 2, 3], [3, 1, 4])\n"
          "plt.show()\n"
        )
      },
    )

    assert error is None
    assert result is not None
    assert result["stdout"] == "ok\n"
    assert result["images"]
    assert result["images"][0]["filename"] == "_plot_1.png"
    assert result["images"][0]["media_type"] == "image/png"
    assert result["images"][0]["data_base64"]

  asyncio.run(_run_test())


def test_code_execute_extra_env_replaces_scalars_and_prepends_path_lists(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("RISK_API_URL", "http://inherited.example:5001")
  monkeypatch.setenv("PYTHONPATH", "/inherited/root")

  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(
      session,
      config=CodeExecutionConfig(
        register_docker=False,
        extra_env={
          "RISK_API_URL": "http://127.0.0.1:5262",
          "PYTHONPATH": str(tmp_path),
        },
      ),
    )

    result, error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {
        "code": (
          "import os\n"
          "print(os.environ['RISK_API_URL'])\n"
          "print(os.environ['PYTHONPATH'].split(os.pathsep)[:2])\n"
        )
      },
    )

    assert error is None
    assert result is not None
    lines = result["stdout"].splitlines()
    assert lines[0] == "http://127.0.0.1:5262"
    assert lines[1] == repr([str(tmp_path), "/inherited/root"])

  asyncio.run(_run_test())


def test_code_execute_respects_image_base64_limit() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(
      session,
      config=CodeExecutionConfig(register_docker=False, max_image_base64_bytes=10),
    )

    result, error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {"code": 'from pathlib import Path\nPath("_plot_large.png").write_bytes(b"x" * 64)'},
    )

    assert error is None
    assert result is not None
    assert len(result["images"]) == 1
    assert result["images"][0]["filename"] == "_plot_large.png"
    assert result["images"][0]["skipped"] is True
    assert "exceeds" in result["images"][0]["reason"]

  asyncio.run(_run_test())


def test_code_execute_streaming_emits_chunk_events() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    event_log = EventLog()
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

    result, error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {
        "code": (
          "import time\n"
          "for idx in range(5):\n"
          "    print(f'line-{idx}', flush=True)\n"
          "    time.sleep(0.05)\n"
        )
      },
      event_log=event_log,
    )

    chunk_events = [entry.event for entry in event_log.entries if entry.event.get("type") == "tool_output_chunk"]
    assert error is None
    assert result is not None
    assert len(chunk_events) == 5
    assert [event["seq"] for event in chunk_events] == [1, 2, 3, 4, 5]
    assert all(event["tool_call_id"] == "code_execute_call" for event in chunk_events)
    assert result["stdout"] == "".join(f"line-{idx}\n" for idx in range(5))

  asyncio.run(_run_test())


class _StubProvider:
  name = "stub"


def _job_runner(spill_dir: Path | None = None) -> AgentRunner:
  """A real runner whose task registry owns the session's background cells."""
  event_log = EventLog()
  return AgentRunner(
    event_log=event_log,
    dispatcher=ToolDispatcher(
      capability_bind=TEST_CAPABILITY_BIND,
      mcp_client=_FakeMcp(),
      local_tool_handlers={},
      event_log=event_log,
      session_id="sess-code-jobs",
    ),
    session_id="sess-code-jobs",
    capability_execution=stub_runner_capability_execution(
      provider=_StubProvider(),
      model="stub-model",
      effort="none",
    ),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
    code_execution_spill_dir_provider=(
      SpillSink(
        root_provider=lambda: str(spill_dir),
        capabilities=SpillCapabilities(code_execute=True, spill_read=True),
      )
      if spill_dir is not None
      else None
    ),
  )


def _record_started_handles(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
  handles: list[Any] = []
  start_bridged = SubprocessBackend.start_bridged

  async def _recording_start_bridged(self: Any, *args: Any, **kwargs: Any) -> Any:
    handle = await start_bridged(self, *args, **kwargs)
    handles.append(handle)
    return handle

  monkeypatch.setattr(SubprocessBackend, "start_bridged", _recording_start_bridged)
  return handles


async def _until(predicate: Any) -> None:
  for _ in range(400):
    if predicate():
      return
    await asyncio.sleep(0.01)
  raise AssertionError("condition not reached")


def test_code_execute_background_cell_is_a_code_job_that_completes_and_is_killed(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async def _run_test() -> None:
    handles = _record_started_handles(monkeypatch)
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    runner = _job_runner()
    bundle = build_code_execution(
      session,
      # This test guards the background-job lifecycle (complete, then kill)
      # through subprocess handles; the kernel backend's job lifecycle is
      # the same registry path, but its handles are not subprocess handles.
      config=CodeExecutionConfig(register_docker=False, register_kernel=False),
      job_runner=lambda: runner,
    )

    try:
      start_result, start_error = await _dispatch_bundle_tool(
        session,
        bundle,
        "code_execute",
        {
          "code": (
            "import time\n"
            "print('started', flush=True)\n"
            "time.sleep(0.3)\n"
            "print('finished', flush=True)\n"
          ),
          "background": True,
        },
      )

      assert start_error is None
      assert start_result is not None
      assert start_result["status"] == "running"
      assert start_result["message"].startswith("The cell runs as a background job")
      assert start_result["backend"] == "subprocess"
      task_id = start_result["task_id"]
      entry = runner._task_registry.get(task_id)
      assert entry is not None
      assert entry.task_type == "code"
      assert entry.asyncio_task is not None
      await asyncio.wait_for(entry.asyncio_task, timeout=10)
      assert entry.state == TaskState.COMPLETED
      payload = runner._background_task_payload(entry)
      assert payload["status"] == "completed"
      assert payload["stdout"] == "started\nfinished\n"
      assert payload["backend"] == "subprocess"
      assert bundle.frame_reader._cells == {}

      start_result, start_error = await _dispatch_bundle_tool(
        session,
        bundle,
        "code_execute",
        {"code": "import time\nprint('begin', flush=True)\ntime.sleep(30)\n", "background": True},
      )
      assert start_error is None
      assert start_result is not None
      kill_entry = runner._task_registry.get(start_result["task_id"])
      assert kill_entry is not None and kill_entry.asyncio_task is not None
      await _until(lambda: len(handles) == 2)

      assert runner.cancel_background_task(kill_entry.task_id) is True
      await asyncio.wait_for(kill_entry.asyncio_task, timeout=10)

      assert kill_entry.state == TaskState.KILLED
      assert handles[1]._backend_data["process"].returncode is not None
      assert bundle.frame_reader._cells == {}
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run_test())


def test_code_execute_background_result_is_the_bounded_model_projection(tmp_path: Path) -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    runner = _job_runner(spill_dir=tmp_path)
    bundle = build_code_execution(
      session,
      config=CodeExecutionConfig(
        register_docker=False,
        default_timeout_ms=_COLD_MATPLOTLIB_TIMEOUT_MS,
      ),
      job_runner=lambda: runner,
    )
    results: dict[str, Any] = {}
    try:
      for name, code in (
        ("bulk", "print('x' * 60000)\n"),
        (
          "plot",
          "import matplotlib.pyplot as plt\n"
          "plt.figure(figsize=(1, 1))\n"
          "plt.plot([1, 2], [2, 1])\n"
          "plt.show()\n",
        ),
      ):
        start_result, start_error = await _dispatch_bundle_tool(
          session, bundle, "code_execute", {"code": code, "background": True},
        )
        assert start_error is None and start_result is not None
        entry = runner._task_registry.get(start_result["task_id"])
        assert entry is not None and entry.asyncio_task is not None
        await asyncio.wait_for(
          entry.asyncio_task,
          timeout=harness_wait_seconds(_COLD_MATPLOTLIB_TIMEOUT_MS / 1000 + 30),
        )
        assert entry.state == TaskState.COMPLETED
        serialized = json.dumps(entry.result, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        assert len(serialized) <= TASK_NOTIFICATION_INLINE_PAYLOAD_MAX_BYTES
        assert entry.notification_delivery_state not in {"payload_omitted", "queue_omitted"}
        results[name] = entry.result
    finally:
      await cleanup_code_execution(session)

    bulk = results["bulk"]
    assert bulk["_runner_truncated"] is True
    assert bulk["original_chars"] > 60_000
    spill_sink = runner._spill_dir_provider
    assert spill_sink is not None
    assert resolve_spill_producer(spill_sink, bulk["spill_ref"]) is not None
    plot = results["plot"]
    image = plot["images"][0]
    assert image["data_base64"] == f"[image: {image['filename']}]"
    assert "iVBORw0KGgo" not in json.dumps(plot)

  asyncio.run(_run_test())


def test_code_execute_background_without_a_job_runner_is_unavailable() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    unavailable = {
      "code": "code_execution_unavailable",
      "message": "background cells need the session's job runner",
    }
    try:
      for job_runner in (None, lambda: None):
        bundle = build_code_execution(
          session,
          config=CodeExecutionConfig(register_docker=False),
          job_runner=job_runner,
        )
        result, error = await _dispatch_bundle_tool(
          session,
          bundle,
          "code_execute",
          {"code": "print('never')", "background": True},
        )
        assert result is None
        assert error == unavailable
      assert session.code_execution_frame_reader is not None
      assert session.code_execution_frame_reader._cells == {}
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run_test())


def test_cleanup_code_execution_kills_its_cells_jobs_without_a_live_turn(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async def _run_test() -> None:
    handles = _record_started_handles(monkeypatch)
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    runner = _job_runner()
    bundle = build_code_execution(
      session,
      # Subprocess handles are this cleanup test's kill witness; see the
      # background-lifecycle test above for why the kernel is pinned off.
      config=CodeExecutionConfig(register_docker=False, register_kernel=False),
      job_runner=lambda: runner,
    )

    start_result, start_error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {
        "code": (
          "import time\n"
          "print('still-running', flush=True)\n"
          "time.sleep(30)\n"
        ),
        "background": True,
      },
    )
    assert start_error is None
    assert start_result is not None
    entry = runner._task_registry.get(start_result["task_id"])
    assert entry is not None
    await _until(lambda: len(handles) == 1)

    # A job no cell launched, observed by a foreground cell's `hank.wait`.
    async def _observed_job(_tool_input: Dict[str, Any], **_kwargs: Any):
      await asyncio.sleep(30)
      return {}, None

    observed, observed_error = await runner._register_background_task(
      tool_input={},
      handler=_observed_job,
      task_type="code",
      task_id_override="ce_observed",
      capability_bind_receipt=runner.capability_execution.bind.to_json(),
      on_complete=None,
    )
    assert observed_error is None and observed is not None
    observed_entry = runner._task_registry.get("ce_observed")
    assert observed_entry is not None and observed_entry.asyncio_task is not None
    waiter = asyncio.create_task(_dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {"code": (
        "try:\n"
        "  hank.wait(['ce_observed'], timeout=30)\n"
        "except Exception as exc:\n"
        "  print(type(exc).__name__)\n"
      )},
    ))
    reader = bundle.frame_reader
    await _until(lambda: any(
      cell.job is None and cell._inflight for cell in reader._cells.values()
    ))
    assert [cell.job for cell in reader._cells.values() if cell.job is not None] == [
      (runner, entry.task_id),
    ]

    # Session expiry and autonomous teardown run cleanup with no live turn.
    session.active_turn = None
    work_dir = Path(session.code_execution_work_dir or "")
    at_removal: list[tuple[Any, Any]] = []
    rmtree = shutil.rmtree

    def _recording_rmtree(path: Any, *args: Any, **kwargs: Any) -> None:
      at_removal.append((entry.state, handles[0]._backend_data["process"].returncode))
      rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", _recording_rmtree)
    await cleanup_code_execution(session)
    await cleanup_code_execution(session)

    assert entry.state == TaskState.KILLED
    assert at_removal[0][0] == TaskState.KILLED
    assert at_removal[0][1] is not None
    assert session.code_execution_work_dir is None
    assert not work_dir.exists()
    # The waiting cell's close does not reach the job it observed.
    assert observed_entry.state == TaskState.RUNNING
    await asyncio.wait_for(waiter, timeout=30)
    assert observed_entry.state == TaskState.RUNNING
    assert runner.cancel_background_task("ce_observed") is True
    await asyncio.wait_for(observed_entry.asyncio_task, timeout=10)

  asyncio.run(_run_test())


def test_strip_code_execute_base64_hook_rewrites_model_history() -> None:
  result_entry = {
    "content": json.dumps(
      {
        "stdout": "",
        "stderr": "",
        "return_code": 0,
        "images": [{"filename": "plot.png", "media_type": "image/png", "data_base64": "Zm9v"}],
      }
    )
  }
  ctx = ToolResultContext(
    tool_name="code_execute",
    tool_input={"code": "print(1)"},
    redacted_tool_input={"code": "print(1)"},
    result=None,
    error=None,
    duration_ms=1,
    tool_call_id="tool_1",
    session_id="sess_1",
    server=None,
    result_entry=result_entry,
  )
  session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
  bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

  bundle.sanitize_hook(ctx)

  payload = json.loads(result_entry["content"])
  assert payload["images"][0]["data_base64"] == "[image: plot.png]"




def test_auto_host_with_missing_docker_image_reports_subprocess_and_why(tmp_path, monkeypatch, caplog) -> None:
  # A real `docker` CLI whose image is absent: `image inspect` exits 1 with the
  # daemon's own message. The walk must still reach subprocess, and both the
  # result and the gateway log must say so and why.
  fake_docker = tmp_path / "docker"
  fake_docker.write_text(
    "#!/bin/sh\n"
    "echo \"Error response from daemon: No such image: $3\" >&2\n"
    "exit 1\n"
  )
  fake_docker.chmod(0o755)
  monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")

  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(
      session,
      # The walk must reach subprocess, so the now-default kernel backend
      # (which the walk would select first) is pinned off; Docker's skip
      # reason is what this test guards.
      config=CodeExecutionConfig(
        docker_image="hank-missing-image:latest", register_kernel=False,
      ),
    )
    try:
      with caplog.at_level("WARNING", logger="agent_gateway.code_execution._handlers"):
        result, error = await _dispatch_bundle_tool(
          session,
          bundle,
          "code_execute",
          {"code": "print('ran')"},
        )
      assert error is None
      assert result is not None
      assert result["stdout"] == "ran\n"
      assert result["backend"] == "subprocess"
      assert result["sandboxed"] is False
      assert result["skipped_backends"] == [{
        "backend": "docker",
        "sandboxed": True,
        "reason": (
          "docker image inspect hank-missing-image:latest exited 1: "
          "Error response from daemon: No such image: hank-missing-image:latest"
        ),
      }]
      assert "selected subprocess (sandboxed=False)" in caplog.text
      assert "No such image: hank-missing-image:latest" in caplog.text

      named, error = await _dispatch_bundle_tool(
        session,
        bundle,
        "code_execute",
        {"code": "print('ran')", "host": "subprocess"},
      )
      assert error is None
      assert named is not None
      assert (named["backend"], named["sandboxed"], named["skipped_backends"]) == ("subprocess", False, [])
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run_test())


def test_docker_availability_probe_memoises_per_image_within_ttl(monkeypatch) -> None:
  probed_images: list[str] = []

  def _unavailable_reason(self) -> str | None:
    probed_images.append(self._image)
    return None if self._image == "ready:latest" else f"no image {self._image}"

  monkeypatch.setattr(
    "agent_gateway.code_execution._backends._docker.DockerBackend.unavailable_reason",
    _unavailable_reason,
  )
  clear_docker_availability_cache()
  try:
    ready = CodeExecutionConfig(docker_image="ready:latest")
    missing = CodeExecutionConfig(docker_image="missing:latest")

    assert docker_backend_unavailable_reason(ready) is None
    assert docker_backend_unavailable_reason(ready) is None
    assert probed_images == ["ready:latest"]

    assert docker_backend_unavailable_reason(missing) == "no image missing:latest"
    assert probed_images == ["ready:latest", "missing:latest"]

    clear_docker_availability_cache()
    assert docker_backend_unavailable_reason(missing, ttl_s=0.0) == "no image missing:latest"
    assert docker_backend_unavailable_reason(missing) == "no image missing:latest"
    assert docker_backend_unavailable_reason(missing) == "no image missing:latest"
    assert probed_images == ["ready:latest", "missing:latest", "missing:latest", "missing:latest"]
  finally:
    clear_docker_availability_cache()
