import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import SessionStore, ToolDispatcher
from agent_gateway.code_execution import (
  CodeExecutionConfig,
  build_code_execution,
  cleanup_code_execution,
  clear_docker_availability_cache,
  docker_backend_unavailable_reason,
)
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.event_log import EventLog
from agent_gateway.runner import ToolResultContext
from agent_gateway.secret_boundary import REDACTED_SECRET, SecretBoundary
from agent_gateway.tool_dispatcher_helpers import ToolResult


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
    mcp_client=_FakeMcp(),
    local_tool_handlers=bundle.handlers,
    role="owner",
    needs_approval=needs_approval or (lambda _name, _tool_input, _qualifier: False),
    event_log=event_log or EventLog(),
    approval_key_qualifier=bundle.approval_qualifier,
  )
  return await dispatcher.dispatch(f"{tool_name}_call", tool_name, tool_input)


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


def test_code_execute_redacts_deep_admitted_subprocess_sidecar(tmp_path: Path) -> None:
  async def _run_test() -> None:
    canary = "CUSTOM-ACTIVE-DEEP-CREDENTIAL"
    credential_file = tmp_path / "credential.txt"
    credential_file.write_text(canary, encoding="utf-8")
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(
      session,
      config=CodeExecutionConfig(register_kernel=False, register_docker=False),
    )
    dispatcher = ToolDispatcher(
      mcp_client=_FakeMcp(),
      local_tool_handlers=bundle.handlers,
      role="owner",
      needs_approval=lambda _name, _tool_input, _qualifier: False,
      approval_key_qualifier=bundle.approval_qualifier,
    )
    dispatcher.bind_secret_boundary(SecretBoundary((canary,)))
    try:
      result, error = await dispatcher.dispatch(
        "deep-sidecar", "code_execute", {
          "host": "subprocess",
          "code": (
            "import json, os\n"
            "from pathlib import Path\n"
            f"credential = Path({str(credential_file)!r}).read_text()\n"
            "leaf = json.dumps({credential: [credential, {'nested': credential}], 'count': 7})\n"
            "nested = leaf\n"
            "for index in range(1200):\n"
            "    nested = '{\"n\":' + nested + '}'\n"
            "    if index % 24 == 0:\n"
            "        nested = '[' + nested + ']'\n"
            # String assembly keeps the child JSON encoder's recursion limit
            # out of the admitted sidecar/parent projection regression.
            "payload = '{\"schema_version\":1,\"function\":\"deep\",\"tool_version\":\"v1\","
            "\"output_sha256\":\"hash\",\"params\":' + nested + ',\"summary\":' + leaf + '}'\n"
            "target = Path(os.environ['AGENT_CODE_EXECUTE_WORK_DIR']) / '.agent_provenance'"
            " / os.environ['AGENT_TELEMETRY_TOOL_CALL_ID']\n"
            "target.mkdir(parents=True, exist_ok=True)\n"
            "(target / 'deep.json').write_text(payload, encoding='utf-8')\n"
            "print(len(payload.encode('utf-8')))\n"
          ),
        },
      )
      assert error is None and result is not None
      assert result["return_code"] == 0
      assert result["stderr"] == ""
      assert 7200 < int(result["stdout"].strip()) < 8192
      assert "computations_dropped" not in result
      assert len(result["computations"]) == 1
      computation = result["computations"][0]
      assert computation["function"] == "deep"
      assert computation["output_sha256"] == "hash"
      value = computation["params"]
      for index in reversed(range(1200)):
        if index % 24 == 0:
          assert isinstance(value, list) and len(value) == 1
          value = value[0]
        assert isinstance(value, dict) and list(value) == ["n"]
        value = value["n"]
      expected_leaf = {
        REDACTED_SECRET: [REDACTED_SECRET, {"nested": REDACTED_SECRET}],
        "count": 7,
      }
      assert value == expected_leaf
      assert computation["summary"] == expected_leaf
      pending = [result]
      while pending:
        value = pending.pop()
        if isinstance(value, str):
          assert canary not in value
        elif isinstance(value, dict):
          pending.extend(value.keys())
          pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
          pending.extend(value)
      assert session.code_execution_work_dir is not None
      sidecar_dir = Path(session.code_execution_work_dir) / ".agent_provenance" / "deep-sidecar"
      assert not sidecar_dir.exists()
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
    bundle = build_code_execution(
      session,
      config=CodeExecutionConfig(register_docker=False, default_timeout_ms=1000, max_timeout_ms=2000),
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
      config=CodeExecutionConfig(register_docker=False, extra_env={"PYTHONPATH": str(support_dir)}),
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


def test_code_execute_background_status_and_cancel_flow() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

    try:
      start_result, start_error = await _dispatch_bundle_tool(
        session,
        bundle,
        "code_execute",
        {
          "code": (
            "import time\n"
            "print('started', flush=True)\n"
            "time.sleep(0.5)\n"
            "print('finished', flush=True)\n"
          ),
          "background": True,
        },
      )

      assert start_error is None
      assert start_result is not None
      assert start_result["status"] == "running"
      task_id = start_result["task_id"]

      running_result = None
      for _ in range(10):
        await asyncio.sleep(0.05)
        running_result, running_error = await _dispatch_bundle_tool(
          session,
          bundle,
          "code_execute_status",
          {"task_id": task_id},
        )
        assert running_error is None
        if running_result and "started" in running_result.get("stdout_tail", ""):
          break
      assert running_result is not None
      assert running_result["status"] == "running"
      assert "started" in running_result["stdout_tail"]

      await asyncio.sleep(0.8)

      complete_result, complete_error = await _dispatch_bundle_tool(
        session,
        bundle,
        "code_execute_status",
        {"task_id": task_id},
      )
      assert complete_error is None
      assert complete_result is not None
      assert complete_result["stdout"] == "started\nfinished\n"
      assert task_id not in session.background_tasks

      start_result, start_error = await _dispatch_bundle_tool(
        session,
        bundle,
        "code_execute",
        {
          "code": (
            "import time\n"
            "print('begin', flush=True)\n"
            "time.sleep(30)\n"
          ),
          "background": True,
        },
      )
      assert start_error is None
      assert start_result is not None
      cancel_task_id = start_result["task_id"]
      task = session.background_tasks[cancel_task_id]

      cancel_result, cancel_error = await _dispatch_bundle_tool(
        session,
        bundle,
        "code_execute_status",
        {"task_id": cancel_task_id, "cancel": True},
      )

      assert cancel_error is None
      assert cancel_result == {"status": "cancelled", "task_id": cancel_task_id}
      assert cancel_task_id not in session.background_tasks
      assert task.handle._backend_data["process"].returncode is not None
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run_test())


def test_code_execute_status_rejects_invalid_cancel_flag_before_task_lookup() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

    for cancel in ("true", 1, None):
      result, error = await _dispatch_bundle_tool(
        session,
        bundle,
        "code_execute_status",
        {"task_id": "missing-task", "cancel": cancel},
      )
      assert result is None
      assert error == {"code": "invalid_input", "message": "cancel must be a boolean"}

  asyncio.run(_run_test())


def test_code_execute_status_rejects_invalid_task_id_type() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

    for task_id in (123, True, None):
      result, error = await _dispatch_bundle_tool(
        session,
        bundle,
        "code_execute_status",
        {"task_id": task_id},
      )
      assert result is None
      assert error == {"code": "invalid_input", "message": "task_id must be a string"}

    result, error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute_status",
      {"task_id": ""},
    )

    assert result is None
    assert error == {"code": "invalid_input", "message": "task_id is required"}

  asyncio.run(_run_test())


def test_cleanup_code_execution_cancels_tasks_and_is_idempotent() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

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
    task = session.background_tasks[start_result["task_id"]]
    work_dir = Path(session.code_execution_work_dir or "")

    await cleanup_code_execution(session)
    await cleanup_code_execution(session)

    assert session.background_tasks == {}
    assert task._terminated is True
    assert task.handle._backend_data["process"].returncode is not None
    assert session.code_execution_work_dir is None
    assert not work_dir.exists()

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
      config=CodeExecutionConfig(docker_image="hank-missing-image:latest"),
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
