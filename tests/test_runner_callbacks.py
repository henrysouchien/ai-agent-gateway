import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, EventLog, ToolDispatcher, ToolResultContext  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.runner_callbacks import (  # noqa: E402
  call_before_stream_complete_hook,
  call_metric_hook,
  call_tool_result_hook,
  call_tool_timing_hook,
)
from agent_gateway.runner_hooks_lifecycle import RunnerHooksLifecycleMixin  # noqa: E402


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


class _StubProvider:
  name = "stub"


class _Logger:
  def __init__(self) -> None:
    self.warnings: list[tuple[Any, ...]] = []

  def warning(self, *args: Any) -> None:
    self.warnings.append(args)


def _run(coro):
  return asyncio.run(coro)


def _make_dispatcher(event_log: EventLog | None = None) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    event_log=event_log or EventLog(),
    session_id="sess-callback",
  )


def test_runner_hooks_lifecycle_methods_are_inherited_from_mixin() -> None:
  assert issubclass(AgentRunner, RunnerHooksLifecycleMixin)
  assert gateway_runner.RunnerHooksLifecycleMixin is RunnerHooksLifecycleMixin

  for method_name in (
    "_call_on_tool_result",
    "_call_on_before_stream_complete",
    "_call_on_tool_timing",
    "_call_metric",
    "_call_credential_refresher",
    "_apply_refreshed_auth_config",
    "_usage_has_tokens",
    "_usage_delta",
    "_build_usage_event",
    "_call_on_usage",
    "_call_on_late_usage_event",
    "_call_on_session_summary",
    "_estimate_usage_cost",
  ):
    assert getattr(AgentRunner, method_name) is getattr(RunnerHooksLifecycleMixin, method_name)


def test_runner_callback_wrappers_resolve_parent_module_helpers(monkeypatch: Any) -> None:
  runner = object.__new__(AgentRunner)
  runner._on_metric = object()
  runner._sid = "sess"
  metric_calls: list[dict[str, Any]] = []

  def _call_metric_hook(callback: Any, *, name: str, value: int, log_session_id: str, logger: Any) -> None:
    metric_calls.append(
      {
        "callback": callback,
        "name": name,
        "value": value,
        "log_session_id": log_session_id,
        "logger": logger,
      }
    )

  monkeypatch.setattr(gateway_runner, "_call_metric_hook", _call_metric_hook)

  AgentRunner._call_metric(runner, "gateway.test", 3)

  assert metric_calls == [
    {
      "callback": runner._on_metric,
      "name": "gateway.test",
      "value": 3,
      "log_session_id": "sess",
      "logger": gateway_runner.log,
    }
  ]


def test_tool_timing_helper_passes_optional_user_id_when_supported() -> None:
  calls: list[dict[str, Any]] = []

  def callback(session_id, tool_name, server, duration_ms, is_error, result_bytes, *, user_id=None):
    calls.append(
      {
        "session_id": session_id,
        "tool_name": tool_name,
        "server": server,
        "duration_ms": duration_ms,
        "is_error": is_error,
        "result_bytes": result_bytes,
        "user_id": user_id,
      }
    )

  call_tool_timing_hook(
    callback,
    accepts_user_id=True,
    accepts_context_surfaces=False,
    session_id="sess-full",
    log_session_id="sess",
    user_id="alice",
    context_surfaces=None,
    tool_name="lookup",
    server="research",
    duration_ms=12,
    is_error=False,
    result_bytes=34,
    logger=_Logger(),
  )

  assert calls == [
    {
      "session_id": "sess-full",
      "tool_name": "lookup",
      "server": "research",
      "duration_ms": 12,
      "is_error": False,
      "result_bytes": 34,
      "user_id": "alice",
    }
  ]


def test_tool_timing_helper_supports_legacy_signature_and_swallows_errors() -> None:
  legacy_calls: list[tuple[Any, ...]] = []
  logger = _Logger()

  def legacy_callback(session_id, tool_name, server, duration_ms, is_error, result_bytes):
    legacy_calls.append((session_id, tool_name, server, duration_ms, is_error, result_bytes))

  call_tool_timing_hook(
    legacy_callback,
    accepts_user_id=False,
    accepts_context_surfaces=False,
    session_id="sess-full",
    log_session_id="sess",
    user_id="alice",
    context_surfaces=None,
    tool_name="legacy",
    server=None,
    duration_ms=5,
    is_error=True,
    result_bytes=6,
    logger=logger,
  )

  def failing_callback(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("offline")

  call_tool_timing_hook(
    failing_callback,
    accepts_user_id=True,
    accepts_context_surfaces=False,
    session_id="sess-full",
    log_session_id="sess",
    user_id="alice",
    context_surfaces=None,
    tool_name="failing",
    server=None,
    duration_ms=1,
    is_error=False,
    result_bytes=2,
    logger=logger,
  )

  assert legacy_calls == [("sess-full", "legacy", None, 5, True, 6)]
  assert logger.warnings
  assert "on_tool_timing hook failed" in logger.warnings[0][0]


def test_tool_timing_helper_passes_context_surfaces_when_supported() -> None:
  calls: list[dict[str, Any]] = []
  surface = {"surface_id": "tool:file_read", "content_hash": "sha256:def"}

  def callback(session_id, tool_name, server, duration_ms, is_error, result_bytes, *, context_surfaces=None):
    calls.append({
      "session_id": session_id,
      "tool_name": tool_name,
      "context_surfaces": context_surfaces,
    })

  call_tool_timing_hook(
    callback,
    accepts_user_id=False,
    accepts_context_surfaces=True,
    session_id="sess-full",
    log_session_id="sess",
    user_id="alice",
    context_surfaces=[surface],
    tool_name="lookup",
    server="research",
    duration_ms=12,
    is_error=False,
    result_bytes=34,
    logger=_Logger(),
  )

  assert calls == [{
    "session_id": "sess-full",
    "tool_name": "lookup",
    "context_surfaces": [surface],
  }]


def test_metric_helper_records_and_swallows_errors() -> None:
  calls: list[tuple[str, int]] = []
  logger = _Logger()

  call_metric_hook(
    lambda name, value: calls.append((name, value)),
    name="gateway.test",
    value=3,
    log_session_id="sess",
    logger=logger,
  )

  def failing_metric(_name: str, _value: int) -> None:
    raise RuntimeError("offline")

  call_metric_hook(failing_metric, name="gateway.fail", value=1, log_session_id="sess", logger=logger)

  assert calls == [("gateway.test", 3)]
  assert logger.warnings
  assert "metric hook failed" in logger.warnings[0][0]


def test_tool_result_helper_filters_blocks_and_swallows_errors() -> None:
  logger = _Logger()
  ctx = object()

  async def callback(active_ctx: object):
    assert active_ctx is ctx
    return [{"type": "text", "text": "ok"}, "ignored", {"type": "json", "value": 1}]

  assert _run(call_tool_result_hook(callback, ctx, log_session_id="sess", logger=logger)) == [
    {"type": "text", "text": "ok"},
    {"type": "json", "value": 1},
  ]
  assert _run(call_tool_result_hook(lambda _ctx: None, ctx, log_session_id="sess", logger=logger)) == []

  def failing_callback(_ctx: object) -> None:
    raise RuntimeError("offline")

  assert _run(call_tool_result_hook(failing_callback, ctx, log_session_id="sess", logger=logger)) == []
  assert logger.warnings
  assert "on_tool_result hook failed" in logger.warnings[0][0]


def test_before_stream_complete_helper_respects_legacy_and_terminal_signatures() -> None:
  event_log = EventLog()
  logger = _Logger()
  calls: list[tuple[str, dict[str, Any] | None]] = []

  def legacy_callback(active_log: EventLog) -> None:
    assert active_log is event_log
    calls.append(("legacy", None))

  async def terminal_callback(active_log: EventLog, terminal_event: dict[str, Any] | None) -> None:
    assert active_log is event_log
    calls.append(("terminal", terminal_event))

  _run(
    call_before_stream_complete_hook(
      legacy_callback,
      event_log,
      {"type": "error", "error": "failed"},
      log_session_id="sess",
      logger=logger,
    )
  )
  _run(
    call_before_stream_complete_hook(
      legacy_callback,
      event_log,
      {"type": "stream_complete"},
      log_session_id="sess",
      logger=logger,
    )
  )
  _run(
    call_before_stream_complete_hook(
      terminal_callback,
      event_log,
      {"type": "error", "error": "failed"},
      log_session_id="sess",
      logger=logger,
    )
  )

  assert calls == [
    ("legacy", None),
    ("terminal", {"type": "error", "error": "failed"}),
  ]
  assert logger.warnings == []


def test_before_stream_complete_helper_swallows_errors() -> None:
  logger = _Logger()

  def failing_callback(_event_log: EventLog, _terminal_event: dict[str, Any] | None) -> None:
    raise RuntimeError("offline")

  _run(
    call_before_stream_complete_hook(
      failing_callback,
      EventLog(),
      {"type": "stream_complete"},
      log_session_id="sess",
      logger=logger,
    )
  )

  assert logger.warnings
  assert "on_before_stream_complete hook failed" in logger.warnings[0][0]


def test_runner_callback_delegates_preserve_session_and_user_id() -> None:
  timing_calls: list[dict[str, Any]] = []
  metric_calls: list[tuple[str, int]] = []

  def timing_callback(session_id, tool_name, server, duration_ms, is_error, result_bytes, *, user_id=None):
    timing_calls.append(
      {
        "session_id": session_id,
        "tool_name": tool_name,
        "server": server,
        "duration_ms": duration_ms,
        "is_error": is_error,
        "result_bytes": result_bytes,
        "user_id": user_id,
      }
    )

  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    provider=_StubProvider(),
    auth_config={"api_key": "k", "model": "stub-model"},
    on_tool_timing=timing_callback,
    on_metric=lambda name, value: metric_calls.append((name, value)),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  runner._call_on_tool_timing(
    tool_name="lookup",
    server="research",
    duration_ms=12,
    is_error=False,
    result_bytes=34,
  )
  runner._call_metric("gateway.test", 2)

  assert timing_calls == [
    {
      "session_id": "sess-parent",
      "tool_name": "lookup",
      "server": "research",
      "duration_ms": 12,
      "is_error": False,
      "result_bytes": 34,
      "user_id": "alice",
    }
  ]
  assert metric_calls == [("gateway.test", 2)]


def test_runner_hook_delegates_preserve_log_and_filter_results() -> None:
  event_log = EventLog()
  tool_result_calls: list[ToolResultContext] = []
  before_complete_calls: list[dict[str, Any] | None] = []

  async def on_tool_result(ctx: ToolResultContext):
    tool_result_calls.append(ctx)
    return [{"type": "text", "text": "ok"}, "ignored"]

  async def on_before_stream_complete(active_log: EventLog, terminal_event: dict[str, Any] | None):
    assert active_log is event_log
    before_complete_calls.append(terminal_event)

  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-parent",
    provider=_StubProvider(),
    auth_config={"api_key": "k", "model": "stub-model"},
    on_tool_result=on_tool_result,
    on_before_stream_complete=on_before_stream_complete,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  ctx = ToolResultContext(
    tool_name="lookup",
    tool_input={"query": "AAPL"},
    result={"ok": True},
    error=None,
    duration_ms=12,
    tool_call_id="tool-1",
    session_id="sess-parent",
    server=None,
    result_entry=None,
  )

  assert _run(runner._call_on_tool_result(ctx)) == [{"type": "text", "text": "ok"}]
  _run(runner._call_on_before_stream_complete({"type": "error", "error": "failed"}))

  assert tool_result_calls == [ctx]
  assert before_complete_calls == [{"type": "error", "error": "failed"}]


def test_runner_before_stream_complete_delegate_allows_missing_log_without_hook() -> None:
  runner = object.__new__(AgentRunner)
  runner._on_before_stream_complete = None

  _run(AgentRunner._call_on_before_stream_complete(runner, {"type": "error", "error": "failed"}))
