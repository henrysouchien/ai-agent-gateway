import asyncio
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, CostEstimate, EventLog, ModelInfo, ToolDispatcher  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.runner import StreamTurnResult  # noqa: E402
from agent_gateway.runner_run_loop import RunnerRunLoopMixin  # noqa: E402
from agent_gateway.runner_state import ToolUseLoopResult  # noqa: E402


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


class _NoCredentialProvider:
  name = "patched-provider"

  def __init__(self) -> None:
    self.seen_config: dict[str, Any] | None = None

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    self.seen_config = dict(config)
    return False


def _make_no_credential_runner(
  provider: _NoCredentialProvider,
  *,
  allow_stub_response: bool = True,
) -> AgentRunner:
  event_log = EventLog()
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    event_log=event_log,
    session_id="sess_run_loop",
  )
  return AgentRunner(
    event_log=event_log,
    dispatcher=dispatcher,
    session_id="sess_run_loop",
    provider=provider,
    auth_config={"api_key": "k"},
    allow_stub_response=allow_stub_response,
    get_tool_definitions=lambda: [],
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )


class _CredentialProvider:
  name = "stub"

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    _ = config
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name)

  def estimate_cost(
    self,
    model: str,
    uncached: int,
    output: int,
    *,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
  ) -> CostEstimate:
    _ = model, uncached, output, cache_read_tokens, cache_creation_tokens
    return CostEstimate()


class _ClientCreationFailureProvider(_CredentialProvider):
  name = "broken-provider"

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    raise RuntimeError("sensitive client construction detail")


def _make_credential_runner(
  provider: _CredentialProvider | None = None,
  *,
  allow_stub_response: bool = True,
) -> AgentRunner:
  event_log = EventLog()
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    event_log=event_log,
    session_id="sess_run_loop",
  )
  return AgentRunner(
    event_log=event_log,
    dispatcher=dispatcher,
    session_id="sess_run_loop",
    provider=provider or _CredentialProvider(),
    auth_config={"api_key": "k", "model": "stub-model"},
    allow_stub_response=allow_stub_response,
    get_tool_definitions=lambda: [],
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )


def test_runner_run_loop_method_is_inherited_from_mixin() -> None:
  assert issubclass(AgentRunner, RunnerRunLoopMixin)
  assert gateway_runner.RunnerRunLoopMixin is RunnerRunLoopMixin
  assert AgentRunner.run is RunnerRunLoopMixin.run


def test_runner_still_reexports_run_loop_constants() -> None:
  assert gateway_runner._MAX_NOTIFICATIONS_PER_TURN == 5
  assert gateway_runner._MAX_TOKENS_CONTINUATIONS == 3
  assert "tool-first response" in gateway_runner._MAX_TOKENS_NUDGE


def test_run_loop_resolves_compatibility_aliases_from_runner_module(monkeypatch) -> None:
  calls: dict[str, Any] = {}

  def fake_default_model(provider_name: str | None) -> str:
    calls["provider_name"] = provider_name
    return "parent-default-model"

  def fake_normalized_run_config(
    auth_config: dict[str, Any],
    *,
    default_model: str,
    model_override: str | None,
  ) -> dict[str, Any]:
    calls["auth_config"] = dict(auth_config)
    calls["default_model"] = default_model
    calls["model_override"] = model_override
    return {"model": "parent-normalized-model", "thinking": False}

  provider = _NoCredentialProvider()
  runner = _make_no_credential_runner(provider)

  async def fake_stub_response(messages: list[dict[str, Any]]) -> None:
    calls["stub_messages"] = list(messages)

  monkeypatch.setattr(gateway_runner, "_get_default_model_for_provider", fake_default_model)
  monkeypatch.setattr(gateway_runner, "_normalized_run_config", fake_normalized_run_config)
  monkeypatch.setattr(runner, "_emit_stub_response", fake_stub_response)

  asyncio.run(
    runner.run(
      messages=[{"role": "user", "content": "hello"}],
      model_override="request-model",
    )
  )

  assert calls["provider_name"] == "patched-provider"
  assert calls["auth_config"] == {"api_key": "k"}
  assert calls["default_model"] == "parent-default-model"
  assert calls["model_override"] == "request-model"
  assert provider.seen_config == {"model": "parent-normalized-model", "thinking": False}
  assert calls["stub_messages"] == [{"role": "user", "content": "hello"}]


def test_run_loop_missing_credential_fails_closed_when_stub_disabled(monkeypatch) -> None:
  runner = _make_no_credential_runner(
    _NoCredentialProvider(),
    allow_stub_response=False,
  )

  async def fail_stub(_messages: list[dict[str, Any]]) -> None:
    raise AssertionError("stub response must not run")

  monkeypatch.setattr(runner, "_emit_stub_response", fail_stub)

  asyncio.run(runner.run(messages=[{"role": "user", "content": "private prompt"}]))

  events = [entry.event for entry in runner._log.entries]
  assert [event for event in events if event.get("type") == "error"] == [{
    "type": "error",
    "error": "Provider startup failed: no active credential configured for provider=patched-provider.",
  }]
  assert not any(event.get("type") in {"text_delta", "stream_complete"} for event in events)
  assert "private prompt" not in str(events)


def test_run_loop_client_creation_fails_closed_when_stub_disabled(monkeypatch) -> None:
  runner = _make_credential_runner(
    _ClientCreationFailureProvider(),
    allow_stub_response=False,
  )

  async def fail_stub(_messages: list[dict[str, Any]]) -> None:
    raise AssertionError("stub response must not run")

  monkeypatch.setattr(runner, "_emit_stub_response", fail_stub)

  asyncio.run(runner.run(messages=[{"role": "user", "content": "private prompt"}]))

  events = [entry.event for entry in runner._log.entries]
  assert [event for event in events if event.get("type") == "error"] == [{
    "type": "error",
    "error": "Provider startup failed: could not create client for provider=broken-provider.",
  }]
  assert not any(event.get("type") in {"text_delta", "stream_complete"} for event in events)
  assert "private prompt" not in str(events)
  assert "sensitive client construction detail" not in str(events)


def test_context_manifest_persists_before_stream_and_uses_both_regular_paths() -> None:
  class Capture:
    def __init__(self) -> None:
      self.calls: list[dict[str, Any]] = []

    def persist(self, **kwargs: Any) -> str:
      self.calls.append({**kwargs, "thread": threading.get_ident()})
      order.append("persist")
      return "sha256:prompt"

  async def case() -> None:
    runner = _make_credential_runner()
    capture = Capture()
    runner._context_capture = capture
    runner._context_surfaces_static = [{"surface_id": "tool:x", "content_hash": "sha256:x"}]
    durable: list[dict[str, Any]] = []

    async def append_durable(event: dict[str, Any]) -> None:
      order.append("durable")
      durable.append(dict(event))

    async def stream_turn(**kwargs: Any):
      order.append("stream")
      assert kwargs["system_prompt"] == [("first", True), ("second", False)]
      return object(), StreamTurnResult(
        full_text="done",
        stop_reason="end_turn",
        content_blocks=[{"type": "text", "text": "done"}],
      )

    runner._append_durable_event = append_durable  # type: ignore[method-assign]
    runner._stream_turn = stream_turn  # type: ignore[method-assign]
    main_thread = threading.get_ident()
    await runner.run(
      messages=[{"role": "user", "content": "hello"}],
      system_prompt=[("first", True), ("second", False)],
    )
    manifests = [entry.event for entry in runner._log.entries if entry.event.get("type") == "context_manifest"]
    assert order[:3] == ["persist", "durable", "stream"]
    assert capture.calls[0]["rendered_system_prompt"] == [("first", True), ("second", False)]
    assert capture.calls[0]["thread"] != main_thread
    assert manifests == [event for event in durable if event.get("type") == "context_manifest"]
    assert manifests[0]["turn"] == 1

  order: list[str] = []
  asyncio.run(case())


def test_context_capture_failure_suppresses_manifest_and_turn_continues() -> None:
  class Capture:
    def persist(self, **_kwargs: Any) -> str:
      raise RuntimeError("unresolved")

  async def case() -> None:
    runner = _make_credential_runner()
    runner._context_capture = Capture()
    streamed = False

    async def stream_turn(**_kwargs: Any):
      nonlocal streamed
      streamed = True
      return object(), StreamTurnResult(
        full_text="done",
        stop_reason="end_turn",
        content_blocks=[{"type": "text", "text": "done"}],
      )

    runner._stream_turn = stream_turn  # type: ignore[method-assign]
    await runner.run(messages=[{"role": "user", "content": "hello"}], system_prompt="system")
    assert streamed
    assert not any(entry.event.get("type") == "context_manifest" for entry in runner._log.entries)

  asyncio.run(case())


def test_run_loop_stops_after_tool_results_when_runner_requests_it(monkeypatch) -> None:
  async def _case() -> None:
    runner = _make_credential_runner()
    stream_calls = {"count": 0}

    async def fake_stream_turn(**kwargs: Any):
      _ = kwargs
      stream_calls["count"] += 1
      if stream_calls["count"] == 1:
        result = StreamTurnResult(
          full_text="",
          stop_reason="tool_use",
          content_blocks=[
            {
              "type": "tool_use",
              "id": "tool-1",
              "name": "fms_report_sniff_test",
              "input": {"judgment": {"ticker": "PCTY"}},
            }
          ],
        )
        result.tool_uses = [("tool-1", "fms_report_sniff_test", {"judgment": {"ticker": "PCTY"}})]
        return object(), result
      return object(), StreamTurnResult(
        full_text="unwanted final prose",
        stop_reason="end_turn",
        content_blocks=[{"type": "text", "text": "unwanted final prose"}],
      )

    async def fake_execute_tool_use_loop(*args: Any, **kwargs: Any) -> ToolUseLoopResult:
      _ = args, kwargs
      runner._stop_after_tool_results_reason = "terminal_fms_result"
      runner._stop_after_tool_results_tool_name = "fms_report_sniff_test"
      return ToolUseLoopResult(
        tool_results_content=[
          {"type": "tool_result", "tool_use_id": "tool-1", "content": '{"status":"noop"}'}
        ],
        tools_used=["fms_report_sniff_test"],
      )

    runner._stream_turn = fake_stream_turn  # type: ignore[method-assign]
    monkeypatch.setattr(gateway_runner, "_execute_tool_use_loop", fake_execute_tool_use_loop)

    await runner.run(messages=[{"role": "user", "content": "Run sniff test"}], system_prompt="x")

    assert stream_calls["count"] == 1
    events = [entry.event for entry in runner._log.entries]
    assert events[-1]["type"] == "stream_complete"
    assistant_text = [
      block.get("text")
      for event in events
      if event.get("type") == "assistant_message"
      for block in event.get("content_blocks", [])
      if block.get("type") == "text"
    ]
    assert "unwanted final prose" not in assistant_text

  asyncio.run(_case())


def test_terminal_fms_result_wins_over_post_tool_budget_boundary(monkeypatch) -> None:
  async def _case() -> None:
    runner = _make_credential_runner()
    stream_calls = {"count": 0}
    exceeded_state = SimpleNamespace(
      total_cost=2.6353,
      budget=2.0,
      reason="parent_budget",
      reason_suffix="",
    )
    runner._cost_accumulator = object()
    monkeypatch.setattr(
      gateway_runner,
      "_budget_cost_progress",
      lambda *_args, **_kwargs: SimpleNamespace(
        last_reported_cost=2.6353,
        exceeded_state=exceeded_state,
      ),
    )
    monkeypatch.setattr(
      gateway_runner,
      "_budget_exceeded_state",
      lambda _accumulator: exceeded_state,
    )

    async def fake_stream_turn(**kwargs: Any):
      _ = kwargs
      stream_calls["count"] += 1
      result = StreamTurnResult(
        full_text="",
        stop_reason="tool_use",
        content_blocks=[
          {
            "type": "tool_use",
            "id": "tool-1",
            "name": "fms_propose_managing_risk",
            "input": {"judgment": {"ticker": "PCTY"}},
          }
        ],
      )
      result.tool_uses = [
        ("tool-1", "fms_propose_managing_risk", {"judgment": {"ticker": "PCTY"}})
      ]
      return object(), result

    async def fake_execute_tool_use_loop(*args: Any, **kwargs: Any) -> ToolUseLoopResult:
      _ = args, kwargs
      runner._stop_after_tool_results_reason = "terminal_fms_result"
      runner._stop_after_tool_results_tool_name = "fms_propose_managing_risk"
      return ToolUseLoopResult(
        tool_results_content=[
          {
            "type": "tool_result",
            "tool_use_id": "tool-1",
            "content": '{"status":"staged","proposal_id":"proposal-1"}',
          }
        ],
        tools_used=["fms_propose_managing_risk"],
      )

    runner._stream_turn = fake_stream_turn  # type: ignore[method-assign]
    monkeypatch.setattr(gateway_runner, "_execute_tool_use_loop", fake_execute_tool_use_loop)

    await runner.run(messages=[{"role": "user", "content": "Run managing risk"}], system_prompt="x")

    assert stream_calls["count"] == 1
    events = [entry.event for entry in runner._log.entries]
    assert events[-1]["type"] == "stream_complete"
    assert not any(event.get("type") == "budget_exceeded" for event in events)
    assert not any(
      event.get("type") == "run_interrupted" and event.get("reason") == "budget_exceeded"
      for event in events
    )

  asyncio.run(_case())
