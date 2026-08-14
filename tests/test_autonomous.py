import asyncio
import inspect
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway  # noqa: E402
import agent_gateway.autonomous as autonomous  # noqa: E402
import agent_gateway.autonomous_output as autonomous_output  # noqa: E402
from agent_gateway import EventLog  # noqa: E402
from agent_gateway.capability_binding import CapabilityBind, CredentialHandle  # noqa: E402
from agent_gateway.capability_execution import BoundCapabilityExecution  # noqa: E402
from agent_gateway.model_registry import (  # noqa: E402
  ModelRegistryEntry,
  ProductModelRegistry,
)
from agent_gateway.providers import ModelInfo, ModelProvider  # noqa: E402
from agent_gateway.session import GatewaySession  # noqa: E402


def _run(coro):
  return asyncio.run(coro)


class _MockResponse:
  def raise_for_status(self) -> None:
    return None


class _RecordingAsyncClient:
  def __init__(self, calls: list[dict[str, Any]], *args: Any, **kwargs: Any) -> None:
    _ = args, kwargs
    self._calls = calls

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc, tb) -> None:
    _ = exc_type, exc, tb

  async def post(self, url: str, json: dict[str, Any] | None = None, headers: dict[str, str] | None = None):
    self._calls.append({"url": url, "json": json, "headers": headers})
    return _MockResponse()


class _StubProvider(ModelProvider):
  name = "stub"

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name)

  def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    if False:
      yield


def _autonomous_bind(
  *,
  provider: str = "stub",
  model: str = "stub-model",
  effort: str = "none",
  run_mode: str = "autonomous",
  capability_id: str = "session.driver",
) -> CapabilityBind:
  return CapabilityBind(
    schema_version="1.0",
    capability_id=capability_id,  # type: ignore[arg-type]
    model_key=f"test.{provider}.{model}",
    provider=provider,
    upstream_model=model,
    adapter=f"test.{provider}",
    protocol_profile="test.reasoning",
    route="test.in_process",
    effort=effort,
    credential_principal="service",
    credential_ref=f"test-service:{provider}",
    run_mode=run_mode,  # type: ignore[arg-type]
    registry_revision="test-autonomous.1",
    policy_revision="test-autonomous.1",
    selection_source=(
      "capability_default"
      if capability_id == "session.driver"
      else "internal_policy"
    ),
  )


def _registry_for_bind(bind: CapabilityBind) -> ProductModelRegistry:
  return ProductModelRegistry(
    schema="product-model-registry/v1",
    revision=bind.registry_revision,
    models={
      bind.model_key: ModelRegistryEntry(
        key=bind.model_key,
        label="Autonomous test model",
        provider=bind.provider,
        upstream_model=bind.upstream_model,
        adapter=bind.adapter,
        protocol_profile=bind.protocol_profile,
        route=bind.route,
        lifecycle="active",
        capabilities={
          bind.capability_id: (
            "user_selectable"
            if bind.capability_id == "session.driver"
            else "internal"
          )
        },
        supported_efforts=frozenset({bind.effort}),
        default_effort=bind.effort,
        features=frozenset({"tools", "streaming"}),
        reported_identities=frozenset({bind.upstream_model}),
      )
    },
  )


def _bound_execution(
  *,
  provider: ModelProvider | None = None,
  bind: CapabilityBind | None = None,
  auth_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
  resolved_provider = provider or _StubProvider()
  resolved_bind = bind or _autonomous_bind(provider=resolved_provider.name)
  resolved_auth_config = (
    auth_config
    if auth_config is not None
    else {
      "provider": resolved_bind.provider,
      "max_tokens": 16_000,
      "api_key": "test-key",
    }
  )
  return {
    "capability_execution": BoundCapabilityExecution(
      bind=resolved_bind,
      registry=_registry_for_bind(resolved_bind),
      adapter=resolved_provider,
      auth_config=resolved_auth_config,
    ),
    "session": GatewaySession(
      session_id="autonomous-test",
      api_key_hash="test",
      created_at=1,
      expires_at=2,
      user_id="alice",
      role="owner",
      tenant_id="test-tenant",
      session_credential_handle=CredentialHandle(
        handle_id=resolved_bind.credential_ref,
        provider=resolved_bind.provider,
        principal="service",
        tenant_id="test-tenant",
        actor_id=None,
      ),
    ),
  }


def test_autonomous_output_exports_preserve_public_parent_surface() -> None:
  helper_names = (
    "RunOutput",
    "build_state_payload",
    "collect_run_output",
    "extract_state_update",
    "format_run_summary",
    "load_state",
    "mark_post_run_guard_failure",
    "run_output_exit_code",
    "run_output_outcome",
    "save_state",
  )

  assert autonomous.RunOutput is autonomous_output.RunOutput
  for name in helper_names:
    assert getattr(agent_gateway, name) is getattr(autonomous, name)


def test_autonomous_output_wrappers_preserve_parent_private_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
  class _PatchedRunOutput:
    def __init__(self, **kwargs: Any) -> None:
      self.kwargs = kwargs

  monkeypatch.setattr(autonomous, "RunOutput", _PatchedRunOutput)
  event_log = EventLog()
  event_log.append({"type": "text_delta", "text": "hello"})

  output = autonomous.collect_run_output(event_log, timed_out=True)

  assert isinstance(output, _PatchedRunOutput)
  assert output.kwargs["response"] == "hello"
  assert output.kwargs["timed_out"] is True

  monkeypatch.setattr(autonomous, "_extract_summary", lambda text, limit=1500: f"patched:{limit}:{text}")
  monkeypatch.setattr(autonomous, "_ensure_string_list", lambda value: [f"patched-list:{value!r}"])
  payload = autonomous.build_state_payload(
    previous_state={},
    model_state={"alerts": ["keep"]},
    run_output=autonomous_output.RunOutput("body", [], {}, None, False),
  )
  assert payload["last_summary"] == "patched:1500:body"
  assert payload["alerts"] == ["patched-list:['keep']"]
  assert "patched:1200:body" in autonomous.format_run_summary(
    autonomous_output.RunOutput("body", [], {}, None, False)
  )

  monkeypatch.setattr(autonomous, "_STATE_JSON_MARKER", "## CUSTOM_STATE")
  text = """
Ignored.

## CUSTOM_STATE
```json
{"status": "patched-marker"}
```
"""
  assert autonomous.extract_state_update(text) == {"status": "patched-marker"}

  read_calls: list[Path] = []
  write_calls: list[tuple[Path, dict[str, Any]]] = []
  monkeypatch.setattr(autonomous, "_read_json_object", lambda path: read_calls.append(path) or {"loaded": str(path)})
  monkeypatch.setattr(autonomous, "_atomic_write_json", lambda path, payload: write_calls.append((path, payload)))

  assert autonomous.load_state(Path("/tmp/state-root"), state_file="custom.json") == {
    "loaded": "/tmp/state-root/custom.json"
  }
  autonomous.save_state(Path("/tmp/state-root"), {"saved": True}, state_file="custom.json")
  assert read_calls == [Path("/tmp/state-root/custom.json")]
  assert write_calls == [(Path("/tmp/state-root/custom.json"), {"saved": True})]


def test_collect_run_output() -> None:
  event_log = EventLog()
  event_log.append({"type": "text_delta", "text": "stale"})
  event_log.append({"type": "tool_call_start", "tool_name": "old_tool"})
  event_log.append({"type": "stream_retry"})
  event_log.append({"type": "text_delta", "text": "Hello"})
  event_log.append({"type": "text_delta", "text": " world"})
  event_log.append({"type": "tool_call_start", "tool_name": "fresh_tool"})
  event_log.append({"type": "budget_exceeded"})
  event_log.append({"type": "max_turns_reached"})
  event_log.append({
    "type": "stream_complete",
    "terminal_disposition": "completed",
    "usage": {"input_tokens": 10, "output_tokens": 20},
  })

  output = autonomous.collect_run_output(event_log, timed_out=False)

  assert output.response == "Hello world"
  assert output.tools_used == ["fresh_tool"]
  assert output.usage == {"input_tokens": 10, "output_tokens": 20}
  assert output.error is None
  assert output.timed_out is False
  assert output.budget_exceeded is True
  assert output.max_turns_reached is True
  assert output.max_tokens_reached is False


def test_collect_run_output_detects_max_tokens_stop_reason() -> None:
  event_log = EventLog()
  event_log.append({"type": "text_delta", "text": "partial"})
  event_log.append({
    "type": "assistant_message",
    "content_blocks": [{"type": "text", "text": "partial"}],
    "stop_reason": "max_tokens",
  })
  event_log.append({
    "type": "stream_complete",
    "terminal_disposition": "completed",
    "usage": {"input_tokens": 1, "output_tokens": 2},
  })

  output = autonomous.collect_run_output(event_log, timed_out=False)

  assert output.response == "partial"
  assert output.max_tokens_reached is True
  assert output.error is None


def test_collect_run_output_detects_operator_pause() -> None:
  event_log = EventLog()
  event_log.append({"type": "text_delta", "text": "paused"})
  event_log.append({"type": "operator_pause", "reason": "operator_pause"})
  event_log.append({
    "type": "stream_complete",
    "terminal_disposition": "completed",
    "usage": {"input_tokens": 1, "output_tokens": 2},
  })

  output = autonomous.collect_run_output(event_log, timed_out=False)

  assert output.response == "paused"
  assert output.operator_paused is True
  assert output.error is None
  assert output.timed_out is False


@pytest.mark.parametrize(
  ("reason", "expected"),
  [
    ("budget_exceeded", "budget_exceeded"),
    ("max_turns_reached", "max_turns_reached"),
    ("operator_pause", "operator_paused"),
  ],
)
def test_collect_run_output_decodes_interrupted_terminal_disposition(
  reason: str,
  expected: str,
) -> None:
  event_log = EventLog()
  event_log.append({"type": "text_delta", "text": "partial"})
  event_log.append({
    "type": "stream_complete",
    "terminal_disposition": "interrupted",
    "reason": reason,
    "usage": {"input_tokens": 1},
  })
  event_log.append({
    "type": "stream_complete",
    "terminal_disposition": "completed",
    "usage": {"input_tokens": 999},
  })

  output = autonomous.collect_run_output(event_log, timed_out=False)

  assert getattr(output, expected) is True
  assert output.usage == {"input_tokens": 1}
  assert autonomous.run_output_outcome(output) != "success"


def test_collect_run_output_fails_closed_on_other_interruption_reason() -> None:
  event_log = EventLog()
  event_log.append({
    "type": "stream_complete",
    "terminal_disposition": "interrupted",
    "reason": "cancelled",
    "usage": {},
  })

  output = autonomous.collect_run_output(event_log, timed_out=False)

  assert output.error == "Autonomous run interrupted: cancelled"
  assert autonomous.run_output_exit_code(output) == 1
  assert autonomous.run_output_outcome(output) == "error"


def test_collect_run_output_fails_closed_on_unknown_terminal_disposition() -> None:
  event_log = EventLog()
  event_log.append({
    "type": "stream_complete",
    "terminal_disposition": "future_value",
    "usage": {},
  })

  output = autonomous.collect_run_output(event_log, timed_out=False)

  assert output.error == "invalid_terminal_disposition: 'future_value'"
  assert autonomous.run_output_exit_code(output) == 1


def test_collect_run_output_fails_closed_on_missing_terminal_disposition() -> None:
  event_log = EventLog()
  event_log.append({"type": "stream_complete", "usage": {}})

  output = autonomous.collect_run_output(event_log, timed_out=False)

  assert output.error == "invalid_terminal_disposition: None"
  assert autonomous.run_output_exit_code(output) == 1


@pytest.mark.parametrize(
  "events",
  [
    [],
    [{"type": "text_delta", "text": "partial"}],
  ],
  ids=["empty-log", "partial-only"],
)
def test_collect_run_output_requires_terminal_event(
  events: list[dict[str, Any]],
) -> None:
  event_log = EventLog()
  for event in events:
    event_log.append(event)

  output = autonomous.collect_run_output(event_log, timed_out=False)

  assert output.error == "missing_terminal_event"
  assert autonomous.run_output_exit_code(output) == 1
  assert autonomous.run_output_outcome(output) == "error"


def test_collect_run_output_first_terminal_error_wins() -> None:
  event_log = EventLog(defer_terminal_close=True)
  event_log.append({"type": "text_delta", "text": "partial"})
  event_log.append({"type": "error", "error": "provider failed"})
  event_log.append({
    "type": "stream_complete",
    "terminal_disposition": "completed",
    "usage": {"input_tokens": 999},
  })

  output = autonomous.collect_run_output(event_log, timed_out=False)

  assert output.error == "provider failed"
  assert output.usage == {}


def test_run_session_timeout() -> None:
  class _SlowRunner:
    async def run(self, **kwargs: Any) -> None:
      _ = kwargs
      await asyncio.sleep(0.2)

  output = _run(
    autonomous.run_session(
      _SlowRunner(),  # type: ignore[arg-type]
      EventLog(),
      max_turns=3,
      timeout_seconds=0.01,
      initial_message="hello",
      system_prompt="You are helpful.",
    )
  )

  assert output.timed_out is True
  assert output.response == ""
  assert output.error is None


def test_run_session_timeout_does_not_wait_for_slow_cancellation(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  event_log = EventLog()
  monkeypatch.setattr(autonomous, "_RUN_SESSION_CANCEL_DRAIN_SECONDS", 0.01)
  monkeypatch.setattr(autonomous, "_RUN_SESSION_FORCE_CLOSE_SECONDS", 0.01)

  class _SlowCancellationRunner:
    force_closed = False
    cancel_seen = False

    async def run(self, **kwargs: Any) -> None:
      _ = kwargs
      try:
        await asyncio.Event().wait()
      except asyncio.CancelledError:
        self.cancel_seen = True
        await asyncio.sleep(1.0)

    async def force_close(self, timeout: float = 2.0) -> None:
      _ = timeout
      self.force_closed = True

  async def _exercise() -> tuple[autonomous.RunOutput, float, _SlowCancellationRunner]:
    runner = _SlowCancellationRunner()
    started = time.monotonic()
    output = await autonomous.run_session(
      runner,  # type: ignore[arg-type]
      event_log,
      max_turns=3,
      timeout_seconds=0.01,
      initial_message="hello",
      system_prompt="You are helpful.",
    )
    return output, time.monotonic() - started, runner

  output, elapsed, runner = _run(_exercise())

  assert output.timed_out is True
  assert output.error is None
  assert runner.cancel_seen is True
  assert runner.force_closed is True
  assert elapsed < 0.2


def test_enrolled_run_session_waits_for_explicit_settlement_handshake() -> None:
  async def _case() -> None:
    event_log = EventLog()

    class _EnrolledRunner:
      top_level_skill_enrolled = True

      def __init__(self) -> None:
        self.run_returned = asyncio.Event()
        self.settlement = asyncio.Event()

      async def run(self, **kwargs: Any) -> None:
        _ = kwargs
        event_log.append({"type": "text_delta", "text": "completed"})
        event_log.append({
          "type": "stream_complete",
          "terminal_disposition": "completed",
          "usage": {},
        })
        self.run_returned.set()

      async def wait_for_top_level_skill_settlement(self) -> None:
        await self.settlement.wait()

    runner = _EnrolledRunner()
    task = asyncio.create_task(
      autonomous.run_session(
        runner,  # type: ignore[arg-type]
        event_log,
        max_turns=3,
        timeout_seconds=None,
        initial_message="hello",
        system_prompt=None,
      )
    )
    await asyncio.wait_for(runner.run_returned.wait(), timeout=1.0)
    await asyncio.sleep(0)
    assert not task.done()
    runner.settlement.set()
    output = await asyncio.wait_for(task, timeout=1.0)
    assert output.response == "completed"
    assert output.error is None

  _run(_case())


def test_enrolled_timeout_waits_for_owned_result_settlement() -> None:
  async def _case() -> None:
    event_log = EventLog()

    class _EnrolledRunner:
      top_level_skill_enrolled = True

      def __init__(self) -> None:
        self.causes: list[str] = []
        self.authoritative_cause: str | None = None
        self.cancel_seen = asyncio.Event()
        self.release = asyncio.Event()
        self.settlement = asyncio.Event()

      def set_server_terminal_cause(self, cause: str) -> bool:
        self.causes.append(cause)
        if self.authoritative_cause is None:
          self.authoritative_cause = cause
        return self.authoritative_cause == cause

      async def run(self, **kwargs: Any) -> None:
        _ = kwargs
        try:
          await asyncio.Event().wait()
        except asyncio.CancelledError:
          self.cancel_seen.set()
          await self.release.wait()
          event_log.append({
            "type": "stream_complete",
            "terminal_disposition": "interrupted",
            "reason": "timeout",
            "server_terminal_cause": "timeout",
            "usage": {},
          })
          self.settlement.set()
          raise

      async def force_close(self, timeout: float = 2.0) -> None:
        _ = timeout

      async def wait_for_top_level_skill_settlement(self) -> None:
        await self.settlement.wait()

    runner = _EnrolledRunner()
    task = asyncio.create_task(
      autonomous.run_session(
        runner,  # type: ignore[arg-type]
        event_log,
        max_turns=3,
        timeout_seconds=0.01,
        initial_message="hello",
        system_prompt=None,
      )
    )
    await asyncio.wait_for(runner.cancel_seen.wait(), timeout=1.0)
    await asyncio.sleep(0)
    assert not task.done()
    runner.release.set()
    output = await asyncio.wait_for(task, timeout=1.0)
    assert runner.causes == ["timeout"]
    assert runner.authoritative_cause == "timeout"
    assert output.timed_out is True
    assert output.error is None

  _run(_case())


def test_caller_cancellation_during_enrolled_timeout_drain_waits_for_settlement(
) -> None:
  async def _case() -> None:
    event_log = EventLog()

    class _EnrolledRunner:
      top_level_skill_enrolled = True

      def __init__(self) -> None:
        self.causes: list[str] = []
        self.authoritative_cause: str | None = None
        self.cancel_seen = asyncio.Event()
        self.release = asyncio.Event()
        self.settlement = asyncio.Event()

      def set_server_terminal_cause(self, cause: str) -> bool:
        self.causes.append(cause)
        if self.authoritative_cause is None:
          self.authoritative_cause = cause
        return self.authoritative_cause == cause

      def classify_server_cancellation_cause(self) -> str:
        return "caller_cancellation"

      async def run(self, **kwargs: Any) -> None:
        _ = kwargs
        try:
          await asyncio.Event().wait()
        except asyncio.CancelledError:
          self.cancel_seen.set()
          await self.release.wait()
          self.settlement.set()
          raise

      async def force_close(self, timeout: float = 2.0) -> None:
        _ = timeout

      async def wait_for_top_level_skill_settlement(self) -> None:
        await self.settlement.wait()

    runner = _EnrolledRunner()
    task = asyncio.create_task(
      autonomous.run_session(
        runner,  # type: ignore[arg-type]
        event_log,
        max_turns=3,
        timeout_seconds=0.01,
        initial_message="hello",
        system_prompt=None,
      )
    )
    await asyncio.wait_for(runner.cancel_seen.wait(), timeout=1.0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    runner.release.set()
    with pytest.raises(asyncio.CancelledError):
      await asyncio.wait_for(task, timeout=1.0)
    assert runner.settlement.is_set()
    assert runner.causes == ["timeout", "caller_cancellation"]
    assert runner.authoritative_cause == "timeout"

  _run(_case())


def test_caller_cancellation_after_completion_fence_does_not_cancel_run(
) -> None:
  async def _case() -> None:
    event_log = EventLog()

    class _EnrolledRunner:
      top_level_skill_enrolled = True

      def __init__(self) -> None:
        self.run_entered = asyncio.Event()
        self.release = asyncio.Event()
        self.settlement = asyncio.Event()
        self.cancel_seen = False

      def set_server_terminal_cause(self, cause: str) -> bool:
        assert cause == "caller_cancellation"
        return False

      def classify_server_cancellation_cause(self) -> str:
        return "caller_cancellation"

      async def run(self, **kwargs: Any) -> None:
        _ = kwargs
        self.run_entered.set()
        try:
          await self.release.wait()
        except asyncio.CancelledError:
          self.cancel_seen = True
          raise
        event_log.append({
          "type": "stream_complete",
          "terminal_disposition": "completed",
          "usage": {},
        })
        self.settlement.set()

      async def wait_for_top_level_skill_settlement(self) -> None:
        await self.settlement.wait()

    runner = _EnrolledRunner()
    task = asyncio.create_task(
      autonomous.run_session(
        runner,  # type: ignore[arg-type]
        event_log,
        max_turns=3,
        timeout_seconds=None,
        initial_message="hello",
        system_prompt=None,
      )
    )
    await asyncio.wait_for(runner.run_entered.wait(), timeout=1.0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert runner.cancel_seen is False
    runner.release.set()
    with pytest.raises(asyncio.CancelledError):
      await asyncio.wait_for(task, timeout=1.0)
    assert runner.cancel_seen is False
    assert runner.settlement.is_set()

  _run(_case())


@pytest.mark.parametrize("timeout_seconds", [0, None, -1])
def test_run_session_no_wall_clock(
  monkeypatch: pytest.MonkeyPatch,
  timeout_seconds: float | None,
) -> None:
  event_log = EventLog()

  class _Runner:
    async def run(self, **kwargs: Any) -> None:
      _ = kwargs
      event_log.append({"type": "text_delta", "text": "completed"})
      event_log.append({
        "type": "stream_complete",
        "terminal_disposition": "completed",
        "usage": {},
      })

  async def _unexpected_wait_for(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("asyncio.wait_for should not wrap non-positive timeouts")

  monkeypatch.setattr(autonomous.asyncio, "wait_for", _unexpected_wait_for)

  output = _run(
    autonomous.run_session(
      _Runner(),  # type: ignore[arg-type]
      event_log,
      max_turns=3,
      timeout_seconds=timeout_seconds,
      initial_message="hello",
      system_prompt="You are helpful.",
    )
  )

  assert output.timed_out is False
  assert output.error is None
  assert output.response == "completed"


def test_run_session_writer_lease_collision_is_info_skip(
  caplog: pytest.LogCaptureFixture,
) -> None:
  event_log = EventLog()

  class _LeaseHeldRunner:
    async def run(self, **kwargs: Any) -> None:
      _ = kwargs
      raise autonomous.WriterLeaseAlreadyHeldError(
        "Writer lease already held for /tmp/agentsess_analyst_1.jsonl"
      )

  with caplog.at_level(logging.INFO, logger="agent_gateway.autonomous"):
    output = _run(
      autonomous.run_session(
        _LeaseHeldRunner(),  # type: ignore[arg-type]
        event_log,
        max_turns=3,
        timeout_seconds=None,
        initial_message="hello",
        system_prompt="You are helpful.",
      )
    )

  assert output.error == (
    "WriterLeaseAlreadyHeldError: Writer lease already held for "
    "/tmp/agentsess_analyst_1.jsonl"
  )
  assert output.exit_reason == "writer_lease_already_held"
  assert "Autonomous run skipped: WriterLeaseAlreadyHeldError" in caplog.text
  assert "Traceback (most recent call last)" not in caplog.text
  assert not [
    record
    for record in caplog.records
    if record.name == "agent_gateway.autonomous" and record.levelno >= logging.WARNING
  ]


def test_run_autonomous_uses_exact_prebound_model(monkeypatch: pytest.MonkeyPatch) -> None:
  captured: dict[str, Any] = {}

  async def _fake_run_session(
    runner,
    event_log,
    *,
    max_turns: int,
    timeout_seconds: float | None,
    initial_message: str,
    system_prompt: str | list[tuple[str, bool]],
  ) -> autonomous.RunOutput:
    _ = event_log, max_turns, timeout_seconds, initial_message, system_prompt
    captured["model"] = (
      runner._capability_execution.bind.upstream_model
    )
    return autonomous.RunOutput("ok", [], {}, None, False)

  monkeypatch.setattr(autonomous, "run_session", _fake_run_session)

  provider = _StubProvider()
  output = _run(
    autonomous.run_autonomous(
      "System",
      "Hello",
      **_bound_execution(provider=provider),
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
  )

  assert output.response == "ok"
  assert captured["model"] == "stub-model"


def test_run_autonomous_binds_trusted_top_level_skill_identity(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  captured: dict[str, Any] = {}

  async def _fake_run_session(
    runner,
    event_log,
    **kwargs: Any,
  ) -> autonomous.RunOutput:
    _ = event_log, kwargs
    captured["skill_run_id"] = runner._skill_run_id
    return autonomous.RunOutput("ok", [], {}, None, False)

  monkeypatch.setattr(autonomous, "run_session", _fake_run_session)
  authority = _bound_execution()
  authority["session"].approval_policy = type(
    "TrustedPolicy",
    (),
    {"policy_bundle_hash": "a" * 64},
  )()

  output = _run(
    autonomous.run_autonomous(
      "System",
      "Hello",
      **authority,
      top_level_skill_name="market-scan",
      skill_run_id="run-skill-market-scan-test",
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
  )

  assert output.response == "ok"
  assert captured["skill_run_id"] == "run-skill-market-scan-test"
  assert authority["session"].run_context.skill == "market-scan"
  assert authority["session"].run_context.run_id == "run-skill-market-scan-test"
  assert authority["session"].run_context.policy_bundle_hash == "a" * 64


def test_run_autonomous_rejects_top_level_skill_without_trusted_policy() -> None:
  with pytest.raises(ValueError, match="trusted policy bundle"):
    _run(
      autonomous.run_autonomous(
        "System",
        "Hello",
        **_bound_execution(),
        top_level_skill_name="market-scan",
        skill_run_id="run-skill-market-scan-test",
        user_id="alice",
        billing_mode="byok",
        rate_table_version="unknown",
      )
    )


def test_run_autonomous_public_surface_requires_prebound_execution() -> None:
  parameters = inspect.signature(autonomous.run_autonomous).parameters

  assert "capability_execution" in parameters
  assert "session" in parameters
  assert {
    "api_key",
    "auth_config",
    "auth_token",
    "bound_auth_config",
    "capability_bind",
    "execution_transport",
    "max_tokens",
    "model",
    "provider",
    "provider_config",
  }.isdisjoint(parameters)


def test_run_autonomous_rejects_non_gateway_session_authority() -> None:
  authority = _bound_execution()
  authority["session"] = object()

  with pytest.raises(TypeError, match="exact GatewaySession"):
    _run(
      autonomous.run_autonomous(
        "System",
        "Hello",
        **authority,
        user_id="alice",
        billing_mode="byok",
        rate_table_version="unknown",
      )
    )


@pytest.mark.parametrize(
  ("bind", "auth_config", "error_match"),
  [
    (
      _autonomous_bind(capability_id="node.explore"),
      None,
      "session.driver",
    ),
    (
      _autonomous_bind(run_mode="interactive"),
      None,
      "autonomous or cron",
    ),
    (
      _autonomous_bind(),
      {
        "provider": "stub",
        "model": "other-model",
        "max_tokens": 16_000,
        "api_key": "test-key",
      },
      "selection|forbidden|model",
    ),
  ],
)
def test_run_autonomous_rejects_bind_mismatch_before_mcp_or_client(
  monkeypatch: pytest.MonkeyPatch,
  bind: CapabilityBind,
  auth_config: dict[str, Any] | None,
  error_match: str,
) -> None:
  provider = _StubProvider()
  client_calls = 0

  def _unexpected_create_client(
    config: dict[str, Any],
    *,
    timeout: float | None = None,
  ) -> Any:
    nonlocal client_calls
    _ = config, timeout
    client_calls += 1
    raise AssertionError("provider client must not be created")

  class _UnexpectedMcpClientManager:
    def __init__(self, **_kwargs: Any) -> None:
      raise AssertionError("MCP manager must not be constructed")

  monkeypatch.setattr(provider, "create_client", _unexpected_create_client)
  monkeypatch.setattr(
    autonomous,
    "McpClientManager",
    _UnexpectedMcpClientManager,
  )

  with pytest.raises((TypeError, ValueError), match=error_match):
    _run(
      autonomous.run_autonomous(
        "System",
        "Hello",
        **_bound_execution(
          provider=provider,
          bind=bind,
          auth_config=auth_config,
        ),
        mcp_servers={"fixture": {"command": "never"}},
        user_id="alice",
        billing_mode="byok",
        rate_table_version="unknown",
      )
    )

  assert client_calls == 0


def test_run_autonomous_rejects_missing_bound_credential_before_client(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  class _MissingCredentialProvider(_StubProvider):
    def has_active_credential(self, config: dict[str, Any]) -> bool:
      _ = config
      return False

    def create_client(
      self,
      config: dict[str, Any],
      *,
      timeout: float | None = None,
    ) -> Any:
      _ = config, timeout
      raise AssertionError("provider client must not be created")

  monkeypatch.setattr(
    autonomous,
    "McpClientManager",
    lambda **_kwargs: pytest.fail("MCP manager must not be constructed"),
  )

  with pytest.raises(ValueError, match="credential material is unavailable"):
    _run(
      autonomous.run_autonomous(
        "System",
        "Hello",
        **_bound_execution(provider=_MissingCredentialProvider()),
        mcp_servers={"fixture": {"command": "never"}},
        user_id="alice",
        billing_mode="byok",
        rate_table_version="unknown",
      )
    )


def test_run_autonomous_sync_rejects_removed_raw_selection_fields() -> None:
  with pytest.raises(TypeError, match="raw execution selection fields: model"):
    autonomous.run_autonomous_sync(
      "System",
      "Hello",
      **_bound_execution(),
      model="other-model",
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )


@pytest.mark.parametrize(
  ("output", "expected"),
  [
    (autonomous.RunOutput("ok", [], {}, None, False), 0),
    (autonomous.RunOutput("ok", [], {}, "boom", False), 1),
    (autonomous.RunOutput("ok", [], {}, None, False, budget_exceeded=True), 2),
    (autonomous.RunOutput("ok", [], {}, None, False, max_turns_reached=True), 3),
    (autonomous.RunOutput("ok", [], {}, None, False, max_tokens_reached=True), 4),
    (autonomous.RunOutput("ok", [], {}, None, False, operator_paused=True), 0),
    (autonomous.RunOutput("ok", [], {}, "guard failed", False, exit_reason="post_run_guard_failed"), 1),
    (autonomous.RunOutput("ok", [], {}, None, True), 124),
  ],
)
def test_run_output_exit_code(output: autonomous.RunOutput, expected: int) -> None:
  assert autonomous.run_output_exit_code(output) == expected


@pytest.mark.parametrize(
  ("output", "expected"),
  [
    (autonomous.RunOutput("ok", [], {}, None, False), "success"),
    (autonomous.RunOutput("ok", [], {}, None, True), "timeout"),
    (autonomous.RunOutput("ok", [], {}, "boom", False), "error"),
    (autonomous.RunOutput("ok", [], {}, None, False, budget_exceeded=True), "budget_exceeded"),
    (autonomous.RunOutput("ok", [], {}, None, False, max_turns_reached=True), "max_turns"),
    (autonomous.RunOutput("ok", [], {}, None, False, max_tokens_reached=True), "max_tokens"),
    (autonomous.RunOutput("ok", [], {}, None, False, operator_paused=True), "operator_pause"),
    (
      autonomous.RunOutput("ok", [], {}, "guard failed", False, exit_reason="post_run_guard_failed"),
      "post_run_guard_failed",
    ),
  ],
)
def test_run_output_outcome(output: autonomous.RunOutput, expected: str) -> None:
  assert autonomous.run_output_outcome(output) == expected


def test_post_run_guard_failure_marks_exit_reason_and_summary() -> None:
  output = autonomous.RunOutput("Agent completed.", [], {}, None, False)

  autonomous.mark_post_run_guard_failure(
    output,
    guard="producer_skill_artifact_postcondition",
    message="Missing standard artifact.",
    details={"expected_file": "skills/demo/out.md"},
  )

  assert autonomous.run_output_exit_code(output) == 1
  assert autonomous.run_output_outcome(output) == "post_run_guard_failed"
  assert output.post_run_guard == {
    "guard": "producer_skill_artifact_postcondition",
    "message": "Missing standard artifact.",
    "expected_file": "skills/demo/out.md",
  }
  summary = autonomous.format_run_summary(output)
  assert "Status: post-run guard failed" in summary
  assert "Exit reason: post_run_guard_failed" in summary
  assert "Post-run guard: producer_skill_artifact_postcondition" in summary
  assert "Error: Missing standard artifact." in summary


def test_extract_state_update() -> None:
  text = """
Summary text.

## STATE_UPDATE_JSON
```json
{"alerts": ["Check filings"], "next_session": ["Review guidance"]}
```
"""

  assert autonomous.extract_state_update(text) == {
    "alerts": ["Check filings"],
    "next_session": ["Review guidance"],
  }


def test_load_save_state(tmp_path: Path) -> None:
  payload = {"skill": "monitor", "count": 2}

  autonomous.save_state(tmp_path, payload)

  assert autonomous.load_state(tmp_path) == payload


def test_build_state_payload() -> None:
  output = autonomous.RunOutput(
    response="Long summary text",
    tools_used=["file_read", "file_read", "web_fetch"],
    usage={"input_tokens": 5},
    error=None,
    timed_out=False,
  )

  payload = autonomous.build_state_payload(
    previous_state={"existing": "value", "alerts": ["keep"], "next_session": [None, "next"]},
    model_state={"alerts": ["fresh"], "new_key": 1},
    run_output=output,
    model_name="claude-opus-4-6",
    briefing_file="briefings/daily.md",
    connected_servers={"prices", "filings"},
    active_servers={"prices"},
  )

  assert payload["existing"] == "value"
  assert payload["new_key"] == 1
  assert payload["model"] == "claude-opus-4-6"
  assert payload["briefing_file"] == "briefings/daily.md"
  assert payload["connected_servers"] == ["filings", "prices"]
  assert payload["active_servers"] == ["prices"]
  assert payload["tools_used"] == ["file_read", "web_fetch"]
  assert payload["usage"] == {"input_tokens": 5}
  assert payload["operator_paused"] is False
  assert payload["max_tokens_reached"] is False
  assert payload["last_outcome"] == "success"
  assert payload["last_summary"] == "Long summary text"
  assert payload["alerts"] == ["fresh"]
  assert payload["next_session"] == ["next"]
  assert "last_run" in payload


@pytest.mark.parametrize(
  ("output", "expected_outcome", "expected_error"),
  [
    (
      autonomous.RunOutput(
        response="partial response",
        tools_used=["file_read"],
        usage={"input_tokens": 5},
        error="provider failed",
        timed_out=False,
      ),
      "error",
      "provider failed",
    ),
    (
      autonomous.RunOutput(
        response="partial response",
        tools_used=["file_read"],
        usage={"input_tokens": 5},
        error=None,
        timed_out=False,
        budget_exceeded=True,
      ),
      "budget_exceeded",
      None,
    ),
  ],
  ids=["terminal-error", "interrupted-budget"],
)
def test_build_state_payload_rejects_model_state_from_non_successful_run(
  output: autonomous.RunOutput,
  expected_outcome: str,
  expected_error: str | None,
) -> None:
  payload = autonomous.build_state_payload(
    previous_state={
      "domain_value": "previous",
      "alerts": ["keep"],
      "error": "stale error",
    },
    model_state={
      "domain_value": "partial",
      "alerts": ["partial"],
      "new_partial_key": True,
    },
    run_output=output,
    model_name="claude-opus-4-6",
  )

  assert payload["domain_value"] == "previous"
  assert payload["alerts"] == ["keep"]
  assert "new_partial_key" not in payload
  assert payload["last_outcome"] == expected_outcome
  if expected_error is None:
    assert "error" not in payload
  else:
    assert payload["error"] == expected_error


def test_format_run_summary() -> None:
  output = autonomous.RunOutput(
    response="This is a compact summary.",
    tools_used=["tool_a", "tool_b"],
    usage={"input_tokens": 11, "output_tokens": 22, "estimated_cost": 0.12},
    error=None,
    timed_out=False,
  )

  message = autonomous.format_run_summary(
    output,
    label="Nightly analyst run",
    state={"briefing_file": "notes/briefing.md", "ideas": 3},
    format_state_fn=lambda state: f"Ideas: {state['ideas']}",
  )

  assert "Nightly analyst run" in message
  assert "Status: completed" in message
  assert "Briefing: notes/briefing.md" in message
  assert "Ideas: 3" in message
  assert "Summary:" in message


def test_format_run_summary_operator_paused_status() -> None:
  output = autonomous.RunOutput("Paused cleanly.", [], {}, None, False, operator_paused=True)

  message = autonomous.format_run_summary(output, label="Nightly analyst run")

  assert "Status: operator paused" in message


def test_format_run_summary_max_tokens_status() -> None:
  output = autonomous.RunOutput("Partial.", [], {}, None, False, max_tokens_reached=True)

  message = autonomous.format_run_summary(output, label="Nightly analyst run")

  assert "Status: max tokens reached" in message


def test_deliver_telegram(monkeypatch: pytest.MonkeyPatch) -> None:
  calls: list[dict[str, Any]] = []
  monkeypatch.setattr(autonomous.httpx, "AsyncClient", lambda *args, **kwargs: _RecordingAsyncClient(calls, *args, **kwargs))

  output = autonomous.RunOutput("Summary", ["tool"], {}, None, False)
  _run(
    autonomous.deliver(
      autonomous.DeliveryConfig(
        telegram_bot_token="bot-token",
        telegram_chat_id="chat-id",
        telegram_label="Run label",
      ),
      output,
      {"briefing_file": "notes/briefing.md"},
    )
  )

  assert len(calls) == 1
  assert calls[0]["url"] == "https://api.telegram.org/botbot-token/sendMessage"
  assert "Run label" in str(calls[0]["json"]["text"])


def test_deliver_telegram_briefing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  calls: list[dict[str, Any]] = []
  monkeypatch.setattr(autonomous.httpx, "AsyncClient", lambda *args, **kwargs: _RecordingAsyncClient(calls, *args, **kwargs))

  briefing_path = tmp_path / "briefing.md"
  content = "A" * 4100
  briefing_path.write_text(content, encoding="utf-8")
  output = autonomous.RunOutput("Summary", [], {}, None, False)

  _run(
    autonomous.deliver(
      autonomous.DeliveryConfig(
        telegram_bot_token="bot-token",
        telegram_chat_id="chat-id",
        briefing_file=briefing_path,
      ),
      output,
      None,
    )
  )

  assert len(calls) == 1 + len(autonomous.split_messages(content))


def test_deliver_webhook(monkeypatch: pytest.MonkeyPatch) -> None:
  calls: list[dict[str, Any]] = []
  monkeypatch.setattr(autonomous.httpx, "AsyncClient", lambda *args, **kwargs: _RecordingAsyncClient(calls, *args, **kwargs))

  output = autonomous.RunOutput("Summary", ["tool"], {"input_tokens": 1}, None, False)
  _run(
    autonomous.deliver(
      autonomous.DeliveryConfig(webhook_url="https://example.com/hook"),
      output,
      {"briefing_file": "notes/briefing.md"},
    )
  )

  assert len(calls) == 1
  assert calls[0]["url"] == "https://example.com/hook"
  assert calls[0]["json"]["outcome"] == "success"
  assert calls[0]["json"]["run_output"]["response"] == "Summary"
  assert calls[0]["json"]["state"]["briefing_file"] == "notes/briefing.md"


def test_deliver_callback() -> None:
  seen: dict[str, Any] = {}

  async def _on_complete(run_output: autonomous.RunOutput, state: dict[str, Any] | None) -> None:
    seen["response"] = run_output.response
    seen["state"] = dict(state or {})

  _run(
    autonomous.deliver(
      autonomous.DeliveryConfig(on_complete=_on_complete),
      autonomous.RunOutput("Summary", [], {}, None, False),
      {"briefing_file": "notes/briefing.md"},
    )
  )

  assert seen == {
    "response": "Summary",
    "state": {"briefing_file": "notes/briefing.md"},
  }


def test_run_autonomous_simple(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  captured: dict[str, Any] = {}
  delivered: dict[str, Any] = {}

  class _StubDispatcher:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      captured["dispatcher_args"] = args
      captured["dispatcher_kwargs"] = kwargs

  class _StubRunner:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      captured["runner_args"] = args
      captured["runner_kwargs"] = kwargs

  async def _fake_run_session(
    runner: Any,
    event_log: EventLog,
    *,
    max_turns: int,
    timeout_seconds: float,
    initial_message: str,
    system_prompt: str | list[tuple[str, bool]],
  ) -> autonomous.RunOutput:
    captured["run_session_runner"] = runner
    captured["run_session_event_log"] = event_log
    captured["run_session_kwargs"] = {
      "max_turns": max_turns,
      "timeout_seconds": timeout_seconds,
      "initial_message": initial_message,
      "system_prompt": system_prompt,
    }
    return autonomous.RunOutput(
      response=(
        "Completed.\n\n"
        "## STATE_UPDATE_JSON\n"
        "```json\n"
        '{"alerts":["Watch earnings"],"next_session":["Review transcript"]}\n'
        "```"
      ),
      tools_used=["file_read"],
      usage={"input_tokens": 1, "output_tokens": 2},
      error=None,
      timed_out=False,
    )

  async def _on_complete(run_output: autonomous.RunOutput, state: dict[str, Any] | None) -> None:
    delivered["response"] = run_output.response
    delivered["state"] = dict(state or {})

  state_dir = tmp_path / "state"
  autonomous.save_state(state_dir, {"existing": "value"})
  monkeypatch.setattr(autonomous, "ToolDispatcher", _StubDispatcher)
  monkeypatch.setattr(autonomous, "AgentRunner", _StubRunner)
  monkeypatch.setattr(autonomous, "run_session", _fake_run_session)

  provider = _StubProvider()
  authority = _bound_execution(provider=provider)
  result = _run(
    autonomous.run_autonomous(
      "You are helpful.",
      "Run the task.",
      **authority,
      tool_handlers={"local_tool": lambda *_args, **_kwargs: None},
      tool_definitions=[{"name": "local_tool", "description": "Local tool", "input_schema": {"type": "object"}}],
      max_turns=7,
      timeout_seconds=12,
      state_dir=state_dir,
      delivery=autonomous.DeliveryConfig(
        on_complete=_on_complete,
        briefing_file="notes/briefing.md",
      ),
      session_id="session-123",
      mcp_meta_inject_servers=frozenset({"idea-workbench-mcp"}),
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
  )

  assert result.response.startswith("Completed.")
  capability_execution = captured["runner_kwargs"]["capability_execution"]
  assert capability_execution.provider is provider
  assert capability_execution.bind.upstream_model == "stub-model"
  assert capability_execution.bind.effort == "none"
  assert captured["runner_kwargs"]["max_tokens_override"] == 16_000
  assert captured["runner_kwargs"]["allow_stub_response"] is False
  assert captured["runner_kwargs"]["session_id"] == "session-123"
  assert captured["dispatcher_kwargs"]["session"] is authority["session"]
  assert (
    captured["dispatcher_kwargs"]["run_context"]
    is authority["session"].run_context
  )
  assert captured["dispatcher_kwargs"]["role"] == authority["session"].role
  assert captured["dispatcher_kwargs"]["mcp_meta_inject_servers"] == frozenset(
    {"idea-workbench-mcp"}
  )
  assert captured["run_session_kwargs"]["initial_message"] == "Run the task."
  assert captured["run_session_kwargs"]["system_prompt"] == "You are helpful."

  saved_state = autonomous.load_state(state_dir)
  assert saved_state["existing"] == "value"
  assert saved_state["alerts"] == ["Watch earnings"]
  assert saved_state["next_session"] == ["Review transcript"]
  assert saved_state["briefing_file"] == "notes/briefing.md"
  assert delivered["state"]["existing"] == "value"


def test_run_autonomous_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
  captured: dict[str, Any] = {}

  class _StubDispatcher:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      _ = args, kwargs

  class _StubRunner:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      _ = args
      captured["runner_kwargs"] = kwargs

  async def _fake_run_session(
    runner: Any,
    event_log: EventLog,
    *,
    max_turns: int,
    timeout_seconds: float | None,
    initial_message: str,
    system_prompt: str | list[tuple[str, bool]],
  ) -> autonomous.RunOutput:
    _ = runner, event_log, initial_message, system_prompt
    captured["run_session_kwargs"] = {
      "max_turns": max_turns,
      "timeout_seconds": timeout_seconds,
    }
    return autonomous.RunOutput(
      response="Completed.",
      tools_used=[],
      usage={},
      error=None,
      timed_out=False,
    )

  monkeypatch.setattr(autonomous, "ToolDispatcher", _StubDispatcher)
  monkeypatch.setattr(autonomous, "AgentRunner", _StubRunner)
  monkeypatch.setattr(autonomous, "run_session", _fake_run_session)

  _run(
    autonomous.run_autonomous(
      "You are helpful.",
      "Run the task.",
      **_bound_execution(),
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
  )

  assert captured["run_session_kwargs"]["max_turns"] == 80
  assert captured["run_session_kwargs"]["timeout_seconds"] is None
  assert captured["runner_kwargs"]["per_turn_timeout"] is None


def test_run_autonomous_forwards_outputs_dir(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  captured: dict[str, Any] = {}

  class _StubDispatcher:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      _ = args, kwargs

  class _StubRunner:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      _ = args, kwargs

  async def _fake_run_session(
    runner: Any,
    event_log: EventLog,
    *,
    max_turns: int,
    timeout_seconds: float,
    initial_message: str,
    system_prompt: str | list[tuple[str, bool]],
  ) -> autonomous.RunOutput:
    _ = runner, event_log, max_turns, timeout_seconds, initial_message, system_prompt
    return autonomous.RunOutput(
      response="Completed.",
      tools_used=[],
      usage={},
      error=None,
      timed_out=False,
    )

  async def _fake_run_agent(_tool_input, **_kwargs):
    return {"response": "ok"}, None

  def _fake_make_run_agent_handler(*args, **kwargs):
    _ = args
    captured["kwargs"] = kwargs
    return _fake_run_agent

  skills_dir = tmp_path / "skills"
  outputs_dir = tmp_path / "outputs"
  skills_dir.mkdir(parents=True, exist_ok=True)

  monkeypatch.setattr(autonomous, "ToolDispatcher", _StubDispatcher)
  monkeypatch.setattr(autonomous, "AgentRunner", _StubRunner)
  monkeypatch.setattr(autonomous, "run_session", _fake_run_session)
  monkeypatch.setattr(autonomous, "make_run_agent_handler", _fake_make_run_agent_handler)

  authority = _bound_execution()
  _run(
    autonomous.run_autonomous(
      "You are helpful.",
      "Run the task.",
      **authority,
      skills_dir=skills_dir,
      outputs_dir=outputs_dir,
      capability_execution_resolver=object(),
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
  )

  assert captured["kwargs"]["outputs_dir"] == outputs_dir
  assert captured["kwargs"]["capability_execution_resolver"] is not None
  assert captured["kwargs"]["parent_session"] is authority["session"]


def test_run_autonomous_skills_dir_registers_send_message_tool_and_builtin_name(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  captured: dict[str, Any] = {}
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir(parents=True, exist_ok=True)

  class _StubMcpClientManager:
    def __init__(
      self,
      *,
      allowed_servers: set[str] | None = None,
      inline_servers: dict[str, dict[str, Any]] | None = None,
      config_path: str | Path | None = None,
      builtin_tool_names: set[str] | None = None,
      timeout_overrides: dict[str, int] | None = None,
      server_aliases: dict[str, str] | None = None,
    ) -> None:
      _ = allowed_servers, inline_servers, config_path, timeout_overrides, server_aliases
      self._builtin_tool_names = set(builtin_tool_names or set())

    async def startup(self) -> None:
      return None

    async def shutdown(self) -> None:
      return None

    def get_server_names(self) -> list[str]:
      return []

    def get_tool_definitions(self) -> list[dict[str, Any]]:
      return []

  async def _fake_run_session(
    runner: Any,
    event_log: EventLog,
    *,
    max_turns: int,
    timeout_seconds: float,
    initial_message: str,
    system_prompt: str | list[tuple[str, bool]],
  ) -> autonomous.RunOutput:
    _ = event_log, max_turns, timeout_seconds, initial_message, system_prompt
    captured["local_handlers"] = set(runner._dispatcher._local)
    captured["tool_defs"] = {tool["name"] for tool in runner._get_tool_definitions()}
    captured["builtin_tool_names"] = set(runner._mcp_client._builtin_tool_names)
    return autonomous.RunOutput(
      response="Completed.",
      tools_used=[],
      usage={},
      error=None,
      timed_out=False,
    )

  monkeypatch.setattr(autonomous, "McpClientManager", _StubMcpClientManager)
  monkeypatch.setattr(autonomous, "run_session", _fake_run_session)

  _run(
    autonomous.run_autonomous(
      "You are helpful.",
      "Run the task.",
      **_bound_execution(),
      skills_dir=skills_dir,
      capability_execution_resolver=object(),
      mcp_servers={"filesystem": {"command": "npx", "args": ["-y", "server"]}},
      trusted_mcp_allowed_servers={"filesystem"},
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
  )

  assert captured["local_handlers"] >= {"run_agent", "get_background_result", "send_message"}
  assert captured["tool_defs"] >= {"run_agent", "get_background_result", "send_message"}
  assert captured["builtin_tool_names"] >= {"run_agent", "get_background_result", "send_message"}


def test_run_autonomous_refuses_child_surface_without_capability_resolver(
  tmp_path: Path,
) -> None:
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir(parents=True, exist_ok=True)

  with pytest.raises(
    ValueError,
    match="requires capability_execution_resolver",
  ):
    _run(
      autonomous.run_autonomous(
        "You are helpful.",
        "Run the task.",
        **_bound_execution(),
        skills_dir=skills_dir,
        user_id="alice",
        billing_mode="byok",
        rate_table_version="unknown",
      )
    )


def test_extract_state_empty_text() -> None:
  assert autonomous.extract_state_update("") == {}


def test_extract_state_non_dict_json() -> None:
  text = """
## STATE_UPDATE_JSON
```json
[1, 2, 3]
```
"""

  assert autonomous.extract_state_update(text) == {}


def test_extract_state_malformed_json() -> None:
  text = """
## STATE_UPDATE_JSON
```json
{"alerts": [}
```
"""

  assert autonomous.extract_state_update(text) == {}


def test_split_messages_empty() -> None:
  assert autonomous.split_messages("") == []


def test_split_messages_force_split_long_line() -> None:
  assert autonomous.split_messages("ABCDEFGHIJ", max_len=4) == ["ABCD", "EFGH", "IJ"]
