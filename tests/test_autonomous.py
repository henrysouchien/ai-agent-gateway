import asyncio
import inspect
import logging
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway  # noqa: E402
import agent_gateway.autonomous as autonomous  # noqa: E402
import agent_gateway.autonomous_output as autonomous_output  # noqa: E402
from agent_gateway import EventLog  # noqa: E402
from agent_gateway.events import event_to_dict  # noqa: E402
from agent_gateway.multi_user.billing import SessionUsageSummary  # noqa: E402
from agent_gateway.session_recap import compute_recap, emit_recap_then_terminal  # noqa: E402
from model_authority.bind import CapabilityBind
from model_authority.binding import CapabilityEffort, CredentialHandle
from agent_gateway.capability_execution import BoundCapabilityExecution  # noqa: E402
from model_authority.registry import ModelRegistryEntry, ProductModelRegistry
from model_authority.schema import SCHEMA
from gateway_test_support.model_defaults import compat_for_profile  # noqa: E402
from gateway_test_support.host_policy import owner_session_host_policy  # noqa: E402, F401
from agent_gateway.mcp_client import _ServerState  # noqa: E402
from agent_gateway.providers import ModelInfo, ModelProvider, StreamEvent  # noqa: E402
from agent_gateway.session import GatewaySession  # noqa: E402
from agent_gateway.skill_limits import SkillExecutionLimits  # noqa: E402
from agent_gateway.tool_policy_registry import (  # noqa: E402
  ToolPolicyImplementationRegistry,
)
from agent_workflow_contracts.tool_registration import (  # noqa: E402
  ToolRegistrationCatalog,
)


def _run(coro):
  return asyncio.run(coro)

def _raw_capability_execution_resolver(**resolver):
  return resolver


def _raw_interceptors(**interceptors):
  return interceptors


def _raw_mcp_session(**session):
  return session


async def _noop_local_tool(*_args, **_kwargs):
  return {}, None


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

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
    _ = client, params
    if False:
      yield StreamEvent(type="message_start")


def _autonomous_bind(
  *,
  provider: str = "stub",
  model: str = "stub-model",
  effort: CapabilityEffort = "none",
  run_mode: str = "autonomous",
  capability_id: str = "session.driver",
) -> CapabilityBind:
  return CapabilityBind(
    schema_version="1.0",
    capability_id=capability_id,
    model_key=f"test.{provider}.{model}",
    provider=provider,
    upstream_model=model,
    adapter=f"test.{provider}",
    protocol_profile="responses.reasoning",
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
    schema=SCHEMA,
    revision=bind.registry_revision,
    models={
      bind.model_key: ModelRegistryEntry(
        key=bind.model_key,
        label="Autonomous test model",
        provider=bind.provider,
        upstream_model=bind.upstream_model,
        adapter=bind.adapter,
        protocol_profile=bind.protocol_profile,
        compat=compat_for_profile(bind.protocol_profile),
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
    "admitted_skill_execution_limits": None,
  }


@pytest.mark.parametrize("admitted_child", [False, True])
def test_run_autonomous_first_handle_read_keeps_executing_job(
  tmp_path: Path,
  owner_session_host_policy,
  admitted_child: bool,
) -> None:
  from dataclasses import replace
  from agent_gateway.admission import (
    capture_job_context,
    current_parent_context,
    root_context,
  )
  from agent_gateway.tool_result_spill import (
    SpillCapabilities,
    SpillSink,
    TOOL_RESULT_READ_TOOL_DEF,
    make_tool_result_read_handler,
    write_spill_set,
  )

  sink = SpillSink(
    lambda: str(tmp_path),
    capabilities=SpillCapabilities(file_read=True, spill_read=True),
  )
  publication = write_spill_set(
    sink=sink,
    tool_name="file_read",
    tool_use_id="first",
    content="autonomous evidence",
    model_max_chars=16000,
  )
  arguments = {
    "spill_ref": publication.spill_ref,
    "member_ref": "file_read_first.txt",
  }
  observed: list[tuple[Any, Any]] = []

  class _FirstReadProvider(_StubProvider):
    async def stream(self, client: Any, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
      yield StreamEvent(type="message_start", input_tokens=10)
      if not observed:
        yield StreamEvent(
          type="tool_use_end",
          tool_id="read-first",
          tool_name="tool_result_read",
          tool_input=arguments,
          raw_block={
            "type": "tool_use",
            "id": "read-first",
            "name": "tool_result_read",
            "input": arguments,
          },
        )
        yield StreamEvent(type="message_end", stop_reason="tool_use")
      else:
        yield StreamEvent(type="text_delta", text="read complete")
        yield StreamEvent(
          type="text_end",
          raw_block={"type": "text", "text": "read complete"},
        )
        yield StreamEvent(type="message_end", stop_reason="end_turn")

  bound = _bound_execution(provider=_FirstReadProvider())
  session = bound["session"]
  job_id = session.session_id
  if admitted_child:
    context = root_context(
      session,
      withheld_tools=frozenset(),
      research_file_id=None,
    )
    job_id = "physical-child-job"
    capture_job_context(session, replace(
      context,
      children=(),
      tool_ids=frozenset({"tool_result_read"}),
      job_id=job_id,
      depth=1,
    ))
  handler = make_tool_result_read_handler(
    lambda: sink,
    caller=lambda: current_parent_context(session),
    journal=lambda: None,
  )

  async def _read(tool_input, **kwargs):
    result = await handler(tool_input, **kwargs)
    observed.append(result)
    return result

  _run(autonomous.run_autonomous(
    "test",
    **bound,
    initial_message="read own evidence",
    user_id="alice",
    billing_mode="byok",
    rate_table_version="test",
    tool_handlers={"tool_result_read": _read},
    tool_definitions=[TOOL_RESULT_READ_TOOL_DEF],
  ))
  result, error = observed[0]
  assert error is None
  assert result["content"] == "autonomous evidence"
  assert current_parent_context(session).job_id == job_id
  if admitted_child:
    assert current_parent_context(session).children == ()


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
  event_log.append({"type": "soft_budget_steer", "kind": "turns", "observed": 100, "threshold": 100})
  event_log.append({"type": "soft_budget_final", "kind": "turns", "observed": 150, "threshold": 100})
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
  assert autonomous.run_output_outcome(output) == "success"
  assert autonomous.run_output_exit_code(output) == 0


def test_collect_approval_park_preserves_state_and_exposes_only_safe_references() -> None:
  event_log = EventLog(defer_terminal_close=True)
  event_log.append({"type": "text_delta", "text": "Waiting for the requested operation."})
  event_log.append({"type": "tool_call_start", "tool_name": "execute_operation"})
  usage = {"input_tokens": 31, "output_tokens": 17, "estimated_cost": 0.02}
  references = {
    "approval_id": "approval-park",
    "job_id": "job-original",
    "original_tool_call_id": "call-original",
    "root_turn_id": "turn-original",
  }
  event_log.append({
    "type": "stream_complete",
    "terminal_disposition": "awaiting_approval",
    "usage": usage,
    **references,
    "nonce": "private-nonce",
    "operands": {"secret": "private-operand"},
    "prepared_tool_call": {"arguments": "private-arguments"},
  })
  event_log.append({
    "type": "stream_complete",
    "terminal_disposition": "completed",
    "usage": {"input_tokens": 1, "output_tokens": 1},
  })
  recap = {"type": "session_recap", "summary": "Approval is pending."}
  event_log.append(recap)

  output = autonomous.collect_run_output(event_log, timed_out=False)

  assert autonomous.run_output_outcome(output) == "awaiting_approval"
  assert output.exit_reason == "awaiting_approval"
  assert output.error is None
  assert output.operator_paused is False
  assert output.timed_out is False
  assert output.response == "Waiting for the requested operation."
  assert output.tools_used == ["execute_operation"]
  assert output.usage == usage
  assert output.session_recap == recap
  assert output.approval_refs == references
  state = autonomous.build_state_payload(
    previous_state={"position": "original", "error": "stale"},
    model_state={"position": "uncommitted", "new_model_field": True},
    run_output=output,
  )
  assert state["position"] == "original"
  assert "new_model_field" not in state
  assert "error" not in state
  assert state["last_outcome"] == "awaiting_approval"
  assert state["approval_refs"] == references
  assert state["usage"] == usage
  summary = autonomous.format_run_summary(output, state=state)
  assert "Status: awaiting approval" in summary
  assert "Approval: approval-park" in summary
  assert "completed" not in summary
  assert "Error:" not in summary
  for private_value in ("private-nonce", "private-operand", "private-arguments"):
    assert private_value not in repr(output)
    assert private_value not in repr(state)
    assert private_value not in summary

  settled_state = autonomous.build_state_payload(
    previous_state=state,
    model_state={"position": "settled"},
    run_output=autonomous.RunOutput("Done.", [], {}, None, False),
  )
  assert settled_state["position"] == "settled"
  assert settled_state["last_outcome"] == "success"
  assert "approval_refs" not in settled_state


def test_length_stopped_run_records_measurement_and_keeps_terminal_status() -> None:
  event_log = EventLog()
  event_log.append({"type": "text_delta", "text": "partial, then continued"})
  event_log.append({
    "type": "assistant_message",
    "content_blocks": [{"type": "text", "text": "partial"}],
    "stop_reason": "max_tokens",
  })
  event_log.append({
    "type": "assistant_message",
    "content_blocks": [{"type": "text", "text": ", then continued"}],
    "stop_reason": "end_turn",
  })
  emit_recap_then_terminal(
    event_log,
    {
      "type": "stream_complete",
      "terminal_disposition": "completed",
      "usage": {"input_tokens": 1, "output_tokens": 2},
    },
    session_id="standalone",
    started_at=1.0,
  )

  output = autonomous.collect_run_output(event_log, timed_out=False)

  assert output.response == "partial, then continued"
  assert output.error is None
  assert autonomous.run_output_outcome(output) == "success"
  assert autonomous.run_output_exit_code(output) == 0
  assert output.session_recap is not None
  assert output.session_recap["tool_calls_summary"]["measurements"] == {
    "max_tokens_stops": 1,
  }


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


@pytest.mark.parametrize("reason", ["cancelled", "soft_budget_abort"])
def test_collect_run_output_fails_closed_on_other_interruption_reason(reason: str) -> None:
  event_log = EventLog()
  event_log.append({
    "type": "stream_complete",
    "terminal_disposition": "interrupted",
    "reason": reason,
    "usage": {},
  })

  output = autonomous.collect_run_output(event_log, timed_out=False)

  assert output.error == f"Autonomous run interrupted: {reason}"
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


@pytest.mark.parametrize(
  "terminal_order",
  ["recap_before_terminal", "recap_after_terminal", "error", "missing"],
)
def test_collect_run_output_carries_recap_to_durable_state(
  terminal_order: str,
  tmp_path: Path,
) -> None:
  event_log = EventLog(defer_terminal_close=True)
  event_log.append({"type": "text_delta", "text": "partial"})
  event_log.append({
    "type": "tool_call_start",
    "tool_call_id": "preview",
    "tool_name": "model_preview",
  })
  event_log.append({
    "type": "tool_call_complete",
    "tool_call_id": "preview",
    "tool_name": "model_preview",
    "is_error": True,
  })
  event_log.append({
    "type": "interceptor_measurement",
    "measurement": {"calls_in_window": 6, "structured_reads": 2},
  })
  summary = SessionUsageSummary(
    user_id="7",
    session_id="standalone",
    request_id="request",
    input_tokens=30,
    output_tokens=40,
    cache_read_tokens=0,
    cache_creation_tokens=0,
    cost=0.63,
    turns=3,
    channel="cli",
    started_at=1.0,
    ended_at=2.0,
  )
  terminal_usage = {"input_tokens": 10, "output_tokens": 20, "estimated_cost": 0.12}
  terminal: dict[str, Any] = {
    "type": "stream_complete",
    "terminal_disposition": "completed",
    "usage": terminal_usage,
  }
  if terminal_order == "recap_before_terminal":
    emit_recap_then_terminal(
      event_log, terminal, session_id="standalone", started_at=1.0, usage=summary,
    )
  else:
    if terminal_order == "recap_after_terminal":
      event_log.append(terminal)
    elif terminal_order == "error":
      event_log.append({"type": "error", "error": "provider failed"})
    event_log.append(event_to_dict(compute_recap(
      event_log,
      session_id="standalone",
      started_at=1.0,
      trigger="turn_end",
      usage=summary,
    )))

  if terminal_order != "missing":
    event_log.append({"type": "text_delta", "text": "not part of the result"})
    event_log.append({"type": "operator_pause"})
    event_log.append({
      "type": "stream_complete",
      "terminal_disposition": "completed",
      "usage": {"input_tokens": 999},
    })

  output = autonomous.collect_run_output(event_log, timed_out=False)
  assert output.response == "partial"
  assert output.tools_used == ["model_preview"]
  assert output.operator_paused is False
  expected_error = {
    "error": "provider failed",
    "missing": "missing_terminal_event",
  }.get(terminal_order)
  expected_usage = {} if expected_error else terminal_usage
  assert output.error == expected_error
  assert output.usage == expected_usage
  assert output.session_recap is not None
  assert output.session_recap["tool_calls_summary"] == {
    "total_calls": 1,
    "successes": 0,
    "errors": 1,
    "by_tool_name": {"model_preview": 1},
    "by_server": {"local": 1},
    "measurements": {"calls_in_window": 6, "structured_reads": 2, "model_calls": 1},
  }
  assert output.session_recap["usage"]["turns"] == 3
  assert output.session_recap["usage"]["cost"] == 0.63

  state = autonomous.build_state_payload(
    previous_state={"session_recap": {"session_id": "previous"}},
    model_state={"session_recap": {"session_id": "model-authored"}},
    run_output=output,
  )
  assert state["session_recap"] == output.session_recap
  autonomous.save_state(tmp_path, state)
  retained = autonomous.load_state(tmp_path)
  assert retained["session_recap"]["session_id"] == "standalone"
  assert retained["session_recap"]["tool_calls_summary"] == output.session_recap["tool_calls_summary"]
  assert retained["session_recap"]["usage"]["turns"] == 3
  assert retained["session_recap"]["usage"]["cost"] == 0.63
  assert retained["usage"] == expected_usage
  assert retained["last_outcome"] == ("error" if expected_error else "success")


def test_collect_run_output_keeps_latest_run_wide_recap_across_retry() -> None:
  event_log = EventLog()
  event_log.append({"type": "text_delta", "text": "stale"})
  event_log.append({
    "type": "tool_call_start", "tool_call_id": "old", "tool_name": "old_tool",
  })
  event_log.append({
    "type": "tool_call_complete", "tool_call_id": "old", "is_error": True,
  })
  event_log.append(event_to_dict(compute_recap(
    event_log, session_id="standalone", started_at=1.0, trigger="explicit",
  )))
  event_log.append({"type": "stream_retry"})
  event_log.append({"type": "text_delta", "text": "fresh"})
  event_log.append({
    "type": "tool_call_start", "tool_call_id": "fresh", "tool_name": "fresh_tool",
  })
  event_log.append({
    "type": "tool_call_complete", "tool_call_id": "fresh", "is_error": False,
  })
  emit_recap_then_terminal(
    event_log,
    {
      "type": "stream_complete",
      "terminal_disposition": "completed",
      "usage": {"input_tokens": 10, "output_tokens": 20, "estimated_cost": 0.12},
    },
    session_id="standalone",
    started_at=1.0,
  )

  output = autonomous.collect_run_output(event_log, timed_out=False)
  assert output.response == "fresh"
  assert output.tools_used == ["fresh_tool"]
  assert output.usage == {"input_tokens": 10, "output_tokens": 20, "estimated_cost": 0.12}
  assert output.session_recap is not None
  assert output.session_recap["tool_calls_summary"]["total_calls"] == 2
  assert output.session_recap["tool_calls_summary"]["successes"] == 1
  assert output.session_recap["tool_calls_summary"]["errors"] == 1
  assert output.session_recap["usage"] is None



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


def test_run_session_no_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
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
    raise AssertionError("asyncio.wait_for must not wrap run_session LLM work")

  monkeypatch.setattr(autonomous.asyncio, "wait_for", _unexpected_wait_for)

  output = _run(
    autonomous.run_session(
      _Runner(),  # type: ignore[arg-type]
      event_log,
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
  from gateway_test_support.capability_execution_test_support import stub_capability_execution_resolver
  captured: dict[str, Any] = {}

  async def _fake_run_session(
    runner,
    event_log,
    *,
    initial_message: str,
    system_prompt: str | list[tuple[str, bool]],
  ) -> autonomous.RunOutput:
    _ = event_log, initial_message, system_prompt
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
  authority["admitted_skill_execution_limits"] = (
    SkillExecutionLimits(None)
  )
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


def test_run_autonomous_top_level_skill_exposes_every_supplied_tool(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  captured: dict[str, Any] = {}
  real_manager_type = autonomous.McpClientManager

  def _real_logical_manager(**kwargs: Any):
    captured["manager_kwargs"] = kwargs
    manager = real_manager_type(**kwargs)
    physical_definitions = [
      {"name": "fmp_fetch", "input_schema": {"type": "object"}},
      {"name": "fmp_search", "input_schema": {"type": "object"}},
    ]
    manager._servers = {
      "fmp-mcp": _ServerState(
        name="fmp-mcp",
        **_raw_mcp_session(session=object()),
        exit_contexts=[],
        tool_definitions=physical_definitions,
        tool_names={
          definition["name"] for definition in physical_definitions
        },
        tool_prefix="",
        config=None,
      )
    }
    logical_aliases = {
      "fetch_financials": "fmp_fetch",
      "search_companies": "fmp_search",
    }
    manager._apply_collision_filtering(
      policy_server_for_tool=lambda tool_name: (
        "market-data-mcp"
        if tool_name in logical_aliases
        else "fmp-mcp"
      )
    )
    manager._started = True
    return manager

  async def _fake_run_session(runner, event_log, **kwargs: Any) -> autonomous.RunOutput:
    _ = event_log, kwargs
    captured["tool_defs"] = {
      definition["name"] for definition in runner._get_tool_definitions()
    }
    captured["local_handlers"] = set(runner._dispatcher._local)
    captured["mcp_scope"] = runner._dispatcher._allowed_mcp_tools_by_server
    return autonomous.RunOutput("ok", [], {}, None, False)

  monkeypatch.setattr(
    autonomous,
    "McpClientManager",
    _real_logical_manager,
  )
  monkeypatch.setattr(autonomous, "run_session", _fake_run_session)
  authority = _bound_execution()
  authority["admitted_skill_execution_limits"] = (
    SkillExecutionLimits(None)
  )
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
      mcp_servers={"market-data-mcp": {"command": "unused"}},
      trusted_mcp_allowed_servers={"market-data-mcp"},
      mcp_logical_server_routes={"market-data-mcp": "fmp-mcp"},
      mcp_logical_tool_aliases={
        "market-data-mcp": {
          "fetch_financials": "fmp_fetch",
          "search_companies": "fmp_search",
        }
      },
      tool_handlers={
        "declared_local": _noop_local_tool,
        "extra_local": _noop_local_tool,
      },
      tool_definitions=[
        {"name": "declared_local", "input_schema": {"type": "object"}},
        {"name": "extra_local", "input_schema": {"type": "object"}},
      ],
      top_level_skill_name="market-scan",
      skill_run_id="run-skill-market-scan-test",
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
  )

  assert output.response == "ok"
  assert captured["tool_defs"] == {
    "declared_local",
    "extra_local",
    "fetch_financials",
    "search_companies",
  }
  assert captured["local_handlers"] == {"declared_local", "extra_local"}
  assert captured["mcp_scope"] is None
  assert captured["manager_kwargs"]["logical_server_routes"] == {
    "market-data-mcp": "fmp-mcp"
  }
  assert set(captured["manager_kwargs"]["inline_servers"]) == {"fmp-mcp"}
  assert captured["manager_kwargs"]["logical_tool_aliases"] == {
    "market-data-mcp": {
      "fetch_financials": "fmp_fetch",
      "search_companies": "fmp_search",
    }
  }


def test_run_autonomous_preserves_manager_alias_precedence_without_logical_routes(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  captured: dict[str, Any] = {}
  real_manager_type = autonomous.McpClientManager

  def _real_alias_manager(**kwargs: Any):
    captured["inline_servers"] = kwargs["inline_servers"]
    manager = real_manager_type(**kwargs)
    captured["canonical_servers"] = manager._canonicalize_server_configs(
      kwargs["inline_servers"]
    )
    manager._started = True
    return manager

  async def _fake_run_session(
    _runner: Any,
    _event_log: EventLog,
    **_kwargs: Any,
  ) -> autonomous.RunOutput:
    return autonomous.RunOutput("ok", [], {}, None, False)

  monkeypatch.setattr(
    autonomous,
    "McpClientManager",
    _real_alias_manager,
  )
  monkeypatch.setattr(autonomous, "run_session", _fake_run_session)

  output = _run(
    autonomous.run_autonomous(
      "System",
      "Hello",
      **_bound_execution(),
      mcp_servers={
        "legacy-server": {"command": "legacy"},
        "canonical-server": {"command": "canonical"},
      },
      trusted_mcp_allowed_servers={"canonical-server"},
      trusted_mcp_server_aliases={
        "legacy-server": "canonical-server"
      },
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
  )

  assert output.response == "ok"
  assert captured["inline_servers"] == {
    "legacy-server": {"command": "legacy"},
    "canonical-server": {"command": "canonical"},
  }
  assert captured["canonical_servers"] == {
    "canonical-server": {"command": "canonical"}
  }


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
    previous_state={
      "existing": "value",
      "alerts": ["keep"],
      "next_session": [None, "next"],
      "session_recap": {"session_id": "previous"},
    },
    model_state={
      "alerts": ["fresh"],
      "new_key": 1,
      "session_recap": {"session_id": "model-authored"},
    },
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
  assert payload["session_recap"] is None
  assert payload["operator_paused"] is False
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
  ],
  ids=["terminal-error"],
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


def test_format_run_summary_preserves_long_reply_whole_for_transport_splitting() -> None:
  reply = "\n".join(f"Finding {index}: " + "x" * 90 for index in range(120))
  output = autonomous.RunOutput(reply, ["tool_a"], {"input_tokens": 10, "output_tokens": 20, "estimated_cost": 0.05}, None, False)
  message = autonomous.format_run_summary(output, label="Nightly analyst run")
  assert reply in message
  chunks = autonomous.split_messages(message, max_len=4096)
  assert len(chunks) > 1
  assert all(len(chunk) <= 4096 for chunk in chunks)
  assert reply in "".join(chunks)


def test_deliver_telegram_sends_a_long_reply_whole_in_limit_sized_messages(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls: list[dict[str, Any]] = []
  monkeypatch.setattr(autonomous.httpx, "AsyncClient", lambda *args, **kwargs: _RecordingAsyncClient(calls, *args, **kwargs))
  reply = "\n".join(f"Finding {index}: " + "x" * 90 for index in range(120))

  _run(
    autonomous.deliver(
      autonomous.DeliveryConfig(
        telegram_bot_token="bot-token",
        telegram_chat_id="chat-id",
        telegram_label="Run label",
      ),
      autonomous.RunOutput(reply, [], {}, None, False),
    )
  )

  texts = [str(call["json"]["text"]) for call in calls]
  assert len(texts) > 1
  assert all(len(text) <= 4096 for text in texts)
  assert reply in "".join(texts)


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


def test_run_autonomous_skills_dir_registers_send_message_tool_and_builtin_name(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  from gateway_test_support.capability_execution_test_support import stub_capability_execution_resolver
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
        logical_server_routes: dict[str, str] | None = None,
        logical_tool_aliases: dict[str, dict[str, str]] | None = None,
        input_preparation_routes: Any = (),
        tool_registration_catalog: Any = None,
      ) -> None:
      _ = (
        allowed_servers,
        inline_servers,
        config_path,
        timeout_overrides,
        server_aliases,
          logical_server_routes,
          logical_tool_aliases,
          input_preparation_routes,
          tool_registration_catalog,
        )
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
    initial_message: str,
    system_prompt: str | list[tuple[str, bool]],
  ) -> autonomous.RunOutput:
    _ = event_log, initial_message, system_prompt
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
      **_raw_capability_execution_resolver(
        capability_execution_resolver=stub_capability_execution_resolver(run_mode="autonomous"),
      ),
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


def test_run_autonomous_pairs_prebound_execution_with_explicit_resolver_for_background_child(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  owner_session_host_policy: Any,
) -> None:
  from dataclasses import FrozenInstanceError

  from agent_gateway.skills import SkillLoader
  from gateway_test_support.capability_execution_test_support import stub_capability_execution_resolver

  skills_dir = tmp_path / "skills"
  skills_dir.mkdir()
  (skills_dir / "resolver-child.md").write_text(
    "---\nexecution_class: node.implement\nname: resolver-child\nversion: '1.0'\n"
    "agent_callable: true\nagent_description: Read.\nmutation_mode: read_only\n"
    "allowed_tools: []\nmcp_tools: {}\n"
    "semantic_metadata:\n  required_context: []\n  tool_refs: []\n"
    "---\nRead and return the result.\n",
    encoding="utf-8",
  )
  loader = SkillLoader(skills_dir)
  operation = next(
    item.snapshot.operation.model_dump(mode="json")
    for item in loader.list_callable_operations()
    if item.snapshot.operation.name == "resolver-child"
  )
  provider = _StubProvider()
  provider.name = "openai"
  resolver = replace(
    stub_capability_execution_resolver(
      default_provider="openai", default_model="gpt-5.6-sol",
      run_mode="autonomous",
    ),
    adapter_resolver=lambda _adapter: provider,
  )
  authority = _bound_execution()
  resolved_execution = resolver.resolve("session.driver")
  execution = replace(
    resolved_execution,
    auth_config={**resolved_execution.auth_config, "max_tokens": 16_000},
    resolver=None,
  )
  execution.validate()
  authority["capability_execution"] = execution

  async def run_parent(runner: Any, _log: EventLog, **_kwargs: Any):
    result, error = await runner._dispatcher.dispatch("delegate", "run_agent", {
      "operation": operation, "objective": "Read.", "background": True,
    })
    assert error is None, error
    assert result is not None and result["task_id"]
    entry = runner._task_registry.get(result["task_id"])
    assert entry is not None
    assert entry.admitted_task is not None
    paired = runner.capability_execution
    assert paired is not execution
    assert paired.resolver is resolver
    assert paired.bind is execution.bind
    assert paired.registry is execution.registry
    assert paired.adapter is execution.adapter
    assert paired.auth_config == execution.auth_config
    paired.validate()
    with pytest.raises(FrozenInstanceError):
      paired.resolver = None
    return autonomous.RunOutput("Spawned.", [], {}, None, False)

  monkeypatch.setattr(autonomous, "run_session", run_parent)
  output = _run(autonomous.run_autonomous(
    "System", "Delegate a read.", **authority,
    skills_dir=skills_dir,
    capability_execution_resolver=resolver,
    user_id="alice", billing_mode="byok", rate_table_version="unknown",
  ))
  assert output.response == "Spawned."
  assert execution.resolver is None


@pytest.mark.parametrize("use_factory", [False, True])
def test_run_autonomous_delegated_code_handler_keeps_its_executable_route(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  use_factory: bool,
  owner_session_host_policy: Any,
) -> None:
  import json

  from agent_gateway import ToolDispatcher
  from agent_gateway.code_execution import (
    CodeExecutionBundle, CodeExecutionConfig, build_code_execution,
    cleanup_code_execution,
  )
  from agent_gateway.skills import SkillLoader
  from agent_gateway.task_registry import TaskRegistry
  from gateway_test_support.capability_execution_test_support import stub_capability_execution_resolver
  owner_session_host_policy.get_local_tool_effect = {
    "code_execute": "read", "parent_read": "read",
  }.get
  owner_session_host_policy.get_read_capability = lambda name, _server: (
    "computation.execute/v1" if name == "code_execute" else None
  )

  captured: dict[str, Any] = {}
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir()
  (skills_dir / "kernel-child.md").write_text(
    "---\nexecution_class: node.implement\nname: kernel-child\nversion: '1.0'\nagent_callable: true\n"
    "agent_description: Compute.\nmutation_mode: read_only\n"
    "allowed_tools: [code_execute]\nmcp_tools: {}\n"
    "semantic_metadata:\n  required_context: []\n"
    "  tool_refs:\n    - kind: local\n      tool_id: code_execute\n"
    "  capability_requirements:\n    - name: computation.execute/v1\n"
    "      required: true\n      binding_modes: [live_tool]\n"
    "---\nCompute and return the result.\n", encoding="utf-8",
  )
  loader = SkillLoader(skills_dir)
  operation = next(
    item.snapshot.operation.model_dump(mode="json")
    for item in loader.list_callable_operations()
    if item.snapshot.operation.name == "kernel-child"
  )
  resolver = stub_capability_execution_resolver(
    default_provider="openai", default_model="gpt-5.6-sol",
    run_mode="autonomous",
  )
  authority = _bound_execution()
  execution = resolver.resolve("session.driver")
  authority["capability_execution"] = replace(
    execution, auth_config={**execution.auth_config, "max_tokens": 16_000},
  )
  parent_bundle = build_code_execution(
    authority["session"],
    config=CodeExecutionConfig(register_docker=False, register_kernel=False),
  )

  async def caller_code_handler(_input: dict[str, Any], **_kwargs: Any):
    return {"stdout": "caller-owned\n"}, None

  parent_handlers = {
    **(parent_bundle.handlers if use_factory else {"code_execute": caller_code_handler}),
    "parent_read": caller_code_handler,
  }
  definitions = [
    *parent_bundle.tool_definitions,
    {"name": "parent_read", "description": "Parent-only read.",
     "input_schema": {"type": "object"}},
  ]

  def bind_code_execution(bundle: CodeExecutionBundle, dispatcher: ToolDispatcher) -> None:
    bundle.bind_dispatcher(
      dispatcher, tool_definitions=dispatcher.get_tool_definitions,
      code_callable=lambda name: name in parent_handlers,
    )

  def child_factory(session: GatewaySession, *, job_runner: Any):
    captured["child_session"] = session
    bundle = build_code_execution(session, config=CodeExecutionConfig(
      register_docker=False, register_kernel=False,
    ), job_runner=job_runner)
    captured["child_bundle"] = bundle
    return bundle

  class ParentRunner:
    def __init__(self, **kwargs: Any) -> None:
      self._dispatcher = kwargs["dispatcher"]
      self._containment = kwargs["containment"]
      self.capability_execution = kwargs["capability_execution"]
      self._get_tool_definitions = kwargs["get_tool_definitions"]
      self._full_session_id = authority["session"].session_id
      self._agent_session_log = object()
      self._spill_dir_provider = None
      self._log = kwargs["event_log"]
      self._task_registry = TaskRegistry()
      self.durable_events: list[dict[str, Any]] = []

    async def _append_durable_event(self, event: dict[str, Any]):
      self.durable_events.append(event)
      return object()

    async def _confirm_durable_skill_event(self, event: dict[str, Any]):
      return event

    async def spawn_sub_agent(self, task: str, **kwargs: Any):
      dispatcher = kwargs["dispatcher"]
      captured["child_definitions"] = dispatcher.get_tool_definitions()
      captured["code_result"] = await dispatcher.dispatch(
        "child-cell", "code_execute", {
          "host": "subprocess",
          "code": (
            "import json\n"
            "print(json.dumps({"
            "'catalog': [row['name'] for row in hank.search('code_execute')], "
            "'parent_reads': hank.search('parent_read')}))"
          ),
        },
      )
      # This fixture stops the provider loop after observing the actual route.
      return None, {"code": "cancelled", "message": "Fixture stopped after child dispatch."}

  async def run_parent(runner: Any, _log: EventLog, **_kwargs: Any):
    if use_factory:
      parent_dispatcher = ToolDispatcher(
        capability_bind=authority['capability_execution'].bind, mcp_client=autonomous._NullMcpClient(), local_tool_handlers=parent_handlers,
        role="owner", approval_key_qualifier=parent_bundle.approval_qualifier,
        get_tool_definitions=lambda: definitions,
      )
      bind_code_execution(parent_bundle, parent_dispatcher)
      result, error = await parent_dispatcher.dispatch(
        "parent-cell", "code_execute", {
          "host": "subprocess",
          "code": "print([row['name'] for row in hank.search('parent_read')])",
        },
      )
      assert error is None
      assert result is not None and result["return_code"] == 0, result
      assert result["stdout"] == "['parent_read']\n"
    _, error = await runner._dispatcher.dispatch("delegate", "run_agent", {
      "operation": operation, "objective": "Compute.", "background": False,
    })
    assert error is not None and error["code"] == "cancelled"
    return autonomous.RunOutput("Completed.", [], {}, None, False)

  monkeypatch.setattr(autonomous, "AgentRunner", ParentRunner)
  monkeypatch.setattr(autonomous, "run_session", run_parent)
  async def run() -> None:
    try:
      await autonomous.run_autonomous(
        "System", "Compute.", **authority,
        skills_dir=skills_dir,
        tool_handlers=parent_handlers,
        tool_definitions=definitions,
        code_execution_factory=child_factory if use_factory else None,
        code_execution_binder=bind_code_execution if use_factory else None,
        **_raw_capability_execution_resolver(capability_execution_resolver=resolver),
        user_id="alice", billing_mode="byok", rate_table_version="unknown",
      )
    finally:
      await cleanup_code_execution(authority["session"])

  _run(run())
  result, error = captured["code_result"]
  assert error is None
  if use_factory:
    assert result["return_code"] == 0, result
    assert json.loads(result["stdout"]) == {
      "catalog": ["code_execute"], "parent_reads": [],
    }
  else:
    assert result["stdout"] == "caller-owned\n"
  assert {definition["name"] for definition in captured["child_definitions"]} == {
    "code_execute",
  }
  if use_factory:
    assert captured["child_session"] is not authority["session"]
    assert (
      captured["child_bundle"].handlers["code_execute"]
      is not parent_bundle.handlers["code_execute"]
    )
    assert captured["child_bundle"].frame_reader is not parent_bundle.frame_reader
    assert captured["child_session"].code_execution_work_dir is None
