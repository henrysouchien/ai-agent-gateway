from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from pydantic import BaseModel

from agent_gateway.capability_binding import (
  CredentialHandle,
)
from agent_gateway.model_registry import (
  CAPABILITY_IDS,
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_gateway.event_log import EventLog, log_has_terminal
from agent_gateway.providers import AnthropicProvider
from agent_gateway.server import MaterializedCredential
from agent_gateway.control_run_lifecycle import CONTROL_RUN_STATE_CLASSIFICATION
from agent_gateway.server_chat_helpers import (
  _chat_turn_state_from_events,
  _dispatch_chat_turn,
  _latest_chat_run_state,
)
from agent_gateway.server_models import ChatMessage, ChatRequest, ChatRuntime, ChatTurnInputs
from agent_gateway.session import GatewaySession, SessionStream


_SERVICE_HANDLE = CredentialHandle(
  handle_id="service:trailing-capture-tests:anthropic",
  provider="anthropic",
  principal="service",
  tenant_id="trailing-capture-tests",
  actor_id=None,
)
_PROVIDER = AnthropicProvider()


def _materialize_service_credential(
  handle: CredentialHandle,
) -> MaterializedCredential:
  assert handle is _SERVICE_HANDLE
  return MaterializedCredential(
    handle=handle,
    auth_config={
      "provider": "anthropic",
      "api_key": "test-key",
      "billing_mode": "byok",
      "rate_table_version": "test-v1",
    },
  )


def _resolve_capability_provider(adapter: str) -> AnthropicProvider:
  if adapter != "anthropic.messages":
    raise RuntimeError(f"unknown test capability adapter: {adapter}")
  return _PROVIDER


def _run(coro):
  return asyncio.run(coro)


async def _collect(log: EventLog) -> list[dict[str, Any]]:
  return [dict(entry.event) async for entry in log.iter_from()]


def _session(*, kind: str = "chat", channel: str | None = None) -> GatewaySession:
  return GatewaySession(
    session_id="sess-trailing",
    api_key_hash="hash",
    created_at=1,
    expires_at=4_000_000_000,
    user_id="alice",
    kind=kind,  # type: ignore[arg-type]
    tenant_id=_SERVICE_HANDLE.tenant_id,
    allow_service_for_interactive=True,
    channel=channel,
    model_entitled_capabilities=CAPABILITY_IDS,
    model_entitled_keys=frozenset(INITIAL_MODEL_REGISTRY.models),
  )


def _inputs(*, profile: str | None = None) -> ChatTurnInputs:
  context = {} if profile is None else {"profile": profile}
  return ChatTurnInputs(
    messages=[ChatMessage(role="user", content="hello")],
    request_id="req-trailing",
    context=context,
    metadata={},
    model_key=None,
  )


async def _no_event(_event: Any) -> None:
  return None


def _runtime_builder(runner_factory):
  async def build_chat_runtime(
    session,
    request,
    channel,
    auth_manager,
  ) -> ChatRuntime:
    _ = (session, channel, auth_manager)
    capability_bind = request.capability_bind
    assert capability_bind is not None

    def build_bound_runner(event_log, session_id, _started_at):
      runner = runner_factory(event_log, session_id)
      runner.capability_execution = request.capability_execution
      return runner

    return ChatRuntime(
      system_prompt="test",
      build_runner=build_bound_runner,
      capability_execution=request.capability_execution,
    )

  setattr(
    build_chat_runtime,
    "_gateway_model_registry",
    INITIAL_MODEL_REGISTRY,
  )
  setattr(
    build_chat_runtime,
    "_gateway_model_selection_policy",
    INITIAL_MODEL_SELECTION_POLICY,
  )
  setattr(build_chat_runtime, "_gateway_tenant_id", _SERVICE_HANDLE.tenant_id)
  setattr(
    build_chat_runtime,
    "_gateway_service_provider_handles",
    {"anthropic": _SERVICE_HANDLE},
  )
  setattr(
    build_chat_runtime,
    "_gateway_service_auth_config_resolver",
    _materialize_service_credential,
  )
  setattr(
    build_chat_runtime,
    "_gateway_capability_adapter_resolver",
    _resolve_capability_provider,
  )
  setattr(build_chat_runtime, "_gateway_channel_profile_allowlist", None)

  return build_chat_runtime


class _AppendingRunner:
  def __init__(self, event_log: EventLog, events: list[dict[str, Any]]) -> None:
    self._event_log = event_log
    self._events = events

  async def run(self, **_kwargs: Any) -> None:
    for event in self._events:
      self._event_log.append(event)


class _FailingRunner:
  async def run(self, **_kwargs: Any) -> None:
    raise RuntimeError("runner failed")


class _DrainingRunner:
  def __init__(self, event_log: EventLog, terminal_emitted: asyncio.Event) -> None:
    self._event_log = event_log
    self._terminal_emitted = terminal_emitted

  async def run(self, **_kwargs: Any) -> None:
    self._event_log.append({
      "type": "stream_complete",
      "terminal_disposition": "completed",
      "usage": {},
    })
    self._terminal_emitted.set()
    await asyncio.Event().wait()


def test_deferred_event_log_accepts_and_yields_entries_after_terminal() -> None:
  async def case() -> None:
    event_log = EventLog(defer_terminal_close=True)
    consumer = asyncio.create_task(_collect(event_log))
    await asyncio.sleep(0)

    assert event_log.append({"type": "stream_complete"}) is not None
    assert event_log.has_terminal is True
    assert event_log.closed is False
    assert event_log.append({"type": "citation_validation", "violation_count": 0}) is not None
    await asyncio.sleep(0)
    assert consumer.done() is False

    event_log.close()
    events = await asyncio.wait_for(consumer, timeout=1.0)
    assert [event["type"] for event in events] == ["stream_complete", "citation_validation"]

  _run(case())


def test_deferred_event_log_close_without_terminal_synthesizes_error_and_ends_consumer() -> None:
  async def case() -> None:
    event_log = EventLog(defer_terminal_close=True)
    consumer = asyncio.create_task(_collect(event_log))
    await asyncio.sleep(0)

    event_log.close()

    events = await asyncio.wait_for(consumer, timeout=1.0)
    assert events == [{"type": "error", "error": "stream closed"}]
    assert event_log.has_terminal is True
    assert event_log.closed is True

  _run(case())


def test_default_event_log_keeps_terminal_close_behavior() -> None:
  async def case() -> None:
    seen: list[dict[str, Any]] = []
    event_log = EventLog(on_event=lambda event, _sid: seen.append(dict(event)))
    assert event_log.defer_terminal_close is False
    assert event_log.append({"type": "stream_complete"}) is not None
    assert event_log.closed is True
    assert event_log.has_terminal is True
    assert event_log.append({"type": "citation_validation"}) is None
    assert [entry.event async for entry in event_log.iter_from()] == [{"type": "stream_complete"}]
    event_log.close()
    assert seen == [{"type": "stream_complete"}]

  _run(case())


def test_log_has_terminal_falls_back_to_closed() -> None:
  assert log_has_terminal(SimpleNamespace(closed=True)) is True
  assert log_has_terminal(SimpleNamespace(closed=False)) is False


def test_stream_complete_disposition_closes_after_nonterminal_interruption() -> None:
  event_log = EventLog()

  interruption_entry = event_log.append({
    "type": "interrupted",
    "reason": "recovered_on_attach",
  })
  trailing_entry = event_log.append({
    "type": "text_delta",
    "text": "recovered",
  })
  terminal_entry = event_log.append({
    "type": "stream_complete",
    "terminal_disposition": "interrupted",
    "reason": "operator_pause",
    "usage": {},
  })

  assert interruption_entry is not None
  assert trailing_entry is not None
  assert terminal_entry is not None
  assert event_log.has_terminal is True
  assert event_log.closed is True
  assert event_log.append({"type": "text_delta", "text": "too late"}) is None


def test_deferred_dispatch_happy_path_closes_after_trailing_event(tmp_path) -> None:
  async def case() -> None:
    event_log = EventLog(defer_terminal_close=True)
    events = [
      {
        "type": "stream_complete",
        "terminal_disposition": "completed",
        "usage": {},
      },
      {"type": "citation_validation", "violation_count": 0},
    ]
    result = await _dispatch_chat_turn(
      _session(),
      _inputs(),
      event_log=event_log,
      on_event=_no_event,
      build_chat_runtime=_runtime_builder(
        lambda log, _sid: _AppendingRunner(log, events)
      ),
      transcript_dir=tmp_path,
    )

    event_types = [entry.event["type"] for entry in event_log.entries]
    assert event_types == ["stream_complete", "citation_validation"]
    assert event_types.count("stream_complete") + event_types.count("error") == 1
    assert all(entry.event.get("error") != "stream closed" for entry in event_log.entries)
    assert event_log.closed is True
    assert result.state == "completed"
    transcript = (tmp_path / "sess-trailing.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["type"] for line in transcript] == ["chat_request"]

  _run(case())


def test_deferred_dispatch_cancelled_mid_drain_keeps_first_terminal() -> None:
  async def case() -> None:
    event_log = EventLog(defer_terminal_close=True)
    terminal_emitted = asyncio.Event()
    task = asyncio.create_task(
      _dispatch_chat_turn(
        _session(),
        _inputs(),
        event_log=event_log,
        on_event=_no_event,
        build_chat_runtime=_runtime_builder(
          lambda log, _sid: _DrainingRunner(log, terminal_emitted)
        ),
        transcript_dir=None,
      )
    )
    await terminal_emitted.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await task

    terminal_events = [
      entry.event for entry in event_log.entries if entry.event.get("type") in {"stream_complete", "error"}
    ]
    assert terminal_events == [{
      "type": "stream_complete",
      "terminal_disposition": "completed",
      "usage": {},
    }]
    assert event_log.closed is True

  _run(case())


def test_deferred_dispatch_runner_exception_emits_exactly_one_terminal() -> None:
  async def case() -> None:
    event_log = EventLog(defer_terminal_close=True)
    await _dispatch_chat_turn(
      _session(),
      _inputs(),
      event_log=event_log,
      on_event=_no_event,
      build_chat_runtime=_runtime_builder(lambda _log, _sid: _FailingRunner()),
      transcript_dir=None,
    )

    terminal_events = [
      entry.event for entry in event_log.entries if entry.event.get("type") in {"stream_complete", "error"}
    ]
    assert terminal_events == [{"type": "error", "error": "runner failed"}]
    assert event_log.closed is True

  _run(case())


@pytest.mark.parametrize(
  ("rejection", "status_code"),
  [
    ("session-kind", 400),
    ("turn-busy", 409),
    ("stream-active", 409),
    ("invalid-profile", 403),
  ],
)
@pytest.mark.parametrize("deferred", [True, False])
def test_dispatch_pre_lifecycle_rejections_preserve_scoped_close(
  rejection: str,
  status_code: int,
  deferred: bool,
) -> None:
  async def case() -> None:
    callbacks: list[dict[str, Any]] = []
    event_log = EventLog(
      defer_terminal_close=deferred,
      on_event=lambda event, _sid: callbacks.append(dict(event)),
    )
    session = _session(kind="control" if rejection == "session-kind" else "chat")
    blocker: asyncio.Task[Any] | None = None
    if rejection == "turn-busy":
      blocker = asyncio.create_task(asyncio.Event().wait())
      session.active_turn = SessionStream(event_log=EventLog(), runner_task=blocker)
    elif rejection == "stream-active":
      session.stream_active = True
    elif rejection == "invalid-profile":
      session.channel = "excel"

    build_runtime = _runtime_builder(lambda _log, _sid: _FailingRunner())
    if rejection == "invalid-profile":
      setattr(build_runtime, "_gateway_channel_profile_allowlist", {"excel": frozenset({"community"})})

    consumer = asyncio.create_task(_collect(event_log))
    await asyncio.sleep(0)
    try:
      with pytest.raises(HTTPException) as exc_info:
        await _dispatch_chat_turn(
          session,
          _inputs(profile="analyst"),
          event_log=event_log,
          on_event=_no_event,
          build_chat_runtime=build_runtime,
          transcript_dir=None,
        )
      assert exc_info.value.status_code == status_code

      if deferred:
        events = await asyncio.wait_for(consumer, timeout=1.0)
        assert events == [{"type": "error", "error": "stream closed"}]
        assert event_log.closed is True
        assert callbacks == events
      else:
        await asyncio.sleep(0)
        assert consumer.done() is False
        assert event_log.closed is False
        assert event_log.entries == []
        assert callbacks == []
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
    finally:
      if blocker is not None:
        blocker.cancel()
        await asyncio.gather(blocker, return_exceptions=True)

  _run(case())


def test_chat_turn_state_uses_terminal_before_trailing_event() -> None:
  events = [
    {"type": "text_delta", "text": "done"},
    {"type": "stream_complete", "terminal_disposition": "completed"},
    {"type": "citation_validation", "violation_count": 0},
  ]
  assert _chat_turn_state_from_events(events) == "completed"


@pytest.mark.parametrize("state", CONTROL_RUN_STATE_CLASSIFICATION)
def test_latest_chat_run_state_recognizes_every_canonical_owner_state(
  state: str,
) -> None:
  session = _session()
  session.event_history.append({"type": "run_state_changed", "state": state})

  assert _latest_chat_run_state(session, session.session_id) == state


def test_latest_chat_run_state_canonicalizes_internal_cancel_alias() -> None:
  session = _session()
  session.event_history.append({"type": "run_state_changed", "state": "killed"})

  assert _latest_chat_run_state(session, session.session_id) == "cancelled"


def test_chat_turn_state_rejects_missing_terminal_disposition() -> None:
  assert _chat_turn_state_from_events([{"type": "stream_complete"}]) == "failed"


def test_chat_turn_state_projects_interrupted_terminal_with_trailing_event() -> None:
  events = [
    {"type": "operator_pause", "safe_boundary": "before_turn"},
    {
      "type": "stream_complete",
      "terminal_disposition": "interrupted",
      "reason": "operator_pause",
    },
    {"type": "citation_validation", "violation_count": 0},
  ]

  assert _chat_turn_state_from_events(events) == "interrupted"


def test_chat_turn_state_uses_first_terminal_error_before_trailing_success() -> None:
  events = [
    {"type": "text_delta", "text": "partial"},
    {"type": "error", "error": "provider failed"},
    {
      "type": "stream_complete",
      "terminal_disposition": "completed",
      "usage": {},
    },
  ]

  assert _chat_turn_state_from_events(events) == "failed"


def test_chat_request_drain_trailing_compatibility() -> None:
  class LegacyChatRequest(BaseModel):
    messages: list[dict[str, str]]

  new_payload = {
    "messages": [{"role": "user", "content": "hello"}],
    "drain_trailing": True,
  }
  assert LegacyChatRequest.model_validate(new_payload).messages == new_payload["messages"]
  assert ChatRequest.model_validate({"messages": new_payload["messages"]}).drain_trailing is False
