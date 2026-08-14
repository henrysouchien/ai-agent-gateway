from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from fastapi import HTTPException

from agent_gateway.capability_binding import CredentialHandle
from agent_gateway.model_registry import (
  CAPABILITY_IDS,
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_gateway.event_log import EventLog
from agent_gateway.providers import AnthropicProvider
from agent_gateway.runner_introspection import (
  exception_traceback_already_logged,
  mark_exception_traceback_logged,
)
from agent_gateway.server import (
  ChatMessage,
  ChatRuntime,
  ChatTurnInputs,
  MaterializedCredential,
  _dispatch_chat_turn,
)
from agent_gateway.session import AuthManager, GatewaySession, SessionStore


_SERVICE_HANDLE = CredentialHandle(
  handle_id="service:dispatch-helper-pr4:anthropic",
  provider="anthropic",
  principal="service",
  tenant_id="dispatch-helper-pr4",
  actor_id=None,
)
_SERVICE_MATERIAL = MaterializedCredential(
  handle=_SERVICE_HANDLE,
  auth_config={
    "provider": "anthropic",
    "api_key": "test-key",
    "billing_mode": "byok",
    "rate_table_version": "test",
  },
)


def _materialize_service_credential(
  handle: CredentialHandle,
) -> MaterializedCredential:
  if handle is not _SERVICE_HANDLE:
    raise RuntimeError("unknown test credential handle")
  return _SERVICE_MATERIAL


_PROVIDER = AnthropicProvider()


def _resolve_capability_adapter(adapter: str) -> AnthropicProvider:
  if adapter != "anthropic.messages":
    raise RuntimeError(f"unknown test capability adapter: {adapter}")
  return _PROVIDER


def _configure_runtime_builder(builder):
  setattr(builder, "_gateway_model_registry", INITIAL_MODEL_REGISTRY)
  setattr(
    builder,
    "_gateway_model_selection_policy",
    INITIAL_MODEL_SELECTION_POLICY,
  )
  setattr(
    builder,
    "_gateway_service_provider_handles",
    {"anthropic": _SERVICE_HANDLE},
  )
  setattr(
    builder,
    "_gateway_service_auth_config_resolver",
    _materialize_service_credential,
  )
  setattr(
    builder,
    "_gateway_capability_adapter_resolver",
    _resolve_capability_adapter,
  )
  setattr(builder, "_gateway_tenant_id", "dispatch-helper-pr4")
  return builder


def _run(coro):
  return asyncio.run(coro)


def _make_session(user_id: str = "alice") -> GatewaySession:
  store = SessionStore(ttl=3600)
  return store.create_session(
    api_key_hash=AuthManager.hash_api_key("gateway-key"),
    user_id=user_id,
    kind="chat",
    tenant_id="dispatch-helper-pr4",
    allow_service_for_interactive=True,
    model_entitled_capabilities=CAPABILITY_IDS,
    model_entitled_keys=frozenset(INITIAL_MODEL_REGISTRY.models),
  )


class _CompletingRunner:
  def __init__(
    self,
    event_log: EventLog,
    captured: dict[str, Any],
    capability_execution: Any,
  ) -> None:
    self._event_log = event_log
    self._captured = captured
    self.capability_execution = capability_execution

  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    self._captured["messages"] = messages
    self._captured["system_prompt"] = system_prompt
    self._captured["max_turns"] = max_turns
    self._event_log.append({"type": "text_delta", "text": "ok"})
    self._event_log.append({
      "type": "stream_complete",
      "terminal_disposition": "completed",
      "usage": {},
    })


class _FailingRunner:
  def __init__(self, event_log: EventLog, capability_execution: Any) -> None:
    self._event_log = event_log
    self.capability_execution = capability_execution

  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    _ = messages, system_prompt, max_turns
    raise RuntimeError("runner failed")


class _MarkedFailingRunner(_FailingRunner):
  def __init__(self, event_log: EventLog, capability_execution: Any) -> None:
    super().__init__(event_log, capability_execution)
    self._error = RuntimeError("runner failed")
    mark_exception_traceback_logged(self._error)

  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    _ = messages, system_prompt, max_turns
    raise self._error


class _SilentRunner:
  def __init__(self, event_log: EventLog, capability_execution: Any) -> None:
    self._event_log = event_log
    self.capability_execution = capability_execution

  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    _ = messages, system_prompt, max_turns
    self._event_log.append({"type": "text_delta", "text": "partial"})


async def _dispatch_with_runner(runner_type: type[Any]):
  session = _make_session()
  event_log = EventLog()

  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, channel, auth_manager
    return ChatRuntime(
      system_prompt="system",
      build_runner=lambda event_log, _sid, _started_at: runner_type(
        event_log,
        request.capability_execution,
      ),
      capability_execution=request.capability_execution,
    )
  _configure_runtime_builder(_build_chat_runtime)

  async def _on_event(_event: dict[str, Any]) -> None:
    return None

  return await _dispatch_chat_turn(
    session,
    ChatTurnInputs(
      messages=[ChatMessage(role="user", content="test")],
      request_id=None,
      context=None,
      metadata=None,
      model_key=None,
    ),
    event_log=event_log,
    on_event=_on_event,
    build_chat_runtime=_build_chat_runtime,
    transcript_dir=None,
  )


def test_dispatch_chat_turn_runs_synchronously_and_fans_out_events() -> None:
  async def case() -> None:
    session = _make_session()
    event_log = EventLog()
    captured: dict[str, Any] = {}
    seen_events: list[dict[str, Any]] = []

    async def _build_chat_runtime(*, session, request, channel, auth_manager):
      _ = channel, auth_manager
      captured["request_user_id"] = request.user_id
      captured["session_user_id"] = session.user_id
      captured["metadata"] = dict(request.metadata)
      return ChatRuntime(
        system_prompt="system",
        build_runner=lambda event_log, _sid, _started_at: _CompletingRunner(
          event_log,
          captured,
          request.capability_execution,
        ),
        capability_execution=request.capability_execution,
      )
    _configure_runtime_builder(_build_chat_runtime)

    async def _on_event(event: dict[str, Any]) -> None:
      seen_events.append(dict(event))

    result = await _dispatch_chat_turn(
      session,
      ChatTurnInputs(
        messages=[ChatMessage(role="user", content="hello")],
        request_id=" req-1 ",
        context={"channel": "tui"},
        metadata={"document_context": {"source_id": "MSFT_10Q"}},
        model_key=None,
      ),
      event_log=event_log,
      on_event=_on_event,
      build_chat_runtime=_build_chat_runtime,
      transcript_dir=None,
    )

    assert result.session_id == session.session_id
    assert result.request_id == "req-1"
    assert result.state == "completed"
    assert captured["request_user_id"] == session.user_id
    assert captured["session_user_id"] == session.user_id
    assert captured["metadata"] == {"document_context": {"source_id": "MSFT_10Q"}}
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert [event["type"] for event in seen_events] == [
      "capability_bound",
      "text_delta",
      "stream_complete",
    ]
    assert [event["type"] for event in session.event_history.snapshot()] == [
      "capability_bound",
      "text_delta",
      "stream_complete",
    ]
    assert event_log.closed is True
    assert session.stream_active is False
    assert session.initial_message == "hello"

  _run(case())


def test_dispatch_chat_turn_clears_stream_active_after_runner_failure() -> None:
  async def case() -> None:
    session = _make_session()
    event_log = EventLog()
    seen_events: list[dict[str, Any]] = []

    async def _build_chat_runtime(*, session, request, channel, auth_manager):
      _ = session, channel, auth_manager
      return ChatRuntime(
        system_prompt="system",
        build_runner=lambda event_log, _sid, _started_at: _FailingRunner(
          event_log,
          request.capability_execution,
        ),
        capability_execution=request.capability_execution,
      )
    _configure_runtime_builder(_build_chat_runtime)

    async def _on_event(event: dict[str, Any]) -> None:
      seen_events.append(dict(event))

    result = await _dispatch_chat_turn(
      session,
      ChatTurnInputs(
        messages=[ChatMessage(role="user", content="fail")],
        request_id=None,
        context=None,
        metadata=None,
        model_key=None,
      ),
      event_log=event_log,
      on_event=_on_event,
      build_chat_runtime=_build_chat_runtime,
      transcript_dir=None,
    )

    assert result.state == "failed"
    assert result.events[-1] == {"type": "error", "error": "runner failed"}
    assert seen_events[-1] == {"type": "error", "error": "runner failed"}
    assert session.event_history.snapshot()[-1] == {"type": "error", "error": "runner failed"}
    assert session.stream_active is False

  _run(case())


def test_dispatch_chat_turn_logs_traceback_when_runner_fails(caplog) -> None:
  caplog.set_level(logging.ERROR, logger="agent_gateway.server")

  result = _run(_dispatch_with_runner(_FailingRunner))

  records = [
    record
    for record in caplog.records
    if record.name == "agent_gateway.server"
    and "chat runner failed" in record.getMessage()
  ]
  assert len(records) == 1
  assert records[0].exc_info
  assert records[0].exc_info[0] is RuntimeError
  assert "Traceback (most recent call last)" in caplog.text
  assert "runner failed" in caplog.text
  assert result.events[-1] == {"type": "error", "error": "runner failed"}
  assert result.state == "failed"


def test_dispatch_chat_turn_does_not_duplicate_premarked_traceback(caplog) -> None:
  caplog.set_level(logging.ERROR, logger="agent_gateway.server")

  result = _run(_dispatch_with_runner(_MarkedFailingRunner))

  records = [
    record
    for record in caplog.records
    if record.name == "agent_gateway.server"
    and "chat runner failed" in record.getMessage()
  ]
  assert len(records) == 1
  assert not records[0].exc_info
  assert result.events[-1] == {"type": "error", "error": "runner failed"}
  assert result.state == "failed"


def test_exception_traceback_marker_helpers_are_best_effort() -> None:
  class _HostileSetattrError(RuntimeError):
    def __setattr__(self, name: str, value: Any) -> None:
      _ = name, value
      raise KeyboardInterrupt

  class _HostileGetattributeError(RuntimeError):
    def __getattribute__(self, name: str) -> Any:
      if name == "_gateway_traceback_logged":
        raise SystemExit
      return super().__getattribute__(name)

  hostile_setattr = _HostileSetattrError("hostile setattr")
  mark_exception_traceback_logged(hostile_setattr)
  assert exception_traceback_already_logged(hostile_setattr) is False

  hostile_getattribute = _HostileGetattributeError("hostile getattribute")
  assert exception_traceback_already_logged(hostile_getattribute) is False

  plain = RuntimeError("plain")
  assert exception_traceback_already_logged(plain) is False
  mark_exception_traceback_logged(plain)
  assert exception_traceback_already_logged(plain) is True


def test_dispatch_chat_turn_warns_when_runner_returns_without_terminal(caplog) -> None:
  caplog.set_level(logging.WARNING, logger="agent_gateway.server")

  result = _run(_dispatch_with_runner(_SilentRunner))

  records = [
    record
    for record in caplog.records
    if record.name == "agent_gateway.server"
    and "completed without a terminal event" in record.getMessage()
  ]
  assert len(records) == 1
  assert not records[0].exc_info
  assert result.events[-1] == {"type": "error", "error": "stream closed"}


def test_dispatch_chat_turn_rejects_concurrent_turn() -> None:
  async def case() -> None:
    session = _make_session()
    session.stream_active = True

    async def _build_chat_runtime(*, session, request, channel, auth_manager):
      _ = session, request, channel, auth_manager
      raise AssertionError("should not build runtime")

    with pytest.raises(HTTPException) as exc_info:
      await _dispatch_chat_turn(
        session,
        ChatTurnInputs(
          messages=[ChatMessage(role="user", content="hi")],
          request_id=None,
          context=None,
          metadata=None,
          model_key=None,
        ),
        event_log=EventLog(),
        on_event=lambda _event: None,  # type: ignore[arg-type]
        build_chat_runtime=_build_chat_runtime,
        transcript_dir=None,
      )

    assert exc_info.value.status_code == 409
    assert session.stream_active is True

  _run(case())
