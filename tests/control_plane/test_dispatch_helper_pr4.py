from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import HTTPException

from agent_gateway.event_log import EventLog
from agent_gateway.server import ChatMessage, ChatRuntime, ChatTurnInputs, _dispatch_chat_turn
from agent_gateway.session import AuthManager, GatewaySession, SessionStore


def _run(coro):
  return asyncio.run(coro)


def _make_session(user_id: str = "alice") -> GatewaySession:
  store = SessionStore(ttl=3600)
  return store.create_session(
    api_key_hash=AuthManager.hash_api_key("gateway-key"),
    user_id=user_id,
    kind="chat",
  )


class _CompletingRunner:
  def __init__(self, event_log: EventLog, captured: dict[str, Any]) -> None:
    self._event_log = event_log
    self._captured = captured

  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    model_override: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    self._captured["messages"] = messages
    self._captured["system_prompt"] = system_prompt
    self._captured["model_override"] = model_override
    self._captured["max_turns"] = max_turns
    self._event_log.append({"type": "text_delta", "text": "ok"})
    self._event_log.append({"type": "stream_complete", "usage": {}})


class _FailingRunner:
  def __init__(self, event_log: EventLog) -> None:
    self._event_log = event_log

  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    model_override: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    _ = messages, system_prompt, model_override, max_turns
    raise RuntimeError("runner failed")


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
        build_runner=lambda event_log, _sid: _CompletingRunner(event_log, captured),
        model_override=request.model,
      )

    async def _on_event(event: dict[str, Any]) -> None:
      seen_events.append(dict(event))

    result = await _dispatch_chat_turn(
      session,
      ChatTurnInputs(
        messages=[ChatMessage(role="user", content="hello")],
        request_id=" req-1 ",
        context={"channel": "tui"},
        metadata={"document_context": {"source_id": "MSFT_10Q"}},
        model="test-model",
      ),
      event_log=event_log,
      on_event=_on_event,
      build_chat_runtime=_build_chat_runtime,
      credentials_resolver=None,
      transcript_dir=None,
    )

    assert result.session_id == session.session_id
    assert result.request_id == "req-1"
    assert result.state == "completed"
    assert captured["request_user_id"] == session.user_id
    assert captured["session_user_id"] == session.user_id
    assert captured["metadata"] == {"document_context": {"source_id": "MSFT_10Q"}}
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert [event["type"] for event in seen_events] == ["text_delta", "stream_complete"]
    assert [event["type"] for event in session.event_history.snapshot()] == ["text_delta", "stream_complete"]
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
      _ = session, request, channel, auth_manager
      return ChatRuntime(
        system_prompt="system",
        build_runner=lambda event_log, _sid: _FailingRunner(event_log),
      )

    async def _on_event(event: dict[str, Any]) -> None:
      seen_events.append(dict(event))

    result = await _dispatch_chat_turn(
      session,
      ChatTurnInputs(
        messages=[ChatMessage(role="user", content="fail")],
        request_id=None,
        context=None,
        metadata=None,
        model=None,
      ),
      event_log=event_log,
      on_event=_on_event,
      build_chat_runtime=_build_chat_runtime,
      credentials_resolver=None,
      transcript_dir=None,
    )

    assert result.state == "failed"
    assert result.events[-1] == {"type": "error", "error": "runner failed"}
    assert seen_events[-1] == {"type": "error", "error": "runner failed"}
    assert session.event_history.snapshot()[-1] == {"type": "error", "error": "runner failed"}
    assert session.stream_active is False

  _run(case())


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
          model=None,
        ),
        event_log=EventLog(),
        on_event=lambda _event: None,  # type: ignore[arg-type]
        build_chat_runtime=_build_chat_runtime,
        credentials_resolver=None,
        transcript_dir=None,
      )

    assert exc_info.value.status_code == 409
    assert session.stream_active is True

  _run(case())
