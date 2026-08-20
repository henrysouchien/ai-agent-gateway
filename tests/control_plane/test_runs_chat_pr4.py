from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from types import SimpleNamespace
from typing import Any

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from agent_gateway.auth import AuthConfig, ResolverResult
from agent_gateway.capability_binding import CredentialHandle
from agent_gateway.model_registry import (
  CAPABILITY_IDS,
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_gateway.control_plane.runs_chat_helpers import _finalize_control_chat_task
from agent_gateway.control_run_lifecycle import (
  is_control_run_resumable_state,
  is_control_run_terminal_state,
)
from agent_gateway.event_log import EventLog
from agent_gateway.server import (
  ChatMessage,
  ChatRequest,
  ChatRuntime,
  GatewayServerConfig,
  MaterializedCredential,
  create_gateway_app,
)
from agent_gateway.session import AuthManager, GatewaySession


API_KEY = "runs-test-key"
_TEST_TENANT_ID = "control-runs-test"
_TEST_SERVICE_HANDLE = CredentialHandle(
  handle_id="control-runs-test-anthropic",
  provider="anthropic",
  principal="service",
  tenant_id=_TEST_TENANT_ID,
  actor_id=None,
)


def _test_gateway_config(**kwargs: Any) -> GatewayServerConfig:
  return GatewayServerConfig(
    tenant_id=_TEST_TENANT_ID,
    allow_service_credentials_for_interactive=True,
    model_registry=INITIAL_MODEL_REGISTRY,
    model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    service_provider_handles={"anthropic": _TEST_SERVICE_HANDLE},
    service_auth_config_resolver=lambda handle: MaterializedCredential(
      handle=handle,
      auth_config={
        "provider": "anthropic",
        "billing_mode": "byok",
        "api_key": "control-runs-test-secret",
      },
    ),
    **kwargs,
  )


class _EchoTurnRunner:
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
    _ = system_prompt, max_turns
    last_user = next((message["content"] for message in reversed(messages) if message.get("role") == "user"), "")
    self._event_log.append({"type": "text_delta", "text": f"echo:{last_user}"})
    self._event_log.append({
      "type": "stream_complete",
      "terminal_disposition": "completed",
      "usage": {},
    })



class _FailingTurnRunner:
  def __init__(self, capability_execution: Any) -> None:
    self.capability_execution = capability_execution

  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    _ = messages, system_prompt, max_turns
    raise RuntimeError("provider stream failed")


class _HangingTurnRunner:
  def __init__(self) -> None:
    self.started = threading.Event()
    self.cancelled = threading.Event()
    self.disconnected = threading.Event()

  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    _ = messages, system_prompt, max_turns
    self.started.set()
    try:
      await asyncio.Event().wait()
    except asyncio.CancelledError:
      self.cancelled.set()
      raise

  async def on_disconnect(self) -> None:
    self.disconnected.set()
    await asyncio.sleep(0)


class _StreamingHangingTurnRunner(_HangingTurnRunner):
  def __init__(self, event_log: EventLog) -> None:
    super().__init__()
    self._event_log = event_log

  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    _ = messages, system_prompt, max_turns
    self.started.set()
    self._event_log.append({"type": "text_delta", "text": "started"})
    try:
      await asyncio.Event().wait()
    except asyncio.CancelledError:
      self.cancelled.set()
      raise


def _make_app(
  captured_contexts: list[dict[str, Any]] | None = None,
  *,
  dispatch_scope_validator: Any | None = None,
  on_startup: Any | None = None,
  on_shutdown: Any | None = None,
  credentials_resolver: Any | None = None,
):
  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, channel, auth_manager
    if captured_contexts is not None:
      captured_contexts.append(dict(request.context))
    return ChatRuntime(
      system_prompt="system",
      build_runner=lambda event_log, _sid, _started_at: _EchoTurnRunner(
        event_log,
        request.capability_execution,
      ),
      capability_execution=request.capability_execution,
    )

  return create_gateway_app(
    _test_gateway_config(
      jwt_secret="runs-pr4-test-secret-0123456789x",
      valid_api_keys={API_KEY},
      build_chat_runtime=_build_chat_runtime,
      dispatch_scope_validator=dispatch_scope_validator,
      on_startup=on_startup,
      on_shutdown=on_shutdown,
      credentials_resolver=credentials_resolver,
    )
  )


def _make_capability_app(captured_requests: list[ChatRequest]):
  async def _credentials_resolver(_api_key: str, _request: Any) -> ResolverResult:
    return ResolverResult(
      user_id="alice",
      channel="tui",
      auth_config=AuthConfig.from_dict(
        {
          "provider": "anthropic",
          "billing_mode": "byok",
          "api_key": "test-user-secret",
        }
      ),
      credential_principal="user",
      risk_user_id=101,
      model_entitled_capabilities=CAPABILITY_IDS,
      model_entitled_keys=frozenset(INITIAL_MODEL_REGISTRY.models),
    )

  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, channel, auth_manager
    captured_requests.append(request)
    return ChatRuntime(
      system_prompt="system",
      build_runner=lambda event_log, _sid, _started_at: _EchoTurnRunner(
        event_log,
        request.capability_execution,
      ),
      capability_execution=request.capability_execution,
    )

  return create_gateway_app(
    GatewayServerConfig(
      jwt_secret="runs-pr4-capability-secret-0123456789",
      valid_api_keys={API_KEY},
      tenant_id="control-parity-test",
      credentials_resolver=_credentials_resolver,
      model_registry=INITIAL_MODEL_REGISTRY,
      model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
      build_chat_runtime=_build_chat_runtime,
    )
  )


def _make_failing_app():
  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, channel, auth_manager
    return ChatRuntime(
      system_prompt="system",
      build_runner=lambda _event_log, _sid, _started_at: _FailingTurnRunner(
        request.capability_execution
      ),
      capability_execution=request.capability_execution,
    )

  return create_gateway_app(
    _test_gateway_config(
      jwt_secret="runs-pr4-test-secret-0123456789x",
      valid_api_keys={API_KEY},
      build_chat_runtime=_build_chat_runtime,
    )
  )


def _make_setup_failing_app():
  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, request, channel, auth_manager
    raise RuntimeError("runtime setup failed")

  return create_gateway_app(
    _test_gateway_config(
      jwt_secret="runs-pr4-test-secret-0123456789x",
      valid_api_keys={API_KEY},
      build_chat_runtime=_build_chat_runtime,
    )
  )


def _make_hanging_app() -> tuple[Any, _HangingTurnRunner]:
  runner = _HangingTurnRunner()

  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, channel, auth_manager
    runner.capability_execution = request.capability_execution
    return ChatRuntime(
      system_prompt="system",
      build_runner=lambda _event_log, _sid, _started_at: runner,
      disconnect_handler=runner.on_disconnect,
      capability_execution=request.capability_execution,
    )

  app = create_gateway_app(
    _test_gateway_config(
      jwt_secret="runs-pr4-test-secret-0123456789x",
      valid_api_keys={API_KEY},
      build_chat_runtime=_build_chat_runtime,
    )
  )
  return app, runner


def _make_streaming_hanging_app() -> tuple[Any, dict[str, _StreamingHangingTurnRunner]]:
  captured: dict[str, _StreamingHangingTurnRunner] = {}

  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, channel, auth_manager

    def _build_runner(
      event_log: EventLog,
      _sid: str,
      _started_at: float,
    ) -> _StreamingHangingTurnRunner:
      runner = _StreamingHangingTurnRunner(event_log)
      runner.capability_execution = request.capability_execution
      captured["runner"] = runner
      return runner

    return ChatRuntime(
      system_prompt="system",
      build_runner=_build_runner,
      disconnect_handler=lambda: captured["runner"].on_disconnect(),
      capability_execution=request.capability_execution,
    )

  app = create_gateway_app(
    _test_gateway_config(
      jwt_secret="runs-pr4-test-secret-0123456789x",
      valid_api_keys={API_KEY},
      build_chat_runtime=_build_chat_runtime,
    )
  )
  return app, captured


def _control_session(client: TestClient, user_id: str) -> dict[str, Any]:
  response = client.post(
    "/api/control/session",
    json={"api_key": API_KEY, "user_id": user_id, "context": {"channel": "tui"}},
  )
  assert response.status_code == 200, response.text
  payload = response.json()
  session = client.app.state.auth.session_store.get_session(payload["session_id"])
  assert session is not None
  session.tenant_id = session.tenant_id or _TEST_TENANT_ID
  session.allow_service_for_interactive = True
  session.model_entitled_capabilities = CAPABILITY_IDS
  session.model_entitled_keys = frozenset(INITIAL_MODEL_REGISTRY.models)
  payload["session_token"] = client.app.state.auth.issue_token(session)
  return payload


def _chat_session(client: TestClient, user_id: str, *, channel: str = "tui") -> dict[str, Any]:
  response = client.post(
    "/api/chat/init",
    json={"api_key": API_KEY, "user_id": user_id, "context": {"channel": channel}},
  )
  assert response.status_code == 200, response.text
  payload = response.json()
  session = client.app.state.auth.session_store.get_session(payload["session_id"])
  assert session is not None
  session.tenant_id = session.tenant_id or _TEST_TENANT_ID
  session.allow_service_for_interactive = True
  session.model_entitled_capabilities = CAPABILITY_IDS
  session.model_entitled_keys = frozenset(INITIAL_MODEL_REGISTRY.models)
  payload["session_token"] = client.app.state.auth.issue_token(session)
  return payload


def _headers(session: dict[str, Any]) -> dict[str, str]:
  return {"Authorization": f"Bearer {session['session_token']}"}


def test_control_chat_finalizer_logs_late_failure_traceback(caplog) -> None:
  caplog.set_level(
    logging.WARNING,
    logger="agent_gateway.control_plane.runs_chat_helpers",
  )

  async def case() -> None:
    async def fail() -> None:
      raise RuntimeError("late control dispatch failure")

    session = GatewaySession(
      session_id="control-chat-finalizer-test",
      api_key_hash="hash",
      created_at=1,
      expires_at=4_000_000_000,
      user_id="alice",
      kind="chat",
      tenant_id=_TEST_TENANT_ID,
      allow_service_for_interactive=True,
      channel="tui",
    )
    task = asyncio.create_task(fail())
    task_key = "control_chat_turn:test"
    session.control_chat_tasks[task_key] = task

    await _finalize_control_chat_task(
      task=task,
      session=session,
      app_state=SimpleNamespace(),
      task_key=task_key,
    )
    assert task_key not in session.control_chat_tasks

  asyncio.run(case())

  records = [
    record
    for record in caplog.records
    if record.name == "agent_gateway.control_plane.runs_chat_helpers"
    and "control chat turn dispatch failed" in record.getMessage()
  ]
  assert len(records) == 1
  assert records[0].exc_info
  assert records[0].exc_info[0] is RuntimeError
  assert "Traceback (most recent call last)" in caplog.text
  assert "late control dispatch failure" in caplog.text


def _wait_until(predicate, *, timeout: float = 1.0) -> bool:
  deadline = time.time() + timeout
  while time.time() < deadline:
    if predicate():
      return True
    time.sleep(0.01)
  return bool(predicate())


def _collect_chat_turn_events(client: TestClient, token: str, message: str) -> list[dict[str, Any]]:
  events: list[dict[str, Any]] = []
  with client.stream(
    "POST",
    "/api/chat",
    headers={"Authorization": f"Bearer {token}"},
    json={"messages": [{"role": "user", "content": message}], "context": {"channel": "tui"}},
  ) as response:
    assert response.status_code == 200, response.text
    for line in response.iter_lines():
      if not line:
        continue
      text = line.decode("utf-8") if isinstance(line, bytes) else line
      if text.startswith("data:"):
        text = text[5:].strip()
      events.append(_unwrap_sse_payload(json.loads(text)))
  return events


def _unwrap_sse_payload(payload: dict[str, Any]) -> dict[str, Any]:
  candidate = payload.get("event")
  if isinstance(payload.get("seq"), int) and isinstance(candidate, dict) and isinstance(candidate.get("type"), str):
    return candidate
  return payload


def test_control_chat_dispatch_reuses_immutable_credential_handle(
  client: TestClient,
  control_plane_app,
  test_control_session: dict[str, Any],
) -> None:
  control = control_plane_app.state.auth.session_store.get_session(
    test_control_session["session_id"]
  )
  assert control is not None
  assert control.session_credential_handle is not None

  response = client.post(
    "/api/control/runs",
    headers={"Authorization": f"Bearer {test_control_session['session_token']}"},
    json={
      "kind": "chat",
      "message": "Start a child chat.",
      "channel": "tui",
      "max_budget_usd": 5.0,
    },
  )

  assert response.status_code == 200, response.text
  assert response.headers["cache-control"] == "private, no-store"
  chat = control_plane_app.state.auth.session_store.get_session(
    response.json()["chat_session_id"]
  )
  assert chat is not None
  assert chat.session_credential_handle is control.session_credential_handle
  assert chat.tenant_id == control.tenant_id == "test-product"
  assert chat.allow_service_for_interactive is True
  assert chat.max_budget_usd == 5.0
  assert response.json()["run"]["max_budget_usd"] == 5.0


def test_control_chat_dispatch_rejects_product_dev_mode(
  client: TestClient,
  test_control_session: dict[str, Any],
) -> None:
  response = client.post(
    "/api/control/runs",
    headers={"Authorization": f"Bearer {test_control_session['session_token']}"},
    json={
      "kind": "chat",
      "message": "Start a developer chat.",
      "channel": "tui",
      "dev_mode": True,
    },
  )

  assert response.status_code == 422, response.text


def test_control_session_token_response_is_private_no_store(
  client: TestClient,
  test_api_key: str,
) -> None:
  response = client.post(
    "/api/control/session",
    json={
      "api_key": test_api_key,
      "context": {"channel": "tui"},
    },
  )

  assert response.status_code == 200, response.text
  assert "session_token" in response.json()
  assert response.headers["cache-control"] == "private, no-store"


def test_control_session_generic_resolver_exception_response_is_value_free() -> None:
  secret = "CUSTOM-ACTIVE-CREDENTIAL-control-session-error-8f21d7"

  async def resolver(_api_key: str, _payload: Any) -> ResolverResult:
    raise RuntimeError(f"resolver failed with {secret}")

  app = _make_app(credentials_resolver=resolver)
  with TestClient(app) as client:
    response = client.post(
      "/api/control/session",
      json={"api_key": API_KEY, "user_id": "alice"},
    )

  assert response.status_code == 500
  assert response.json() == {
    "error": "credentials_unavailable",
    "message": "Credential resolver unavailable",
    "user_id": "alice",
  }
  assert secret not in response.text


def test_control_chat_start_and_continuation_bind_requested_model_and_effort() -> None:
  captured_requests: list[ChatRequest] = []
  app = _make_capability_app(captured_requests)

  with TestClient(app) as client:
    control = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "chat",
        "message": "Start with the alternate model.",
        "channel": "tui",
        "model_key": "anthropic.claude-opus-5",
        "catalog_revision": INITIAL_MODEL_REGISTRY.revision,
        "effort": "LOW",
        "skill": "business-model-construction",
      },
    )
    assert start.status_code == 200, start.text

    continuation = client.post(
      f"/api/control/runs/{start.json()['chat_session_id']}/messages",
      headers=_headers(control),
      json={
        "messages": [{"role": "user", "content": "Continue with the default model."}],
        "model_key": "anthropic.claude-sonnet-5",
        "catalog_revision": INITIAL_MODEL_REGISTRY.revision,
        "effort": "none",
        "context": {
          "skill": "build-model",
          "stage_skill_route": {
            "route_kind": "stage",
            "skill_name": "build-model",
          },
        },
      },
    )

  assert continuation.status_code == 200, continuation.text
  assert len(captured_requests) == 2
  start_bind = captured_requests[0].capability_bind
  continuation_bind = captured_requests[1].capability_bind
  assert start_bind is not None
  assert continuation_bind is not None
  assert (start_bind.upstream_model, start_bind.effort) == ("claude-opus-5", "low")
  assert captured_requests[0].context["stage_skill_route"] == {
    "route_kind": "stage",
    "skill_name": "business-model-construction",
  }
  assert captured_requests[1].context["skill"] == "business-model-construction"
  assert captured_requests[1].context["stage_skill_route"] == {
    "route_kind": "stage",
    "skill_name": "business-model-construction",
  }
  assert (continuation_bind.upstream_model, continuation_bind.effort) == (
    "claude-sonnet-5",
    "none",
  )


def test_control_chat_rejects_invalid_effort_before_dispatch() -> None:
  captured_requests: list[ChatRequest] = []
  app = _make_capability_app(captured_requests)

  with TestClient(app) as client:
    control = _control_session(client, "alice")
    invalid_start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "chat",
        "message": "Do not dispatch.",
        "channel": "tui",
        "effort": "turbo",
      },
    )
    assert invalid_start.status_code == 422, invalid_start.text
    assert captured_requests == []

    start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "chat",
        "message": "Start normally.",
        "channel": "tui",
      },
    )
    assert start.status_code == 200, start.text
    target = app.state.auth.session_store.get_session(start.json()["chat_session_id"])
    assert target is not None

    invalid_continuation = client.post(
      f"/api/control/runs/{target.session_id}/messages",
      headers=_headers(control),
      json={
        "messages": [{"role": "user", "content": "Do not continue."}],
        "request_id": "invalid-effort",
        "effort": "turbo",
      },
    )

  assert invalid_continuation.status_code == 422, invalid_continuation.text
  assert len(captured_requests) == 1
  assert not any(
    event.get("type") == "parent_message_sent"
    and event.get("message_id") == "invalid-effort"
    for event in target.event_history.snapshot()
  )


def test_capability_refused_start_rolls_back_registered_chat_session() -> None:
  captured_requests: list[ChatRequest] = []
  app = _make_capability_app(captured_requests)

  with TestClient(app) as client:
    control = _control_session(client, "alice")
    sessions_before = set(app.state.auth.session_store.sessions)

    refused = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "chat",
        "message": "Do not retain this refused run.",
        "channel": "tui",
        "model_key": "openai.gpt-5-4-mini-sdk",
        "catalog_revision": INITIAL_MODEL_REGISTRY.revision,
        "effort": "none",
      },
    )

    assert refused.status_code == 400, refused.text
    assert refused.json()["detail"]["error_code"] == "capability_model_not_allowed"
    assert set(app.state.auth.session_store.sessions) == sessions_before
    assert captured_requests == []


def test_capability_refusal_does_not_poison_continuation_dedupe() -> None:
  captured_requests: list[ChatRequest] = []
  app = _make_capability_app(captured_requests)

  with TestClient(app) as client:
    control = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "chat",
        "message": "Start normally.",
        "channel": "tui",
      },
    )
    assert start.status_code == 200, start.text
    target = app.state.auth.session_store.get_session(start.json()["chat_session_id"])
    assert target is not None

    refused = client.post(
      f"/api/control/runs/{target.session_id}/messages",
      headers=_headers(control),
      json={
        "messages": [{"role": "user", "content": "Try an unapproved model."}],
        "request_id": "retry-after-refusal",
        "model_key": "openai.gpt-5-4-mini-sdk",
        "catalog_revision": INITIAL_MODEL_REGISTRY.revision,
        "effort": "none",
      },
    )

    assert refused.status_code == 400, refused.text
    assert refused.json()["detail"]["error_code"] == "capability_model_not_allowed"
    assert len(captured_requests) == 1
    assert not any(
      event.get("type") == "parent_message_sent"
      and event.get("message_id") == "retry-after-refusal"
      for event in target.event_history.snapshot()
    )

    retried = client.post(
      f"/api/control/runs/{target.session_id}/messages",
      headers=_headers(control),
      json={
        "messages": [{"role": "user", "content": "Retry with an approved model."}],
        "request_id": "retry-after-refusal",
        "model_key": "anthropic.claude-opus-5",
        "catalog_revision": INITIAL_MODEL_REGISTRY.revision,
        "effort": "high",
      },
    )

  assert retried.status_code == 200, retried.text
  assert retried.json()["delivery_status"] == "delivered"
  assert len(captured_requests) == 2
  parent_events = [
    event
    for event in target.event_history.snapshot()
    if event.get("type") == "parent_message_sent"
    and event.get("message_id") == "retry-after-refusal"
  ]
  assert len(parent_events) == 1
  retried_bind = captured_requests[-1].capability_bind
  assert retried_bind is not None
  assert (retried_bind.upstream_model, retried_bind.effort) == (
    "claude-opus-5",
    "high",
  )


def _consume_chat_turn(client: TestClient, token: str, message: str) -> None:
  _collect_chat_turn_events(client, token, message)


def test_chat_stream_body_iterator_close_keeps_dispatch_task_until_session_expiry() -> None:
  async def case() -> None:
    app, captured = _make_streaming_hanging_app()
    auth = app.state.auth
    session = auth.session_store.create_session(
      AuthManager.hash_api_key(API_KEY),
      user_id="alice",
      kind="chat",
      tenant_id=_TEST_TENANT_ID,
      allow_service_for_interactive=True,
      model_entitled_capabilities=CAPABILITY_IDS,
      model_entitled_keys=frozenset(INITIAL_MODEL_REGISTRY.models),
    )
    token = auth.issue_token(session)
    route = next(route for route in app.routes if getattr(route, "path", None) == "/api/chat")
    request = Request(
      {
        "type": "http",
        "method": "POST",
        "path": "/api/chat",
        "headers": [(b"authorization", f"Bearer {token}".encode("utf-8"))],
        "query_string": b"",
        "app": app,
      }
    )
    response = await route.endpoint(
      request,
      body=ChatRequest(
        messages=[ChatMessage(role="user", content="stream then disconnect")],
        context={"channel": "tui"},
      ),
    )

    chunk = await asyncio.wait_for(response.body_iterator.__anext__(), timeout=0.5)
    assert b"text_delta" in chunk
    close = getattr(response.body_iterator, "aclose", None)
    assert callable(close)
    await close()

    await asyncio.sleep(0)
    running = [
      task
      for task in asyncio.all_tasks()
      if task is not asyncio.current_task()
      and not task.done()
      and "_dispatch_chat_turn" in getattr(task.get_coro(), "__qualname__", repr(task.get_coro()))
    ]
    try:
      assert running
      runner = captured["runner"]
      assert not runner.disconnected.is_set()
      assert not runner.cancelled.is_set()
      assert session.stream_active is True
      assert session.active_turn is not None
      assert session.active_turn.is_running
      assert session.active_turn.subscribers == {}

      await auth.session_store.expire_session_async(session.session_id)
      await asyncio.sleep(0)
      leaked = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and "_dispatch_chat_turn" in getattr(task.get_coro(), "__qualname__", repr(task.get_coro()))
      ]
      assert leaked == []
      assert runner.disconnected.is_set()
      assert runner.cancelled.is_set()
      assert session.stream_active is False
      assert session.active_turn is None
    finally:
      for task in running:
        task.cancel()
      if running:
        await asyncio.gather(*running, return_exceptions=True)
      await app.state.user_event_bus.shutdown()

  asyncio.run(case())


def test_list_runs_returns_user_chat_sessions_and_excludes_control_sessions() -> None:
  app = _make_app()
  with TestClient(app) as client:
    alice_control = _control_session(client, "alice")
    alice_empty_chat = _chat_session(client, "alice")
    alice_chat = _chat_session(client, "alice")
    bob_chat = _chat_session(client, "bob")

    alice_session = app.state.auth.session_store.get_session(alice_chat["session_id"])
    alice_empty_session = app.state.auth.session_store.get_session(alice_empty_chat["session_id"])
    bob_session = app.state.auth.session_store.get_session(bob_chat["session_id"])
    assert alice_session is not None
    assert alice_empty_session is not None
    assert bob_session is not None
    alice_session.initial_message = "hello alice"
    alice_session.event_history.append({
      "type": "stream_complete",
      "terminal_disposition": "completed",
      "usage": {},
    })
    bob_session.event_history.append({
      "type": "stream_complete",
      "terminal_disposition": "completed",
      "usage": {},
    })

    response = client.get("/api/control/runs?kind=chat", headers=_headers(alice_control))
    assert response.status_code == 200, response.text
    runs = response.json()["runs"]

    assert [run["run_id"] for run in runs] == [alice_chat["session_id"]]
    assert runs[0]["kind"] == "chat"
    assert runs[0]["session_id"] == alice_chat["session_id"]
    assert runs[0]["user_id"] == "alice"
    assert runs[0]["state"] == "completed"
    assert runs[0]["initial_message"] == "hello alice"

    control_filtered = client.get("/api/control/runs?kind=control", headers=_headers(alice_control))
    assert control_filtered.status_code == 200
    assert control_filtered.json() == {"runs": []}

    state_filtered = client.get("/api/control/runs?state=running", headers=_headers(alice_control))
    assert state_filtered.status_code == 200
    assert state_filtered.json() == {"runs": []}

    empty_detail = client.get(f"/api/control/runs/{alice_empty_chat['session_id']}", headers=_headers(alice_control))
    empty_logs = client.get(
      f"/api/control/runs/{alice_empty_chat['session_id']}/logs",
      headers=_headers(alice_control),
    )
    empty_message = client.post(
      f"/api/control/runs/{alice_empty_chat['session_id']}/messages",
      headers=_headers(alice_control),
      json={"messages": [{"role": "user", "content": "continue blank session"}]},
    )

    assert empty_detail.status_code == 404
    assert empty_logs.status_code == 404
    assert empty_message.status_code == 404


def test_chat_interrupted_terminal_projects_interrupted_control_state() -> None:
  app = _make_app()
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")
    session = app.state.auth.session_store.get_session(
      chat["session_id"]
    )
    assert session is not None
    session.event_history.append({
      "type": "operator_pause",
      "safe_boundary": "before_turn",
    })
    session.event_history.append({
      "type": "stream_complete",
      "terminal_disposition": "interrupted",
      "reason": "operator_pause",
      "usage": {},
    })

    response = client.get(
      f"/api/control/runs/{chat['session_id']}",
      headers=_headers(control),
    )

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "interrupted"


def test_chat_budget_stop_projects_budget_limited_control_state() -> None:
  app = _make_app()
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")
    session = app.state.auth.session_store.get_session(chat["session_id"])
    assert session is not None
    session.event_history.append({
      "type": "budget_exceeded",
      "total_cost": 0.42,
      "budget": 0.4,
      "reason": "sdk_max_budget_usd",
    })
    session.event_history.append({
      "type": "stream_complete",
      "terminal_disposition": "interrupted",
      "reason": "budget_exceeded",
      "usage": {},
    })

    response = client.get(
      f"/api/control/runs/{chat['session_id']}",
      headers=_headers(control),
    )
    logs = client.get(
      f"/api/control/runs/{chat['session_id']}/logs",
      headers=_headers(control),
    )

    assert response.status_code == 200, response.text
    state = response.json()["state"]
    assert state == "budget_limited"
    assert is_control_run_terminal_state(state) is True
    assert is_control_run_resumable_state(state) is False
    # The wire vocabulary is frozen: only the recorded run state is remapped.
    assert logs.status_code == 200, logs.text
    logged = [json.loads(line) for line in logs.json()["log_lines"]]
    assert [event["type"] for event in logged][-2:] == [
      "budget_exceeded",
      "stream_complete",
    ]
    assert logged[-1]["reason"] == "budget_exceeded"
    assert "budget_limited" not in logs.text


def test_chat_interrupted_terminal_without_budget_reason_stays_interrupted() -> None:
  app = _make_app()
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")
    session = app.state.auth.session_store.get_session(chat["session_id"])
    assert session is not None
    session.event_history.append({
      "type": "stream_complete",
      "terminal_disposition": "interrupted",
      "usage": {},
    })

    response = client.get(
      f"/api/control/runs/{chat['session_id']}",
      headers=_headers(control),
    )

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "interrupted"


def test_chat_budget_stop_with_channel_error_projects_failed() -> None:
  app = _make_app()
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")
    session = app.state.auth.session_store.get_session(chat["session_id"])
    assert session is not None
    for event in [
      {
        "type": "budget_exceeded",
        "total_cost": 0.42,
        "budget": 0.4,
        "reason": "sdk_max_budget_usd",
      },
      {"type": "stream_error", "error": "control event channel corrupted"},
      {
        "type": "stream_complete",
        "terminal_disposition": "interrupted",
        "reason": "budget_exceeded",
        "usage": {},
      },
    ]:
      session.event_history.append(event)

    response = client.get(
      f"/api/control/runs/{chat['session_id']}",
      headers=_headers(control),
    )

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "failed"


def test_chat_first_terminal_error_wins_over_trailing_completed_event() -> None:
  app = _make_app()
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")
    session = app.state.auth.session_store.get_session(
      chat["session_id"]
    )
    assert session is not None
    for event in [
      {"type": "text_delta", "text": "partial"},
      {"type": "error", "error": "provider failed"},
      {
        "type": "stream_complete",
        "terminal_disposition": "completed",
        "usage": {},
      },
    ]:
      session.event_history.append(event)

    response = client.get(
      f"/api/control/runs/{chat['session_id']}",
      headers=_headers(control),
    )

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "failed"


def test_chat_missing_terminal_disposition_projects_failed_control_state() -> None:
  app = _make_app()
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")
    session = app.state.auth.session_store.get_session(chat["session_id"])
    assert session is not None
    session.event_history.append({"type": "stream_complete", "usage": {}})

    response = client.get(
      f"/api/control/runs/{chat['session_id']}",
      headers=_headers(control),
    )

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "failed"


def test_chat_dispatch_persists_redacted_dispatch_scope() -> None:
  captured_contexts: list[dict[str, Any]] = []
  app = _make_app(captured_contexts)
  dispatch_scope = {
    "kind": "portfolio",
    "source": "active_default",
    "portfolio_name": "core",
    "portfolio_id": None,
    "display_name": "Core Portfolio",
  }

  with TestClient(app) as client:
    control = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "chat",
        "message": "Start portfolio chat.",
        "channel": "tui",
        "context": {"purpose": "risk-review"},
        "dispatch_scope": dispatch_scope,
      },
    )

    assert start.status_code == 200, start.text
    start_payload = start.json()
    assert start_payload["run"]["dispatch_scope"] == dispatch_scope
    chat_session_id = start_payload["chat_session_id"]
    session = app.state.auth.session_store.get_session(chat_session_id)
    assert session is not None
    assert session.dispatch_scope == dispatch_scope

    detail = client.get(f"/api/control/runs/{chat_session_id}", headers=_headers(control))
    assert detail.status_code == 200, detail.text
    assert detail.json()["dispatch_scope"] == dispatch_scope
    listed = client.get("/api/control/runs?kind=chat", headers=_headers(control))
    assert listed.status_code == 200, listed.text
    assert listed.json()["runs"][0]["dispatch_scope"] == dispatch_scope

    follow_up = client.post(
      f"/api/control/runs/{chat_session_id}/messages",
      headers=_headers(control),
      json={
        "messages": [{"role": "user", "content": "Continue portfolio chat."}],
        "context": {"purpose": "follow-up"},
      },
    )
    assert follow_up.status_code == 200, follow_up.text
    assert follow_up.json()["run"]["dispatch_scope"] == dispatch_scope
    assert captured_contexts[-1]["purpose"] == "follow-up"
    assert captured_contexts[-1]["channel"] == "tui"
    assert captured_contexts[-1]["portfolio_name"] == "core"
    assert captured_contexts[-1]["dispatch_scope"] == dispatch_scope


def test_chat_dispatch_rejects_unknown_dispatch_scope_before_runtime() -> None:
  captured_contexts: list[dict[str, Any]] = []

  async def validator(_session, scope: dict[str, Any]) -> dict[str, Any]:
    assert scope["portfolio_name"] == "unknown"
    raise HTTPException(
      status_code=422,
      detail={
        "error": "dispatch_scope_portfolio_not_visible",
        "field": "dispatch_scope.portfolio_name",
      },
    )

  app = _make_app(captured_contexts, dispatch_scope_validator=validator)

  with TestClient(app) as client:
    control = _control_session(client, "alice")
    response = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "chat",
        "message": "Start portfolio chat.",
        "channel": "tui",
        "dispatch_scope": {
          "kind": "portfolio",
          "source": "user_selected",
          "portfolio_name": "unknown",
        },
      },
    )

  assert response.status_code == 422, response.text
  assert response.json()["detail"]["error"] == "dispatch_scope_portfolio_not_visible"
  assert captured_contexts == []


def test_chat_dispatch_uses_canonicalized_dispatch_scope() -> None:
  captured_contexts: list[dict[str, Any]] = []

  def validator(_session, scope: dict[str, Any]) -> dict[str, Any]:
    return {
      **scope,
      "portfolio_id": "portfolio-1",
      "display_name": "Canonical Portfolio",
    }

  app = _make_app(captured_contexts, dispatch_scope_validator=validator)

  with TestClient(app) as client:
    control = _control_session(client, "alice")
    response = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "chat",
        "message": "Start portfolio chat.",
        "channel": "tui",
        "dispatch_scope": {
          "kind": "portfolio",
          "source": "user_selected",
          "portfolio_name": "taxable_combined",
          "display_name": "Client Portfolio",
        },
      },
    )

  assert response.status_code == 200, response.text
  canonical_scope = {
    "kind": "portfolio",
    "source": "user_selected",
    "portfolio_name": "taxable_combined",
    "portfolio_id": "portfolio-1",
    "display_name": "Canonical Portfolio",
  }
  assert response.json()["run"]["dispatch_scope"] == canonical_scope
  assert captured_contexts[-1]["dispatch_scope"] == canonical_scope
  assert captured_contexts[-1]["portfolio_name"] == "taxable_combined"


def test_chat_dispatch_rejects_context_authority_fields() -> None:
  app = _make_app()

  with TestClient(app) as client:
    control = _control_session(client, "alice")
    response = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "chat",
        "message": "Start portfolio chat.",
        "channel": "tui",
        "context": {"nested": {"portfolioName": "spoofed"}},
      },
    )

  assert response.status_code == 422, response.text


def test_chat_continuation_rejects_context_authority_fields() -> None:
  app = _make_app()

  with TestClient(app) as client:
    control = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "chat",
        "message": "Start chat.",
        "channel": "tui",
      },
    )
    assert start.status_code == 200, start.text
    chat_session_id = start.json()["chat_session_id"]

    response = client.post(
      f"/api/control/runs/{chat_session_id}/messages",
      headers=_headers(control),
      json={
        "messages": [{"role": "user", "content": "Continue."}],
        "context": {"dispatchScope": {"portfolio_name": "spoofed"}},
      },
    )

  assert response.status_code == 422, response.text


def test_get_run_returns_chat_run_shape_and_404s_unknown_or_cross_user() -> None:
  app = _make_app()
  with TestClient(app) as client:
    alice_control = _control_session(client, "alice")
    alice_chat = _chat_session(client, "alice")
    bob_chat = _chat_session(client, "bob")

    session = app.state.auth.session_store.get_session(alice_chat["session_id"])
    assert session is not None
    session.channel = "tui"
    session.initial_message = "first"
    session.event_history.append(
      {
        "type": "skill_result_captured",
        "skill_run_id": "skill-1",
        "skill": "model-review",
        "cost_usd": 0.09,
        "verdict_echo": {
          "verdict_token": "PRICE_TARGET_SET",
          "confidence": "HIGH",
          "one_line_summary": "summary",
        },
      }
    )

    response = client.get(f"/api/control/runs/{alice_chat['session_id']}", headers=_headers(alice_control))
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload == {
      "kind": "chat",
      "run_id": alice_chat["session_id"],
      "session_id": alice_chat["session_id"],
      "agent": "hank",
      "channel": "tui",
      "user_id": "alice",
      "owner_user_id": "alice",
      "raw_user_id": "alice",
      "user_slug": "alice",
      "risk_user_id": None,
      "user_email": None,
      "user_aliases": ["alice"],
      "identity_status": "legacy_user_id_fallback",
      "state": "starting",
      "started_at": payload["started_at"],
      "ended_at": None,
      "cost_usd": 0.09,
      "max_budget_usd": None,
      "initial_message": "first",
      "skill_run_ids": ["skill-1"],
      "current_verdict": {
        "verdict_token": "PRICE_TARGET_SET",
        "confidence": "HIGH",
        "one_line_summary": "summary",
        "skill_run_id": "skill-1",
      },
      "pending_approval": None,
      "latest_tool_result": None,
      "dispatch_scope": None,
    }

    unknown = client.get("/api/control/runs/not-a-run", headers=_headers(alice_control))
    cross_user = client.get(f"/api/control/runs/{bob_chat['session_id']}", headers=_headers(alice_control))

    assert unknown.status_code == 404
    assert cross_user.status_code == 404


def test_chat_run_cost_accumulates_completed_turns_and_live_partial() -> None:
  app = _make_app()
  with TestClient(app) as client:
    alice_control = _control_session(client, "alice")
    alice_chat = _chat_session(client, "alice")

    session = app.state.auth.session_store.get_session(alice_chat["session_id"])
    assert session is not None
    session.channel = "tui"
    session.initial_message = "first"
    session.event_history.append({
      "type": "stream_complete",
      "terminal_disposition": "completed",
      "usage": {"estimated_cost": 0.10},
    })
    session.event_history.append({
      "type": "stream_complete",
      "terminal_disposition": "completed",
      "usage": {"estimated_cost": 0.20},
    })
    session.stream_active = True
    session.event_history.append({"type": "turn_complete", "turn": 1, "usage": {"estimated_cost": 0.03}})

    response = client.get(f"/api/control/runs/{alice_chat['session_id']}", headers=_headers(alice_control))

  assert response.status_code == 200, response.text
  payload = response.json()
  assert payload["state"] == "running"
  assert payload["cost_usd"] == 0.33


def test_chat_run_cost_accumulates_child_and_parent_stream_totals() -> None:
  app = _make_app()
  with TestClient(app) as client:
    alice_control = _control_session(client, "alice")
    alice_chat = _chat_session(client, "alice")

    session = app.state.auth.session_store.get_session(alice_chat["session_id"])
    assert session is not None
    session.channel = "tui"
    session.initial_message = "parent with sub-agent"
    session.event_history.append(
      {
        "type": "stream_complete",
        "sub_agent_id": "sub0:child",
        "usage": {"estimated_cost": 0.20},
      }
    )
    session.event_history.append(
      {
        "type": "stream_complete",
        "usage": {"estimated_cost": 0.03},
      }
    )

    response = client.get(f"/api/control/runs/{alice_chat['session_id']}", headers=_headers(alice_control))

  assert response.status_code == 200, response.text
  assert response.json()["cost_usd"] == 0.23


def test_get_run_logs_reads_session_event_history_tail_and_enforces_user_scope() -> None:
  app = _make_app()
  with TestClient(app) as client:
    alice_control = _control_session(client, "alice")
    alice_chat = _chat_session(client, "alice")
    bob_chat = _chat_session(client, "bob")

    alice_session = app.state.auth.session_store.get_session(alice_chat["session_id"])
    bob_session = app.state.auth.session_store.get_session(bob_chat["session_id"])
    assert alice_session is not None
    assert bob_session is not None
    for index in range(3):
      alice_session.event_history.append({"type": "event", "index": index})
    bob_session.event_history.append({"type": "event", "index": "bob"})

    response = client.get(
      f"/api/control/runs/{alice_chat['session_id']}/logs?tail=2",
      headers=_headers(alice_control),
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["run_id"] == alice_chat["session_id"]
    assert payload["more_available"] is True
    assert [json.loads(line)["index"] for line in payload["log_lines"]] == [1, 2]

    cross_user = client.get(
      f"/api/control/runs/{bob_chat['session_id']}/logs",
      headers=_headers(alice_control),
    )
    assert cross_user.status_code == 404


def test_chat_session_event_history_retains_events_across_three_turns() -> None:
  app = _make_app()
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")

    streamed_event_types: list[str | None] = []
    for index in range(3):
      streamed_event_types.extend(
        event.get("type") for event in _collect_chat_turn_events(client, chat["session_token"], f"turn-{index}")
      )

    session = app.state.auth.session_store.get_session(chat["session_id"])
    assert session is not None
    retained = session.event_history.snapshot()
    retained_text = [event.get("text") for event in retained if event.get("type") == "text_delta"]
    retained_states = [event.get("state") for event in retained if event.get("type") == "run_state_changed"]

    assert retained_text == ["echo:turn-0", "echo:turn-1", "echo:turn-2"]
    assert retained_states == ["running", "completed", "running", "completed", "running", "completed"]
    assert [event.get("type") for event in retained].count("stream_complete") == 3
    assert "run_state_changed" not in streamed_event_types

    run = client.get(f"/api/control/runs/{chat['session_id']}", headers=_headers(control))
    assert run.status_code == 200, run.text
    assert run.json()["state"] == "completed"
    assert run.json()["ended_at"] is not None

    response = client.get(
      f"/api/control/runs/{chat['session_id']}/logs?tail=20",
      headers=_headers(control),
    )
    assert response.status_code == 200, response.text
    log_text = "\n".join(response.json()["log_lines"])
    assert "echo:turn-0" in log_text
    assert "echo:turn-1" in log_text
    assert "echo:turn-2" in log_text
    assert '"state": "completed"' in log_text


def test_failed_normal_chat_turn_records_terminal_control_lifecycle() -> None:
  app = _make_failing_app()
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")

    _consume_chat_turn(client, chat["session_token"], "fail")

    session = app.state.auth.session_store.get_session(chat["session_id"])
    assert session is not None
    retained = session.event_history.snapshot()
    retained_states = [event.get("state") for event in retained if event.get("type") == "run_state_changed"]

    assert retained_states == ["running", "failed"]
    assert retained[-2]["type"] == "error"
    assert retained[-1]["state"] == "failed"

    run = client.get(f"/api/control/runs/{chat['session_id']}", headers=_headers(control))
    assert run.status_code == 200, run.text
    payload = run.json()
    assert payload["state"] == "failed"
    assert payload["ended_at"] is not None


def test_setup_failure_records_terminal_control_lifecycle() -> None:
  app = _make_setup_failing_app()
  with TestClient(app, raise_server_exceptions=False) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")

    response = client.post(
      "/api/chat",
      headers=_headers(chat),
      json={"messages": [{"role": "user", "content": "fail during setup"}], "context": {"channel": "tui"}},
    )

    assert response.status_code == 200
    session = app.state.auth.session_store.get_session(chat["session_id"])
    assert session is not None
    retained = session.event_history.snapshot()
    retained_states = [event.get("state") for event in retained if event.get("type") == "run_state_changed"]

    assert retained_states == ["running", "failed"]
    assert any(event.get("type") == "error" and "runtime setup failed" in event.get("error", "") for event in retained)

    run = client.get(f"/api/control/runs/{chat['session_id']}", headers=_headers(control))
    assert run.status_code == 200, run.text
    payload = run.json()
    assert payload["state"] == "failed"
    assert payload["ended_at"] is not None


def test_active_later_chat_turn_overrides_prior_terminal_lifecycle() -> None:
  app = _make_app()
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")
    session = app.state.auth.session_store.get_session(chat["session_id"])
    assert session is not None
    session.event_history.append({"type": "run_state_changed", "state": "running", "ts": int(time.time()) - 3})
    session.event_history.append({
      "type": "stream_complete",
      "terminal_disposition": "completed",
      "usage": {},
    })
    session.event_history.append({"type": "run_state_changed", "state": "completed", "ts": int(time.time()) - 2})
    session.event_history.append({"type": "run_state_changed", "state": "running", "ts": int(time.time()) - 1})
    session.stream_active = True

    run = client.get(f"/api/control/runs/{chat['session_id']}", headers=_headers(control))
    assert run.status_code == 200, run.text
    payload = run.json()
    assert payload["state"] == "running"
    assert payload["ended_at"] is None


def test_delete_running_control_chat_cancels_background_task_and_disconnects() -> None:
  app, runner = _make_hanging_app()
  with TestClient(app) as client:
    control = _control_session(client, "alice")

    response = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "chat",
        "message": "keep running",
        "channel": "tui",
        "deadline_sec": 1,
      },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["run"]["state"] == "running"
    assert runner.started.is_set()

    chat_session_id = payload["chat_session_id"]
    session = app.state.auth.session_store.get_session(chat_session_id)
    assert session is not None
    assert session.stream_active is True
    assert any(key.startswith("control_chat_turn:") for key in session.control_chat_tasks)
    assert not any(key.startswith("control_chat_turn:") for key in session.background_tasks)

    deleted = client.delete(f"/api/control/runs/{chat_session_id}", headers=_headers(control))
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["state"] == "cancelled"
    retained_states = [event.get("state") for event in session.event_history.snapshot() if event.get("type") == "run_state_changed"]
    assert retained_states == ["running", "cancelled"]

    assert runner.disconnected.wait(timeout=1.0)
    assert runner.cancelled.wait(timeout=1.0)
    assert _wait_until(lambda: session.stream_active is False)
    assert _wait_until(lambda: not any(key.startswith("control_chat_turn:") for key in session.control_chat_tasks))
    assert app.state.auth.session_store.get_session(chat_session_id) is None


def test_listing_elapsed_running_chat_is_read_only_until_explicit_cancel() -> None:
  app, runner = _make_hanging_app()
  with TestClient(app) as client:
    control = _control_session(client, "alice")

    response = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "chat",
        "message": "expire while running",
        "channel": "tui",
        "deadline_sec": 1,
      },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["run"]["state"] == "running"
    assert runner.started.is_set()

    chat_session_id = payload["chat_session_id"]
    session = app.state.auth.session_store.get_session(chat_session_id)
    assert session is not None
    assert session.stream_active is True
    assert any(key.startswith("control_chat_turn:") for key in session.control_chat_tasks)

    session.expires_at = 0
    listed = client.get("/api/control/runs?kind=chat", headers=_headers(control))
    assert listed.status_code == 200, listed.text
    assert [run["run_id"] for run in listed.json()["runs"]] == []

    assert session._expired is False
    assert chat_session_id in app.state.auth.session_store.sessions
    assert runner.disconnected.is_set() is False
    assert runner.cancelled.is_set() is False

    deleted = client.delete(
      f"/api/control/runs/{chat_session_id}",
      headers=_headers(control),
    )
    assert deleted.status_code == 200, deleted.text

    assert runner.disconnected.wait(timeout=1.0)
    assert runner.cancelled.wait(timeout=1.0)
    assert _wait_until(lambda: session.stream_active is False)
    assert _wait_until(lambda: not any(key.startswith("control_chat_turn:") for key in session.control_chat_tasks))
    assert app.state.auth.session_store.get_session(chat_session_id) is None
