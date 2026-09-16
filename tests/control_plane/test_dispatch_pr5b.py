from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from gateway_test_support.control_plane_identity import fake_identity_resolver, fake_mcp_user_key_lookup

from agent_gateway.capability_binding import (
  CredentialHandle,
)
from agent_gateway.capability_execution import BoundCapabilityExecution
from agent_gateway.model_registry import (
  CAPABILITY_IDS,
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_gateway.event_log import EventLog
from agent_gateway.runner import AgentRunner
from agent_gateway.server import (
  ChatRequest,
  ChatRuntime,
  GatewayServerConfig,
  MaterializedCredential,
  create_gateway_app,
)
from agent_gateway.session import AuthManager, GatewaySession


API_KEY = "dispatch-pr5b-key"
_SERVICE_HANDLE = CredentialHandle(
  handle_id="service:dispatch-pr5b:anthropic",
  provider="anthropic",
  principal="service",
  tenant_id="dispatch-pr5b",
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


class _EchoRunner(AgentRunner):
  def __init__(
    self,
    event_log: EventLog,
    captured: dict[str, Any],
    capability_execution: BoundCapabilityExecution,
  ) -> None:
    self._event_log = event_log
    self._captured = captured
    self._capability_execution = capability_execution
    self._selected_content_bindings_bound = False

  async def run(
    self,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None = None,
    max_turns: int | None = None,
    *,
    resume_initial_messages: list[dict[str, Any]] | None = None,
  ) -> None:
    _ = system_prompt, max_turns, resume_initial_messages
    self._captured["messages"] = messages
    last_user = next((message["content"] for message in reversed(messages) if message.get("role") == "user"), "")
    self._event_log.append({"type": "text_delta", "text": f"echo:{last_user}"})
    self._event_log.append({
      "type": "stream_complete",
      "terminal_disposition": "completed",
      "usage": {},
    })


def _make_app(captured: dict[str, Any] | None = None) -> FastAPI:
  captured = captured if captured is not None else {}

  async def _build_chat_runtime(
    session: GatewaySession,
    request: ChatRequest,
    channel: str | None,
    auth_manager: AuthManager | None,
    *,
    storage_root: Path | None = None,
  ) -> ChatRuntime:
    _ = auth_manager
    captured["request_context"] = dict(request.context or {})
    captured["channel"] = channel
    captured["session_id"] = session.session_id
    capability_execution = request.capability_execution
    assert capability_execution is not None
    return ChatRuntime(
      system_prompt="system",
      build_runner=lambda event_log, _sid, _started_at: _EchoRunner(
        event_log,
        captured,
        capability_execution,
      ),
      capability_execution=capability_execution,
    )

  return create_gateway_app(
    GatewayServerConfig(
      identity_resolver=fake_identity_resolver,
      mcp_user_key_lookup=fake_mcp_user_key_lookup,
      jwt_secret="dispatch-pr5b-test-secret-0123456789",
      valid_api_keys={API_KEY},
      tenant_id="dispatch-pr5b",
      allow_service_credentials_for_interactive=True,
      model_registry=INITIAL_MODEL_REGISTRY,
      model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
      service_provider_handles={"anthropic": _SERVICE_HANDLE},
      service_auth_config_resolver=_materialize_service_credential,
      build_chat_runtime=_build_chat_runtime,
    )
  )


def _control_session(app: FastAPI, client: TestClient, user_id: str) -> dict[str, Any]:
  response = client.post(
    "/api/control/session",
    json={"api_key": API_KEY, "user_id": user_id, "context": {"channel": "tui"}},
  )
  assert response.status_code == 200, response.text
  return _with_model_entitlements(app, response.json())


def _chat_session(app: FastAPI, client: TestClient, user_id: str) -> dict[str, Any]:
  response = client.post(
    "/api/chat/init",
    json={"api_key": API_KEY, "user_id": user_id, "context": {"channel": "tui"}},
  )
  assert response.status_code == 200, response.text
  return _with_model_entitlements(app, response.json())


def _with_model_entitlements(
  app: FastAPI,
  session_payload: dict[str, Any],
) -> dict[str, Any]:
  session = app.state.auth.session_store.get_session(
    session_payload["session_id"]
  )
  assert session is not None
  session.model_entitled_capabilities = CAPABILITY_IDS
  session.model_entitled_keys = frozenset(INITIAL_MODEL_REGISTRY.models)
  payload = dict(session_payload)
  payload["session_token"] = app.state.auth.issue_token(session)
  return payload


def _headers(session_payload: dict[str, Any]) -> dict[str, str]:
  return {"Authorization": f"Bearer {session_payload['session_token']}"}


def test_chat_dispatch_mints_chat_session_token_and_returns_chat_run() -> None:
  captured: dict[str, Any] = {}
  app = _make_app(captured)
  with TestClient(app) as client:
    control = _control_session(app, client, "alice")

    response = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "chat",
        "message": "review AAPL",
        "channel": "tui",
        "skill": "earnings-review",
        "ticker": "AAPL",
      },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["chat_session_token"]
    assert payload["chat_session_id"] == payload["run"]["session_id"]
    assert payload["chat_session_expires_at"] > 0
    assert "task_id" not in payload

    run = payload["run"]
    assert run["kind"] == "chat"
    assert run["run_id"] == payload["chat_session_id"]
    assert run["channel"] == "tui"
    assert run["user_id"] == "alice"
    assert run["owner_user_id"] == "alice"
    assert run["raw_user_id"] == "alice"
    assert run["user_aliases"] == ["alice"]
    assert run["identity_status"] == "legacy_user_id_fallback"
    assert run["state"] == "completed"
    assert run["ended_at"] is not None
    assert run["initial_message"] == "review AAPL"

    session = app.state.auth.session_store.get_session(payload["chat_session_id"])
    assert session is not None
    assert session.kind == "chat"
    assert session.channel == "tui"
    lifecycle_states = [
      event.get("state") for event in session.event_history.snapshot() if event.get("type") == "run_state_changed"
    ]
    assert lifecycle_states == ["running", "completed"]
    assert captured["messages"] == [{"role": "user", "content": "review AAPL"}]
    assert captured["request_context"]["skill"] == "earnings-review"
    assert captured["request_context"]["ticker"] == "AAPL"

    verified = app.state.auth.verify_token(payload["chat_session_token"])
    assert verified.session_id == payload["chat_session_id"]
    assert verified.kind == "chat"


def test_chat_dispatch_requires_control_session_token() -> None:
  app = _make_app()
  with TestClient(app) as client:
    chat = _chat_session(app, client, "alice")

    response = client.post(
      "/api/control/runs",
      headers=_headers(chat),
      json={"kind": "chat", "message": "hello", "channel": "tui"},
    )

    assert response.status_code == 401
