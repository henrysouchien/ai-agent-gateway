import asyncio
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.auth import AuthConfig, NoCredentialError
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app


class _StubRunner:
  def __init__(self, event_log, run_calls: list[dict[str, Any]]) -> None:
    self._event_log = event_log
    self._run_calls = run_calls

  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    model_override: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    self._run_calls.append(
      {
        "messages": messages,
        "system_prompt": system_prompt,
        "model_override": model_override,
        "max_turns": max_turns,
      }
    )
    self._event_log.append({"type": "stream_complete", "usage": {}})


def _make_app(
  *,
  credentials_resolver=None,
  resolver_timeout_seconds: float = 5.0,
):
  captured_requests: list[Any] = []
  run_calls: list[dict[str, Any]] = []

  async def _build_chat_runtime(session, request, channel, auth_manager):
    _ = channel, auth_manager
    captured_requests.append({"session": session, "request": request})
    return ChatRuntime(
      system_prompt="test",
      build_runner=lambda event_log, _sid: _StubRunner(event_log, run_calls),
    )

  app = create_gateway_app(
    GatewayServerConfig(
      auth_config={"model": "claude-sonnet-4-6"},
      credentials_resolver=credentials_resolver,
      resolver_timeout_seconds=resolver_timeout_seconds,
      build_chat_runtime=_build_chat_runtime,
    )
  )
  return app, captured_requests, run_calls


def _init_session(client: TestClient, *, user_id: str | None = None, context: dict[str, Any] | None = None):
  payload: dict[str, Any] = {"api_key": "gateway-key"}
  if user_id is not None:
    payload["user_id"] = user_id
  if context is not None:
    payload["context"] = context
  response = client.post("/api/chat/init", json=payload)
  assert response.status_code == 200, response.text
  return response.json()


def _consume_chat_stream(client: TestClient, token: str, payload: dict[str, Any]) -> None:
  with client.stream(
    "POST",
    "/api/chat",
    headers={"Authorization": f"Bearer {token}"},
    json=payload,
  ) as response:
    assert response.status_code == 200, response.text
    list(response.iter_lines())


def _resolver_for(user_id: str) -> AuthConfig:
  return AuthConfig.from_dict(
    {
      "provider": "anthropic",
      "billing_mode": "byok",
      "api_key": f"key-{user_id}",
      "model": "claude-sonnet-4-6",
      "max_tokens": 16000,
    }
  )


def test_chat_init_resolves_user_id_top_level_then_context_then_default() -> None:
  app, _captured_requests, _run_calls = _make_app()

  with TestClient(app) as client:
    top = _init_session(client, user_id="top-level", context={"user_id": "legacy"})
    legacy = _init_session(client, context={"user_id": "legacy"})
    default = _init_session(client)

  top_session = app.state.auth.session_store.get_session(top["session_id"])
  legacy_session = app.state.auth.session_store.get_session(legacy["session_id"])
  default_session = app.state.auth.session_store.get_session(default["session_id"])

  assert top_session.user_id == "top-level"
  assert legacy_session.user_id == "legacy"
  assert default_session.user_id == "_default"


def test_strict_mode_rejects_default_user_at_chat_init() -> None:
  async def _resolver(user_id: str, _init_request):
    return _resolver_for(user_id)

  app, _captured_requests, _run_calls = _make_app(credentials_resolver=_resolver)

  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "gateway-key"})

  assert response.status_code == 400
  assert response.json()["error"] == "strict_mode_default_user"


def test_chat_init_calls_resolver_and_stores_auth_config() -> None:
  calls: list[tuple[str, Any]] = []

  async def _resolver(user_id: str, init_request):
    calls.append((user_id, init_request))
    return _resolver_for(user_id)

  app, _captured_requests, _run_calls = _make_app(credentials_resolver=_resolver)

  with TestClient(app) as client:
    init_response = _init_session(client, user_id="alice")

  session = app.state.auth.session_store.get_session(init_response["session_id"])
  assert calls and calls[0][0] == "alice"
  assert session.user_id == "alice"
  assert session.auth_config == {
    "provider": "anthropic",
    "billing_mode": "byok",
    "api_key": "key-alice",
    "model": "claude-sonnet-4-6",
    "max_tokens": 16000,
  }


def test_chat_rejects_cross_user_reuse_in_strict_mode() -> None:
  async def _resolver(user_id: str, _init_request):
    return _resolver_for(user_id)

  app, _captured_requests, _run_calls = _make_app(credentials_resolver=_resolver)

  with TestClient(app) as client:
    init_response = _init_session(client, user_id="alice")
    response = client.post(
      "/api/chat",
      headers={"Authorization": f"Bearer {init_response['session_token']}"},
      json={"messages": [{"role": "user", "content": "hi"}], "user_id": "bob"},
    )

  assert response.status_code == 401
  assert response.json()["error"] == "cross_user_reuse"
  assert response.json()["session_user"] == "alice"
  assert response.json()["request_user"] == "bob"


def test_chat_requires_user_id_in_strict_mode() -> None:
  async def _resolver(user_id: str, _init_request):
    return _resolver_for(user_id)

  app, _captured_requests, _run_calls = _make_app(credentials_resolver=_resolver)

  with TestClient(app) as client:
    init_response = _init_session(client, user_id="alice")
    response = client.post(
      "/api/chat",
      headers={"Authorization": f"Bearer {init_response['session_token']}"},
      json={"messages": [{"role": "user", "content": "hi"}]},
    )

  assert response.status_code == 400
  assert response.json()["error"] == "missing_user_id"


def test_non_strict_mode_defaults_missing_user_id_to_jwt_bound_value() -> None:
  app, captured_requests, _run_calls = _make_app()

  with TestClient(app) as client:
    init_response = _init_session(client, user_id="alice")
    _consume_chat_stream(
      client,
      init_response["session_token"],
      {"messages": [{"role": "user", "content": "hello"}]},
    )

  assert captured_requests[0]["request"].user_id == "alice"


def test_chat_request_id_uses_consumer_value_or_gateway_uuid() -> None:
  app, captured_requests, _run_calls = _make_app()

  with TestClient(app) as client:
    init_response = _init_session(client, user_id="alice")
    token = init_response["session_token"]
    _consume_chat_stream(
      client,
      token,
      {
        "messages": [{"role": "user", "content": "hello"}],
        "request_id": "req-123",
      },
    )
    _consume_chat_stream(
      client,
      token,
      {"messages": [{"role": "user", "content": "hello again"}]},
    )

  assert captured_requests[0]["request"].request_id == "req-123"
  assert captured_requests[1]["request"].request_id != "req-123"
  assert str(uuid.UUID(captured_requests[1]["request"].request_id)) == captured_requests[1]["request"].request_id


def test_credentials_resolver_timeout_returns_structured_error() -> None:
  async def _resolver(_user_id: str, _init_request):
    await asyncio.sleep(0.05)
    return _resolver_for("slow")

  app, _captured_requests, _run_calls = _make_app(
    credentials_resolver=_resolver,
    resolver_timeout_seconds=0.01,
  )

  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "gateway-key", "user_id": "alice"})

  assert response.status_code == 504
  assert response.json()["error"] == "credentials_timeout"
  assert response.json()["user_id"] == "alice"


def test_credentials_resolver_raise_returns_structured_error() -> None:
  async def _resolver(user_id: str, _init_request):
    raise NoCredentialError(f"User {user_id} has no credential configured.")

  app, _captured_requests, _run_calls = _make_app(credentials_resolver=_resolver)

  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "gateway-key", "user_id": "alice"})

  assert response.status_code == 401
  payload = response.json()
  assert payload["error"] == "credentials_unavailable"
  assert payload["reason"] == "User alice has no credential configured."
