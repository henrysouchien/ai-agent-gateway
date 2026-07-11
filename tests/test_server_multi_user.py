import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import (  # noqa: E402
  AgentRunner,
  AgentSessionLog,
  CostEstimate,
  ModelInfo,
  ModelProvider,
  SessionContextBuilder,
  StreamEvent,
  ToolDispatcher,
)
from agent_gateway.auth import AuthConfig, NoCredentialError, ProviderCredentialFailure, ResolverResult  # noqa: E402
from agent_gateway import server as server_module  # noqa: E402
from agent_gateway import server_artifact_helpers as artifact_helpers_module  # noqa: E402
from agent_gateway import server_artifact_routes as artifact_routes_module  # noqa: E402
from agent_gateway import server_chat_helpers as chat_helpers_module  # noqa: E402
from agent_gateway import server_models as models_module  # noqa: E402
from agent_gateway.server import ChatInitRequest, ChatRuntime, GatewayServerConfig, create_gateway_app  # noqa: E402


def test_server_parent_aliases_moved_helpers() -> None:
  assert server_module.ChatRuntime is models_module.ChatRuntime
  assert server_module.GatewayServerConfig is models_module.GatewayServerConfig
  assert server_module._artifact_json_response is artifact_helpers_module._artifact_json_response
  assert server_module._error_payload is artifact_helpers_module._error_payload
  assert server_module._server_artifact_routes is artifact_routes_module
  assert server_module._dispatch_chat_turn is chat_helpers_module._dispatch_chat_turn
  assert callable(server_module._stream_subscriber_sse)
  assert callable(server_module._register_stream_subscriber)
  assert server_module._init_approval_subsystem is chat_helpers_module._init_approval_subsystem
  assert server_module.SQLiteApprovalStore is chat_helpers_module.SQLiteApprovalStore
  assert server_module.resolve_policy is chat_helpers_module.resolve_policy


def test_server_artifact_routes_use_parent_namespace_helpers(monkeypatch) -> None:
  calls: dict[str, Any] = {}
  sentinel_request = object()

  def _auth_dependency(request):
    calls["request"] = request
    return "alice"

  def _request_filters(request):
    calls["filters_request"] = request
    return {"origin_kind": "product"}

  def _artifact_paths(user_id: str, *, ticker: str, skill: str):
    calls["paths"] = (user_id, ticker, skill)
    return ["artifact-a"]

  def _json_response(artifact, *, user_id: str, filters: dict[str, Any]):
    calls["json"] = (artifact, user_id, filters)
    return JSONResponse({"ok": True})

  monkeypatch.setattr(server_module, "_artifact_auth_dependency", _auth_dependency)
  monkeypatch.setattr(server_module, "_artifact_request_filters", _request_filters)
  monkeypatch.setattr(server_module, "artifact_json_paths_for_request", _artifact_paths)
  monkeypatch.setattr(server_module, "_artifact_json_response", _json_response)

  response = server_module._server_artifact_routes.artifact_latest_response(
    server_module.__dict__,
    sentinel_request,
    "PCTY",
    "earnings-scenarios",
  )

  assert response.status_code == 200
  assert calls == {
    "request": sentinel_request,
    "filters_request": sentinel_request,
    "paths": ("alice", "PCTY", "earnings-scenarios"),
    "json": ("artifact-a", "alice", {"origin_kind": "product"}),
  }


class _StubRunner:
  def __init__(self, event_log, run_calls: list[dict[str, Any]]) -> None:
    self._event_log = event_log
    self._run_calls = run_calls
    self._credential_refresher = None

  def set_credential_refresher(self, callback) -> None:
    self._credential_refresher = callback

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
    if self._credential_refresher is not None:
      await self._credential_refresher(
        ProviderCredentialFailure(
          provider="anthropic",
          kind="rate_limit",
          status_code=429,
          message="rate limit exceeded",
        )
      )
    self._event_log.append({"type": "stream_complete", "usage": {}})


def _make_app(
  *,
  credentials_resolver=None,
  credentials_refresh_resolver=None,
  resolver_timeout_seconds: float = 5.0,
  channel_profile_allowlist: dict[str, frozenset[str]] | None = None,
  transcript_dir: Path | None = None,
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
      credentials_refresh_resolver=credentials_refresh_resolver,
      resolver_timeout_seconds=resolver_timeout_seconds,
      channel_profile_allowlist=channel_profile_allowlist,
      build_chat_runtime=_build_chat_runtime,
      transcript_dir=transcript_dir,
    )
  )
  return app, captured_requests, run_calls


def _init_session(
  client: TestClient,
  *,
  user_id: str | None = "alice",
  user_email: str | None = None,
  context: dict[str, Any] | None = None,
):
  payload: dict[str, Any] = {"api_key": "gateway-key"}
  if user_id is not None:
    payload["user_id"] = user_id
  if user_email is not None:
    payload["user_email"] = user_email
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


def _auth_config_for(user_id: str) -> AuthConfig:
  return AuthConfig.from_dict(
    {
      "provider": "anthropic",
      "billing_mode": "byok",
      "api_key": f"key-{user_id}",
      "model": "claude-sonnet-4-6",
      "max_tokens": 16000,
    }
  )


class _NoopMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


class _ReplayProvider(ModelProvider):
  name = "stub"

  def __init__(self, responses: list[str]) -> None:
    self.responses = list(responses)
    self.captured_messages: list[list[dict[str, Any]]] = []

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    _ = config
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> object:
    _ = config, timeout
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name)

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    **kwargs: Any,
  ) -> dict[str, Any]:
    _ = model, system_prompt, tools, max_tokens, kwargs
    self.captured_messages.append([dict(message) for message in messages])
    return {}

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    if not self.responses:
      raise AssertionError("unexpected extra provider call")
    text = self.responses.pop(0)
    yield StreamEvent(type="message_start", input_tokens=10)
    yield StreamEvent(type="text_delta", text=text)
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": text})
    yield StreamEvent(type="usage_update", output_tokens=5)
    yield StreamEvent(type="message_end", stop_reason="end_turn")

  def estimate_cost(
    self,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
  ) -> CostEstimate:
    _ = model, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens
    return CostEstimate()


def test_chat_stream_rebuilds_multi_turn_memory_from_session_log(tmp_path: Path) -> None:
  provider = _ReplayProvider(["AAPL market cap is about $4.4T.", "AAPL P/E is about 36x."])
  session_log = AgentSessionLog(path=tmp_path / "sessions" / "interactive.jsonl")

  async def _build_chat_runtime(session, request, channel, auth_manager):
    _ = session, request, channel, auth_manager

    def _build_runner(event_log, session_id):
      return AgentRunner(
        event_log=event_log,
        dispatcher=ToolDispatcher(
          mcp_client=_NoopMcpClient(),
          local_tool_handlers={},
          event_log=event_log,
          session_id=session_id,
        ),
        session_id=session_id,
        provider=provider,
        auth_config={"api_key": "k", "model": "stub-model"},
        get_tool_definitions=lambda: [],
        user_id=session.user_id,
        billing_mode="byok",
        rate_table_version="unknown",
        agent_session_log=session_log,
        context_builder=SessionContextBuilder(
          agent_session_log=session_log,
          tail_window_seconds=None,
        ),
      )

    return ChatRuntime(system_prompt="system", build_runner=_build_runner)

  app = create_gateway_app(
    GatewayServerConfig(
      auth_config={"model": "stub-model"},
      allowed_models={"stub-model"},
      build_chat_runtime=_build_chat_runtime,
    )
  )
  client = TestClient(app)
  init = _init_session(client)

  _consume_chat_stream(
    client,
    init["session_token"],
    {"messages": [{"role": "user", "content": "What is AAPL market cap?"}]},
  )
  _consume_chat_stream(
    client,
    init["session_token"],
    {
      "messages": [
        {"role": "user", "content": "CLIENT FABRICATED USER"},
        {"role": "assistant", "content": "CLIENT FABRICATED ASSISTANT"},
        {"role": "user", "content": "What is its P/E?"},
      ],
    },
  )

  assert len(provider.captured_messages) == 2
  second_input = json.dumps(provider.captured_messages[1], default=str)
  assert "What is AAPL market cap?" in second_input
  assert "AAPL market cap is about $4.4T." in second_input
  assert "What is its P/E?" in second_input
  assert "CLIENT FABRICATED USER" not in second_input
  assert "CLIENT FABRICATED ASSISTANT" not in second_input


def _resolver_result_for(user_id: str, *, channel: str = "excel") -> ResolverResult:
  return ResolverResult(
    user_id=user_id,
    channel=channel,
    auth_config=_auth_config_for(user_id),
    risk_user_id=abs(hash(user_id)) % 10_000 + 1,
    role="owner",
  )


def test_chat_init_requires_top_level_user_id_without_resolver() -> None:
  app, _captured_requests, _run_calls = _make_app()

  with TestClient(app) as client:
    top = _init_session(
      client,
      user_id="top-level",
      user_email="top@example.com",
      context={"user_id": "legacy"},
    )
    legacy = client.post("/api/chat/init", json={"api_key": "gateway-key", "context": {"user_id": "legacy"}})
    missing = client.post("/api/chat/init", json={"api_key": "gateway-key"})

  top_session = app.state.auth.session_store.get_session(top["session_id"])

  assert top_session.user_id == "top-level"
  assert top_session.user_email == "top@example.com"
  assert legacy.status_code == 400
  assert legacy.json()["error"] == "missing_user_id"
  assert missing.status_code == 400
  assert missing.json()["error"] == "missing_user_id"


def test_chat_init_request_accepts_byok_fields_without_channel_field() -> None:
  request = ChatInitRequest(
    api_key="gateway-key",
    anthropic_auth_mode="oauth",
    anthropic_api_key="sk-ant-api",
    anthropic_auth_token="sk-ant-oat",
  )
  fields = ChatInitRequest.model_fields if hasattr(ChatInitRequest, "model_fields") else ChatInitRequest.__fields__

  assert request.anthropic_auth_mode == "oauth"
  assert request.anthropic_api_key == "sk-ant-api"
  assert request.anthropic_auth_token == "sk-ant-oat"
  assert "channel" not in fields


def test_chat_init_rejects_reserved_user_returned_by_resolver() -> None:
  calls: list[tuple[str, Any]] = []

  async def _resolver(api_key: str, init_request):
    calls.append((api_key, init_request))
    return _resolver_result_for("_default")

  app, _captured_requests, _run_calls = _make_app(credentials_resolver=_resolver)

  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "gateway-key"})

  assert calls and calls[0][0] == "gateway-key"
  assert response.status_code == 400
  assert response.json()["error"] == "missing_user_id"


def test_chat_init_accepts_resolver_derived_user_without_request_user_id() -> None:
  async def _resolver(api_key: str, init_request):
    assert api_key == "gateway-key"
    assert init_request.user_id is None
    return _resolver_result_for("alice", channel="excel")

  app, _captured_requests, _run_calls = _make_app(credentials_resolver=_resolver)

  with TestClient(app) as client:
    init_response = _init_session(client, user_id=None)

  session = app.state.auth.session_store.get_session(init_response["session_id"])
  assert session.user_id == "alice"
  assert session.channel == "excel"
  assert session.is_public is False


def test_channel_profile_allowlist_none_preserves_profile_resolution() -> None:
  async def _resolver(_api_key: str, init_request):
    return _resolver_result_for(init_request.user_id, channel="x")

  app, captured_requests, _run_calls = _make_app(credentials_resolver=_resolver)

  with TestClient(app) as client:
    init_response = _init_session(client, user_id="alice")
    _consume_chat_stream(
      client,
      init_response["session_token"],
      {
        "user_id": "alice",
        "messages": [{"role": "user", "content": "hello"}],
        "context": {"profile": "analyst"},
      },
    )
    _consume_chat_stream(
      client,
      init_response["session_token"],
      {
        "user_id": "alice",
        "messages": [{"role": "user", "content": "hello"}],
        "context": {"profile": "community"},
      },
    )

  assert [item["request"].context["profile"] for item in captured_requests] == ["analyst", "community"]


def test_channel_profile_allowlist_rejects_disallowed_session_channel_profile(monkeypatch) -> None:
  async def _resolver(_api_key: str, init_request):
    return _resolver_result_for(init_request.user_id, channel="x")

  app, captured_requests, _run_calls = _make_app(
    credentials_resolver=_resolver,
    channel_profile_allowlist={"x": frozenset({"community"})},
  )
  subscriber_calls: list[Any] = []
  original_register = server_module._register_stream_subscriber

  def _record_register(*args, **kwargs):
    subscriber_calls.append((args, kwargs))
    return original_register(*args, **kwargs)

  monkeypatch.setattr(server_module, "_register_stream_subscriber", _record_register)

  with TestClient(app, raise_server_exceptions=False) as client:
    init_response = _init_session(client, user_id="alice")
    response = client.post(
      "/api/chat",
      headers={"Authorization": f"Bearer {init_response['session_token']}"},
      json={
        "user_id": "alice",
        "messages": [{"role": "user", "content": "hello"}],
        "context": {"channel": "unrestricted-client-claim", "profile": "analyst"},
        "drain_trailing": True,
      },
    )

  assert response.status_code == 403
  assert response.json()["detail"] == "Profile 'analyst' is not permitted on channel 'x'"
  assert captured_requests == []
  session = app.state.auth.session_store.get_session(init_response["session_id"])
  assert session.stream_active is False
  assert session.active_turn is None
  assert subscriber_calls == []


def test_discord_channel_profile_allowlist_rejects_analyst_and_hank_alias() -> None:
  async def _resolver(_api_key: str, init_request):
    return _resolver_result_for(init_request.user_id, channel="discord")

  app, captured_requests, _run_calls = _make_app(
    credentials_resolver=_resolver,
    channel_profile_allowlist={"discord": frozenset({"community"})},
  )

  with TestClient(app, raise_server_exceptions=False) as client:
    init_response = _init_session(client, user_id="alice")
    for requested_profile, resolved_profile in (("analyst", "analyst"), ("hank", "analyst")):
      response = client.post(
        "/api/chat",
        headers={"Authorization": f"Bearer {init_response['session_token']}"},
        json={
          "user_id": "alice",
          "messages": [{"role": "user", "content": "hello"}],
          "context": {"profile": requested_profile},
        },
      )
      assert response.status_code == 403
      assert response.json()["detail"] == (
        f"Profile {resolved_profile!r} is not permitted on channel 'discord'"
      )

  assert captured_requests == []


def test_discord_hank_community_alias_resolves_to_community_profile() -> None:
  async def _resolver(_api_key: str, init_request):
    return _resolver_result_for(init_request.user_id, channel="discord")

  app, captured_requests, _run_calls = _make_app(
    credentials_resolver=_resolver,
    channel_profile_allowlist={"discord": frozenset({"community"})},
  )

  with TestClient(app) as client:
    init_response = _init_session(client, user_id="alice")
    _consume_chat_stream(
      client,
      init_response["session_token"],
      {
        "user_id": "alice",
        "messages": [{"role": "user", "content": "hello"}],
        "context": {"profile": "hank-community"},
      },
    )

  assert [item["request"].context["profile"] for item in captured_requests] == ["community"]


def test_discord_session_channel_profile_allowlist_ignores_claimed_context_channel() -> None:
  async def _resolver(_api_key: str, init_request):
    return _resolver_result_for(init_request.user_id, channel="discord")

  app, captured_requests, _run_calls = _make_app(
    credentials_resolver=_resolver,
    channel_profile_allowlist={"discord": frozenset({"community"})},
  )

  with TestClient(app, raise_server_exceptions=False) as client:
    init_response = _init_session(client, user_id="alice")
    response = client.post(
      "/api/chat",
      headers={"Authorization": f"Bearer {init_response['session_token']}"},
      json={
        "user_id": "alice",
        "messages": [{"role": "user", "content": "hello"}],
        "context": {"channel": "web", "profile": "analyst"},
      },
    )

  assert response.status_code == 403
  assert response.json()["detail"] == "Profile 'analyst' is not permitted on channel 'discord'"
  assert captured_requests == []


def test_channel_profile_allowlist_absent_channel_is_unrestricted() -> None:
  async def _resolver(_api_key: str, init_request):
    return _resolver_result_for(init_request.user_id, channel="y")

  app, captured_requests, _run_calls = _make_app(
    credentials_resolver=_resolver,
    channel_profile_allowlist={"x": frozenset({"community"})},
  )

  with TestClient(app) as client:
    init_response = _init_session(client, user_id="alice")
    _consume_chat_stream(
      client,
      init_response["session_token"],
      {
        "user_id": "alice",
        "messages": [{"role": "user", "content": "hello"}],
        "context": {"profile": "analyst"},
      },
    )

  assert [item["request"].context["profile"] for item in captured_requests] == ["analyst"]


def test_chat_init_calls_resolver_and_stores_auth_config() -> None:
  calls: list[tuple[str, Any]] = []

  async def _resolver(api_key: str, init_request):
    calls.append((api_key, init_request))
    return _resolver_result_for(init_request.user_id, channel="web")

  app, _captured_requests, _run_calls = _make_app(credentials_resolver=_resolver)

  with TestClient(app) as client:
    init_response = _init_session(client, user_id="alice", user_email="alice@example.com")

  session = app.state.auth.session_store.get_session(init_response["session_id"])
  verified_session, claims = app.state.auth.verify_token_with_payload(init_response["session_token"])
  assert calls and calls[0][0] == "gateway-key"
  assert session.user_id == "alice"
  assert session.user_email == "alice@example.com"
  assert session.channel == "web"
  assert session.is_public is False
  assert verified_session.user_email == "alice@example.com"
  assert claims["user_email"] == "alice@example.com"
  assert claims["channel"] == "web"
  assert claims["is_public"] is False
  assert session.auth_config == {
    "provider": "anthropic",
    "billing_mode": "byok",
    "api_key": "key-alice",
    "model": "claude-sonnet-4-6",
    "max_tokens": 16000,
  }


def test_chat_refresh_resolver_updates_session_auth_config() -> None:
  refresh_requests: list[Any] = []

  async def _resolver(_api_key: str, init_request):
    return ResolverResult(
      user_id=init_request.user_id,
      channel="web",
      auth_config=AuthConfig.from_dict(
        {
          "provider": "anthropic",
          "billing_mode": "byok",
          "rate_table_version": "unknown",
          "api_key": "key-alice",
          "model": "claude-sonnet-4-6",
          "max_tokens": 16000,
        }
      ),
      risk_user_id=101,
      role="owner",
    )

  async def _refresh(request):
    refresh_requests.append(request)
    return AuthConfig.from_dict(
      {
        "provider": "anthropic",
        "billing_mode": "metered",
        "rate_table_version": "2026-04-08",
        "api_key": "key-rotated",
        "model": "claude-sonnet-4-6",
        "max_tokens": 16000,
      }
    )

  app, _captured_requests, _run_calls = _make_app(
    credentials_resolver=_resolver,
    credentials_refresh_resolver=_refresh,
  )

  with TestClient(app) as client:
    init_response = _init_session(client, user_id="alice")
    _consume_chat_stream(
      client,
      init_response["session_token"],
      {"messages": [{"role": "user", "content": "hi"}], "user_id": "alice", "request_id": "req-refresh"},
    )

  session = app.state.auth.session_store.get_session(init_response["session_id"])
  assert session.auth_config["api_key"] == "key-rotated"
  assert session.auth_config["billing_mode"] == "byok"
  assert session.auth_config["rate_table_version"] == "unknown"
  assert refresh_requests
  assert refresh_requests[0].user_id == "alice"
  assert refresh_requests[0].request_id == "req-refresh"
  assert refresh_requests[0].billing_mode == "byok"
  assert refresh_requests[0].rate_table_version == "unknown"
  assert refresh_requests[0].failure.kind == "rate_limit"


def test_chat_rejects_cross_user_reuse_in_strict_mode() -> None:
  async def _resolver(_api_key: str, init_request):
    return _resolver_result_for(init_request.user_id)

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
  async def _resolver(_api_key: str, init_request):
    return _resolver_result_for(init_request.user_id)

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


def test_non_strict_mode_rejects_missing_init_user_id() -> None:
  app, _captured_requests, _run_calls = _make_app()

  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "gateway-key"})

  assert response.status_code == 400
  assert response.json()["error"] == "missing_user_id"


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


def test_chat_request_drain_trailing_constructs_deferred_event_log() -> None:
  app, _captured_requests, _run_calls = _make_app()

  with TestClient(app) as client:
    init_response = _init_session(client, user_id="alice")
    _consume_chat_stream(
      client,
      init_response["session_token"],
      {
        "messages": [{"role": "user", "content": "hello"}],
        "drain_trailing": True,
      },
    )

  session = app.state.auth.session_store.get_session(init_response["session_id"])
  assert session.active_turn is not None
  assert session.active_turn.event_log.defer_terminal_close is True
  assert session.active_turn.event_log.closed is True


def test_chat_metadata_reaches_runtime_and_transcript(tmp_path: Path) -> None:
  app, captured_requests, _run_calls = _make_app(transcript_dir=tmp_path)
  metadata = {
    "contextual_chat": {
      "id": "ctx-holding-1",
      "schemaVersion": "contextual-chat-v1",
      "surface": "portfolio",
      "objectType": "holding",
      "objectId": "AAPL",
      "visibleText": "AAPL row",
    }
  }

  with TestClient(app) as client:
    init_response = _init_session(client, user_id="alice")
    _consume_chat_stream(
      client,
      init_response["session_token"],
      {
        "messages": [{"role": "user", "content": "Review this row"}],
        "metadata": metadata,
      },
    )

  assert captured_requests[0]["request"].metadata == metadata
  transcript_path = tmp_path / f"{init_response['session_id']}.jsonl"
  transcript_events = [
    json.loads(line)
    for line in transcript_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
  ]
  chat_request = next(event for event in transcript_events if event.get("type") == "chat_request")
  assert chat_request["metadata"] == metadata


def test_credentials_resolver_timeout_returns_structured_error() -> None:
  async def _resolver(_api_key: str, _init_request):
    await asyncio.sleep(0.05)
    return _resolver_result_for("slow")

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
  async def _resolver(_api_key: str, init_request):
    user_id = init_request.user_id
    raise NoCredentialError(f"User {user_id} has no credential configured.")

  app, _captured_requests, _run_calls = _make_app(credentials_resolver=_resolver)

  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "gateway-key", "user_id": "alice"})

  assert response.status_code == 401
  payload = response.json()
  assert payload["error"] == "credentials_unavailable"
  assert payload["reason"] == "User alice has no credential configured."


def test_gateway_server_config_accepts_on_session_created_hook() -> None:
  def _hook(_session, _api_key: str, _request) -> None:
    return None

  config = GatewayServerConfig(build_chat_runtime=lambda *_args, **_kwargs: None, on_session_created=_hook)

  assert config.on_session_created is _hook


def test_on_session_created_exception_expires_session_and_reraises() -> None:
  expired_session_ids: list[str] = []

  def _hook(session, _api_key: str, _request) -> None:
    raise HTTPException(status_code=401, detail=f"reject {session.session_id}")

  app, _captured_requests, _run_calls = _make_app()
  app.state.gateway_config.on_session_created = _hook
  original_expire_session = app.state.auth.session_store.expire_session

  def _expire_session(session_id: str) -> None:
    expired_session_ids.append(session_id)
    original_expire_session(session_id)

  app.state.auth.session_store.expire_session = _expire_session

  with TestClient(app) as client:
    response = client.post("/api/chat/init", json={"api_key": "gateway-key", "user_id": "alice"})

  assert response.status_code == 401
  assert expired_session_ids
  assert app.state.auth.session_store.sessions == {}


def test_auth_config_to_dict_is_called_before_create_session_boundary() -> None:
  class _TrackedAuthConfig:
    calls = 0

    def to_dict(self) -> dict[str, Any]:
      self.calls += 1
      return {
        "provider": "anthropic",
        "billing_mode": "byok",
        "api_key": "tracked-key",
        "model": "claude-sonnet-4-6",
      }

  tracked = _TrackedAuthConfig()

  async def _resolver(_api_key: str, _init_request):
    return ResolverResult(
      user_id="alice",
      channel="excel",
      auth_config=tracked,
      risk_user_id=101,
      role="owner",
    )

  app, _captured_requests, _run_calls = _make_app(credentials_resolver=_resolver)

  with TestClient(app) as client:
    init_response = _init_session(client, user_id=None)

  session = app.state.auth.session_store.get_session(init_response["session_id"])
  assert tracked.calls == 1
  assert session.auth_config == {
    "provider": "anthropic",
    "billing_mode": "byok",
    "api_key": "tracked-key",
    "model": "claude-sonnet-4-6",
  }
