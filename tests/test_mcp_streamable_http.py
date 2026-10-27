import asyncio
import hashlib
import hmac
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest
from hank_contracts.agent_api import (
  AGENT_API_CLAIM_HEADERS,
  canonical_agent_api_claim_bytes,
)
from mcp.shared.auth import OAuthToken

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.claim_signing_authority as claim_authority_module
import agent_gateway.mcp_client as mcp_client_module
from agent_gateway.claim_signing_authority import GatewayClaimSigningAuthority
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.mcp_client_config import McpEnvExpansionError
from agent_gateway.session import GatewaySession


def _run(coro):
  return asyncio.run(coro)


def test_startup_allows_streamable_http_server_type(tmp_path) -> None:
  config_path = tmp_path / "claude.json"
  config_path.write_text(
    '{"mcpServers": {"finance-cli": {"type": "streamable-http", "url": "https://cashnerd.ai/mcp"}}}',
    encoding="utf-8",
  )
  manager = McpClientManager(config_path=config_path, allowed_servers={"finance-cli"})
  calls: list[str] = []

  async def _fake_connect_or_warn(name, config):
    calls.append(name)
    return None

  async def _main():
    original = manager._connect_or_warn
    manager._connect_or_warn = _fake_connect_or_warn
    try:
      await manager.startup()
    finally:
      manager._connect_or_warn = original

  _run(_main())

  assert calls == ["finance-cli"]


@pytest.mark.parametrize("terminate_on_close", [False, True])
def test_connect_streamable_http_lists_catalog_pages_and_closes_session(
  monkeypatch, terminate_on_close,
) -> None:
  requests: list[httpx2.Request] = []
  cursors: list[str | None] = []
  http_clients: list[httpx2.AsyncClient] = []

  async def respond(request: httpx2.Request) -> httpx2.Response:
    requests.append(request)
    assert request.headers["Authorization"] == "Bearer secret-token"
    if request.method == "GET":
      return httpx2.Response(405)
    if request.method == "DELETE":
      return httpx2.Response(200)
    payload = json.loads(request.content)
    if payload["method"] == "notifications/initialized":
      return httpx2.Response(202)
    if payload["method"] == "initialize":
      result = {
        "protocolVersion": "2025-11-25",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "paged-finance", "version": "1"},
      }
    else:
      assert payload["method"] == "tools/list"
      cursor = payload.get("params", {}).get("cursor")
      cursors.append(cursor)
      if cursor is None:
        result = {
          "tools": [{
            "name": "remote_tool",
            "description": "Remote tool",
            "inputSchema": {"type": "object", "properties": {"ticker": {"type": "string"}}},
            "_meta": {"audience": ["assistant"]},
          }],
          "nextCursor": "page-2",
        }
      else:
        assert cursor == "page-2"
        result = {
          "tools": [{
            "name": "second_tool",
            "inputSchema": {"type": "object", "properties": {}},
          }],
        }
    return httpx2.Response(
      200,
      headers={"Mcp-Session-Id": "mcp-session-1"},
      json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
    )

  def http_client(**kwargs):
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond), **kwargs)
    http_clients.append(client)
    return client

  monkeypatch.setattr(
    mcp_client_module, "httpx2",
    SimpleNamespace(AsyncClient=http_client, Timeout=httpx2.Timeout),
  )
  monkeypatch.setenv("CASHNERD_MCP_TOKEN", "secret-token")

  async def scenario():
    manager = McpClientManager(config_path=None, startup_timeout=1)
    state = await manager._connect(
      "finance-cli",
      {
        "type": "streamable-http",
        "url": "https://cashnerd.ai/mcp",
        "headers": {"Authorization": "Bearer ${CASHNERD_MCP_TOKEN}"},
        "run_scoped_env": ["CASHNERD_MCP_TOKEN"],
        "timeout": 7,
        "sse_read_timeout": 45,
        "terminate_on_close": terminate_on_close,
      },
    )
    try:
      assert state.tool_names == {"remote_tool", "second_tool"}
      assert state.tool_definitions[0]["input_schema"]["properties"]["ticker"]["type"] == "string"
      assert state.tool_metadata["remote_tool"] == {"audience": ["assistant"]}
      assert all("_meta" not in tool and "meta" not in tool for tool in state.tool_definitions)
      assert cursors == [None, "page-2"]
    finally:
      await manager._close_contexts(state.exit_contexts)
    assert http_clients[0].is_closed
    assert sum(request.method == "DELETE" for request in requests) == int(terminate_on_close)

  _run(scenario())


def test_streamable_http_refuses_undeclared_ambient_header_token(monkeypatch) -> None:
  monkeypatch.setenv("CASHNERD_MCP_TOKEN", "ambient-operator-token")
  connected = False

  def http_client(**_kwargs):
    nonlocal connected
    connected = True
    raise AssertionError("undeclared credentials must fail before connecting")

  monkeypatch.setattr(
    mcp_client_module, "httpx2",
    SimpleNamespace(AsyncClient=http_client, Timeout=httpx2.Timeout),
  )

  async def scenario():
    manager = McpClientManager(config_path=None, startup_timeout=1)
    with pytest.raises(McpEnvExpansionError, match="CASHNERD_MCP_TOKEN"):
      await manager._connect(
        "finance-cli",
        {
          "type": "streamable-http",
          "url": "https://cashnerd.ai/mcp",
          "headers": {"Authorization": "Bearer ${CASHNERD_MCP_TOKEN}"},
        },
      )

  _run(scenario())
  assert connected is False


def test_oauth_auth_reuses_persistent_tokens_with_httpx2(tmp_path) -> None:
  cache_path = tmp_path / "oauth.json"
  config = {
    "oauth": {
      "cache_path": str(cache_path),
      "scopes": ["openid", "email"],
      "callback_port": 8765,
      "client_name": "advisor",
    }
  }
  manager = McpClientManager(config_path=None)

  async def scenario():
    original = manager._build_http_auth("finance-cli", "https://cashnerd.ai/mcp", config)
    assert original is not None
    await original.context.storage.set_tokens(
      OAuthToken(access_token="persisted-token", token_type="Bearer"),
    )
    restored = manager._build_http_auth("finance-cli", "https://cashnerd.ai/mcp", config)
    different_endpoint = manager._build_http_auth(
      "finance-cli", "https://cashnerd.ai/other-mcp", config,
    )
    assert different_endpoint is not None
    assert await different_endpoint.context.storage.get_tokens() is None

    async def respond(request: httpx2.Request) -> httpx2.Response:
      assert request.headers["Authorization"] == "Bearer persisted-token"
      return httpx2.Response(200, json={"authenticated": True})

    async with httpx2.AsyncClient(
      auth=restored, transport=httpx2.MockTransport(respond),
    ) as client:
      response = await client.get("https://cashnerd.ai/mcp")
    assert response.json() == {"authenticated": True}

  _run(scenario())


_CLAIM_KEY = "per-user-http-claim-test-key-at-least-32-bytes"
_RISK_URL = "https://risk.example.test"
_PER_USER_RISK_ENTRY = {
  "type": "streamable-http",
  "url": "${RISK_API_URL}/api/mcp/portfolio-reads-mcp/",
  "per_user": True,
  "run_scoped_env": ["RISK_API_URL"],
}


def _claim_fields(request: httpx2.Request) -> dict[str, str]:
  return {
    field: request.headers[header]
    for field, header in AGENT_API_CLAIM_HEADERS.items()
  }


def _claim_is_signed_with_test_key(fields: dict[str, str]) -> bool:
  canonical = canonical_agent_api_claim_bytes(
    audience=fields["audience"],
    issued_at=int(fields["issued_at"]),
    expiry=int(fields["expiry"]),
    user_id=fields["user_id"],
    user_email=fields["user_email"],
    nonce=fields["nonce"],
  )
  expected = hmac.new(_CLAIM_KEY.encode(), canonical, hashlib.sha256).hexdigest()
  return hmac.compare_digest(expected, fields["signature"])


def _risk_mcp_endpoint(requests: list[httpx2.Request]):
  """A stateless streamable-HTTP MCP server with one `get_positions` tool."""

  async def respond(request: httpx2.Request) -> httpx2.Response:
    requests.append(request)
    if request.method == "GET":
      return httpx2.Response(405)
    if request.method == "DELETE":
      return httpx2.Response(200)
    payload = json.loads(request.content)
    if payload["method"] == "notifications/initialized":
      return httpx2.Response(202)
    if payload["method"] == "initialize":
      result = {
        "protocolVersion": "2025-11-25",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "portfolio-reads-mcp", "version": "1"},
      }
    elif payload["method"] == "tools/list":
      result = {
        "tools": [{
          "name": "get_positions",
          "description": "Positions",
          "inputSchema": {"type": "object", "properties": {}},
        }],
      }
    else:
      assert payload["method"] == "tools/call"
      result = {"content": [{"type": "text", "text": "positions"}], "isError": False}
    return httpx2.Response(
      200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
    )

  return respond


def _risk_session(user_id: int, email: str) -> GatewaySession:
  return GatewaySession(
    session_id=f"session-{user_id}",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id=str(user_id),
    user_email=email,
    risk_user_id=user_id,
    owner_user_id=str(user_id),
    role="owner",
  )


def _serve_risk_mcp(monkeypatch, requests: list[httpx2.Request]) -> None:
  def http_client(**kwargs):
    return httpx2.AsyncClient(
      transport=httpx2.MockTransport(_risk_mcp_endpoint(requests)), **kwargs,
    )

  monkeypatch.setattr(
    mcp_client_module, "httpx2",
    SimpleNamespace(AsyncClient=http_client, Timeout=httpx2.Timeout),
  )
  monkeypatch.setattr(
    claim_authority_module, "_configured_authority",
    GatewayClaimSigningAuthority(_CLAIM_KEY),
  )
  monkeypatch.setenv("RISK_API_URL", _RISK_URL)
  monkeypatch.delenv("AGENT_API_CLAIM_TTL_SECONDS", raising=False)


def test_per_user_http_entry_signs_each_request_for_the_sessions_user(monkeypatch) -> None:
  requests: list[httpx2.Request] = []
  _serve_risk_mcp(monkeypatch, requests)
  alice = _risk_session(7, "alice@example.test")
  bob = _risk_session(8, "bob@example.test")

  async def scenario():
    manager = McpClientManager(
      config_path=None,
      startup_timeout=5,
      inline_servers={"portfolio-reads-mcp": _PER_USER_RISK_ENTRY},
    )
    await manager.startup()
    try:
      (tool,) = [definition["name"] for definition in manager.get_tool_definitions()]
      for session in (alice, alice, bob):
        result, error = await manager.call_tool(tool, {}, gateway_session=session)
        assert error is None
    finally:
      await manager.shutdown()

  _run(scenario())

  assert requests
  assert {str(request.url) for request in requests} == {
    f"{_RISK_URL}/api/mcp/portfolio-reads-mcp/"
  }
  claims = [_claim_fields(request) for request in requests]
  assert all(_claim_is_signed_with_test_key(claim) for claim in claims)
  assert len({claim["nonce"] for claim in claims}) == len(claims)

  def calls_of(user_id: str) -> list[httpx2.Request]:
    return [
      request for request in requests
      if request.headers[AGENT_API_CLAIM_HEADERS["user_id"]] == user_id
      and json.loads(request.content or b"{}").get("method") == "tools/call"
    ]

  def initializations_of(user_id: str) -> list[httpx2.Request]:
    return [
      request for request in requests
      if request.headers[AGENT_API_CLAIM_HEADERS["user_id"]] == user_id
      and json.loads(request.content or b"{}").get("method") == "initialize"
    ]

  # Two calls on one connection: one handshake, two distinct fresh claims.
  assert len(initializations_of("7")) == 1
  assert len(calls_of("7")) == 2
  assert {
    request.headers[AGENT_API_CLAIM_HEADERS["user_email"]]
    for request in calls_of("7")
  } == {"alice@example.test"}
  # A second user gets a connection and claims of their own.
  assert len(initializations_of("8")) == 1
  assert [
    request.headers[AGENT_API_CLAIM_HEADERS["user_email"]]
    for request in calls_of("8")
  ] == ["bob@example.test"]


def test_per_user_http_claims_expire_after_the_agent_api_claim_ttl(monkeypatch) -> None:
  requests: list[httpx2.Request] = []
  _serve_risk_mcp(monkeypatch, requests)
  monkeypatch.setenv("AGENT_API_CLAIM_TTL_SECONDS", "123")

  async def scenario():
    manager = McpClientManager(
      config_path=None,
      startup_timeout=5,
      inline_servers={"portfolio-reads-mcp": _PER_USER_RISK_ENTRY},
    )
    await manager.startup()
    await manager.shutdown()

  _run(scenario())

  assert requests
  for claim in map(_claim_fields, requests):
    assert int(claim["expiry"]) - int(claim["issued_at"]) == 123


def test_streamable_http_refuses_undeclared_ambient_url_variable(monkeypatch) -> None:
  monkeypatch.setenv("RISK_API_URL", "https://ambient.example.test")
  monkeypatch.setattr(
    claim_authority_module, "_configured_authority",
    GatewayClaimSigningAuthority(_CLAIM_KEY),
  )
  connected = False

  def http_client(**_kwargs):
    nonlocal connected
    connected = True
    raise AssertionError("an undeclared url variable must fail before connecting")

  monkeypatch.setattr(
    mcp_client_module, "httpx2",
    SimpleNamespace(AsyncClient=http_client, Timeout=httpx2.Timeout),
  )
  undeclared = {
    key: value for key, value in _PER_USER_RISK_ENTRY.items()
    if key != "run_scoped_env"
  }

  async def scenario():
    manager = McpClientManager(config_path=None, startup_timeout=1)
    with pytest.raises(McpEnvExpansionError, match="RISK_API_URL"):
      await manager._connect("portfolio-reads-mcp", undeclared)

  _run(scenario())
  assert connected is False


def test_per_user_http_connect_refuses_unset_risk_api_url(monkeypatch) -> None:
  """The connect path must use the refusing HTTP expander, not expand_env_refs.

  A declared-but-unset ${RISK_API_URL} is "" in the scoped env. expand_env_refs
  would turn the url into a relative path; expand_http_template names the
  variable and refuses before any client is built.
  """
  monkeypatch.delenv("RISK_API_URL", raising=False)
  monkeypatch.setattr(
    claim_authority_module, "_configured_authority",
    GatewayClaimSigningAuthority(_CLAIM_KEY),
  )
  connected = False

  def http_client(**_kwargs):
    nonlocal connected
    connected = True
    raise AssertionError("unset RISK_API_URL must fail before a relative-URL connect")

  monkeypatch.setattr(
    mcp_client_module, "httpx2",
    SimpleNamespace(AsyncClient=http_client, Timeout=httpx2.Timeout),
  )

  async def scenario():
    manager = McpClientManager(config_path=None, startup_timeout=1)
    with pytest.raises(
      McpEnvExpansionError,
      match="undeclared environment variable: RISK_API_URL",
    ):
      await manager._connect("portfolio-reads-mcp", dict(_PER_USER_RISK_ENTRY))

  _run(scenario())
  assert connected is False


def test_per_user_streamable_http_entry_cannot_also_configure_oauth() -> None:
  manager = McpClientManager(config_path=None)
  with pytest.raises(ValueError, match="oauth"):
    manager._prepare_server_config_for_startup(
      {**_PER_USER_RISK_ENTRY, "oauth": True}
    )


def test_manager_built_for_one_user_signs_every_request_as_that_user(monkeypatch) -> None:
  requests: list[httpx2.Request] = []
  _serve_risk_mcp(monkeypatch, requests)
  run_signer = claim_authority_module.configured_gateway_claim_signing_authority(
    required=True,
  ).bind_user(user_id="42", user_email="run@example.test")

  async def scenario():
    manager = McpClientManager(
      config_path=None,
      startup_timeout=5,
      inline_servers={"portfolio-reads-mcp": _PER_USER_RISK_ENTRY},
      user_claim_signer=run_signer,
    )
    await manager.startup()
    try:
      (tool,) = [definition["name"] for definition in manager.get_tool_definitions()]
      result, error = await manager.call_tool(
        tool, {}, gateway_session=_risk_session(7, "alice@example.test"),
      )
      assert error is None
    finally:
      await manager.shutdown()

  _run(scenario())

  claims = [_claim_fields(request) for request in requests]
  assert len(claims) > 2
  assert {(claim["user_id"], claim["user_email"]) for claim in claims} == {
    ("42", "run@example.test"),
  }
  assert all(_claim_is_signed_with_test_key(claim) for claim in claims)


def test_bound_signer_manager_signs_as_the_bound_user_for_a_session_without_a_per_user_subject(
  monkeypatch,
) -> None:
  # A run, the autonomous runner and a placed runtime dispatch with a gateway
  # session that carries no canonical Risk subject; the bound signer is the
  # only identity such a manager's per-user HTTP server is called as.
  requests: list[httpx2.Request] = []
  _serve_risk_mcp(monkeypatch, requests)
  run_signer = claim_authority_module.configured_gateway_claim_signing_authority(
    required=True,
  ).bind_user(user_id="42", user_email="run@example.test")
  run_session = GatewaySession(
    session_id="run-session",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id="run-user",
    role="owner",
  )

  async def scenario():
    manager = McpClientManager(
      config_path=None,
      startup_timeout=5,
      inline_servers={"portfolio-reads-mcp": _PER_USER_RISK_ENTRY},
      user_claim_signer=run_signer,
    )
    await manager.startup()
    try:
      (tool,) = [definition["name"] for definition in manager.get_tool_definitions()]
      return await manager.call_tool(tool, {}, gateway_session=run_session)
    finally:
      await manager.shutdown()

  result, error = _run(scenario())

  assert error is None
  calls = [
    request for request in requests
    if json.loads(request.content or b"{}").get("method") == "tools/call"
  ]
  assert len(calls) == 1
  assert {
    (claim["user_id"], claim["user_email"])
    for claim in map(_claim_fields, requests)
  } == {("42", "run@example.test")}


def test_changed_claim_identity_rotates_the_per_user_http_connection(monkeypatch) -> None:
  requests: list[httpx2.Request] = []
  _serve_risk_mcp(monkeypatch, requests)
  before = _risk_session(7, "alice@example.test")
  after = _risk_session(7, "alice.renamed@example.test")
  drain_reasons: list[str] = []

  async def scenario():
    manager = McpClientManager(
      config_path=None,
      startup_timeout=5,
      inline_servers={"portfolio-reads-mcp": _PER_USER_RISK_ENTRY},
    )
    drain = manager._schedule_drain

    def record_drain(state, reason):
      drain_reasons.append(reason)
      drain(state, reason)

    manager._schedule_drain = record_drain
    await manager.startup()
    try:
      (tool,) = [definition["name"] for definition in manager.get_tool_definitions()]
      for session in (before, before, after):
        _result, error = await manager.call_tool(tool, {}, gateway_session=session)
        assert error is None
    finally:
      await manager.shutdown()

  _run(scenario())

  user_initializations = [
    request for request in requests
    if json.loads(request.content or b"{}").get("method") == "initialize"
    and request.headers[AGENT_API_CLAIM_HEADERS["user_id"]] == "7"
  ]
  assert [
    request.headers[AGENT_API_CLAIM_HEADERS["user_email"]]
    for request in user_initializations
  ] == ["alice@example.test", "alice.renamed@example.test"]
  assert drain_reasons == ["binding_changed"]


def test_autonomous_claim_signer_signs_every_http_request_with_a_fresh_broker_claim(
  monkeypatch,
) -> None:
  from hank_contracts.agent_api import AGENT_API_CLAIM_ENV_VARS
  from agent_gateway.autonomous_claim_broker import AutonomousClaimSigner

  requests: list[httpx2.Request] = []
  _serve_risk_mcp(monkeypatch, requests)
  signer = object.__new__(AutonomousClaimSigner)
  signer._user_id = "42"
  signer._user_email = "child@example.test"
  calls: list[tuple[object, int]] = []

  def sign_claim(self: AutonomousClaimSigner, *, ttl_seconds: int) -> dict[str, str]:
    calls.append((self, ttl_seconds))
    nonce = f"broker-nonce-{len(calls)}"
    return {
      AGENT_API_CLAIM_ENV_VARS["audience"]: "agent-api",
      AGENT_API_CLAIM_ENV_VARS["issued_at"]: "1",
      AGENT_API_CLAIM_ENV_VARS["expiry"]: "2",
      AGENT_API_CLAIM_ENV_VARS["user_id"]: "42",
      AGENT_API_CLAIM_ENV_VARS["user_email"]: "child@example.test",
      AGENT_API_CLAIM_ENV_VARS["nonce"]: nonce,
      AGENT_API_CLAIM_ENV_VARS["signature"]: "sig",
    }

  monkeypatch.setattr(AutonomousClaimSigner, "sign_claim", sign_claim)

  async def scenario():
    manager = McpClientManager(
      config_path=None,
      startup_timeout=5,
      inline_servers={"portfolio-reads-mcp": _PER_USER_RISK_ENTRY},
      user_claim_signer=signer,
    )
    await manager.startup()
    try:
      (tool,) = [definition["name"] for definition in manager.get_tool_definitions()]
      result, error = await manager.call_tool(
        tool, {}, gateway_session=_risk_session(7, "alice@example.test"),
      )
      assert error is None
      assert result is not None
    finally:
      await manager.shutdown()

  _run(scenario())

  assert requests
  assert len(calls) == len(requests)
  assert len(calls) > 1
  assert all(recorded_signer is signer for recorded_signer, _ttl in calls)
  assert len({
    request.headers[AGENT_API_CLAIM_HEADERS["nonce"]] for request in requests
  }) == len(requests)
  assert {
    request.headers[AGENT_API_CLAIM_HEADERS["user_id"]] for request in requests
  } == {"42"}
  assert {
    request.headers[AGENT_API_CLAIM_HEADERS["user_email"]] for request in requests
  } == {"child@example.test"}


def test_per_user_http_capacity_error_is_not_a_sheets_error(monkeypatch) -> None:
  requests: list[httpx2.Request] = []
  _serve_risk_mcp(monkeypatch, requests)
  monkeypatch.setattr(mcp_client_module, "PER_USER_INSTANCE_CAP", 1)

  async def scenario():
    manager = McpClientManager(
      config_path=None,
      startup_timeout=5,
      inline_servers={"portfolio-reads-mcp": _PER_USER_RISK_ENTRY},
    )
    await manager.startup()
    held = None
    try:
      first = mcp_client_module._PerUserGatewaySubject.from_gateway_session(
        _risk_session(7, "alice@example.test"),
      )
      second = mcp_client_module._PerUserGatewaySubject.from_gateway_session(
        _risk_session(8, "bob@example.test"),
      )
      held = await manager._get_per_user_server("portfolio-reads-mcp", first)
      held.active_calls = 1
      with pytest.raises(mcp_client_module._PerUserMcpError) as raised:
        await manager._get_per_user_server("portfolio-reads-mcp", second)
      return raised.value
    finally:
      if held is not None:
        held.active_calls = 0
      await manager.shutdown()

  error = _run(scenario())

  assert error.code == "mcp_user_authority_unavailable"
  assert "Google Sheets" not in str(error)
  assert "1 of 1 instances in use" in str(error)


def test_startup_reports_an_unreachable_per_user_http_server_and_keeps_the_others(
  monkeypatch,
) -> None:
  requests: list[httpx2.Request] = []
  serve = _risk_mcp_endpoint(requests)

  async def respond(request: httpx2.Request) -> httpx2.Response:
    if request.url.host == "refused.example.test":
      raise httpx2.ConnectError("All connection attempts failed", request=request)
    return await serve(request)

  def http_client(**kwargs):
    return httpx2.AsyncClient(transport=httpx2.MockTransport(respond), **kwargs)

  monkeypatch.setattr(
    mcp_client_module, "httpx2",
    SimpleNamespace(AsyncClient=http_client, Timeout=httpx2.Timeout),
  )
  monkeypatch.setattr(
    claim_authority_module, "_configured_authority",
    GatewayClaimSigningAuthority(_CLAIM_KEY),
  )
  monkeypatch.setenv("RISK_API_URL", "https://refused.example.test")

  async def scenario():
    manager = McpClientManager(
      config_path=None,
      startup_timeout=5,
      inline_servers={
        "research-corpus-mcp": {
          "type": "streamable-http",
          "url": "${RISK_API_URL}/api/mcp/research-mcp/",
          "per_user": True,
          "run_scoped_env": ["RISK_API_URL"],
        },
        "other-mcp": {
          "type": "streamable-http",
          "url": "https://other.example.test/mcp/",
        },
      },
    )
    await manager.startup()
    try:
      return (
        [definition["name"] for definition in manager.get_tool_definitions()],
        manager.get_startup_diagnostics(),
      )
    finally:
      await manager.shutdown()

  tools, diagnostics = _run(scenario())

  assert tools == ["get_positions"]
  assert set(diagnostics) == {"research-corpus-mcp"}
  assert "All connection attempts failed" in diagnostics["research-corpus-mcp"]["message"]
