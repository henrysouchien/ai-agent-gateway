from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx2
import pytest
from hank_contracts.agent_api import (
  AGENT_API_CLAIM_AUDIENCE,
  AGENT_API_CLAIM_ENV_VARS,
)

import agent_gateway.claim_mint_client as claim_mint_module
import agent_gateway.claim_signing_authority as claim_authority_module
import agent_gateway.mcp_client as mcp_client_module
import agent_gateway.mcp_client_connections as connection_helpers
from agent_gateway.autonomous_claim_broker import AutonomousClaimSigner
from agent_gateway.claim_signing_authority import GatewayClaimSigningAuthority
from .test_mcp_client_per_user import _definition, _gateway_session, _manager, _subject
from .test_mcp_streamable_http import (
  _CLAIM_KEY,
  _claim_fields,
  _claim_is_signed_with_test_key,
  _risk_mcp_endpoint,
)


_MINT_URL = "https://gateway.example.test"
_MCP_URL = "https://risk.example.test/mcp"
_ROSTER_KEY = "per-user-roster-key"


def _http_manager():
  manager = _manager(operation="get_positions")
  definition = _definition({
    "type": "streamable-http",
    "url": _MCP_URL,
    "per_user": True,
    "run_scoped_env": ["AGENT_API_CLAIM_MINT_URL", "GATEWAY_API_KEY"],
  })
  definition.name = "remote-mcp"
  manager._servers = {"remote-mcp": definition}
  manager._tool_to_server = {"tool": "remote-mcp"}
  return manager


def _remote_claim(*, user_id: str, user_email: str, generation: int) -> dict[str, str]:
  fields = {
    "audience": AGENT_API_CLAIM_AUDIENCE,
    "issued_at": "100",
    "expiry": "400",
    "user_id": user_id,
    "user_email": user_email,
    "nonce": f"remote-nonce-{generation}",
    "signature": f"remote-signature-{generation}",
  }
  return {AGENT_API_CLAIM_ENV_VARS[field]: value for field, value in fields.items()}


def _unexpected_local_authority(*_args, **_kwargs):
  raise AssertionError("remote mint must not access a local signer or authority")


def test_per_user_tool_call_remotely_mints_fresh_subject_claims_without_local_authority(
  monkeypatch,
):
  session = _gateway_session()
  session.user_email = "alice@example.test"
  assert session.user_email is not None
  email = session.user_email
  assert isinstance(email, str)
  mint_requests: list[httpx2.Request] = []
  mcp_requests: list[httpx2.Request] = []
  issued_claims: list[dict[str, str]] = []

  async def mint_endpoint(request: httpx2.Request) -> httpx2.Response:
    mint_requests.append(request)
    assert request.method == "POST"
    assert str(request.url) == f"{_MINT_URL}/api/claims/mint"
    assert request.headers["Authorization"] == f"Bearer {_ROSTER_KEY}"
    # Identity is selected by the roster bearer, never supplied in a mint body.
    assert request.content == b""
    claim = _remote_claim(
      user_id=session.user_id,
      user_email=email,
      generation=len(mint_requests),
    )
    issued_claims.append(claim)
    return httpx2.Response(200, json=claim)

  def mint_http_client(**kwargs):
    return httpx2.AsyncClient(
      transport=httpx2.MockTransport(mint_endpoint), **kwargs,
    )

  def mcp_http_client(**kwargs):
    return httpx2.AsyncClient(
      transport=httpx2.MockTransport(_risk_mcp_endpoint(mcp_requests)), **kwargs,
    )

  monkeypatch.setenv("AGENT_API_CLAIM_MINT_URL", _MINT_URL)
  monkeypatch.setenv("GATEWAY_API_KEY", _ROSTER_KEY)
  monkeypatch.delenv("AGENT_API_USER_CLAIM_HMAC_KEY", raising=False)
  monkeypatch.setattr(claim_authority_module, "_configured_authority", None)
  monkeypatch.setattr(
    mcp_client_module, "configured_gateway_claim_signing_authority",
    _unexpected_local_authority,
  )
  monkeypatch.setattr(
    mcp_client_module, "sign_gateway_user_claim", _unexpected_local_authority,
  )
  monkeypatch.setattr(AutonomousClaimSigner, "sign_claim", _unexpected_local_authority)
  monkeypatch.setattr(
    claim_mint_module, "httpx2",
    SimpleNamespace(
      AsyncClient=mint_http_client,
      Auth=httpx2.Auth,
      HTTPError=httpx2.HTTPError,
      InvalidURL=httpx2.InvalidURL,
    ),
  )
  monkeypatch.setattr(
    mcp_client_module, "httpx2",
    SimpleNamespace(AsyncClient=mcp_http_client, Timeout=httpx2.Timeout),
  )

  async def scenario():
    manager = _http_manager()
    assert manager._user_claim_signer is None
    try:
      # Retain the real call -> acquire -> spawn -> host -> HTTP/SDK path;
      # only the two external HTTP endpoints are replaced by transports.
      for _ in range(2):
        result, error = await manager.call_tool("tool", {}, gateway_session=session)
        assert error is None
        assert result == {"text": "positions"}
    finally:
      await manager.shutdown()

  asyncio.run(scenario())

  assert claim_authority_module._configured_authority is None
  assert len(mcp_requests) > 2
  assert len(mint_requests) == len(mcp_requests)
  claims = [_claim_fields(request) for request in mcp_requests]
  assert {(claim["user_id"], claim["user_email"]) for claim in claims} == {
    (session.user_id, session.user_email),
  }
  assert len({claim["nonce"] for claim in claims}) == len(claims)
  assert {claim["signature"] for claim in claims} == {
    claim[AGENT_API_CLAIM_ENV_VARS["signature"]] for claim in issued_claims
  }
  assert all("Authorization" not in request.headers for request in mcp_requests)
  methods = [json.loads(request.content or b"{}").get("method") for request in mcp_requests]
  assert methods.count("initialize") == 1
  assert methods.count("tools/call") == 2



@pytest.mark.parametrize(
  ("stated_user_id", "accepted"),
  [
    pytest.param(1, True, id="positive-int"),
    pytest.param(True, False, id="bool"),
    pytest.param(0, False, id="zero"),
    pytest.param("1", False, id="str"),
    pytest.param(1.0, False, id="float"),
    pytest.param({"a": 1}, False, id="mapping"),
    pytest.param(None, False, id="none"),
  ],
)
def test_standalone_caller_without_gateway_session_mints_remote_claim_from_meta_user_id(
  monkeypatch, stated_user_id, accepted,
):
  mint_requests: list[httpx2.Request] = []
  mcp_requests: list[httpx2.Request] = []

  async def mint_endpoint(request: httpx2.Request) -> httpx2.Response:
    mint_requests.append(request)
    claim = _remote_claim(
      user_id="1",
      user_email="henry@example.test",
      generation=len(mint_requests),
    )
    return httpx2.Response(200, json=claim)

  def mint_http_client(**kwargs):
    return httpx2.AsyncClient(
      transport=httpx2.MockTransport(mint_endpoint), **kwargs,
    )

  def mcp_http_client(**kwargs):
    return httpx2.AsyncClient(
      transport=httpx2.MockTransport(_risk_mcp_endpoint(mcp_requests)), **kwargs,
    )

  monkeypatch.setenv("AGENT_API_CLAIM_MINT_URL", _MINT_URL)
  monkeypatch.setenv("GATEWAY_API_KEY", _ROSTER_KEY)
  monkeypatch.delenv("AGENT_API_USER_CLAIM_HMAC_KEY", raising=False)
  monkeypatch.setattr(claim_authority_module, "_configured_authority", None)
  monkeypatch.setattr(
    mcp_client_module, "configured_gateway_claim_signing_authority",
    _unexpected_local_authority,
  )
  monkeypatch.setattr(
    claim_mint_module, "httpx2",
    SimpleNamespace(
      AsyncClient=mint_http_client,
      Auth=httpx2.Auth,
      HTTPError=httpx2.HTTPError,
      InvalidURL=httpx2.InvalidURL,
    ),
  )
  monkeypatch.setattr(
    mcp_client_module, "httpx2",
    SimpleNamespace(AsyncClient=mcp_http_client, Timeout=httpx2.Timeout),
  )

  async def scenario():
    manager = _http_manager()
    try:
      result, error = await manager.call_tool(
        "tool", {}, meta={"user_id": stated_user_id}, gateway_session=None,
      )
      if accepted:
        assert error is None
        assert result == {"text": "positions"}
        assert ("remote-mcp", "1") in manager._per_user_servers
      else:
        assert result is None
        assert error is not None
        assert error["sub_code"] == "missing_user_identity"
        assert manager._per_user_servers == {}
      missing_result, missing_error = await manager.call_tool(
        "tool", {}, meta={}, gateway_session=None,
      )
      assert missing_result is None
      assert missing_error is not None
      assert missing_error["sub_code"] == "missing_user_identity"
    finally:
      await manager.shutdown()

  asyncio.run(scenario())
  assert len(mint_requests) == len(mcp_requests)
  assert [json.loads(req.content or b"{}").get("method") for req in mcp_requests].count(
    "tools/call"
  ) == int(accepted)

@pytest.mark.parametrize("signer_kind", ["authority", "gateway", "autonomous"])
def test_absent_mint_url_preserves_each_legacy_per_user_signer(monkeypatch, signer_kind):
  monkeypatch.delenv("AGENT_API_CLAIM_MINT_URL", raising=False)
  monkeypatch.setenv("GATEWAY_API_KEY", _ROSTER_KEY)
  monkeypatch.setenv("AGENT_API_CLAIM_TTL_SECONDS", "123")
  authority = GatewayClaimSigningAuthority(_CLAIM_KEY)
  monkeypatch.setattr(claim_authority_module, "_configured_authority", authority)
  monkeypatch.setattr(
    connection_helpers, "mint_gateway_user_claim", _unexpected_local_authority,
  )
  manager = _http_manager()
  expected_user_id, expected_email = "7", "alice@example.test"
  broker_calls = []
  if signer_kind == "gateway":
    expected_user_id, expected_email = "42", "bound@example.test"
    manager._user_claim_signer = authority.bind_user(
      user_id=expected_user_id, user_email=expected_email,
    )
  elif signer_kind == "autonomous":
    expected_user_id, expected_email = "43", "child@example.test"
    signer = object.__new__(AutonomousClaimSigner)
    signer._user_id = expected_user_id
    signer._user_email = expected_email
    manager._user_claim_signer = signer

    def sign_claim(self, *, ttl_seconds):
      broker_calls.append((self, ttl_seconds))
      return _remote_claim(
        user_id=expected_user_id,
        user_email=expected_email,
        generation=len(broker_calls),
      )

    monkeypatch.setattr(AutonomousClaimSigner, "sign_claim", sign_claim)

  requests: list[httpx2.Request] = []

  def respond(request):
    requests.append(request)
    return httpx2.Response(200)

  async def scenario():
    auth = manager._user_claim_http_auth("7", "alice@example.test")
    async with httpx2.AsyncClient(
      auth=auth, transport=httpx2.MockTransport(respond),
    ) as client:
      for _ in range(2):
        response = await client.post(_MCP_URL)
        assert response.status_code == 200

  asyncio.run(scenario())

  claims = [_claim_fields(request) for request in requests]
  assert {(claim["user_id"], claim["user_email"]) for claim in claims} == {
    (expected_user_id, expected_email),
  }
  assert len({claim["nonce"] for claim in claims}) == 2
  if signer_kind == "autonomous":
    assert broker_calls == [(manager._user_claim_signer, 123)] * 2
  else:
    assert all(_claim_is_signed_with_test_key(claim) for claim in claims)
    assert all(int(claim["expiry"]) - int(claim["issued_at"]) == 123 for claim in claims)


def test_absent_mint_url_still_requires_local_authority_when_no_signer(monkeypatch):
  monkeypatch.delenv("AGENT_API_CLAIM_MINT_URL", raising=False)
  monkeypatch.setattr(claim_authority_module, "_configured_authority", None)
  manager = _http_manager()
  with pytest.raises(RuntimeError, match="claim-signing authority is not installed"):
    manager._user_claim_http_auth("7", "alice@example.test")


@pytest.mark.parametrize("mint_url,api_key,error", [
  ("", _ROSTER_KEY, "AGENT_API_CLAIM_MINT_URL"),
  ("  ", _ROSTER_KEY, "AGENT_API_CLAIM_MINT_URL"),
  (_MINT_URL, None, "GATEWAY_API_KEY"),
  (_MINT_URL, "", "GATEWAY_API_KEY"),
  (_MINT_URL, "  ", "GATEWAY_API_KEY"),
])
@pytest.mark.parametrize("configured_signer", [False, True])
def test_invalid_remote_config_refuses_per_user_spawn_without_local_fallback(
  monkeypatch, mint_url, api_key, error, configured_signer,
):
  monkeypatch.setenv("AGENT_API_CLAIM_MINT_URL", mint_url)
  if api_key is None:
    monkeypatch.delenv("GATEWAY_API_KEY", raising=False)
  else:
    monkeypatch.setenv("GATEWAY_API_KEY", api_key)
  manager = _http_manager()
  if configured_signer:
    manager._user_claim_signer = GatewayClaimSigningAuthority(_CLAIM_KEY).bind_user(
      user_id="7", user_email="alice@example.test",
    )
  monkeypatch.setattr(claim_authority_module, "_configured_authority", None)
  monkeypatch.setattr(
    mcp_client_module, "configured_gateway_claim_signing_authority",
    _unexpected_local_authority,
  )
  monkeypatch.setattr(
    mcp_client_module, "sign_gateway_user_claim", _unexpected_local_authority,
  )
  monkeypatch.setattr(
    connection_helpers, "mint_gateway_user_claim", _unexpected_local_authority,
  )
  monkeypatch.setattr(manager, "_connect_streamable_http", _unexpected_local_authority)

  async def scenario():
    with pytest.raises(ValueError, match=error):
      await manager._spawn_per_user_server("remote-mcp", _subject())

  asyncio.run(scenario())
