# ruff: noqa: E402

from __future__ import annotations
from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND

import asyncio
import base64
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_workflow_contracts.models import AdmittedResearchBinding, AdmittedTenant
from agent_gateway.admission import capture_job_context, root_context_from_identity
from agent_gateway.approval_route import NO_APPROVAL_ROUTE
from agent_gateway.execution_identity import DispatchIdentity
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.approval_policy import RunContext
from agent_gateway.investment_capability_claim import (
  INVESTMENT_CAPABILITY_FACADE_TOOLS,
  INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV,
  INVESTMENT_CAPABILITY_CLAIM_PUBLIC_KEY_ENV,
  INVESTMENT_JOB_PRINCIPAL_TOOLS,
  InvestmentCapabilityGrant,
  InvestmentCapabilityClaimError,
  configure_investment_capability_grants,
  investment_capability_grants,
  issue_investment_capability_claim,
  issue_investment_selected_content_claim,
  investment_capability_signing_available,
  issue_investment_job_principal_claim,
)
from agent_gateway.skill_limits import (
  ActiveSkillAdmission,
  SkillExecutionLimits,
)
from agent_gateway.skill_context import reset_current_skill, set_current_skill
from agent_gateway.tool_dispatcher import ToolDispatcher
from agent_gateway.session import GatewaySession
from gateway_test_support.host_policy import owner_session_host_policy


_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
_PRIVATE_KEY_MATERIAL = base64.urlsafe_b64encode(
  _PRIVATE_KEY.private_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PrivateFormat.Raw,
    encryption_algorithm=serialization.NoEncryption(),
  )
).rstrip(b"=").decode("ascii")
_PUBLIC_KEY = _PRIVATE_KEY.public_key()
_PUBLIC_KEY_BYTES = _PUBLIC_KEY.public_bytes(
  encoding=serialization.Encoding.Raw,
  format=serialization.PublicFormat.Raw,
)
_PUBLIC_KEY_MATERIAL = base64.urlsafe_b64encode(
  _PUBLIC_KEY_BYTES
).rstrip(b"=").decode("ascii")
_KEY_ID = f"ed25519-sha256:{hashlib.sha256(_PUBLIC_KEY_BYTES).hexdigest()}"
_POLICY_BUNDLE_HASH = hashlib.sha256(b"test-policy-bundle").hexdigest()
_TENANT = AdmittedTenant(tenant_id="tenant:desk", storage_id="sia-alice", role="owner")

_READ_FACADE_TOOLS = (
  "cancel_investment_run",
  "get_investment_artifact",
  "get_investment_capability",
  "get_investment_run",
  "list_investment_artifacts",
  "list_investment_capabilities",
)
# What the market-scan and quant-research notes declare; the product
# composition root binds the same projection from the compiled catalog.
_DECLARED_GRANTS = {
  "market-scan": InvestmentCapabilityGrant(
    capability_ids=("fingerprint_screen", "insider_buying", "quality_screen"),
    effect_classes=("artifact_write", "read", "state_write"),
    tool_names=(*_READ_FACADE_TOOLS, "start_investment_run"),
  ),
  "quant-research": InvestmentCapabilityGrant(
    capability_ids=("quant_research",),
    effect_classes=("read", "research_write", "state_write"),
    tool_names=(*_READ_FACADE_TOOLS, "start_quant_research"),
  ),
}


@pytest.fixture(autouse=True)
def _bind_declared_grants(owner_session_host_policy):
  previous = investment_capability_grants()
  configure_investment_capability_grants(_DECLARED_GRANTS)
  yield
  configure_investment_capability_grants(previous)



def _limits_for_skill(
  skill: str | None,
) -> SkillExecutionLimits | None:
  if skill is None:
    return None
  if skill == "quant-research":
    return SkillExecutionLimits(32_000)
  return SkillExecutionLimits(None)


def _set_signing_key(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv(
    INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV,
    _PRIVATE_KEY_MATERIAL,
  )


def _decode_segment(value: str) -> dict[str, Any]:
  padded = value + ("=" * (-len(value) % 4))
  return json.loads(base64.urlsafe_b64decode(padded).decode("ascii"))


def test_signer_availability_uses_the_existing_key_loader(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv(INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV, raising=False)
  assert investment_capability_signing_available() is False

  monkeypatch.setenv(INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV, "malformed")
  assert investment_capability_signing_available() is False

  _set_signing_key(monkeypatch)
  assert investment_capability_signing_available() is True


def _decode_claim(token: str) -> tuple[dict[str, Any], dict[str, Any]]:
  header, payload, _signature = token.split(".")
  return _decode_segment(header), _decode_segment(payload)


class _FakeMcpClient(McpClientManager):
  def __init__(
    self,
    *,
    tool_name: str,
    original_tool_name: str | None = None,
  ) -> None:
    super().__init__(config_path=None)
    self.tool_name = tool_name
    self.original_tool_name = original_tool_name or tool_name
    self.calls: list[dict[str, Any]] = []

  def is_mcp_tool(self, name: str) -> bool:
    return name == self.tool_name

  def get_server_for_tool(self, name: str) -> str | None:
    return "idea-workbench-mcp" if name == self.tool_name else None

  def get_original_tool_name(self, name: str) -> str:
    return self.original_tool_name if name == self.tool_name else name

  async def call_tool(
    self,
    name: str,
    tool_input: object,
    meta: dict[str, Any] | None = None,
    abort_event: asyncio.Event | None = None,
    gateway_session: object | None = None,
    allow_uncertain_replay: bool = True,
    trusted_dispatch_scope: Mapping[str, object] | None = None,
  ) -> tuple[object | None, dict[str, object] | None]:
    _ = abort_event, gateway_session, allow_uncertain_replay, trusted_dispatch_scope
    self.calls.append({"name": name, "tool_input": tool_input, "meta": meta})
    return {"ok": True}, None

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


def _dispatcher(
  mcp: _FakeMcpClient,
  *,
  skill: str | None = "quant-research",
  policy_bundle_hash: str = _POLICY_BUNDLE_HASH,
  research_file_id: int | None = 2,
) -> ToolDispatcher:
  session = GatewaySession(
    session_id="session-123",
    api_key_hash="test-api-key-hash",
    created_at=1_800_000_000,
    expires_at=1_800_000_600,
    user_id=_TENANT.storage_id,
    owner_user_id=_TENANT.storage_id,
    tenant_id=_TENANT.tenant_id,
    risk_user_id=42,
    channel="excel",
    role="owner",
  )
  capture_job_context(
    session,
    root_context_from_identity(
      tenant=_TENANT,
      job_id="agent-run-123",
      tool_ids=None,
      withheld_tools=frozenset(),
      research_binding=AdmittedResearchBinding(research_file_id=2),
      approval_route=NO_APPROVAL_ROUTE,
    ),
  )
  return ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=mcp,
    session_id="session-123",
    user_id="sia-alice",
    risk_user_id=42,
    channel="excel",
    role="owner",
    mcp_meta_inject_servers=frozenset({"idea-workbench-mcp"}),
    session=session,
    run_context=RunContext(
      user_id="sia-alice",
      request_id="request-123",
      session_id="session-123",
      run_id="agent-run-123",
      profile=skill or "analyst",
      channel="excel",
      skill=skill,
      admitted_skill_execution_limits=_limits_for_skill(skill),
      research_file_id=research_file_id,
      policy_bundle_hash=policy_bundle_hash,
    ),
  )


def _dispatch(
  dispatcher: ToolDispatcher,
  *,
  tool_call_id: str,
  tool_name: str,
  tool_input: dict[str, Any] | None = None,
):
  if tool_input is None:
    tool_input = (
      {"request": {"research_file_id": 2}}
      if tool_name == "start_quant_research"
      else {}
    )
  return asyncio.run(
    dispatcher.dispatch(
      tool_call_id,
      tool_name,
      tool_input,
      advertised_tool_names=frozenset({tool_name}),
      skill_run_id="skill-run-123",
    )
  )


def _assert_names_grant_and_issuance_route(error: dict[str, Any], *, subject: str) -> None:
  assert error["code"] == "investment_capability_claim_unavailable"
  message = error["message"]
  assert subject in message
  assert "market-scan" in message
  assert "quant-research" in message
  assert "run_agent" in message
  assert "agent-operation/market-scan" in message
  assert "agent-operation/quant-research" in message
  assert error["required_grants"] == ["market-scan", "quant-research"]
  assert error["route"] == "run_agent"
  assert error["operations"] == [
    "agent-operation/market-scan",
    "agent-operation/quant-research",
  ]


def test_claim_unavailable_error_names_grant_and_issuance_route() -> None:
  from agent_gateway.investment_capability_claim import (
    investment_capability_claim_unavailable_error,
  )

  error = investment_capability_claim_unavailable_error(
    subject="tool 'get_investment_run'",
  )
  _assert_names_grant_and_issuance_route(
    error,
    subject="tool 'get_investment_run'",
  )


def test_facade_tool_boundary_is_exact() -> None:
  assert INVESTMENT_CAPABILITY_FACADE_TOOLS == frozenset({
    "cancel_investment_run",
    "get_investment_artifact",
    "get_investment_capability",
    "get_investment_run",
    "list_investment_artifacts",
    "list_investment_capabilities",
    "start_investment_run",
    "start_quant_research",
  })


def test_selected_content_claim_is_the_exact_minimal_v2_contract(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _set_signing_key(monkeypatch)

  token = issue_investment_selected_content_claim(
    user_id="sia-alice",
    artifact_id="artifact:quant-1",
    view="excerpt",
    now=1_800_000_000,
  )

  header, payload = _decode_claim(token)
  assert header == {"alg": "EdDSA", "kid": _KEY_ID, "typ": "JWT"}
  assert payload == {
    "artifact_id": "artifact:quant-1",
    "aud": "investment-tools",
    "exp": 1_800_000_060,
    "iat": 1_800_000_000,
    "iss": "ai-excel-addin",
    "purpose": "selected_content_read",
    "sub": "sia-alice",
    "tool_name": "get_investment_artifact",
    "v": 2,
    "view": "excerpt",
  }
  header_segment, payload_segment, signature_segment = token.split(".")
  signature = base64.urlsafe_b64decode(
    signature_segment + "=" * (-len(signature_segment) % 4)
  )
  _PUBLIC_KEY.verify(
    signature,
    f"{header_segment}.{payload_segment}".encode("ascii"),
  )


@pytest.mark.parametrize(
  ("artifact_id", "view", "now"),
  [
    ("", "summary", 1_800_000_000),
    ("artifact-1", "summary", True),
  ],
)
def test_selected_content_claim_rejects_noncanonical_coordinates(
  monkeypatch: pytest.MonkeyPatch,
  artifact_id: str,
  view: str,
  now: object,
) -> None:
  _set_signing_key(monkeypatch)

  with pytest.raises(InvestmentCapabilityClaimError):
    issue_investment_selected_content_claim(
      user_id="sia-alice",
      artifact_id=artifact_id,
      view=view,
      now=now,  # type: ignore[arg-type]
    )


def test_claim_grant_is_the_bound_skill_declaration(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _set_signing_key(monkeypatch)
  configure_investment_capability_grants({
    "screen-digest": InvestmentCapabilityGrant(
      capability_ids=("quality_screen",),
      effect_classes=("read",),
      tool_names=("get_investment_run",),
    ),
  })
  inputs: dict[str, Any] = {
    "user_id": "42",
    "tenant": _TENANT,
    "session_id": "session-123",
    "skill_run_id": "skill-run-123",
    "channel": "excel",
    "request_id": "request-123",
    "jti": "tool-call-declared",
    "admitted_skill_execution_limits": SkillExecutionLimits(None),
    "policy_bundle_hash": _POLICY_BUNDLE_HASH,
  }

  token = issue_investment_capability_claim(
    **inputs,
    tool_name="get_investment_run",
    skill="screen-digest",
  )
  _header, payload = _decode_claim(token)
  assert payload["allowed_capability_ids"] == ["quality_screen"]
  assert payload["effect_classes"] == ["read"]
  assert payload["skill"] == "screen-digest"

  with pytest.raises(InvestmentCapabilityClaimError, match="unsupported tool"):
    issue_investment_capability_claim(
      **inputs,
      tool_name="start_investment_run",
      skill="screen-digest",
    )
  # No gateway-side list survives the binding: an undeclared skill holds no grant.
  with pytest.raises(InvestmentCapabilityClaimError, match="unsupported skill"):
    issue_investment_capability_claim(
      **inputs,
      tool_name="get_investment_run",
      skill="market-scan",
    )
  assert "approved_budget" not in inspect.signature(
    issue_investment_capability_claim
  ).parameters


def test_market_scan_claim_is_exact_and_has_no_quant_budget_or_origin(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _set_signing_key(monkeypatch)

  token = issue_investment_capability_claim(
    user_id="42",
    tenant=_TENANT,
    session_id="session-123",
    skill_run_id="skill-run-123",
    channel="excel",
    tool_name="start_investment_run",
    request_id="request-123",
    jti="tool-call-market-scan",
    skill="market-scan",
    admitted_skill_execution_limits=_limits_for_skill("market-scan"),
    policy_bundle_hash=_POLICY_BUNDLE_HASH,
    now=1_800_000_000,
  )

  _header, payload = _decode_claim(token)
  assert payload == {
    "allowed_capability_ids": [
      "fingerprint_screen",
      "insider_buying",
      "quality_screen",
    ],
    "aud": "investment-tools",
    "channel": "excel",
    "effect_classes": ["artifact_write", "read", "state_write"],
    "exp": 1_800_000_060,
    "iat": 1_800_000_000,
    "iss": "ai-excel-addin",
    "jti": "tool-call-market-scan",
    "kind": "facade",
    "policy_bundle_hash": _POLICY_BUNDLE_HASH,
    "request_id": "request-123",
    "session_id": "session-123",
    "skill": "market-scan",
    "skill_run_id": "skill-run-123",
    "sub": "42",
    "tenant": _TENANT.model_dump(mode="json"),
    "tool_name": "start_investment_run",
    "user_id": "42",
    "v": 4,
  }


@pytest.mark.parametrize(
  ("skill", "tool_name", "research_file_id"),
  [
    ("market-scan", "start_quant_research", 2),
    ("quant-research", "start_investment_run", None),
  ],
)
def test_cross_skill_submission_tools_fail_closed_before_signing(
  monkeypatch: pytest.MonkeyPatch,
  skill: str,
  tool_name: str,
  research_file_id: int | None,
) -> None:
  _set_signing_key(monkeypatch)

  with pytest.raises(
    InvestmentCapabilityClaimError,
    match="unsupported tool",
  ):
    issue_investment_capability_claim(
      user_id="42",
      tenant=_TENANT,
      session_id="session-123",
      skill_run_id="skill-run-123",
      channel="excel",
      tool_name=tool_name,
      request_id="request-123",
      jti="tool-call-cross-skill",
      skill=skill,
      admitted_skill_execution_limits=_limits_for_skill(skill),
      policy_bundle_hash=_POLICY_BUNDLE_HASH,
      research_file_id=research_file_id,
    )


def test_claim_is_canonical_signed_short_lived_and_exactly_bound(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _set_signing_key(monkeypatch)

  token = issue_investment_capability_claim(
    user_id="42",
    tenant=_TENANT,
    session_id="session-123",
    skill_run_id="skill-run-123",
    channel="excel",
    tool_name="start_quant_research",
    request_id="request-123",
    jti="tool-call-123",
    skill="quant-research",
    admitted_skill_execution_limits=_limits_for_skill("quant-research"),
    policy_bundle_hash=_POLICY_BUNDLE_HASH,
    research_file_id=2,
    now=1_800_000_000,
  )

  header, payload = _decode_claim(token)
  assert header == {"alg": "EdDSA", "kid": _KEY_ID, "typ": "JWT"}
  assert payload == {
    "allowed_capability_ids": ["quant_research"],
    "approved_budget": {
      "max_tokens": 32_000,
      "max_wall_clock_seconds": 5_400,
    },
    "aud": "investment-tools",
    "channel": "excel",
    "effect_classes": ["read", "research_write", "state_write"],
    "exp": 1_800_000_060,
    "iat": 1_800_000_000,
    "iss": "ai-excel-addin",
    "jti": "tool-call-123",
    "kind": "facade",
    "policy_bundle_hash": _POLICY_BUNDLE_HASH,
    "request_id": "request-123",
    "research_file_id": 2,
    "session_id": "session-123",
    "skill": "quant-research",
    "skill_run_id": "skill-run-123",
    "sub": "42",
    "tenant": _TENANT.model_dump(mode="json"),
    "tool_name": "start_quant_research",
    "user_id": "42",
    "v": 4,
  }
  signing_input, supplied_signature = token.rsplit(".", 1)
  signature_bytes = base64.urlsafe_b64decode(
    supplied_signature + ("=" * (-len(supplied_signature) % 4))
  )
  _PUBLIC_KEY.verify(signature_bytes, signing_input.encode("ascii"))
  assert _PRIVATE_KEY_MATERIAL not in token


@pytest.mark.parametrize("tool_name", sorted(INVESTMENT_CAPABILITY_FACADE_TOOLS))
def test_dispatch_fails_closed_for_every_facade_tool_without_private_key(
  monkeypatch: pytest.MonkeyPatch,
  tool_name: str,
) -> None:
  monkeypatch.delenv(
    INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV,
    raising=False,
  )
  mcp = _FakeMcpClient(tool_name=tool_name)

  result, error = _dispatch(
    _dispatcher(mcp),
    tool_call_id=f"call-{tool_name}",
    tool_name=tool_name,
  )

  assert result is None
  assert error is not None
  _assert_names_grant_and_issuance_route(
    error,
    subject=f"tool '{tool_name}'",
  )
  assert mcp.calls == []


def test_dispatch_fails_closed_for_malformed_private_key_without_leaking_it(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  malformed_key = "malformed-private-key"
  monkeypatch.setenv(
    INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV,
    malformed_key,
  )
  mcp = _FakeMcpClient(tool_name="list_investment_capabilities")

  result, error = _dispatch(
    _dispatcher(mcp),
    tool_call_id="call-weak",
    tool_name="list_investment_capabilities",
  )

  assert result is None
  assert error is not None
  assert error["code"] == "investment_capability_claim_unavailable"
  assert malformed_key not in json.dumps(error)
  assert mcp.calls == []


def test_dispatch_injects_claim_bound_to_original_routed_tool(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _set_signing_key(monkeypatch)
  routed_name = "idea_workbench__get_investment_run"
  mcp = _FakeMcpClient(
    tool_name=routed_name,
    original_tool_name="get_investment_run",
  )

  result, error = _dispatch(
    _dispatcher(mcp),
    tool_call_id="call-exact-tool",
    tool_name=routed_name,
  )

  assert error is None
  assert result == {"ok": True}
  assert len(mcp.calls) == 1
  meta = mcp.calls[0]["meta"]
  assert isinstance(meta, dict)
  header, payload = _decode_claim(meta["investment_capability_claim"])
  assert header == {"alg": "EdDSA", "kid": _KEY_ID, "typ": "JWT"}
  assert payload["tool_name"] == "get_investment_run"
  assert payload["jti"] == "call-exact-tool"
  assert payload["request_id"] == "request-123"
  assert payload["sub"] == payload["user_id"] == "42"
  assert payload["session_id"] == "session-123"
  assert payload["skill_run_id"] == "skill-run-123"
  assert payload["channel"] == "excel"
  assert payload["skill"] == "quant-research"
  assert payload["allowed_capability_ids"] == ["quant_research"]
  assert payload["approved_budget"] == {
    "max_tokens": 32_000,
    "max_wall_clock_seconds": 5_400,
  }
  assert payload["effect_classes"] == ["read", "research_write", "state_write"]
  assert payload["policy_bundle_hash"] == _POLICY_BUNDLE_HASH
  assert "research_file_id" not in payload


@pytest.mark.parametrize(
  "tool_name",
  [
    "list_investment_capabilities",
    "get_investment_capability",
    "get_investment_run",
    "cancel_investment_run",
    "start_quant_research",
  ],
)
def test_quant_skill_claim_is_issued_for_quant_read_and_cancel_routes(
  monkeypatch: pytest.MonkeyPatch,
  tool_name: str,
) -> None:
  _set_signing_key(monkeypatch)
  mcp = _FakeMcpClient(tool_name=tool_name)

  result, error = _dispatch(
    _dispatcher(mcp),
    tool_call_id=f"call-{tool_name}",
    tool_name=tool_name,
  )

  assert error is None
  assert result == {"ok": True}
  _header, payload = _decode_claim(
    mcp.calls[0]["meta"]["investment_capability_claim"]
  )
  assert payload["tool_name"] == tool_name
  assert payload["allowed_capability_ids"] == ["quant_research"]
  assert payload["effect_classes"] == ["read", "research_write", "state_write"]
  if tool_name == "start_quant_research":
    assert payload["research_file_id"] == 2
  else:
    assert "research_file_id" not in payload


def test_quant_invalid_request_is_returned_after_one_signed_mcp_call(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  class _RejectingMcpClient(_FakeMcpClient):
    async def call_tool(
      self,
      name: str,
      tool_input: object,
      meta: dict[str, Any] | None = None,
      abort_event: asyncio.Event | None = None,
      gateway_session: object | None = None,
      allow_uncertain_replay: bool = True,
      trusted_dispatch_scope: Mapping[str, object] | None = None,
    ) -> tuple[object | None, dict[str, object] | None]:
      _ = abort_event, gateway_session, allow_uncertain_replay, trusted_dispatch_scope
      self.calls.append({
        "name": name,
        "tool_input": tool_input,
        "meta": meta,
      })
      return None, {
        "code": "invalid_request",
        "message": "source_refs[0] uses an unsupported reference scheme",
      }

  _set_signing_key(monkeypatch)
  mcp = _RejectingMcpClient(tool_name="start_quant_research")

  result, error = _dispatch(
    _dispatcher(mcp),
    tool_call_id="call-invalid-quant-request",
    tool_name="start_quant_research",
    tool_input={
      "request": {
        "research_file_id": 2,
        "source_refs": ["fmp:income-statement"],
      },
    },
  )

  assert result is None
  assert error == {
    "code": "invalid_request",
    "message": "source_refs[0] uses an unsupported reference scheme",
  }
  assert len(mcp.calls) == 1
  assert mcp.calls[0]["name"] == "start_quant_research"
  _header, payload = _decode_claim(
    mcp.calls[0]["meta"]["investment_capability_claim"]
  )
  assert payload["tool_name"] == "start_quant_research"
  assert payload["research_file_id"] == 2


@pytest.mark.parametrize(
  "tool_name",
  [
    "cancel_investment_run",
    "get_investment_artifact",
    "get_investment_capability",
    "get_investment_run",
    "list_investment_artifacts",
    "list_investment_capabilities",
    "start_investment_run",
  ],
)
def test_market_scan_claim_is_issued_for_only_its_exact_facade_routes(
  monkeypatch: pytest.MonkeyPatch,
  tool_name: str,
) -> None:
  _set_signing_key(monkeypatch)
  mcp = _FakeMcpClient(tool_name=tool_name)

  result, error = _dispatch(
    _dispatcher(mcp, skill="market-scan", research_file_id=None),
    tool_call_id=f"call-market-{tool_name}",
    tool_name=tool_name,
  )

  assert error is None
  assert result == {"ok": True}
  _header, payload = _decode_claim(
    mcp.calls[0]["meta"]["investment_capability_claim"]
  )
  assert payload["tool_name"] == tool_name
  assert payload["skill"] == "market-scan"
  assert payload["allowed_capability_ids"] == [
    "fingerprint_screen",
    "insider_buying",
    "quality_screen",
  ]
  assert payload["effect_classes"] == [
    "artifact_write",
    "read",
    "state_write",
  ]
  assert "approved_budget" not in payload
  assert "research_file_id" not in payload


def test_model_cannot_supply_investment_run_provenance_refs(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _set_signing_key(monkeypatch)
  mcp = _FakeMcpClient(tool_name="start_investment_run")

  result, error = _dispatch(
    _dispatcher(mcp, skill="market-scan", research_file_id=None),
    tool_call_id="call-market-forged-refs",
    tool_name="start_investment_run",
    tool_input={
      "capability_id": "quality_screen",
      "external_refs": {"source_schedule_id": "forged"},
    },
  )

  assert result is None
  assert error == {
    "code": "investment_external_refs_not_allowed",
    "message": (
      "Investment run provenance references cannot be supplied by the model."
    ),
  }
  assert mcp.calls == []


def test_market_scan_quant_submission_fails_closed_before_mcp(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _set_signing_key(monkeypatch)
  mcp = _FakeMcpClient(tool_name="start_quant_research")

  result, error = _dispatch(
    _dispatcher(mcp, skill="market-scan", research_file_id=2),
    tool_call_id="call-market-quant",
    tool_name="start_quant_research",
  )

  assert result is None
  assert error is not None
  assert error["code"] == "investment_capability_claim_unavailable"
  assert mcp.calls == []


@pytest.mark.parametrize(
  "research_file_id",
  [None, True, "2", 0, -1, 1 << 63],
)
def test_quant_claim_rejects_missing_or_malformed_trusted_research_origin(
  monkeypatch: pytest.MonkeyPatch,
  research_file_id: object,
) -> None:
  _set_signing_key(monkeypatch)

  with pytest.raises(
    InvestmentCapabilityClaimError,
    match="requires research_file_id",
  ):
    issue_investment_capability_claim(
      user_id="42",
      tenant=_TENANT,
      session_id="session-123",
      skill_run_id="skill-run-123",
      channel="excel",
      tool_name="start_quant_research",
      request_id="request-123",
      jti="tool-call-123",
      skill="quant-research",
      admitted_skill_execution_limits=_limits_for_skill("quant-research"),
      policy_bundle_hash=_POLICY_BUNDLE_HASH,
      research_file_id=research_file_id,  # pyright: ignore[reportArgumentType]  # negative: malformed research origin rejection
    )


@pytest.mark.parametrize("model_research_file_id", [3, True, "2", None])
def test_quant_dispatch_rejects_model_origin_mismatch_before_mcp(
  monkeypatch: pytest.MonkeyPatch,
  model_research_file_id: object,
) -> None:
  _set_signing_key(monkeypatch)
  mcp = _FakeMcpClient(tool_name="start_quant_research")
  request: dict[str, Any] = {}
  if model_research_file_id is not None:
    request["research_file_id"] = model_research_file_id

  result, error = _dispatch(
    _dispatcher(mcp, research_file_id=2),
    tool_call_id="call-origin-mismatch",
    tool_name="start_quant_research",
    tool_input={"request": request},
  )

  assert result is None
  assert error is not None
  assert error["code"] == "investment_capability_claim_unavailable"
  assert "research_file_id" not in json.dumps(error)
  assert mcp.calls == []


@pytest.mark.parametrize(
  "trusted_research_file_id",
  [None, True, "2", 0, 1 << 63],
)
def test_quant_dispatch_rejects_malformed_trusted_origin_before_mcp(
  monkeypatch: pytest.MonkeyPatch,
  trusted_research_file_id: object,
) -> None:
  _set_signing_key(monkeypatch)
  mcp = _FakeMcpClient(tool_name="start_quant_research")

  result, error = _dispatch(
    _dispatcher(mcp, research_file_id=trusted_research_file_id),  # pyright: ignore[reportArgumentType]  # negative: malformed trusted origin rejection
    tool_call_id="call-invalid-trusted-origin",
    tool_name="start_quant_research",
    tool_input={"request": {"research_file_id": 2}},
  )

  assert result is None
  assert error is not None
  assert error["code"] == "investment_capability_claim_unavailable"
  assert mcp.calls == []


@pytest.mark.parametrize(
  ("limits", "approved_budget"),
  [
    (
      SkillExecutionLimits(100_000_001),
      {
        "max_tokens": 100_000_001,
        "max_wall_clock_seconds": 5_400,
      },
    ),
    (
      SkillExecutionLimits(None),
      {"max_wall_clock_seconds": 5_400},
    ),
    (
      None,
      {"max_wall_clock_seconds": 5_400},
    ),
  ],
)
def test_quant_claim_signs_the_admitted_budget_numbers_present(
  monkeypatch: pytest.MonkeyPatch,
  limits: SkillExecutionLimits | None,
  approved_budget: dict[str, int | float],
) -> None:
  _set_signing_key(monkeypatch)

  token = issue_investment_capability_claim(
    user_id="42",
    tenant=_TENANT,
    session_id="session-123",
    skill_run_id="skill-run-123",
    channel="excel",
    tool_name="start_quant_research",
    request_id="request-123",
    jti="tool-call-123",
    skill="quant-research",
    admitted_skill_execution_limits=limits,
    policy_bundle_hash=_POLICY_BUNDLE_HASH,
    research_file_id=2,
    now=1_800_000_000,
  )

  _header, payload = _decode_claim(token)
  assert payload["approved_budget"] == approved_budget
  assert payload["exp"] == 1_800_000_060


@pytest.mark.parametrize("tool_name", sorted(INVESTMENT_CAPABILITY_FACADE_TOOLS))
@pytest.mark.parametrize("skill", [None, "unknown-skill"])
def test_facade_dispatch_fails_closed_without_a_known_trusted_skill(
  monkeypatch: pytest.MonkeyPatch,
  skill: str | None,
  tool_name: str,
) -> None:
  _set_signing_key(monkeypatch)
  token = set_current_skill(None)
  try:
    mcp = _FakeMcpClient(tool_name=tool_name)
    result, error = _dispatch(
      _dispatcher(mcp, skill=skill),
      tool_call_id="call-unknown-skill",
      tool_name=tool_name,
    )
  finally:
    reset_current_skill(token)

  assert result is None
  assert error is not None
  _assert_names_grant_and_issuance_route(
    error,
    subject=f"tool '{tool_name}'",
  )
  assert mcp.calls == []


def test_active_gateway_skill_context_can_supply_missing_run_context_skill(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _set_signing_key(monkeypatch)
  token = set_current_skill(ActiveSkillAdmission(
    "quant-research",
    _limits_for_skill("quant-research"),  # type: ignore[arg-type]
  ))
  try:
    mcp = _FakeMcpClient(tool_name="get_investment_run")
    result, error = _dispatch(
      _dispatcher(mcp, skill=None),
      tool_call_id="call-active-skill",
      tool_name="get_investment_run",
    )
  finally:
    reset_current_skill(token)

  assert error is None
  assert result == {"ok": True}
  _header, payload = _decode_claim(
    mcp.calls[0]["meta"]["investment_capability_claim"]
  )
  assert payload["skill"] == "quant-research"


def test_conflicting_trusted_skill_contexts_fail_closed(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _set_signing_key(monkeypatch)
  token = set_current_skill(ActiveSkillAdmission(
    "different-skill",
    SkillExecutionLimits(32_000),
  ))
  try:
    mcp = _FakeMcpClient(tool_name="get_investment_run")
    result, error = _dispatch(
      _dispatcher(mcp),
      tool_call_id="call-conflicting-skill",
      tool_name="get_investment_run",
    )
  finally:
    reset_current_skill(token)

  assert result is None
  assert error is not None
  assert error["code"] == "investment_capability_claim_unavailable"
  assert mcp.calls == []


@pytest.mark.parametrize(
  "policy_bundle_hash",
  ["unknown", _POLICY_BUNDLE_HASH.upper(), "0" * 63],
)
def test_invalid_policy_bundle_identity_fails_closed(
  monkeypatch: pytest.MonkeyPatch,
  policy_bundle_hash: str,
) -> None:
  _set_signing_key(monkeypatch)
  mcp = _FakeMcpClient(tool_name="get_investment_run")

  result, error = _dispatch(
    _dispatcher(mcp, policy_bundle_hash=policy_bundle_hash),
    tool_call_id="call-unknown-policy",
    tool_name="get_investment_run",
  )

  assert result is None
  assert error is not None
  assert error["code"] == "investment_capability_claim_unavailable"
  assert mcp.calls == []


def test_signed_grant_tampering_invalidates_signature(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _set_signing_key(monkeypatch)
  token = issue_investment_capability_claim(
    user_id="42",
    tenant=_TENANT,
    session_id="session-123",
    skill_run_id="skill-run-123",
    channel="excel",
    tool_name="start_investment_run",
    request_id="request-123",
    jti="tool-call-123",
    skill="market-scan",
    admitted_skill_execution_limits=_limits_for_skill("market-scan"),
    policy_bundle_hash=_POLICY_BUNDLE_HASH,
    now=1_800_000_000,
  )
  header_segment, payload_segment, supplied_signature = token.split(".")
  payload = _decode_segment(payload_segment)
  payload["allowed_capability_ids"] = ["fingerprint_screen"]
  tampered_payload = base64.urlsafe_b64encode(
    json.dumps(
      payload,
      ensure_ascii=True,
      sort_keys=True,
      separators=(",", ":"),
    ).encode("ascii")
  ).rstrip(b"=").decode("ascii")
  signature_bytes = base64.urlsafe_b64decode(
    supplied_signature + ("=" * (-len(supplied_signature) % 4))
  )
  with pytest.raises(InvalidSignature):
    _PUBLIC_KEY.verify(
      signature_bytes,
      f"{header_segment}.{tampered_payload}".encode("ascii"),
    )


def test_legacy_workbench_tools_do_not_require_or_receive_claim(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv(
    INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV,
    raising=False,
  )
  mcp = _FakeMcpClient(tool_name="search_findings")

  result, error = _dispatch(
    _dispatcher(mcp),
    tool_call_id="call-legacy-read",
    tool_name="search_findings",
  )

  assert error is None
  assert result == {"ok": True}
  assert "investment_capability_claim" not in mcp.calls[0]["meta"]


def test_private_key_validation_error_never_echoes_key_material(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  malformed_key = "recognizable-malformed-private-key"
  monkeypatch.setenv(
    INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV,
    malformed_key,
  )

  with pytest.raises(InvestmentCapabilityClaimError) as exc_info:
    issue_investment_capability_claim(
      user_id="42",
      tenant=_TENANT,
      session_id="session-123",
      skill_run_id="skill-run-123",
      channel="excel",
      tool_name="get_investment_run",
      request_id="request-123",
      jti="tool-call-123",
      skill="quant-research",
      admitted_skill_execution_limits=_limits_for_skill("quant-research"),
      policy_bundle_hash=_POLICY_BUNDLE_HASH,
    )

  assert malformed_key not in str(exc_info.value)


def test_legacy_shared_secret_and_public_key_cannot_enable_signing(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv(
    INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV,
    raising=False,
  )
  monkeypatch.setenv(
    "INVESTMENT_CAPABILITY_CLAIM_SECRET",
    "legacy-shared-secret-that-is-long-enough-to-have-worked",
  )
  monkeypatch.setenv(
    INVESTMENT_CAPABILITY_CLAIM_PUBLIC_KEY_ENV,
    _PUBLIC_KEY_MATERIAL,
  )

  with pytest.raises(InvestmentCapabilityClaimError):
    issue_investment_capability_claim(
      user_id="42",
      tenant=_TENANT,
      session_id="session-123",
      skill_run_id="skill-run-123",
      channel="excel",
      tool_name="get_investment_run",
      request_id="request-123",
      jti="tool-call-123",
      skill="quant-research",
      admitted_skill_execution_limits=_limits_for_skill("quant-research"),
      policy_bundle_hash=_POLICY_BUNDLE_HASH,
    )


@pytest.mark.parametrize(
  ("field", "value"),
  [
    ("user_id", "user\nspoof"),
    ("session_id", "s" * 257),
    ("channel", "c" * 65),
  ],
)
def test_claim_rejects_fields_the_server_cannot_accept(
  monkeypatch: pytest.MonkeyPatch,
  field: str,
  value: str,
) -> None:
  _set_signing_key(monkeypatch)
  inputs = {
    "user_id": "42",
    "tenant": _TENANT,
    "session_id": "session-123",
    "skill_run_id": "skill-run-123",
    "channel": "excel",
    "tool_name": "get_investment_run",
    "request_id": "request-123",
    "jti": "tool-call-123",
    "skill": "quant-research",
    "admitted_skill_execution_limits": _limits_for_skill("quant-research"),
    "policy_bundle_hash": _POLICY_BUNDLE_HASH,
  }
  inputs[field] = value

  with pytest.raises(InvestmentCapabilityClaimError):
    issue_investment_capability_claim(**inputs)




@pytest.mark.parametrize("tool_name", sorted(INVESTMENT_JOB_PRINCIPAL_TOOLS))
def test_job_principal_claim_is_exact_and_requires_no_business_skill_grant(
  monkeypatch: pytest.MonkeyPatch,
  tool_name: str,
) -> None:
  _set_signing_key(monkeypatch)
  configure_investment_capability_grants({})
  token = issue_investment_job_principal_claim(
    user_id="42",
    tenant=_TENANT,
    session_id="session-123",
    channel="excel",
    tool_name=tool_name,
    request_id="request-123",
    jti="tool-call-123",
    now=1_800_000_000,
  )
  header, payload = _decode_claim(token)
  assert INVESTMENT_JOB_PRINCIPAL_TOOLS == frozenset({
    "submit_research_task", "send_research_task_message",
    "start_research_pipeline", "start_research_benchmark",
  })
  assert INVESTMENT_JOB_PRINCIPAL_TOOLS.isdisjoint(INVESTMENT_CAPABILITY_FACADE_TOOLS)
  assert header == {"alg": "EdDSA", "kid": _KEY_ID, "typ": "JWT"}
  assert payload == {
    "aud": "investment-tools",
    "channel": "excel",
    "exp": 1_800_000_060,
    "iat": 1_800_000_000,
    "iss": "ai-excel-addin",
    "jti": "tool-call-123",
    "kind": "job_principal",
    "request_id": "request-123",
    "session_id": "session-123",
    "sub": "42",
    "tenant": _TENANT.model_dump(mode="json"),
    "tool_name": tool_name,
    "user_id": "42",
    "v": 3,
  }
  signing_input, supplied_signature = token.rsplit(".", 1)
  signature_bytes = base64.urlsafe_b64decode(
    supplied_signature + "=" * (-len(supplied_signature) % 4)
  )
  _PUBLIC_KEY.verify(signature_bytes, signing_input.encode("ascii"))


@pytest.mark.parametrize(
  ("tenant", "tool_name"),
  [(None, "submit_research_task"), (_TENANT, "start_investment_run")],
)
def test_job_principal_claim_requires_authenticated_tenant_and_exact_control(
  monkeypatch: pytest.MonkeyPatch,
  tenant: AdmittedTenant | None,
  tool_name: str,
) -> None:
  _set_signing_key(monkeypatch)
  with pytest.raises(InvestmentCapabilityClaimError):
    issue_investment_job_principal_claim(
      user_id="42",
      tenant=tenant,
      session_id="session-123",
      channel="excel",
      tool_name=tool_name,
      request_id="request-123",
      jti="tool-call-123",
    )


@pytest.mark.parametrize("tool_name", sorted(INVESTMENT_JOB_PRINCIPAL_TOOLS))
@pytest.mark.parametrize("active_skill", [False, True])
def test_owner_dispatch_uses_job_principal_with_or_without_active_skill(
  monkeypatch: pytest.MonkeyPatch,
  tool_name: str,
  active_skill: bool,
) -> None:
  _set_signing_key(monkeypatch)
  configure_investment_capability_grants({})
  skill = (
    ActiveSkillAdmission("quant-research", SkillExecutionLimits(32_000))
    if active_skill else None
  )
  token = set_current_skill(skill)
  try:
    mcp = _FakeMcpClient(tool_name=tool_name)
    result, error = _dispatch(
      _dispatcher(mcp, skill=None),
      tool_call_id="call-job-principal",
      tool_name=tool_name,
    )
  finally:
    reset_current_skill(token)
  assert error is None
  assert result == {"ok": True}
  assert len(mcp.calls) == 1
  signed_claim = mcp.calls[0]["meta"]["investment_capability_claim"]
  _header, payload = _decode_claim(signed_claim)
  assert payload["v"] == 3
  assert payload["kind"] == "job_principal"
  assert payload["tool_name"] == tool_name
  assert payload["sub"] == payload["user_id"] == "42"
  assert payload["tenant"] == _TENANT.model_dump(mode="json")
  assert payload["tenant"]["storage_id"] != payload["sub"]
  assert payload["jti"] == "call-job-principal"
  assert not {"skill", "allowed_capability_ids", "effect_classes", "approved_budget"} & payload.keys()
  signing_input, supplied_signature = signed_claim.rsplit(".", 1)
  _PUBLIC_KEY.verify(
    base64.urlsafe_b64decode(supplied_signature + "=" * (-len(supplied_signature) % 4)),
    signing_input.encode("ascii"),
  )


@pytest.mark.parametrize("tool_name", sorted(INVESTMENT_JOB_PRINCIPAL_TOOLS))
def test_sessionless_job_dispatch_returns_typed_issuer_unavailable(
  monkeypatch: pytest.MonkeyPatch,
  tool_name: str,
) -> None:
  _set_signing_key(monkeypatch)
  configure_investment_capability_grants({})
  mcp = _FakeMcpClient(tool_name=tool_name)
  dispatcher = ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=mcp,
    session=None,
    session_id="sessionless-request",
    user_id="sia-alice",
    risk_user_id=42,
    channel="excel",
    role="owner",
    mcp_meta_inject_servers=frozenset({"idea-workbench-mcp"}),
    run_context=RunContext(
      user_id="sia-alice",
      request_id="request-sessionless",
      session_id="sessionless-request",
      run_id="sessionless-job",
      profile="analyst",
      channel="excel",
    ),
  )
  result, error = _dispatch(
    dispatcher, tool_call_id="call-sessionless-job", tool_name=tool_name,
  )
  assert result is None
  assert error is not None
  assert error["code"] == "investment_capability_claim_unavailable"
  assert mcp.calls == []


@pytest.mark.parametrize("tool_name", sorted(INVESTMENT_JOB_PRINCIPAL_TOOLS))
def test_uncaptured_session_returns_typed_job_principal_gap(
  monkeypatch: pytest.MonkeyPatch,
  tool_name: str,
) -> None:
  _set_signing_key(monkeypatch)
  mcp = _FakeMcpClient(tool_name=tool_name)
  session = GatewaySession(
    session_id="uncaptured-session",
    api_key_hash="test-api-key-hash",
    created_at=1_800_000_000,
    expires_at=1_800_000_600,
    user_id=_TENANT.storage_id,
    risk_user_id=42,
    channel="excel",
    role="owner",
  )
  dispatcher = ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=mcp,
    session=session,
    session_id=session.session_id,
    user_id=session.user_id,
    risk_user_id=session.risk_user_id,
    channel="excel",
    role="owner",
    mcp_meta_inject_servers=frozenset({"idea-workbench-mcp"}),
  )
  result, error = _dispatch(
    dispatcher, tool_call_id="call-uncaptured-job", tool_name=tool_name,
  )
  assert result is None
  assert error is not None
  assert error["code"] == "investment_capability_claim_unavailable"
  assert mcp.calls == []


@pytest.mark.parametrize("tool_name", sorted(INVESTMENT_JOB_PRINCIPAL_TOOLS))
def test_session_without_admitted_tenant_cannot_mint_job_principal(
  monkeypatch: pytest.MonkeyPatch,
  tool_name: str,
) -> None:
  _set_signing_key(monkeypatch)
  mcp = _FakeMcpClient(tool_name=tool_name)
  dispatcher = _dispatcher(mcp, skill=None)
  capture_job_context(
    dispatcher._session,
    root_context_from_identity(
      tenant=None,
      job_id="agent-run-123",
      tool_ids=None,
      withheld_tools=frozenset(),
      research_binding=AdmittedResearchBinding(),
      approval_route=NO_APPROVAL_ROUTE,
    ),
  )
  result, error = _dispatch(
    dispatcher, tool_call_id="call-unowned-job", tool_name=tool_name,
  )
  assert result is None
  assert error is not None
  assert error["code"] == "investment_capability_claim_unavailable"
  assert mcp.calls == []


@pytest.mark.parametrize("tool_name", sorted(INVESTMENT_JOB_PRINCIPAL_TOOLS))
@pytest.mark.parametrize("tenant", [_TENANT, None])
def test_sessionless_dispatch_identity_retains_actual_admitted_tenant(
  monkeypatch: pytest.MonkeyPatch,
  tool_name: str,
  tenant: AdmittedTenant | None,
) -> None:
  _set_signing_key(monkeypatch)
  configure_investment_capability_grants({})
  admitted = root_context_from_identity(
    tenant=tenant,
    job_id="sessionless-owned-job",
    tool_ids=None,
    withheld_tools=frozenset(),
    research_binding=AdmittedResearchBinding(),
    approval_route=NO_APPROVAL_ROUTE,
  )
  identity = DispatchIdentity(
    context=admitted if tenant is not None else None,
    execution=None,
    session=None,
    user_id="sia-alice",
    risk_user_id=42,
    channel="excel",
  )
  mcp = _FakeMcpClient(tool_name=tool_name)
  dispatcher = ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=mcp,
    identity=identity,
    session_id="sessionless-request",
    role="owner",
    mcp_meta_inject_servers=frozenset({"idea-workbench-mcp"}),
    run_context=RunContext(
      user_id="sia-alice",
      request_id="request-sessionless-owned",
      session_id="sessionless-request",
      run_id=admitted.job_id,
      profile="analyst",
      channel="excel",
    ),
  )
  result, error = _dispatch(
    dispatcher, tool_call_id="call-sessionless-owned", tool_name=tool_name,
  )
  if tenant is None:
    assert result is None
    assert error is not None
    assert error["code"] == "investment_capability_claim_unavailable"
    assert mcp.calls == []
    return
  assert error is None
  assert result == {"ok": True}
  assert len(mcp.calls) == 1
  signed_claim = mcp.calls[0]["meta"]["investment_capability_claim"]
  _header, payload = _decode_claim(signed_claim)
  assert payload["v"] == 3
  assert payload["kind"] == "job_principal"
  assert payload["tool_name"] == tool_name
  assert payload["tenant"] == tenant.model_dump(mode="json")
  assert payload["sub"] == payload["user_id"] == "42"
  assert payload["tenant"]["storage_id"] != payload["sub"]
  assert not {"skill", "allowed_capability_ids", "effect_classes", "approved_budget"} & payload.keys()
  signing_input, supplied_signature = signed_claim.rsplit(".", 1)
  _PUBLIC_KEY.verify(
    base64.urlsafe_b64decode(supplied_signature + "=" * (-len(supplied_signature) % 4)),
    signing_input.encode("ascii"),
  )


@pytest.mark.parametrize("tool_name", sorted(INVESTMENT_JOB_PRINCIPAL_TOOLS))
def test_in_process_child_context_precedes_parent_credential_session(
  monkeypatch: pytest.MonkeyPatch,
  tool_name: str,
) -> None:
  from dataclasses import replace

  from agent_workflow_contracts import CatalogToolEntry, PlatformToolCatalog, SemanticCapabilityRequirement
  from agent_gateway.admission import (
    AdmittedContext, AdmissionOwners, OperationRegistration, SpawnRequest,
    admit, context_for_child, current_parent_context,
  )
  from agent_gateway.capability_resolution import OperationDeclaration, resolve_operation_authority
  from agent_gateway.semantic_capability_routing import capability_for_tool
  from gateway_test_support.capability_execution_test_support import stub_capability_execution_resolver

  _set_signing_key(monkeypatch)
  configure_investment_capability_grants({})
  mcp = _FakeMcpClient(tool_name=tool_name)
  parent_dispatcher = _dispatcher(mcp, skill=None)
  parent_session = parent_dispatcher._session
  assert isinstance(parent_session, GatewaySession)
  parent_context = current_parent_context(parent_session)
  control_capability = capability_for_tool(
    canonical_name=mcp.get_original_tool_name(tool_name),
    server_id=mcp.get_server_for_tool(tool_name),
    effect="write",
  )
  assert control_capability is not None
  catalog = PlatformToolCatalog(tools=(
    CatalogToolEntry(
      tool_id=tool_name,
      canonical_name=tool_name,
      origin="mcp",
      server_id="idea-workbench-mcp",
      effect="write",
      capability=control_capability,
    ),
  ))
  parent_context = replace(
    parent_context, tool_ids=frozenset(entry.tool_id for entry in catalog.tools),
  )
  capture_job_context(parent_session, parent_context)
  assert parent_context.tool_ids == frozenset({tool_name})
  declaration = OperationDeclaration(
    operation_name="explore",
    grant_id="grant:child-job-control",
    workspace_scope="workspace_write",
    tool_ceiling=frozenset({tool_name}),
    required_capabilities=(
      SemanticCapabilityRequirement(name=control_capability, binding_modes=("live_tool",)),
    ),
  )
  admitted = admit(
    SpawnRequest(operation="explore", task="Control the owned research job"),
    parent_context,
    owners=AdmissionOwners(
      operation=OperationRegistration(
        name="explore",
        execution_class="node.explore",
        route_ceiling=frozenset({tool_name}),
        operation_routes=frozenset(),
      ),
      capability_execution_resolver=stub_capability_execution_resolver(
        default_provider="openai", default_model="gpt-5.6-sol", default_effort="high",
      ),
      session=parent_session,
      route_registered=lambda route_id: route_id == tool_name,
      resolve_tool_authority=lambda tool_ids: resolve_operation_authority(
        replace(declaration, tool_ceiling=tool_ids), catalog=catalog,
      ),
      placement="in_process",
    ),
  )
  assert isinstance(admitted, AdmittedContext)
  child_context = context_for_child(admitted, job_id="actual-child-job")
  assert child_context.tenant is not None
  assert child_context.depth == parent_context.depth + 1
  assert child_context.tool_ids == frozenset({tool_name})
  assert child_context.job_id != parent_context.job_id
  identity = DispatchIdentity(
    context=child_context,
    execution=None,
    session=parent_session,
    user_id=_TENANT.storage_id,
    risk_user_id=42,
    channel="excel",
  )
  child_dispatcher = ToolDispatcher(
    capability_bind=admitted.wire.capability_bind,
    mcp_client=mcp,
    identity=identity,
    session_id=parent_session.session_id,
    role="owner",
    mcp_meta_inject_servers=frozenset({"idea-workbench-mcp"}),
    run_context=RunContext(
      user_id=_TENANT.storage_id,
      request_id="request-child-control",
      session_id=parent_session.session_id,
      run_id=child_context.job_id,
      profile="analyst",
      channel="excel",
    ),
  )
  executing_context = child_dispatcher._executing_context()
  assert executing_context is not None
  assert executing_context is child_context
  assert executing_context.job_id == "actual-child-job"
  assert current_parent_context(parent_session) is parent_context
  assert current_parent_context(parent_session).job_id == "agent-run-123"
  result, error = _dispatch(
    child_dispatcher, tool_call_id="call-child-control", tool_name=tool_name,
  )
  assert error is None
  assert result == {"ok": True}
  assert len(mcp.calls) == 1
  _header, payload = _decode_claim(mcp.calls[0]["meta"]["investment_capability_claim"])
  assert payload["v"] == 3
  assert payload["kind"] == "job_principal"
  assert payload["tenant"] == child_context.tenant.model_dump(mode="json")
  assert payload["sub"] == "42"
  assert payload["tool_name"] == tool_name
  assert current_parent_context(parent_session).job_id == "agent-run-123"



@pytest.mark.parametrize("role", ["owner", "invite"])
def test_claim_signs_canonical_tenant_without_conflating_risk_subject(
  monkeypatch: pytest.MonkeyPatch,
  role: str,
) -> None:
  _set_signing_key(monkeypatch)
  tenant = AdmittedTenant.model_validate({
    "tenant_id": None,
    "storage_id": "sia-alice",
    "role": role,
  })
  token = issue_investment_capability_claim(
    user_id="42",
    tenant=tenant,
    session_id="session-123",
    skill_run_id="skill-run-123",
    channel="excel",
    tool_name="get_investment_run",
    request_id="request-123",
    jti="tool-call-123",
    skill="market-scan",
    admitted_skill_execution_limits=_limits_for_skill("market-scan"),
    policy_bundle_hash=_POLICY_BUNDLE_HASH,
    now=1_800_000_000,
  )
  _header, payload = _decode_claim(token)
  assert payload["tenant"] == {
    "tenant_id": None,
    "storage_id": "sia-alice",
    "role": role,
  }
  assert payload["sub"] == payload["user_id"] == "42"
  assert payload["allowed_capability_ids"] == sorted(_DECLARED_GRANTS["market-scan"].capability_ids)
  assert payload["effect_classes"] == sorted(_DECLARED_GRANTS["market-scan"].effect_classes)


def test_claim_requires_explicit_authenticated_tenant(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _set_signing_key(monkeypatch)
  with pytest.raises(InvestmentCapabilityClaimError, match="authenticated tenant"):
    issue_investment_capability_claim(
      user_id="42",
      tenant=None,
      session_id="session-123",
      skill_run_id="skill-run-123",
      channel="excel",
      tool_name="get_investment_run",
      request_id="request-123",
      jti="tool-call-123",
      skill="market-scan",
      admitted_skill_execution_limits=_limits_for_skill("market-scan"),
      policy_bundle_hash=_POLICY_BUNDLE_HASH,
    )
