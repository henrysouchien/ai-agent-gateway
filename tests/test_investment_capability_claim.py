# ruff: noqa: E402

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.investment_capability_claim as claim_module
from agent_gateway.approval_policy import RunContext
from agent_gateway.investment_capability_claim import (
  INVESTMENT_CAPABILITY_FACADE_TOOLS,
  INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV,
  INVESTMENT_CAPABILITY_CLAIM_PUBLIC_KEY_ENV,
  INVESTMENT_CAPABILITY_SKILL_GRANTS,
  InvestmentCapabilitySkillGrant,
  InvestmentCapabilityClaimError,
  issue_investment_capability_claim,
  issue_investment_selected_content_claim,
  investment_capability_signing_available,
)
from agent_gateway.skill_context import reset_current_skill, set_current_skill
from agent_gateway.tool_dispatcher import ToolDispatcher


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


class _FakeMcpClient:
  def __init__(
    self,
    *,
    tool_name: str,
    original_tool_name: str | None = None,
  ) -> None:
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
    tool_input: dict[str, Any],
    meta: dict[str, Any] | None = None,
  ):
    self.calls.append({"name": name, "tool_input": tool_input, "meta": meta})
    return {"ok": True}, None

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


def _dispatcher(
  mcp: _FakeMcpClient,
  *,
  skill: str | None = "quant-research",
  policy_bundle_hash: str = _POLICY_BUNDLE_HASH,
  research_file_id: object = 2,
) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=mcp,
    session_id="session-123",
    user_id="sia-alice",
    risk_user_id=42,
    channel="excel",
    role="owner",
    mcp_meta_inject_servers=frozenset({"idea-workbench-mcp"}),
    run_context=RunContext(
      user_id="sia-alice",
      request_id="request-123",
      session_id="session-123",
      run_id="agent-run-123",
      profile=skill or "quant-research",
      channel="excel",
      skill=skill,
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
    ("artifact-1", "schema", 1_800_000_000),
    ("artifact-1", "SUMMARY", 1_800_000_000),
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


def test_investment_capability_skill_grants_are_exact_and_gateway_owned() -> None:
  assert set(INVESTMENT_CAPABILITY_SKILL_GRANTS) == {
    "market-scan",
    "quant-research",
  }

  market_grant = INVESTMENT_CAPABILITY_SKILL_GRANTS["market-scan"]
  assert market_grant.allowed_capability_ids == (
    "fingerprint_screen",
    "insider_buying",
    "quality_screen",
  )
  assert market_grant.allowed_tool_names == (
    "cancel_investment_run",
    "get_investment_artifact",
    "get_investment_capability",
    "get_investment_run",
    "list_investment_artifacts",
    "list_investment_capabilities",
    "start_investment_run",
  )
  assert market_grant.effect_classes == (
    "artifact_write",
    "read",
    "state_write",
  )
  assert market_grant.max_wall_clock_seconds is None
  assert "quant_research" not in market_grant.allowed_capability_ids
  assert "start_quant_research" not in market_grant.allowed_tool_names
  assert "research_write" not in market_grant.effect_classes

  grant = INVESTMENT_CAPABILITY_SKILL_GRANTS["quant-research"]
  assert grant.allowed_capability_ids == ("quant_research",)
  assert grant.allowed_tool_names == (
    "cancel_investment_run",
    "get_investment_artifact",
    "get_investment_capability",
    "get_investment_run",
    "list_investment_artifacts",
    "list_investment_capabilities",
    "start_quant_research",
  )
  assert grant.effect_classes == ("read", "research_write", "state_write")
  assert grant.max_wall_clock_seconds == 5_400
  assert "fingerprint_screen" not in grant.allowed_capability_ids
  assert "quality_screen" not in grant.allowed_capability_ids
  assert "artifact_write" not in grant.effect_classes
  assert "start_investment_run" not in grant.allowed_tool_names
  assert "approved_budget" not in inspect.signature(
    issue_investment_capability_claim
  ).parameters


def test_market_scan_claim_is_exact_and_has_no_quant_budget_or_origin(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _set_signing_key(monkeypatch)

  token = issue_investment_capability_claim(
    user_id="42",
    session_id="session-123",
    skill_run_id="skill-run-123",
    channel="excel",
    tool_name="start_investment_run",
    request_id="request-123",
    jti="tool-call-market-scan",
    skill="market-scan",
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
    "policy_bundle_hash": _POLICY_BUNDLE_HASH,
    "request_id": "request-123",
    "session_id": "session-123",
    "skill": "market-scan",
    "skill_run_id": "skill-run-123",
    "sub": "42",
    "tool_name": "start_investment_run",
    "user_id": "42",
    "v": 1,
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
      session_id="session-123",
      skill_run_id="skill-run-123",
      channel="excel",
      tool_name=tool_name,
      request_id="request-123",
      jti="tool-call-cross-skill",
      skill=skill,
      policy_bundle_hash=_POLICY_BUNDLE_HASH,
      research_file_id=research_file_id,
    )


def test_claim_is_canonical_signed_short_lived_and_exactly_bound(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _set_signing_key(monkeypatch)

  token = issue_investment_capability_claim(
    user_id="42",
    session_id="session-123",
    skill_run_id="skill-run-123",
    channel="excel",
    tool_name="start_quant_research",
    request_id="request-123",
    jti="tool-call-123",
    skill="quant-research",
    policy_bundle_hash=_POLICY_BUNDLE_HASH,
    research_file_id=2,
    now=1_800_000_000,
  )

  header, payload = _decode_claim(token)
  assert header == {"alg": "EdDSA", "kid": _KEY_ID, "typ": "JWT"}
  assert payload == {
    "allowed_capability_ids": ["quant_research"],
    "approved_budget": {
      "max_cost_usd": 20.0,
      "max_tokens": 32_000,
      "max_turns": 20,
      "max_wall_clock_seconds": 5_400,
    },
    "aud": "investment-tools",
    "channel": "excel",
    "effect_classes": ["read", "research_write", "state_write"],
    "exp": 1_800_000_060,
    "iat": 1_800_000_000,
    "iss": "ai-excel-addin",
    "jti": "tool-call-123",
    "policy_bundle_hash": _POLICY_BUNDLE_HASH,
    "request_id": "request-123",
    "research_file_id": 2,
    "session_id": "session-123",
    "skill": "quant-research",
    "skill_run_id": "skill-run-123",
    "sub": "42",
    "tool_name": "start_quant_research",
    "user_id": "42",
    "v": 1,
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
  assert error == {
    "code": "investment_capability_claim_unavailable",
    "message": f"Trusted investment capability identity is unavailable for tool '{tool_name}'.",
  }
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
    "max_cost_usd": 20.0,
    "max_tokens": 32_000,
    "max_turns": 20,
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
      tool_input: dict[str, Any],
      meta: dict[str, Any] | None = None,
    ):
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
      session_id="session-123",
      skill_run_id="skill-run-123",
      channel="excel",
      tool_name="start_quant_research",
      request_id="request-123",
      jti="tool-call-123",
      skill="quant-research",
      policy_bundle_hash=_POLICY_BUNDLE_HASH,
      research_file_id=research_file_id,
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
    _dispatcher(mcp, research_file_id=trusted_research_file_id),
    tool_call_id="call-invalid-trusted-origin",
    tool_name="start_quant_research",
    tool_input={"request": {"research_file_id": 2}},
  )

  assert result is None
  assert error is not None
  assert error["code"] == "investment_capability_claim_unavailable"
  assert mcp.calls == []


def test_issuer_rejects_malformed_gateway_budget_policy(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _set_signing_key(monkeypatch)
  monkeypatch.setattr(
    claim_module,
    "INVESTMENT_CAPABILITY_SKILL_GRANTS",
    {
      "quant-research": InvestmentCapabilitySkillGrant(
        allowed_capability_ids=("quant_research",),
        allowed_tool_names=("start_quant_research",),
        effect_classes=("read", "research_write", "state_write"),
        max_wall_clock_seconds=0,
      ),
    },
  )

  with pytest.raises(
    InvestmentCapabilityClaimError,
    match="invalid approved budget",
  ):
    issue_investment_capability_claim(
      user_id="42",
      session_id="session-123",
      skill_run_id="skill-run-123",
      channel="excel",
      tool_name="start_quant_research",
      request_id="request-123",
      jti="tool-call-123",
      skill="quant-research",
      policy_bundle_hash=_POLICY_BUNDLE_HASH,
      research_file_id=2,
    )


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
  assert error["code"] == "investment_capability_claim_unavailable"
  assert mcp.calls == []


def test_active_gateway_skill_context_can_supply_missing_run_context_skill(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _set_signing_key(monkeypatch)
  token = set_current_skill("quant-research")
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
  token = set_current_skill("different-skill")
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
    session_id="session-123",
    skill_run_id="skill-run-123",
    channel="excel",
    tool_name="start_investment_run",
    request_id="request-123",
    jti="tool-call-123",
    skill="market-scan",
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
      session_id="session-123",
      skill_run_id="skill-run-123",
      channel="excel",
      tool_name="get_investment_run",
      request_id="request-123",
      jti="tool-call-123",
      skill="quant-research",
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
      session_id="session-123",
      skill_run_id="skill-run-123",
      channel="excel",
      tool_name="get_investment_run",
      request_id="request-123",
      jti="tool-call-123",
      skill="quant-research",
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
    "session_id": "session-123",
    "skill_run_id": "skill-run-123",
    "channel": "excel",
    "tool_name": "get_investment_run",
    "request_id": "request-123",
    "jti": "tool-call-123",
    "skill": "quant-research",
    "policy_bundle_hash": _POLICY_BUNDLE_HASH,
  }
  inputs[field] = value

  with pytest.raises(InvestmentCapabilityClaimError):
    issue_investment_capability_claim(**inputs)
