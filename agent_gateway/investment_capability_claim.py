from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import time
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from agent_workflow_contracts.models import AdmittedTenant

from .skill_limits import SkillExecutionLimits
from hank_contracts.investment_claims import (
  INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV as INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV,
  INVESTMENT_CAPABILITY_CLAIM_PUBLIC_KEY_ENV as INVESTMENT_CAPABILITY_CLAIM_PUBLIC_KEY_ENV,
  INVESTMENT_CAPABILITY_CLAIM_ISSUER as INVESTMENT_CAPABILITY_CLAIM_ISSUER,
  INVESTMENT_CAPABILITY_CLAIM_AUDIENCE as INVESTMENT_CAPABILITY_CLAIM_AUDIENCE,
  INVESTMENT_CAPABILITY_CLAIM_TTL_SECONDS as INVESTMENT_CAPABILITY_CLAIM_TTL_SECONDS,
  INVESTMENT_CAPABILITY_CLAIM_KEY_BYTES as INVESTMENT_CAPABILITY_CLAIM_KEY_BYTES,
  INVESTMENT_SELECTED_CONTENT_CLAIM_PURPOSE as INVESTMENT_SELECTED_CONTENT_CLAIM_PURPOSE,
  INVESTMENT_CAPABILITY_CLAIM_VERSION as INVESTMENT_CAPABILITY_CLAIM_VERSION,
  INVESTMENT_CAPABILITY_CLAIM_KIND as INVESTMENT_CAPABILITY_CLAIM_KIND,
  INVESTMENT_JOB_PRINCIPAL_CLAIM_VERSION as INVESTMENT_JOB_PRINCIPAL_CLAIM_VERSION,
  INVESTMENT_JOB_PRINCIPAL_CLAIM_KIND as INVESTMENT_JOB_PRINCIPAL_CLAIM_KIND,
  INVESTMENT_CAPABILITY_CLAIM_SERVER as INVESTMENT_CAPABILITY_CLAIM_SERVER,
  RISK_CAPABILITY_CLAIM_AUDIENCE as RISK_CAPABILITY_CLAIM_AUDIENCE,
  RISK_JOB_CAPABILITY_CLAIM_VERSION as RISK_JOB_CAPABILITY_CLAIM_VERSION,
  RISK_JOB_CAPABILITY_CLAIM_KIND as RISK_JOB_CAPABILITY_CLAIM_KIND,
  INVESTMENT_CAPABILITY_FACADE_TOOLS as INVESTMENT_CAPABILITY_FACADE_TOOLS,
  INVESTMENT_JOB_PRINCIPAL_TOOLS as INVESTMENT_JOB_PRINCIPAL_TOOLS,
  InvestmentCapabilityGrant as InvestmentCapabilityGrant,
)

logger = logging.getLogger(__name__)

INVESTMENT_READBACK_FACADE_TOOLS: Final[frozenset[str]] = frozenset({
  "get_investment_run",
  "list_investment_artifacts",
  "get_investment_artifact",
})

_MAX_ID_CHARS = 256
_MAX_CHANNEL_CHARS = 64
_MAX_RESEARCH_FILE_ID = (1 << 63) - 1

_POLICY_BUNDLE_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_skill_grants: Mapping[str, InvestmentCapabilityGrant] = MappingProxyType({})


def configure_investment_capability_grants(
  grants: Mapping[str, InvestmentCapabilityGrant],
) -> None:
  """Bind the application's skill-name -> declared-grant projection.

  Process-scoped like ``configure_server_policy``: the composition root that
  compiles the skill catalog binds the grants its notes declare. Unbound, no
  skill holds a grant.
  """

  global _skill_grants
  _skill_grants = grants


def investment_capability_grants() -> Mapping[str, InvestmentCapabilityGrant]:
  """The bound skill-name -> declared-grant projection."""

  return _skill_grants


class InvestmentCapabilityClaimError(RuntimeError):
  """A trusted investment-capability claim could not be issued."""


def investment_capability_claim_unavailable_error(
  *,
  subject: str,
) -> dict[str, Any]:
  """Named refusal for a missing investment-capability grant.

  The grant set is the bound projection of the skill notes that declare
  ``semantic_metadata.investment_grant``. Callers carry this payload; they
  do not restate the grants.
  """

  grants = tuple(sorted(_skill_grants))
  operations = tuple(f"agent-operation/{name}" for name in grants)
  grant_list = " or ".join(grants)
  operation_list = " or ".join(operations)
  return {
    "code": "investment_capability_claim_unavailable",
    "message": (
      f"Trusted investment capability identity is unavailable for {subject}. "
      f"Requires an admitted investment-capability grant ({grant_list}) "
      f"issued inside run_agent with {operation_list}."
    ),
    "required_grants": list(grants),
    "route": "run_agent",
    "operations": list(operations),
    "next_call": (
      {"tool_name": "run_agent", "arguments": {"operation": operations[0]}}
      if operations
      else None
    ),
  }



def _b64url(value: bytes) -> str:
  return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _canonical_json(value: dict[str, object]) -> bytes:
  return json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
  ).encode("ascii")


def _required_text(
  value: object,
  field: str,
  *,
  maximum: int = _MAX_ID_CHARS,
) -> str:
  if not isinstance(value, str):
    raise InvestmentCapabilityClaimError(
      f"investment capability claim requires {field}"
    )
  text = value.strip()
  if (
    not text
    or len(text) > maximum
    or any(ord(character) < 32 or ord(character) == 127 for character in text)
  ):
    raise InvestmentCapabilityClaimError(
      f"investment capability claim requires {field}"
    )
  return text


def _decode_key_material(value: object) -> bytes:
  if not isinstance(value, str) or not value or value != value.strip() or "=" in value:
    raise InvestmentCapabilityClaimError(
      "investment capability claim signing is unavailable"
    )
  try:
    raw = base64.b64decode(
      value + ("=" * (-len(value) % 4)),
      altchars=b"-_",
      validate=True,
    )
  except (TypeError, ValueError) as exc:
    raise InvestmentCapabilityClaimError(
      "investment capability claim signing is unavailable"
    ) from exc
  if (
    len(raw) != INVESTMENT_CAPABILITY_CLAIM_KEY_BYTES
    or _b64url(raw) != value
  ):
    raise InvestmentCapabilityClaimError(
      "investment capability claim signing is unavailable"
    )
  return raw


def _signing_key() -> tuple[Ed25519PrivateKey, str]:
  raw_private_key = _decode_key_material(
    os.environ.get(INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV)
  )
  try:
    private_key = Ed25519PrivateKey.from_private_bytes(raw_private_key)
  except ValueError as exc:
    raise InvestmentCapabilityClaimError(
      "investment capability claim signing is unavailable"
    ) from exc
  raw_public_key = private_key.public_key().public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw,
  )
  key_id = f"ed25519-sha256:{hashlib.sha256(raw_public_key).hexdigest()}"
  return private_key, key_id


def _sign(payload: dict[str, object]) -> str:
  private_key, key_id = _signing_key()
  header: dict[str, object] = {"alg": "EdDSA", "kid": key_id, "typ": "JWT"}
  signing_input = b".".join((
    _b64url(_canonical_json(header)).encode("ascii"),
    _b64url(_canonical_json(payload)).encode("ascii"),
  ))
  signature = private_key.sign(signing_input)
  return f"{signing_input.decode('ascii')}.{_b64url(signature)}"


def investment_capability_signing_available() -> bool:
  """Whether the hydrated private key can issue Investment claims."""

  try:
    _signing_key()
  except InvestmentCapabilityClaimError:
    return False
  return True


def _approved_budget(
  execution_limits: SkillExecutionLimits | None,
) -> dict[str, int | float]:
  """Sign the quant containment bound and admitted token limit, as present.

  Cost and turns are not minted. investment-tools records those as
  measurements; the issuer refuses none.
  """

  numbers: dict[str, int | float | None] = {
    "max_wall_clock_seconds": 5_400,
    "max_tokens": (
      None if execution_limits is None else execution_limits.max_tokens
    ),
  }
  return {
    field: value for field, value in numbers.items() if value is not None
  }


def issue_investment_capability_claim(
  *,
  user_id: str,
  tenant: AdmittedTenant | None,
  session_id: str,
  skill_run_id: str,
  channel: str,
  tool_name: str,
  request_id: str,
  jti: str,
  skill: str | None = None,
  admitted_skill_execution_limits: SkillExecutionLimits | None,
  policy_bundle_hash: str | None,
  research_file_id: int | None = None,
  now: int | None = None,
) -> str:
  """Issue the short-lived gateway assertion consumed by investment-tools.

  The model never supplies this payload. The gateway binds the executing
  job's admitted tenant and the exact routed MCP tool before dispatch.
  ``user_id`` is the resolved Risk subject, not the tenant's storage identity.
  """

  subject = _required_text(user_id, "user_id")
  if not isinstance(tenant, AdmittedTenant):
    raise InvestmentCapabilityClaimError(
      "investment capability claim requires an authenticated tenant"
    )
  session = _required_text(session_id, "session_id")
  skill_run = _required_text(skill_run_id, "skill_run_id")
  routed_channel = _required_text(
    channel,
    "channel",
    maximum=_MAX_CHANNEL_CHARS,
  )
  routed_tool = _required_text(tool_name, "tool_name")
  request = _required_text(request_id, "request_id")
  token_id = _required_text(jti, "jti")
  # The bundle hash is provenance, not authority (the grants below are). A run
  # with no approval route has no policy bundle, so the claim carries none
  # rather than refusing the gateway's own run context.
  trusted_policy_bundle: str | None = policy_bundle_hash
  if (
    not isinstance(policy_bundle_hash, str)
    or _POLICY_BUNDLE_HASH_PATTERN.fullmatch(policy_bundle_hash) is None
  ):
    logger.info(
      "investment capability claim for %s carries no policy bundle (run context: %r)",
      routed_tool,
      policy_bundle_hash,
    )
    trusted_policy_bundle = None
  if skill is None and routed_tool in INVESTMENT_READBACK_FACADE_TOOLS:
    matching_grants = [
      grant for grant in _skill_grants.values()
      if routed_tool in grant.tool_names and "read" in grant.effect_classes
    ]
    if not matching_grants:
      raise InvestmentCapabilityClaimError(
        "investment capability claim cannot bind an unsupported tool"
      )
    capability_ids = sorted({
      capability for grant in matching_grants for capability in grant.capability_ids
    })
    effect_classes = ["read"]
    trusted_skill = "session_readback"
  else:
    trusted_skill = _required_text(skill, "skill")
    grant = _skill_grants.get(trusted_skill)
    if grant is None:
      raise InvestmentCapabilityClaimError(
        "investment capability claim cannot bind an unsupported skill"
      )
    if routed_tool not in grant.tool_names:
      raise InvestmentCapabilityClaimError(
        "investment capability claim cannot bind an unsupported tool"
      )
    capability_ids = sorted(grant.capability_ids)
    effect_classes = sorted(grant.effect_classes)
  if "quant_research" in capability_ids:
    effective_limits = (
      admitted_skill_execution_limits
      if admitted_skill_execution_limits is not None or skill is not None
      else SkillExecutionLimits(32_000)
    )
    approved_budget = _approved_budget(effective_limits)
  else:
    approved_budget = None
  if routed_tool == "start_quant_research":
    if (
      isinstance(research_file_id, bool)
      or not isinstance(research_file_id, int)
      or not 1 <= research_file_id <= _MAX_RESEARCH_FILE_ID
    ):
      raise InvestmentCapabilityClaimError(
        "investment capability claim requires research_file_id"
      )
  elif research_file_id is not None:
    raise InvestmentCapabilityClaimError(
      "investment capability claim cannot bind research_file_id to this tool"
    )

  issued_at = int(time.time()) if now is None else now
  if isinstance(issued_at, bool) or not isinstance(issued_at, int):
    raise InvestmentCapabilityClaimError(
      "investment capability claim requires an integer issuance time"
    )

  payload: dict[str, object] = {
    "allowed_capability_ids": capability_ids,
    "aud": INVESTMENT_CAPABILITY_CLAIM_AUDIENCE,
    "channel": routed_channel,
    "effect_classes": effect_classes,
    "exp": issued_at + INVESTMENT_CAPABILITY_CLAIM_TTL_SECONDS,
    "iat": issued_at,
    "iss": INVESTMENT_CAPABILITY_CLAIM_ISSUER,
    "jti": token_id,
    "kind": INVESTMENT_CAPABILITY_CLAIM_KIND,
    "policy_bundle_hash": trusted_policy_bundle,
    "request_id": request,
    "session_id": session,
    "skill": trusted_skill,
    "skill_run_id": skill_run,
    "sub": subject,
    "tenant": tenant.model_dump(mode="json"),
    "tool_name": routed_tool,
    "user_id": subject,
    "v": INVESTMENT_CAPABILITY_CLAIM_VERSION,
  }
  if approved_budget is not None:
    payload["approved_budget"] = approved_budget
  if routed_tool == "start_quant_research":
    payload["research_file_id"] = research_file_id
  return _sign(payload)


def issue_investment_job_principal_claim(
  *,
  user_id: str,
  tenant: AdmittedTenant | None,
  session_id: str,
  channel: str,
  tool_name: str,
  request_id: str,
  jti: str,
  now: int | None = None,
) -> str:
  """Bind a role-authorized job control to the executing authenticated principal.

  The dispatcher owns role/route authorization. This form carries no business
  skill or capability grant; Risk subject and tenant storage identity are distinct.
  """
  subject = _required_text(user_id, "user_id")
  if not isinstance(tenant, AdmittedTenant):
    raise InvestmentCapabilityClaimError(
      "investment job principal claim requires an authenticated tenant"
    )
  session = _required_text(session_id, "session_id")
  routed_channel = _required_text(channel, "channel", maximum=_MAX_CHANNEL_CHARS)
  routed_tool = _required_text(tool_name, "tool_name")
  request = _required_text(request_id, "request_id")
  token_id = _required_text(jti, "jti")
  if routed_tool not in INVESTMENT_JOB_PRINCIPAL_TOOLS:
    raise InvestmentCapabilityClaimError(
      "investment job principal claim cannot bind an unsupported control"
    )
  issued_at = int(time.time()) if now is None else now
  if isinstance(issued_at, bool) or not isinstance(issued_at, int):
    raise InvestmentCapabilityClaimError(
      "investment job principal claim requires an integer issuance time"
    )
  payload: dict[str, object] = {
    "aud": INVESTMENT_CAPABILITY_CLAIM_AUDIENCE,
    "channel": routed_channel,
    "exp": issued_at + INVESTMENT_CAPABILITY_CLAIM_TTL_SECONDS,
    "iat": issued_at,
    "iss": INVESTMENT_CAPABILITY_CLAIM_ISSUER,
    "jti": token_id,
    "kind": INVESTMENT_JOB_PRINCIPAL_CLAIM_KIND,
    "request_id": request,
    "session_id": session,
    "sub": subject,
    "tenant": tenant.model_dump(mode="json"),
    "tool_name": routed_tool,
    "user_id": subject,
    "v": INVESTMENT_JOB_PRINCIPAL_CLAIM_VERSION,
  }
  return _sign(payload)

def issue_risk_job_capability_claim(
  *,
  user_id: str,
  tenant: AdmittedTenant,
  session_id: str,
  channel: str = "code_execute",
  granted_operations: Sequence[str],
  cell_token: str,
  now: int | None = None,
) -> str:
  """Issue the short-lived gateway job capability claim consumed by native Risk."""
  subject = _required_text(user_id, "user_id")
  if not isinstance(tenant, AdmittedTenant):
    raise InvestmentCapabilityClaimError(
      "risk job capability claim requires an authenticated tenant"
    )
  session = _required_text(session_id, "session_id")
  token_id = _required_text(cell_token, "cell_token")
  issued_at = int(time.time()) if now is None else now
  payload: dict[str, object] = {
    "aud": RISK_CAPABILITY_CLAIM_AUDIENCE,
    "channel": channel,
    "exp": issued_at + INVESTMENT_CAPABILITY_CLAIM_TTL_SECONDS,
    "granted_operations": sorted(set(granted_operations)),
    "iat": issued_at,
    "iss": INVESTMENT_CAPABILITY_CLAIM_ISSUER,
    "jti": token_id,
    "cell_token": token_id,
    "kind": RISK_JOB_CAPABILITY_CLAIM_KIND,
    "session_id": session,
    "sub": subject,
    "tenant": tenant.model_dump(mode="json"),
    "user_id": subject,
    "v": RISK_JOB_CAPABILITY_CLAIM_VERSION,
  }
  return _sign(payload)



def issue_investment_selected_content_claim(
  *,
  user_id: str,
  artifact_id: str,
  view: str,
  now: int | None = None,
) -> str:
  """Authorize one exact bounded Investment artifact view for admission."""

  subject = _required_text(user_id, "user_id")
  selected_artifact_id = _required_text(artifact_id, "artifact_id")
  selected_view = _required_text(view, "view")
  issued_at = int(time.time()) if now is None else now
  if isinstance(issued_at, bool) or not isinstance(issued_at, int):
    raise InvestmentCapabilityClaimError(
      "investment selected-content claim requires an integer issuance time"
    )

  payload: dict[str, object] = {
    "artifact_id": selected_artifact_id,
    "aud": INVESTMENT_CAPABILITY_CLAIM_AUDIENCE,
    "exp": issued_at + INVESTMENT_CAPABILITY_CLAIM_TTL_SECONDS,
    "iat": issued_at,
    "iss": INVESTMENT_CAPABILITY_CLAIM_ISSUER,
    "purpose": INVESTMENT_SELECTED_CONTENT_CLAIM_PURPOSE,
    "sub": subject,
    "tool_name": "get_investment_artifact",
    "v": 2,
    "view": selected_view,
  }
  return _sign(payload)


__all__ = [
  "INVESTMENT_CAPABILITY_CLAIM_AUDIENCE",
  "INVESTMENT_CAPABILITY_CLAIM_ISSUER",
  "INVESTMENT_CAPABILITY_CLAIM_KEY_BYTES",
  "INVESTMENT_CAPABILITY_CLAIM_KIND",
  "INVESTMENT_CAPABILITY_CLAIM_VERSION",
  "INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV",
  "INVESTMENT_CAPABILITY_CLAIM_PUBLIC_KEY_ENV",
  "INVESTMENT_CAPABILITY_CLAIM_SERVER",
  "INVESTMENT_CAPABILITY_CLAIM_TTL_SECONDS",
  "INVESTMENT_SELECTED_CONTENT_CLAIM_PURPOSE",
  "INVESTMENT_CAPABILITY_FACADE_TOOLS",
  "INVESTMENT_READBACK_FACADE_TOOLS",
  "INVESTMENT_JOB_PRINCIPAL_CLAIM_KIND",
  "INVESTMENT_JOB_PRINCIPAL_CLAIM_VERSION",
  "INVESTMENT_JOB_PRINCIPAL_TOOLS",
  "InvestmentCapabilityGrant",
  "InvestmentCapabilityClaimError",
  "configure_investment_capability_grants",
  "investment_capability_grants",
  "issue_investment_capability_claim",
  "issue_investment_job_principal_claim",
  "issue_investment_selected_content_claim",
  "issue_risk_job_capability_claim",
  "investment_capability_claim_unavailable_error",
  "investment_capability_signing_available",
]
