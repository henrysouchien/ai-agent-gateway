from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from types import MappingProxyType
from typing import Any, Final, Mapping, NamedTuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .skill_limits import SkillExecutionLimits
from .skill_context import current_skill

INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV: Final = (
  "INVESTMENT_CAPABILITY_CLAIM_ED25519_PRIVATE_KEY"
)
INVESTMENT_CAPABILITY_CLAIM_PUBLIC_KEY_ENV: Final = (
  "INVESTMENT_CAPABILITY_CLAIM_ED25519_PUBLIC_KEY"
)
INVESTMENT_CAPABILITY_CLAIM_ISSUER: Final = "ai-excel-addin"
INVESTMENT_CAPABILITY_CLAIM_AUDIENCE: Final = "investment-tools"
INVESTMENT_CAPABILITY_CLAIM_TTL_SECONDS: Final = 60
INVESTMENT_CAPABILITY_CLAIM_KEY_BYTES: Final = 32
INVESTMENT_SELECTED_CONTENT_CLAIM_PURPOSE: Final = "selected_content_read"
_MAX_ID_CHARS = 256
_MAX_CHANNEL_CHARS = 64
_MAX_TOKEN_CHARS = 16_384
_MAX_RESEARCH_FILE_ID = (1 << 63) - 1
_POLICY_BUNDLE_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")

INVESTMENT_CAPABILITY_FACADE_TOOLS: Final = frozenset({
  "cancel_investment_run",
  "get_investment_artifact",
  "get_investment_capability",
  "get_investment_run",
  "list_investment_artifacts",
  "list_investment_capabilities",
  "start_investment_run",
  "start_quant_research",
})

INVESTMENT_CAPABILITY_CLAIM_SERVER: Final = "idea-workbench-mcp"


class InvestmentCapabilitySkillGrant(NamedTuple):
  allowed_capability_ids: tuple[str, ...]
  allowed_tool_names: tuple[str, ...]
  effect_classes: tuple[str, ...]
  max_wall_clock_seconds: int | None


# This mapping is gateway policy, never model input. A skill receives only the
# narrow capability/effect set reviewed here, and the receiving server further
# intersects it with each tool wrapper's route maximum.
INVESTMENT_CAPABILITY_SKILL_GRANTS: Final[Mapping[str, InvestmentCapabilitySkillGrant]] = (
  MappingProxyType({
    "market-scan": InvestmentCapabilitySkillGrant(
      allowed_capability_ids=(
        "fingerprint_screen",
        "insider_buying",
        "quality_screen",
      ),
      allowed_tool_names=(
        "cancel_investment_run",
        "get_investment_artifact",
        "get_investment_capability",
        "get_investment_run",
        "list_investment_artifacts",
        "list_investment_capabilities",
        "start_investment_run",
      ),
      effect_classes=("artifact_write", "read", "state_write"),
      max_wall_clock_seconds=None,
    ),
    "quant-research": InvestmentCapabilitySkillGrant(
      allowed_capability_ids=("quant_research",),
      allowed_tool_names=(
        "cancel_investment_run",
        "get_investment_artifact",
        "get_investment_capability",
        "get_investment_run",
        "list_investment_artifacts",
        "list_investment_capabilities",
        "start_quant_research",
      ),
      effect_classes=("read", "research_write", "state_write"),
      max_wall_clock_seconds=5_400,
    ),
  })
)


class InvestmentCapabilityClaimError(RuntimeError):
  """A trusted investment-capability claim could not be issued."""


def investment_capability_claim_unavailable_error(
  *,
  subject: str,
) -> dict[str, Any]:
  """Named refusal for a missing investment-capability grant.

  The grant set and issuance route are derived from
  ``INVESTMENT_CAPABILITY_SKILL_GRANTS``. Callers carry this payload; they
  do not restate the grants.
  """

  grants = tuple(sorted(INVESTMENT_CAPABILITY_SKILL_GRANTS))
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
  }


def investment_capability_server_load_error(
  server_name: str,
) -> dict[str, Any] | None:
  """Refuse loading a claim-gated server the session cannot hold a grant for."""

  if server_name != INVESTMENT_CAPABILITY_CLAIM_SERVER:
    return None
  if current_skill() in INVESTMENT_CAPABILITY_SKILL_GRANTS:
    return None
  return investment_capability_claim_unavailable_error(
    subject=f"server '{server_name}'",
  )



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


def investment_capability_signing_available() -> bool:
  """Whether the hydrated private key can issue Investment claims."""

  try:
    _signing_key()
  except InvestmentCapabilityClaimError:
    return False
  return True


def _approved_budget(
  execution_limits: SkillExecutionLimits,
  grant: InvestmentCapabilitySkillGrant,
) -> dict[str, int | float]:
  if type(execution_limits) is not SkillExecutionLimits:
    raise InvestmentCapabilityClaimError(
      "investment capability skill policy is unavailable"
    )
  max_cost_usd = execution_limits.max_budget_usd
  max_tokens = execution_limits.max_tokens
  max_turns = execution_limits.max_turns
  if (
    max_cost_usd is None
    or not 0 < max_cost_usd <= 10_000
    or max_tokens is None
    or not 1 <= max_tokens <= 100_000_000
    or max_turns is None
    or not 1 <= max_turns <= 10_000
  ):
    raise InvestmentCapabilityClaimError(
      "investment capability skill policy has an invalid approved budget"
    )
  wall_clock = grant.max_wall_clock_seconds
  if (
    isinstance(wall_clock, bool)
    or not isinstance(wall_clock, int)
    or not 1 <= wall_clock <= 604_800
  ):
    raise InvestmentCapabilityClaimError(
      "investment capability skill policy has an invalid approved budget"
    )
  return {
    "max_cost_usd": float(max_cost_usd),
    "max_tokens": max_tokens,
    "max_turns": max_turns,
    "max_wall_clock_seconds": wall_clock,
  }


def issue_investment_capability_claim(
  *,
  user_id: str,
  session_id: str,
  skill_run_id: str,
  channel: str,
  tool_name: str,
  request_id: str,
  jti: str,
  skill: str,
  admitted_skill_execution_limits: SkillExecutionLimits | None,
  policy_bundle_hash: str,
  research_file_id: int | None = None,
  now: int | None = None,
) -> str:
  """Issue the short-lived gateway assertion consumed by investment-tools.

  The model never supplies this payload. The gateway binds authenticated
  session identity and the exact routed MCP tool immediately before dispatch.
  """

  subject = _required_text(user_id, "user_id")
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
  trusted_skill = _required_text(skill, "skill")
  trusted_policy_bundle = _required_text(
    policy_bundle_hash,
    "policy_bundle_hash",
  )
  if _POLICY_BUNDLE_HASH_PATTERN.fullmatch(trusted_policy_bundle) is None:
    raise InvestmentCapabilityClaimError(
      "investment capability claim requires policy_bundle_hash"
    )
  if routed_tool not in INVESTMENT_CAPABILITY_FACADE_TOOLS:
    raise InvestmentCapabilityClaimError(
      "investment capability claim cannot bind an unsupported tool"
    )
  grant = INVESTMENT_CAPABILITY_SKILL_GRANTS.get(trusted_skill)
  if grant is None:
    raise InvestmentCapabilityClaimError(
      "investment capability claim cannot bind an unsupported skill"
    )
  if routed_tool not in grant.allowed_tool_names:
    raise InvestmentCapabilityClaimError(
      "investment capability claim cannot bind an unsupported tool"
    )
  if "quant_research" in grant.allowed_capability_ids:
    if type(admitted_skill_execution_limits) is not SkillExecutionLimits:
      raise InvestmentCapabilityClaimError(
        "investment capability skill policy is unavailable"
      )
    approved_budget = _approved_budget(
      admitted_skill_execution_limits,
      grant,
    )
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

  private_key, key_id = _signing_key()
  header: dict[str, object] = {
    "alg": "EdDSA",
    "kid": key_id,
    "typ": "JWT",
  }
  payload: dict[str, object] = {
    "allowed_capability_ids": sorted(grant.allowed_capability_ids),
    "aud": INVESTMENT_CAPABILITY_CLAIM_AUDIENCE,
    "channel": routed_channel,
    "effect_classes": sorted(grant.effect_classes),
    "exp": issued_at + INVESTMENT_CAPABILITY_CLAIM_TTL_SECONDS,
    "iat": issued_at,
    "iss": INVESTMENT_CAPABILITY_CLAIM_ISSUER,
    "jti": token_id,
    "policy_bundle_hash": trusted_policy_bundle,
    "request_id": request,
    "session_id": session,
    "skill": trusted_skill,
    "skill_run_id": skill_run,
    "sub": subject,
    "tool_name": routed_tool,
    "user_id": subject,
    "v": 1,
  }
  if approved_budget is not None:
    payload["approved_budget"] = approved_budget
  if routed_tool == "start_quant_research":
    payload["research_file_id"] = research_file_id
  signing_input = b".".join((
    _b64url(_canonical_json(header)).encode("ascii"),
    _b64url(_canonical_json(payload)).encode("ascii"),
  ))
  signature = private_key.sign(signing_input)
  token = f"{signing_input.decode('ascii')}.{_b64url(signature)}"
  if len(token) > _MAX_TOKEN_CHARS:
    raise InvestmentCapabilityClaimError(
      "investment capability claim exceeds its transport bound"
    )
  return token


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
  selected_view = _required_text(view, "view", maximum=32)
  if selected_view not in {"summary", "excerpt"}:
    raise InvestmentCapabilityClaimError(
      "investment selected-content claim requires a supported view"
    )
  issued_at = int(time.time()) if now is None else now
  if isinstance(issued_at, bool) or not isinstance(issued_at, int):
    raise InvestmentCapabilityClaimError(
      "investment selected-content claim requires an integer issuance time"
    )

  private_key, key_id = _signing_key()
  header: dict[str, object] = {
    "alg": "EdDSA",
    "kid": key_id,
    "typ": "JWT",
  }
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
  signing_input = b".".join((
    _b64url(_canonical_json(header)).encode("ascii"),
    _b64url(_canonical_json(payload)).encode("ascii"),
  ))
  signature = private_key.sign(signing_input)
  token = f"{signing_input.decode('ascii')}.{_b64url(signature)}"
  if len(token) > _MAX_TOKEN_CHARS:
    raise InvestmentCapabilityClaimError(
      "investment selected-content claim exceeds its transport bound"
    )
  return token


__all__ = [
  "INVESTMENT_CAPABILITY_CLAIM_AUDIENCE",
  "INVESTMENT_CAPABILITY_CLAIM_ISSUER",
  "INVESTMENT_CAPABILITY_CLAIM_KEY_BYTES",
  "INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV",
  "INVESTMENT_CAPABILITY_CLAIM_PUBLIC_KEY_ENV",
  "INVESTMENT_CAPABILITY_CLAIM_SERVER",
  "INVESTMENT_CAPABILITY_CLAIM_TTL_SECONDS",
  "INVESTMENT_SELECTED_CONTENT_CLAIM_PURPOSE",
  "INVESTMENT_CAPABILITY_FACADE_TOOLS",
  "INVESTMENT_CAPABILITY_SKILL_GRANTS",
  "InvestmentCapabilitySkillGrant",
  "InvestmentCapabilityClaimError",
  "issue_investment_capability_claim",
  "issue_investment_selected_content_claim",
  "investment_capability_claim_unavailable_error",
  "investment_capability_server_load_error",
  "investment_capability_signing_available",
]
