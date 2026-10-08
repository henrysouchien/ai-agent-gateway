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

INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV: Final = (
  "INVESTMENT_CAPABILITY_CLAIM_ED25519_PRIVATE_KEY"
)
INVESTMENT_CAPABILITY_CLAIM_PUBLIC_KEY_ENV: Final = (
  "INVESTMENT_CAPABILITY_CLAIM_ED25519_PUBLIC_KEY"
)
INVESTMENT_CAPABILITY_CLAIM_ISSUER: Final = "ai-excel-addin"
INVESTMENT_CAPABILITY_CLAIM_AUDIENCE: Final = "investment-tools"
# ``exp`` is the replay bound of a signed claim: a captured token stops
# authorizing a user's investment tools 60 seconds after issuance, and
# investment-tools rejects any claim whose exp - iat exceeds this.
INVESTMENT_CAPABILITY_CLAIM_TTL_SECONDS: Final = 60
INVESTMENT_CAPABILITY_CLAIM_KEY_BYTES: Final = 32
INVESTMENT_SELECTED_CONTENT_CLAIM_PURPOSE: Final = "selected_content_read"
_MAX_ID_CHARS = 256
_MAX_CHANNEL_CHARS = 64
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


class InvestmentCapabilityGrant(NamedTuple):
  """One skill's investment grant, projected from its own skill note.

  ``capability_ids`` and ``effect_classes`` are the note's
  ``semantic_metadata.investment_grant``; ``tool_names`` are the tools its
  ``tool_refs`` declare on ``INVESTMENT_CAPABILITY_CLAIM_SERVER``.
  investment-tools owns both vocabularies and intersects every claim with the
  routed tool's own maximum, so the gateway restates neither.
  """

  capability_ids: tuple[str, ...]
  effect_classes: tuple[str, ...]
  tool_names: tuple[str, ...]


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
  grant = _skill_grants.get(trusted_skill)
  if grant is None:
    raise InvestmentCapabilityClaimError(
      "investment capability claim cannot bind an unsupported skill"
    )
  if routed_tool not in grant.tool_names:
    raise InvestmentCapabilityClaimError(
      "investment capability claim cannot bind an unsupported tool"
    )
  if "quant_research" in grant.capability_ids:
    approved_budget = _approved_budget(admitted_skill_execution_limits)
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
    "allowed_capability_ids": sorted(grant.capability_ids),
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
  return f"{signing_input.decode('ascii')}.{_b64url(signature)}"


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
  return f"{signing_input.decode('ascii')}.{_b64url(signature)}"


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
  "InvestmentCapabilityGrant",
  "InvestmentCapabilityClaimError",
  "configure_investment_capability_grants",
  "investment_capability_grants",
  "issue_investment_capability_claim",
  "issue_investment_selected_content_claim",
  "investment_capability_claim_unavailable_error",
  "investment_capability_signing_available",
]
