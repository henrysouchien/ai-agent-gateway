"""Offline verification for risk_module commercial execution claims."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import json
import re
import time
from typing import Callable, Mapping
from uuid import UUID

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key
import jwt


COMMERCIAL_CLAIM_ALGORITHM = "EdDSA"
COMMERCIAL_CLAIM_ISSUER = "risk-module-commercial-control"
COMMERCIAL_CLAIM_AUDIENCE = "hank-agent-gateway"
_STABLE_CODE = re.compile(r"^[a-z][a-z0-9._:-]{0,127}$")
_SUBJECT = re.compile(r"^user:[1-9][0-9]*$")
COMMERCIAL_CLAIM_REQUIRED_FIELDS = frozenset({
  "schema_version", "kid", "iss", "aud", "sub", "environment", "surface",
  "commercial_account_id", "agreement_id", "agreement_terms_revision",
  "offer_code", "effective_scopes", "entitlement_revision", "payer_policy_version",
  "budget_policy_version", "shadow_rate_version", "manifest_version",
  "authorized_work_start_deadline", "usage_accept_until", "iat", "exp", "jti",
})


class CommercialClaimError(ValueError):
  def __init__(self, code: str) -> None:
    self.code = code
    super().__init__(code)


@dataclass(frozen=True)
class CommercialContextState:
  active: bool
  entitlement_revision: int


@dataclass(frozen=True)
class CommercialClaimTrustSnapshot:
  public_keys_by_id: Mapping[str, bytes]
  environment: str
  manifest_versions: frozenset[str]
  payer_policy_versions: frozenset[str]
  budget_policy_versions: frozenset[str]
  shadow_rate_versions: frozenset[str]
  current_agreement_terms_revision: Callable[[UUID], int | None]
  resolve_context_state: Callable[[UUID], CommercialContextState] | None = None

  def __post_init__(self) -> None:
    if self.environment not in {"dev", "staging", "prod"}:
      raise ValueError("commercial trust environment is invalid")
    if not self.public_keys_by_id:
      raise ValueError("commercial trust snapshot requires verification keys")
    for key_id, pem in self.public_keys_by_id.items():
      if not _STABLE_CODE.fullmatch(key_id):
        raise ValueError("commercial verification key id is invalid")
      key = load_pem_public_key(pem)
      if not isinstance(key, Ed25519PublicKey):
        raise ValueError("commercial verification keys must be Ed25519")
    for versions in (
      self.manifest_versions, self.payer_policy_versions,
      self.budget_policy_versions, self.shadow_rate_versions,
    ):
      if not versions or any(not _valid_version(value) for value in versions):
        raise ValueError("commercial trust versions must be non-empty bounded strings")


@dataclass(frozen=True)
class VerifiedCommercialClaim:
  schema_version: int
  key_id: str
  subject: str
  environment: str
  surface: str
  commercial_account_id: UUID
  agreement_id: UUID
  agreement_terms_revision: int
  offer_code: str
  effective_scopes: tuple[str, ...]
  entitlement_revision: int
  payer_policy_version: str
  budget_policy_version: str
  shadow_rate_version: str
  manifest_version: str
  authorized_work_start_deadline: int
  usage_accept_until: int
  issued_at: int
  expires_at: int
  context_id: UUID


class CommercialClaimVerifier:
  def __init__(self, trust: CommercialClaimTrustSnapshot) -> None:
    self._trust = trust
    self._keys = {
      key_id: load_pem_public_key(pem)
      for key_id, pem in trust.public_keys_by_id.items()
    }

  def verify_for_work_start(
    self,
    token: str,
    *,
    now: int | None = None,
    require_live_context: bool = False,
  ) -> VerifiedCommercialClaim:
    current = int(time.time()) if now is None else now
    if type(current) is not int:
      raise CommercialClaimError("invalid_verification_time")
    self._validate_compact_token(token)
    header = self._header(token)
    key_id = header.get("kid")
    if header.get("alg") != COMMERCIAL_CLAIM_ALGORITHM:
      raise CommercialClaimError("wrong_algorithm")
    if not isinstance(key_id, str) or key_id not in self._keys:
      raise CommercialClaimError("unknown_key")
    try:
      payload = jwt.decode(
        token,
        self._keys[key_id],
        algorithms=[COMMERCIAL_CLAIM_ALGORITHM],
        issuer=COMMERCIAL_CLAIM_ISSUER,
        audience=COMMERCIAL_CLAIM_AUDIENCE,
        options={
          "require": sorted(COMMERCIAL_CLAIM_REQUIRED_FIELDS),
          "verify_exp": False,
          "verify_iat": False,
        },
      )
    except jwt.PyJWTError as exc:
      raise CommercialClaimError("invalid_signature_or_registered_claim") from exc
    claim = self._validate_payload(payload, key_id=key_id)
    if current < claim.issued_at:
      raise CommercialClaimError("claim_not_yet_valid")
    if current >= claim.expires_at or current > claim.authorized_work_start_deadline:
      raise CommercialClaimError("start_authority_expired")
    if claim.environment != self._trust.environment:
      raise CommercialClaimError("environment_mismatch")
    if claim.manifest_version not in self._trust.manifest_versions:
      raise CommercialClaimError("manifest_skew")
    if claim.payer_policy_version not in self._trust.payer_policy_versions:
      raise CommercialClaimError("unknown_payer_policy")
    if claim.budget_policy_version not in self._trust.budget_policy_versions:
      raise CommercialClaimError("unknown_budget_policy")
    if claim.shadow_rate_version not in self._trust.shadow_rate_versions:
      raise CommercialClaimError("unknown_rate_policy")
    current_terms_revision = self._trust.current_agreement_terms_revision(
      claim.agreement_id
    )
    if current_terms_revision is None:
      raise CommercialClaimError("unknown_agreement")
    if current_terms_revision != claim.agreement_terms_revision:
      raise CommercialClaimError("agreement_terms_stale")
    if require_live_context:
      resolver = self._trust.resolve_context_state
      if resolver is None:
        raise CommercialClaimError("live_context_check_unavailable")
      state = resolver(claim.context_id)
      if not state.active:
        raise CommercialClaimError("execution_context_revoked")
      if state.entitlement_revision != claim.entitlement_revision:
        raise CommercialClaimError("entitlement_revision_stale")
    return claim

  def verify_for_irreversible_submission(
    self, token: str, *, now: int | None = None
  ) -> VerifiedCommercialClaim:
    return self.verify_for_work_start(
      token, now=now, require_live_context=True
    )

  def recheck_verified_for_irreversible_submission(
    self,
    claim: VerifiedCommercialClaim,
    *,
    now: int | None = None,
  ) -> None:
    """Freshly recheck token-free verified authority before a provider mutation."""
    current = int(time.time()) if now is None else now
    if type(current) is not int:
      raise CommercialClaimError("invalid_verification_time")
    if current < claim.issued_at or current >= claim.expires_at:
      raise CommercialClaimError("start_authority_expired")
    current_terms_revision = self._trust.current_agreement_terms_revision(
      claim.agreement_id
    )
    if current_terms_revision is None:
      raise CommercialClaimError("unknown_agreement")
    if current_terms_revision != claim.agreement_terms_revision:
      raise CommercialClaimError("agreement_terms_stale")
    resolver = self._trust.resolve_context_state
    if resolver is None:
      raise CommercialClaimError("live_context_check_unavailable")
    state = resolver(claim.context_id)
    if not state.active:
      raise CommercialClaimError("execution_context_revoked")
    if state.entitlement_revision != claim.entitlement_revision:
      raise CommercialClaimError("entitlement_revision_stale")

  def uses_context_resolver(self, resolver: Callable) -> bool:
    return self._trust.resolve_context_state == resolver

  @staticmethod
  def _validate_compact_token(token: str) -> None:
    if type(token) is not str:
      raise CommercialClaimError("invalid_token_type")
    try:
      encoded = token.encode("ascii")
    except UnicodeEncodeError as exc:
      raise CommercialClaimError("invalid_compact_token") from exc
    segments = token.split(".")
    if (
      not 1 <= len(encoded) <= 4096
      or len(segments) != 3
      or any(not segment for segment in segments)
      or any(character.isspace() for character in token)
    ):
      raise CommercialClaimError("invalid_compact_token")

  @staticmethod
  def _header(token: str) -> dict:
    try:
      segment = token.split(".", 1)[0]
      padding = "=" * (-len(segment) % 4)
      raw = base64.b64decode(
        segment + padding, altchars=b"-_", validate=True
      )
      header = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
      raise CommercialClaimError("malformed_token") from exc
    if not isinstance(header, dict):
      raise CommercialClaimError("malformed_token")
    if set(header) != {"alg", "kid", "typ"}:
      raise CommercialClaimError("unsupported_protected_header")
    if header.get("typ") != "JWT":
      raise CommercialClaimError("wrong_token_type")
    return header

  @staticmethod
  def _validate_payload(payload: dict, *, key_id: str) -> VerifiedCommercialClaim:
    if set(payload) != COMMERCIAL_CLAIM_REQUIRED_FIELDS:
      raise CommercialClaimError("unknown_or_missing_claim")
    if payload.get("schema_version") != 1:
      raise CommercialClaimError("unknown_schema")
    if payload.get("kid") != key_id:
      raise CommercialClaimError("key_id_mismatch")
    if payload.get("iss") != COMMERCIAL_CLAIM_ISSUER:
      raise CommercialClaimError("wrong_issuer")
    if payload.get("aud") != COMMERCIAL_CLAIM_AUDIENCE:
      raise CommercialClaimError("wrong_audience")
    subject = payload.get("sub")
    if not isinstance(subject, str) or not _SUBJECT.fullmatch(subject):
      raise CommercialClaimError("invalid_subject")
    environment = payload.get("environment")
    if environment not in {"dev", "staging", "prod"}:
      raise CommercialClaimError("invalid_environment")
    surface = _require_code(payload, "surface")
    offer = _require_code(payload, "offer_code")
    account_id = _require_uuid(payload, "commercial_account_id")
    agreement_id = _require_uuid(payload, "agreement_id")
    context_id = _require_uuid(payload, "jti")
    terms_revision = _require_positive_int(payload, "agreement_terms_revision")
    entitlement_revision = _require_positive_int(payload, "entitlement_revision")
    issued_at = _require_positive_int(payload, "iat")
    expires_at = _require_positive_int(payload, "exp")
    start_deadline = _require_positive_int(payload, "authorized_work_start_deadline")
    usage_accept_until = _require_positive_int(payload, "usage_accept_until")
    if expires_at <= issued_at or expires_at - issued_at > 300:
      raise CommercialClaimError("invalid_claim_lifetime")
    if not issued_at <= start_deadline <= expires_at <= usage_accept_until:
      raise CommercialClaimError("invalid_claim_window")
    raw_scopes = payload.get("effective_scopes")
    if not isinstance(raw_scopes, list) or not raw_scopes:
      raise CommercialClaimError("invalid_scopes")
    scopes = tuple(raw_scopes)
    if (
      any(not isinstance(scope, str) or not _STABLE_CODE.fullmatch(scope) for scope in scopes)
      or tuple(sorted(set(scopes))) != scopes
    ):
      raise CommercialClaimError("invalid_scopes")
    return VerifiedCommercialClaim(
      schema_version=1, key_id=key_id, subject=subject, environment=environment,
      surface=surface, commercial_account_id=account_id, agreement_id=agreement_id,
      agreement_terms_revision=terms_revision, offer_code=offer,
      effective_scopes=scopes, entitlement_revision=entitlement_revision,
      payer_policy_version=_require_version(payload, "payer_policy_version"),
      budget_policy_version=_require_version(payload, "budget_policy_version"),
      shadow_rate_version=_require_version(payload, "shadow_rate_version"),
      manifest_version=_require_version(payload, "manifest_version"),
      authorized_work_start_deadline=start_deadline,
      usage_accept_until=usage_accept_until, issued_at=issued_at,
      expires_at=expires_at, context_id=context_id,
    )


def _valid_version(value: object) -> bool:
  return isinstance(value, str) and 1 <= len(value) <= 256 and not value.isspace()


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
  result = {}
  for key, value in pairs:
    if key in result:
      raise json.JSONDecodeError("duplicate object key", key, 0)
    result[key] = value
  return result


def _require_version(payload: dict, field: str) -> str:
  value = payload.get(field)
  if not _valid_version(value):
    raise CommercialClaimError(f"invalid_{field}")
  return value


def _require_code(payload: dict, field: str) -> str:
  value = payload.get(field)
  if not isinstance(value, str) or not _STABLE_CODE.fullmatch(value):
    raise CommercialClaimError(f"invalid_{field}")
  return value


def _require_positive_int(payload: dict, field: str) -> int:
  value = payload.get(field)
  if type(value) is not int or value <= 0:
    raise CommercialClaimError(f"invalid_{field}")
  return value


def _require_uuid(payload: dict, field: str) -> UUID:
  value = payload.get(field)
  if not isinstance(value, str):
    raise CommercialClaimError(f"invalid_{field}")
  try:
    return UUID(value)
  except ValueError as exc:
    raise CommercialClaimError(f"invalid_{field}") from exc
