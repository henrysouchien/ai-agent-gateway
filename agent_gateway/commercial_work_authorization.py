"""Offline verification for one signed commercial gateway work start."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import time
from typing import Callable
from uuid import UUID

import jwt

from hank_contracts.commercial_claims import VerifiedCommercialClaim
from hank_contracts.commercial_work import (
  _CODE,
  _load_ed25519_public_key,
  WORK_AUTHORIZATION_ALGORITHM,
  WORK_AUTHORIZATION_AUDIENCE,
  WORK_AUTHORIZATION_ISSUER,
  WORK_AUTHORIZATION_REQUIRED_FIELDS as WORK_AUTHORIZATION_REQUIRED_FIELDS,
  WORK_AUTHORIZATION_V3_REQUIRED_FIELDS as WORK_AUTHORIZATION_V3_REQUIRED_FIELDS,
  WorkAuthorizationError as WorkAuthorizationError,
  WorkAuthorizationTrustSnapshot as WorkAuthorizationTrustSnapshot,
  VerifiedWorkAuthorization as VerifiedWorkAuthorization,
)
_OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,254}$")
_BASE64URL_SEGMENT = re.compile(r"^[A-Za-z0-9_-]+$")


class WorkAuthorizationVerifier:
  def __init__(
    self,
    trust: WorkAuthorizationTrustSnapshot,
    *,
    clock: Callable[[], int] | None = None,
  ) -> None:
    self._keys = {
      key_id: _load_ed25519_public_key(pem)
      for key_id, pem in trust.public_keys_by_id.items()
    }
    self._environment = trust.environment
    self._manifest_version = trust.manifest_version
    self._retired_at_by_key_id = dict(trust.retired_at_by_key_id)
    self._capability_required_scopes = {
      capability: frozenset(scopes)
      for capability, scopes in trust.capability_required_scopes.items()
    }
    self._capability_allowed_operations = {
      capability: frozenset(operations)
      for capability, operations in trust.capability_allowed_operations.items()
    }
    self._clock = clock or (lambda: int(time.time()))

  def verify_for_attach(
    self,
    token: str,
    *,
    execution_claim: VerifiedCommercialClaim,
    request_id: str,
    session_id: str,
    operation: str,
    billing_mode: str,
    capability_id: str | None,
    credential_principal: str,
  ) -> VerifiedWorkAuthorization:
    current = self._clock()
    if type(current) is not int:
      raise WorkAuthorizationError("invalid_verification_time")
    for name, value in (("request", request_id), ("session", session_id)):
      if not isinstance(value, str) or not _OPAQUE_ID.fullmatch(value):
        raise WorkAuthorizationError(f"invalid_{name}_id")
    if not isinstance(operation, str) or not _CODE.fullmatch(operation):
      raise WorkAuthorizationError("invalid_operation")
    if billing_mode not in {"byok", "metered"}:
      raise WorkAuthorizationError("invalid_billing_mode")
    if capability_id is not None and (
      not isinstance(capability_id, str) or not _CODE.fullmatch(capability_id)
    ):
      raise WorkAuthorizationError("invalid_capability_id")

    encoded = self._validate_compact_token(token)
    header = self._header(token)
    key_id = header.get("kid")
    if header.get("alg") != WORK_AUTHORIZATION_ALGORITHM:
      raise WorkAuthorizationError("wrong_algorithm")
    if not isinstance(key_id, str) or key_id not in self._keys:
      raise WorkAuthorizationError("unknown_key")
    self._validate_payload_json(token)
    try:
      payload = jwt.decode(
        token,
        self._keys[key_id],
        algorithms=[WORK_AUTHORIZATION_ALGORITHM],
        issuer=WORK_AUTHORIZATION_ISSUER,
        audience=WORK_AUTHORIZATION_AUDIENCE,
        options={
          "require": ["iss", "aud", "iat", "exp", "jti"],
          "verify_exp": False,
          "verify_iat": False,
        },
      )
    except jwt.PyJWTError as exc:
      raise WorkAuthorizationError(
        "invalid_signature_or_registered_claim"
      ) from exc
    authorization = self._validate_payload(
      payload,
      key_id=key_id,
      token_sha256="sha256:" + hashlib.sha256(encoded).hexdigest(),
    )
    if current < authorization.issued_at:
      raise WorkAuthorizationError("authorization_not_yet_valid")
    if current >= authorization.expires_at:
      raise WorkAuthorizationError("start_authority_expired")
    retired_at = self._retired_at_by_key_id[key_id]
    if retired_at is not None and authorization.issued_at > retired_at:
      raise WorkAuthorizationError("key_retired")
    if authorization.environment != self._environment:
      raise WorkAuthorizationError("environment_mismatch")
    if execution_claim.environment != self._environment:
      raise WorkAuthorizationError("execution_claim_environment_mismatch")
    if (
      current < execution_claim.issued_at
      or current >= execution_claim.expires_at
      or current > execution_claim.authorized_work_start_deadline
    ):
      raise WorkAuthorizationError("execution_claim_start_authority_expired")
    if authorization.execution_context_id != execution_claim.context_id:
      raise WorkAuthorizationError("execution_claim_mismatch")
    if (
      authorization.issued_at < execution_claim.issued_at
      or authorization.expires_at > execution_claim.authorized_work_start_deadline
    ):
      raise WorkAuthorizationError("execution_claim_window_mismatch")
    if execution_claim.manifest_version != self._manifest_version:
      raise WorkAuthorizationError("manifest_skew")
    if authorization.request_id != request_id:
      raise WorkAuthorizationError("request_mismatch")
    if authorization.session_id != session_id:
      raise WorkAuthorizationError("session_mismatch")
    if authorization.operation != operation:
      raise WorkAuthorizationError("operation_mismatch")
    if authorization.billing_mode != billing_mode:
      raise WorkAuthorizationError("billing_mode_mismatch")
    if authorization.capability_id != capability_id:
      raise WorkAuthorizationError("capability_mismatch")
    # Money: a commercial start may run only the capabilities and operations
    # the signed claim's scopes fund, else one paid claim funds unpriced work.
    required_scopes = self._capability_required_scopes.get(
      authorization.capability_id or ""
    )
    if required_scopes is None or not required_scopes.issubset(
      execution_claim.effective_scopes
    ):
      raise WorkAuthorizationError("capability_not_authorized")
    allowed_operations = self._capability_allowed_operations[
      authorization.capability_id or ""
    ]
    if authorization.operation not in allowed_operations:
      raise WorkAuthorizationError("operation_not_authorized")
    # Money (rulings-index WA-ACCOUNT-BUDGET): the ticket binds the account,
    # not the provider or model the call happens to use.
    if authorization.credential_principal != credential_principal:
      raise WorkAuthorizationError("credential_principal_mismatch")
    return authorization

  @staticmethod
  def _validate_compact_token(token: str) -> bytes:
    if type(token) is not str:
      raise WorkAuthorizationError("invalid_token_type")
    try:
      encoded = token.encode("ascii")
    except UnicodeEncodeError as exc:
      raise WorkAuthorizationError("invalid_compact_token") from exc
    segments = token.split(".")
    if (
      not 1 <= len(encoded) <= 4096
      or len(segments) != 3
      or any(not segment for segment in segments)
      or any(character.isspace() for character in token)
    ):
      raise WorkAuthorizationError("invalid_compact_token")
    for segment in segments:
      _decode_base64url_segment(segment)
    return encoded

  @staticmethod
  def _header(token: str) -> dict:
    header = _decode_json_segment(token.split(".", 1)[0])
    if set(header) != {"alg", "kid", "typ"}:
      raise WorkAuthorizationError("unsupported_protected_header")
    if header.get("typ") != "JWT":
      raise WorkAuthorizationError("wrong_token_type")
    return header

  @staticmethod
  def _validate_payload_json(token: str) -> None:
    _decode_json_segment(token.split(".")[1])

  @staticmethod
  def _validate_payload(
    payload: dict, *, key_id: str, token_sha256: str
  ) -> VerifiedWorkAuthorization:
    fields = set(payload)
    version = payload.get("schema_version")
    # V1 and V2 tickets are not accepted. The gate is off, so none are live.
    if type(version) is int and version in (1, 2):
      raise WorkAuthorizationError("unknown_schema")
    if fields != WORK_AUTHORIZATION_V3_REQUIRED_FIELDS:
      raise WorkAuthorizationError("unknown_or_missing_claim")
    if type(version) is not int or version != 3:
      raise WorkAuthorizationError("unknown_schema")
    schema_version = 3
    if payload.get("kid") != key_id:
      raise WorkAuthorizationError("key_id_mismatch")
    if payload.get("iss") != WORK_AUTHORIZATION_ISSUER:
      raise WorkAuthorizationError("wrong_issuer")
    if payload.get("aud") != WORK_AUTHORIZATION_AUDIENCE:
      raise WorkAuthorizationError("wrong_audience")
    environment = payload.get("environment")
    if environment not in {"dev", "staging", "prod"}:
      raise WorkAuthorizationError("invalid_environment")
    authorization_id = _require_uuid(payload, "jti")
    context_id = _require_uuid(payload, "execution_claim_jti")
    workflow_id = _require_uuid(payload, "workflow_run_id")
    attempt_group_id = _require_uuid(payload, "workflow_attempt_group_id")
    attempt_number = _require_positive_int(payload, "workflow_attempt_number")
    retry_of = _optional_uuid(payload, "retry_of_workflow_run_id")
    attempt_kind = payload.get("workflow_attempt_kind")
    if attempt_kind not in {"initial", "user_retry", "automatic_retry"}:
      raise WorkAuthorizationError("invalid_attempt_kind")
    observability = payload.get("primary_inference_observability")
    if observability not in {
      "hank_metered", "hank_byok_observed", "external_unobserved", "none",
    }:
      raise WorkAuthorizationError("invalid_observability")
    funding_route_id = _require_uuid(payload, "funding_route_id")
    provider = _require_code(payload, "provider")
    billing_mode = payload.get("billing_mode")
    reservation_id = _optional_uuid(payload, "reservation_id")
    if billing_mode not in {"byok", "metered"}:
      raise WorkAuthorizationError("invalid_billing_mode")
    # Money: metered work spends Hank funds, billable only against the
    # risk_module budget reservation the authorization names.
    if (billing_mode == "metered") != (reservation_id is not None):
      raise WorkAuthorizationError("invalid_authority_shape")
    operation = _require_code(payload, "operation")
    capability = _optional_code(payload, "capability_id")
    request_id = _require_opaque_id(payload, "request_id")
    session_id = _require_opaque_id(payload, "session_id")
    issued_at = _require_positive_int(payload, "iat")
    expires_at = _require_positive_int(payload, "exp")
    if expires_at <= issued_at:
      raise WorkAuthorizationError("invalid_authorization_lifetime")
    credential_principal = payload.get("credential_principal")
    if credential_principal not in {"user", "service"}:
      raise WorkAuthorizationError("invalid_credential_principal")
    reserved_spend = payload.get("reserved_spend_microusd")
    # Money: metered work stops at the reservation hold; byok has no hold.
    if billing_mode == "byok":
      if reserved_spend is not None:
        raise WorkAuthorizationError("invalid_authority_shape")
    elif type(reserved_spend) is not int or reserved_spend < 0:
      raise WorkAuthorizationError("invalid_authority_shape")
    return VerifiedWorkAuthorization(
      schema_version=schema_version,
      key_id=key_id,
      token_sha256=token_sha256,
      authorization_id=authorization_id,
      environment=environment,
      execution_context_id=context_id,
      workflow_run_id=workflow_id,
      workflow_attempt_group_id=attempt_group_id,
      workflow_attempt_number=attempt_number,
      retry_of_workflow_run_id=retry_of,
      workflow_attempt_kind=attempt_kind,
      primary_inference_observability=observability,
      funding_route_id=funding_route_id,
      provider=provider,
      billing_mode=billing_mode,
      reservation_id=reservation_id,
      operation=operation,
      capability_id=capability,
      request_id=request_id,
      session_id=session_id,
      issued_at=issued_at,
      expires_at=expires_at,
      reserved_spend_microusd=reserved_spend,
      credential_principal=credential_principal,
    )


def _decode_json_segment(segment: str) -> dict:
  try:
    raw = _decode_base64url_segment(segment)
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
  except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise WorkAuthorizationError("malformed_token") from exc
  if not isinstance(value, dict):
    raise WorkAuthorizationError("malformed_token")
  return value


def _decode_base64url_segment(segment: str) -> bytes:
  if not _BASE64URL_SEGMENT.fullmatch(segment):
    raise WorkAuthorizationError("invalid_compact_token")
  try:
    padding = "=" * (-len(segment) % 4)
    raw = base64.b64decode(segment + padding, altchars=b"-_", validate=True)
  except binascii.Error as exc:
    raise WorkAuthorizationError("invalid_compact_token") from exc
  canonical = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
  if canonical != segment:
    raise WorkAuthorizationError("invalid_compact_token")
  return raw


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
  value = {}
  for key, item in pairs:
    if key in value:
      raise json.JSONDecodeError("duplicate object key", key, 0)
    value[key] = item
  return value


def _require_positive_int(payload: dict, field: str) -> int:
  value = payload.get(field)
  if type(value) is not int or value <= 0:
    raise WorkAuthorizationError(f"invalid_{field}")
  return value


def _require_uuid(payload: dict, field: str) -> UUID:
  value = payload.get(field)
  if not isinstance(value, str):
    raise WorkAuthorizationError(f"invalid_{field}")
  try:
    parsed = UUID(value)
  except ValueError as exc:
    raise WorkAuthorizationError(f"invalid_{field}") from exc
  if str(parsed) != value:
    raise WorkAuthorizationError(f"invalid_{field}")
  return parsed


def _optional_uuid(payload: dict, field: str) -> UUID | None:
  return None if payload.get(field) is None else _require_uuid(payload, field)


def _require_code(payload: dict, field: str) -> str:
  value = payload.get(field)
  if not isinstance(value, str) or not _CODE.fullmatch(value):
    raise WorkAuthorizationError(f"invalid_{field}")
  return value


def _optional_code(payload: dict, field: str) -> str | None:
  return None if payload.get(field) is None else _require_code(payload, field)


def _require_opaque_id(payload: dict, field: str) -> str:
  value = payload.get(field)
  if not isinstance(value, str) or not _OPAQUE_ID.fullmatch(value):
    raise WorkAuthorizationError(f"invalid_{field}")
  return value


__all__ = [
  "VerifiedWorkAuthorization",
  "WORK_AUTHORIZATION_ALGORITHM",
  "WORK_AUTHORIZATION_AUDIENCE",
  "WORK_AUTHORIZATION_ISSUER",
  "WORK_AUTHORIZATION_REQUIRED_FIELDS",
  "WORK_AUTHORIZATION_V3_REQUIRED_FIELDS",
  "WorkAuthorizationError",
  "WorkAuthorizationTrustSnapshot",
  "WorkAuthorizationVerifier",
]
