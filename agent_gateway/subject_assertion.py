"""Offline verification for risk_module web-session subject assertions."""

from __future__ import annotations

import base64
import binascii
from collections import OrderedDict
import json
import threading
import time
from typing import Annotated, Any, Literal, Mapping
from uuid import UUID

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import HTTPException
import jwt
from pydantic import (
  BaseModel,
  ConfigDict,
  Field,
  StrictInt,
  StrictStr,
  ValidationError,
  ValidationInfo,
  field_validator,
)


from hank_contracts.subject_assertion import (
  SUBJECT_ASSERTION_ALGORITHM,
  SUBJECT_ASSERTION_ISSUER,
  SUBJECT_ASSERTION_AUDIENCE,
  SUBJECT_ASSERTION_PUBLIC_KEYS_ENV,
  SUBJECT_ASSERTION_MAX_LIFETIME_SECONDS,
  SUBJECT_ASSERTION_CLOCK_SKEW_SECONDS,
  SUBJECT_ASSERTION_REPLAY_CACHE_SIZE,
  _KEY_ID,
  _SUBJECT,
  _REQUIRED_CLAIMS,
  _normalize_email,
  SubjectAssertionError,
  VerifiedGatewaySubject,
)


_CLAIM_ERROR_CODES = {
  "schema_version": "invalid_schema_version",
  "sub": "invalid_subject",
  "channel": "invalid_channel",
  "iat": "invalid_lifetime",
  "exp": "invalid_lifetime",
}


class _SubjectAssertionClaims(BaseModel):
  """Exact claim set of a verified assertion; every claim's type and bounds."""

  model_config = ConfigDict(extra="forbid", strict=True)

  schema_version: Annotated[StrictInt, Field(ge=1, le=1)]
  sub: StrictStr
  channel: Literal["web"]
  iat: StrictInt
  exp: StrictInt
  # Checked by PyJWT (iss, aud), ``_normalize_email`` and ``_canonical_uuid``.
  iss: Any
  aud: Any
  email: Any
  request_id: Any
  jti: Any

  @field_validator("sub")
  @classmethod
  def _subject_shape(cls, value: str) -> str:
    if not _SUBJECT.fullmatch(value):
      raise ValueError("subject must be a canonical positive integer")
    return value

  @field_validator("exp")
  @classmethod
  def _lifetime_bounds(cls, value: int, info: ValidationInfo) -> int:
    issued_at = info.data.get("iat")
    if issued_at is not None and (
      value <= issued_at or value - issued_at > SUBJECT_ASSERTION_MAX_LIFETIME_SECONDS
    ):
      raise ValueError("assertion lifetime out of bounds")
    return value


def _claims_error_code(exc: ValidationError) -> str:
  errors = exc.errors(include_input=False, include_url=False)
  if any(err["type"] in ("missing", "extra_forbidden") or not err["loc"] for err in errors):
    return "invalid_claims"
  for field in _CLAIM_ERROR_CODES:
    if any(err["loc"][0] == field for err in errors):
      return _CLAIM_ERROR_CODES[field]
  return "invalid_claims"


def _canonical_uuid(value: object, *, code: str) -> str:
  if not isinstance(value, str):
    raise SubjectAssertionError(code)
  try:
    parsed = UUID(value)
  except (ValueError, AttributeError) as exc:
    raise SubjectAssertionError(code) from exc
  canonical = str(parsed)
  if canonical != value:
    raise SubjectAssertionError(code)
  return canonical


def _decode_public_key(value: object) -> Ed25519PublicKey:
  if not isinstance(value, str) or not value or value != value.strip():
    raise SubjectAssertionError("invalid_public_key")
  try:
    raw = base64.b64decode(
      value + ("=" * (-len(value) % 4)),
      altchars=b"-_",
      validate=True,
    )
  except (TypeError, ValueError, binascii.Error) as exc:
    raise SubjectAssertionError("invalid_public_key") from exc
  canonical = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
  if len(raw) != 32 or canonical != value:
    raise SubjectAssertionError("invalid_public_key")
  return Ed25519PublicKey.from_public_bytes(raw)


def _unique_key_ids(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for key_id, encoded in pairs:
    if key_id in result:
      raise ValueError("duplicate public key id")
    result[key_id] = encoded
  return result


def load_subject_assertion_public_keys(
  value: str,
) -> dict[str, Ed25519PublicKey]:
  try:
    parsed = json.loads(value, object_pairs_hook=_unique_key_ids)
  except (TypeError, ValueError) as exc:
    raise SubjectAssertionError("invalid_public_keys_config") from exc
  if not isinstance(parsed, dict) or not parsed:
    raise SubjectAssertionError("invalid_public_keys_config")
  keys: dict[str, Ed25519PublicKey] = {}
  for key_id, encoded in parsed.items():
    if not isinstance(key_id, str) or not _KEY_ID.fullmatch(key_id):
      raise SubjectAssertionError("invalid_key_id")
    keys[key_id] = _decode_public_key(encoded)
  return keys


class GatewaySubjectAssertionVerifier:
  """Verify, bind, and consume one short-lived web-session assertion."""

  def __init__(
    self,
    public_keys_by_id: Mapping[str, Ed25519PublicKey],
    *,
    replay_cache_size: int = SUBJECT_ASSERTION_REPLAY_CACHE_SIZE,
  ) -> None:
    if not public_keys_by_id:
      raise ValueError("subject assertion verifier requires public keys")
    if replay_cache_size <= 0:
      raise ValueError("subject assertion replay cache size must be positive")
    self._keys = dict(public_keys_by_id)
    self._replay_cache_size = replay_cache_size
    self._consumed: OrderedDict[tuple[str, str], int] = OrderedDict()
    self._lock = threading.Lock()

  @classmethod
  def from_environment(
    cls,
    env: Mapping[str, str],
    *,
    required: bool,
  ) -> "GatewaySubjectAssertionVerifier | None":
    raw = str(env.get(SUBJECT_ASSERTION_PUBLIC_KEYS_ENV) or "").strip()
    if not raw:
      if required:
        raise SubjectAssertionError("public_keys_not_configured")
      return None
    return cls(load_subject_assertion_public_keys(raw))

  def verify(
    self,
    token: str,
    *,
    payload_user_id: object,
    payload_email: object,
    payload_request_id: object,
    claimed_channel: object,
  ) -> VerifiedGatewaySubject:
    current = int(time.time())
    if not isinstance(token, str) or not token or token != token.strip():
      raise SubjectAssertionError("assertion_required")
    try:
      header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
      raise SubjectAssertionError("invalid_token") from exc
    if set(header) != {"alg", "kid", "typ"}:
      raise SubjectAssertionError("invalid_header")
    key_id = header.get("kid")
    if (
      header.get("alg") != SUBJECT_ASSERTION_ALGORITHM
      or header.get("typ") != "JWT"
      or not isinstance(key_id, str)
      or key_id not in self._keys
    ):
      raise SubjectAssertionError("invalid_header")
    try:
      claims = jwt.decode(
        token,
        self._keys[key_id],
        algorithms=[SUBJECT_ASSERTION_ALGORITHM],
        issuer=SUBJECT_ASSERTION_ISSUER,
        audience=SUBJECT_ASSERTION_AUDIENCE,
        leeway=SUBJECT_ASSERTION_CLOCK_SKEW_SECONDS,
        # ``email`` is structurally required but explicitly nullable; PyJWT's
        # require option treats a null value as missing, so the exact-key check
        # below owns that one claim. PyJWT's time-claim validators coerce with
        # ``int(...)`` and raise ``TypeError`` on a non-scalar claim, outside
        # its own exception hierarchy.
        options={"require": sorted(_REQUIRED_CLAIMS - {"email"})},
      )
    except (jwt.PyJWTError, TypeError, OverflowError) as exc:
      raise SubjectAssertionError("verification_failed") from exc
    try:
      typed = _SubjectAssertionClaims.model_validate(claims)
    except ValidationError as exc:
      raise SubjectAssertionError(_claims_error_code(exc)) from exc
    subject = typed.sub
    expires_at = typed.exp
    email = _normalize_email(claims.get("email"))
    request_id = _canonical_uuid(claims.get("request_id"), code="invalid_request_id")
    jti = _canonical_uuid(claims.get("jti"), code="invalid_jti")

    supplied_user_id = str(payload_user_id or "").strip()
    if supplied_user_id and supplied_user_id != subject:
      raise SubjectAssertionError("subject_mismatch")
    if payload_email is not None and _normalize_email(payload_email) != email:
      raise SubjectAssertionError("email_mismatch")
    supplied_request_id = str(payload_request_id or "").strip()
    if supplied_request_id != request_id:
      raise SubjectAssertionError("request_id_mismatch")
    supplied_channel = str(claimed_channel or "").strip().lower()
    if supplied_channel and supplied_channel != "web":
      raise SubjectAssertionError("channel_mismatch")

    self._consume_once(key_id=key_id, jti=jti, expires_at=expires_at, now=current)
    return VerifiedGatewaySubject(
      user_id=subject,
      risk_user_id=int(subject),
      email=email,
      channel="web",
      request_id=request_id,
    )

  def _consume_once(self, *, key_id: str, jti: str, expires_at: int, now: int) -> None:
    replay_key = (key_id, jti)
    with self._lock:
      expired = [key for key, expiry in self._consumed.items() if expiry <= now]
      for key in expired:
        self._consumed.pop(key, None)
      if replay_key in self._consumed:
        raise SubjectAssertionError("replayed")
      if len(self._consumed) >= self._replay_cache_size:
        # Never evict an unexpired jti: doing so would make a replay valid again.
        # Capacity pressure therefore fails closed until an entry expires.
        raise SubjectAssertionError("replay_cache_saturated")
      self._consumed[replay_key] = expires_at


class SubjectVerificationUnavailable(HTTPException):
  """The web subject verifier cannot answer: a typed 503 with the repair.

  The credentials resolver raises this where it needs a verifier the process
  does not have (``not_configured``) or whose replay cache is full
  (``replay_cache_saturated``); the gateway's credential-error mappers project
  ``error_body`` so the caller sees the cause and the next call, not a generic
  authentication failure.
  """

  _DETAIL = {
    "not_configured": "Web subject assertion verification is not configured",
    "replay_cache_saturated": "Web subject assertion verification is temporarily unavailable",
  }
  _NEXT_CALL = {
    "not_configured": (
      f"set {SUBJECT_ASSERTION_PUBLIC_KEYS_ENV} to a JSON object of distinct key ids to "
      "Ed25519 public keys (the gateway startup log names the invalid_public_keys_config "
      "cause when the value is invalid), restart the gateway, then retry POST /api/chat/init"
    ),
    "replay_cache_saturated": (
      "retry POST /api/chat/init with a freshly signed subject assertion once earlier "
      "assertions have expired"
    ),
  }

  def __init__(self, reason: Literal["not_configured", "replay_cache_saturated"]) -> None:
    super().__init__(status_code=503, detail=self._DETAIL[reason])
    self.reason = reason

  def error_body(self, *, user_id: str | None = None) -> dict[str, str]:
    body = {
      "error": "credentials_unavailable",
      "message": str(self.detail),
      "reason": f"web_subject_verification_{self.reason}",
      "next_call": self._NEXT_CALL[self.reason],
    }
    if user_id is not None:
      body["user_id"] = user_id
    return body


__all__ = [
  "GatewaySubjectAssertionVerifier",
  "SubjectVerificationUnavailable",
  "SUBJECT_ASSERTION_ALGORITHM",
  "SUBJECT_ASSERTION_AUDIENCE",
  "SUBJECT_ASSERTION_ISSUER",
  "SUBJECT_ASSERTION_MAX_LIFETIME_SECONDS",
  "SUBJECT_ASSERTION_PUBLIC_KEYS_ENV",
  "SubjectAssertionError",
  "VerifiedGatewaySubject",
  "load_subject_assertion_public_keys",
]
