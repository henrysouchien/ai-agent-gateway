"""Offline verification for risk_module web-session subject assertions."""

from __future__ import annotations

import base64
import binascii
from collections import OrderedDict
from dataclasses import dataclass
import json
import re
import threading
import time
from typing import Mapping
from uuid import UUID

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
import jwt


SUBJECT_ASSERTION_ALGORITHM = "EdDSA"
SUBJECT_ASSERTION_ISSUER = "risk-module-web-auth"
SUBJECT_ASSERTION_AUDIENCE = "agent-gateway-session-init"
SUBJECT_ASSERTION_PUBLIC_KEYS_ENV = "GATEWAY_SUBJECT_ASSERTION_ED25519_PUBLIC_KEYS"
SUBJECT_ASSERTION_MAX_LIFETIME_SECONDS = 90
SUBJECT_ASSERTION_CLOCK_SKEW_SECONDS = 5
SUBJECT_ASSERTION_REPLAY_CACHE_SIZE = 10_000
_KEY_ID = re.compile(r"^[a-z][a-z0-9._:-]{0,127}$")
_SUBJECT = re.compile(r"^[1-9][0-9]*$")
_REQUIRED_CLAIMS = frozenset({
  "schema_version",
  "iss",
  "aud",
  "sub",
  "email",
  "channel",
  "request_id",
  "iat",
  "exp",
  "jti",
})


class SubjectAssertionError(ValueError):
  """A subject assertion is missing, invalid, expired, or already consumed."""

  def __init__(self, code: str) -> None:
    self.code = code
    super().__init__(code)


@dataclass(frozen=True, slots=True)
class VerifiedGatewaySubject:
  """Token-free identity derived from verified upstream authority."""

  user_id: str
  risk_user_id: int
  email: str | None
  channel: str
  request_id: str

  @classmethod
  def trusted_internal(
    cls,
    *,
    owner_user_id: str,
    email: str | None,
    channel: str,
    request_id: str,
  ) -> "VerifiedGatewaySubject":
    """Build identity from an already-verified gateway-owned control record."""
    subject = str(owner_user_id or "").strip()
    if not _SUBJECT.fullmatch(subject):
      raise SubjectAssertionError("invalid_trusted_subject")
    normalized_channel = str(channel or "").strip().lower()
    if normalized_channel != "web":
      raise SubjectAssertionError("invalid_trusted_channel")
    normalized_request_id = str(request_id or "").strip()
    if not normalized_request_id:
      raise SubjectAssertionError("invalid_trusted_request_id")
    return cls(
      user_id=subject,
      risk_user_id=int(subject),
      email=_normalize_email(email),
      channel=normalized_channel,
      request_id=normalized_request_id,
    )


def _normalize_email(value: object) -> str | None:
  if value is None:
    return None
  if not isinstance(value, str):
    raise SubjectAssertionError("invalid_email")
  normalized = value.strip().lower()
  if not normalized or len(normalized) > 320:
    raise SubjectAssertionError("invalid_email")
  return normalized


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


def load_subject_assertion_public_keys(
  value: str,
) -> dict[str, Ed25519PublicKey]:
  try:
    parsed = json.loads(value)
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
    now: int | None = None,
  ) -> VerifiedGatewaySubject:
    current = int(time.time()) if now is None else now
    if type(current) is not int:
      raise SubjectAssertionError("invalid_verification_time")
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
        # below owns that one claim.
        options={"require": sorted(_REQUIRED_CLAIMS - {"email"})},
      )
    except jwt.PyJWTError as exc:
      raise SubjectAssertionError("verification_failed") from exc
    if not isinstance(claims, dict) or set(claims) != _REQUIRED_CLAIMS:
      raise SubjectAssertionError("invalid_claims")
    if claims.get("schema_version") != 1 or type(claims.get("schema_version")) is not int:
      raise SubjectAssertionError("invalid_schema_version")
    subject = claims.get("sub")
    if not isinstance(subject, str) or not _SUBJECT.fullmatch(subject):
      raise SubjectAssertionError("invalid_subject")
    if claims.get("channel") != "web":
      raise SubjectAssertionError("invalid_channel")
    issued_at = claims.get("iat")
    expires_at = claims.get("exp")
    if type(issued_at) is not int or type(expires_at) is not int:
      raise SubjectAssertionError("invalid_lifetime")
    if (
      expires_at <= issued_at
      or expires_at - issued_at > SUBJECT_ASSERTION_MAX_LIFETIME_SECONDS
      or issued_at > current + SUBJECT_ASSERTION_CLOCK_SKEW_SECONDS
    ):
      raise SubjectAssertionError("invalid_lifetime")
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


__all__ = [
  "GatewaySubjectAssertionVerifier",
  "SUBJECT_ASSERTION_ALGORITHM",
  "SUBJECT_ASSERTION_AUDIENCE",
  "SUBJECT_ASSERTION_ISSUER",
  "SUBJECT_ASSERTION_MAX_LIFETIME_SECONDS",
  "SUBJECT_ASSERTION_PUBLIC_KEYS_ENV",
  "SubjectAssertionError",
  "VerifiedGatewaySubject",
  "load_subject_assertion_public_keys",
]
