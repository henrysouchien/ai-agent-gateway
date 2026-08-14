from __future__ import annotations

import base64
import json
import time
from uuid import uuid4

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import jwt
import pytest

from agent_gateway.subject_assertion import (
  GatewaySubjectAssertionVerifier,
  SUBJECT_ASSERTION_AUDIENCE,
  SUBJECT_ASSERTION_ISSUER,
  SubjectAssertionError,
  load_subject_assertion_public_keys,
)


def _authority(
  *, replay_cache_size: int = 10_000
) -> tuple[Ed25519PrivateKey, GatewaySubjectAssertionVerifier]:
  private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
  public_raw = private_key.public_key().public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw,
  )
  public_value = base64.urlsafe_b64encode(public_raw).rstrip(b"=").decode("ascii")
  keys = load_subject_assertion_public_keys(
    json.dumps({"risk-web-v1": public_value})
  )
  return private_key, GatewaySubjectAssertionVerifier(
    keys,
    replay_cache_size=replay_cache_size,
  )


def _token(
  private_key: Ed25519PrivateKey,
  *,
  subject: str = "101",
  email: str | None = "user@example.com",
  request_id: str,
  issued_at: int,
  expires_at: int,
  audience: str = SUBJECT_ASSERTION_AUDIENCE,
) -> str:
  return jwt.encode(
    {
      "schema_version": 1,
      "iss": SUBJECT_ASSERTION_ISSUER,
      "aud": audience,
      "sub": subject,
      "email": email,
      "channel": "web",
      "request_id": request_id,
      "iat": issued_at,
      "exp": expires_at,
      "jti": str(uuid4()),
    },
    private_key,
    algorithm="EdDSA",
    headers={"kid": "risk-web-v1", "typ": "JWT"},
  )


def test_verifier_derives_subject_and_consumes_assertion_once() -> None:
  private_key, verifier = _authority()
  current = int(time.time())
  request_id = str(uuid4())
  token = _token(
    private_key,
    request_id=request_id,
    issued_at=current,
    expires_at=current + 60,
  )

  subject = verifier.verify(
    token,
    payload_user_id="101",
    payload_email=" USER@example.com ",
    payload_request_id=request_id,
    claimed_channel="web",
    now=current,
  )

  assert subject.user_id == "101"
  assert subject.risk_user_id == 101
  assert subject.email == "user@example.com"
  with pytest.raises(SubjectAssertionError, match="replayed"):
    verifier.verify(
      token,
      payload_user_id="101",
      payload_email="user@example.com",
      payload_request_id=request_id,
      claimed_channel="web",
      now=current,
    )


@pytest.mark.parametrize(
  ("override", "code"),
  [
    ({"payload_user_id": "202"}, "subject_mismatch"),
    ({"payload_email": "other@example.com"}, "email_mismatch"),
    ({"payload_request_id": str(uuid4())}, "request_id_mismatch"),
    ({"claimed_channel": "mcp"}, "channel_mismatch"),
  ],
)
def test_verifier_rejects_payload_authority_mismatch(
  override: dict[str, str],
  code: str,
) -> None:
  private_key, verifier = _authority()
  current = int(time.time())
  request_id = str(uuid4())
  token = _token(
    private_key,
    request_id=request_id,
    issued_at=current,
    expires_at=current + 60,
  )
  arguments = {
    "payload_user_id": "101",
    "payload_email": "user@example.com",
    "payload_request_id": request_id,
    "claimed_channel": "web",
    "now": current,
    **override,
  }

  with pytest.raises(SubjectAssertionError, match=code):
    verifier.verify(token, **arguments)


def test_verifier_rejects_wrong_audience_and_expired_assertion() -> None:
  private_key, verifier = _authority()
  current = int(time.time())
  request_id = str(uuid4())

  for token in (
    _token(
      private_key,
      request_id=request_id,
      issued_at=current,
      expires_at=current + 60,
      audience="wrong-service",
    ),
    _token(
      private_key,
      request_id=request_id,
      issued_at=current - 120,
      expires_at=current - 60,
    ),
  ):
    with pytest.raises(SubjectAssertionError, match="verification_failed"):
      verifier.verify(
        token,
        payload_user_id="101",
        payload_email="user@example.com",
        payload_request_id=request_id,
        claimed_channel="web",
        now=current,
      )


def test_replay_cache_fails_closed_instead_of_evicting_live_jti() -> None:
  private_key, verifier = _authority(replay_cache_size=1)
  current = int(time.time())
  request_ids = [str(uuid4()), str(uuid4())]
  tokens = [
    _token(
      private_key,
      request_id=request_id,
      issued_at=current,
      expires_at=current + 60,
    )
    for request_id in request_ids
  ]
  verifier.verify(
    tokens[0],
    payload_user_id="101",
    payload_email="user@example.com",
    payload_request_id=request_ids[0],
    claimed_channel="web",
    now=current,
  )

  with pytest.raises(SubjectAssertionError, match="replay_cache_saturated"):
    verifier.verify(
      tokens[1],
      payload_user_id="101",
      payload_email="user@example.com",
      payload_request_id=request_ids[1],
      claimed_channel="web",
      now=current,
    )
  with pytest.raises(SubjectAssertionError, match="replayed"):
    verifier.verify(
      tokens[0],
      payload_user_id="101",
      payload_email="user@example.com",
      payload_request_id=request_ids[0],
      claimed_channel="web",
      now=current,
    )
