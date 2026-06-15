import sys
from pathlib import Path

import jwt

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.session import AuthManager, JWT_ALGORITHM, SessionStore

_JWT_SECRET = "jwt-secret-with-at-least-32-bytes"


def test_session_channel_fields_and_jwt_round_trip_new_claims() -> None:
  store = SessionStore(ttl=3600)
  auth = AuthManager(secret=_JWT_SECRET, valid_keys={"gateway-key"}, session_store=store)
  session = store.create_session(
    api_key_hash="hash",
    user_id="alice",
    user_email="alice@example.com",
    risk_user_id=101,
    role="invite",
  )

  assert session.channel is None
  assert session.is_public is False
  assert session.kind == "chat"

  session.channel = "public"
  session.is_public = True
  token = auth.issue_token(session)
  verified_session, claims = auth.verify_token_with_payload(token)

  assert verified_session is session
  assert claims["session_id"] == session.session_id
  assert claims["api_key_hash"] == "hash"
  assert claims["created_at"] == session.created_at
  assert claims["expires_at"] == session.expires_at
  assert claims["user_id"] == "alice"
  assert claims["user_email"] == "alice@example.com"
  assert claims["risk_user_id"] == 101
  assert claims["role"] == "invite"
  assert claims["channel"] == "public"
  assert claims["is_public"] is True
  assert session.channel == "public"
  assert session.is_public is True


def test_auth_manager_rejects_short_hs256_secret() -> None:
  store = SessionStore(ttl=3600)

  try:
    AuthManager(secret="too-short", valid_keys={"gateway-key"}, session_store=store)
  except ValueError as exc:
    assert "at least 32 bytes" in str(exc)
  else:
    raise AssertionError("AuthManager accepted a short JWT signing secret")


def test_session_store_ttl_override_and_kind_are_store_only() -> None:
  store = SessionStore(ttl=3600)
  auth = AuthManager(secret=_JWT_SECRET, valid_keys={"gateway-key"}, session_store=store)
  session = store.create_session(
    api_key_hash="hash",
    user_id="alice",
    kind="control",
    ttl_seconds=900,
  )

  token = auth.issue_token(session)
  claims = jwt.decode(token, _JWT_SECRET, algorithms=[JWT_ALGORITHM])

  assert session.kind == "control"
  assert session.expires_at == session.created_at + 900
  assert "kind" not in claims


def test_old_jwt_without_channel_claims_still_validates_with_defaults() -> None:
  store = SessionStore(ttl=3600)
  auth = AuthManager(secret=_JWT_SECRET, valid_keys={"gateway-key"}, session_store=store)
  session = store.create_session(api_key_hash="hash", user_id="alice")
  session.channel = "excel"
  session.is_public = True
  old_payload = {
    "session_id": session.session_id,
    "api_key_hash": session.api_key_hash,
    "created_at": session.created_at,
    "expires_at": session.expires_at,
    "user_id": session.user_id,
    "user_email": session.user_email,
  }
  token = jwt.encode(old_payload, _JWT_SECRET, algorithm=JWT_ALGORITHM)

  verified_session, claims = auth.verify_token_with_payload(token)

  assert verified_session is session
  assert claims["channel"] is None
  assert claims["is_public"] is False
  assert claims["risk_user_id"] == 0
  assert claims["role"] == "owner"
  assert session.channel is None
  assert session.is_public is False
