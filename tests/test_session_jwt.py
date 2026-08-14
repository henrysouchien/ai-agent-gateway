# ruff: noqa: E402

import sys
from pathlib import Path

import jwt
import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.session import AuthManager, JWT_ALGORITHM, SessionStore
from agent_gateway.session_capabilities import TEAM_WORKSPACE_WRITE_CAPABILITY

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
    capabilities=frozenset({TEAM_WORKSPACE_WRITE_CAPABILITY}),
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
  assert claims["capabilities"] == [TEAM_WORKSPACE_WRITE_CAPABILITY]
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


@pytest.mark.parametrize(
  "claim_name",
  ("risk_user_id", "role", "capabilities", "channel", "is_public", "schema_version"),
)
def test_session_jwt_rejects_missing_required_claim(claim_name: str) -> None:
  store = SessionStore(ttl=3600)
  auth = AuthManager(secret=_JWT_SECRET, valid_keys={"gateway-key"}, session_store=store)
  session = store.create_session(api_key_hash="hash", user_id="alice")
  claims = jwt.decode(
    auth.issue_token(session),
    _JWT_SECRET,
    algorithms=[JWT_ALGORITHM],
  )
  claims.pop(claim_name)
  token = jwt.encode(claims, _JWT_SECRET, algorithm=JWT_ALGORITHM)

  with pytest.raises(HTTPException) as exc_info:
    auth.verify_token(token)

  assert exc_info.value.status_code == 401
  assert exc_info.value.detail == "Invalid session payload"


def test_session_jwt_rejects_capability_claim_mismatch() -> None:
  store = SessionStore(ttl=3600)
  auth = AuthManager(secret=_JWT_SECRET, valid_keys={"gateway-key"}, session_store=store)
  session = store.create_session(api_key_hash="hash", user_id="alice")
  claims = jwt.decode(auth.issue_token(session), _JWT_SECRET, algorithms=[JWT_ALGORITHM])
  claims["capabilities"] = [TEAM_WORKSPACE_WRITE_CAPABILITY]
  token = jwt.encode(claims, _JWT_SECRET, algorithm=JWT_ALGORITHM)

  with pytest.raises(HTTPException) as exc_info:
    auth.verify_token(token)

  assert exc_info.value.status_code == 401
  assert exc_info.value.detail == "Session capabilities mismatch"
