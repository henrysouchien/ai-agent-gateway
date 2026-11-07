# ruff: noqa: E402

import asyncio
import sys
from pathlib import Path

import jwt
import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.session import (
  AGENT_TOOL_CALLBACK_PURPOSE,
  AUTONOMOUS_RUNTIME_SESSION_PURPOSE,
  AuthManager,
  JWT_ALGORITHM,
  SessionStore,
  caller_kind_for_channel,
)
from agent_gateway.session_capabilities import TEAM_WORKSPACE_WRITE_CAPABILITY
from agent_gateway.ttl_policy import SESSION_TTL

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


def test_tool_callback_bearer_is_the_sessions_identity_marked_agent_once_per_turn() -> None:
  """An agent tool's callback into a product route presents a gateway-signed bearer
  with purpose=agent_tool_callback: the session's own identity, role and channel,
  minted once per turn (product-labelling §3, design agent-tool-callback-caller-kind)."""
  store = SessionStore(ttl=3600)
  auth = AuthManager(secret=_JWT_SECRET, valid_keys={"gateway-key"}, session_store=store)
  session = store.create_session(
    api_key_hash="hash", user_id="alice", user_email="alice@example.com",
    risk_user_id=101, role="owner",
  )
  session.channel = "excel"
  session.caller_kind = caller_kind_for_channel("excel")
  assert session.caller_kind == "human"
  own = auth.issue_token(session)
  assert callable(session.tool_callback_token_issuer)

  delegated = session.tool_callback_token_issuer("turn-1")
  assert delegated != own
  assert session.tool_callback_token_issuer("turn-1") == delegated  # one mint per turn
  assert session.tool_callback_token_issuer("turn-2") != delegated

  verified, kind = auth.verify_token_caller_kind(delegated)
  assert verified is session and kind == "agent"
  _own_session, own_kind = auth.verify_token_caller_kind(own)
  assert own_kind == "human"
  claims = jwt.decode(delegated, _JWT_SECRET, algorithms=[JWT_ALGORITHM])
  assert claims["purpose"] == AGENT_TOOL_CALLBACK_PURPOSE
  assert claims["turn_id"] == "turn-1"
  assert (claims["user_id"], claims["risk_user_id"], claims["role"], claims["channel"]) == (
    "alice", 101, "owner", "excel",
  )
  # Verifying the delegated bearer never replaces the session's own token.
  assert session.session_token == own



@pytest.mark.parametrize(
  ("caller_kind", "purpose", "expected"),
  [
    ("human", None, "human"),
    ("human", "normalizer", "human"),  # a turn's context purpose never changes who the caller is
    ("agent", None, "agent"),
    ("agent", AUTONOMOUS_RUNTIME_SESSION_PURPOSE, "agent"),
    (None, None, None),
    (None, "normalizer", None),
  ],
)
def test_caller_kind_is_positive_on_both_sides(caller_kind, purpose, expected) -> None:
  """C1 condition 4: human only for a session its door minted as a human client's;
  an agent client and a runtime child are agents; a session no door classified is
  unresolved (the reader refuses), never attributed by absence. `purpose` is the
  turn's context purpose, rewritten per /api/chat turn (DEFECT-DESK b2c72ce2), and
  plays no part."""
  store = SessionStore(ttl=3600)
  auth = AuthManager(secret=_JWT_SECRET, valid_keys={"gateway-key"}, session_store=store)
  session = store.create_session(api_key_hash="hash", user_id="alice", risk_user_id=101, role="owner")
  session.caller_kind = caller_kind
  session.purpose = purpose
  _session, kind = auth.verify_token_caller_kind(auth.issue_token(session))
  assert kind == expected
  # The delegated bearer is an agent's whatever the session's own kind.
  _session, delegated_kind = auth.verify_token_caller_kind(
    auth.issue_tool_callback_token(session, turn_id="turn-1")
  )
  assert delegated_kind == "agent"


@pytest.mark.parametrize(
  ("channel", "caller_kind"),
  [
    ("excel", "human"), ("web", "human"), ("cli", "human"),
    ("tui", "human"), ("telegram", "human"), ("discord", "human"),
    ("mcp", "agent"), ("public", None), (None, None),
  ],
)
def test_chat_init_caller_kind_follows_the_client_channel(channel, caller_kind) -> None:
  assert caller_kind_for_channel(channel) == caller_kind


def test_auth_manager_rejects_short_hs256_secret() -> None:
  store = SessionStore(ttl=3600)

  try:
    AuthManager(secret="too-short", valid_keys={"gateway-key"}, session_store=store)
  except ValueError as exc:
    assert "at least 32 bytes" in str(exc)
  else:
    raise AssertionError("AuthManager accepted a short JWT signing secret")


def test_auth_manager_with_configured_keys_refuses_an_unknown_key() -> None:
  auth = AuthManager(secret=_JWT_SECRET, valid_keys={"gateway-key"}, session_store=SessionStore(ttl=3600))

  auth.validate_api_key("gateway-key")
  with pytest.raises(HTTPException) as exc_info:
    auth.validate_api_key("any")
  assert exc_info.value.status_code == 401


def test_session_store_ttl_override_and_kind_are_store_only() -> None:
  def ttl_resolver(_subject: str, _ttl_name: str) -> int | None:
    raise AssertionError("explicit TTL must bypass entitlement resolution")

  store = SessionStore(ttl=3600, ttl_resolver=ttl_resolver)
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


def test_session_store_resolves_ttl_per_normalized_user_with_default_fallback() -> None:
  calls: list[tuple[str, str]] = []

  def ttl_resolver(subject: str, ttl_name: str) -> int | None:
    calls.append((subject, ttl_name))
    return {"alice": 120, "bob": 480}.get(subject)

  store = SessionStore(ttl=3600, ttl_resolver=ttl_resolver)
  sessions = [
    store.create_session(api_key_hash="hash", user_id=user_id)
    for user_id in ("alice", "bob", "carol")
  ]

  assert [item.expires_at - item.created_at for item in sessions] == [120, 480, 3600]
  assert calls == [("alice", SESSION_TTL), ("bob", SESSION_TTL), ("carol", SESSION_TTL)]


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


def test_expired_session_runs_immediate_cleanup_once_and_waits_for_blockers() -> None:
  async def scenario() -> None:
    store = SessionStore(ttl=3600)
    session = store.create_session(api_key_hash="hash", user_id="alice")
    immediate: list[str] = []
    final: list[str] = []
    finalized = asyncio.Event()
    active = [1]
    store.add_on_expiry(lambda item: immediate.append(item.session_id))

    def record_final(item) -> None:
      final.append(item.session_id)
      finalized.set()

    store.add_on_final_expiry(record_final)
    store.add_expiry_blocker(lambda item: active[0])

    await store.expire_session_async(session.session_id)
    assert session._expired is True
    assert store.get_session(session.session_id) is None
    assert session.session_id in store.sessions
    assert immediate == [session.session_id]
    assert final == []

    await store.expire_session_async(session.session_id)
    assert immediate == [session.session_id]
    assert final == []

    active[0] = 0
    await asyncio.wait_for(finalized.wait(), timeout=1)
    assert session.session_id not in store.sessions
    assert immediate == [session.session_id]
    assert final == [session.session_id]

  asyncio.run(scenario())


def test_expiry_remains_visible_until_async_final_hook_completes() -> None:
  async def scenario() -> None:
    store = SessionStore(ttl=3600)
    session = store.create_session(api_key_hash="hash", user_id="alice")
    final_started = asyncio.Event()
    release_final = asyncio.Event()

    async def final_hook(_session) -> None:
      final_started.set()
      await release_final.wait()

    store.add_on_final_expiry(final_hook)
    expiry = asyncio.create_task(store.expire_session_async(session.session_id))
    await final_started.wait()

    assert session.session_id in store.sessions
    assert store.get_session(session.session_id) is None

    release_final.set()
    await expiry
    assert session.session_id not in store.sessions

  asyncio.run(scenario())


def test_final_expiry_hook_failure_retains_and_retries_from_failed_hook() -> None:
  async def scenario() -> None:
    store = SessionStore(ttl=3600)
    session = store.create_session(api_key_hash="hash", user_id="alice")
    calls: list[str] = []
    failing_attempts = 0
    finalized = asyncio.Event()

    def first(_session) -> None:
      calls.append("first")

    def fail_once(_session) -> None:
      nonlocal failing_attempts
      failing_attempts += 1
      calls.append(f"retry:{failing_attempts}")
      if failing_attempts == 1:
        raise OSError("teardown unavailable")

    def last(_session) -> None:
      calls.append("last")
      finalized.set()

    store.add_on_final_expiry(first)
    store.add_on_final_expiry(fail_once)
    store.add_on_final_expiry(last)

    await store.expire_session_async(session.session_id)

    assert session.session_id in store.sessions
    assert store.get_session(session.session_id) is None
    assert calls == ["first", "retry:1"]

    await asyncio.wait_for(finalized.wait(), timeout=1)
    assert calls == ["first", "retry:1", "retry:2", "last"]
    assert session.session_id not in store.sessions

  asyncio.run(scenario())


def test_sync_expiry_defers_async_hooks_without_exposing_session() -> None:
  store = SessionStore(ttl=3600)
  session = store.create_session(api_key_hash="hash", user_id="alice")
  immediate: list[str] = []
  final: list[str] = []
  store.add_on_expiry(lambda item: immediate.append(item.session_id))
  store.add_on_final_expiry(lambda item: final.append(item.session_id))

  store.expire_session(session.session_id)

  assert store.get_session(session.session_id) is None
  assert session.session_id in store.sessions
  assert immediate == []
  assert final == []
  asyncio.run(store.expire_session_async(session.session_id))
  assert session.session_id not in store.sessions
  assert immediate == [session.session_id]
  assert final == [session.session_id]


@pytest.mark.parametrize("blocker_value", (True, -1, "0", None))
def test_expiry_blocker_malformed_or_unknown_retains_session(
  blocker_value: object,
) -> None:
  async def scenario() -> None:
    store = SessionStore(ttl=3600)
    session = store.create_session(api_key_hash="hash", user_id="alice")
    store.add_expiry_blocker(lambda _session: blocker_value)  # type: ignore[return-value]

    await store.expire_session_async(session.session_id)

    assert session._expired is True
    assert session.session_id in store.sessions
    assert store.get_session(session.session_id) is None

  asyncio.run(scenario())


def test_session_token_stays_valid_past_expires_at(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  now = [1_000]
  monkeypatch.setattr("agent_gateway.session.time.time", lambda: now[0])
  store = SessionStore(ttl=120)
  auth = AuthManager(secret=_JWT_SECRET, valid_keys={"gateway-key"}, session_store=store)
  session = store.create_session(api_key_hash="hash", user_id="alice")
  token = auth.issue_token(session)

  now[0] = session.expires_at + 3600

  assert auth.verify_token(token) is session
  assert store.visible_sessions_snapshot() == (session,)

  store.expire_session(session.session_id)
  with pytest.raises(HTTPException) as exc_info:
    auth.verify_token(token)
  assert exc_info.value.status_code == 401
