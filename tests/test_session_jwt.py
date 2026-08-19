# ruff: noqa: E402

import asyncio
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


def test_visible_session_snapshot_matches_auth_expiry_without_mutation(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  now = [1_000]
  monkeypatch.setattr("agent_gateway.session.time.time", lambda: now[0])
  store = SessionStore(ttl=10)
  current = store.create_session(api_key_hash="hash", user_id="current")
  retained_expired = store.create_session(api_key_hash="hash", user_id="retained")
  elapsed = store.create_session(
    api_key_hash="hash",
    user_id="elapsed",
    ttl_seconds=1,
  )
  retained_expired._expired = True
  now[0] = elapsed.expires_at
  clock_reads = 0

  def captured_now() -> int:
    nonlocal clock_reads
    clock_reads += 1
    return now[0]

  monkeypatch.setattr("agent_gateway.session.time.time", captured_now)
  before = dict(store.sessions)

  assert store.visible_sessions_snapshot() == (current,)
  assert clock_reads == 1
  assert store.sessions == before
  assert elapsed._expired is False
  assert retained_expired._expired is True


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


def test_expiry_blocker_exception_retains_session_and_token_stays_rejected(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async def scenario() -> None:
    store = SessionStore(ttl=3600)
    auth = AuthManager(
      secret=_JWT_SECRET,
      valid_keys={"gateway-key"},
      session_store=store,
    )
    session = store.create_session(api_key_hash="hash", user_id="alice")
    token = auth.issue_token(session)
    store.add_expiry_blocker(
      lambda _session: (_ for _ in ()).throw(RuntimeError("unknown"))
    )
    monkeypatch.setattr(
      "agent_gateway.session.time.time",
      lambda: session.expires_at,
    )

    with pytest.raises(HTTPException) as exc_info:
      auth.verify_token(token)
    await asyncio.sleep(0)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Session expired"
    assert session.session_id in store.sessions
    assert store.get_session(session.session_id) is None

  asyncio.run(scenario())
