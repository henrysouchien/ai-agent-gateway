from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterator
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import pytest

from agent_gateway.capability_execution import CapabilityExecutionResolver, MaterializedCredential
from agent_gateway.providers.credential_pool import (
  PoolAccountCandidate,
  ProviderCredentialPool,
  REFRESH_FAILURE_BLOCK_SECONDS,
  get_provider_credential_pool,
  load_pool_accounts,
  rotate_pool_candidate,
  rotate_provider_credential,
  select_pool_candidate,
  remove_pool_account,
  upsert_pool_account,
)
from agent_gateway.providers.google_antigravity import GoogleAntigravityProvider
from agent_gateway.providers.google_antigravity_oauth import (
  resolve_antigravity_pool_candidates,
  save_antigravity_token_record,
)
from agent_workflow_contracts.models import SelectionProvenance
from model_authority.binding import CredentialHandle
from model_authority.selection import (
  CapabilityDefault, FallbackEntry, TierModelsDefault, TierSelectionPolicy, capture_selection_snapshot,
)
from gateway_test_support.capability_execution_test_support import (
  stub_bound_capability_execution, stub_capability_execution_resolver,
)


@pytest.mark.parametrize(
  ("filename", "material"),
  [("oauth.json", {"access_token": "legacy-token", "refresh_token": "legacy-refresh"}),
   ("credentials.json", {"api_key": "legacy-key"})],
)
def test_legacy_record_reads_as_account_and_migrates_without_loss(
  tmp_path: Path, filename: str, material: dict[str, str],
) -> None:
  store = tmp_path / filename
  legacy = {"email": "first@example.com", "source": "subscription", **material}
  store.write_text(json.dumps(legacy), encoding="utf-8")

  accounts = load_pool_accounts(store)
  assert accounts == [{**legacy, "identity": "first@example.com"}]

  upsert_pool_account(store, {"email": "second@example.com", **material})
  envelope = json.loads(store.read_text(encoding="utf-8"))
  assert envelope["version"] == 1
  assert len(envelope["accounts"]) == 2
  assert envelope["accounts"][0] == accounts[0]
  assert stat.S_IMODE(store.stat().st_mode) == 0o600

  upsert_pool_account(store, {"email": "first@example.com", **material, "source": "renewed"})
  updated = load_pool_accounts(store)
  assert len(updated) == 2
  assert [seat["identity"] for seat in updated] == ["first@example.com", "second@example.com"]
  assert updated[0]["source"] == "renewed"


def test_gemini_enrollment_preserves_seats_and_updates_matching_identity(tmp_path: Path) -> None:
  store = tmp_path / "google_antigravity" / "oauth.json"
  first = {"email": "first@example.com", "access_token": "token-first", "project_id": "project-first"}
  second = {"email": "second@example.com", "access_token": "token-second", "project_id": "project-second"}
  save_antigravity_token_record(store, first)
  save_antigravity_token_record(store, second)
  accounts = load_pool_accounts(store)
  assert [(seat["email"], seat["project_id"]) for seat in accounts] == [
    ("first@example.com", "project-first"), ("second@example.com", "project-second"),
  ]
  save_antigravity_token_record(store, {**first, "access_token": "token-first-renewed"})
  accounts = load_pool_accounts(store)
  assert len(accounts) == 2
  assert accounts[0]["access_token"] == "token-first-renewed"
  assert accounts[1]["access_token"] == "token-second"
  assert accounts[1]["project_id"] == "project-second"


def test_pool_mutations_do_not_self_deadlock_in_subprocess(tmp_path: Path) -> None:
  store = tmp_path / "oauth.json"
  package_root = Path(__file__).resolve().parents[1]
  env = dict(os.environ)
  env["PYTHONPATH"] = os.pathsep.join([str(package_root), env.get("PYTHONPATH", "")])
  script = """
import json
from pathlib import Path
import sys
from agent_gateway.providers.credential_pool import upsert_pool_account, load_pool_accounts, remove_pool_account
path = Path(sys.argv[1])
upsert_pool_account(path, {"identity": "first", "access_token": "one"})
upsert_pool_account(path, {"identity": "second", "access_token": "two"})
assert len(load_pool_accounts(path)) == 2
assert remove_pool_account(path, "first")
assert [seat["identity"] for seat in load_pool_accounts(path)] == ["second"]
assert remove_pool_account(path, "second")
assert load_pool_accounts(path) == []
print(json.dumps({"accounts": load_pool_accounts(path), "exists": path.exists()}))
"""
  result = subprocess.run(
    [sys.executable, "-c", script, str(store)], env=env,
    capture_output=True, text=True, timeout=5, check=True,
  )
  assert json.loads(result.stdout) == {"accounts": [], "exists": False}


def _candidates() -> list[PoolAccountCandidate]:
  return [
    PoolAccountCandidate(
      provider="google_antigravity", key=identity,
      record={"identity": identity, "access_token": token, "project_id": project},
    )
    for identity, token, project in [
      ("first@example.com", "token-first", "project-first"),
      ("second@example.com", "token-second", "project-second"),
    ]
  ]


def test_pool_parks_accounts_until_cooldown_and_never_shortens_block() -> None:
  pool = ProviderCredentialPool()
  pool.block("first", until=150)
  pool.block("first", until=120)
  assert pool.blocked_until("first") == 150
  assert pool.blocked_until("unknown") == 0
  assert pool.first_unblocked(["first", "second"], now=100) == "second"
  pool.block("second", until=160)
  assert pool.first_unblocked(["first", "second"], now=100) == ""
  assert pool.first_unblocked(["first", "second"], now=150) == "first"
  pool.clear()
  assert pool.first_unblocked(["first", "second"], now=100) == "first"


def test_select_and_rotate_keep_token_and_project_of_unblocked_seat() -> None:
  first, second = _candidates()
  pool = ProviderCredentialPool()
  assert select_pool_candidate([first, second], pool=pool, now=100) is first
  selected = rotate_pool_candidate(
    [first, second], current_key=first.token, until=200, pool=pool, now=100,
  )
  assert selected is not None
  assert selected is second
  assert selected.record["project_id"] == "project-second"
  assert pool.blocked_until(first.key) == 200
  assert select_pool_candidate([first, second], pool=pool, now=100) is second
  assert rotate_pool_candidate(
    [first, second], current_key=second.account_identity, until=200, pool=pool, now=100,
  ) is None
  assert select_pool_candidate([first, second], pool=pool, now=200) is first


def test_selection_refresh_failure_parks_first_and_serves_second() -> None:
  first, second = _candidates()
  pool = ProviderCredentialPool()
  refreshed: list[str] = []

  def refresh(candidate: PoolAccountCandidate) -> PoolAccountCandidate:
    refreshed.append(candidate.key)
    if candidate is first:
      raise RuntimeError("refresh revoked")
    return candidate

  selected = select_pool_candidate([first, second], pool=pool, refresh_fn=refresh, now=100)
  assert selected is not None
  assert selected is second
  assert refreshed == [first.key, second.key]
  assert pool.blocked_until(first.key) == 100 + REFRESH_FAILURE_BLOCK_SECONDS
  assert selected.record["project_id"] == "project-second"


def test_rotation_refresh_failure_parks_sibling_and_serves_next() -> None:
  first, second = _candidates()
  third = PoolAccountCandidate(
    provider="google_antigravity", key="third@example.com",
    record={"identity": "third@example.com", "access_token": "token-third", "project_id": "project-third"},
  )
  pool = ProviderCredentialPool()

  def refresh(candidate: PoolAccountCandidate) -> PoolAccountCandidate:
    if candidate is second:
      raise RuntimeError("refresh revoked")
    return candidate

  selected = rotate_pool_candidate(
    [first, second, third], current_key=first.key, until=200,
    pool=pool, refresh_fn=refresh, now=100,
  )
  assert selected is third
  assert pool.blocked_until(first.key) == 200
  assert pool.blocked_until(second.key) == 100 + REFRESH_FAILURE_BLOCK_SECONDS


@pytest.fixture
def gemini_pool(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Iterator[tuple[ProviderCredentialPool, Path]]:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  for name in (
    "GOOGLE_ANTIGRAVITY_AUTH_STORE_PATH", "GOOGLE_ANTIGRAVITY_AUTH_TOKEN",
    "GOOGLE_ANTIGRAVITY_PROJECT_ID", "GEMINI_AUTH_TOKEN", "GOOGLE_AUTH_TOKEN",
  ):
    monkeypatch.delenv(name, raising=False)
  pool = get_provider_credential_pool("google_antigravity")
  pool.clear()
  store = tmp_path / "google_antigravity" / "oauth.json"
  for candidate in _candidates():
    save_antigravity_token_record(store, candidate.record)
  try:
    yield pool, store
  finally:
    pool.clear()


def _gemini_upstream_model() -> str:
  from model_authority.current import INITIAL_MODEL_REGISTRY

  entry = next(
    entry
    for entry in INITIAL_MODEL_REGISTRY.models.values()
    if entry.provider == "google_antigravity"
  )
  return entry.upstream_model


def _gemini_config(store: Path) -> dict[str, Any]:
  return {
    "provider": "google_antigravity", "auth_mode": "oauth", "auth_token": "token-first",
    "api_key": "", "project_id": "project-first", "account_identity": "first@example.com",
    "auth_store_path": str(store), "max_tokens": 4096,
  }


@pytest.mark.parametrize("failure_kind", ["rate_limit", "quota"])
def test_gemini_usage_limit_rotates_with_second_seats_project(
  gemini_pool: tuple[ProviderCredentialPool, Path], failure_kind: str,
) -> None:
  pool, store = gemini_pool
  reset_at = time.time() + 600
  rotated = rotate_provider_credential(
    "google_antigravity", _gemini_config(store), failure_kind=failure_kind, reset_at=reset_at,
  )
  assert rotated is not None
  assert rotated["auth_token"] == "token-second"
  assert rotated["project_id"] == "project-second"
  assert rotated["account_identity"] == "second@example.com"
  assert pool.blocked_until("first@example.com") == reset_at


def test_rebind_unparked_account_moves_to_second_gemini_seat(gemini_pool) -> None:
  pool, store = gemini_pool
  execution = stub_bound_capability_execution(
    provider=GoogleAntigravityProvider(), model=_gemini_upstream_model(), effort="high",
    auth_config=_gemini_config(store),
  )
  pool.block("first@example.com", until=time.time() + 600)
  rebound = CapabilityExecutionResolver._rebind_unparked_account(execution)
  assert rebound is not None
  assert rebound.auth_config["auth_token"] == "token-second"
  assert rebound.auth_config["project_id"] == "project-second"
  assert rebound.auth_config["account_identity"] == "second@example.com"
  assert rebound.bind.model_key == execution.bind.model_key
  assert rebound.bind.credential_principal == execution.bind.credential_principal


def test_rebind_unparked_account_does_not_serve_fully_parked_gemini_pool(gemini_pool) -> None:
  pool, store = gemini_pool
  execution = stub_bound_capability_execution(
    provider=GoogleAntigravityProvider(), model=_gemini_upstream_model(), effort="high",
    auth_config=_gemini_config(store),
  )
  candidates, _ = resolve_antigravity_pool_candidates(execution.auth_config)
  assert len(candidates) == 2
  for candidate in candidates:
    pool.block(candidate.key, until=time.time() + 600)
  assert CapabilityExecutionResolver._rebind_unparked_account(execution) is None


def test_model_fallback_records_fully_parked_gemini_provider_and_serves_next(gemini_pool) -> None:
  pool, store = gemini_pool
  gemini_model = _gemini_upstream_model()
  resolver = stub_capability_execution_resolver(
    default_provider="openai", default_model="gpt-primary", default_effort="none",
    extra_models=(("google_antigravity", gemini_model), ("openai", "gpt-backup")),
  )
  models = resolver.registry.models
  primary_key = next(key for key, entry in models.items() if entry.upstream_model == "gpt-primary")
  gemini_key = next(key for key, entry in models.items() if entry.upstream_model == gemini_model)
  backup_key = next(key for key, entry in models.items() if entry.upstream_model == "gpt-backup")
  old_materializer = resolver.credential_materializer

  def materialize(handle: CredentialHandle) -> MaterializedCredential:
    if handle.provider == "google_antigravity":
      return MaterializedCredential(handle=handle, auth_config=_gemini_config(store))
    return old_materializer(handle)

  slot = resolver.selection_policy.capabilities["session.driver"]
  policy = replace(
    resolver.selection_policy,
    capabilities={
      **resolver.selection_policy.capabilities,
      "session.driver": replace(slot, default=CapabilityDefault(kind="tier", tier="deep")),
    },
    tiers={
      "deep": TierSelectionPolicy(
        default=TierModelsDefault((primary_key,), "none"),
        allowed_model_keys=frozenset({primary_key, gemini_key, backup_key}),
        fallback=(FallbackEntry(model=gemini_key), FallbackEntry(model=backup_key)),
      ),
    },
  )
  resolver = replace(resolver, selection_policy=policy, credential_materializer=materialize)
  primary = resolver.resolve("session.driver")
  for candidate in _candidates():
    pool.block(candidate.key, until=time.time() + 600)

  hop = CapabilityExecutionResolver.rebind_model_fallback(
    primary, capture_selection_snapshot(policy),
    selection_provenance=SelectionProvenance(role=None, role_source=None, effort_source=None, preset=None),
    history_fits=lambda _execution: True,
  )
  assert hop is not None
  assert hop.execution.bind.model_key == backup_key
  assert [(skip.model_key, skip.code) for skip in hop.skipped] == [(gemini_key, "credential_pool_parked")]


def test_two_xai_device_code_logins_preserve_both_seats(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import asyncio
  import httpx
  from agent_gateway.providers.xai_oauth import (
    DEFAULT_XAI_OAUTH_CLIENT_ID,
    DEFAULT_XAI_OAUTH_ISSUER,
    DEFAULT_XAI_OAUTH_SCOPE,
    load_xai_accounts,
    load_xai_token_record,
    login_xai_device_code,
  )

  store = tmp_path / "xai" / "oauth.json"
  monkeypatch.setenv("XAI_AUTH_STORE_PATH", str(store))
  tokens = [
    ("xai-access-seat-1", "xai-refresh-seat-1"),
    ("xai-access-seat-2", "xai-refresh-seat-2"),
  ]

  for access_tok, refresh_tok in tokens:
    def handler(request: httpx.Request, a: str = access_tok, r: str = refresh_tok) -> httpx.Response:
      if request.url.path.endswith("/openid-configuration"):
        return httpx.Response(200, json={
          "device_authorization_endpoint": f"{DEFAULT_XAI_OAUTH_ISSUER}/oauth2/device",
          "token_endpoint": f"{DEFAULT_XAI_OAUTH_ISSUER}/oauth2/token",
        })
      if request.url.path.endswith("/oauth2/device"):
        return httpx.Response(200, json={
          "device_code": "dev-123",
          "user_code": "USER-123",
          "verification_uri": f"{DEFAULT_XAI_OAUTH_ISSUER}/activate",
          "expires_in": 600,
          "interval": 1,
        })
      if request.url.path.endswith("/oauth2/token"):
        return httpx.Response(200, json={
          "access_token": a,
          "refresh_token": r,
          "expires_in": 3600,
          "scope": DEFAULT_XAI_OAUTH_SCOPE,
        })
      return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
      asyncio.run(login_xai_device_code(client=client))
    finally:
      asyncio.run(client.aclose())

  accounts = load_xai_accounts(store)
  assert [acc["access_token"] for acc in accounts] == ["xai-access-seat-1", "xai-access-seat-2"]
  single_view = load_xai_token_record(store)
  assert single_view is not None
  assert "accounts" not in single_view
  assert "identity" not in single_view
  assert single_view["client_id"] == DEFAULT_XAI_OAUTH_CLIENT_ID


def test_xai_enrollment_keeps_quarantined_seat_unchanged(tmp_path: Path) -> None:
  from agent_gateway.providers.xai_oauth import (
    load_xai_token_record,
    upsert_xai_token_record,
  )

  store = tmp_path / "xai" / "oauth.json"
  store.parent.mkdir(parents=True)
  quarantined_a = {
    "access_token": "xai-access-a",
    "refresh_token": "xai-refresh-a",
    "expires_at": 1790000000,
    "scope": "openid profile email",
    "issuer": "https://auth.x.ai",
    "client_id": "xai-client",
    "refresh_pending": True,
  }
  store.write_text(json.dumps(quarantined_a))
  seat_b = {
    "access_token": "xai-access-b",
    "refresh_token": "xai-refresh-b",
    "expires_at": 1790003600,
    "scope": "openid profile email",
    "issuer": "https://auth.x.ai",
    "client_id": "xai-client",
  }
  accounts = upsert_xai_token_record(store, seat_b)
  assert len(accounts) == 2
  retained_a = next(
    acc for acc in accounts if acc.get("refresh_token") == "xai-refresh-a"
  )
  assert retained_a["refresh_pending"] is True
  assert retained_a["access_token"] == "xai-access-a"
  single_view = load_xai_token_record(store)
  assert single_view is not None
  assert "refresh_pending" not in single_view
  assert single_view["access_token"] == "xai-access-b"


def test_selected_xai_account_never_inherits_sibling_quarantine(tmp_path: Path) -> None:
  from agent_gateway.providers.xai_oauth import (
    oauth_record_from_config,
    upsert_xai_token_record,
  )

  store = tmp_path / "xai" / "oauth.json"
  monkeypatch_scope = {"XAI_AUTH_STORE_PATH": str(store)}
  base = {
    "expires_at": 1790003600,
    "scope": "openid profile email",
    "issuer": "https://auth.x.ai",
    "client_id": "xai-client",
  }
  upsert_xai_token_record(store, {
    "identity": "seat-a", "access_token": "xai-access-a",
    "refresh_token": "xai-refresh-a", "refresh_pending": True, **base,
  })
  upsert_xai_token_record(store, {
    "identity": "seat-b", "access_token": "xai-access-b",
    "refresh_token": "xai-refresh-b", **base,
  })
  record_b, _settings = oauth_record_from_config(
    {"account_identity": "seat-b"}, environ=monkeypatch_scope,
  )
  assert record_b is not None
  assert record_b["access_token"] == "xai-access-b"
  assert "refresh_pending" not in record_b
  record_default, _ = oauth_record_from_config({}, environ=monkeypatch_scope)
  assert record_default is not None
  assert record_default["access_token"] == "xai-access-a"
  assert record_default.get("refresh_pending") is True
  assert "accounts" not in record_default


def test_xai_refresh_lock_serializes_with_pool_removal(tmp_path: Path) -> None:
  import asyncio
  import threading
  from agent_gateway.providers.xai_oauth import _store_lock

  store = tmp_path / "xai" / "oauth.json"
  upsert_pool_account(store, {"identity": "seat-a", "access_token": "tok-a"})
  acquired = threading.Event()
  release = threading.Event()
  removed = threading.Event()
  errors: list[BaseException] = []

  def hold_lock() -> None:
    async def _hold() -> None:
      async with _store_lock(store):
        acquired.set()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, release.wait, 2)
    try:
      asyncio.run(_hold())
    except BaseException as exc:
      errors.append(exc)

  def run_removal() -> None:
    try:
      remove_pool_account(store, "seat-a")
      removed.set()
    except BaseException as exc:
      errors.append(exc)

  holder = threading.Thread(target=hold_lock)
  holder.start()
  assert acquired.wait(timeout=2)
  remover = threading.Thread(target=run_removal)
  remover.start()
  remover.join(timeout=0.3)
  assert not removed.is_set()  # removal blocks while refresh holds the lock
  release.set()
  holder.join(timeout=2)
  remover.join(timeout=2)
  assert not errors
  assert removed.is_set()
  assert load_pool_accounts(store) == []


def test_bound_credential_without_pool_is_never_parked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  for name in ("GOOGLE_ANTIGRAVITY_AUTH_STORE_PATH", "GOOGLE_AUTH_STORE_PATH", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AUTH_TOKEN", "GEMINI_AUTH_TOKEN"):
    monkeypatch.delenv(name, raising=False)
  execution = stub_bound_capability_execution(
    provider=GoogleAntigravityProvider(), model=_gemini_upstream_model(), effort="high",
    auth_config={
      "provider": "google_antigravity", "auth_mode": "oauth",
      "auth_token": "request-scoped-token", "api_key": "",
      "project_id": "request-project", "max_tokens": 4096,
    },
  )
  assert CapabilityExecutionResolver._rebind_unparked_account(execution) is execution


def test_explicit_custom_store_rotation_stays_within_its_own_accounts(tmp_path: Path) -> None:
  custom = tmp_path / "google_antigravity" / "custom.json"
  canonical = tmp_path / "google_antigravity" / "oauth.json"
  for seat, path in (("first@example.com", custom), ("second@example.com", custom)):
    upsert_pool_account(path, {
      "identity": seat, "access_token": f"token-{seat.split('@')[0]}",
      "project_id": f"project-{seat.split('@')[0]}",
    })
  upsert_pool_account(canonical, {
    "identity": "third@example.com", "access_token": "token-third", "project_id": "project-third",
  })
  rotated = rotate_provider_credential(
    "google_antigravity",
    {
      "provider": "google_antigravity", "auth_mode": "oauth",
      "auth_token": "token-first", "api_key": "",
      "project_id": "project-first", "account_identity": "first@example.com",
      "auth_store_path": str(custom), "max_tokens": 4096,
    },
    failure_kind="rate_limit", reset_at=time.time() + 600,
  )
  assert rotated is not None
  assert rotated["account_identity"] == "second@example.com"
  assert rotated["project_id"] == "project-second"


def test_whole_store_removal_serializes_with_writers(tmp_path: Path) -> None:
  from agent_gateway.providers.credential_pool import store_write_lock
  import threading

  store = tmp_path / "google" / "credentials.json"
  upsert_pool_account(store, {"identity": "seat", "api_key": "key"})
  locked = threading.Event()
  release = threading.Event()
  removed = threading.Event()
  errors: list[BaseException] = []

  def hold() -> None:
    with store_write_lock(store):
      locked.set()
      release.wait(timeout=2)

  def run_removal() -> None:
    try:
      if remove_pool_account(store):
        removed.set()
    except BaseException as exc:
      errors.append(exc)

  holder = threading.Thread(target=hold)
  holder.start()
  assert locked.wait(timeout=2)
  remover = threading.Thread(target=run_removal)
  remover.start()
  remover.join(timeout=0.2)
  assert not removed.is_set()
  assert store.exists()
  release.set()
  holder.join(timeout=2)
  remover.join(timeout=2)
  assert not errors
  assert removed.is_set()
  assert not store.exists()


def test_xai_concurrent_pool_resolution_does_not_deadlock_active_refresh(
  tmp_path: Path,
) -> None:
  import asyncio
  import httpx
  from agent_gateway.providers.xai_oauth import (
    DEFAULT_XAI_OAUTH_CLIENT_ID,
    DEFAULT_XAI_OAUTH_ISSUER,
    DEFAULT_XAI_OAUTH_SCOPE,
     refresh_xai_oauth_token,
    resolve_xai_oauth_settings,
    resolve_xai_pool_candidates,
    upsert_xai_token_record,
  )

  store = tmp_path / "xai" / "oauth.json"
  env = {"XAI_AUTH_STORE_PATH": str(store)}
  expired_record = {
    "identity": "seat-a",
    "access_token": "xai-expired-access",
    "refresh_token": "xai-refresh-1",
    "expires_at": time.time() - 60,
    "scope": DEFAULT_XAI_OAUTH_SCOPE,
    "issuer": DEFAULT_XAI_OAUTH_ISSUER,
    "client_id": DEFAULT_XAI_OAUTH_CLIENT_ID,
    "token_endpoint": f"{DEFAULT_XAI_OAUTH_ISSUER}/oauth2/token",
  }
  upsert_xai_token_record(store, expired_record)
  settings = resolve_xai_oauth_settings(environ=env)
  in_flight = asyncio.Event()

  async def handler(_request: httpx.Request) -> httpx.Response:
    in_flight.set()
    await asyncio.sleep(0.15)
    return httpx.Response(200, json={
      "access_token": "xai-refreshed-access",
      "refresh_token": "xai-refresh-2",
      "expires_in": 3600,
      "scope": DEFAULT_XAI_OAUTH_SCOPE,
    })

  async def run_both() -> tuple[dict[str, Any], int]:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
      refresh_task = asyncio.create_task(
        refresh_xai_oauth_token(expired_record, settings=settings, client=client, force=True)
      )
      await asyncio.wait_for(in_flight.wait(), timeout=1.0)
      candidates, _ = resolve_xai_pool_candidates(environ=env)
      refreshed = await asyncio.wait_for(refresh_task, timeout=1.0)
      return refreshed, len(candidates)

  refreshed, count = asyncio.run(run_both())
  assert refreshed["access_token"] == "xai-refreshed-access"
  assert count == 1
