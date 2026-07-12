from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import sqlite3
from uuid import uuid4

import pytest

from agent_gateway import work_authorization_consumption as consumption_module
from agent_gateway.commercial_work_authorization import VerifiedWorkAuthorization
from agent_gateway.work_authorization_consumption import (
  WorkAuthorizationAlreadyAttached,
  WorkAuthorizationConsumptionConflict,
  WorkAuthorizationConsumptionError,
  WorkAuthorizationConsumptionStore,
)


NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)


def _verified(**changes) -> VerifiedWorkAuthorization:
  token_marker = changes.pop("token_marker", uuid4().hex)
  facts = {
    "schema_version": 1,
    "key_id": "work-signing-v1",
    "token_sha256": "sha256:" + hashlib.sha256(token_marker.encode()).hexdigest(),
    "authorization_id": uuid4(),
    "environment": "prod",
    "execution_context_id": uuid4(),
    "workflow_run_id": uuid4(),
    "workflow_attempt_group_id": uuid4(),
    "workflow_attempt_number": 1,
    "retry_of_workflow_run_id": None,
    "workflow_attempt_kind": "initial",
    "primary_inference_observability": "hank_metered",
    "funding_route_id": uuid4(),
    "provider": "anthropic",
    "billing_mode": "metered",
    "reservation_id": uuid4(),
    "operation": "messages.create",
    "capability_id": "portfolio.review",
    "request_id": "request-1",
    "session_id": "session-1",
    "issued_at": int(NOW.timestamp()),
    "expires_at": int((NOW + timedelta(minutes=2)).timestamp()),
  }
  facts.update(changes)
  if facts["workflow_attempt_kind"] == "initial":
    facts["workflow_attempt_group_id"] = facts["workflow_run_id"]
  return VerifiedWorkAuthorization(**facts)


def _store(path, *, now=NOW + timedelta(seconds=1)):
  return WorkAuthorizationConsumptionStore(path, clock=lambda: now)


def test_attach_is_durable_token_free_and_append_only(tmp_path) -> None:
  path = tmp_path / "commercial" / "usage.sqlite3"
  store = _store(path)
  authority = _verified(token_marker="raw-token-must-not-persist")

  record = store.attach_once(authority)

  assert record.authorization_id == authority.authorization_id
  assert record.schema_version == 1
  assert record.token_sha256 == authority.token_sha256
  assert record.content_sha256.startswith("sha256:")
  assert not hasattr(record, "token")
  assert store.get(authority.authorization_id) == record
  assert store.health()["ok"] is True
  assert path.stat().st_mode & 0o777 == 0o600
  for suffix in ("-wal", "-shm"):
    sidecar = type(path)(str(path) + suffix)
    if sidecar.exists():
      assert sidecar.stat().st_mode & 0o777 == 0o600
  assert b"raw-token-must-not-persist" not in path.read_bytes()
  with sqlite3.connect(path) as connection:
    columns = {
      row[1]
      for row in connection.execute(
        "PRAGMA table_info(commercial_work_authorization_consumptions)"
      )
    }
    assert "token" not in columns
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
      connection.execute(
        "UPDATE commercial_work_authorization_consumptions "
        "SET request_id = 'tampered'"
      )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
      connection.execute("DELETE FROM commercial_work_authorization_consumptions")


def test_exact_replay_is_already_attached_never_a_second_allow(tmp_path) -> None:
  store = _store(tmp_path / "usage.sqlite3")
  authority = _verified()
  first = store.attach_once(authority)

  with pytest.raises(WorkAuthorizationAlreadyAttached) as raised:
    store.attach_once(authority)

  assert raised.value.record == first
  assert store.health()["consumption_count"] == 1


def test_reused_authorization_workflow_or_digest_conflicts(tmp_path) -> None:
  store = _store(tmp_path / "usage.sqlite3")
  authority = _verified()
  store.attach_once(authority)

  conflicts = (
    replace(authority, request_id="request-2"),
    _verified(workflow_run_id=authority.workflow_run_id),
    _verified(token_sha256=authority.token_sha256),
  )
  for conflict in conflicts:
    with pytest.raises(WorkAuthorizationConsumptionConflict):
      store.attach_once(conflict)
  assert store.health()["consumption_count"] == 1


def test_concurrent_exact_attach_has_one_winner_and_one_replay(tmp_path) -> None:
  store = _store(tmp_path / "usage.sqlite3")
  authority = _verified()

  def attach():
    try:
      store.attach_once(authority)
      return "attached"
    except WorkAuthorizationAlreadyAttached:
      return "already_attached"

  with ThreadPoolExecutor(max_workers=2) as executor:
    outcomes = sorted(executor.map(lambda _: attach(), range(2)))
  assert outcomes == ["already_attached", "attached"]
  assert store.health()["consumption_count"] == 1


def test_attach_rechecks_expiry_after_verification(tmp_path) -> None:
  authority = _verified()
  store = _store(tmp_path / "expired.sqlite3", now=NOW + timedelta(minutes=2))
  with pytest.raises(WorkAuthorizationConsumptionError, match="expired"):
    store.attach_once(authority)
  store = _store(tmp_path / "not-yet.sqlite3", now=NOW - timedelta(seconds=1))
  with pytest.raises(WorkAuthorizationConsumptionError, match="expired"):
    store.attach_once(authority)
  assert store.health()["consumption_count"] == 0

  with pytest.raises(TypeError):
    store.attach_once(authority, attached_at=NOW)


def test_fsync_crossing_expiry_returns_no_allow_and_consumes_authority(tmp_path) -> None:
  authority = _verified(expires_at=int((NOW + timedelta(seconds=2)).timestamp()))
  observed_times = iter((NOW + timedelta(seconds=1), NOW + timedelta(seconds=2)))
  store = WorkAuthorizationConsumptionStore(
    tmp_path / "usage.sqlite3",
    clock=lambda: next(observed_times),
  )

  with pytest.raises(WorkAuthorizationConsumptionError, match="expired during"):
    store.attach_once(authority)

  assert store.get(authority.authorization_id) is not None
  assert store.health()["consumption_count"] == 1


def test_schema_drift_fails_closed_without_replacing_table(tmp_path) -> None:
  path = tmp_path / "usage.sqlite3"
  with sqlite3.connect(path) as connection:
    connection.execute(
      "CREATE TABLE commercial_work_authorization_consumptions "
      "(authorization_id TEXT PRIMARY KEY)"
    )
  with pytest.raises(WorkAuthorizationConsumptionError, match="schema"):
    WorkAuthorizationConsumptionStore(path)
  with sqlite3.connect(path) as connection:
    columns = connection.execute(
      "PRAGMA table_info(commercial_work_authorization_consumptions)"
    ).fetchall()
  assert [row[1] for row in columns] == ["authorization_id"]


def test_same_columns_with_weakened_constraints_fail_closed(tmp_path) -> None:
  path = tmp_path / "usage.sqlite3"
  weakened = consumption_module._TABLE_SQL.replace(
    "CHECK ((billing_mode = 'metered') = (reservation_id IS NOT NULL))",
    "CHECK (1 = 1)",
  )
  with sqlite3.connect(path) as connection:
    connection.execute(weakened)
  with pytest.raises(WorkAuthorizationConsumptionError, match="constraints"):
    WorkAuthorizationConsumptionStore(path)


def test_same_columns_with_case_changed_digest_constraint_fail_closed(
  tmp_path,
) -> None:
  path = tmp_path / "usage.sqlite3"
  changed = consumption_module._TABLE_SQL.replace(
    "*[^0-9a-f]*",
    "*[^0-9A-F]*",
    1,
  )
  with sqlite3.connect(path) as connection:
    connection.execute(changed)
  with pytest.raises(WorkAuthorizationConsumptionError, match="constraints"):
    WorkAuthorizationConsumptionStore(path)


@pytest.mark.parametrize("busy_timeout", (0, -1))
def test_store_configuration_is_fail_closed(tmp_path, busy_timeout) -> None:
  with pytest.raises(ValueError, match="busy timeout"):
    WorkAuthorizationConsumptionStore(
      tmp_path / "usage.sqlite3", busy_timeout_ms=busy_timeout
    )
  with pytest.raises(ValueError, match="synchronous"):
    WorkAuthorizationConsumptionStore(
      tmp_path / "usage.sqlite3", synchronous="NORMAL"
    )


def test_store_fails_closed_when_permissions_cannot_be_secured(
  tmp_path, monkeypatch
) -> None:
  def denied(*_args, **_kwargs):
    raise PermissionError("denied")

  monkeypatch.setattr(consumption_module.os, "chmod", denied)
  with pytest.raises(WorkAuthorizationConsumptionError, match="permissions"):
    WorkAuthorizationConsumptionStore(tmp_path / "usage.sqlite3")
