from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import sqlite3
from uuid import UUID, uuid4

import pytest

from agent_gateway import work_authorization_consumption as consumption_module
from hank_contracts.commercial_work import (
  VerifiedWorkAuthorization,
  WorkAuthorizationError,
)
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
    store.attach_once(authority, attached_at=NOW)  # pyright: ignore[reportCallIssue]  # negative: forged attachment timestamp rejection


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


def test_previous_v1_store_is_rebuilt_and_old_row_reads_back(tmp_path) -> None:
  path = tmp_path / "usage.sqlite3"
  v1 = _verified()
  previous_columns = consumption_module._V1_COLUMNS
  values = consumption_module._values(
    v1,
    consumption_module._content_sha256(v1),
    "2026-07-11T12:00:01.000000Z",
  )[: len(previous_columns)]
  with sqlite3.connect(path) as connection:
    connection.execute(consumption_module._PREVIOUS_TABLE_SQL)
    for trigger in consumption_module._TRIGGERS:
      connection.execute(trigger)
    connection.execute(
      "INSERT INTO commercial_work_authorization_consumptions "
      f"({', '.join(previous_columns)}) "
      f"VALUES ({', '.join('?' for _ in previous_columns)})",
      values,
    )

  store = _store(path)

  migrated = store.get(v1.authorization_id)
  assert migrated is not None
  assert migrated.schema_version == 1
  assert migrated.model_key is None
  assert migrated.credential_principal is None
  assert migrated.reserved_spend_microusd is None
  assert migrated.request_id == v1.request_id
  with pytest.raises(WorkAuthorizationAlreadyAttached):
    store.attach_once(v1)
  v3 = _verified(
    schema_version=3,
    credential_principal="user",
    reserved_spend_microusd=250_000,
  )
  record = store.attach_once(v3)
  assert record.schema_version == 3
  assert record.model_key is None
  assert (record.credential_principal, record.reserved_spend_microusd) == (
    "user",
    250_000,
  )


def test_previous_v2_store_is_rebuilt_and_old_row_reads_back(tmp_path) -> None:
  path = tmp_path / "usage.sqlite3"
  v2 = _verified(schema_version=2)
  columns = consumption_module._V2_COLUMNS
  values = list(consumption_module._values(
    v2,
    consumption_module._content_sha256(v2),
    "2026-07-11T12:00:01.000000Z",
  )[: len(columns)])
  model_key_index = columns.index("model_key")
  principal_index = columns.index("credential_principal")
  values[model_key_index] = "anthropic.claude-sonnet-5-5"
  values[principal_index] = "service"
  with sqlite3.connect(path) as connection:
    connection.execute(consumption_module._V2_TABLE_SQL)
    for trigger in consumption_module._TRIGGERS:
      connection.execute(trigger)
    connection.execute(
      "INSERT INTO commercial_work_authorization_consumptions "
      f"({', '.join(columns)}) "
      f"VALUES ({', '.join('?' for _ in columns)})",
      values,
    )

  store = _store(path)
  migrated = store.get(v2.authorization_id)
  assert migrated is not None
  assert migrated.schema_version == 2
  assert migrated.model_key == "anthropic.claude-sonnet-5-5"
  assert migrated.credential_principal == "service"
  assert migrated.reserved_spend_microusd is None
  assert migrated.provider == v2.provider
  assert migrated.request_id == v2.request_id


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
      tmp_path / "usage.sqlite3", synchronous="NORMAL"  # pyright: ignore[reportArgumentType]  # negative: unsupported SQLite mode rejection
    )


def test_store_fails_closed_when_permissions_cannot_be_secured(
  tmp_path, monkeypatch
) -> None:
  def denied(*_args, **_kwargs):
    raise PermissionError("denied")

  monkeypatch.setattr(consumption_module.os, "chmod", denied)
  with pytest.raises(WorkAuthorizationConsumptionError, match="permissions"):
    WorkAuthorizationConsumptionStore(tmp_path / "usage.sqlite3")

@pytest.mark.parametrize(
  "observability",
  ("hank_metered", "hank_byok_observed", "external_unobserved", "none"),
)
def test_attach_admits_all_verifier_observabilities(tmp_path, observability) -> None:
  path = tmp_path / "usage.sqlite3"
  store = _store(path)
  auth = _verified(
    schema_version=3,
    primary_inference_observability=observability,
    credential_principal="user",
    billing_mode="byok",
    reservation_id=None,
    reserved_spend_microusd=None,
  )
  record = store.attach_once(auth)
  assert record.primary_inference_observability == observability


def test_store_migrates_v3_metered_observability_table(tmp_path) -> None:
  path = tmp_path / "usage.sqlite3"
  old_auth = _verified(
    schema_version=3,
    primary_inference_observability="hank_byok_observed",
    credential_principal="user",
    billing_mode="byok",
    reservation_id=None,
    reserved_spend_microusd=None,
  )
  content_sha = consumption_module._content_sha256(old_auth)
  values = consumption_module._values(old_auth, content_sha, "2026-07-11T12:00:01Z")
  with sqlite3.connect(path) as connection:
    connection.execute(consumption_module._V3_METERED_OBSERVABILITY_TABLE_SQL)
    cols = ", ".join(consumption_module._COLUMNS)
    q = ", ".join("?" for _ in consumption_module._COLUMNS)
    connection.execute(
      f"INSERT INTO commercial_work_authorization_consumptions ({cols}) VALUES ({q})",
      values,
    )
  store = _store(path)
  # Pre-existing row survived migration intact
  migrated = store.get(old_auth.authorization_id)
  assert migrated is not None
  assert migrated.primary_inference_observability == "hank_byok_observed"

  # New row with external_unobserved attaches cleanly
  auth = _verified(
    schema_version=3,
    primary_inference_observability="external_unobserved",
    credential_principal="user",
    billing_mode="byok",
    reservation_id=None,
    reserved_spend_microusd=None,
  )
  record = store.attach_once(auth)
  assert record.primary_inference_observability == "external_unobserved"


def test_verify_for_attach_then_attach_admits_external_unobserved(tmp_path) -> None:
  # Loaded by path: under full-suite collection the name ``tests`` is the
  # repository's root test package, not this one.
  import importlib.util
  from pathlib import Path

  spec = importlib.util.spec_from_file_location(
    "_gateway_commercial_work_authorization_helpers",
    Path(__file__).with_name("test_commercial_work_authorization.py"),
  )
  assert spec is not None and spec.loader is not None
  helpers = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(helpers)
  _claim = helpers._claim
  _keypair = helpers._keypair
  _payload = helpers._payload
  _token = helpers._token
  _trust = helpers._trust
  from agent_gateway.commercial_work_authorization import WorkAuthorizationVerifier

  private, public = _keypair()
  now_ts = int((NOW + timedelta(seconds=1)).timestamp())
  verifier = WorkAuthorizationVerifier(
    _trust(public, retired_at_by_key_id={"work-signing-v1": now_ts + 60}),
    clock=lambda: now_ts,
  )
  payload = _payload(
    schema_version=3,
    primary_inference_observability="external_unobserved",
    credential_principal="user",
    billing_mode="byok",
    reservation_id=None,
    reserved_spend_microusd=None,
    iat=now_ts - 10,
    exp=now_ts + 300,
  )
  token = _token(private, payload)
  verified = verifier.verify_for_attach(
    token,
    execution_claim=_claim(
      context_id=UUID(payload["execution_claim_jti"]),
      issued_at=now_ts - 10,
      expires_at=now_ts + 300,
      authorized_work_start_deadline=now_ts + 300,
    ),
    request_id="request-1",
    session_id="session-1",
    operation="messages.create",
    billing_mode="byok",
    capability_id="portfolio.review",
    credential_principal="user",
  )
  assert verified.primary_inference_observability == "external_unobserved"

  path = tmp_path / "usage.sqlite3"
  store = _store(path)
  record = store.attach_once(verified)
  assert record.primary_inference_observability == "external_unobserved"
