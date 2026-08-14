from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import stat

import pytest
from agent_workflow_contracts import CapabilityBind

from agent_gateway import usage_outbox as usage_outbox_module
from agent_gateway.commercial_contract import canonical_usage_payload_sha256
from agent_gateway.multi_user.billing import SessionUsageSummary
from agent_gateway.usage_outbox import (
  CommercialUsageOutbox,
  CommercialUsageOutboxConflict,
  CommercialUsageOutboxError,
)
from agent_gateway.usage_reconciliation import CommercialUsageReconciliationTracker


NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)


def _mode(path: Path) -> int:
  return stat.S_IMODE(path.stat().st_mode)


def test_outbox_requires_an_owner_private_storage_directory(tmp_path) -> None:
  storage = tmp_path / "broad-storage"
  storage.mkdir(mode=0o755)

  with pytest.raises(CommercialUsageOutboxError, match="owner-private"):
    CommercialUsageOutbox(storage / "usage.sqlite3")


def test_outbox_hardens_sqlite_main_wal_and_shm_files(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "private-storage" / "usage.sqlite3")

  connection = outbox._connect()
  try:
    connection.execute("BEGIN IMMEDIATE")
    paths = (
      outbox.path,
      Path(f"{outbox.path}-wal"),
      Path(f"{outbox.path}-shm"),
    )
    assert all(path.is_file() for path in paths)
    assert {_mode(path) for path in paths} == {0o600}
  finally:
    connection.close()

  assert _mode(outbox.path.parent) == 0o700


def test_reconciliation_schemas_are_declared_as_package_data() -> None:
  pyproject = Path(__file__).parents[1] / "pyproject.toml"
  contents = pyproject.read_text(encoding="utf-8")
  assert '"contracts/usage-reconciliation-v2/*.json"' in contents
  assert '"contracts/usage-reconciliation-v3/*.json"' in contents


def _payload(event_id: str, *, request_id: str = "req_001") -> dict:
  payload = {
    "schema_version": 1,
    "source_product": "hank-agent-gateway",
    "source_event_id": event_id,
    "environment": "prod",
    "occurred_at": "2026-07-11T11:59:00Z",
    "execution_context_id": "33333333-3333-4333-8333-333333333333",
    "request_id": request_id,
    "session_id": "sess_001",
    "parent_turn_id": None,
    "workflow_run_id": "44444444-4444-4444-8444-444444444444",
    "reservation_id": "55555555-5555-4555-8555-555555555555",
    "funding_route_id": "66666666-6666-4666-8666-666666666666",
    "channel": "mcp",
    "provider": "anthropic",
    "operation": "messages.create",
    "model": "claude-sonnet-test",
    "capability_id": "portfolio.review",
    "usage_state": "succeeded",
    "uncached_input_tokens": 100,
    "billable_output_tokens": 20,
    "reasoning_tokens_observed": 5,
    "cache_write_tokens": 0,
    "cache_read_tokens": 10,
    "is_batch": False,
    "provider_units": None,
    "separately_billed_tool_cost_usd": "0",
    "producer_estimated_cost_usd": "0.002",
    "provider_reported_cost_usd": None,
    "cost_observation_kind": "producer_estimate",
    "producer_rate_version": "rates-v1",
    "shadow_rate_version": "shadow-v1",
    "raw_billing_mode": "metered",
  }
  payload["source_payload_sha256"] = canonical_usage_payload_sha256(payload)
  return payload


def _v2_payload(event_id: str) -> dict:
  payload = {
    **_payload(event_id),
    "schema_version": 2,
    "workflow_attempt_group_id": "44444444-4444-4444-8444-444444444444",
    "workflow_attempt_number": 1,
    "retry_of_workflow_run_id": None,
    "workflow_attempt_kind": "initial",
    "work_authorization_id": "77777777-7777-4777-8777-777777777777",
  }
  payload["source_payload_sha256"] = canonical_usage_payload_sha256(payload)
  return payload


def _v3_payload(event_id: str) -> dict:
  bind = CapabilityBind(
    schema_version="1.0",
    capability_id="portfolio.review",
    model_key="anthropic.claude-sonnet-test",
    provider="anthropic",
    upstream_model="claude-sonnet-test",
    adapter="anthropic.sdk.messages",
    protocol_profile="anthropic.messages",
    route="direct",
    effort="none",
    credential_principal="user",
    credential_ref="test-credential",
    run_mode="interactive",
    registry_revision="test-v1",
    policy_revision="test-v1",
    selection_source="explicit_user",
  ).receipt()
  payload = {
    **_payload(event_id),
    "schema_version": 3,
    "capability_bind": bind,
    "provider_reported_model": "claude-sonnet-test-20260801",
    "workflow_attempt_group_id": None,
    "workflow_attempt_number": None,
    "retry_of_workflow_run_id": None,
    "workflow_attempt_kind": None,
    "work_authorization_id": None,
  }
  payload["source_payload_sha256"] = canonical_usage_payload_sha256(payload)
  return payload


def test_outbox_preserves_exact_v3_bind_and_reported_identity_bytes(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  payload = _v3_payload("evt_v3")
  outbox.enqueue_batch([payload], created_at=NOW)

  stored = outbox.get("evt_v3")
  assert stored is not None
  assert stored.payload == payload
  assert stored.payload_json == json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
  )
  assert stored.payload["capability_bind"] == payload["capability_bind"]
  assert stored.payload["provider_reported_model"] == (
    "claude-sonnet-test-20260801"
  )


def test_outbox_migration_pragmas_and_health(tmp_path) -> None:
  path = tmp_path / "commercial" / "usage.sqlite3"
  outbox = CommercialUsageOutbox(path)
  outbox.enqueue_batch([_payload("evt_001")], created_at=NOW - timedelta(seconds=30))

  health = outbox.health(now=NOW)
  assert health["ok"] is True
  assert health["schema_version"] == 3
  assert health["journal_mode"] == "wal"
  assert health["synchronous"] >= 2
  assert health["counts"] == {"pending": 1}
  assert health["backlog_count"] == 1
  assert health["oldest_backlog_age_seconds"] == 30
  assert path.stat().st_mode & 0o777 == 0o600

  with sqlite3.connect(path) as connection:
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
    columns = {
      row[1] for row in connection.execute("PRAGMA table_info(commercial_usage_outbox)")
    }
  assert {"payload_json", "payload_sha256", "state", "sending_lease_token"} <= columns
  CommercialUsageOutbox(path)


def test_atomic_batch_idempotency_and_conflict_rollback(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  first = _payload("evt_001")
  second = _payload("evt_002")
  outbox.enqueue_batch([first, second], created_at=NOW)
  outbox.enqueue_batch([first, second], created_at=NOW + timedelta(seconds=1))
  equivalent = dict(first, producer_estimated_cost_usd=0.002)
  assert equivalent["source_payload_sha256"] == canonical_usage_payload_sha256(equivalent)
  outbox.enqueue_batch([equivalent], created_at=NOW + timedelta(seconds=2))
  assert outbox.health(now=NOW)["counts"] == {"pending": 2}
  assert outbox.get("evt_001").payload == first

  conflicting = _payload("evt_002", request_id="req_conflict")
  with pytest.raises(CommercialUsageOutboxConflict):
    outbox.enqueue_batch([_payload("evt_003"), conflicting], created_at=NOW)
  assert outbox.get("evt_003") is None
  assert outbox.get("evt_002").payload == second


def test_source_partition_uses_normalized_reconciliation_identity(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  first = _payload("evt_001")
  later = _payload("evt_002")
  later["occurred_at"] = "2026-07-11T12:01:00+00:00"
  later["source_payload_sha256"] = canonical_usage_payload_sha256(later)
  outbox.enqueue_batch([first, later], created_at=NOW)

  rows = outbox.source_partition(
    environment="prod",
    source_product="hank-agent-gateway",
    occurred_from=datetime(2026, 7, 11, 11, 58, tzinfo=timezone.utc),
    occurred_until=datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc),
  )
  assert [row.event_id for row in rows] == ["evt_001"]
  assert rows[0].occurred_at == "2026-07-11T11:59:00.000000Z"


def test_payload_bytes_are_immutable_across_fenced_state_transitions(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  payload = _v3_payload("evt_001")
  outbox.enqueue_batch([payload], created_at=NOW)
  original_json = outbox.get("evt_001").payload_json

  leased = outbox.lease_batch(limit=10, lease_for=timedelta(seconds=30), now=NOW)
  assert len(leased) == 1
  row = leased[0]
  assert row.state == "sending"
  assert row.attempt_count == 1
  with sqlite3.connect(outbox.path) as connection:
    with pytest.raises(sqlite3.IntegrityError, match="source evidence is immutable"):
      connection.execute(
        "UPDATE commercial_usage_outbox SET payload_json = '{}' WHERE event_id = 'evt_001'"
      )
  assert outbox.mark_accepted(
    "evt_001", "wrong-token", ingest_status="accepted",
    canonical_event_id="canonical-001", accepted_at=NOW,
  ) is False
  assert outbox.mark_accepted(
    "evt_001", row.sending_lease_token, ingest_status="accepted",
    canonical_event_id="canonical-001",
    accepted_at=NOW + timedelta(seconds=1),
  ) is True
  accepted = outbox.get("evt_001")
  assert accepted.state == "accepted"
  assert accepted.payload_json == original_json
  assert accepted.ingest_status == "accepted"
  assert accepted.canonical_event_id == "canonical-001"
  assert json.loads(original_json) == payload
  with sqlite3.connect(outbox.path) as connection:
    with pytest.raises(sqlite3.IntegrityError, match="terminal evidence is immutable"):
      connection.execute(
        "UPDATE commercial_usage_outbox SET payload_json = '{}' WHERE event_id = 'evt_001'"
      )
    with pytest.raises(sqlite3.IntegrityError, match="terminal evidence is immutable"):
      connection.execute(
        "UPDATE commercial_usage_outbox SET canonical_event_id = 'forged' "
        "WHERE event_id = 'evt_001'"
      )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
      connection.execute(
        "DELETE FROM commercial_usage_outbox WHERE event_id = 'evt_001'"
      )


def test_database_rejects_accepted_state_without_canonical_receipt(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  outbox.enqueue_batch([_v3_payload("evt_001")], created_at=NOW)
  leased = outbox.lease_batch(
    limit=1, lease_for=timedelta(seconds=30), now=NOW
  )[0]
  with sqlite3.connect(outbox.path) as connection:
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
      connection.execute(
        """
        UPDATE commercial_usage_outbox
           SET state = 'accepted', sending_lease_token = NULL,
               sending_lease_expires_at = NULL, accepted_at = ?,
               ingest_status = 'accepted', ingest_decided_at = ?
         WHERE event_id = ? AND sending_lease_token = ?
        """,
        (
          "2026-07-11T12:01:00.000000Z",
          "2026-07-11T12:01:00.000000Z",
          leased.event_id,
          leased.sending_lease_token,
        ),
      )


def test_lease_dead_letters_non_v3_payloads_instead_of_shipping(
  tmp_path, caplog
) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  outbox.enqueue_batch(
    [_payload("evt_v1"), _v2_payload("evt_v2"), _v3_payload("evt_v3")],
    created_at=NOW,
  )

  with caplog.at_level("ERROR", logger="agent_gateway.usage_outbox"):
    leased = outbox.lease_batch(limit=10, lease_for=timedelta(seconds=30), now=NOW)

  # Only the v3 payload ships; older queued payloads are refused loudly.
  assert [row.event_id for row in leased] == ["evt_v3"]
  for event_id, version in (("evt_v1", 1), ("evt_v2", 2)):
    row = outbox.get(event_id)
    assert row.state == "dead"
    assert row.last_error == f"unshippable_payload_schema_version:{version}"
    assert row.sending_lease_token is None
    assert row.ingest_status is None
  logged = [record.getMessage() for record in caplog.records]
  assert any("evt_v1" in message and "dead-lettered" in message for message in logged)
  assert any("evt_v2" in message and "dead-lettered" in message for message in logged)

  # Dead-lettered rows are terminal: they never re-enter the ship path.
  assert outbox.mark_accepted(
    "evt_v3",
    leased[0].sending_lease_token,
    ingest_status="accepted",
    canonical_event_id="canonical-v3",
    accepted_at=NOW,
  ) is True
  assert outbox.lease_batch(
    limit=10, lease_for=timedelta(seconds=30), now=NOW + timedelta(minutes=5)
  ) == []
  # Evidence bytes remain immutable in place.
  assert outbox.get("evt_v1").payload["schema_version"] == 1


def test_retry_dead_and_expired_lease_recovery(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  outbox.enqueue_batch(
    [_v3_payload("evt_001"), _v3_payload("evt_002")], created_at=NOW
  )
  first = outbox.lease_batch(limit=1, lease_for=timedelta(seconds=5), now=NOW)[0]
  assert outbox.mark_retryable(
    first.event_id,
    first.sending_lease_token,
    next_attempt_at=NOW + timedelta(seconds=10),
    error="temporary",
  ) is True
  second = outbox.lease_batch(limit=2, lease_for=timedelta(seconds=5), now=NOW)[0]
  assert second.event_id == "evt_002"
  assert outbox.mark_dead(second.event_id, second.sending_lease_token, error="malformed")

  assert outbox.lease_batch(
    limit=2, lease_for=timedelta(seconds=5), now=NOW + timedelta(seconds=9)
  ) == []
  retried = outbox.lease_batch(
    limit=2, lease_for=timedelta(seconds=5), now=NOW + timedelta(seconds=11)
  )[0]
  assert retried.event_id == "evt_001"
  assert retried.attempt_count == 2

  recovered = outbox.lease_batch(
    limit=2, lease_for=timedelta(seconds=5), now=NOW + timedelta(seconds=17)
  )[0]
  assert recovered.event_id == "evt_001"
  assert recovered.attempt_count == 3
  assert recovered.last_error == "sending lease expired"
  assert outbox.mark_dead("evt_001", retried.sending_lease_token, error="stale fence") is False


def test_fractional_lease_expiry_uses_lexically_stable_utc_timestamps(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  outbox.enqueue_batch([_v3_payload("evt_001")], created_at=NOW)
  leased = outbox.lease_batch(
    limit=1, lease_for=timedelta(microseconds=500_000), now=NOW
  )[0]
  assert outbox.lease_batch(
    limit=1, lease_for=timedelta(seconds=1),
    now=NOW + timedelta(microseconds=250_000),
  ) == []
  recovered = outbox.lease_batch(
    limit=1, lease_for=timedelta(seconds=1),
    now=NOW + timedelta(microseconds=500_000),
  )[0]
  assert recovered.event_id == leased.event_id
  assert recovered.sending_lease_token != leased.sending_lease_token


def test_enqueue_rejects_tampered_digest_before_writing(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  payload = _payload("evt_bad")
  payload["request_id"] = "tampered"
  with pytest.raises(ValueError, match="digest"):
    outbox.enqueue_batch([payload])
  assert outbox.health()["backlog_count"] == 0


def test_migration_rejects_version_claim_with_wrong_schema_shape(tmp_path) -> None:
  path = tmp_path / "usage.sqlite3"
  with sqlite3.connect(path) as connection:
    connection.execute("CREATE TABLE commercial_usage_outbox (event_id TEXT PRIMARY KEY)")
    connection.execute("PRAGMA user_version=1")
  with pytest.raises(CommercialUsageOutboxError, match="table"):
    CommercialUsageOutbox(path)


def test_valid_v1_rows_upgrade_with_explicit_legacy_acceptance_evidence(tmp_path) -> None:
  path = tmp_path / "usage.sqlite3"
  pending = _payload("evt_pending")
  accepted = _payload("evt_accepted")
  with sqlite3.connect(path) as connection:
    connection.execute(usage_outbox_module._V1_OUTBOX_TABLE_SQL)
    for index_sql in usage_outbox_module._V1_OUTBOX_INDEX_SQL.values():
      connection.execute(index_sql)
    connection.executemany(
      """
      INSERT INTO commercial_usage_outbox (
        event_id, payload_sha256, payload_json, state, attempt_count,
        accepted_at, created_at
      ) VALUES (?, ?, ?, ?, 0, ?, ?)
      """,
      [
        (
          pending["source_event_id"], pending["source_payload_sha256"],
          json.dumps(pending, sort_keys=True, separators=(",", ":")),
          "pending", None, "2026-07-11T12:00:00.000000Z",
        ),
        (
          accepted["source_event_id"], accepted["source_payload_sha256"],
          json.dumps(accepted, sort_keys=True, separators=(",", ":")),
          "accepted", "2026-07-11T12:01:00.000000Z",
          "2026-07-11T12:00:00.000000Z",
        ),
      ],
    )
    connection.execute("PRAGMA user_version=1")

  outbox = CommercialUsageOutbox(path)
  assert outbox.get("evt_pending").state == "pending"
  migrated = outbox.get("evt_accepted")
  assert migrated.ingest_status == "legacy_unknown"
  assert migrated.canonical_event_id is None
  assert migrated.environment == "prod"
  assert outbox.health(now=NOW)["schema_version"] == 3


def test_v1_migration_rejects_corrupt_payload_identity_without_upgrading(tmp_path) -> None:
  path = tmp_path / "usage.sqlite3"
  payload = _payload("evt_payload")
  with sqlite3.connect(path) as connection:
    connection.execute(usage_outbox_module._V1_OUTBOX_TABLE_SQL)
    for index_sql in usage_outbox_module._V1_OUTBOX_INDEX_SQL.values():
      connection.execute(index_sql)
    connection.execute(
      """
      INSERT INTO commercial_usage_outbox (
        event_id, payload_sha256, payload_json, state, attempt_count, created_at
      ) VALUES (?, ?, ?, 'pending', 0, ?)
      """,
      (
        "evt_table", "sha256:" + "a" * 64,
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        "2026-07-11T12:00:00.000000Z",
      ),
    )
    connection.execute("PRAGMA user_version=1")

  with pytest.raises(CommercialUsageOutboxError, match="identity or digest"):
    CommercialUsageOutbox(path)
  with sqlite3.connect(path) as connection:
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    assert connection.execute(
      "SELECT event_id FROM commercial_usage_outbox"
    ).fetchone()[0] == "evt_table"


def test_migration_rejects_same_columns_without_v1_constraints(tmp_path) -> None:
  path = tmp_path / "usage.sqlite3"
  with sqlite3.connect(path) as connection:
    connection.execute("""
      CREATE TABLE commercial_usage_outbox (
        event_id TEXT, payload_sha256 TEXT, payload_json TEXT, state TEXT,
        attempt_count INTEGER, next_attempt_at TEXT, sending_lease_token TEXT,
        sending_lease_expires_at TEXT, accepted_at TEXT, last_error TEXT,
        created_at TEXT
      )
    """)
    connection.execute("PRAGMA user_version=1")
  with pytest.raises(CommercialUsageOutboxError, match="table"):
    CommercialUsageOutbox(path)


def test_health_reports_actual_schema_drift(tmp_path) -> None:
  path = tmp_path / "usage.sqlite3"
  outbox = CommercialUsageOutbox(path)
  with sqlite3.connect(path) as connection:
    connection.execute("PRAGMA user_version=0")

  health = outbox.health(now=NOW)
  assert health["ok"] is False
  assert health["schema_version"] == 0
  assert health["schema_valid"] is False


def test_v3_claim_rejects_missing_reconciliation_evidence_table(tmp_path) -> None:
  path = tmp_path / "usage.sqlite3"
  CommercialUsageOutbox(path)
  with sqlite3.connect(path) as connection:
    connection.execute("DROP TABLE commercial_usage_reconciliation_reports")
  with pytest.raises(CommercialUsageOutboxError, match="version 3"):
    CommercialUsageOutbox(path)


def test_v3_claim_rejects_same_name_noop_guard_trigger(tmp_path) -> None:
  path = tmp_path / "usage.sqlite3"
  CommercialUsageOutbox(path)
  with sqlite3.connect(path) as connection:
    connection.execute("DROP TRIGGER trg_commercial_usage_outbox_immutable_source")
    connection.execute("""
      CREATE TRIGGER trg_commercial_usage_outbox_immutable_source
      BEFORE UPDATE ON commercial_usage_outbox BEGIN SELECT 1; END
    """)
  with pytest.raises(CommercialUsageOutboxError, match="version 3"):
    CommercialUsageOutbox(path)


def test_reconciliation_reports_are_append_only_replay_safe_revisions(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  tracker = CommercialUsageReconciliationTracker(
    request_id="req_001", session_id="sess_001"
  )
  tracker.record_batch([_payload("evt_001")])
  summary = SessionUsageSummary(
    user_id="user-001", session_id="sess_001", request_id="req_001",
    input_tokens=100, output_tokens=20, cache_read_tokens=10,
    cache_creation_tokens=0, cost=0.002, turns=1, channel="mcp",
    started_at=NOW.timestamp() - 60, ended_at=NOW.timestamp(),
    usage_event_count=1, usage_event_ids=("evt_001",),
    capability_bind=_v3_payload("bind")["capability_bind"],
  )
  report = tracker.compare(summary)
  first, replayed = outbox.record_reconciliation_report(report, recorded_at=NOW)
  replay, replayed_again = outbox.record_reconciliation_report(
    report, recorded_at=NOW + timedelta(seconds=1)
  )
  tracker.mark_late("evt_001")
  changed_report = tracker.compare(summary)
  second, second_replayed = outbox.record_reconciliation_report(
    changed_report, recorded_at=NOW + timedelta(seconds=2)
  )
  stored_replay, stored_replayed = outbox.record_reconciliation_report(
    first.payload, recorded_at=NOW + timedelta(seconds=3)
  )
  assert replayed is False and replayed_again is True and second_replayed is False
  assert stored_replayed is True and stored_replay.report_id == first.report_id
  assert replay.report_id == first.report_id
  assert (first.revision, first.supersedes_report_sha256) == (1, None)
  assert second.revision == 2
  assert second.supersedes_report_sha256 == first.report_sha256
  assert second.payload["evidence_revision"] == 2
  assert second.payload["supersedes_report_sha256"] == first.report_sha256
  assert outbox.current_reconciliation_report(
    environment="prod", source_product="hank-agent-gateway",
    request_id="req_001", session_id="sess_001"
  ) == second
  assert outbox.reconciliation_reports(
    environment="prod",
    source_product="hank-agent-gateway",
    recorded_from=NOW - timedelta(seconds=1),
    recorded_until=NOW + timedelta(seconds=3),
  ) == (first, second)
  with sqlite3.connect(outbox.path) as connection:
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
      connection.execute(
        "UPDATE commercial_usage_reconciliation_reports SET status = 'match' "
        "WHERE report_id = ?",
        (second.report_id,),
      )
    with pytest.raises(sqlite3.IntegrityError, match="lineage must be linear"):
      connection.execute(
        """
        INSERT INTO commercial_usage_reconciliation_reports (
          report_id, environment, source_product, request_id, session_id,
          revision, supersedes_report_sha256, content_sha256, report_sha256,
          payload_json, status, recorded_at
        ) SELECT ?, environment, source_product, request_id, session_id,
                 4, ?, ?, ?, payload_json, status, recorded_at
            FROM commercial_usage_reconciliation_reports WHERE report_id = ?
        """,
        (
          "forged-report", "sha256:" + "a" * 64, "sha256:" + "b" * 64,
          "sha256:" + "c" * 64, second.report_id,
        ),
      )


def test_reconciliation_v2_is_validated_and_persisted_with_attempt_lineage(
  tmp_path,
) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  tracker = CommercialUsageReconciliationTracker(
    request_id="req_001", session_id="sess_001"
  )
  tracker.record_batch([_v2_payload("evt_v2_001")])
  summary = SessionUsageSummary(
    user_id="user-001", session_id="sess_001", request_id="req_001",
    input_tokens=100, output_tokens=20, cache_read_tokens=10,
    cache_creation_tokens=0, cost=0.002, turns=1, channel="mcp",
    started_at=NOW.timestamp() - 60, ended_at=NOW.timestamp(),
    usage_event_count=1, usage_event_ids=("evt_v2_001",),
    capability_bind=_v3_payload("bind")["capability_bind"],
  )

  stored, replayed = outbox.record_reconciliation_report(
    tracker.compare(summary), recorded_at=NOW
  )

  assert replayed is False
  assert stored.payload["evidence_schema_version"] == 2
  assert stored.payload["workflow_attempt_number"] == 1
  assert stored.payload["event_lines"][0]["source_schema_version"] == 2
  assert stored.payload["event_lines"][0]["work_authorization_id"] == (
    "77777777-7777-4777-8777-777777777777"
  )
  forged = dict(stored.payload)
  forged["event_lines"] = [
    {
      **stored.payload["event_lines"][0],
      "work_authorization_id": "88888888-8888-4888-8888-888888888888",
    }
  ]
  with pytest.raises(ValueError, match="attempt lineage is inconsistent"):
    outbox.record_reconciliation_report(forged)

  other_attempt = CommercialUsageReconciliationTracker(
    request_id="req_001", session_id="sess_001"
  )
  other_payload = {
    **_v2_payload("evt_v2_001"),
    "work_authorization_id": "88888888-8888-4888-8888-888888888888",
  }
  other_payload["source_payload_sha256"] = canonical_usage_payload_sha256(
    other_payload
  )
  other_attempt.record_batch([other_payload])
  with pytest.raises(ValueError, match="cannot change within a revision chain"):
    outbox.record_reconciliation_report(other_attempt.compare(summary))


def test_reconciliation_v2_rejects_noncanonical_manifest_order(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  tracker = CommercialUsageReconciliationTracker(
    request_id="req_001", session_id="sess_001"
  )
  root = _v2_payload("evt_v2_a")
  unit = {
    **_v2_payload("evt_v2_b"),
    "uncached_input_tokens": 0,
    "billable_output_tokens": 0,
    "reasoning_tokens_observed": None,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0,
    "provider_units": "1",
    "producer_estimated_cost_usd": None,
  }
  unit["source_payload_sha256"] = canonical_usage_payload_sha256(unit)
  tracker.record_batch([root, unit])
  summary = SessionUsageSummary(
    user_id="user-001", session_id="sess_001", request_id="req_001",
    input_tokens=100, output_tokens=20, cache_read_tokens=10,
    cache_creation_tokens=0, cost=0.002, turns=1, channel="mcp",
    started_at=NOW.timestamp() - 60, ended_at=NOW.timestamp(),
    usage_event_count=1, usage_event_ids=("evt_v2_a",),
    capability_bind=_v3_payload("bind")["capability_bind"],
  )
  reversed_manifest = tracker.compare(summary).as_dict()
  reversed_manifest["event_lines"] = list(
    reversed(reversed_manifest["event_lines"])
  )
  reversed_manifest["observed_source_event_ids"] = list(
    reversed(reversed_manifest["observed_source_event_ids"])
  )

  with pytest.raises(ValueError, match="identity lists are not canonical"):
    outbox.record_reconciliation_report(reversed_manifest)
  assert outbox.current_reconciliation_report(
    environment="prod", source_product="hank-agent-gateway",
    request_id="req_001", session_id="sess_001",
  ) is None


def test_reconciliation_report_rejects_self_inconsistent_totals_and_status(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  tracker = CommercialUsageReconciliationTracker(
    request_id="req_001", session_id="sess_001"
  )
  tracker.record_batch([_payload("evt_001")])
  report = tracker.compare(SessionUsageSummary(
    user_id="user-001", session_id="sess_001", request_id="req_001",
    input_tokens=100, output_tokens=20, cache_read_tokens=10,
    cache_creation_tokens=0, cost=0.002, turns=1, channel="mcp",
    started_at=NOW.timestamp() - 60, ended_at=NOW.timestamp(),
    usage_event_count=1, usage_event_ids=("evt_001",),
    capability_bind=_v3_payload("bind")["capability_bind"],
  )).as_dict()
  report.update(commercial_event_count=999, input_token_delta=123, status="match")

  with pytest.raises(ValueError, match="inconsistent"):
    outbox.record_reconciliation_report(report)
  invalid_conflict = tracker.compare(SessionUsageSummary(
    user_id="user-001", session_id="sess_001", request_id="req_001",
    input_tokens=100, output_tokens=20, cache_read_tokens=10,
    cache_creation_tokens=0, cost=0.002, turns=1, channel="mcp",
    started_at=NOW.timestamp() - 60, ended_at=NOW.timestamp(),
    usage_event_count=1, usage_event_ids=("evt_001",),
    capability_bind=_v3_payload("bind")["capability_bind"],
  )).as_dict()
  invalid_conflict.update(
    conflicting_event_id_count=1,
    conflicting_source_event_ids=["not-observed"],
    status="mismatch",
  )
  with pytest.raises(ValueError, match="inconsistent"):
    outbox.record_reconciliation_report(invalid_conflict)
  non_finite_time = tracker.compare(SessionUsageSummary(
    user_id="user-001", session_id="sess_001", request_id="req_001",
    input_tokens=100, output_tokens=20, cache_read_tokens=10,
    cache_creation_tokens=0, cost=0.002, turns=1, channel="mcp",
    started_at=float("nan"), ended_at=NOW.timestamp(),
    usage_event_count=1, usage_event_ids=("evt_001",),
    capability_bind=_v3_payload("bind")["capability_bind"],
  ))
  with pytest.raises(ValueError, match="inconsistent"):
    outbox.record_reconciliation_report(non_finite_time)
  assert outbox.reconciliation_reports(
    environment="prod", source_product="hank-agent-gateway",
    recorded_from=NOW - timedelta(seconds=1),
    recorded_until=NOW + timedelta(seconds=1),
  ) == ()


def test_reconciliation_shipments_are_fenced_ordered_and_terminal(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  outbox.enable_reconciliation_shipping()
  tracker = CommercialUsageReconciliationTracker(
    request_id="req_001", session_id="sess_001"
  )
  tracker.record_batch([_payload("evt_001")])
  summary = SessionUsageSummary(
    user_id="user-001", session_id="sess_001", request_id="req_001",
    input_tokens=100, output_tokens=20, cache_read_tokens=10,
    cache_creation_tokens=0, cost=0.002, turns=1, channel="mcp",
    started_at=NOW.timestamp() - 60, ended_at=NOW.timestamp(),
    usage_event_count=1, usage_event_ids=("evt_001",),
    capability_bind=_v3_payload("bind")["capability_bind"],
  )
  first, _ = outbox.record_reconciliation_report(tracker.compare(summary), recorded_at=NOW)
  tracker.mark_late("evt_001")
  second, _ = outbox.record_reconciliation_report(
    tracker.compare(summary), recorded_at=NOW + timedelta(seconds=1)
  )

  leased_first = outbox.lease_reconciliation_batch(
    limit=10, lease_for=timedelta(seconds=30), now=NOW + timedelta(seconds=2)
  )
  assert [row.report_id for row in leased_first] == [first.report_id]
  assert leased_first[0].envelope == {
    "report_sha256": first.report_sha256,
    "evidence": first.payload,
  }
  assert outbox.mark_reconciliation_accepted(
    first.report_id, "wrong-token", ingest_status="accepted", accepted_at=NOW
  ) is False
  assert outbox.mark_reconciliation_accepted(
    first.report_id, leased_first[0].sending_lease_token or "",
    ingest_status="accepted", accepted_at=NOW + timedelta(seconds=2),
  ) is True

  leased_second = outbox.lease_reconciliation_batch(
    limit=10, lease_for=timedelta(seconds=30), now=NOW + timedelta(seconds=3)
  )
  assert [row.report_id for row in leased_second] == [second.report_id]
  assert outbox.mark_reconciliation_dead(
    second.report_id, leased_second[0].sending_lease_token or "",
    error="conflict", ingest_status="conflict",
    reason_code="usage_reconciliation.revision_conflict",
    decided_at=NOW + timedelta(seconds=3),
  ) is True
  with sqlite3.connect(outbox.path) as connection:
    with pytest.raises(sqlite3.IntegrityError, match="shipment is terminal"):
      connection.execute(
        "UPDATE commercial_usage_reconciliation_shipments "
        "SET attempt_count = 99 WHERE report_id = ?",
        (second.report_id,),
      )
  assert outbox.health()["reconciliation_shipment_counts"] == {
    "accepted": 1, "dead": 1,
  }


def test_v2_upgrade_backfills_pending_reconciliation_shipments(tmp_path) -> None:
  path = tmp_path / "usage.sqlite3"
  outbox = CommercialUsageOutbox(path)
  tracker = CommercialUsageReconciliationTracker(
    request_id="req_001", session_id="sess_001"
  )
  tracker.record_batch([_payload("evt_001")])
  report, _ = outbox.record_reconciliation_report(
    tracker.compare(SessionUsageSummary(
      user_id="user-001", session_id="sess_001", request_id="req_001",
      input_tokens=100, output_tokens=20, cache_read_tokens=10,
      cache_creation_tokens=0, cost=0.002, turns=1, channel="mcp",
      started_at=NOW.timestamp() - 60, ended_at=NOW.timestamp(),
      usage_event_count=1, usage_event_ids=("evt_001",),
      capability_bind=_v3_payload("bind")["capability_bind"],
    )),
    recorded_at=NOW,
  )
  with sqlite3.connect(path) as connection:
    connection.execute("DROP TABLE commercial_usage_reconciliation_shipments")
    connection.execute("PRAGMA user_version=2")

  upgraded = CommercialUsageOutbox(path)
  assert upgraded.health()["schema_version"] == 3
  assert upgraded.health()["reconciliation_shipment_counts"] == {"held": 1}
  assert upgraded.health()["reconciliation_shipment_backlog_count"] == 0
  assert upgraded.enable_reconciliation_shipping() == 1
  leased = upgraded.lease_reconciliation_batch(
    limit=1, lease_for=timedelta(seconds=30), now=NOW + timedelta(seconds=1)
  )
  assert [row.report_id for row in leased] == [report.report_id]


def test_v2_upgrade_rejects_preexisting_v3_target_artifacts(tmp_path) -> None:
  path = tmp_path / "usage.sqlite3"
  CommercialUsageOutbox(path)
  with sqlite3.connect(path) as connection:
    connection.execute("DROP TABLE commercial_usage_reconciliation_shipments")
    connection.execute(
      "CREATE TABLE commercial_usage_reconciliation_shipments "
      "(report_id TEXT PRIMARY KEY, state TEXT NOT NULL)"
    )
    connection.execute("PRAGMA user_version=2")

  with pytest.raises(CommercialUsageOutboxError, match="version 2"):
    CommercialUsageOutbox(path)
