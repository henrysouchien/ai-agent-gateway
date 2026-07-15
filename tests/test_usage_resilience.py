from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import multiprocessing

import pytest

from agent_gateway.commercial_contract import canonical_usage_payload_sha256
from agent_gateway.usage_outbox import CommercialUsageOutbox
from agent_gateway.usage_resilience import (
  CommercialUsageCircuitBreaker,
  CommercialUsageCircuitOpen,
  CommercialUsageEmergencySpool,
  CommercialUsageSpoolError,
  ResilientCommercialUsageSink,
)


NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)


def _trip_breaker_in_child(path: str) -> None:
  CommercialUsageCircuitBreaker(path).trip("child worker incident", now=NOW)


def _payload(event_id: str) -> dict:
  payload = {
    "schema_version": 1, "source_product": "hank-agent-gateway",
    "source_event_id": event_id, "environment": "prod",
    "occurred_at": "2026-07-11T11:59:00Z",
    "execution_context_id": "33333333-3333-4333-8333-333333333333",
    "request_id": "req_001", "session_id": "sess_001", "parent_turn_id": None,
    "workflow_run_id": "wf_001", "reservation_id": "res_001",
    "funding_route_id": "fund_001", "channel": "mcp", "provider": "anthropic",
    "operation": "messages.create", "model": "claude-sonnet-test",
    "capability_id": "portfolio.review", "usage_state": "succeeded",
    "uncached_input_tokens": 10, "billable_output_tokens": 2,
    "reasoning_tokens_observed": 1, "cache_write_tokens": 0, "cache_read_tokens": 0,
    "is_batch": False, "provider_units": None,
    "separately_billed_tool_cost_usd": "0", "producer_estimated_cost_usd": "0.001",
    "provider_reported_cost_usd": None, "cost_observation_kind": "producer_estimate",
    "producer_rate_version": "rates-v1", "shadow_rate_version": "shadow-v1",
    "raw_billing_mode": "metered",
  }
  payload["source_payload_sha256"] = canonical_usage_payload_sha256(payload)
  return payload


def test_checksum_spool_replays_batches_once_with_atomic_cursor(tmp_path) -> None:
  spool = CommercialUsageEmergencySpool(tmp_path / "emergency.spool")
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  spool.append_batch([_payload("evt_001")])
  spool.append_batch([_payload("evt_002"), _payload("evt_003")])

  assert spool.pending_batches() == 2
  assert spool.replay_into(outbox, limit=1) == 1
  assert spool.pending_batches() == 1
  assert spool.replay_into(outbox) == 1
  assert spool.pending_batches() == 0
  assert spool.replay_into(outbox) == 0
  assert outbox.health(now=NOW)["counts"] == {"pending": 3}
  assert spool.cursor_path.stat().st_mode & 0o777 == 0o600


def test_circuit_state_is_process_shared_and_byok_grace_is_bounded(tmp_path) -> None:
  state = tmp_path / "circuit.json"
  first = CommercialUsageCircuitBreaker(state, byok_grace_seconds=30)
  sibling = CommercialUsageCircuitBreaker(state, byok_grace_seconds=30)
  assert first.trip("primary failed", now=NOW) is True
  assert sibling.snapshot.tripped is True
  with pytest.raises(CommercialUsageCircuitOpen, match="primary failed"):
    sibling.assert_work_allowed("metered", now=NOW)
  sibling.assert_work_allowed("byok", now=NOW + timedelta(seconds=30))
  with pytest.raises(CommercialUsageCircuitOpen):
    sibling.assert_work_allowed("byok", now=NOW + timedelta(seconds=30, microseconds=1))
  sibling._reset_with_evidence(
    expected_tripped_at=sibling.snapshot.tripped_at or "",
    operator_id="test-operator",
    reconciliation_evidence_id="test:reconciliation:clean",
  )
  first.assert_work_allowed("metered", now=NOW)


def test_circuit_trip_is_visible_to_sibling_process(tmp_path) -> None:
  state = tmp_path / "circuit.json"
  context = multiprocessing.get_context("fork")
  process = context.Process(target=_trip_breaker_in_child, args=(str(state),))
  process.start()
  process.join(timeout=5)
  assert process.exitcode == 0
  sibling = CommercialUsageCircuitBreaker(state)
  with pytest.raises(CommercialUsageCircuitOpen, match="child worker incident"):
    sibling.assert_work_allowed("metered", now=NOW)


def test_corrupt_circuit_state_fails_closed(tmp_path) -> None:
  state = tmp_path / "circuit.json"
  state.write_text("{bad", encoding="utf-8")
  breaker = CommercialUsageCircuitBreaker(state)
  assert breaker.snapshot.tripped is True
  with pytest.raises(CommercialUsageCircuitOpen, match="state is corrupt"):
    breaker.assert_work_allowed("metered", now=NOW)


def test_process_safe_concurrent_append_preserves_every_frame(tmp_path) -> None:
  spool = CommercialUsageEmergencySpool(tmp_path / "emergency.spool")
  with ThreadPoolExecutor(max_workers=8) as executor:
    list(executor.map(
      lambda index: spool.append_batch([_payload(f"evt_{index:03d}")]),
      range(24),
    ))
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  assert spool.pending_batches() == 24
  assert spool.replay_into(outbox) == 24
  assert outbox.health(now=NOW)["counts"] == {"pending": 24}


def test_spool_checksum_corruption_fails_closed(tmp_path) -> None:
  spool = CommercialUsageEmergencySpool(tmp_path / "emergency.spool")
  spool.append_batch([_payload("evt_001")])
  data = bytearray(spool.path.read_bytes())
  data[-3] ^= 1
  spool.path.write_bytes(data)
  with pytest.raises(CommercialUsageSpoolError, match="checksum"):
    spool.pending_batches()


@pytest.mark.parametrize("cut_bytes", [1, 20])
def test_partial_spool_frame_is_never_treated_as_durable(tmp_path, cut_bytes) -> None:
  spool = CommercialUsageEmergencySpool(tmp_path / "emergency.spool")
  spool.append_batch([_payload("evt_001")])
  data = spool.path.read_bytes()
  spool.path.write_bytes(data[:-cut_bytes])
  with pytest.raises(CommercialUsageSpoolError, match="incomplete"):
    spool.pending_batches()


def test_replay_crash_after_enqueue_before_cursor_update_is_idempotent(tmp_path, monkeypatch) -> None:
  spool = CommercialUsageEmergencySpool(tmp_path / "emergency.spool")
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  spool.append_batch([_payload("evt_001")])
  original = spool._write_cursor
  monkeypatch.setattr(spool, "_write_cursor", lambda **kwargs: (_ for _ in ()).throw(OSError("crash")))
  with pytest.raises(OSError, match="crash"):
    spool.replay_into(outbox)
  assert outbox.get("evt_001") is not None
  monkeypatch.setattr(spool, "_write_cursor", original)
  assert spool.replay_into(outbox) == 1
  assert spool.pending_batches() == 0
  assert outbox.health(now=NOW)["counts"] == {"pending": 1}


def test_foreign_and_non_boundary_replay_cursors_fail_closed(tmp_path) -> None:
  spool = CommercialUsageEmergencySpool(tmp_path / "emergency.spool")
  spool.append_batch([_payload("evt_001")])
  header = spool.path.read_bytes().split(b"\n", 1)[0].decode("ascii")
  file_id = header.split(" ", 1)[1]
  spool.cursor_path.write_text(
    json.dumps({"version": 1, "file_id": "foreign", "offset": len(header) + 1}),
    encoding="utf-8",
  )
  with pytest.raises(CommercialUsageSpoolError, match="identity"):
    spool.pending_batches()
  spool.cursor_path.write_text(
    json.dumps({"version": 1, "file_id": file_id, "offset": len(header) + 2}),
    encoding="utf-8",
  )
  with pytest.raises(CommercialUsageSpoolError, match="frame"):
    spool.pending_batches()


def test_spool_size_limit_models_disk_full_failure(tmp_path) -> None:
  spool = CommercialUsageEmergencySpool(
    tmp_path / "emergency.spool", max_batch_bytes=100, max_spool_bytes=200
  )
  with pytest.raises(CommercialUsageSpoolError, match="frame limit"):
    spool.append_batch([_payload("evt_001")])


def test_resilient_sink_spools_primary_failure_trips_and_never_raises(tmp_path) -> None:
  class FailedOutbox:
    def enqueue_batch(self, payloads):
      raise OSError("disk full")

    def health(self):
      return {"backlog_count": 0, "storage_bytes": 0}

  alerts = []
  breaker = CommercialUsageCircuitBreaker(tmp_path / "circuit.json")
  spool = CommercialUsageEmergencySpool(tmp_path / "emergency.spool")
  sink = ResilientCommercialUsageSink(
    outbox=FailedOutbox(), spool=spool, circuit_breaker=breaker,
    max_backlog=100, max_storage_bytes=1_000_000,
    alert=lambda code, details: alerts.append((code, details)),
  )

  assert sink([_payload("evt_001")]) == "emergency_spool"

  assert spool.pending_batches() == 1
  assert breaker.snapshot.tripped is True
  with pytest.raises(CommercialUsageCircuitOpen):
    sink.assert_work_allowed("metered")
  sink.assert_work_allowed("byok")
  assert [item[0] for item in alerts] == [
    "commercial_usage.primary_outbox_failed", "commercial_usage.emergency_spool_used",
  ]


def test_all_durability_failure_still_preserves_paid_result_and_blocks_future_work(tmp_path) -> None:
  class FailedOutbox:
    def enqueue_batch(self, payloads):
      raise OSError("primary failed")

    def health(self):
      return {"backlog_count": 0, "storage_bytes": 0}

  class FailedSpool:
    path = tmp_path / "missing.spool"

    def append_batch(self, payloads):
      raise OSError("spool failed")

  alerts = []
  breaker = CommercialUsageCircuitBreaker(
    tmp_path / "circuit.json", byok_grace_seconds=0
  )
  sink = ResilientCommercialUsageSink(
    outbox=FailedOutbox(), spool=FailedSpool(), circuit_breaker=breaker,
    max_backlog=100, max_storage_bytes=1_000_000,
    alert=lambda code, details: alerts.append(code),
  )

  assert sink([_payload("evt_001")]) == "lost"
  assert breaker.snapshot.reason.startswith("all commercial usage durability failed")
  with pytest.raises(CommercialUsageCircuitOpen):
    sink.assert_work_allowed("metered")
  with pytest.raises(CommercialUsageCircuitOpen):
    sink.assert_work_allowed("byok")
  assert alerts[-1] == "commercial_usage.all_durability_failed"


def test_backlog_and_storage_high_water_trip_before_new_metered_work(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  outbox.enqueue_batch([_payload("evt_001")], created_at=NOW)
  breaker = CommercialUsageCircuitBreaker(tmp_path / "circuit.json")
  alerts = []
  sink = ResilientCommercialUsageSink(
    outbox=outbox,
    spool=CommercialUsageEmergencySpool(tmp_path / "emergency.spool"),
    circuit_breaker=breaker,
    max_backlog=1,
    max_storage_bytes=1_000_000_000,
    alert=lambda code, details: alerts.append(code),
  )
  with pytest.raises(CommercialUsageCircuitOpen, match="backlog"):
    sink.assert_work_allowed("metered")
  assert alerts == ["commercial_usage.backlog_high_water"]


def test_reconciliation_shipment_high_water_trips_before_new_metered_work(
  tmp_path,
) -> None:
  class ReconciliationBacklog:
    def health(self):
      return {
        "ok": True,
        "backlog_count": 0,
        "reconciliation_shipment_backlog_count": 1,
        "storage_bytes": 0,
      }

  alerts = []
  sink = ResilientCommercialUsageSink(
    outbox=ReconciliationBacklog(),
    spool=CommercialUsageEmergencySpool(tmp_path / "emergency.spool"),
    circuit_breaker=CommercialUsageCircuitBreaker(tmp_path / "circuit.json"),
    max_backlog=1,
    max_storage_bytes=1_000_000,
    alert=lambda code, details: alerts.append(code),
  )
  with pytest.raises(CommercialUsageCircuitOpen, match="reconciliation shipment"):
    sink.assert_work_allowed("metered")
  assert alerts == ["commercial_usage.reconciliation_backlog_high_water"]


def test_outbox_health_failure_trips_shared_breaker_and_alerts(tmp_path) -> None:
  class BrokenHealthOutbox:
    def health(self):
      raise OSError("database corrupt")

  alerts = []
  breaker = CommercialUsageCircuitBreaker(tmp_path / "circuit.json")
  sink = ResilientCommercialUsageSink(
    outbox=BrokenHealthOutbox(),
    spool=CommercialUsageEmergencySpool(tmp_path / "emergency.spool"),
    circuit_breaker=breaker, max_backlog=10, max_storage_bytes=1_000_000,
    alert=lambda code, details: alerts.append(code),
  )
  with pytest.raises(CommercialUsageCircuitOpen, match="health failed"):
    sink.assert_work_allowed("metered")
  assert breaker.snapshot.tripped is True
  assert alerts == ["commercial_usage.outbox_health_failed"]


def test_replay_must_drain_emergency_spool_before_guarded_reset_allows_work(tmp_path) -> None:
  durable = CommercialUsageOutbox(tmp_path / "usage.sqlite3")

  class FlakyOutbox:
    failed = True

    def enqueue_batch(self, payloads):
      if self.failed:
        raise OSError("temporary primary outage")
      durable.enqueue_batch(payloads)

    def health(self):
      return durable.health(now=NOW)

  primary = FlakyOutbox()
  breaker = CommercialUsageCircuitBreaker(tmp_path / "circuit.json")
  spool = CommercialUsageEmergencySpool(tmp_path / "emergency.spool")
  sink = ResilientCommercialUsageSink(
    outbox=primary, spool=spool, circuit_breaker=breaker,
    max_backlog=100, max_storage_bytes=1_000_000,
  )
  sink([_payload("evt_001")])
  incident_at = breaker.snapshot.tripped_at or ""
  with pytest.raises(CommercialUsageCircuitOpen, match="replay is pending"):
    sink.reset_after_recovery(
      expected_tripped_at=incident_at,
      operator_id="test-operator",
      reconciliation_evidence_id="test:reconciliation:clean",
      max_backlog_count=1,
    )

  primary.failed = False
  assert sink.replay_emergency() == 1
  assert durable.get("evt_001") is not None
  sink.reset_after_recovery(
    expected_tripped_at=incident_at,
    operator_id="test-operator",
    reconciliation_evidence_id="test:reconciliation:clean",
    max_backlog_count=1,
  )
  sink.assert_work_allowed("metered")
