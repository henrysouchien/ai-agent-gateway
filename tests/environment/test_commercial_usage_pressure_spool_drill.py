from __future__ import annotations

from pathlib import Path

import pytest

from agent_gateway.commercial_contract import canonical_usage_payload_sha256
from agent_gateway.usage_outbox import CommercialUsageOutbox
from agent_gateway.usage_resilience import (
    CommercialUsageCircuitBreaker,
    CommercialUsageCircuitOpen,
    CommercialUsageDurability,
    CommercialUsageEmergencySpool,
    CommercialUsageSpoolError,
    ResilientCommercialUsageSink,
)


def _payload(event_id: str) -> dict:
    payload = {
        "schema_version": 1,
        "source_product": "hank-agent-gateway",
        "source_event_id": event_id,
        "environment": "prod",
        "occurred_at": "2026-07-13T19:00:00Z",
        "execution_context_id": "33333333-3333-4333-8333-333333333333",
        "request_id": "gateway-pressure-drill",
        "session_id": "gateway-pressure-drill",
        "parent_turn_id": None,
        "workflow_run_id": "44444444-4444-4444-8444-444444444444",
        "reservation_id": "55555555-5555-4555-8555-555555555555",
        "funding_route_id": "66666666-6666-4666-8666-666666666666",
        "channel": "mcp",
        "provider": "openai",
        "operation": "responses.create",
        "model": "gpt-5",
        "capability_id": "research.read",
        "usage_state": "succeeded",
        "uncached_input_tokens": 100,
        "billable_output_tokens": 20,
        "reasoning_tokens_observed": 5,
        "cache_write_tokens": 0,
        "cache_read_tokens": 0,
        "is_batch": False,
        "provider_units": None,
        "separately_billed_tool_cost_usd": "0",
        "producer_estimated_cost_usd": "0.002",
        "provider_reported_cost_usd": None,
        "cost_observation_kind": "producer_estimate",
        "producer_rate_version": "gateway-pressure-drill-v1",
        "shadow_rate_version": "openai-2026-07-01",
        "raw_billing_mode": "metered",
    }
    payload["source_payload_sha256"] = canonical_usage_payload_sha256(payload)
    return payload


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_real_gateway_outbox_pressure_spool_recovery(tmp_path: Path) -> None:
    spool_root = tmp_path / "spool-recovery"
    durable_outbox = CommercialUsageOutbox(spool_root / "usage.sqlite3")

    class RecoverablePrimary:
        failed = True

        def enqueue_batch(self, payloads):
            if self.failed:
                raise OSError("isolated primary durability outage")
            durable_outbox.enqueue_batch(payloads)

        def health(self):
            return durable_outbox.health()

    primary = RecoverablePrimary()
    spool = CommercialUsageEmergencySpool(spool_root / "emergency.spool")
    breaker = CommercialUsageCircuitBreaker(spool_root / "circuit.json")
    sink = ResilientCommercialUsageSink(
        outbox=primary,
        spool=spool,
        circuit_breaker=breaker,
        max_backlog=10,
        max_storage_bytes=10_000_000,
    )

    assert sink([_payload("gateway-pressure-spool-001")]) == "emergency_spool"
    assert spool.pending_batches() == 1
    incident_at = breaker.snapshot.tripped_at or ""
    assert incident_at
    assert _mode(spool.path) == 0o600
    assert _mode(breaker.path) == 0o600
    with pytest.raises(CommercialUsageCircuitOpen, match="replay is pending"):
        sink.reset_after_recovery(
            expected_tripped_at=incident_at,
            operator_id="gateway-pressure-drill",
            reconciliation_evidence_id="gateway-pressure-drill:reconciled",
            max_backlog_count=1,
        )

    primary.failed = False
    assert sink.replay_emergency() == 1
    assert spool.pending_batches() == 0
    assert durable_outbox.health()["counts"] == {"pending": 1}
    assert sink.replay_emergency() == 0
    assert durable_outbox.health()["counts"] == {"pending": 1}
    connection = durable_outbox._connect()
    try:
        private_paths = (
            durable_outbox.path,
            Path(f"{durable_outbox.path}-wal"),
            Path(f"{durable_outbox.path}-shm"),
            spool.path,
            spool.lock_path,
            spool.cursor_path,
            breaker.path,
            breaker.lock_path,
        )
        assert all(path.is_file() for path in private_paths)
        assert {_mode(path) for path in private_paths} == {0o600}
        assert _mode(spool_root) == 0o700
    finally:
        connection.close()
    reset = sink.reset_after_recovery(
        expected_tripped_at=incident_at,
        operator_id="gateway-pressure-drill",
        reconciliation_evidence_id="gateway-pressure-drill:reconciled",
        max_backlog_count=1,
    )
    assert reset.tripped is False
    assert reset.last_reset_prior_tripped_at == incident_at
    assert reset.last_reset_at
    assert reset.last_reset_operator_id == "gateway-pressure-drill"
    assert reset.last_reset_evidence_id == "gateway-pressure-drill:reconciled"
    assert sink([_payload("gateway-pressure-resumed-001")]) == "outbox"
    assert durable_outbox.health()["counts"] == {"pending": 2}

    backlog_root = tmp_path / "backlog-pressure"
    backlog = CommercialUsageDurability.create(
        outbox_path=backlog_root / "usage.sqlite3",
        spool_path=backlog_root / "emergency.spool",
        circuit_state_path=backlog_root / "circuit.json",
        max_backlog=1,
        max_storage_bytes=10_000_000,
    )
    assert backlog.sink([_payload("gateway-pressure-backlog-001")]) == "outbox"
    with pytest.raises(CommercialUsageCircuitOpen, match="backlog"):
        backlog.sink.assert_work_allowed("metered")
    assert backlog.circuit_breaker.snapshot.tripped is True

    disk_root = tmp_path / "disk-pressure"
    disk = CommercialUsageDurability.create(
        outbox_path=disk_root / "usage.sqlite3",
        spool_path=disk_root / "emergency.spool",
        circuit_state_path=disk_root / "circuit.json",
        max_backlog=10,
        max_storage_bytes=1,
    )
    with pytest.raises(CommercialUsageCircuitOpen, match="disk high-water"):
        disk.sink.assert_work_allowed("metered")
    assert "disk high-water" in (disk.circuit_breaker.snapshot.reason or "")

    corrupt_root = tmp_path / "corrupt-spool"
    corrupt = CommercialUsageEmergencySpool(corrupt_root / "emergency.spool")
    corrupt.append_batch([_payload("gateway-pressure-corrupt-001")])
    with corrupt.path.open("r+b") as stream:
        stream.seek(-2, 2)
        original = stream.read(1)
        stream.seek(-1, 1)
        stream.write(b"x" if original != b"x" else b"y")
        stream.flush()
    with pytest.raises(CommercialUsageSpoolError, match="checksum mismatch"):
        corrupt.pending_batches()
