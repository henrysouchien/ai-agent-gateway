from __future__ import annotations

from dataclasses import replace

import pytest

from agent_gateway.multi_user.billing import SessionUsageSummary
from agent_gateway.usage_reconciliation import CommercialUsageReconciliationTracker


def _summary(**changes) -> SessionUsageSummary:
  values = {
    "user_id": "alice", "session_id": "sess_001", "request_id": "req_001",
    "input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 10,
    "cache_creation_tokens": 5, "cost": 0.01, "turns": 1, "channel": "mcp",
    "started_at": 1.0, "ended_at": 2.0, "drain_complete": True,
    "in_flight_task_count": 0,
    "usage_event_count": 1, "usage_event_ids": ("evt_001",),
  }
  values.update(changes)
  return SessionUsageSummary(**values)


def _base_payload(event_id: str = "evt_001") -> dict:
  return {
    "environment": "prod",
    "source_product": "hank-agent-gateway",
    "execution_context_id": "context-001",
    "workflow_run_id": "workflow-001",
    "occurred_at": "2026-07-11T12:00:00Z",
    "source_event_id": event_id,
    "source_payload_sha256": f"sha256:{event_id}",
    "request_id": "req_001",
    "session_id": "sess_001",
    "uncached_input_tokens": 100,
    "billable_output_tokens": 20,
    "reasoning_tokens_observed": 4,
    "cache_read_tokens": 10,
    "cache_write_tokens": 5,
    "provider_units": None,
    "producer_estimated_cost_usd": "0.01",
  }


def _unit_payload() -> dict:
  return {
    **_base_payload("evt_unit_001"),
    "uncached_input_tokens": 0,
    "billable_output_tokens": 0,
    "reasoning_tokens_observed": None,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0,
    "provider_units": "2",
    "producer_estimated_cost_usd": None,
  }


def _v2_payload(event_id: str = "evt_v2_001") -> dict:
  return {
    **_base_payload(event_id),
    "schema_version": 2,
    "execution_context_id": "33333333-3333-4333-8333-333333333333",
    "workflow_run_id": "44444444-4444-4444-8444-444444444444",
    "workflow_attempt_group_id": "44444444-4444-4444-8444-444444444444",
    "workflow_attempt_number": 1,
    "retry_of_workflow_run_id": None,
    "workflow_attempt_kind": "initial",
    "work_authorization_id": "55555555-5555-4555-8555-555555555555",
  }


def test_reconciliation_matches_tokens_estimate_counts_and_separate_units() -> None:
  tracker = CommercialUsageReconciliationTracker(
    request_id="req_001", session_id="sess_001"
  )
  tracker.record_batch([_base_payload(), _unit_payload()])

  report = tracker.compare(_summary())

  assert report.status == "match"
  assert report.commercial_event_count == 2
  assert report.provider_call_event_count == 1
  assert report.separate_unit_event_count == 1
  assert str(report.provider_units) == "2"
  assert report.reasoning_tokens_observed == 4
  assert report.missing_event_id_count == 0
  assert report.estimate_delta_usd == 0
  assert report.summary_emitted_as_cost_event is False
  assert report.as_dict()["summary_emitted_as_cost_event"] is False


def test_reconciliation_reports_missing_mismatch_late_and_incomplete() -> None:
  tracker = CommercialUsageReconciliationTracker(
    request_id="req_001", session_id="sess_001"
  )
  tracker.record_batch([_base_payload()])
  tracker.mark_late("evt_001")

  mismatch = tracker.compare(_summary(
    turns=2, input_tokens=99, cost=0.02,
    usage_event_count=2, usage_event_ids=("evt_001", "evt_missing"),
  ))
  assert mismatch.status == "mismatch"
  assert mismatch.missing_event_id_count == 1
  assert mismatch.missing_source_event_ids == ("evt_missing",)
  assert mismatch.input_token_delta == 1
  assert mismatch.estimate_delta_usd < 0
  assert mismatch.late_source_event_ids == ("evt_001",)

  incomplete = tracker.compare(replace(_summary(), drain_complete=False, in_flight_task_count=1))
  assert incomplete.status == "incomplete"


def test_reconciliation_deduplicates_identical_ids_and_flags_conflicts() -> None:
  tracker = CommercialUsageReconciliationTracker(
    request_id="req_001", session_id="sess_001"
  )
  payload = _base_payload()
  tracker.record_batch([payload])
  tracker.record_batch([dict(payload)])
  tracker.record_batch([{**payload, "source_payload_sha256": "sha256:conflict"}])

  report = tracker.compare(_summary())

  assert report.commercial_event_count == 1
  assert report.conflicting_event_id_count == 1
  assert report.conflicting_source_event_ids == ("evt_001",)
  assert report.status == "mismatch"


def test_reconciliation_rejects_cross_request_or_session_facts() -> None:
  tracker = CommercialUsageReconciliationTracker(
    request_id="req_001", session_id="sess_001"
  )
  try:
    tracker.record_batch([{**_base_payload(), "request_id": "other"}])
  except ValueError as exc:
    assert "identity mismatch" in str(exc)
  else:
    raise AssertionError("cross-request reconciliation payload was accepted")


def test_reconciliation_distinguishes_emergency_durability_from_lost_fact() -> None:
  tracker = CommercialUsageReconciliationTracker(
    request_id="req_001", session_id="sess_001"
  )
  tracker.record_batch([_base_payload()], durability="lost")
  lost = tracker.compare(_summary())
  assert lost.status == "mismatch"
  assert lost.durability_lost_event_count == 1
  assert lost.missing_event_id_count == 1
  assert lost.durable_provider_call_event_count == 0

  tracker.record_batch([_base_payload()], durability="emergency_spool")
  recovered = tracker.compare(_summary())
  assert recovered.status == "match"
  assert recovered.emergency_spooled_event_count == 1
  assert recovered.durability_lost_event_count == 0


def test_parent_with_provider_units_remains_provider_call_and_cannot_double_children() -> None:
  tracker = CommercialUsageReconciliationTracker(
    request_id="req_001", session_id="sess_001"
  )
  parent = {**_base_payload(), "provider_units": "2"}
  tracker.record_batch([parent])
  report = tracker.compare(_summary())
  assert report.provider_call_event_count == 1
  assert report.separate_unit_event_count == 0
  assert report.provider_units == 2


def test_reconciliation_requires_exact_provider_event_identity_set() -> None:
  tracker = CommercialUsageReconciliationTracker(
    request_id="req_001", session_id="sess_001"
  )
  tracker.record_batch([_base_payload()])

  report = tracker.compare(_summary(usage_event_ids=()))

  assert report.provider_call_count_delta == 0
  assert report.missing_event_id_count == 0
  assert report.status == "mismatch"


def test_reconciliation_v2_carries_exact_attempt_authority_on_report_and_lines() -> None:
  tracker = CommercialUsageReconciliationTracker(
    request_id="req_001", session_id="sess_001"
  )
  payload = _v2_payload()
  tracker.record_batch([payload])

  report = tracker.compare(_summary(
    usage_event_ids=("evt_v2_001",),
  ))
  document = report.as_dict()

  assert report.evidence_schema_version == 2
  assert document["workflow_attempt_group_id"] == payload["workflow_attempt_group_id"]
  assert document["workflow_attempt_number"] == 1
  assert document["retry_of_workflow_run_id"] is None
  assert document["workflow_attempt_kind"] == "initial"
  assert document["work_authorization_id"] == payload["work_authorization_id"]
  assert document["event_lines"][0]["source_schema_version"] == 2
  assert document["event_lines"][0]["work_authorization_id"] == (
    payload["work_authorization_id"]
  )


def test_reconciliation_v2_rejects_mixed_or_drifting_attempt_authority() -> None:
  tracker = CommercialUsageReconciliationTracker(
    request_id="req_001", session_id="sess_001"
  )
  tracker.record_batch([_v2_payload()])
  legacy = {
    **_base_payload("evt_v1"),
    "execution_context_id": "33333333-3333-4333-8333-333333333333",
    "workflow_run_id": "44444444-4444-4444-8444-444444444444",
  }
  with pytest.raises(ValueError, match="mix usage schema versions"):
    tracker.record_batch([legacy])
  with pytest.raises(ValueError, match="attempt identity mismatch"):
    tracker.record_batch([{
      **_v2_payload("evt_v2_other"),
      "work_authorization_id": "66666666-6666-4666-8666-666666666666",
    }])


def test_reconciliation_rejects_a_batch_atomically() -> None:
  tracker = CommercialUsageReconciliationTracker(
    request_id="req_001", session_id="sess_001"
  )
  first = _v2_payload()
  drifting_second = {
    **_v2_payload("evt_v2_drift"),
    "work_authorization_id": "66666666-6666-4666-8666-666666666666",
  }

  with pytest.raises(ValueError, match="attempt identity mismatch"):
    tracker.record_batch([first, drifting_second])
  with pytest.raises(ValueError, match="source lineage is unavailable"):
    tracker.compare(_summary(usage_event_ids=("evt_v2_001",)))

  tracker.record_batch([first])
  report = tracker.compare(_summary(usage_event_ids=("evt_v2_001",)))
  assert report.commercial_event_count == 1
  assert report.observed_source_event_ids == ("evt_v2_001",)
