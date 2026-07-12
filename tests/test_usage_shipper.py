from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import random
import sqlite3

import httpx
import pytest

from agent_gateway import usage_outbox as usage_outbox_module
from agent_gateway.commercial_contract import canonical_usage_payload_sha256
from agent_gateway.multi_user.billing import SessionUsageSummary
from agent_gateway.usage_outbox import CommercialUsageOutbox
from agent_gateway.usage_reconciliation import CommercialUsageReconciliationTracker
from agent_gateway.usage_shipper import (
  CommercialUsageReconciliationIngestClient,
  CommercialUsageReconciliationShipper,
  CommercialUsageIngestClient,
  CommercialUsagePermanentDeliveryError,
  CommercialUsageResponseError,
  CommercialUsageShipper,
  CommercialUsageShipperConfig,
  UsageAcceptance,
  _signature_message,
  ReconciliationAcceptance,
)


NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)
SECRET = b"s" * 32


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


def _record_reconciliation(
  outbox: CommercialUsageOutbox,
  *,
  request_id: str = "req_001",
  session_id: str = "sess_001",
  event_id: str = "evt_001",
):
  payload = _payload(event_id)
  payload["request_id"] = request_id
  payload["session_id"] = session_id
  payload["source_payload_sha256"] = canonical_usage_payload_sha256(payload)
  tracker = CommercialUsageReconciliationTracker(
    request_id=request_id, session_id=session_id
  )
  tracker.record_batch([payload])
  summary = SessionUsageSummary(
    user_id="user-001", session_id=session_id, request_id=request_id,
    input_tokens=10, output_tokens=2, cache_read_tokens=0,
    cache_creation_tokens=0, cost=0.001, turns=1, channel="mcp",
    started_at=NOW.timestamp() - 60, ended_at=NOW.timestamp(),
    usage_event_count=1, usage_event_ids=(event_id,),
  )
  report, _ = outbox.record_reconciliation_report(tracker.compare(summary), recorded_at=NOW)
  return tracker, summary, report


def test_ingest_client_signs_exact_body_and_validates_complete_response() -> None:
  observed = {}

  async def handler(request: httpx.Request) -> httpx.Response:
    body = request.content
    observed["body"] = body
    observed["headers"] = request.headers
    return httpx.Response(200, json={"results": [{
      "environment": "prod", "source_event_id": "evt_001", "status": "accepted",
      "canonical_event_id": "canonical_1", "reason_code": None,
    }]})

  async def exercise():
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
      client = CommercialUsageIngestClient(
        base_url="https://risk.internal", key_id="usage-v1", secret=SECRET,
        environment="prod", http_client=http_client, now=lambda: NOW,
        nonce=lambda: "nonce-001",
      )
      return await client.send_batch([_payload("evt_001")])

  results = asyncio.run(exercise())
  assert results[0].status == "accepted"
  headers = observed["headers"]
  message = _signature_message(
    method="POST", path="/internal/commercial/usage-events:batch",
    timestamp=str(int(NOW.timestamp())), nonce="nonce-001", body=observed["body"],
  )
  expected = hmac.new(SECRET, message, hashlib.sha256).hexdigest()
  assert headers["x-hank-request-signature"] == f"v1={expected}"
  assert json.loads(observed["body"])["events"][0]["source_event_id"] == "evt_001"


def test_reconciliation_client_signs_separate_target_and_validates_receipt(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  _, _, report = _record_reconciliation(outbox)
  observed = {}

  async def handler(request: httpx.Request) -> httpx.Response:
    observed["request"] = request
    return httpx.Response(200, json={"results": [{
      "schema_version": 1,
      "environment": report.environment,
      "source_product": report.source_product,
      "request_id": report.request_id,
      "session_id": report.session_id,
      "evidence_revision": report.revision,
      "report_sha256": report.report_sha256,
      "status": "accepted",
      "reason_code": None,
    }]})

  async def exercise():
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
      client = CommercialUsageReconciliationIngestClient(
        base_url="https://risk.internal", key_id="usage-v1", secret=SECRET,
        environment="prod", http_client=http_client, now=lambda: NOW,
        nonce=lambda: "nonce-reconciliation",
      )
      return await client.send_batch([{
        "report_sha256": report.report_sha256,
        "evidence": report.payload,
      }])

  results = asyncio.run(exercise())
  assert results[0].status == "accepted"
  request = observed["request"]
  assert request.url.path == "/internal/commercial/usage-reconciliation:batch"
  expected = hmac.new(
    SECRET,
    _signature_message(
      method="POST", path=request.url.path, timestamp=str(int(NOW.timestamp())),
      nonce="nonce-reconciliation", body=request.content,
    ),
    hashlib.sha256,
  ).hexdigest()
  assert request.headers["x-hank-request-signature"] == f"v1={expected}"


def test_reconciliation_client_rejects_non_string_reason_code(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  _, _, report = _record_reconciliation(outbox)

  async def handler(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"results": [{
      "schema_version": 1,
      "environment": report.environment,
      "source_product": report.source_product,
      "request_id": report.request_id,
      "session_id": report.session_id,
      "evidence_revision": report.revision,
      "report_sha256": report.report_sha256,
      "status": "accepted",
      "reason_code": 123,
    }]})

  async def exercise():
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
      client = CommercialUsageReconciliationIngestClient(
        base_url="https://risk.internal", key_id="usage-v1", secret=SECRET,
        environment="prod", http_client=http_client, now=lambda: NOW,
        nonce=lambda: "nonce-bad-reason",
      )
      await client.send_batch([{
        "report_sha256": report.report_sha256,
        "evidence": report.payload,
      }])

  with pytest.raises(CommercialUsageResponseError, match="identity/status"):
    asyncio.run(exercise())


@pytest.mark.parametrize("results", [[], [
  {"environment": "prod", "source_event_id": "other", "status": "accepted", "canonical_event_id": "c"}
], [
  {"environment": "prod", "source_event_id": "evt_001", "status": "accepted", "canonical_event_id": "c"},
  {"environment": "prod", "source_event_id": "evt_001", "status": "duplicate", "canonical_event_id": "c"},
]])
def test_ingest_client_rejects_partial_unknown_or_duplicate_results(results) -> None:
  async def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"results": results})

  async def exercise():
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
      client = CommercialUsageIngestClient(
        base_url="https://risk.internal", key_id="usage-v1", secret=SECRET,
        environment="prod", http_client=http_client,
      )
      await client.send_batch([_payload("evt_001")])

  with pytest.raises(CommercialUsageResponseError):
    asyncio.run(exercise())


class _Sender:
  def __init__(self, results=None, error: Exception | None = None):
    self.results = results
    self.error = error
    self.payloads = []

  async def send_batch(self, payloads):
    self.payloads.append(payloads)
    if self.error:
      raise self.error
    return self.results


def test_shipper_maps_acceptance_duplicate_conflict_retry_and_terminal(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  ids = ["accepted", "duplicate", "conflict", "retry", "terminal"]
  outbox.enqueue_batch([_payload(event_id) for event_id in ids], created_at=NOW)
  statuses = ["accepted", "duplicate", "conflict", "rejected_retryable", "rejected_terminal"]
  sender = _Sender([
    UsageAcceptance(
      "prod", event_id, status,
      "canonical" if status in {"accepted", "duplicate"} else None,
      "unknown_rate_version" if status == "rejected_retryable" else "reason",
    )
    for event_id, status in zip(ids, statuses)
  ])
  metrics = []
  shipper = CommercialUsageShipper(
    outbox=outbox, sender=sender,
    config=CommercialUsageShipperConfig(batch_size=10, jitter_ratio=0),
    metric=lambda name, value: metrics.append((name, value)),
  )

  assert asyncio.run(shipper.run_once(now=NOW)) == 5
  assert outbox.get("accepted").state == "accepted"
  assert outbox.get("duplicate").state == "accepted"
  assert outbox.get("conflict").state == "dead"
  assert outbox.get("terminal").state == "dead"
  assert outbox.get("retry").state == "retryable"
  assert outbox.get("retry").next_attempt_at == "2026-07-11T12:00:01.000000Z"
  assert (outbox.get("accepted").ingest_status,
          outbox.get("accepted").canonical_event_id) == ("accepted", "canonical")
  assert outbox.get("duplicate").ingest_status == "duplicate"
  assert outbox.get("conflict").ingest_status == "conflict"
  assert outbox.get("conflict").ingest_reason_code == "reason"
  assert outbox.get("terminal").ingest_status == "rejected_terminal"
  assert outbox.get("retry").ingest_status is None
  assert ("commercial_usage.backlog", 5) in metrics
  assert ("commercial_usage.oldest_backlog_age_seconds", 0) in metrics
  assert ("commercial_usage.ingest_lag_seconds", 60) in metrics
  assert ("commercial_usage.rejected_retryable", 1) in metrics
  assert ("commercial_usage.rejected_terminal", 1) in metrics
  assert ("commercial_usage.unknown_rate_version", 1) in metrics


def test_transport_error_retries_indefinitely_with_exponential_bounded_jitter(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  outbox.enqueue_batch([_payload("evt_001")], created_at=NOW)
  sender = _Sender(error=TimeoutError("offline"))
  shipper = CommercialUsageShipper(
    outbox=outbox, sender=sender,
    config=CommercialUsageShipperConfig(
      batch_size=1, base_backoff_seconds=2, max_backoff_seconds=5,
      jitter_ratio=0, max_event_attempts=1,
    ),
    random_source=random.Random(1),
  )

  asyncio.run(shipper.run_once(now=NOW))
  first = outbox.get("evt_001")
  assert first.state == "retryable"
  assert first.next_attempt_at == "2026-07-11T12:00:02.000000Z"
  asyncio.run(shipper.run_once(now=NOW + timedelta(seconds=2)))
  second = outbox.get("evt_001")
  assert second.state == "retryable"
  assert second.next_attempt_at == "2026-07-11T12:00:06.000000Z"


def test_retryable_result_moves_to_dead_after_attempt_limit(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  outbox.enqueue_batch([_payload("evt_001")], created_at=NOW)
  sender = _Sender([UsageAcceptance(
    "prod", "evt_001", "rejected_retryable", None, "unknown_rate"
  )])
  shipper = CommercialUsageShipper(
    outbox=outbox, sender=sender,
    config=CommercialUsageShipperConfig(
      batch_size=1, base_backoff_seconds=1, max_backoff_seconds=1,
      jitter_ratio=0, max_event_attempts=2,
    ),
  )
  asyncio.run(shipper.run_once(now=NOW))
  asyncio.run(shipper.run_once(now=NOW + timedelta(seconds=1)))
  row = outbox.get("evt_001")
  assert row.state == "dead"
  assert row.attempt_count == 2
  assert row.last_error.startswith("attempts_exhausted")


def test_stale_lease_fence_prevents_late_shipper_completion(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  outbox.enqueue_batch([_payload("evt_001")], created_at=NOW)
  stale = outbox.lease_batch(limit=1, lease_for=timedelta(seconds=1), now=NOW)[0]
  fresh = outbox.lease_batch(
    limit=1, lease_for=timedelta(seconds=10), now=NOW + timedelta(seconds=2)
  )[0]
  assert outbox.mark_accepted(
    "evt_001", stale.sending_lease_token, ingest_status="accepted",
    canonical_event_id="canonical-001", accepted_at=NOW,
  ) is False
  assert outbox.mark_accepted(
    "evt_001", fresh.sending_lease_token, ingest_status="accepted",
    canonical_event_id="canonical-001", accepted_at=NOW,
  ) is True


def test_shipper_quarantines_ambiguous_sender_results_for_retry(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  outbox.enqueue_batch([_payload("evt_001")], created_at=NOW)
  duplicate = UsageAcceptance("prod", "evt_001", "accepted", "canonical", None)
  shipper = CommercialUsageShipper(
    outbox=outbox, sender=_Sender([duplicate, duplicate]),
    config=CommercialUsageShipperConfig(jitter_ratio=0),
  )

  asyncio.run(shipper.run_once(now=NOW))

  row = outbox.get("evt_001")
  assert row.state == "retryable"
  assert row.last_error.startswith("ambiguous_response")


def test_permanent_poison_event_isolated_without_blocking_later_valid_row(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  outbox.enqueue_batch([_payload("poison"), _payload("valid")], created_at=NOW)

  class IsolatingSender:
    async def send_batch(self, payloads):
      if any(payload["source_event_id"] == "poison" for payload in payloads):
        raise CommercialUsagePermanentDeliveryError("body too large")
      return [UsageAcceptance(
        "prod", payload["source_event_id"], "accepted", "canonical", None
      ) for payload in payloads]

  shipper = CommercialUsageShipper(
    outbox=outbox, sender=IsolatingSender(),
    config=CommercialUsageShipperConfig(batch_size=10, jitter_ratio=0),
  )

  asyncio.run(shipper.run_once(now=NOW))

  assert outbox.get("poison").state == "dead"
  assert outbox.get("valid").state == "accepted"


def test_reconciliation_shipper_preserves_revision_order_and_terminal_receipts(
  tmp_path,
) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  tracker, summary, first = _record_reconciliation(outbox)
  tracker.mark_late("evt_001")
  second, _ = outbox.record_reconciliation_report(
    tracker.compare(summary), recorded_at=NOW + timedelta(seconds=1)
  )

  class Sender:
    def __init__(self) -> None:
      self.calls = []

    async def send_batch(self, reports):
      self.calls.append(reports)
      evidence = reports[0]["evidence"]
      status = "accepted" if evidence["evidence_revision"] == 1 else "conflict"
      return [ReconciliationAcceptance(
        environment=evidence["environment"],
        source_product=evidence["source_product"],
        request_id=evidence["request_id"],
        session_id=evidence["session_id"],
        evidence_revision=evidence["evidence_revision"],
        report_sha256=reports[0]["report_sha256"],
        status=status,
        reason_code=(
          None if status == "accepted"
          else "usage_reconciliation.revision_conflict"
        ),
      )]

  sender = Sender()
  shipper = CommercialUsageReconciliationShipper(
    outbox=outbox, sender=sender,
    config=CommercialUsageShipperConfig(batch_size=10, jitter_ratio=0),
  )
  assert asyncio.run(shipper.run_once(now=NOW + timedelta(seconds=2))) == 1
  assert sender.calls[0][0]["report_sha256"] == first.report_sha256
  assert asyncio.run(shipper.run_once(now=NOW + timedelta(seconds=3))) == 1
  assert sender.calls[1][0]["report_sha256"] == second.report_sha256
  assert outbox.health()["reconciliation_shipment_counts"] == {
    "accepted": 1,
    "dead": 1,
  }


def test_reconciliation_shipper_isolates_malformed_local_envelope(tmp_path) -> None:
  outbox = CommercialUsageOutbox(tmp_path / "usage.sqlite3")
  _, _, poison = _record_reconciliation(outbox)
  _, _, valid = _record_reconciliation(
    outbox, request_id="req_002", session_id="sess_002", event_id="evt_002"
  )

  class Sender:
    def __init__(self) -> None:
      self.calls = []

    async def send_batch(self, reports):
      self.calls.append(reports)
      evidence = reports[0]["evidence"]
      return [ReconciliationAcceptance(
        environment=evidence["environment"],
        source_product=evidence["source_product"],
        request_id=evidence["request_id"],
        session_id=evidence["session_id"],
        evidence_revision=evidence["evidence_revision"],
        report_sha256=reports[0]["report_sha256"],
        status="accepted",
        reason_code=None,
      )]

  sender = Sender()
  shipper = CommercialUsageReconciliationShipper(
    outbox=outbox, sender=sender,
    config=CommercialUsageShipperConfig(batch_size=10, jitter_ratio=0),
  )
  with sqlite3.connect(outbox.path) as connection:
    connection.execute("DROP TRIGGER trg_commercial_usage_reconciliation_no_update")
    connection.execute(
      "UPDATE commercial_usage_reconciliation_reports SET payload_json = 'not-json' "
      "WHERE report_id = ?",
      (poison.report_id,),
    )
    connection.execute(
      usage_outbox_module._REPORT_IMMUTABLE_TRIGGERS[
        "trg_commercial_usage_reconciliation_no_update"
      ]
    )

  assert asyncio.run(shipper.run_once(now=NOW + timedelta(seconds=1))) == 2
  assert len(sender.calls) == 1
  assert sender.calls[0][0]["report_sha256"] == valid.report_sha256
  assert outbox.health()["reconciliation_shipment_counts"] == {
    "accepted": 1,
    "dead": 1,
  }
