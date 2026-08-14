from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import multiprocessing
import os

from agent_gateway.commercial_contract import canonical_usage_payload_sha256
from agent_gateway.usage_outbox import CommercialUsageOutbox
from agent_gateway.usage_shipper import (
  CommercialUsageShipper,
  CommercialUsageShipperConfig,
  UsageAcceptance,
)


NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)


def _payload(event_id: str) -> dict:
  # Shippable version: the outbox ship gate dead-letters anything but v3.
  payload = {
    "schema_version": 3, "source_product": "hank-agent-gateway",
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


def _enqueue_then_exit(path: str, payload: dict) -> None:
  CommercialUsageOutbox(path).enqueue_batch([payload], created_at=NOW)
  os._exit(0)


def _lease_then_exit(path: str) -> None:
  CommercialUsageOutbox(path).lease_batch(
    limit=1, lease_for=timedelta(seconds=1), now=NOW
  )
  os._exit(0)


def _accept_then_exit(path: str, event_id: str, token: str) -> None:
  CommercialUsageOutbox(path).mark_accepted(
    event_id, token, ingest_status="accepted",
    canonical_event_id=f"canonical-{event_id}", accepted_at=NOW,
  )
  os._exit(0)


def _post_then_die_before_response(path: str, remote_path: str) -> None:
  class Sender:
    async def send_batch(self, payloads):
      descriptor = os.open(remote_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
      try:
        body = (json.dumps({"source_event_id": payloads[0]["source_event_id"]}) + "\n").encode()
        os.write(descriptor, body)
        os.fsync(descriptor)
      finally:
        os.close(descriptor)
      os._exit(23)

  shipper = CommercialUsageShipper(
    outbox=CommercialUsageOutbox(path), sender=Sender(),
    config=CommercialUsageShipperConfig(
      batch_size=1, lease_seconds=1, jitter_ratio=0,
    ),
  )
  asyncio.run(shipper.run_once(now=NOW))


def _run_process(target, *args, expected_exit=0) -> None:
  process = multiprocessing.get_context("fork").Process(target=target, args=args)
  process.start()
  process.join(timeout=10)
  assert process.exitcode == expected_exit


def test_process_kill_after_insert_and_lease_loses_no_event(tmp_path) -> None:
  path = str(tmp_path / "usage.sqlite3")
  _run_process(_enqueue_then_exit, path, _payload("evt_001"))
  outbox = CommercialUsageOutbox(path)
  assert outbox.get("evt_001").state == "pending"

  _run_process(_lease_then_exit, path)
  assert outbox.get("evt_001").state == "sending"
  reclaimed = outbox.lease_batch(
    limit=1, lease_for=timedelta(seconds=10), now=NOW + timedelta(seconds=2)
  )[0]
  assert reclaimed.event_id == "evt_001"
  assert reclaimed.attempt_count == 2


def test_process_kill_after_remote_post_before_response_does_not_double_ingest(tmp_path) -> None:
  path = str(tmp_path / "usage.sqlite3")
  remote = tmp_path / "remote.jsonl"
  outbox = CommercialUsageOutbox(path)
  outbox.enqueue_batch([_payload("evt_001")], created_at=NOW)
  _run_process(_post_then_die_before_response, path, str(remote), expected_exit=23)
  assert outbox.get("evt_001").state == "sending"
  assert len(remote.read_text(encoding="utf-8").splitlines()) == 1

  class DuplicateSender:
    async def send_batch(self, payloads):
      return [UsageAcceptance("prod", "evt_001", "duplicate", "canonical_1", None)]

  shipper = CommercialUsageShipper(
    outbox=outbox, sender=DuplicateSender(),
    config=CommercialUsageShipperConfig(batch_size=1, lease_seconds=10, jitter_ratio=0),
  )
  asyncio.run(shipper.run_once(now=NOW + timedelta(seconds=2)))
  assert outbox.get("evt_001").state == "accepted"
  assert len(remote.read_text(encoding="utf-8").splitlines()) == 1


def test_process_kill_after_accept_transition_remains_accepted(tmp_path) -> None:
  path = str(tmp_path / "usage.sqlite3")
  outbox = CommercialUsageOutbox(path)
  outbox.enqueue_batch([_payload("evt_001")], created_at=NOW)
  leased = outbox.lease_batch(limit=1, lease_for=timedelta(seconds=10), now=NOW)[0]
  _run_process(
    _accept_then_exit, path, leased.event_id, leased.sending_lease_token
  )
  assert outbox.get("evt_001").state == "accepted"
  assert outbox.lease_batch(
    limit=1, lease_for=timedelta(seconds=10), now=NOW + timedelta(seconds=20)
  ) == []
