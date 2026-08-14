from __future__ import annotations

import ast
import importlib
from pathlib import Path
from uuid import uuid4

import pytest
from agent_workflow_contracts import CapabilityBind

from agent_gateway.commercial_claims import VerifiedCommercialClaim
from agent_gateway.commercial_work_authorization import VerifiedWorkAuthorization
from agent_gateway.commercial_work_start import CommercialWorkStartContext
from agent_gateway.commercial_contract import canonical_usage_payload_sha256
from agent_gateway.commercial_usage import (
  CommercialUsageLineage,
  CommercialUsageProducer,
)
from agent_gateway.multi_user.billing import SessionUsageSummary, UsageEvent
from agent_gateway.usage_reconciliation import CommercialUsageReconciliationTracker
from agent_gateway.runner_usage import call_usage_event_hook
from agent_gateway.providers import ModelInfo, ModelProvider, StreamEvent
from agent_gateway.usage_outbox import CommercialUsageOutboxError
from agent_gateway.usage_resilience import (
  CommercialUsageCircuitOpen,
  CommercialUsageDurability,
)
from tests.capability_execution_test_support import stub_bound_capability_execution


send_prompt_module = importlib.import_module("agent_gateway.send_prompt")


NOW = 1_780_000_000


def _claim() -> VerifiedCommercialClaim:
  return VerifiedCommercialClaim(
    schema_version=1,
    key_id="commercial-signing-v1",
    subject="user:123",
    environment="prod",
    surface="hp1",
    commercial_account_id=uuid4(),
    agreement_id=uuid4(),
    agreement_terms_revision=2,
    offer_code="hp1_pro",
    effective_scopes=("read",),
    entitlement_revision=42,
    payer_policy_version="hp1_customer_host@v1",
    budget_policy_version="hp1_pro_budget@v1",
    shadow_rate_version="commercial_rates@2026-09-01",
    manifest_version="mcp_exposure@2026-09-01",
    authorized_work_start_deadline=NOW + 300,
    usage_accept_until=NOW + 3600,
    issued_at=NOW,
    expires_at=NOW + 300,
    context_id=uuid4(),
  )


def _lineage(**changes) -> CommercialUsageLineage:
  facts = {
    "source_product": "hank-agent-gateway",
    "workflow_run_id": "wf_001",
    "reservation_id": "res_001",
    "funding_route_id": "fund_001",
    "operation": "messages.create",
    "capability_id": "portfolio.review",
  }
  facts.update(changes)
  return CommercialUsageLineage(**facts)


def _event(**changes) -> UsageEvent:
  capability_bind = CapabilityBind(
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
  facts = {
    "user_id": "123",
    "session_id": "sess_001",
    "request_id": "req_001",
    "parent_turn_id": "turn_001",
    "timestamp": float(NOW),
    "model": "claude-sonnet-test",
    "provider": "anthropic",
    "capability_bind": capability_bind,
    "provider_reported_model": None,
    "input_tokens": 1000,
    "output_tokens": 200,
    "reasoning_tokens_observed": 50,
    "cache_read_tokens": 300,
    "cache_creation_tokens": 100,
    "cost_usd": 0.0125,
    "rate_table_version": "anthropic-test-v1",
    "billing_mode": "metered",
    "channel": "mcp",
    "event_id": "evt_001",
  }
  facts.update(changes)
  return UsageEvent(**facts)


def _work_start(*, attempt_number: int = 1) -> CommercialWorkStartContext:
  claim = _claim()
  workflow_run_id = uuid4()
  attempt_group_id = workflow_run_id if attempt_number == 1 else uuid4()
  retry_of = None if attempt_number == 1 else uuid4()
  authorization = VerifiedWorkAuthorization(
    schema_version=1,
    key_id="work-signing-v1",
    token_sha256="sha256:" + "a" * 64,
    authorization_id=uuid4(),
    environment="prod",
    execution_context_id=claim.context_id,
    workflow_run_id=workflow_run_id,
    workflow_attempt_group_id=attempt_group_id,
    workflow_attempt_number=attempt_number,
    retry_of_workflow_run_id=retry_of,
    workflow_attempt_kind="initial" if attempt_number == 1 else "automatic_retry",
    primary_inference_observability="hank_metered",
    funding_route_id=uuid4(),
    provider="anthropic",
    billing_mode="metered",
    reservation_id=uuid4(),
    operation="messages.create",
    capability_id="portfolio.review",
    request_id="req_001",
    session_id="sess_001",
    issued_at=NOW,
    expires_at=NOW + 120,
  )
  return CommercialWorkStartContext(
    claim=claim,
    authorization=authorization,
    consumption=object(),
  )


@pytest.mark.asyncio
async def test_producer_emits_complete_canonical_delta_without_reasoning_double_count() -> None:
  emitted = []
  producer = CommercialUsageProducer(
    enabled=True, claim=_claim(), lineage=_lineage(), sink=emitted.extend
  )
  payload = await producer.emit(_event())

  assert payload is not None
  assert emitted == [payload]
  assert payload["schema_version"] == 3
  assert payload["capability_bind"] == _event().capability_bind
  assert payload["provider"] == payload["capability_bind"]["provider"]
  assert payload["model"] == payload["capability_bind"]["upstream_model"]
  assert payload["capability_id"] == payload["capability_bind"]["capability_id"]
  assert payload["provider_reported_model"] is None
  assert all(payload[field] is None for field in (
    "workflow_attempt_group_id",
    "workflow_attempt_number",
    "retry_of_workflow_run_id",
    "workflow_attempt_kind",
    "work_authorization_id",
  ))
  assert payload["uncached_input_tokens"] == 1000
  assert payload["billable_output_tokens"] == 200
  assert payload["reasoning_tokens_observed"] == 50
  assert payload["source_payload_sha256"] == canonical_usage_payload_sha256(payload)
  assert payload["producer_estimated_cost_usd"] == "0.0125"
  assert "commercial_account_id" not in payload
  assert "payer_class" not in payload


@pytest.mark.asyncio
async def test_producer_keeps_reported_model_distinct_from_exact_bind() -> None:
  producer = CommercialUsageProducer(
    enabled=True, claim=_claim(), lineage=_lineage(), sink=lambda _events: None
  )
  payload = await producer.emit(
    _event(provider_reported_model="claude-sonnet-test-20260801")
  )

  assert payload is not None
  assert payload["provider_reported_model"] == "claude-sonnet-test-20260801"
  assert payload["model"] == "claude-sonnet-test"
  assert payload["capability_bind"]["upstream_model"] == "claude-sonnet-test"


@pytest.mark.asyncio
async def test_producer_rejects_capability_projection_drift() -> None:
  producer = CommercialUsageProducer(
    enabled=True,
    claim=_claim(),
    lineage=_lineage(capability_id="different.capability"),
    sink=lambda _events: None,
  )
  with pytest.raises(ValueError, match="capability projection differs"):
    await producer.emit(_event())


@pytest.mark.asyncio
async def test_verified_work_start_emits_attempt_authoritative_usage_v3() -> None:
  emitted = []
  work_start = _work_start(attempt_number=2)
  producer = CommercialUsageProducer(
    enabled=True,
    claim=None,
    lineage=None,
    sink=emitted.extend,
    work_start=work_start,
  )

  payload = await producer.emit(_event())

  assert payload is not None
  authorization = work_start.authorization
  assert payload["schema_version"] == 3
  assert payload["capability_bind"] == _event().capability_bind
  assert payload["provider_reported_model"] is None
  assert payload["workflow_run_id"] == str(authorization.workflow_run_id)
  assert payload["workflow_attempt_group_id"] == str(
    authorization.workflow_attempt_group_id
  )
  assert payload["workflow_attempt_number"] == 2
  assert payload["retry_of_workflow_run_id"] == str(
    authorization.retry_of_workflow_run_id
  )
  assert payload["workflow_attempt_kind"] == "automatic_retry"
  assert payload["work_authorization_id"] == str(authorization.authorization_id)
  assert payload["source_payload_sha256"] == canonical_usage_payload_sha256(payload)
  assert emitted == [payload]


@pytest.mark.asyncio
@pytest.mark.parametrize(
  "event_changes",
  (
    {"request_id": "other-request"},
    {"session_id": "other-session"},
    {"provider": "openai"},
    {"billing_mode": "byok"},
  ),
)
async def test_usage_v3_rejects_drift_from_verified_work_start(event_changes) -> None:
  producer = CommercialUsageProducer(
    enabled=True,
    claim=None,
    lineage=None,
    sink=lambda _events: None,
    work_start=_work_start(),
  )
  with pytest.raises(ValueError, match="differs"):
    await producer.emit(_event(**event_changes))


@pytest.mark.asyncio
async def test_default_off_is_noop_and_hank_funded_lineage_fails_closed() -> None:
  disabled = CommercialUsageProducer(enabled=False, claim=None, lineage=None, sink=None)
  assert await disabled.emit(_event()) is None

  producer = CommercialUsageProducer(
    enabled=True,
    claim=_claim(),
    lineage=_lineage(reservation_id=None),
    sink=lambda _: None,
  )
  with pytest.raises(ValueError, match="requires reservation lineage"):
    await producer.emit(_event())


@pytest.mark.asyncio
async def test_reasoning_cache_and_sink_failures_are_not_silently_dropped() -> None:
  async def failed_sink(_):
    raise OSError("durable sink unavailable")

  producer = CommercialUsageProducer(
    enabled=True, claim=_claim(), lineage=_lineage(), sink=failed_sink
  )
  with pytest.raises(ValueError, match="informational output subset"):
    await producer.emit(_event(reasoning_tokens_observed=201))
  with pytest.raises(ValueError, match="cannot be negative"):
    await producer.emit(_event(input_tokens=-1))
  with pytest.raises(OSError, match="durable sink unavailable"):
    await producer.emit(_event())


@pytest.mark.asyncio
async def test_source_identity_and_provider_reported_cost_semantics() -> None:
  producer = CommercialUsageProducer(
    enabled=True, claim=_claim(), lineage=_lineage(), sink=lambda _: None
  )

  with pytest.raises(ValueError, match="source and request identity"):
    await producer.emit(_event(request_id=None))

  payload = await producer.emit(
    _event(
      provider_reported_cost_usd="0.0100",
      separately_billed_tool_cost_usd="0.0025",
      provider_units="2",
      is_batch=True,
    )
  )
  assert payload is not None
  assert payload["provider_reported_cost_usd"] == "0.0100"
  assert payload["cost_observation_kind"] == "provider_response"
  assert payload["separately_billed_tool_cost_usd"] == "0.0025"
  assert payload["provider_units"] == "2"
  assert payload["is_batch"] is True


@pytest.mark.asyncio
async def test_typed_provider_units_are_one_atomic_sink_batch() -> None:
  batches = []
  producer = CommercialUsageProducer(
    enabled=True, claim=_claim(), lineage=_lineage(), sink=batches.append
  )
  await producer.emit(_event(provider_unit_deltas={"web_search": 2, "web_fetch": 1}))

  assert len(batches) == 1
  assert [item["operation"] for item in batches[0]] == [
    "messages.create", "web_fetch", "web_search",
  ]
  with pytest.raises(ValueError, match="allocated, not aggregate"):
    await producer.emit(_event(
      provider_unit_deltas={"web_search": 1},
      separately_billed_tool_cost_usd="0.01",
    ))
  with pytest.raises(ValueError, match="cannot coexist"):
    await producer.emit(_event(
      provider_units="2", provider_unit_deltas={"web_search": 1},
    ))


@pytest.mark.asyncio
async def test_resilient_durability_constructs_request_scoped_producer(tmp_path) -> None:
  durability = CommercialUsageDurability.create(
    outbox_path=tmp_path / "commercial.sqlite3",
    spool_path=tmp_path / "commercial.spool",
    circuit_state_path=tmp_path / "commercial.circuit",
    max_backlog=100,
    max_storage_bytes=10_000_000,
  )
  producer = durability.producer(claim=_claim(), lineage=_lineage())

  payload = await producer.emit(_event())

  assert payload is not None
  stored = durability.outbox.get(payload["source_event_id"])
  assert stored is not None
  assert stored.state == "pending"
  assert stored.payload == payload

  with pytest.raises(CommercialUsageOutboxError, match="must use CommercialUsageDurability"):
    durability.outbox.producer(claim=_claim(), lineage=_lineage())


@pytest.mark.asyncio
async def test_resilient_durability_persists_verified_usage_v3(tmp_path) -> None:
  durability = CommercialUsageDurability.create(
    outbox_path=tmp_path / "commercial.sqlite3",
    spool_path=tmp_path / "commercial.spool",
    circuit_state_path=tmp_path / "commercial.circuit",
    max_backlog=100,
    max_storage_bytes=10_000_000,
  )
  work_start = _work_start(attempt_number=2)
  producer = durability.producer(work_start=work_start)

  payload = await producer.emit(_event())

  assert payload is not None
  stored = durability.outbox.get(payload["source_event_id"])
  assert stored is not None
  assert stored.payload["schema_version"] == 3
  assert stored.payload["workflow_attempt_number"] == 2
  assert stored.payload["workflow_attempt_group_id"] == str(
    work_start.authorization.workflow_attempt_group_id
  )
  assert stored.payload["work_authorization_id"] == str(
    work_start.authorization.authorization_id
  )


def test_commercial_producer_delegates_pre_spend_work_guard() -> None:
  class GuardedSink:
    def __call__(self, payloads):
      raise AssertionError("sink should not be called")

    def assert_work_allowed(self, billing_mode):
      raise CommercialUsageCircuitOpen(f"blocked:{billing_mode}")

  producer = CommercialUsageProducer(
    enabled=True, claim=_claim(), lineage=_lineage(), sink=GuardedSink()
  )
  with pytest.raises(CommercialUsageCircuitOpen, match="blocked:metered"):
    producer.assert_work_allowed("metered")


@pytest.mark.asyncio
async def test_session_reconciliation_callback_never_emits_second_cost_event() -> None:
  batches = []
  reports = []
  tracker = CommercialUsageReconciliationTracker(
    request_id="req_001", session_id="sess_001"
  )
  producer = CommercialUsageProducer(
    enabled=True, claim=_claim(), lineage=_lineage(), sink=batches.append,
    reconciliation_tracker=tracker, on_reconciliation=reports.append,
  )
  await producer.emit(_event())
  report = await producer.reconcile(SessionUsageSummary(
    user_id="123", session_id="sess_001", request_id="req_001",
    input_tokens=1000, output_tokens=200, cache_read_tokens=300,
    cache_creation_tokens=100, cost=0.0125, turns=1, channel="mcp",
    started_at=NOW - 1, ended_at=NOW,
    usage_event_count=1, usage_event_ids=("evt_001",),
    capability_bind=_event().capability_bind,
  ))

  assert report.status == "match"
  assert reports == [report]
  assert len(batches) == 1
  assert len(batches[0]) == 1
  late_report = await producer.mark_late("evt_001")
  assert late_report.status == "mismatch"
  assert late_report.late_source_event_ids == ("evt_001",)
  assert reports == [report, late_report]
  assert len(batches) == 1


@pytest.mark.asyncio
async def test_resilient_producer_persists_default_reconciliation_revisions(
  tmp_path,
) -> None:
  durability = CommercialUsageDurability.create(
    outbox_path=tmp_path / "commercial-usage.sqlite3",
    spool_path=tmp_path / "commercial-usage.spool",
    circuit_state_path=tmp_path / "commercial-usage-circuit.json",
    max_backlog=100,
    max_storage_bytes=10_000_000,
  )
  producer = durability.producer(claim=_claim(), lineage=_lineage())
  await producer.emit(_event())
  summary = SessionUsageSummary(
    user_id="123", session_id="sess_001", request_id="req_001",
    input_tokens=1000, output_tokens=200, cache_read_tokens=300,
    cache_creation_tokens=100, cost=0.0125, turns=1, channel="mcp",
    started_at=NOW - 1, ended_at=NOW,
    usage_event_count=1, usage_event_ids=("evt_001",),
    capability_bind=_event().capability_bind,
  )
  report = await producer.reconcile(summary)
  assert report.status == "match"
  first = durability.outbox.current_reconciliation_report(
    environment="prod", source_product="hank-agent-gateway",
    request_id="req_001", session_id="sess_001"
  )
  assert first is not None
  assert first.revision == 1
  assert first.payload["event_lines"][0]["source_event_id"] == "evt_001"

  late = await producer.mark_late("evt_001")
  assert late.status == "mismatch"
  reopened = CommercialUsageDurability.create(
    outbox_path=tmp_path / "commercial-usage.sqlite3",
    spool_path=tmp_path / "commercial-usage.spool",
    circuit_state_path=tmp_path / "commercial-usage-circuit.json",
    max_backlog=100,
    max_storage_bytes=10_000_000,
  ).outbox.current_reconciliation_report(
    environment="prod", source_product="hank-agent-gateway",
    request_id="req_001", session_id="sess_001"
  )
  assert reopened is not None
  assert reopened.revision == 2
  assert reopened.supersedes_report_sha256 == first.report_sha256
  assert reopened.payload["late_source_event_ids"] == ["evt_001"]


@pytest.mark.asyncio
async def test_resilient_producer_persists_reconciliation_with_custom_observer(
  tmp_path,
) -> None:
  observed = []
  durability = CommercialUsageDurability.create(
    outbox_path=tmp_path / "commercial-usage.sqlite3",
    spool_path=tmp_path / "commercial-usage.spool",
    circuit_state_path=tmp_path / "commercial-usage-circuit.json",
    max_backlog=100,
    max_storage_bytes=10_000_000,
  )
  producer = durability.producer(
    claim=_claim(), lineage=_lineage(), on_reconciliation=observed.append,
  )
  await producer.emit(_event())
  report = await producer.reconcile(SessionUsageSummary(
    user_id="123", session_id="sess_001", request_id="req_001",
    input_tokens=1000, output_tokens=200, cache_read_tokens=300,
    cache_creation_tokens=100, cost=0.0125, turns=1, channel="mcp",
    started_at=NOW - 1, ended_at=NOW,
    usage_event_count=1, usage_event_ids=("evt_001",),
    capability_bind=_event().capability_bind,
  ))

  assert observed == [report]
  assert durability.outbox.current_reconciliation_report(
    environment="prod", source_product="hank-agent-gateway",
    request_id="req_001", session_id="sess_001",
  ) is not None


@pytest.mark.asyncio
async def test_disabled_producer_reconciliation_is_a_noop() -> None:
  producer = CommercialUsageProducer(
    enabled=False, claim=None, lineage=None, sink=None,
  )
  summary = SessionUsageSummary(
    user_id="123", session_id="sess_001", request_id="req_001",
    input_tokens=0, output_tokens=0, cache_read_tokens=0,
    cache_creation_tokens=0, cost=0, turns=0, channel="mcp",
    started_at=NOW - 1, ended_at=NOW,
  )

  assert await producer.reconcile(summary) is None


@pytest.mark.asyncio
async def test_reconciliation_persistence_failure_trips_future_work_without_retrying_result(
  tmp_path, monkeypatch,
) -> None:
  alerts = []
  durability = CommercialUsageDurability.create(
    outbox_path=tmp_path / "commercial-usage.sqlite3",
    spool_path=tmp_path / "commercial-usage.spool",
    circuit_state_path=tmp_path / "commercial-usage-circuit.json",
    max_backlog=100,
    max_storage_bytes=10_000_000,
    alert=lambda code, payload: alerts.append((code, payload)),
  )
  producer = durability.producer(claim=_claim(), lineage=_lineage())
  await producer.emit(_event())
  monkeypatch.setattr(
    durability.outbox,
    "record_reconciliation_report",
    lambda _report: (_ for _ in ()).throw(OSError("disk full")),
  )
  report = await producer.reconcile(SessionUsageSummary(
    user_id="123", session_id="sess_001", request_id="req_001",
    input_tokens=1000, output_tokens=200, cache_read_tokens=300,
    cache_creation_tokens=100, cost=0.0125, turns=1, channel="mcp",
    started_at=NOW - 1, ended_at=NOW,
    usage_event_count=1, usage_event_ids=("evt_001",),
    capability_bind=_event().capability_bind,
  ))
  assert report.status == "match"
  assert durability.circuit_breaker.snapshot.tripped is True
  assert [code for code, _payload in alerts] == [
    "commercial_usage.reconciliation_evidence_failed"
  ]
  with pytest.raises(CommercialUsageCircuitOpen):
    producer.assert_work_allowed("metered")


@pytest.mark.asyncio
async def test_shared_runner_hook_produces_before_legacy_observer() -> None:
  order = []

  class Aggregator:
    async def record(self, _):
      order.append("aggregate")
      return True

  producer = CommercialUsageProducer(
    enabled=True,
    claim=_claim(),
    lineage=_lineage(),
    sink=lambda _: order.append("commercial"),
  )
  await call_usage_event_hook(
    Aggregator(), _event(), is_summary_emitted=lambda: False,
    on_usage=lambda _: order.append("legacy"), on_late_usage_event=None,
    emit_metric=lambda *_: None, dlq_path=None, log_session_id="sess", logger=None,
    commercial_usage_producer=producer,
  )
  assert order == ["commercial", "aggregate", "legacy"]


@pytest.mark.asyncio
async def test_send_prompt_real_producer_emits_metered_canonical_identity() -> None:
  class Provider(ModelProvider):
    name = "anthropic"

    def has_active_credential(self, config):
      return True

    def get_model_info(self, model):
      return ModelInfo(
        id=model,
        provider=self.name,
        supports_thinking=True,
      )

    def create_client(self, config, *, timeout=None):
      return object()

    def build_request_params(self, **kwargs):
      return {}

    async def stream(self, client, params):
      yield StreamEvent(type="message_start", input_tokens=10, cache_read_tokens=2)
      yield StreamEvent(
        type="usage_update", output_tokens=4,
        provider_unit_deltas={"web_search": 2, "web_fetch": 1},
      )
      yield StreamEvent(type="message_end", stop_reason="end_turn")

    def estimate_cost(self, model, input_tokens, output_tokens, **kwargs):
      return type("Cost", (), {"total": 0.0042})()

    async def close_client(self, client, timeout=2.0):
      return None

  emitted = []
  provider = Provider()
  capability_execution = stub_bound_capability_execution(
    provider=provider,
    model="claude-sonnet-4-6",
    effort="none",
    credential_principal="user",
    auth_config={
      "max_tokens": 4096,
      "auth_mode": "api",
      "api_key": "k",
    },
  )
  producer = CommercialUsageProducer(
    enabled=True,
    claim=_claim(),
    lineage=_lineage(capability_id="session.driver"),
    sink=emitted.extend,
  )

  await send_prompt_module.send_prompt(
    "hello",
    capability_execution=capability_execution,
    user_id="alice",
    session_id="sess-1",
    request_id="req-1",
    rate_table_version="rates-v1", billing_mode="metered", channel="mcp",
    commercial_usage_producer=producer,
  )

  assert len(emitted) == 3
  assert emitted[0]["request_id"] == "req-1"
  assert emitted[0]["raw_billing_mode"] == "metered"
  assert emitted[0]["channel"] == "mcp"
  assert emitted[0]["producer_rate_version"] == "rates-v1"
  assert [(event["operation"], event["provider_units"]) for event in emitted[1:]] == [
    ("web_fetch", 1), ("web_search", 2),
  ]
  assert len({event["source_event_id"] for event in emitted}) == 3
  assert all(event["cost_observation_kind"] == "unknown" for event in emitted[1:])
  assert all(event["producer_estimated_cost_usd"] is None for event in emitted[1:])


def _awaits_emit(source: str, receiver: str) -> bool:
  tree = ast.parse(source)
  return any(
    isinstance(node, ast.Await)
    and isinstance(node.value, ast.Call)
    and isinstance(node.value.func, ast.Attribute)
    and node.value.func.attr == "emit"
    and isinstance(node.value.func.value, ast.Name)
    and node.value.func.value.id == receiver
    for node in ast.walk(tree)
  )


def _passes_keyword(source: str, keyword: str) -> bool:
  tree = ast.parse(source)
  return any(
    isinstance(node, ast.Call)
    and any(item.arg == keyword for item in node.keywords)
    for node in ast.walk(tree)
  )


def test_production_usage_path_inventory_reaches_shared_commercial_hook() -> None:
  package = Path(__file__).resolve().parents[1] / "agent_gateway"
  emitters = {
    "runner_usage.py": "commercial_usage_producer",
    "sdk_runner_context.py": "producer",
    "send_prompt.py": "commercial_usage_producer",
  }
  for filename, receiver in emitters.items():
    source = (package / filename).read_text(encoding="utf-8")
    assert _awaits_emit(source, receiver), f"production usage path bypasses commercial hook: {filename}"

  for filename in ("runner_sub_agents.py", "autonomous.py"):
    source = (package / filename).read_text(encoding="utf-8")
    assert _passes_keyword(source, "commercial_usage_producer"), filename
  easy_source = (package / "easy.py").read_text(encoding="utf-8")
  assert "commercial_usage_producer_factory" in easy_source
  assert _passes_keyword(easy_source, "commercial_usage_producer")

  sdk_stream = (package / "sdk_runner_stream.py").read_text(encoding="utf-8")
  assert {"message_start", "message_delta"} <= {
    node.value
    for node in ast.walk(ast.parse(sdk_stream))
    if isinstance(node, ast.Constant) and isinstance(node.value, str)
  }
  assert not _awaits_emit("async def bypass(producer):\n  return None\n", "producer")
  for filename in ("runner_stream_turn.py", "sdk_runner.py", "send_prompt.py"):
    assert "assert_work_allowed" in (package / filename).read_text(encoding="utf-8")
  assert (package / "sdk_runner.py").read_text(encoding="utf-8").count(
    "commercial_guard(self._billing_mode)"
  ) >= 2
