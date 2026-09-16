import asyncio
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import httpx
import pytest

from agent_workflow_contracts import (
  AgentOperationRef,
  AttemptRef,
  OrdinaryDelegationTaskRef,
  OutcomeRequirement,
  ResultRequirement,
  TaskResultProvenance,
  sha256_digest,
)

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import (  # noqa: E402
  AgentRunner,
  AgentSessionLog,
  EventLog,
  GatewaySession,
  ModelInfo,
  ModelProvider,
  ToolDispatcher,
)
from agent_gateway.capability_execution import BoundCapabilityExecution  # noqa: E402
from agent_gateway.commercial_usage import CommercialUsageProducer  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.multi_user.billing import (  # noqa: E402
  SessionUsageSummary,
  UsageEvent,
  _UsageAggregator,
)
from agent_gateway.mcp_client import McpClientManager  # noqa: E402
from agent_gateway.providers import CodexProvider, CostEstimate, OpenAIProvider, StreamEvent  # noqa: E402
from agent_gateway.runner_usage import (  # noqa: E402
  apply_message_start_usage,
  apply_usage_update,
  build_usage_event,
  call_late_usage_event_hook,
  call_session_summary_hook,
  call_usage_event_hook,
  empty_usage_totals,
  estimate_usage_cost,
  turn_usage_payload,
  usage_delta,
  usage_delta_state,
  usage_has_tokens,
  usage_snapshot,
)
from agent_gateway.runner_budget import CostAccumulator  # noqa: E402
from agent_gateway.task_registry import TaskRegistry, TaskState  # noqa: E402
from gateway_test_support.capability_execution_test_support import (  # noqa: E402
  stub_bound_capability_execution,
)


def test_runner_usage_wrappers_resolve_parent_module_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(gateway_runner, "_usage_has_tokens", lambda usage_totals: usage_totals.get("patched") == 1)
  monkeypatch.setattr(gateway_runner, "_usage_delta", lambda before, after: {"patched": after["x"] - before["x"]})

  assert AgentRunner._usage_has_tokens({"patched": 1}) is True
  assert AgentRunner._usage_delta({"x": 2}, {"x": 5}) == {"patched": 3}

  runner = object.__new__(AgentRunner)
  runner._usage_user_id = "alice"
  runner._full_session_id = "sess"
  runner._request_id = "req"
  runner._parent_turn_id = "turn"
  runner._provider = _UsageProvider()
  runner._rate_table_version = "v1"
  runner._billing_mode = "metered"
  runner._channel = "web"
  runner._estimate_usage_cost = lambda _model, _usage_totals: CostEstimate(total=0.5)  # type: ignore[method-assign]
  monkeypatch.setattr(gateway_runner, "time", SimpleNamespace(time=lambda: 42.0))
  monkeypatch.setattr(
    gateway_runner,
    "_build_usage_event",
    lambda **kwargs: {"patched": kwargs},
  )

  event = AgentRunner._build_usage_event(runner, model="model", usage_totals={"input_tokens": 1})
  assert isinstance(event, dict)

  assert event["patched"]["timestamp"] == 42.0
  assert event["patched"]["cost_total"] == 0.5
  assert event["patched"]["provider_name"] == "stub"


def _run(coro):
  return asyncio.run(coro)


def test_compaction_event_seam_counts_once_rearms_and_shares_root_total() -> None:
  async def case() -> None:
    aggregator = _UsageAggregator(
      user_id="alice",
      session_id="sess-parent",
      request_id="req-123",
      channel="web",
    )
    parent = object.__new__(AgentRunner)
    child = object.__new__(AgentRunner)
    parent._aggregator = aggregator
    child._aggregator = aggregator
    parent._log = EventLog()
    child._log = EventLog()
    parent._context_pressure_next_reminder_pct = 90
    child._context_pressure_next_reminder_pct = 80

    parent._append({"type": "status", "message": "ordinary event"})
    parent._append({"type": "compaction", "chars": 12})
    child._append({"type": "compaction", "chars": 34})

    summary = await aggregator.snapshot()
    assert summary.compaction_count == 2
    assert parent._context_pressure_next_reminder_pct == 60
    assert child._context_pressure_next_reminder_pct == 60
    assert [
      entry.event["type"]
      for entry in parent._log.entries
    ] == ["status", "compaction"]
    assert [
      entry.event["type"]
      for entry in child._log.entries
    ] == ["compaction"]

    await aggregator.close()
    with pytest.raises(
      RuntimeError,
      match="closed before compaction",
    ):
      parent._append({"type": "compaction", "chars": 56})
    assert (await aggregator.snapshot()).compaction_count == 2

  _run(case())


def _subagent_identity(physical_task_id: str) -> dict[str, Any]:
  operation = AgentOperationRef(
    namespace="agent-operation",
    name="usage-test-child",
    version="1.0",
    digest=sha256_digest({"operation": "usage-test-child"}),
  )
  logical_task = OrdinaryDelegationTaskRef(
    delegation_id=f"delegation:{physical_task_id}",
    operation=operation,
  )
  attempt = AttemptRef(
    attempt_number=1,
    attempt_id=f"attempt:{physical_task_id}:1",
    physical_task_id=physical_task_id,
  )
  digest = sha256_digest({
    "logical_task": logical_task.model_dump(mode="json"),
    "attempt": attempt.model_dump(mode="json"),
  })
  return {
    "logical_task": logical_task,
    "attempt": attempt,
    "result_requirement": ResultRequirement(
      mode="narrative",
      terminal_narrative="required",
      outcome=OutcomeRequirement(required=False, source="none"),
    ),
    "result_provenance": TaskResultProvenance(
      admitted_task_digest=digest,
      model_bind_digest=digest,
      capability_binding_digest=digest,
      tool_grant_digest=digest,
    ),
  }


def _child_execution(
  provider: ModelProvider,
  *,
  model: str = "claude-sonnet-4-6",
) -> BoundCapabilityExecution:
  return stub_bound_capability_execution(
    provider=provider,
    model=model,
    effort="none",
    capability_id="node.implement",
    credential_principal="user",
    auth_config={
      "api_key": "k",
    },
  )


def _runner_execution(provider: ModelProvider) -> BoundCapabilityExecution:
  return stub_bound_capability_execution(
    provider=provider,
    model="claude-sonnet-4-6",
    effort="none",
    auth_config={"api_key": "k"},
  )


def _usage_event() -> UsageEvent:
  bind = _runner_execution(_UsageProvider()).bind
  return UsageEvent(
    user_id="alice",
    session_id="sess-parent",
    request_id="req-123",
    parent_turn_id=None,
    timestamp=123.0,
    model="claude-sonnet-4-6",
    input_tokens=10,
    output_tokens=5,
    cache_read_tokens=1,
    cache_creation_tokens=2,
    cost_usd=0.01,
    rate_table_version="2026-04-08",
    billing_mode="metered",
    channel="web",
    provider=bind.provider,
    capability_bind=bind.to_json(),
    provider_reported_model=None,
  )


def _session_summary() -> SessionUsageSummary:
  bind = _runner_execution(_UsageProvider()).bind
  return SessionUsageSummary(
    user_id="alice",
    session_id="sess-parent",
    request_id="req-123",
    input_tokens=10,
    output_tokens=5,
    cache_read_tokens=1,
    cache_creation_tokens=2,
    cost=0.01,
    turns=1,
    channel="web",
    started_at=100.0,
    ended_at=123.0,
    model="claude-sonnet-4-6",
    provider="stub",
    capability_bind=bind.to_json(),
    usage_event_count=1,
    usage_event_ids=("usage-summary-event-1",),
    rate_table_version="2026-04-08",
    billing_mode="metered",
  )


class _RecordingUsageAggregator:
  def __init__(self, *, recorded: bool = True) -> None:
    self.recorded = recorded
    self.events: list[UsageEvent] = []

  async def record(self, event: UsageEvent) -> bool:
    self.events.append(event)
    return self.recorded


def _null_mcp_client() -> McpClientManager:
  return McpClientManager(config_path=None)


class _UsageProvider(ModelProvider):
  name = "stub"

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(
      id=model,
      provider=self.name,
      input_cost_per_mtok=1.0,
      output_cost_per_mtok=2.0,
      cache_read_cost_per_mtok=0.5,
      cache_write_cost_per_mtok=0.75,
    )

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    **kwargs: Any,
  ) -> dict[str, Any]:
    _ = model, messages, system_prompt, tools, max_tokens, kwargs
    return {}

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    yield StreamEvent(type="message_start", input_tokens=100, cache_read_tokens=10, cache_creation_tokens=5)
    yield StreamEvent(type="text_delta", text="hello ")
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": "hello "})
    yield StreamEvent(type="usage_update", output_tokens=50)
    yield StreamEvent(type="message_end", stop_reason="end_turn")


class _RetryAfterUsageProvider(_UsageProvider):
  def __init__(self) -> None:
    self.calls = 0

  def is_retryable_error(self, exc: Exception) -> bool:
    return isinstance(exc, TimeoutError)

  async def stream(self, client: Any, params: dict[str, Any]):
    self.calls += 1
    if self.calls == 1:
      provider_unit_deltas = {"web_search": 2}
    else:
      provider_unit_deltas = {"web_search": 1, "web_fetch": 4}
    yield StreamEvent(
      type="message_start",
      input_tokens=10,
      provider_unit_deltas=provider_unit_deltas,
    )
    yield StreamEvent(type="usage_update", output_tokens=2)
    if self.calls == 1:
      raise TimeoutError("retry me")
    yield StreamEvent(type="message_end", stop_reason="end_turn")


def test_retry_emits_failed_billable_delta_before_success(monkeypatch: pytest.MonkeyPatch) -> None:
  states = []

  class Producer:
    async def emit(self, event, *, usage_state="succeeded"):
      states.append((
        event.input_tokens,
        event.output_tokens,
        event.provider_unit_deltas,
        usage_state,
      ))

  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_MAX", 1)
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", 0.0)
  event_log = EventLog()
  provider = _RetryAfterUsageProvider()
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-retry",
    capability_execution=_runner_execution(provider),
    user_id="alice", request_id="req-retry", billing_mode="metered",
    rate_table_version="v1", channel="web",
    commercial_usage_producer=Producer(),
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  assert states == [
    (10, 2, {"web_search": 2}, "failed_billable"),
    (10, 2, {"web_search": 1, "web_fetch": 4}, "succeeded"),
  ]


def test_terminal_watchdog_emits_accumulated_failed_billable_usage(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  states = []

  class HangingProvider(_UsageProvider):
    async def stream(self, client: Any, params: dict[str, Any]):
      yield StreamEvent(type="message_start", input_tokens=10)
      yield StreamEvent(type="usage_update", output_tokens=2)
      await asyncio.Event().wait()

  class Producer:
    async def emit(self, event, *, usage_state="succeeded"):
      states.append((event.input_tokens, event.output_tokens, usage_state))

  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_MAX", 0)
  monkeypatch.setattr(gateway_runner, "STREAM_GUARD_POLL_INTERVAL", 0.001)
  event_log = EventLog()
  provider = HangingProvider()
  runner = AgentRunner(
    event_log=event_log, dispatcher=_make_dispatcher(event_log),
    session_id="sess-watchdog",
    capability_execution=_runner_execution(provider),
    user_id="alice", request_id="req-watchdog", billing_mode="metered",
    rate_table_version="v1", channel="web", per_turn_timeout=0.01,
    commercial_usage_producer=Producer(),
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  assert states == [(10, 2, "failed_billable")]


def test_disconnect_on_retry_persists_canceled_partial_delta_first(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  states = []
  runner_ref = []

  class DisconnectingProvider(_UsageProvider):
    def is_retryable_error(self, exc: Exception) -> bool:
      return isinstance(exc, TimeoutError)

    async def stream(self, client: Any, params: dict[str, Any]):
      yield StreamEvent(type="message_start", input_tokens=10)
      yield StreamEvent(type="usage_update", output_tokens=2)
      runner_ref[0]._disconnected = True
      raise TimeoutError("disconnect during provider call")

  class Producer:
    async def emit(self, event, *, usage_state="succeeded"):
      states.append((event.input_tokens, event.output_tokens, usage_state))

  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_MAX", 1)
  event_log = EventLog()
  provider = DisconnectingProvider()
  runner = AgentRunner(
    event_log=event_log, dispatcher=_make_dispatcher(event_log),
    session_id="sess-disconnect",
    capability_execution=_runner_execution(provider),
    user_id="alice", request_id="req-disconnect", billing_mode="metered",
    rate_table_version="v1", channel="web",
    commercial_usage_producer=Producer(),
  )
  runner_ref.append(runner)

  with pytest.raises(TimeoutError, match="disconnect during provider call"):
    _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  assert states == [(10, 2, "canceled")]


def test_usage_helper_functions_match_runner_usage_contract() -> None:
  assert empty_usage_totals() == {
    "input_tokens": 0,
    "output_tokens": 0,
    "reasoning_tokens_observed": 0,
    "provider_units": 0,
    "provider_unit_deltas": {},
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
  }

  before = {
    "input_tokens": 20,
    "output_tokens": 10,
    "cache_read_input_tokens": 5,
    "cache_creation_input_tokens": 2,
  }
  after = {
    "input_tokens": "30",
    "output_tokens": 9,
    "cache_read_input_tokens": 8,
    "cache_creation_input_tokens": 7,
  }

  delta = usage_delta(before, after)

  assert delta == {
    "input_tokens": 10,
    "output_tokens": 0,
    "reasoning_tokens_observed": 0,
    "provider_units": 0,
    "provider_unit_deltas": {},
    "cache_read_input_tokens": 3,
    "cache_creation_input_tokens": 5,
  }
  assert usage_has_tokens(delta) is True
  assert usage_has_tokens({key: 0 for key in delta}) is False
  assert AgentRunner._usage_delta(before, after) == delta
  assert AgentRunner._usage_has_tokens(delta) is True


def test_usage_snapshot_owns_nested_provider_unit_deltas() -> None:
  usage = empty_usage_totals()
  apply_message_start_usage(
    usage,
    input_tokens=10,
    cache_creation_tokens=0,
    cache_read_tokens=0,
    provider_units=2,
    provider_unit_deltas={"search": 2},
  )

  snapshot = usage_snapshot(usage)
  apply_usage_update(
    usage,
    output_tokens=1,
    provider_units=5,
    provider_unit_deltas={"search": 1, "image": 4},
  )

  assert snapshot["provider_unit_deltas"] == {"search": 2}
  assert usage["provider_unit_deltas"] == {"search": 3, "image": 4}
  assert snapshot["provider_unit_deltas"] is not usage["provider_unit_deltas"]


def test_usage_delta_state_returns_delta_and_token_flag() -> None:
  before = {
    "input_tokens": 20,
    "output_tokens": 10,
    "cache_read_input_tokens": 5,
    "cache_creation_input_tokens": 2,
  }
  after = {
    "input_tokens": "25",
    "output_tokens": 10,
    "cache_read_input_tokens": 3,
    "cache_creation_input_tokens": 2,
  }

  state = usage_delta_state(before, after)

  assert state.usage == {
    "input_tokens": 5,
    "output_tokens": 0,
    "reasoning_tokens_observed": 0,
    "provider_units": 0,
    "provider_unit_deltas": {},
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
  }
  assert state.has_tokens is True


def test_usage_delta_state_reports_false_for_empty_delta() -> None:
  usage = {
    "input_tokens": 20,
    "output_tokens": 10,
    "cache_read_input_tokens": 5,
    "cache_creation_input_tokens": 2,
  }

  state = usage_delta_state(usage, dict(usage))

  assert state.usage == {
    "input_tokens": 0,
    "output_tokens": 0,
    "reasoning_tokens_observed": 0,
    "provider_units": 0,
    "provider_unit_deltas": {},
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
  }
  assert state.has_tokens is False


def test_apply_message_start_usage_mutates_existing_totals() -> None:
  usage = {
    "input_tokens": 20,
    "output_tokens": 4,
    "cache_read_input_tokens": 5,
    "cache_creation_input_tokens": 2,
  }

  result = apply_message_start_usage(
    usage,
    input_tokens=30,
    cache_read_tokens=8,
    cache_creation_tokens=7,
  )

  assert result is usage
  assert usage == {
    "input_tokens": 50,
    "output_tokens": 4,
    "cache_read_input_tokens": 13,
    "cache_creation_input_tokens": 9,
    "provider_units": 0,
    "provider_unit_deltas": {},
  }


def test_apply_usage_update_mutates_output_tokens_only() -> None:
  usage = {
    "input_tokens": 20,
    "output_tokens": 4,
    "cache_read_input_tokens": 5,
    "cache_creation_input_tokens": 2,
  }

  result = apply_usage_update(usage, output_tokens=11)

  assert result is usage
  assert usage == {
    "input_tokens": 20,
    "output_tokens": 15,
    "reasoning_tokens_observed": 0,
    "provider_units": 0,
    "provider_unit_deltas": {},
    "cache_read_input_tokens": 5,
    "cache_creation_input_tokens": 2,
  }


def test_turn_usage_payload_copies_usage_and_rounds_cost() -> None:
  usage = {
    "input_tokens": 10,
    "output_tokens": 5,
    "provider_unit_deltas": {"search": 2},
    "cache_read_input_tokens": 3,
    "cache_creation_input_tokens": 2,
  }

  payload = turn_usage_payload(usage, estimated_cost=0.123456)
  usage["input_tokens"] = 99
  usage["provider_unit_deltas"]["search"] = 99

  assert payload == {
    "input_tokens": 10,
    "output_tokens": 5,
    "provider_unit_deltas": {"search": 2},
    "cache_read_input_tokens": 3,
    "cache_creation_input_tokens": 2,
    "estimated_cost": 0.1235,
  }


def test_turn_usage_payload_omits_cost_when_not_provided() -> None:
  assert turn_usage_payload({
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
  }) == {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
  }


def test_estimate_usage_cost_passes_uncached_and_cache_token_counts() -> None:
  calls: list[tuple[Any, ...]] = []

  class _CostProvider:
    def estimate_cost(
      self,
      model: str,
      input_tokens: int,
      output_tokens: int,
      *,
      cache_read_tokens: int = 0,
      cache_creation_tokens: int = 0,
    ) -> CostEstimate:
      calls.append((model, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens))
      return CostEstimate(total=0.42)

  cost = estimate_usage_cost(
    _CostProvider(),
    "claude-sonnet-4-6",
    {
      "input_tokens": 100,
      "output_tokens": 50,
      "cache_read_input_tokens": 10,
      "cache_creation_input_tokens": 5,
    },
  )

  assert cost.total == 0.42
  assert calls == [("claude-sonnet-4-6", 100, 50, 10, 5)]


def test_build_usage_event_helper_sets_billing_fields() -> None:
  bind = _runner_execution(_UsageProvider()).bind
  event = build_usage_event(
    user_id="alice",
    session_id="sess-parent",
    request_id="req-123",
    parent_turn_id="tool-run-agent-1",
    timestamp=123.5,
    model="claude-sonnet-4-6",
    provider_name="stub",
    usage_totals={
      "input_tokens": 100,
      "output_tokens": 50,
      "cache_read_input_tokens": 10,
      "cache_creation_input_tokens": 5,
      "capability_bind": bind.to_json(),
      "provider_reported_model": None,
    },
    cost_total=0.25,
    rate_table_version="2026-04-08",
    billing_mode="metered",
    channel="web",
  )

  assert event.user_id == "alice"
  assert event.session_id == "sess-parent"
  assert event.request_id == "req-123"
  assert event.parent_turn_id == "tool-run-agent-1"
  assert event.timestamp == 123.5
  assert event.model == "claude-sonnet-4-6"
  assert event.provider == "stub"
  assert event.capability_bind == bind.to_json()
  assert event.provider_reported_model is None
  assert event.input_tokens == 100
  assert event.output_tokens == 50
  assert event.cache_read_tokens == 10
  assert event.cache_creation_tokens == 5
  assert event.cost_usd == 0.25
  assert event.rate_table_version == "2026-04-08"
  assert event.billing_mode == "metered"
  assert event.channel == "web"


def test_call_usage_event_hook_records_and_invokes_async_callback() -> None:
  event = _usage_event()
  aggregator = _RecordingUsageAggregator()
  received: list[UsageEvent] = []
  metrics: list[tuple[str, int]] = []

  async def _on_usage(usage_event: UsageEvent) -> None:
    received.append(usage_event)

  _run(
    call_usage_event_hook(
      aggregator,
      event,
      is_summary_emitted=lambda: False,
      on_usage=_on_usage,
      on_late_usage_event=None,
      emit_metric=lambda name, value: metrics.append((name, value)),
      dlq_path=None,
      log_session_id="sess-parent",
      logger=logging.getLogger("test_runner_on_usage"),
    )
  )

  assert aggregator.events == [event]
  assert received == [event]
  assert metrics == []


@pytest.mark.parametrize("recorded, summary_emitted", [(False, False), (True, True)])
def test_call_usage_event_hook_routes_late_events(recorded: bool, summary_emitted: bool) -> None:
  event = _usage_event()
  aggregator = _RecordingUsageAggregator(recorded=recorded)
  late_events: list[UsageEvent] = []

  def _unexpected_on_usage(_usage_event: UsageEvent) -> None:
    raise AssertionError("on_usage should not run for late usage events")

  _run(
    call_usage_event_hook(
      aggregator,
      event,
      is_summary_emitted=lambda: summary_emitted,
      on_usage=_unexpected_on_usage,
      on_late_usage_event=late_events.append,
      emit_metric=lambda _name, _value: None,
      dlq_path=None,
      log_session_id="sess-parent",
      logger=logging.getLogger("test_runner_on_usage"),
    )
  )

  assert aggregator.events == [event]
  assert late_events == [event]


def test_call_usage_event_hook_checks_summary_flag_after_record() -> None:
  event = _usage_event()
  summary_emitted = False
  late_events: list[UsageEvent] = []

  class _FlippingUsageAggregator:
    async def record(self, usage_event: UsageEvent) -> bool:
      nonlocal summary_emitted
      assert usage_event == event
      summary_emitted = True
      return True

  def _unexpected_on_usage(_usage_event: UsageEvent) -> None:
    raise AssertionError("on_usage should not run after summary emission")

  _run(
    call_usage_event_hook(
      _FlippingUsageAggregator(),
      event,
      is_summary_emitted=lambda: summary_emitted,
      on_usage=_unexpected_on_usage,
      on_late_usage_event=late_events.append,
      emit_metric=lambda _name, _value: None,
      dlq_path=None,
      log_session_id="sess-parent",
      logger=logging.getLogger("test_runner_on_usage"),
    )
  )

  assert late_events == [event]


def test_call_usage_event_hook_failure_records_metric_and_dlq(tmp_path: Path) -> None:
  event = _usage_event()
  metrics: list[tuple[str, int]] = []
  dlq_path = tmp_path / "usage_dlq.jsonl"

  def _failing_on_usage(_usage_event: UsageEvent) -> None:
    raise RuntimeError("ledger offline")

  _run(
    call_usage_event_hook(
      _RecordingUsageAggregator(),
      event,
      is_summary_emitted=lambda: False,
      on_usage=_failing_on_usage,
      on_late_usage_event=None,
      emit_metric=lambda name, value: metrics.append((name, value)),
      dlq_path=dlq_path,
      log_session_id="sess-parent",
      logger=logging.getLogger("test_runner_on_usage"),
    )
  )

  payload = json.loads(dlq_path.read_text(encoding="utf-8").strip())
  assert metrics == [("gateway.usage_event_dropped", 1)]
  assert payload["usage_event_schema_version"] == 2
  assert payload["event"]["event_id"] == event.event_id
  assert payload["event"]["capability_bind"] == event.capability_bind
  assert payload["event"]["provider_reported_model"] is None
  assert payload["event"]["user_id"] == "alice"


def test_late_usage_and_summary_helpers_support_async_callbacks() -> None:
  event = _usage_event()
  summary = _session_summary()
  late_events: list[UsageEvent] = []
  summaries: list[SessionUsageSummary] = []

  async def _on_late_usage_event(usage_event: UsageEvent) -> None:
    late_events.append(usage_event)

  async def _on_session_summary(session_summary: SessionUsageSummary) -> None:
    summaries.append(session_summary)

  _run(
    call_late_usage_event_hook(
      _on_late_usage_event,
      event,
      log_session_id="sess-parent",
      logger=logging.getLogger("test_runner_on_usage"),
    )
  )
  _run(
    call_session_summary_hook(
      _on_session_summary,
      summary,
      log_session_id="sess-parent",
      logger=logging.getLogger("test_runner_on_usage"),
    )
  )

  assert late_events == [event]
  assert summaries == [summary]


def test_reconciliation_failure_is_nonfatal_observable_and_summary_still_runs() -> None:
  summaries = []
  metrics = []

  class Producer(CommercialUsageProducer):
    async def reconcile(self, summary: SessionUsageSummary) -> None:
      raise RuntimeError("comparison failed")

  _run(call_session_summary_hook(
    summaries.append,
    _session_summary(),
    log_session_id="sess-parent",
    logger=logging.getLogger("test_runner_on_usage"),
    commercial_usage_producer=Producer(
      enabled=False,
      claim=None,
      lineage=None,
      sink=None,
    ),
    emit_metric=lambda name, value: metrics.append((name, value)),
  ))

  assert len(summaries) == 1
  assert metrics == [("gateway.commercial_usage_reconciliation_error", 1)]


class _OneTurnTextProvider(_UsageProvider):
  def __init__(self) -> None:
    self.requests: list[list[dict[str, Any]]] = []

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    **kwargs: Any,
  ) -> dict[str, Any]:
    _ = model, system_prompt, tools, max_tokens, kwargs
    self.requests.append([dict(message) for message in messages])
    return {"call_index": len(self.requests)}

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    yield StreamEvent(type="message_start", input_tokens=10)
    text = "rough 25 bps"
    yield StreamEvent(type="text_delta", text=text)
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": text})
    yield StreamEvent(type="usage_update", output_tokens=5)
    yield StreamEvent(type="message_end", stop_reason="end_turn")


class _FailingAfterUsageProvider(_UsageProvider):
  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    yield StreamEvent(
      type="message_start",
      input_tokens=40,
      cache_read_tokens=4,
      cache_creation_tokens=3,
      provider_unit_deltas={"web_search": 2},
    )
    yield StreamEvent(
      type="usage_update",
      output_tokens=7,
      provider_unit_deltas={"web_search": 1, "web_fetch": 4},
    )
    raise RuntimeError("stream exploded")
    yield  # pragma: no cover


def _make_dispatcher(
  event_log: EventLog | None = None,
) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=_null_mcp_client(),
    local_tool_handlers={},
    event_log=event_log or EventLog(),
    session_id="sess-parent",
    get_tool_definitions=None,
  )


def test_runner_tool_timing_forwards_tool_call_and_request_ids() -> None:
  timing_calls: list[dict[str, Any]] = []

  def on_tool_timing(
    session_id,
    tool_name,
    server,
    duration_ms,
    is_error,
    result_bytes,
    *,
    user_id=None,
    tool_call_id=None,
    request_id=None,
  ):
    timing_calls.append(
      {
        "session_id": session_id,
        "tool_name": tool_name,
        "server": server,
        "duration_ms": duration_ms,
        "is_error": is_error,
        "result_bytes": result_bytes,
        "user_id": user_id,
        "tool_call_id": tool_call_id,
        "request_id": request_id,
      }
    )

  provider = _UsageProvider()
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    on_tool_timing=on_tool_timing,
    user_id="alice",
    request_id="req-123",
    billing_mode="metered",
    rate_table_version="2026-04-08",
    channel="web",
  )

  runner._call_on_tool_timing(
    tool_name="documents_search",
    server="portfolio-reads-mcp",
    duration_ms=12,
    is_error=False,
    result_bytes=34,
    tool_call_id="toolu-123",
    request_id=runner._request_id,
  )

  assert timing_calls == [
    {
      "session_id": "sess-parent",
      "tool_name": "documents_search",
      "server": "portfolio-reads-mcp",
      "duration_ms": 12,
      "is_error": False,
      "result_bytes": 34,
      "user_id": "alice",
      "tool_call_id": "toolu-123",
      "request_id": "req-123",
    }
  ]


def test_runner_build_usage_event_preserves_timestamp_and_cost_delegates(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  provider = _UsageProvider()
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    user_id="alice",
    request_id="req-123",
    billing_mode="metered",
    rate_table_version="2026-04-08",
    channel="web",
  )
  monkeypatch.setattr(gateway_runner.time, "time", lambda: 456.75)
  runner._estimate_usage_cost = lambda _model, _usage_totals: CostEstimate(total=0.125)  # type: ignore[method-assign]

  event = runner._build_usage_event(
    model="claude-sonnet-4-6",
    usage_totals={
      "input_tokens": 100,
      "output_tokens": 50,
      "cache_read_input_tokens": 10,
      "cache_creation_input_tokens": 5,
      "capability_bind": runner.capability_execution.bind.to_json(),
      "provider_reported_model": None,
    },
  )

  assert event.timestamp == 456.75
  assert event.cost_usd == 0.125
  assert event.provider == "stub"


def test_no_tool_final_answer_completes_in_one_provider_request_without_guard() -> None:
  event_log = EventLog()
  provider = _OneTurnTextProvider()
  durable_events: list[dict[str, Any]] = []

  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  async def _append_durable_event(event: dict[str, Any]):
    durable_events.append(dict(event))
    return SimpleNamespace(seq=len(durable_events))

  runner._append_durable_event = _append_durable_event

  _run(runner.run(messages=[{"role": "user", "content": "compare margin bps"}]))

  assert len(provider.requests) == 1
  assert not any(entry.event.get("type") == "runtime_guard" for entry in event_log.entries)
  assistant_messages = [
    event
    for event in durable_events
    if event.get("type") == "assistant_message"
  ]
  assert len(assistant_messages) == 1
  assert assistant_messages[0]["content_blocks"] == [{"type": "text", "text": "rough 25 bps"}]
  assert not any(event.get("type") == "runtime_guard" for event in durable_events)


def test_on_usage_fires_once_per_turn_with_usage_event_fields() -> None:
  events: list[UsageEvent] = []
  event_log = EventLog()
  provider = _UsageProvider()
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    on_usage=events.append,
    user_id="alice",
    request_id="req-123",
    billing_mode="metered",
    rate_table_version="2026-04-08",
    channel="web",
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  assert len(events) == 1
  event = events[0]
  assert event.user_id == "alice"
  assert event.session_id == "sess-parent"
  assert event.request_id == "req-123"
  assert event.parent_turn_id is None
  assert event.model == "claude-sonnet-4-6"
  assert event.input_tokens == 100
  assert event.output_tokens == 50
  assert event.cache_read_tokens == 10
  assert event.cache_creation_tokens == 5
  assert event.cost_usd == pytest.approx(0.00020875)
  assert event.rate_table_version == "2026-04-08"
  assert event.billing_mode == "metered"
  assert event.channel == "web"
  assert event.provider == "stub"


@pytest.mark.parametrize("budget", [None, 4.0])
def test_session_cost_sums_request_tiers_instead_of_repricing_total_tokens(budget) -> None:
  priced_provider = OpenAIProvider()

  class _TieredUsageProvider(_UsageProvider):
    name = "openai"

    def __init__(self) -> None:
      self.calls = 0

    def get_model_info(self, model: str) -> ModelInfo:
      return priced_provider.get_model_info(model)

    async def stream(self, client: Any, params: dict[str, Any]):
      self.calls += 1
      yield StreamEvent(type="message_start", input_tokens=150_000)
      yield StreamEvent(type="text_delta", text="response")
      yield StreamEvent(type="text_end", raw_block={"type": "text", "text": "response"})
      yield StreamEvent(type="usage_update", output_tokens=1_000)
      yield StreamEvent(type="message_end", stop_reason="max_tokens" if self.calls == 1 else "end_turn")

  usage_events: list[UsageEvent] = []
  event_log = EventLog()
  provider = _TieredUsageProvider()
  accumulator = CostAccumulator(budget) if budget is not None else None
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-request-tiers",
    capability_execution=stub_bound_capability_execution(
      provider=provider, model="gpt-6-astra", effort="low", auth_config={"api_key": "k"},
    ),
    _cost_accumulator=accumulator,
    on_usage=usage_events.append,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="2026-09-11",
  )
  _run(runner.run(messages=[{"role": "user", "content": "continue until complete"}]))

  assert [event.cost_usd for event in usage_events] == pytest.approx([1.55, 1.55])
  completion = next(entry.event for entry in event_log.entries if entry.event.get("type") == "stream_complete")
  assert completion["usage"]["estimated_cost"] == pytest.approx(3.10)
  assert completion["terminal_disposition"] == "completed"
  if accumulator is not None:
    assert accumulator.total == pytest.approx(3.10)


def test_portable_compaction_cost_keeps_request_tiers_separate(tmp_path: Path) -> None:
  priced_provider = CodexProvider()

  class _CompactionUsageProvider(_UsageProvider):
    name = "codex"

    def __init__(self) -> None:
      self.calls = 0

    def get_model_info(self, model: str) -> ModelInfo:
      return replace(priced_provider.get_model_info(model), context_window=30_000)

    async def stream(self, client: Any, params: dict[str, Any]):
      self.calls += 1
      is_compaction = self.calls == 1
      text = (
        "<summary>" + "Keep the prior research and continue the task. " * 10 + "</summary>"
        if is_compaction else "Research complete."
      )
      yield StreamEvent(type="message_start", input_tokens=260_000 if is_compaction else 15_000)
      yield StreamEvent(type="text_delta", text=text)
      yield StreamEvent(type="text_end", raw_block={"type": "text", "text": text})
      yield StreamEvent(type="usage_update", output_tokens=1_000)
      yield StreamEvent(type="message_end", stop_reason="end_turn")

  usage_events: list[UsageEvent] = []
  summaries: list[SessionUsageSummary] = []
  event_log = EventLog()
  durable_log = AgentSessionLog(path=tmp_path / "compaction-pricing.jsonl")
  provider = _CompactionUsageProvider()
  accumulator = CostAccumulator(4.0)
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-compaction-tiers",
    capability_execution=stub_bound_capability_execution(
      provider=provider, model="gpt-6-astra", effort="low", auth_config={"api_key": "k"},
    ),
    _cost_accumulator=accumulator,
    on_usage=usage_events.append,
    on_session_summary=summaries.append,
    agent_session_log=durable_log,
    compaction_trigger=24_000,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="2026-09-11",
  )
  _run(runner.run(messages=[{"role": "user", "content": "Earlier research. " * 6_000}]))

  events = [entry.event for entry in event_log.entries]
  completion = next(event for event in events if event.get("type") == "stream_complete")
  assert (completion["usage"]["estimated_cost"], completion["terminal_disposition"]) == (
    pytest.approx(2.85), "completed",
  )
  assert not any(event.get("type") == "budget_exceeded" for event in events)
  assert accumulator.total == pytest.approx(2.85)
  assert [event.cost_usd for event in usage_events] == pytest.approx([2.65, 0.20])
  assert [(event.input_tokens, event.output_tokens) for event in usage_events] == [
    (260_000, 1_000), (15_000, 1_000),
  ]
  assert len(summaries) == 1
  assert summaries[0].cost == pytest.approx(2.85)
  assert summaries[0].compaction_count == 1
  turn_complete = next(event for event in events if event.get("type") == "turn_complete")
  for usage in (completion["usage"], turn_complete["usage"]):
    assert usage["input_tokens"] == 275_000
    assert usage["output_tokens"] == 2_000
    assert usage["estimated_cost"] == pytest.approx(2.85)
  assistant_messages, _ = _run(durable_log.query(event_types={"assistant_message"}, order="asc"))
  assert assistant_messages[0].event["usage"]["estimated_cost"] == pytest.approx(2.85)
  assert assistant_messages[0].event["content_blocks"][0]["type"] == "compaction"


def test_runner_preserves_cumulative_typed_provider_unit_deltas(
  tmp_path: Path,
) -> None:
  class _ProviderUnitUsageProvider(_UsageProvider):
    async def stream(self, client: Any, params: dict[str, Any]):
      _ = client, params
      yield StreamEvent(
        type="message_start",
        input_tokens=100,
        cache_read_tokens=10,
        cache_creation_tokens=5,
        provider_unit_deltas={"web_search": 2},
      )
      yield StreamEvent(type="text_delta", text="usage probe complete")
      yield StreamEvent(
        type="text_end",
        raw_block={"type": "text", "text": "usage probe complete"},
      )
      yield StreamEvent(
        type="usage_update",
        input_tokens=7,
        output_tokens=50,
        reasoning_tokens=11,
        cache_read_tokens=4,
        cache_creation_tokens=2,
        provider_unit_deltas={"web_search": 1, "web_fetch": 4},
      )
      yield StreamEvent(type="message_end", stop_reason="end_turn")

  usage_events: list[UsageEvent] = []
  event_log = EventLog()
  durable_log = AgentSessionLog(tmp_path / "provider-unit-usage.jsonl")
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-provider-unit-usage",
    capability_execution=_runner_execution(_ProviderUnitUsageProvider()),
    agent_session_log=durable_log,
    on_usage=usage_events.append,
    user_id="alice",
    request_id="req-provider-unit-usage",
    billing_mode="metered",
    rate_table_version="2026-04-08",
    channel="web",
  )

  _run(runner.run(messages=[{"role": "user", "content": "exercise usage"}]))

  assert len(usage_events) == 1
  assert usage_events[0].provider_units is None
  assert usage_events[0].provider_unit_deltas == {
    "web_search": 3,
    "web_fetch": 4,
  }
  turn_complete = next(
    entry.event
    for entry in event_log.entries
    if entry.event.get("type") == "turn_complete"
  )
  assert turn_complete["usage"]["provider_units"] == 0
  assert turn_complete["usage"]["provider_unit_deltas"] == {
    "web_search": 3,
    "web_fetch": 4,
  }
  durable_entries, _ = _run(durable_log.query(order="asc"))
  assistant = next(
    entry.event
    for entry in durable_entries
    if entry.event.get("type") == "assistant_message"
  )
  assert assistant["usage"]["provider_units"] == 0
  assert assistant["usage"]["provider_unit_deltas"] == {
    "web_search": 3,
    "web_fetch": 4,
  }


def test_stream_turn_failure_emits_partial_usage_and_rolls_back_totals() -> None:
  events: list[UsageEvent] = []
  event_log = EventLog()
  provider = _FailingAfterUsageProvider()
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    on_usage=events.append,
    user_id="alice",
    request_id="req-123",
    billing_mode="metered",
    rate_table_version="2026-04-08",
    channel="web",
  )
  usage_totals = {
    "input_tokens": 5,
    "output_tokens": 2,
    "reasoning_tokens_observed": 0,
    "provider_units": 0,
    "provider_unit_deltas": {"existing": 1},
    "cache_read_input_tokens": 1,
    "cache_creation_input_tokens": 0,
  }
  initial_usage_totals = usage_snapshot(usage_totals)

  result = _run(
    runner._stream_turn(
      client=object(),
      config={
        "model": "claude-sonnet-4-6",
        "effort": "none",
        "thinking_enabled_requested": False,
        "auth_mode": "api",
      },
      model_info=runner._provider.get_model_info("claude-sonnet-4-6"),
      system_prompt=None,
      current_messages=[{"role": "user", "content": "hello"}],
      base_kwargs={"tools": []},
      max_tokens=1024,
      turn_count=1,
      turn_t0=gateway_runner.time.time(),
      turn_t0_mono=gateway_runner.time.monotonic(),
      system_chars=0,
      tools_chars=0,
      usage_totals=usage_totals,
    )
  )

  assert result is None
  assert usage_totals == initial_usage_totals
  assert len(events) == 1
  event = events[0]
  assert event.input_tokens == 40
  assert event.output_tokens == 7
  assert event.provider_units is None
  assert event.provider_unit_deltas == {"web_search": 3, "web_fetch": 4}
  assert event.cache_read_tokens == 4
  assert event.cache_creation_tokens == 3
  assert event.cost_usd == pytest.approx(0.00005825)
  error_events = [entry.event for entry in event_log.entries if entry.event["type"] == "error"]
  assert len(error_events) == 1
  assert "stream exploded" in error_events[0]["error"]


@pytest.mark.parametrize("cancel_during_settlement", [False, True])
@pytest.mark.parametrize("close_fails", [False, True])
def test_standalone_stream_error_closes_client_after_settlement(
  monkeypatch: pytest.MonkeyPatch,
  caplog: pytest.LogCaptureFixture,
  cancel_during_settlement: bool,
  close_fails: bool,
) -> None:
  async def case() -> None:
    connection_closed = asyncio.Event()

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
      try:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        await writer.drain()
        await reader.read()
      finally:
        writer.close()
        await writer.wait_closed()
        connection_closed.set()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    class HttpFailureProvider(_FailingAfterUsageProvider):
      async def stream(self, client: Any, params: dict[str, Any]):
        response = await client.get(f"http://127.0.0.1:{port}/")
        assert response.content == b"ok"
        async for event in super().stream(client, params):
          yield event

      async def close_client(self, client: Any, timeout: float = 2.0) -> None:
        await client.aclose()
        if close_fails:
          raise RuntimeError("provider close exploded")

    provider = HttpFailureProvider()
    event_log = EventLog()
    usage_events: list[UsageEvent] = []
    runner = AgentRunner(
      event_log=event_log,
      dispatcher=_make_dispatcher(event_log),
      session_id="standalone-error-cleanup",
      capability_execution=_runner_execution(provider),
      on_usage=usage_events.append,
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
    client = httpx.AsyncClient(trust_env=False)
    runner._set_client(client)
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    release_executor = threading.Event()
    blocker = loop.run_in_executor(None, release_executor.wait)
    settlement_started = asyncio.Event()
    settle = runner._run_durable_session_settlement

    async def observed_settlement(*args: Any, **kwargs: Any) -> Any:
      settlement_started.set()
      return await settle(*args, **kwargs)

    monkeypatch.setattr(runner, "_run_durable_session_settlement", observed_settlement)
    usage_totals = empty_usage_totals()
    task = asyncio.create_task(runner._stream_turn(
      client=client,
      config={
        "model": "claude-sonnet-4-6",
        "effort": "none",
        "auth_mode": "api",
      },
      model_info=provider.get_model_info("claude-sonnet-4-6"),
      system_prompt=None,
      current_messages=[{"role": "user", "content": "hello"}],
      base_kwargs={"tools": []},
      max_tokens=1024,
      turn_count=1,
      turn_t0=gateway_runner.time.time(),
      turn_t0_mono=gateway_runner.time.monotonic(),
      system_chars=0,
      tools_chars=0,
      usage_totals=usage_totals,
    ))
    try:
      await asyncio.wait_for(settlement_started.wait(), timeout=5.0)
      pool = client._transport._pool
      assert (client.is_closed, len(pool.connections), runner._active_client is client) == (
        False, 1, True,
      )
      if cancel_during_settlement:
        for _ in range(2):
          task.cancel()
          await asyncio.sleep(0)
        assert not task.done()
      release_executor.set()
      if cancel_during_settlement:
        with pytest.raises(asyncio.CancelledError):
          await asyncio.wait_for(task, timeout=5.0)
        assert task.cancelled()
        if close_fails:
          assert any(
            record.name == "agent_gateway.runner"
            and record.levelno >= logging.WARNING
            and "provider close exploded" in record.getMessage()
            for record in caplog.records
          )
      elif close_fails:
        with pytest.raises(RuntimeError, match="provider close exploded"):
          await asyncio.wait_for(task, timeout=5.0)
        assert not task.cancelled()
      else:
        assert await asyncio.wait_for(task, timeout=5.0) is None
      assert (client.is_closed, len(pool.connections), runner._active_client) == (
        True, 0, None,
      )
      await asyncio.wait_for(connection_closed.wait(), timeout=5.0)
      errors = [entry.event for entry in event_log.entries if entry.event["type"] == "error"]
      assert len(errors) == 1
      assert "stream exploded" in errors[0]["error"]
      assert [(event.input_tokens, event.output_tokens) for event in usage_events] == [(40, 7)]
      assert usage_totals == empty_usage_totals()
    finally:
      release_executor.set()
      await blocker
      task.cancel()
      await asyncio.gather(task, return_exceptions=True)
      await client.aclose()
      server.close()
      await server.wait_closed()
      await asyncio.wait_for(connection_closed.wait(), timeout=5.0)

  _run(case())


def test_on_usage_failure_does_not_block_chat_response(tmp_path: Path) -> None:
  event_log = EventLog()

  def _failing_on_usage(_event: UsageEvent) -> None:
    raise RuntimeError("ledger offline")

  provider = _UsageProvider()
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    on_usage=_failing_on_usage,
    user_id="alice",
    request_id="req-123",
    usage_ledger_dlq_path=tmp_path / "usage_dlq.jsonl",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  event_types = [entry.event["type"] for entry in event_log.entries]
  assert "stream_complete" in event_types


def test_on_usage_failure_writes_to_dlq_spool(tmp_path: Path) -> None:
  spool_path = tmp_path / "usage_dlq.jsonl"

  async def _failing_on_usage(_event: UsageEvent) -> None:
    raise RuntimeError("db unavailable")

  provider = _UsageProvider()
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    on_usage=_failing_on_usage,
    user_id="alice",
    request_id="req-123",
    usage_ledger_dlq_path=spool_path,
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  payload = json.loads(spool_path.read_text(encoding="utf-8").strip())
  assert payload["usage_event_schema_version"] == 2
  assert payload["event"]["user_id"] == "alice"
  assert payload["event"]["request_id"] == "req-123"
  assert payload["event"]["session_id"] == "sess-parent"
  assert payload["event"]["input_tokens"] == 100
  assert payload["event"]["output_tokens"] == 50
  assert payload["event"]["capability_bind"] == runner.capability_execution.bind.to_json()
  assert payload["event"]["provider_reported_model"] is None


def test_spawn_sub_agent_emits_usage_with_parent_turn_id(tmp_path: Path) -> None:
  events: list[UsageEvent] = []
  provider = _UsageProvider()
  parent_runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=AgentSessionLog(path=tmp_path / "usage-parent.jsonl"),
    workspace_dir=str(tmp_path),
    on_usage=events.append,
    user_id="alice",
    request_id="req-123",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  sub_session = GatewaySession(
    session_id="sub0:sess-parent",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id="alice",
    auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
  )
  result, error = _run(
    parent_runner.spawn_sub_agent(
      "Collect usage",
      capability_execution=_child_execution(_UsageProvider()),
      **_subagent_identity("sub0:sess-parent"),
      skill_name="test-child",
      dispatcher=_make_dispatcher(),
      sub_session=sub_session,
      max_turns=1,
      timeout=5.0,
      parent_turn_id="tool-run-agent-1",
    )
  )

  assert error is None
  assert result is not None
  assert len(events) == 1
  assert events[0].session_id == "sub0:sess-parent"
  assert events[0].request_id == "req-123"
  assert events[0].parent_turn_id == "tool-run-agent-1"
  assert events[0].provider == "stub"


def test_run_appends_turn_complete_event_to_event_log() -> None:
  event_log = EventLog()
  durable_events: list[dict[str, Any]] = []
  provider = _UsageProvider()
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  async def _append_durable_event(event: dict[str, Any]):
    durable_events.append(dict(event))
    return SimpleNamespace(seq=len(durable_events))

  runner._append_durable_event = _append_durable_event

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  turn_complete = [entry.event for entry in event_log.entries if entry.event.get("type") == "turn_complete"]
  assistant_messages = [event for event in durable_events if event.get("type") == "assistant_message"]
  assert len(turn_complete) == 1
  assert len(assistant_messages) == 1
  assert turn_complete[0]["turn"] == 1
  assert turn_complete[0]["usage"] == {
    "input_tokens": 100,
    "output_tokens": 50,
    "reasoning_tokens_observed": 0,
    "provider_units": 0,
    "provider_unit_deltas": {},
    "cache_read_input_tokens": 10,
    "cache_creation_input_tokens": 5,
    "estimated_cost": 0.0002,
    "capability_bind": runner.capability_execution.bind.to_json(),
  }
  assert assistant_messages[0]["model"] == "claude-sonnet-4-6"
  assert assistant_messages[0]["provider"] == "stub"


def test_native_compaction_block_counts_once_in_session_summary(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  class _NativeCompactionProvider(_UsageProvider):
    def __init__(self) -> None:
      self.calls = 0
      self.system_prompts: list[Any] = []

    def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
      self.system_prompts.append(kwargs["system_prompt"])
      return {}

    async def stream(self, client: Any, params: dict[str, Any]):
      _ = client, params
      self.calls += 1
      yield StreamEvent(type="message_start", input_tokens=100)
      if self.calls == 1:
        yield StreamEvent(
          type="compaction",
          raw_block={
            "type": "compaction",
            "content": "native summary",
          },
        )
      else:
        yield StreamEvent(type="text_delta", text="done")
        yield StreamEvent(
          type="text_end",
          raw_block={"type": "text", "text": "done"},
        )
      yield StreamEvent(type="usage_update", output_tokens=5)
      yield StreamEvent(
        type="message_end",
        stop_reason=(
          "compaction"
          if self.calls == 1
          else "end_turn"
        ),
      )

  summaries: list[SessionUsageSummary] = []
  event_log = EventLog()
  durable_log = AgentSessionLog(
    path=tmp_path / "native-compaction.jsonl"
  )
  provider = _NativeCompactionProvider()
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-native-compaction",
    capability_execution=_runner_execution(provider),
    user_id="alice",
    request_id="req-native-compaction",
    on_session_summary=summaries.append,
    billing_mode="byok",
    rate_table_version="unknown",
    agent_session_log=durable_log,
    compaction_trigger=None,
  )
  monkeypatch.setattr(
    gateway_runner,
    "_model_context_window",
    lambda _model_info: 1_000,
  )

  def token_snapshot(
    *,
    system_text: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
  ) -> SimpleNamespace:
    return SimpleNamespace(
      system_text=system_text,
      messages_text="",
      tools_text="",
      system_chars=len(system_text),
      tools_chars=0,
      est_system_tokens=1,
      est_messages_tokens=599,
      est_tools_tokens=0,
      est_total_tokens=600,
      message_count=len(messages),
      tool_count=len(tools),
    )

  monkeypatch.setattr(
    gateway_runner,
    "_token_estimate_snapshot",
    token_snapshot,
  )

  _run(
    runner.run(
      messages=[{
        "role": "user",
        "content": "trigger native compaction",
      }],
      system_prompt="base prompt",
      max_turns=2,
    )
  )

  assert len(summaries) == 1
  assert summaries[0].compaction_count == 1
  assert provider.calls == 2
  assert all(
    "Context at 60%" in str(prompt)
    for prompt in provider.system_prompts
  )
  assert sum(
    entry.event.get("type") == "compaction"
    for entry in event_log.entries
  ) == 1
  assistant_messages, _ = _run(
    durable_log.query(
      event_types={"assistant_message"},
      order="asc",
    )
  )
  assert assistant_messages[0].event["content_blocks"] == [{
    "type": "compaction",
    "content": "native summary",
  }]


def test_runner_emits_session_summary_once_after_run() -> None:
  summaries: list[SessionUsageSummary] = []
  provider = _UsageProvider()
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    user_id="alice",
    request_id="req-summary",
    channel="web",
    on_session_summary=summaries.append,
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  assert len(summaries) == 1
  summary = summaries[0]
  assert summary.user_id == "alice"
  assert summary.session_id == "sess-parent"
  assert summary.request_id == "req-summary"
  assert summary.input_tokens == 100
  assert summary.output_tokens == 50
  assert summary.cache_read_tokens == 10
  assert summary.cache_creation_tokens == 5
  assert summary.cost == pytest.approx(0.00020875)
  assert summary.turns == 1
  assert summary.channel == "web"
  assert summary.rate_table_version == "unknown"
  assert summary.billing_mode == "byok"
  assert summary.drain_complete is True
  assert summary.in_flight_task_count == 0
  assert summary.compaction_count == 0
  assert summary.model == "claude-sonnet-4-6"
  assert summary.provider == "stub"


def test_runner_session_summary_reports_failed_drain_and_in_flight_tasks() -> None:
  class _DrainFailureRunner(AgentRunner):
    async def _shutdown_background_tasks(self, was_cancelled: bool) -> None:
      _ = was_cancelled
      raise RuntimeError("drain failed")

  async def case() -> None:
    summaries: list[SessionUsageSummary] = []
    pending_task = asyncio.create_task(asyncio.Event().wait())
    task_registry = TaskRegistry()
    pending_entry = task_registry.register("run_agent", task_id="bg-pending")
    pending_entry.asyncio_task = pending_task
    pending_entry.started_at = 0.0
    task_registry.transition("bg-pending", TaskState.RUNNING)

    provider = _UsageProvider()
    runner = _DrainFailureRunner(
      event_log=EventLog(),
      dispatcher=_make_dispatcher(),
      session_id="sess-parent",
      capability_execution=_runner_execution(provider),
      user_id="alice",
      request_id="req-summary-drain",
      on_session_summary=summaries.append,
      billing_mode="byok",
      rate_table_version="unknown",
    )
    runner._task_registry = task_registry

    try:
      with pytest.raises(RuntimeError, match="drain failed"):
        await runner.run(
          messages=[{"role": "user", "content": "hello"}],
          max_turns=1,
        )

      assert len(summaries) == 1
      assert summaries[0].drain_complete is False
      assert summaries[0].in_flight_task_count == 1
    finally:
      pending_task.cancel()
      await asyncio.gather(pending_task, return_exceptions=True)

  _run(case())


def test_runner_is_single_use() -> None:
  provider = _UsageProvider()
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  with pytest.raises(RuntimeError, match="single-use"):
    _run(runner.run(messages=[{"role": "user", "content": "hello again"}]))


def test_sub_runner_with_parent_aggregator_does_not_emit_own_summary() -> None:
  parent_summaries: list[SessionUsageSummary] = []
  child_summaries: list[SessionUsageSummary] = []
  parent_provider = _UsageProvider()
  parent_runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    capability_execution=_runner_execution(parent_provider),
    user_id="alice",
    request_id="req-parent",
    on_session_summary=parent_summaries.append,
    billing_mode="byok",
    rate_table_version="unknown",
  )
  child_provider = _UsageProvider()
  child_runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sub0:sess-parent",
    capability_execution=_runner_execution(child_provider),
    user_id="alice",
    request_id="req-parent",
    on_session_summary=child_summaries.append,
    _parent_aggregator=parent_runner._aggregator,
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(child_runner.run(messages=[{"role": "user", "content": "child"}]))
  parent_summary = _run(parent_runner._aggregator.snapshot())

  assert child_summaries == []
  assert parent_summary.input_tokens == 100
  assert parent_summary.turns == 1
  assert parent_summary.model == "claude-sonnet-4-6"
  assert parent_summary.provider == "stub"
  assert parent_summary.rate_table_version == "unknown"
  assert parent_summary.billing_mode == "byok"


@pytest.mark.parametrize("timeout", [0, None, -1])
def test_spawn_sub_agent_no_wall_clock(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  timeout: float | None,
) -> None:
  async def _unexpected_wait_for(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("asyncio.wait_for should not wrap non-positive timeouts")

  monkeypatch.setattr(gateway_runner.asyncio, "wait_for", _unexpected_wait_for)
  parent_provider = _UsageProvider()
  parent_runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    capability_execution=_runner_execution(parent_provider),
    agent_session_log=AgentSessionLog(path=tmp_path / "timeout-parent.jsonl"),
    workspace_dir=str(tmp_path),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  result, error = _run(
    parent_runner.spawn_sub_agent(
      "Collect usage",
      capability_execution=_child_execution(_UsageProvider()),
      **_subagent_identity("sub0:sess-parent"),
      skill_name="test-child",
      dispatcher=_make_dispatcher(),
      max_turns=1,
      timeout=timeout,
    )
  )

  assert error is None
  assert result is not None
  assert result.execution.status == "succeeded"
  assert result.values.terminal_narrative is not None


def test_failed_response_usage_reaches_runner_as_failed_billable() -> None:
  """Provider-level `response.failed` billing must survive the runner seam.

  r9 of the OpenAI Responses cutover flagged that finding 2 (billable usage lost on
  failed responses) was proven only inside the provider mapper. This covers the seam:
  the fixed provider yields `_terminal_events()` -- message_start, usage_update,
  message_end(error) -- and only THEN raises, so the runner must have already
  accumulated the tokens and must bill them as `failed_billable`, not drop them.
  """
  states: list[tuple[int, int, str]] = []

  class _FailedResponseProvider(_UsageProvider):
    async def stream(self, client: Any, params: dict[str, Any]):
      _ = client, params
      # Mirrors openai.py stream(): terminal events first, then the saved error.
      yield StreamEvent(type="message_start", input_tokens=100, cache_read_tokens=10)
      yield StreamEvent(type="usage_update", output_tokens=42)
      yield StreamEvent(type="message_end", stop_reason="error")
      raise RuntimeError("response.failed: server_error")

    def is_retryable_error(self, exc: Exception) -> bool:
      _ = exc
      return False

  class Producer:
    async def emit(self, event, *, usage_state="succeeded"):
      states.append((event.input_tokens, event.output_tokens, usage_state))

  event_log = EventLog()
  provider = _FailedResponseProvider()
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-failed-response",
    capability_execution=_runner_execution(provider),
    user_id="alice", request_id="req-failed", billing_mode="metered",
    rate_table_version="v1", channel="web",
    commercial_usage_producer=Producer(),
  )

  # The runner absorbs a non-retryable stream error (logs it, ends the turn) rather
  # than propagating -- so billing, not the exception, is the only signal that the
  # failed response was accounted for.
  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  # The nonzero usage carried by the failed response must be billed, not dropped.
  assert states, "failed response emitted no usage event at all"
  assert all(state == "failed_billable" for _, _, state in states), states
  assert any(output > 0 for _, output, _ in states), (
    f"failed response billed zero output tokens: {states}"
  )
