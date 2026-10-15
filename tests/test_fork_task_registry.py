from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
import gc
from pathlib import Path
import weakref

import pytest

from agent_gateway.fork_ledger import ForkLedger
from agent_gateway import AgentSessionLog
from agent_gateway.final_narrative_artifact import publish_final_narrative
from agent_gateway.runner_background_tasks import (
  task_completed_event_payload,
  task_correlation_payload,
  task_registered_event_payload,
)
from agent_gateway.sub_agent_narrative_result import task_result_from_execution
from agent_gateway.task_registry import TaskEntry, TaskRegistry, task_state_for_result
from agent_workflow_contracts import (
  AgentOperationRef,
  AttemptRef,
  OrdinaryDelegationTaskRef,
  OutcomeRequirement,
  ResultRequirement,
  sha256_digest,
)
from gateway_test_support.admitted_authority_test_support import (
  provenance_of,
  sealed_admitted_task,
)
from agent_gateway.fork_task_registry import (
  ForkTaskRegistry,
  learn_fork_cost_estimate_usd,
)


def _clock() -> int:
  return int(
    datetime(2026, 7, 27, 12, tzinfo=timezone.utc).timestamp()
    * 1_000_000_000
  )


def _ledger(tmp_path: Path) -> ForkLedger:
  return ForkLedger(
    tmp_path / "fork-ledger.sqlite3",
    process_instance_id="boot-1",
    clock_ns=_clock,
  )


def _completed_fork_entry(tmp_path: Path, fork_id: str, cost: Decimal | float) -> TaskEntry:
  logical = OrdinaryDelegationTaskRef(
    delegation_id=fork_id,
    operation=AgentOperationRef(
      namespace="agent-operation", name="test-fork", version="1.0",
      digest=sha256_digest({"operation": "test-fork"}),
    ),
  )
  attempt = AttemptRef(
    physical_task_id=fork_id, attempt_id=f"{fork_id}:attempt:1", attempt_number=1,
  )
  requirement = ResultRequirement(
    mode="narrative", terminal_narrative="required",
    outcome=OutcomeRequirement(required=False, source="none"),
  )
  admitted = sealed_admitted_task(
    logical_task=logical, attempt=attempt, result_requirement=requirement,
  )
  assert admitted.model_bind is not None
  entry = TaskEntry(
    task_id=fork_id, task_type="agent", admitted_task=admitted,
    capability_bind_receipt=admitted.model_bind.to_json(),
    metadata={"admitted_task": admitted.model_dump(mode="json")},
  )
  journal = AgentSessionLog(tmp_path / f"{fork_id}.jsonl")
  correlation = task_correlation_payload(entry, runner_id=None, role="writer")
  registration = journal.append_sync(task_registered_event_payload(
    entry, correlation_payload=correlation, agent_name="test-fork",
    parent_session_id="session-1",
  ))
  usage = journal.append_sync({
    "type": "turn_complete", "usage": {"cost_usd": float(cost)},
  })
  text = "Completed the inherited learning task."
  terminal_message = journal.append_sync({
    "type": "assistant_message", "stop_reason": "end_turn",
    "content_blocks": [{"type": "text", "text": text}],
  })
  terminal = journal.append_sync({"type": "stream_complete"})
  narrative = publish_final_narrative(
    workspace_dir=tmp_path, sub_agent_id=fork_id,
    terminal_event_seq=terminal_message.seq, text=text,
  )
  result = task_result_from_execution(
    (usage, terminal_message, terminal),
    logical_task=logical, attempt=attempt, requirement=requirement,
    provenance=provenance_of(admitted), final_narrative=narrative,
    stalled=False, admitted_task=admitted,
  )
  entry.result = result.model_dump(mode="json")
  completion = journal.append_sync(task_completed_event_payload(
    entry, task_state_for_result(result), correlation_payload=correlation,
    completed_at=entry.started_at,
  ))
  registry = TaskRegistry()
  registry.load_from_events([registration.event, completion.event])
  restored = registry.get(fork_id)
  assert restored is not None
  return restored


def _registry(
  tmp_path: Path,
  spawn,
  **kwargs,
) -> tuple[ForkLedger, ForkTaskRegistry, list[tuple[str, dict]]]:
  ledger = _ledger(tmp_path)
  events: list[tuple[str, dict]] = []
  registry = ForkTaskRegistry(
    ledger,
    spawn_fork=spawn,
    telemetry=lambda event, fields: events.append((event, dict(fields))),
    **kwargs,
  )
  return ledger, registry, events


@pytest.mark.asyncio
async def test_per_session_cap_skips_second_without_queue_or_reservation(
  tmp_path: Path,
) -> None:
  release = asyncio.Event()

  async def spawn(_fork_id, _handoff):
    await release.wait()
    return _completed_fork_entry(tmp_path, _fork_id, Decimal("0.50"))

  ledger, registry, events = _registry(tmp_path, spawn)
  first = registry.submit(
    fork_id="fork-1",
    session_id="session-1",
    owner="owner-1",
    handoff={"messages": ["one"]},
    on_settled=None,
  )
  second = registry.submit(
    fork_id="fork-2",
    session_id="session-1",
    owner="owner-1",
    handoff={"messages": ["two"]},
    on_settled=None,
  )

  assert first.launched
  assert second.skip_reason == "session_cap"
  assert ledger.get_admission("fork-2") is None
  assert ("learning_fork_skipped", {
    "fork_id": "fork-2",
    "reason": "session_cap",
  }) in events
  release.set()
  await registry.shutdown()


@pytest.mark.asyncio
async def test_concurrent_sessions_launch_and_emit_in_flight_count(
  tmp_path: Path,
) -> None:
  release = asyncio.Event()

  async def spawn(_fork_id, _handoff):
    await release.wait()
    return _completed_fork_entry(tmp_path, _fork_id, 0)

  ledger, registry, events = _registry(tmp_path, spawn)
  for index in range(3):
    assert registry.submit(
      fork_id=f"fork-{index}",
      session_id=f"session-{index}",
      owner="owner-1",
      handoff={},
      on_settled=None,
    ).launched

  launched = [fields for event, fields in events if event == "learning_fork_launched"]
  assert [fields["in_flight"] for fields in launched] == [1, 2, 3]
  assert ledger.get_admission("fork-2") is not None
  release.set()
  await registry.shutdown()


@pytest.mark.asyncio
async def test_large_handoff_launches_whole_and_emits_its_bytes(
  tmp_path: Path,
) -> None:
  calls = []

  async def spawn(_fork_id, handoff):
    calls.append(handoff)
    return _completed_fork_entry(tmp_path, _fork_id, 0)

  _ledger_, registry, events = _registry(tmp_path, spawn)
  handoff = {"transcript": "x" * 100_000}

  decision = registry.submit(
    fork_id="large",
    session_id="session-1",
    owner="owner-1",
    handoff=handoff,
    on_settled=None,
  )
  await registry.shutdown()

  assert decision.launched
  assert decision.retained_bytes is not None
  assert decision.retained_bytes > 100_000
  assert calls == [handoff]
  assert handoff["transcript"] == "x" * 100_000
  launched = next(
    fields for event, fields in events if event == "learning_fork_launched"
  )
  assert launched["retained_bytes"] == decision.retained_bytes
  # The decision carries every count the event reports, for the session log.
  assert (
    decision.in_flight,
    decision.day_invocations,
    decision.day_charged_microusd,
  ) == (
    launched["in_flight"],
    launched["day_invocations"],
    launched["day_charged_microusd"],
  )


@pytest.mark.asyncio
async def test_owned_task_survives_request_task_cancellation(
  tmp_path: Path,
) -> None:
  started = asyncio.Event()
  release = asyncio.Event()
  completed = asyncio.Event()

  async def spawn(_fork_id, _handoff):
    started.set()
    await release.wait()
    completed.set()
    return _completed_fork_entry(tmp_path, _fork_id, Decimal("0.25"))

  ledger, registry, _events = _registry(tmp_path, spawn)

  async def request_task() -> None:
    assert registry.submit(
      fork_id="fork-1",
      session_id="session-1",
      owner="owner-1",
      handoff={"transcript": "retained"},
      on_settled=None,
    ).launched
    await asyncio.Future()

  caller = asyncio.create_task(request_task())
  await started.wait()
  caller.cancel()
  with pytest.raises(asyncio.CancelledError):
    await caller
  assert registry.active_count == 1

  release.set()
  await completed.wait()
  await registry.shutdown()
  record = ledger.get_admission("fork-1")
  assert record is not None
  assert record.state == "settled"
  assert record.settled_usd == Decimal("0.250000")


@pytest.mark.asyncio
async def test_fork_spend_above_estimate_settles_exactly(
  tmp_path: Path,
) -> None:
  async def spawn(_fork_id, _handoff):
    return _completed_fork_entry(tmp_path, _fork_id, Decimal("5.125"))

  ledger, registry, events = _registry(
    tmp_path,
    spawn,
    per_fork_cost_estimate_usd="2.00",
  )
  assert registry.submit(
    fork_id="fork-1",
    session_id="session-1",
    owner="owner-1",
    handoff={},
    on_settled=None,
  ).launched
  await registry.shutdown()

  record = ledger.get_admission("fork-1")
  assert record is not None
  assert record.state == "settled"
  assert record.max_reserved_usd == Decimal("2.000000")
  assert record.settled_usd == Decimal("5.125000")
  assert ("learning_fork_completed", {"fork_id": "fork-1"}) in events



@pytest.mark.asyncio
async def test_shutdown_waits_for_cooperative_task(
  tmp_path: Path,
) -> None:
  release = asyncio.Event()
  exited = asyncio.Event()

  async def spawn(_fork_id, _handoff):
    await release.wait()
    exited.set()
    return _completed_fork_entry(tmp_path, _fork_id, 0)

  _ledger_value, registry, _events = _registry(
    tmp_path,
    spawn,
    shutdown_timeout_seconds=1,
  )
  registry.submit(
    fork_id="fork-1",
    session_id="session-1",
    owner="owner-1",
    handoff={},
    on_settled=None,
  )
  shutdown = asyncio.create_task(registry.shutdown())
  await asyncio.sleep(0)
  assert not shutdown.done()
  release.set()
  await shutdown
  assert exited.is_set()
  assert registry.active_count == 0


@pytest.mark.asyncio
async def test_shutdown_cancels_at_bound_and_abandons(
  tmp_path: Path,
) -> None:
  cancelled = asyncio.Event()

  async def spawn(_fork_id, _handoff):
    try:
      await asyncio.Future()
    except asyncio.CancelledError:
      cancelled.set()
      raise

  ledger, registry, _events = _registry(
    tmp_path,
    spawn,
    shutdown_timeout_seconds=0.02,
  )
  registry.submit(
    fork_id="fork-1",
    session_id="session-1",
    owner="owner-1",
    handoff={},
    on_settled=None,
  )

  await asyncio.wait_for(registry.shutdown(), timeout=0.5)

  assert cancelled.is_set()
  assert registry.active_count == 0
  record = ledger.get_admission("fork-1")
  assert record is not None
  assert record.state == "abandoned"
  assert record.settled_usd == record.max_reserved_usd


@pytest.mark.asyncio
async def test_handoff_reference_released_on_completion(
  tmp_path: Path,
) -> None:
  class Handoff:
    pass

  async def spawn(_fork_id, _handoff):
    return _completed_fork_entry(tmp_path, _fork_id, 0)

  _ledger_value, registry, _events = _registry(tmp_path, spawn)
  handoff = Handoff()
  reference = weakref.ref(handoff)
  registry.submit(
    fork_id="fork-1",
    session_id="session-1",
    owner="owner-1",
    handoff=handoff,
    on_settled=None,
  )
  del handoff

  await registry.shutdown()
  gc.collect()

  assert registry.retained_handoff_count == 0
  assert reference() is None


@pytest.mark.asyncio
async def test_daily_invocations_and_charge_are_emitted_never_refused(
  tmp_path: Path,
) -> None:
  release = asyncio.Event()

  async def spawn(_fork_id, _handoff):
    await release.wait()
    return _completed_fork_entry(tmp_path, _fork_id, 0)

  ledger, registry, events = _registry(
    tmp_path,
    spawn,
    per_fork_cost_estimate_usd="2.00",
  )
  for index in range(13):
    assert registry.submit(
      fork_id=f"fork-{index}",
      session_id=f"session-{index}",
      owner="owner-1",
      handoff={},
      on_settled=None,
    ).launched

  launched = [fields for event, fields in events if event == "learning_fork_launched"]
  assert [fields["day_invocations"] for fields in launched] == list(range(1, 14))
  assert launched[-1]["day_charged_microusd"] == 13 * 2_000_000
  assert ledger.get_admission("fork-12") is not None
  release.set()
  await registry.shutdown()


def test_config_defaults_and_invalid_values(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv("HANK_LEARN_FORK_BUDGET_USD", raising=False)
  assert learn_fork_cost_estimate_usd() == Decimal("2.00")
  monkeypatch.setenv("HANK_LEARN_FORK_BUDGET_USD", "nan")
  with pytest.raises(ValueError, match="finite and positive"):
    learn_fork_cost_estimate_usd()
