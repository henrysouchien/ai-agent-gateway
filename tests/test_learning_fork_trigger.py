from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_gateway.fork_ledger import ForkLedger
from agent_gateway.fork_task_registry import ForkTaskRegistry
from agent_gateway.learning_fork_trigger import (
  claim_learning_receipts,
  evaluate_learning_fork_trigger,
  owner_operated_interactive_analyst,
  settle_learning_receipts,
  submit_learning_fork_after_turn,
)
from agent_gateway.runner_fork_agents import (
  LEARNING_FORK_ALLOWED_TOOLS,
  build_learning_fork_tool_decisions,
  cross_check_learning_memory_writes,
)
from agent_gateway.runner_notifications import build_notification_reminder
from agent_gateway.sub_agent_result_contract import LearningReport
from agent_gateway.task_registry import NotificationQueue


def _clock() -> int:
  return int(
    datetime(2026, 7, 27, 12, tzinfo=timezone.utc).timestamp()
    * 1_000_000_000
  )


def _ledger(tmp_path: Path) -> ForkLedger:
  return ForkLedger(
    tmp_path / "fork-ledger.sqlite3",
    process_instance_id="test-process",
    clock_ns=_clock,
  )


def _runner(
  ledger: ForkLedger,
  registry: ForkTaskRegistry,
  *,
  role: str = "owner",
  billing_mode: str = "metered",
  principal: str = "service",
  profile: str = "analyst",
) -> SimpleNamespace:
  session = SimpleNamespace(
    session_id="session-1",
    user_id="owner-1",
    owner_user_id="owner-1",
    role=role,
    learn_memory_nudge_turns=0,
    learn_skill_nudge_iters=0,
    learning_fork_ledger=ledger,
    learning_fork_registry=registry,
  )
  dispatcher = SimpleNamespace(
    _session=session,
    _run_context=SimpleNamespace(profile=profile),
  )
  return SimpleNamespace(
    _gateway_session=session,
    _dispatcher=dispatcher,
    _capability_execution=SimpleNamespace(
      bind=SimpleNamespace(
        credential_principal=principal,
        run_mode="interactive",
      ),
    ),
    _billing_mode=billing_mode,
    _fork_mode=False,
    _request_id="turn-1",
    _notification_queue=NotificationQueue(),
    _on_metric=None,
  )


def test_hermes_counters_increment_reset_disable_and_trip_combined() -> None:
  first = evaluate_learning_fork_trigger(
    memory_turns=0,
    skill_iters=0,
    tool_calling_iters=1,
    foreground_memory_write=False,
    completed=True,
    real_final_response=True,
    errored=False,
    aborted=False,
    cancelled=False,
    enabled=True,
    memory_threshold=2,
    skill_threshold=2,
  )
  assert (first.memory_turns, first.skill_iters) == (1, 1)
  assert not first.should_submit

  second = evaluate_learning_fork_trigger(
    memory_turns=first.memory_turns,
    skill_iters=first.skill_iters,
    tool_calling_iters=1,
    foreground_memory_write=False,
    completed=True,
    real_final_response=True,
    errored=False,
    aborted=False,
    cancelled=False,
    enabled=True,
    memory_threshold=2,
    skill_threshold=2,
  )
  assert second.should_submit
  assert second.reason == "tripped"

  memory_reset = evaluate_learning_fork_trigger(
    memory_turns=9,
    skill_iters=7,
    tool_calling_iters=2,
    foreground_memory_write=True,
    completed=True,
    real_final_response=True,
    errored=False,
    aborted=False,
    cancelled=False,
    enabled=True,
    memory_threshold=10,
    skill_threshold=10,
  )
  assert memory_reset.memory_turns == 0
  assert memory_reset.skill_iters == 9

  disabled = evaluate_learning_fork_trigger(
    memory_turns=20,
    skill_iters=20,
    tool_calling_iters=1,
    foreground_memory_write=False,
    completed=True,
    real_final_response=True,
    errored=False,
    aborted=False,
    cancelled=False,
    enabled=True,
    memory_threshold=0,
    skill_threshold=10,
  )
  assert not disabled.should_submit
  assert disabled.reason == "counter_disabled"


@pytest.mark.parametrize("terminal_flag", ("errored", "aborted", "cancelled"))
def test_no_fire_or_counter_advance_for_unsuccessful_turns(
  terminal_flag: str,
) -> None:
  flags = {"errored": False, "aborted": False, "cancelled": False}
  flags[terminal_flag] = True
  decision = evaluate_learning_fork_trigger(
    memory_turns=10,
    skill_iters=10,
    tool_calling_iters=3,
    foreground_memory_write=False,
    completed=True,
    real_final_response=True,
    enabled=True,
    memory_threshold=10,
    skill_threshold=10,
    **flags,
  )
  assert not decision.should_submit
  assert decision.reason == terminal_flag
  assert (decision.memory_turns, decision.skill_iters) == (10, 10)


def test_learning_policy_is_closed_and_fail_closed() -> None:
  wire = [
    {"name": name}
    for name in sorted({
      *LEARNING_FORK_ALLOWED_TOOLS,
      "memory_store",
      "memory_delete",
      "memory_sync",
      "invoke_skill",
      "run_agent",
      "unclassified_future_tool",
    })
  ]
  decisions = {
    item.tool: item.decision
    for item in build_learning_fork_tool_decisions(wire)
  }
  assert {
    name for name, decision in decisions.items() if decision == "allow"
  } == LEARNING_FORK_ALLOWED_TOOLS
  assert all(
    decisions[name] == "deny"
    for name in {
      "memory_store",
      "memory_delete",
      "memory_sync",
      "invoke_skill",
      "run_agent",
      "unclassified_future_tool",
    }
  )


def test_memory_write_claim_without_event_evidence_adds_caveat() -> None:
  payload = {
    "summary": "Reviewed the session.",
    "findings": [],
    "artifacts": [],
    "caveats": [],
    "decision": "memory_update",
    "memory_writes": [
      {"path": "learning/notes/claimed.md", "summary": "Preference"}
    ],
    "skill_draft_candidate": None,
    "rationale": "A durable preference was present.",
  }
  checked = cross_check_learning_memory_writes(payload, ())

  assert checked["memory_writes"] == payload["memory_writes"]
  assert len(checked["caveats"]) == 1
  assert "claimed without evidence" in checked["caveats"][0]
  validated = LearningReport.model_validate(checked)
  assert validated.decision == "memory_update"


@pytest.mark.asyncio
async def test_receipt_claim_revert_redelivery_ack_and_parent_isolation(
  tmp_path: Path,
) -> None:
  async def spawn(_fork_id, _handoff):
    return Decimal("0")

  ledger = _ledger(tmp_path)
  registry = ForkTaskRegistry(
    ledger,
    spawn_fork=spawn,
    enabled=True,
  )
  assert ledger.write_receipt(
    fork_id="fork-1",
    session_id="session-1",
    owner="owner-1",
    receipt_text="Self-learning fork: drafted skill 'durable-x' (pending review)",
  )
  first_runner = _runner(ledger, registry)
  first = claim_learning_receipts(first_runner)
  assert first is not None
  reminder = build_notification_reminder(
    first_runner._notification_queue,
    max_count=5,
  )
  assert "Self-learning fork:" in reminder
  assert "SECRET DRAFT BODY" not in reminder
  assert first_runner._notification_queue.pending_count == 1

  settle_learning_receipts(first_runner, first, success=False)
  second_runner = _runner(ledger, registry)
  second_runner._request_id = "turn-2"
  second = claim_learning_receipts(second_runner)
  assert second is not None
  assert [claim.fork_id for claim in second.claims] == ["fork-1"]
  assert claim_learning_receipts(second_runner) is None

  settle_learning_receipts(second_runner, second, success=True)
  third_runner = _runner(ledger, registry)
  third_runner._request_id = "turn-3"
  assert claim_learning_receipts(third_runner) is None


@pytest.mark.asyncio
async def test_launch_resets_only_skill_counter_and_survives_caller_cancel(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import asyncio

  started = asyncio.Event()
  release = asyncio.Event()

  async def spawn(_fork_id, handoff):
    started.set()
    await release.wait()
    handoff.receipt_text = "Self-learning fork: nothing to save"
    return Decimal("0")

  ledger = _ledger(tmp_path)
  registry = ForkTaskRegistry(
    ledger,
    spawn_fork=spawn,
    enabled=True,
  )
  runner = _runner(ledger, registry)
  monkeypatch.setenv("HANK_LEARN_MEMORY_NUDGE_TURNS", "1")
  monkeypatch.setenv("HANK_LEARN_SKILL_NUDGE_ITERS", "1")

  async def request() -> None:
    launch = submit_learning_fork_after_turn(
      runner,
      handoff={"messages": ["parent only"]},
      tool_calling_iters=1,
      foreground_memory_write=False,
      completed=True,
      real_final_response=True,
      errored=False,
      aborted=False,
      cancelled=False,
    )
    assert launch is not None and launch.launched
    await asyncio.Future()

  caller = asyncio.create_task(request())
  await started.wait()
  caller.cancel()
  with pytest.raises(asyncio.CancelledError):
    await caller
  assert registry.active_count == 1
  assert runner._gateway_session.learn_memory_nudge_turns == 1
  assert runner._gateway_session.learn_skill_nudge_iters == 0

  release.set()
  await registry.shutdown()


@pytest.mark.asyncio
async def test_trigger_failure_is_best_effort_and_non_owner_byok_default_off(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls = []

  async def spawn(_fork_id, _handoff):
    calls.append(_fork_id)
    return Decimal("0")

  ledger = _ledger(tmp_path)
  registry = ForkTaskRegistry(
    ledger,
    spawn_fork=spawn,
    enabled=True,
  )
  owner = _runner(ledger, registry)
  assert owner_operated_interactive_analyst(owner, owner._gateway_session)
  for runner in (
    _runner(ledger, registry, role="invite"),
    _runner(ledger, registry, billing_mode="byok"),
    _runner(ledger, registry, principal="user"),
    _runner(ledger, registry, profile="advisor"),
  ):
    assert not owner_operated_interactive_analyst(
      runner,
      runner._gateway_session,
    )

  monkeypatch.setenv("HANK_LEARN_MEMORY_NUDGE_TURNS", "1")
  monkeypatch.setenv("HANK_LEARN_SKILL_NUDGE_ITERS", "1")
  monkeypatch.setenv("HANK_LEARN_FORK_ENABLED", "0")
  assert submit_learning_fork_after_turn(
    owner,
    handoff={},
    tool_calling_iters=1,
    foreground_memory_write=False,
    completed=True,
    real_final_response=True,
    errored=False,
    aborted=False,
    cancelled=False,
  ) is None
  assert calls == []

  monkeypatch.delenv("HANK_LEARN_FORK_ENABLED")
  monkeypatch.setenv("HANK_LEARN_MEMORY_NUDGE_TURNS", "invalid")
  assert submit_learning_fork_after_turn(
    owner,
    handoff={},
    tool_calling_iters=1,
    foreground_memory_write=False,
    completed=True,
    real_final_response=True,
    errored=False,
    aborted=False,
    cancelled=False,
  ) is None
