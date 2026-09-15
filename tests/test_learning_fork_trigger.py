from __future__ import annotations

import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_gateway.approval_policy import RunContext
from agent_gateway.fork_ledger import ForkLedger
from agent_gateway.fork_task_registry import ForkTaskRegistry
from agent_gateway.learning_fork_trigger import (
  claim_learning_receipts,
  evaluate_learning_fork_trigger,
  owner_operated_interactive_analyst,
  settle_learning_receipts,
  submit_learning_fork_after_turn,
)
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.runner_fork_agents import (
  LEARNING_FORK_ALLOWED_TOOLS,
  build_learning_fork_tool_decisions,
  cross_check_learning_memory_writes,
)
from agent_gateway.runner_notifications import build_notification_reminder
from agent_gateway.task_registry import NotificationQueue
from agent_gateway.tool_dispatcher import ToolDispatcher

ROOT = Path(__file__).resolve().parents[3]
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
  sys.path.insert(0, str(API_DIR))

import memory  # noqa: E402
from agent.interactive.tool_dispatcher import ExcelToolDispatcher  # noqa: E402
from agent.shared.learning_forks import configure_learning_forks  # noqa: E402
from agent.shared.learning_report import LearningReport  # noqa: E402
from agent.shared.tool_handlers import fork_memory_write  # noqa: E402
from agent.shared.tool_handlers.fork_memory_write import (  # noqa: E402
  scope_fork_memory_write_handler,
)
from excel_mcp.relay import ChannelRegistry  # noqa: E402


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
  configure_learning_forks(session)
  dispatcher = SimpleNamespace(
    _session=session,
    run_context=SimpleNamespace(profile=profile),
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


async def _unused_spawn(_fork_id, _handoff):
  raise AssertionError("no fork is launched here")


@pytest.mark.asyncio
async def test_excel_wrapper_is_eligible_and_fork_clone_writes_its_note(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The two seams that kept the fork wired shut on the Excel path.

  Eligibility must read the profile through the wrapper, and the fork clone
  must scope ``memory_write`` where dispatch actually reads it, so the note
  lands through the real wrapper while the parent keeps its stock handler.
  """

  workspace = tmp_path / "tenant-workspace"
  index_calls: list[Path] = []

  class _Store:
    def index_memory_file(self, file_path, memory_dir, *, metadata=None):
      del memory_dir, metadata
      index_calls.append(file_path)
      return {"indexed": True, "chunks": 1}

  monkeypatch.setattr(memory, "get_workspace_dir", lambda _user_id=None: workspace)
  monkeypatch.setattr(memory, "get_memory_store", lambda _user_id=None: _Store())
  monkeypatch.setattr(fork_memory_write, "_utc_date", lambda: "2026-09-05")
  stock_calls: list[dict[str, object]] = []

  async def stock_memory_write(tool_input, **_kwargs):
    stock_calls.append(dict(tool_input))
    return {"file": tool_input["file"]}, None

  base = ToolDispatcher(
    mcp_client=McpClientManager(config_path=None),
    local_tool_handlers={"memory_write": stock_memory_write},
    role="owner",
    run_context=RunContext(
      user_id="owner-1",
      request_id="turn-1",
      profile="analyst",
    ),
  )

  async def _no_addin_route(_request):
    raise AssertionError("no Excel route is exercised")

  wrapper = ExcelToolDispatcher(
    base=base,
    channel_registry=ChannelRegistry(),
    execute_addin=_no_addin_route,
  )
  ledger = _ledger(tmp_path)
  registry = ForkTaskRegistry(ledger, spawn_fork=_unused_spawn, enabled=True)
  runner = _runner(ledger, registry)
  runner._dispatcher = wrapper

  assert owner_operated_interactive_analyst(runner, runner._gateway_session)

  fork_id = "learn-0123abcd"
  fork_dispatcher = runner._dispatcher.with_scoped_local_handler(
    "memory_write",
    lambda stock: scope_fork_memory_write_handler(
      stock,
      fork_id=fork_id,
      user_id="owner-1",
    ),
  )
  assert isinstance(fork_dispatcher, ExcelToolDispatcher)
  assert fork_dispatcher.run_context is wrapper.run_context

  result, error = await fork_dispatcher.dispatch(
    "call-1",
    "memory_write",
    {"file": "daily/requested.md", "content": "learned: prefer FCF yield"},
  )
  assert error is None
  assert result is not None
  note = workspace / "notes" / result["file"]
  assert note.name == f"2026-09-05-{fork_id}-1.md"
  assert "learned: prefer FCF yield" in note.read_text(encoding="utf-8")
  assert index_calls == [note]
  assert stock_calls == []

  result, error = await wrapper.dispatch(
    "call-2",
    "memory_write",
    {"file": "daily/parent.md", "content": "parent write"},
  )
  assert error is None
  assert result == {"file": "daily/parent.md"}
  assert stock_calls == [{"file": "daily/parent.md", "content": "parent write"}]
  assert sorted(p.name for p in (workspace / "notes" / "learning").glob("*.md")) == [
    f"2026-09-05-{fork_id}-1.md",
  ]

  with pytest.raises(KeyError):
    wrapper.with_scoped_local_handler("file_write", lambda stock: stock)
