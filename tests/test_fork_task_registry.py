from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
import gc
from pathlib import Path
import weakref

import pytest

from agent_gateway.fork_ledger import ForkLedger
from agent_gateway.fork_task_registry import (
  ForkTaskRegistry,
  fork_handoff_max_bytes,
  learn_fork_budget_usd,
  learn_fork_daily_budget_usd,
  learn_fork_daily_invocation_quota,
  learn_fork_enabled,
  learn_fork_global_concurrency,
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
    owner_operated_interactive=True,
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
    return Decimal("0.50")

  ledger, registry, events = _registry(tmp_path, spawn)
  first = registry.submit(
    fork_id="fork-1",
    session_id="session-1",
    owner="owner-1",
    handoff={"messages": ["one"]},
  )
  second = registry.submit(
    fork_id="fork-2",
    session_id="session-1",
    owner="owner-1",
    handoff={"messages": ["two"]},
  )

  assert first.launched
  assert second.skip_reason == "session_cap"
  assert ledger.get_admission("fork-2") is None
  assert ("learning_fork_skipped", {
    "fork_id": "fork-2",
    "reason": "session_cap",
    "retained_bytes": None,
  }) in events
  release.set()
  await registry.shutdown()


@pytest.mark.asyncio
async def test_global_cap_skips_other_session_without_reservation(
  tmp_path: Path,
) -> None:
  release = asyncio.Event()

  async def spawn(_fork_id, _handoff):
    await release.wait()
    return 0

  ledger, registry, _events = _registry(
    tmp_path,
    spawn,
    global_concurrency=1,
  )
  assert registry.submit(
    fork_id="fork-1",
    session_id="session-1",
    owner="owner-1",
    handoff={},
  ).launched
  skipped = registry.submit(
    fork_id="fork-2",
    session_id="session-2",
    owner="owner-1",
    handoff={},
  )

  assert skipped.skip_reason == "global_concurrency"
  assert ledger.get_admission("fork-2") is None
  release.set()
  await registry.shutdown()


@pytest.mark.asyncio
async def test_oversized_handoff_is_skipped_never_truncated(
  tmp_path: Path,
) -> None:
  calls = []

  async def spawn(_fork_id, handoff):
    calls.append(handoff)
    return 0

  ledger, registry, events = _registry(
    tmp_path,
    spawn,
    handoff_max_bytes=32,
  )
  handoff = {"transcript": "x" * 100}

  decision = registry.submit(
    fork_id="oversized",
    session_id="session-1",
    owner="owner-1",
    handoff=handoff,
  )

  assert decision.skip_reason == "handoff_too_large"
  assert decision.retained_bytes is not None
  assert decision.retained_bytes > 32
  assert handoff["transcript"] == "x" * 100
  assert calls == []
  assert ledger.get_admission("oversized") is None
  assert events[-1][1]["reason"] == "handoff_too_large"


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
    return "0.25"

  ledger, registry, _events = _registry(tmp_path, spawn)

  async def request_task() -> None:
    assert registry.submit(
      fork_id="fork-1",
      session_id="session-1",
      owner="owner-1",
      handoff={"transcript": "retained"},
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
async def test_shutdown_waits_for_cooperative_task(
  tmp_path: Path,
) -> None:
  release = asyncio.Event()
  exited = asyncio.Event()

  async def spawn(_fork_id, _handoff):
    await release.wait()
    exited.set()
    return 0

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
    return 0

  _ledger_value, registry, _events = _registry(tmp_path, spawn)
  handoff = Handoff()
  reference = weakref.ref(handoff)
  registry.submit(
    fork_id="fork-1",
    session_id="session-1",
    owner="owner-1",
    handoff=handoff,
  )
  del handoff

  await registry.shutdown()
  gc.collect()

  assert registry.retained_handoff_count == 0
  assert reference() is None


@pytest.mark.asyncio
async def test_daily_ledger_refusals_skip_without_open_reservation(
  tmp_path: Path,
) -> None:
  release = asyncio.Event()

  async def spawn(_fork_id, _handoff):
    await release.wait()
    return 0

  ledger, registry, events = _registry(
    tmp_path,
    spawn,
    global_concurrency=3,
    daily_invocation_quota=1,
  )
  registry.submit(
    fork_id="admitted",
    session_id="session-1",
    owner="owner-1",
    handoff={},
  )
  quota = registry.submit(
    fork_id="quota-refused",
    session_id="session-2",
    owner="owner-1",
    handoff={},
  )
  assert quota.skip_reason == "daily_invocation_quota"
  assert ledger.get_admission("quota-refused") is None
  assert events[-1][1]["reason"] == "daily_invocation_quota"
  release.set()
  await registry.shutdown()


@pytest.mark.asyncio
async def test_daily_budget_refusal_skips_without_open_reservation(
  tmp_path: Path,
) -> None:
  async def spawn(_fork_id, _handoff):
    return 0

  ledger, registry, events = _registry(
    tmp_path,
    spawn,
    per_fork_budget_usd="2.00",
    daily_budget_usd="1.00",
  )

  decision = registry.submit(
    fork_id="budget-refused",
    session_id="session-1",
    owner="owner-1",
    handoff={},
  )

  assert decision.skip_reason == "daily_budget"
  assert ledger.get_admission("budget-refused") is None
  assert events[-1][1]["reason"] == "daily_budget"


def test_config_defaults_and_invalid_values(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  names = (
    "HANK_LEARN_FORK_ENABLED",
    "HANK_LEARN_FORK_BUDGET_USD",
    "HANK_LEARN_FORK_DAILY_BUDGET_USD",
    "HANK_LEARN_FORK_DAILY_INVOCATION_QUOTA",
    "HANK_LEARN_FORK_GLOBAL_CONCURRENCY",
    "HANK_FORK_HANDOFF_MAX_BYTES",
  )
  for name in names:
    monkeypatch.delenv(name, raising=False)

  assert not learn_fork_enabled()
  assert learn_fork_enabled(owner_operated_interactive=True)
  assert learn_fork_budget_usd() == Decimal("2.00")
  assert learn_fork_daily_budget_usd() == Decimal("10.00")
  assert learn_fork_daily_invocation_quota() == 12
  assert learn_fork_global_concurrency() == 2
  assert fork_handoff_max_bytes() == 33_554_432

  monkeypatch.setenv("HANK_LEARN_FORK_ENABLED", "maybe")
  with pytest.raises(ValueError, match="must be a boolean"):
    learn_fork_enabled()
  monkeypatch.setenv("HANK_LEARN_FORK_BUDGET_USD", "nan")
  with pytest.raises(ValueError, match="finite and positive"):
    learn_fork_budget_usd()
  monkeypatch.setenv("HANK_LEARN_FORK_GLOBAL_CONCURRENCY", "0")
  with pytest.raises(ValueError, match="positive integer"):
    learn_fork_global_concurrency()
