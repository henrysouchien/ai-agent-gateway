import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.heartbeat as heartbeat
from agent_gateway import HeartbeatConfig, HeartbeatLoop, RunOutput, TickResult, strip_heartbeat_ok


def _run(coro):
  return asyncio.run(coro)


def _output(
  response: str = "Action needed",
  *,
  error: str | None = None,
  timed_out: bool = False,
  budget_exceeded: bool = False,
  max_turns_reached: bool = False,
) -> RunOutput:
  return RunOutput(
    response=response,
    tools_used=[],
    usage={},
    error=error,
    timed_out=timed_out,
    budget_exceeded=budget_exceeded,
    max_turns_reached=max_turns_reached,
  )


def _sequence_run_fn(items: list[RunOutput | Exception]):
  remaining = list(items)

  async def _run_fn() -> RunOutput:
    item = remaining.pop(0)
    if isinstance(item, Exception):
      raise item
    return item

  return _run_fn


def test_strip_heartbeat_ok() -> None:
  cases = [
    ("HEARTBEAT_OK\n", "", True),
    ("Action needed\nHEARTBEAT_OK", "Action needed", True),
    ("HEARTBEAT_OK\nAction needed\nHEARTBEAT_OK", "Action needed", True),
    ("Action needed", "Action needed", False),
    ("Action\nHEARTBEAT_OK\nMore detail", "Action\nHEARTBEAT_OK\nMore detail", False),
    ("  HEARTBEAT_OK \nAction needed", "Action needed", True),
  ]

  for raw, expected_text, expected_found in cases:
    stripped, found = strip_heartbeat_ok(raw)
    assert stripped == expected_text
    assert found is expected_found


def test_strip_heartbeat_ok_whitespace_residue() -> None:
  stripped, found = strip_heartbeat_ok("HEARTBEAT_OK\n   \n")

  assert found is True
  assert stripped.strip() == ""


def test_is_checklist_empty(tmp_path: Path) -> None:
  empty_path = tmp_path / "empty.md"
  headers_only_path = tmp_path / "headers.md"
  content_path = tmp_path / "content.md"
  missing_path = tmp_path / "missing.md"

  empty_path.write_text("", encoding="utf-8")
  headers_only_path.write_text("# Heartbeat\n\n## Tasks\n", encoding="utf-8")
  content_path.write_text("# Heartbeat\n\n- Review the latest alert\n", encoding="utf-8")

  assert heartbeat.is_checklist_empty(empty_path) is True
  assert heartbeat.is_checklist_empty(headers_only_path) is True
  assert heartbeat.is_checklist_empty(content_path) is False
  assert heartbeat.is_checklist_empty(missing_path) is True


def test_is_within_active_hours() -> None:
  noon_utc = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
  early_utc = datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc)

  assert heartbeat.is_within_active_hours(None, "UTC", now=noon_utc) is True
  assert heartbeat.is_within_active_hours((9, 17), "UTC", now=noon_utc) is True
  assert heartbeat.is_within_active_hours((9, 17), "UTC", now=early_utc) is False
  assert heartbeat.is_within_active_hours((8, 8), "UTC", now=noon_utc) is False


def test_is_within_active_hours_overnight() -> None:
  late_utc = datetime(2024, 1, 1, 23, 0, tzinfo=timezone.utc)
  early_utc = datetime(2024, 1, 1, 5, 0, tzinfo=timezone.utc)
  noon_utc = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

  assert heartbeat.is_within_active_hours((22, 6), "UTC", now=late_utc) is True
  assert heartbeat.is_within_active_hours((22, 6), "UTC", now=early_utc) is True
  assert heartbeat.is_within_active_hours((22, 6), "UTC", now=noon_utc) is False


def test_is_within_active_hours_invalid_tz(caplog: pytest.LogCaptureFixture) -> None:
  noon_utc = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

  with caplog.at_level(logging.WARNING, logger="agent_gateway.heartbeat"):
    active = heartbeat.is_within_active_hours((9, 17), "Invalid/Timezone", now=noon_utc)

  assert active is True
  assert "Invalid heartbeat timezone" in caplog.text


def test_tick_skipped_outside_hours() -> None:
  async def case() -> None:
    results: list[TickResult] = []
    run_calls = 0

    async def run_fn() -> RunOutput:
      nonlocal run_calls
      run_calls += 1
      return _output()

    async def on_tick(result: TickResult, state: dict[str, Any]) -> None:
      results.append(result)
      assert state["last_outcome"] == "skipped"

    loop = HeartbeatLoop(
      run_fn,
      HeartbeatConfig(active_hours=(8, 8), max_ticks=1),
      on_tick=on_tick,
    )
    await loop.start()

    assert run_calls == 0
    assert len(results) == 1
    assert results[0].skipped is True
    assert results[0].skip_reason == "outside_active_hours"
    assert results[0].output is None
    assert loop.tick_count == 1
    assert loop.state["last_skip_reason"] == "outside_active_hours"

  _run(case())


def test_tick_skipped_empty_checklist(tmp_path: Path) -> None:
  checklist_path = tmp_path / "HEARTBEAT.md"
  checklist_path.write_text("# Heartbeat\n\n## Checklist\n", encoding="utf-8")

  async def case() -> None:
    results: list[TickResult] = []
    run_calls = 0

    async def run_fn() -> RunOutput:
      nonlocal run_calls
      run_calls += 1
      return _output()

    async def on_tick(result: TickResult, _state: dict[str, Any]) -> None:
      results.append(result)

    loop = HeartbeatLoop(
      run_fn,
      HeartbeatConfig(checklist_path=checklist_path, max_ticks=1),
      on_tick=on_tick,
    )
    await loop.start()

    assert run_calls == 0
    assert len(results) == 1
    assert results[0].skipped is True
    assert results[0].skip_reason == "empty_checklist"
    assert loop.state["last_skip_reason"] == "empty_checklist"

  _run(case())


def test_tick_alert_delivery() -> None:
  async def case() -> None:
    alert_calls: list[tuple[RunOutput, dict[str, Any]]] = []
    quiet_calls: list[RunOutput] = []
    error_calls: list[RunOutput | Exception] = []
    tick_results: list[TickResult] = []

    async def run_fn() -> RunOutput:
      return _output("Action needed now")

    async def on_alert(output: RunOutput, state: dict[str, Any]) -> None:
      alert_calls.append((output, state))

    async def on_quiet(output: RunOutput, _state: dict[str, Any]) -> None:
      quiet_calls.append(output)

    async def on_error(error: RunOutput | Exception, _state: dict[str, Any]) -> None:
      error_calls.append(error)

    async def on_tick(result: TickResult, _state: dict[str, Any]) -> None:
      tick_results.append(result)

    loop = HeartbeatLoop(
      run_fn,
      HeartbeatConfig(max_ticks=1),
      on_alert=on_alert,
      on_quiet=on_quiet,
      on_error=on_error,
      on_tick=on_tick,
    )
    await loop.start()

    assert len(alert_calls) == 1
    assert alert_calls[0][0].response == "Action needed now"
    assert alert_calls[0][1]["last_outcome"] == "alert"
    assert quiet_calls == []
    assert error_calls == []
    assert len(tick_results) == 1
    assert tick_results[0].alert is True
    assert tick_results[0].stripped_response == "Action needed now"
    assert loop.state["consecutive_errors"] == 0

  _run(case())


def test_tick_quiet_suppression() -> None:
  async def case() -> None:
    quiet_calls: list[RunOutput] = []
    alert_calls: list[RunOutput] = []
    tick_results: list[TickResult] = []

    async def run_fn() -> RunOutput:
      return _output("HEARTBEAT_OK\n   ")

    async def on_quiet(output: RunOutput, _state: dict[str, Any]) -> None:
      quiet_calls.append(output)

    async def on_alert(output: RunOutput, _state: dict[str, Any]) -> None:
      alert_calls.append(output)

    async def on_tick(result: TickResult, _state: dict[str, Any]) -> None:
      tick_results.append(result)

    loop = HeartbeatLoop(
      run_fn,
      HeartbeatConfig(max_ticks=1),
      on_alert=on_alert,
      on_quiet=on_quiet,
      on_tick=on_tick,
    )
    await loop.start()

    assert len(quiet_calls) == 1
    assert alert_calls == []
    assert len(tick_results) == 1
    assert tick_results[0].alert is False
    assert tick_results[0].stripped_response.strip() == ""
    assert loop.state["last_outcome"] == "quiet"

  _run(case())


def test_tick_ok_with_long_content() -> None:
  async def case() -> None:
    alert_calls: list[RunOutput] = []

    async def run_fn() -> RunOutput:
      return _output("HEARTBEAT_OK\n" + ("A" * 500))

    async def on_alert(output: RunOutput, _state: dict[str, Any]) -> None:
      alert_calls.append(output)

    loop = HeartbeatLoop(
      run_fn,
      HeartbeatConfig(max_ticks=1),
      on_alert=on_alert,
    )
    await loop.start()

    assert len(alert_calls) == 1
    assert loop.state["last_outcome"] == "alert"

  _run(case())


def test_tick_error_from_run_output() -> None:
  async def case() -> None:
    error_calls: list[RunOutput | Exception] = []
    tick_results: list[TickResult] = []

    async def run_fn() -> RunOutput:
      return _output(error="boom")

    async def on_error(error: RunOutput | Exception, _state: dict[str, Any]) -> None:
      error_calls.append(error)

    async def on_tick(result: TickResult, _state: dict[str, Any]) -> None:
      tick_results.append(result)

    loop = HeartbeatLoop(
      run_fn,
      HeartbeatConfig(max_ticks=1),
      on_error=on_error,
      on_tick=on_tick,
    )
    await loop.start()

    assert len(error_calls) == 1
    assert isinstance(error_calls[0], RunOutput)
    assert len(tick_results) == 1
    assert tick_results[0].error == "boom"
    assert loop.state["consecutive_errors"] == 1
    assert loop.state["last_outcome"] == "error"

  _run(case())


def test_tick_error_from_timeout() -> None:
  async def case() -> None:
    error_calls: list[RunOutput | Exception] = []
    tick_results: list[TickResult] = []

    async def run_fn() -> RunOutput:
      return _output(timed_out=True)

    async def on_error(error: RunOutput | Exception, _state: dict[str, Any]) -> None:
      error_calls.append(error)

    async def on_tick(result: TickResult, _state: dict[str, Any]) -> None:
      tick_results.append(result)

    loop = HeartbeatLoop(
      run_fn,
      HeartbeatConfig(max_ticks=1),
      on_error=on_error,
      on_tick=on_tick,
    )
    await loop.start()

    assert len(error_calls) == 1
    assert isinstance(error_calls[0], RunOutput)
    assert len(tick_results) == 1
    assert tick_results[0].error == "Heartbeat run timed out"
    assert loop.state["consecutive_errors"] == 1

  _run(case())


def test_tick_error_from_exception() -> None:
  async def case() -> None:
    error_calls: list[RunOutput | Exception] = []
    tick_results: list[TickResult] = []

    async def run_fn() -> RunOutput:
      raise RuntimeError("boom")

    async def on_error(error: RunOutput | Exception, _state: dict[str, Any]) -> None:
      error_calls.append(error)

    async def on_tick(result: TickResult, _state: dict[str, Any]) -> None:
      tick_results.append(result)

    loop = HeartbeatLoop(
      run_fn,
      HeartbeatConfig(max_ticks=1),
      on_error=on_error,
      on_tick=on_tick,
    )
    await loop.start()

    assert len(error_calls) == 1
    assert isinstance(error_calls[0], RuntimeError)
    assert len(tick_results) == 1
    assert tick_results[0].output is None
    assert tick_results[0].error == "RuntimeError: boom"
    assert loop.state["consecutive_errors"] == 1

  _run(case())


def test_tick_budget_exceeded_is_alert() -> None:
  async def case() -> None:
    alert_calls: list[RunOutput] = []
    error_calls: list[RunOutput | Exception] = []
    tick_results: list[TickResult] = []

    async def run_fn() -> RunOutput:
      return _output("HEARTBEAT_OK\n", budget_exceeded=True)

    async def on_alert(output: RunOutput, _state: dict[str, Any]) -> None:
      alert_calls.append(output)

    async def on_error(error: RunOutput | Exception, _state: dict[str, Any]) -> None:
      error_calls.append(error)

    async def on_tick(result: TickResult, _state: dict[str, Any]) -> None:
      tick_results.append(result)

    loop = HeartbeatLoop(
      run_fn,
      HeartbeatConfig(max_ticks=1),
      on_alert=on_alert,
      on_error=on_error,
      on_tick=on_tick,
    )
    await loop.start()

    assert len(alert_calls) == 1
    assert error_calls == []
    assert len(tick_results) == 1
    assert tick_results[0].alert is True
    assert tick_results[0].stripped_response.strip() == ""
    assert loop.state["last_outcome"] == "budget_exceeded"
    assert loop.state["consecutive_errors"] == 0

  _run(case())


def test_backoff_replaces_interval() -> None:
  async def case() -> None:
    tick_times: list[float] = []

    async def run_fn() -> RunOutput:
      return _output(error="boom")

    async def on_tick(result: TickResult, _state: dict[str, Any]) -> None:
      assert result.error == "boom"
      tick_times.append(asyncio.get_running_loop().time())

    loop = HeartbeatLoop(
      run_fn,
      HeartbeatConfig(interval_seconds=0.4, backoff_steps=[0.05], max_ticks=2),
      on_tick=on_tick,
    )
    await loop.start()

    assert len(tick_times) == 2
    delta = tick_times[1] - tick_times[0]
    assert 0.02 <= delta < 0.2
    assert loop.state["consecutive_errors"] == 2

  _run(asyncio.wait_for(case(), timeout=2.0))


def test_backoff_reset_on_success() -> None:
  async def case() -> None:
    tick_times: list[float] = []
    run_fn = _sequence_run_fn([
      _output(error="boom"),
      _output("Action needed"),
      _output("Action needed again"),
    ])

    async def on_tick(_result: TickResult, _state: dict[str, Any]) -> None:
      tick_times.append(asyncio.get_running_loop().time())

    loop = HeartbeatLoop(
      run_fn,
      HeartbeatConfig(interval_seconds=0.3, backoff_steps=[0.05], max_ticks=3),
      on_tick=on_tick,
    )
    await loop.start()

    assert len(tick_times) == 3
    first_gap = tick_times[1] - tick_times[0]
    second_gap = tick_times[2] - tick_times[1]
    assert 0.02 <= first_gap < 0.2
    assert second_gap >= 0.2
    assert loop.state["consecutive_errors"] == 0

  _run(asyncio.wait_for(case(), timeout=2.5))


def test_stop_interrupts_sleep() -> None:
  async def case() -> None:
    first_tick = asyncio.Event()
    run_calls = 0

    async def run_fn() -> RunOutput:
      nonlocal run_calls
      run_calls += 1
      return _output("Action needed")

    async def on_tick(result: TickResult, _state: dict[str, Any]) -> None:
      if result.tick_number == 1:
        first_tick.set()

    loop = HeartbeatLoop(
      run_fn,
      HeartbeatConfig(interval_seconds=10.0),
      on_tick=on_tick,
    )
    task = asyncio.create_task(loop.start())

    await asyncio.wait_for(first_tick.wait(), timeout=0.5)
    loop.stop()
    await asyncio.wait_for(task, timeout=0.5)

    assert run_calls == 1
    assert loop.running is False

  _run(asyncio.wait_for(case(), timeout=2.0))


def test_max_ticks() -> None:
  async def case() -> None:
    run_calls = 0
    tick_numbers: list[int] = []

    async def run_fn() -> RunOutput:
      nonlocal run_calls
      run_calls += 1
      return _output("Action needed")

    async def on_tick(result: TickResult, _state: dict[str, Any]) -> None:
      tick_numbers.append(result.tick_number)

    loop = HeartbeatLoop(
      run_fn,
      HeartbeatConfig(interval_seconds=0.01, max_ticks=3),
      on_tick=on_tick,
    )
    await loop.start()

    assert run_calls == 3
    assert tick_numbers == [1, 2, 3]
    assert loop.tick_count == 3

  _run(case())


def test_state_persistence(tmp_path: Path) -> None:
  state_path = tmp_path / "heartbeat_state.json"

  async def run_once() -> HeartbeatLoop:
    async def run_fn() -> RunOutput:
      return _output("Action needed")

    loop = HeartbeatLoop(
      run_fn,
      HeartbeatConfig(state_dir=tmp_path, max_ticks=1),
    )
    await loop.start()
    return loop

  first_loop = _run(run_once())
  first_state = json.loads(state_path.read_text(encoding="utf-8"))

  assert first_loop.tick_count == 1
  assert first_state["tick_count"] == 1
  assert first_state["last_outcome"] == "alert"

  second_loop = _run(run_once())
  second_state = json.loads(state_path.read_text(encoding="utf-8"))

  assert second_loop.tick_count == 2
  assert second_state["tick_count"] == 2
  assert second_state["consecutive_errors"] == 0


def test_callback_sync_and_async() -> None:
  async def case() -> None:
    callback_modes: list[str] = []

    async def sync_run_fn() -> RunOutput:
      return _output("Sync alert")

    def sync_on_alert(_output_value: RunOutput, _state: dict[str, Any]) -> None:
      callback_modes.append("sync")

    sync_loop = HeartbeatLoop(
      sync_run_fn,
      HeartbeatConfig(max_ticks=1),
      on_alert=sync_on_alert,
    )
    await sync_loop.start()

    async def async_run_fn() -> RunOutput:
      return _output("Async alert")

    async def async_on_alert(_output_value: RunOutput, _state: dict[str, Any]) -> None:
      callback_modes.append("async")

    async_loop = HeartbeatLoop(
      async_run_fn,
      HeartbeatConfig(max_ticks=1),
      on_alert=async_on_alert,
    )
    await async_loop.start()

    assert callback_modes == ["sync", "async"]

  _run(case())


def test_callback_exception_does_not_kill_loop(caplog: pytest.LogCaptureFixture) -> None:
  async def case() -> None:
    run_calls = 0
    alert_calls = 0
    tick_numbers: list[int] = []

    async def run_fn() -> RunOutput:
      nonlocal run_calls
      run_calls += 1
      return _output("Action needed")

    def bad_on_alert(_output_value: RunOutput, _state: dict[str, Any]) -> None:
      nonlocal alert_calls
      alert_calls += 1
      raise ValueError("broken callback")

    async def on_tick(result: TickResult, _state: dict[str, Any]) -> None:
      tick_numbers.append(result.tick_number)

    loop = HeartbeatLoop(
      run_fn,
      HeartbeatConfig(interval_seconds=0.01, max_ticks=2),
      on_alert=bad_on_alert,
      on_tick=on_tick,
    )
    await loop.start()

    assert run_calls == 2
    assert alert_calls == 2
    assert tick_numbers == [1, 2]

  with caplog.at_level(logging.WARNING, logger="agent_gateway.heartbeat"):
    _run(case())

  assert "Alert callback failed (non-fatal): broken callback" in caplog.text
