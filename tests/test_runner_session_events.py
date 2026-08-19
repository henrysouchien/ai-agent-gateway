import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, AgentSessionLog  # noqa: E402
from agent_gateway.agent_session_log_records import EVENT_SCHEMA_VERSION  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.runner_session_lifecycle import RunnerSessionLifecycleMixin  # noqa: E402
from agent_gateway.product_config import gateway_product_id  # noqa: E402
from agent_gateway.runner_session_events import (  # noqa: E402
  build_assistant_message_event,
  build_attach_event,
  build_budget_exceeded_event,
  build_budget_exceeded_text_event,
  build_chat_done_log_data,
  build_context_pressure_reminder,
  build_context_warning_log_data,
  build_detach_event,
  build_error_event,
  build_interrupted_event,
  build_max_turns_reached_event,
  build_max_turns_text_event,
  build_orphan_tool_call_interrupted_events,
  build_operator_pause_event,
  build_run_error_event,
  build_runtime_guard_event,
  build_stream_complete_event,
  build_stream_retry_event,
  build_stub_response_events,
  build_tool_call_complete_event,
  build_tool_call_start_event,
  build_turn_complete_log_data,
  build_token_estimate_log_data,
  build_turn_complete_event,
  build_user_message_event,
  build_write_lease_metadata,
  durable_event_payload,
  release_write_lease,
  run_detach_reason,
  run_interrupted_reason,
  shutdown_interrupted_reason,
  write_lease_metadata,
)
from agent_gateway.skill_completion_wal import SkillCompletionWalCorruptError  # noqa: E402


def test_context_pressure_reminder_copy_is_model_actionable() -> None:
  assert build_context_pressure_reminder(pct=60) == (
    "Context at 60% — prefer delegating further reading; "
    "large results will spill."
  )


def _run(coro):
  return asyncio.run(coro)


def test_runner_session_lifecycle_methods_are_inherited_from_mixin() -> None:
  assert issubclass(AgentRunner, RunnerSessionLifecycleMixin)
  assert gateway_runner.RunnerSessionLifecycleMixin is RunnerSessionLifecycleMixin

  for method_name in (
    "_append_durable_event",
    "_rebuild_task_registry_from_log",
    "_lookup_task_in_log",
    "_emit_attach_event",
    "_append_user_message_event",
    "_append_assistant_message_event",
    "_emit_stream_retry_event",
    "_emit_error_event",
    "_emit_run_error_event",
    "_emit_interrupted_event",
    "_shutdown_interrupted_reason",
    "_emit_detach_event",
    "_emit_operator_pause_event",
    "_acquire_writer_lease_and_recover",
    "_write_lease_metadata",
    "_release_write_lease",
  ):
    assert getattr(AgentRunner, method_name) is getattr(RunnerSessionLifecycleMixin, method_name)


@pytest.mark.parametrize(
  "outer_version",
  (True, False, 2.0, "2", 1, 3, None),
)
def test_top_level_durable_envelopes_require_exact_current_integer_version(
  outer_version: object,
) -> None:
  runner = object.__new__(AgentRunner)
  event = {
    "type": "stream_complete",
    "event_schema_version": outer_version,
    "runner_id": "runner-1",
    "role": "writer",
  }

  with pytest.raises(RuntimeError, match="invalid schema"):
    runner._expected_durable_top_level_event(event)
  with pytest.raises(RuntimeError, match="invalid schema"):
    runner._append_exact_durable_top_level_envelope_sync(event)
  with pytest.raises(SkillCompletionWalCorruptError, match="invalid schema"):
    runner._durable_top_level_wrapper(event)

  assert EVENT_SCHEMA_VERSION == 2


def test_runner_session_lifecycle_resolves_parent_module_event_builders(monkeypatch: Any) -> None:
  runner = object.__new__(AgentRunner)
  runner._gateway_session_id = "sess"
  runner._client_kind = "cli"
  runner._durable_attach_emitted = False
  appended: list[dict[str, Any]] = []

  async def _append_durable_event(event: dict[str, Any]):
    appended.append(event)
    return SimpleNamespace(seq=1)

  runner._append_durable_event = _append_durable_event  # type: ignore[method-assign]

  def _build_attach_event(**kwargs: Any) -> dict[str, Any]:
    return {"type": "patched_attach", **kwargs}

  monkeypatch.setattr(gateway_runner, "_build_attach_event", _build_attach_event)
  monkeypatch.setattr(gateway_runner, "time", SimpleNamespace(time=lambda: 12.5))
  monkeypatch.setattr(gateway_runner, "socket", SimpleNamespace(gethostname=lambda: "patched-host"))

  _run(AgentRunner._emit_attach_event(runner))

  assert appended == [
    {
      "type": "patched_attach",
      "gateway_session_id": "sess",
      "started_at": 12.5,
      "client_kind": "cli",
      "hostname": "patched-host",
    }
  ]
  assert runner._durable_attach_emitted


def test_runner_append_durable_event_resolves_parent_module_payload_helpers(monkeypatch: Any) -> None:
  appended: list[dict[str, Any]] = []

  class _SessionLog:
    async def append(self, payload: dict[str, Any]):
      appended.append(payload)
      return SimpleNamespace(seq=7)

  runner = object.__new__(AgentRunner)
  runner._agent_session_log = _SessionLog()
  runner._runner_id = "runner-1"
  runner._role = "writer"
  runner._sub_agent_id = None
  runner._last_durable_seq = 0

  def _durable_event_payload(event: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return {"type": event["type"], "patched": True, **kwargs}

  monkeypatch.setattr(gateway_runner, "_durable_event_payload", _durable_event_payload)
  monkeypatch.setattr(gateway_runner, "gateway_product_id", lambda: "patched-product")

  entry = _run(AgentRunner._append_durable_event(runner, {"type": "custom"}))

  assert entry.seq == 7
  assert runner._last_durable_seq == 7
  assert appended == [
    {
      "type": "custom",
      "patched": True,
      "runner_id": "runner-1",
      "role": "writer",
      "sub_agent_id": None,
      "product_id": "patched-product",
    }
  ]


def test_runner_writer_recovery_resolves_parent_module_risk_helper(tmp_path: Path, monkeypatch: Any) -> None:
  events: list[dict[str, Any]] = []

  class _SessionLog:
    path = tmp_path / "session.jsonl"
    write_lease_path = tmp_path / "session.jsonl.write_lease"

    async def query(self, **kwargs: Any):
      if kwargs.get("event_types") == {"tool_call_start", "tool_call_complete", "tool_call_interrupted"}:
        return [SimpleNamespace(event={"type": "tool_call_start", "tool_call_id": "tool-1"})], None
      return [], None

  runner = object.__new__(AgentRunner)
  runner._agent_session_log = _SessionLog()
  runner._role = "writer"
  runner._write_lease_file = None
  runner._last_durable_seq = 0

  async def _append_durable_event(event: dict[str, Any]):
    events.append(event)
    return SimpleNamespace(seq=len(events))

  runner._append_durable_event = _append_durable_event  # type: ignore[method-assign]

  def _build_orphan_events(entries: Any, *, discovered_at: float, tool_risk_for_tool: Any) -> list[dict[str, Any]]:
    assert len(entries) == 1
    return [{"type": "patched_orphan", "risk": tool_risk_for_tool("write_tool"), "discovered_at": discovered_at}]

  monkeypatch.setattr(gateway_runner, "_build_orphan_tool_call_interrupted_events", _build_orphan_events)
  monkeypatch.setattr(gateway_runner, "_get_tool_risk_value", lambda tool_name: f"patched:{tool_name}")
  monkeypatch.setattr(gateway_runner, "time", SimpleNamespace(time=lambda: 42.0))

  _run(AgentRunner._acquire_writer_lease_and_recover(runner))

  assert events == [{"type": "patched_orphan", "risk": "patched:write_tool", "discovered_at": 42.0}]
  AgentRunner._release_write_lease(runner)


def test_runner_release_write_lease_resolves_parent_module_helper(monkeypatch: Any) -> None:
  calls: list[Any] = []
  runner = object.__new__(AgentRunner)
  runner._write_lease_file = object()

  def _release_write_lease(write_lease_file: Any, *, clear_write_lease_file: Any) -> bool:
    calls.append(write_lease_file)
    clear_write_lease_file()
    return True

  monkeypatch.setattr(gateway_runner, "_release_write_lease", _release_write_lease)

  AgentRunner._release_write_lease(runner)

  assert calls
  assert runner._write_lease_file is None


def test_session_event_builders_copy_inputs_and_shape_payloads() -> None:
  content_blocks: list[dict[str, Any]] = [{"type": "text", "text": "hello"}]
  usage = {"input_tokens": 3, "output_tokens": 5}

  assistant = build_assistant_message_event(
    content_blocks=content_blocks,
    stop_reason="end_turn",
    model="model-1",
    provider="stub",
    usage=usage,
  )
  content_blocks.append({"type": "text", "text": "mutated"})
  usage["output_tokens"] = 9

  assert build_attach_event(
    gateway_session_id="sess",
    started_at=1.25,
    client_kind="cli",
    hostname="host",
  ) == {
    "type": "attach",
    "gateway_session_id": "sess",
    "started_at": 1.25,
    "client_kind": "cli",
    "hostname": "host",
  }
  assert build_write_lease_metadata(
    runner_id="runner-1",
    gateway_session_id="sess",
    started_at=1.5,
    hostname="host",
  ) == {
    "runner_id": "runner-1",
    "gateway_session_id": "sess",
    "started_at": 1.5,
    "hostname": "host",
  }
  assert build_user_message_event(
    content="question",
    client_kind="web",
    received_at=2.5,
    selected_content=(),
  ) == {
    "type": "user_message",
    "content": "question",
    "client_kind": "web",
    "received_at": 2.5,
    "selected_content": [],
  }
  assert assistant == {
    "type": "assistant_message",
    "content_blocks": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "model": "model-1",
    "provider": "stub",
    "usage": {"input_tokens": 3, "output_tokens": 5},
  }
  assert build_stream_retry_event(attempt=2, error="stall") == {
    "type": "stream_retry",
    "attempt": 2,
    "error": "stall",
  }
  assert build_error_event("failed") == {"type": "error", "error": "failed"}
  assert build_run_error_event(phase="close", error_type="RuntimeError", error="offline") == {
    "type": "run_error",
    "phase": "close",
    "error_type": "RuntimeError",
    "error": "offline",
  }
  assert build_detach_event(reason="completed", ended_at=3.75) == {
    "type": "detach",
    "reason": "completed",
    "ended_at": 3.75,
  }
  assert build_operator_pause_event("before_turn") == {
    "type": "operator_pause",
    "reason": "operator_pause",
    "safe_boundary": "before_turn",
  }


def test_assistant_message_event_records_exact_logical_response_lineage() -> None:
  origin = build_assistant_message_event(
    content_blocks=[{"type": "text", "text": "first"}],
    stop_reason="max_tokens",
    model="model-1",
    provider="stub",
    usage={},
    logical_response_id="logical-1",
    logical_response_segment_ordinal=0,
  )
  continuation = build_assistant_message_event(
    content_blocks=[{"type": "text", "text": "second"}],
    stop_reason="end_turn",
    model="model-1",
    provider="stub",
    usage={},
    logical_response_id="logical-1",
    logical_response_segment_ordinal=1,
    continued_from_assistant_message_seq=17,
  )

  assert origin["logical_response_id"] == "logical-1"
  assert origin["logical_response_segment_ordinal"] == 0
  assert "continued_from_assistant_message_seq" not in origin
  assert continuation["logical_response_segment_ordinal"] == 1
  assert continuation["continued_from_assistant_message_seq"] == 17

  with pytest.raises(ValueError, match="requires logical_response_id"):
    build_assistant_message_event(
      content_blocks=[],
      stop_reason="end_turn",
      model="model-1",
      provider="stub",
      usage={},
      logical_response_segment_ordinal=0,
    )


def test_write_lease_metadata_helper_writes_only_for_active_writer(tmp_path: Path) -> None:
  lease_path = tmp_path / "session.jsonl.write_lease.meta"
  session_log = SimpleNamespace(write_lease_meta_path=lease_path)

  assert write_lease_metadata(
    session_log,
    role="writer",
    runner_id="runner-1",
    gateway_session_id="sess",
    started_at=1.5,
    hostname="host",
  )
  assert json.loads(lease_path.read_text(encoding="utf-8")) == {
    "gateway_session_id": "sess",
    "hostname": "host",
    "runner_id": "runner-1",
    "started_at": 1.5,
  }

  lease_path.unlink()
  assert not write_lease_metadata(
    session_log,
    role="reader",
    runner_id="runner-1",
    gateway_session_id="sess",
    started_at=2.0,
    hostname="host",
  )
  assert not lease_path.exists()

  assert not write_lease_metadata(
    session_log,
    role="writer",
    runner_id=None,
    gateway_session_id="sess",
    started_at=2.0,
    hostname="host",
  )
  assert not lease_path.exists()


def test_runner_write_lease_metadata_delegates_to_session_event_helper(
  tmp_path: Path,
  monkeypatch: Any,
) -> None:
  lease_path = tmp_path / "session.jsonl.write_lease.meta"
  runner = AgentRunner.__new__(AgentRunner)
  runner._agent_session_log = SimpleNamespace(write_lease_meta_path=lease_path)
  runner._role = "writer"
  runner._runner_id = "runner-1"
  runner._gateway_session_id = "sess"

  monkeypatch.setattr(gateway_runner.time, "time", lambda: 3.5)
  monkeypatch.setattr(gateway_runner.socket, "gethostname", lambda: "host")

  runner._write_lease_metadata()

  assert json.loads(lease_path.read_text(encoding="utf-8")) == {
    "gateway_session_id": "sess",
    "hostname": "host",
    "runner_id": "runner-1",
    "started_at": 3.5,
  }


def test_runner_write_lease_metadata_preserves_early_noop_guard(monkeypatch: Any) -> None:
  runner = AgentRunner.__new__(AgentRunner)
  runner._agent_session_log = None

  def fail_time() -> float:
    raise AssertionError("time should not be read for missing session log")

  def fail_hostname() -> str:
    raise AssertionError("hostname should not be read for missing session log")

  monkeypatch.setattr(gateway_runner.time, "time", fail_time)
  monkeypatch.setattr(gateway_runner.socket, "gethostname", fail_hostname)

  runner._write_lease_metadata()


def test_release_write_lease_closes_and_clears_file() -> None:
  calls: list[str] = []
  write_lease_file = SimpleNamespace(close=lambda: calls.append("closed"))

  assert release_write_lease(write_lease_file, clear_write_lease_file=lambda: calls.append("cleared"))
  assert calls == ["closed", "cleared"]


def test_release_write_lease_preserves_noop_and_error_cleanup() -> None:
  calls: list[str] = []

  assert not release_write_lease(None, clear_write_lease_file=lambda: calls.append("cleared"))
  assert calls == []

  def fail_close() -> None:
    calls.append("closed")
    raise RuntimeError("close failed")

  with pytest.raises(RuntimeError, match="close failed"):
    release_write_lease(SimpleNamespace(close=fail_close), clear_write_lease_file=lambda: calls.append("cleared"))

  assert calls == ["closed", "cleared"]


def test_runner_release_write_lease_delegates_and_clears_on_error() -> None:
  calls: list[str] = []
  runner = AgentRunner.__new__(AgentRunner)
  runner._write_lease_file = SimpleNamespace(close=lambda: calls.append("closed"))

  assert gateway_runner._release_write_lease is release_write_lease

  runner._release_write_lease()

  assert calls == ["closed"]
  assert runner._write_lease_file is None

  def fail_close() -> None:
    calls.append("failed-close")
    raise RuntimeError("close failed")

  runner._write_lease_file = SimpleNamespace(close=fail_close)

  with pytest.raises(RuntimeError, match="close failed"):
    runner._release_write_lease()

  assert calls == ["closed", "failed-close"]
  assert runner._write_lease_file is None


def test_stub_response_events_build_text_chunks_and_terminal_event() -> None:
  events = build_stub_response_events(
    [
      {"role": "system", "content": "ignore"},
      {"role": "user", "content": "first"},
      {"role": "assistant", "content": "middle"},
      {"role": "user", "content": "latest prompt"},
    ],
    provider_name="stub",
  )

  assert "".join(event.get("text", "") for event in events if event.get("type") == "text_delta") == (
    "Stub response (no Stub credential configured). You asked: latest prompt "
  )
  assert events[-1] == {
    "type": "stream_complete",
    "terminal_disposition": "completed",
    "usage": {},
  }


def test_stub_response_events_default_prompt_without_user_message() -> None:
  events = build_stub_response_events([], provider_name="stub")

  assert "".join(event.get("text", "") for event in events if event.get("type") == "text_delta") == (
    "Stub response (no Stub credential configured). You asked: your request "
  )
  assert events[-1]["type"] == "stream_complete"


def test_tool_lifecycle_event_builders_shape_payloads_and_copy_result_fields() -> None:
  redacted_input = {"query": "MSFT"}
  display = {"title": "Lookup MSFT"}
  start_event = build_tool_call_start_event(
    tool_call_id="toolu_1",
    tool_name="lookup",
    tool_input=redacted_input,
    call_index=3,
    server="research",
    started_at=12.5,
    parent_assistant_message_seq=7,
  )
  start_event["display"] = display

  assert start_event == {
    "type": "tool_call_start",
    "tool_call_id": "toolu_1",
    "tool_name": "lookup",
    "tool_input": {"query": "MSFT"},
    "execution_location": "backend",
    "call_index": 3,
    "server": "research",
    "started_at": 12.5,
    "parent_assistant_message_seq": 7,
    "display": {"title": "Lookup MSFT"},
  }
  assert build_tool_call_start_event(
    tool_call_id="toolu_2",
    tool_name="lookup",
    tool_input={},
    call_index=0,
    server=None,
    started_at=1.0,
    parent_assistant_message_seq=None,
  ) == {
    "type": "tool_call_start",
    "tool_call_id": "toolu_2",
    "tool_name": "lookup",
    "tool_input": {},
    "execution_location": "backend",
    "call_index": 0,
    "server": None,
    "started_at": 1.0,
    "parent_assistant_message_seq": None,
  }

  result = {"ok": True}
  semantic_error = {"code": "low_match"}
  dispatch = {
    "outcome": "error_semantic",
    "attempts": 1,
    "route_id": "mcp:research/lookup",
    "sources": [],
  }
  complete_event = build_tool_call_complete_event(
    tool_call_id="toolu_1",
    tool_name="lookup",
    result=result,
    error=None,
    duration_ms=42,
    server="research",
    dispatch=dispatch,
    semantic_error=semantic_error,
  )
  result["ok"] = False
  semantic_error["code"] = "mutated"
  dispatch["outcome"] = "mutated"

  assert complete_event == {
    "type": "tool_call_complete",
    "tool_call_id": "toolu_1",
    "tool_name": "lookup",
    "result": {"ok": True},
    "error": None,
    "duration_ms": 42,
    "server": "research",
    "is_error": True,
    "semantic_error": {"code": "low_match"},
    "dispatch": {
      "outcome": "error_semantic",
      "attempts": 1,
      "route_id": "mcp:research/lookup",
      "sources": [],
    },
  }
  assert build_tool_call_complete_event(
    tool_call_id="toolu_2",
    tool_name="lookup",
    result="done",
    error={"code": "tool_error"},
    duration_ms=5,
    server=None,
    dispatch={
      "outcome": "error_semantic",
      "attempts": 1,
      "route_id": "local/lookup",
      "sources": (),
    },
  ) == {
    "type": "tool_call_complete",
    "tool_call_id": "toolu_2",
    "tool_name": "lookup",
    "result": "done",
    "error": {"code": "tool_error"},
    "duration_ms": 5,
    "server": None,
    "is_error": True,
    "dispatch": {
      "outcome": "error_semantic",
      "attempts": 1,
      "route_id": "local/lookup",
      "sources": [],
    },
  }


def test_tool_call_complete_requires_a_dispatch_record() -> None:
  with pytest.raises(TypeError):
    build_tool_call_complete_event(  # type: ignore[call-arg]
      tool_call_id="toolu_3",
      tool_name="lookup",
      result={"status": "success"},
      error=None,
      duration_ms=1,
      server=None,
    )


def test_run_terminal_and_limit_event_builders_shape_payloads() -> None:
  usage = {
    "input_tokens": 10,
    "output_tokens": 20,
    "cache_creation_input_tokens": 3,
    "cache_read_input_tokens": 4,
  }
  turn_usage = {"input_tokens": 1}

  assert build_turn_complete_event(turn=2, usage=turn_usage) == {
    "type": "turn_complete",
    "turn": 2,
    "usage": {"input_tokens": 1},
  }
  turn_usage["input_tokens"] = 9
  assert build_max_turns_reached_event(turn_count=3, max_turns=2) == {
    "type": "max_turns_reached",
    "turn_count": 3,
    "max_turns": 2,
  }
  assert build_max_turns_text_event("wrap up") == {
    "type": "text_delta",
    "text": "\n\n[Max turns reached]\nwrap up",
  }
  assert build_max_turns_text_event(None) == {
    "type": "text_delta",
    "text": "\n\n[Sub-agent reached maximum turn limit]",
  }
  assert build_runtime_guard_event(guard="final_answer", message="continue") == {
    "type": "runtime_guard",
    "guard": "final_answer",
    "message": "continue",
  }
  assert build_budget_exceeded_event(total_cost=1.23456, budget=1.0, reason="parent_budget") == {
    "type": "budget_exceeded",
    "total_cost": 1.2346,
    "budget": 1.0,
    "reason": "parent_budget",
  }
  assert build_budget_exceeded_text_event(total_cost=1.23456, budget=1.0, reason_suffix=" (parent budget)") == {
    "type": "text_delta",
    "text": "\n\n[Budget limit reached: $1.2346 >= $1.0000 (parent budget)]",
  }
  assert build_stream_complete_event(usage_totals=usage, estimated_cost=0.12345) == {
    "type": "stream_complete",
    "terminal_disposition": "completed",
    "usage": {
      "input_tokens": 10,
      "output_tokens": 20,
      "cache_creation_input_tokens": 3,
      "cache_read_input_tokens": 4,
      "estimated_cost": 0.1235,
    },
  }


def test_context_warning_log_data_payloads_support_initial_and_turn_estimates() -> None:
  assert build_context_warning_log_data(
    session_id="sess-1",
    est_tokens=810,
    context_limit=1000,
  ) == {
    "event": "context_warning",
    "session_id": "sess-1",
    "est_tokens": 810,
    "limit": 1000,
    "pct": 81.0,
  }
  assert build_context_warning_log_data(
    session_id="sess-1",
    turn=3,
    est_tokens=812,
    context_limit=1000,
  ) == {
    "event": "context_warning",
    "session_id": "sess-1",
    "turn": 3,
    "est_tokens": 812,
    "limit": 1000,
    "pct": 81.2,
  }


def test_token_estimate_log_data_payloads_support_initial_and_turn_estimates() -> None:
  assert build_token_estimate_log_data(
    session_id="sess-1",
    est_system_tokens=10,
    est_messages_tokens=20,
    est_tools_tokens=30,
    est_total_tokens=60,
    message_count=2,
    tool_count=3,
  ) == {
    "event": "token_estimate",
    "session_id": "sess-1",
    "est_system_tokens": 10,
    "est_messages_tokens": 20,
    "est_tools_tokens": 30,
    "est_total_tokens": 60,
    "message_count": 2,
    "tool_count": 3,
  }
  assert build_token_estimate_log_data(
    session_id="sess-1",
    turn=3,
    est_system_tokens=10,
    est_messages_tokens=25,
    est_tools_tokens=35,
    est_total_tokens=70,
    message_count=4,
    tool_count=5,
  ) == {
    "event": "token_estimate",
    "session_id": "sess-1",
    "turn": 3,
    "est_system_tokens": 10,
    "est_messages_tokens": 25,
    "est_tools_tokens": 35,
    "est_total_tokens": 70,
    "message_count": 4,
    "tool_count": 5,
  }


def test_turn_complete_log_data_shapes_structured_logging_payload() -> None:
  tools = ["lookup", "write"]

  payload = build_turn_complete_log_data(
    session_id="sess-1",
    turn=2,
    elapsed_s=1.24,
    ttft_s=0.236,
    text_chars=42,
    tools=tools,
    stop_reason="tool_use",
  )
  tools.append("mutated")

  assert payload == {
    "event": "turn_complete",
    "session_id": "sess-1",
    "turn": 2,
    "elapsed_s": 1.2,
    "ttft_s": 0.24,
    "text_chars": 42,
    "tools": ["lookup", "write"],
    "stop_reason": "tool_use",
  }
  assert build_turn_complete_log_data(
    session_id="sess-1",
    turn=3,
    elapsed_s=1.25,
    ttft_s=None,
    text_chars=0,
    tools=[],
    stop_reason=None,
  ) == {
    "event": "turn_complete",
    "session_id": "sess-1",
    "turn": 3,
    "elapsed_s": 1.2,
    "ttft_s": None,
    "text_chars": 0,
    "tools": [],
    "stop_reason": None,
  }


def test_chat_done_log_data_shapes_structured_logging_payload_and_copies_tools() -> None:
  usage = {
    "input_tokens": 10,
    "output_tokens": 20,
    "cache_creation_input_tokens": 3,
    "cache_read_input_tokens": 4,
  }
  tools = ["lookup", "write"]

  payload = build_chat_done_log_data(
    session_id="sess-1",
    elapsed_s=12.34,
    turns=5,
    tools=tools,
    usage_totals=usage,
    cost=1.23456,
  )
  tools.append("mutated")

  assert payload == {
    "event": "chat_done",
    "session_id": "sess-1",
    "elapsed_s": 12.3,
    "turns": 5,
    "tools": ["lookup", "write"],
    "tokens_in": 10,
    "tokens_out": 20,
    "cache_read": 4,
    "cache_write": 3,
    "cost": 1.2346,
  }


def test_durable_and_interrupted_payload_helpers_preserve_existing_fields() -> None:
  payload = durable_event_payload(
    {"type": "custom", "runner_id": "existing", "role": "existing-role"},
    runner_id="runner-1",
    role="writer",
    sub_agent_id="sub-1",
    product_id="hank",
  )

  assert payload == {
    "type": "custom",
    "runner_id": "existing",
    "role": "existing-role",
    "sub_agent_id": "sub-1",
    "product_id": "hank",
  }
  assert build_interrupted_event(
    reason="signal_SIGTERM",
    runner_id="runner-1",
    role="writer",
    last_completed_seq=4,
    recovered_by_runner_id="runner-2",
    recovered_at=5.5,
    extra_fields={"shutdown": {"signal": 15}},
  ) == {
    "type": "interrupted",
    "reason": "signal_SIGTERM",
    "runner_id": "runner-1",
    "role": "writer",
    "last_completed_seq": 4,
    "recovered_by_runner_id": "runner-2",
    "recovered_at": 5.5,
    "shutdown": {"signal": 15},
  }


def test_run_detach_reason_preserves_clean_cancelled_and_error_reasons() -> None:
  assert run_detach_reason(clean_detach_reason="completed", run_error=None) == "completed"
  assert run_detach_reason(clean_detach_reason="operator_pause", run_error=None) == "operator_pause"
  assert run_detach_reason(
    clean_detach_reason="completed",
    run_error=asyncio.CancelledError(),
  ) == "cancelled"
  assert run_detach_reason(
    clean_detach_reason="operator_pause",
    run_error=RuntimeError("boom"),
  ) == "error"


def test_run_interrupted_reason_preserves_shutdown_reason_except_cancelled_sub_agent() -> None:
  shutdown_extra_fields = {"shutdown": {"signal_name": "SIGTERM"}}

  assert run_interrupted_reason(
    run_error=RuntimeError("boom"),
    role="writer",
    shutdown_reason="signal_SIGTERM",
    shutdown_extra_fields=shutdown_extra_fields,
  ) == ("signal_SIGTERM", shutdown_extra_fields)
  assert run_interrupted_reason(
    run_error=asyncio.CancelledError(),
    role="writer",
    shutdown_reason="graceful_shutdown",
    shutdown_extra_fields={},
  ) == ("graceful_shutdown", {})
  assert run_interrupted_reason(
    run_error=asyncio.CancelledError(),
    role="sub_agent",
    shutdown_reason="signal_SIGTERM",
    shutdown_extra_fields=shutdown_extra_fields,
  ) == ("sub_agent_cancelled", {})


def test_orphan_tool_call_interrupted_events_select_unresolved_starts() -> None:
  entries = [
    SimpleNamespace(event={"type": "tool_call_start", "tool_call_id": "", "tool_name": "skip"}),
    SimpleNamespace(event={"type": "tool_call_start", "tool_call_id": "resolved", "tool_name": "lookup"}),
    SimpleNamespace(event={"type": "tool_call_complete", "tool_call_id": "resolved"}),
    SimpleNamespace(
      event={
        "type": "tool_call_start",
        "tool_call_id": "orphan",
        "tool_name": "write_tool",
        "tool_input": {"symbol": "MSFT"},
        "started_at": 12.5,
        "runner_id": "runner-old",
        "role": "writer",
        "sub_agent_id": "sub-1",
      }
    ),
    SimpleNamespace(event={"type": "tool_call_start", "tool_call_id": "orphan", "tool_name": "ignored-later"}),
    SimpleNamespace(event={"type": "tool_call_start", "tool_call_id": "already-interrupted", "tool_name": "lookup"}),
    SimpleNamespace(event={"type": "tool_call_interrupted", "tool_call_id": "already-interrupted"}),
  ]

  synthetic = build_orphan_tool_call_interrupted_events(
    entries,
    discovered_at=20.0,
    tool_risk_for_tool=lambda tool_name: f"risk:{tool_name}",
  )

  assert synthetic == [
    {
      "type": "tool_call_interrupted",
      "tool_call_id": "orphan",
      "tool_name": "write_tool",
      "tool_input": {"symbol": "MSFT"},
      "original_started_at": 12.5,
      "discovered_at": 20.0,
      "tool_risk": "risk:write_tool",
      "runner_id": "runner-old",
      "role": "writer",
      "sub_agent_id": "sub-1",
    }
  ]


def test_shutdown_interrupted_reason_normalizes_signal_payloads() -> None:
  assert shutdown_interrupted_reason(None) == ("graceful_shutdown", {})
  assert shutdown_interrupted_reason({}) == ("graceful_shutdown", {})
  assert shutdown_interrupted_reason({"signal_name": " SIGTERM ", "pid": 123}) == (
    "signal_SIGTERM",
    {"shutdown": {"signal_name": " SIGTERM ", "pid": 123}},
  )
  assert shutdown_interrupted_reason({"signal": 15}) == ("signal_SIG15", {"shutdown": {"signal": 15}})
  assert shutdown_interrupted_reason({"pid": 123}) == ("signal_unknown", {"shutdown": {"pid": 123}})


def test_runner_durable_event_delegate_stamps_envelope(tmp_path: Path, monkeypatch) -> None:
  monkeypatch.setenv("PRODUCT_ID", "hank")
  gateway_product_id.cache_clear()
  runner = object.__new__(AgentRunner)
  runner._agent_session_log = AgentSessionLog(path=tmp_path / "runner.jsonl")
  runner._runner_id = "runner-1"
  runner._role = "writer"
  runner._sub_agent_id = None
  runner._last_durable_seq = 0

  try:
    entry = _run(AgentRunner._append_durable_event(runner, {"type": "custom"}))
    entries, _ = _run(runner._agent_session_log.query(order="asc"))
  finally:
    gateway_product_id.cache_clear()

  assert entry is not None
  assert runner._last_durable_seq == entry.seq == 1
  assert {key: entries[0].event.get(key) for key in ("type", "runner_id", "role", "product_id")} == {
    "type": "custom",
    "runner_id": "runner-1",
    "role": "writer",
    "product_id": "hank",
  }


def test_runner_stub_response_delegate_appends_built_events(monkeypatch) -> None:
  runner = object.__new__(AgentRunner)
  runner._provider = type("Provider", (), {"name": "stub"})()
  appended: list[dict[str, Any]] = []
  sleep_delays: list[float] = []
  runner._append = appended.append

  async def _fake_sleep(delay: float) -> None:
    sleep_delays.append(delay)

  monkeypatch.setattr("agent_gateway.runner.asyncio.sleep", _fake_sleep)

  _run(AgentRunner._emit_stub_response(runner, [{"role": "user", "content": "hello"}]))

  assert [event["type"] for event in appended][-1] == "stream_complete"
  assert appended[-1] == {
    "type": "stream_complete",
    "terminal_disposition": "completed",
    "usage": {},
  }
  assert sleep_delays == [0.05] * (len(appended) - 1)
