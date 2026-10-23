import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, AgentSessionLog, LogEntry  # noqa: E402
from agent_gateway.agent_session_log_records import EVENT_SCHEMA_VERSION  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.providers import ModelProvider  # noqa: E402
from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND  # noqa: E402
from agent_gateway.runner_session_events import (  # noqa: E402
  build_assistant_message_event,
  build_attach_event,
  build_chat_done_log_data,
  build_context_pressure_reminder,
  build_context_warning_log_data,
  build_detach_event,
  build_error_event,
  build_interrupted_event,
  build_soft_budget_final_event,
  build_soft_budget_steer_event,
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
  terminal_closure_decision,
  run_interrupted_reason,
  shutdown_interrupted_reason,
  write_lease_metadata,
)
from agent_gateway.skill_completion_wal import SkillCompletionWalCorruptError  # noqa: E402
from agent_gateway.event_adapter import adapt_event  # noqa: E402
from agent_gateway.event_log import EventLog  # noqa: E402
from agent_gateway.multi_user.billing import SessionUsageSummary  # noqa: E402
from agent_gateway.skill_lifecycle import TopLevelSkillLifecycleMetadata  # noqa: E402
from agent_gateway.tool_dispatcher_approval_lifecycle import ApprovalPark  # noqa: E402
from agent_gateway.server_chat_helpers import _chat_turn_state_from_events  # noqa: E402


def test_context_pressure_reminder_copy_is_model_actionable() -> None:
  assert build_context_pressure_reminder(pct=60) == (
    "Context at 60% — prefer delegating further reading; "
    "large results will spill."
  )


def _run(coro):
  return asyncio.run(coro)

class _StubProvider(ModelProvider):
  name = "stub"




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
  runner._gateway_session = None
  runner._client_kind = "cli"
  runner._durable_attach_emitted = False
  appended: list[dict[str, Any]] = []

  async def _append_durable_event(event: dict[str, Any]):
    appended.append(event)
    return SimpleNamespace(seq=1)

  runner._append_durable_event = _append_durable_event

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
      "user_id": None,
      "channel": None,
      "api_key_hash": None,
    }
  ]
  assert runner._durable_attach_emitted


def test_runner_append_durable_event_resolves_parent_module_payload_helpers(
  tmp_path: Path,
  monkeypatch: Any,
) -> None:
  session_log = AgentSessionLog(tmp_path / "session.jsonl")
  appended: list[dict[str, object]] = []
  runner = object.__new__(AgentRunner)
  runner._agent_session_log = session_log
  runner._runner_id = "runner-1"
  runner._role = "writer"
  runner._sub_agent_id = None
  runner._last_durable_seq = 0
  original_append = session_log.append

  async def _recording_append(payload: dict[str, object]) -> LogEntry:
    appended.append(payload)
    return await original_append(payload)

  monkeypatch.setattr(session_log, "append", _recording_append)

  def _durable_event_payload(
    event: dict[str, Any],
    **kwargs: Any,
  ) -> dict[str, Any]:
    return {"type": event["type"], "patched": True, **kwargs}

  monkeypatch.setattr(
    gateway_runner,
    "_durable_event_payload",
    _durable_event_payload,
  )
  monkeypatch.setattr(
    gateway_runner,
    "gateway_product_id",
    lambda: "patched-product",
  )

  entry = _run(
    AgentRunner._append_durable_event(runner, {"type": "custom"})
  )

  assert entry is not None
  assert entry.seq == 1
  assert runner._last_durable_seq == 1
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
  session_log = AgentSessionLog(tmp_path / "session.jsonl")
  _run(session_log.append({
    "type": "tool_call_start",
    "tool_call_id": "tool-1",
  }))

  runner = object.__new__(AgentRunner)
  runner._agent_session_log = session_log
  runner._role = "writer"
  runner._write_lease_file = None
  runner._last_durable_seq = 0

  async def _append_durable_event(event: dict[str, Any]):
    events.append(event)
    return SimpleNamespace(seq=len(events))

  runner._append_durable_event = _append_durable_event

  def _build_orphan_events(entries: Any, *, discovered_at: float, tool_risk_for_tool: Any) -> list[dict[str, Any]]:
    assert len(entries) == 1
    return [{"type": "patched_orphan", "risk": tool_risk_for_tool("write_tool"), "discovered_at": discovered_at}]

  monkeypatch.setattr(gateway_runner, "_build_orphan_tool_call_interrupted_events", _build_orphan_events)
  monkeypatch.setattr(gateway_runner, "_get_tool_risk_value", lambda tool_name: f"patched:{tool_name}")
  monkeypatch.setattr(gateway_runner, "time", SimpleNamespace(time=lambda: 42.0))

  _run(AgentRunner._acquire_writer_lease_and_recover(runner))

  assert events == [{"type": "patched_orphan", "risk": "patched:write_tool", "discovered_at": 42.0}]
  AgentRunner._release_write_lease(runner)


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
    user_id="alice",
    channel="cli",
    api_key_hash="abc123",
  ) == {
    "type": "attach",
    "gateway_session_id": "sess",
    "started_at": 1.25,
    "client_kind": "cli",
    "hostname": "host",
    "user_id": "alice",
    "channel": "cli",
    "api_key_hash": "abc123",
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
  assert build_stream_retry_event(
    attempt=2,
    error="stall",
    attempt_elapsed_s=12.5,
    backoff_s=8.0,
  ) == {
    "type": "stream_retry",
    "attempt": 2,
    "error": "stall",
    "attempt_elapsed_s": 12.5,
    "backoff_s": 8.0,
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
  session_log = AgentSessionLog(path=tmp_path / "session.jsonl")
  runner._agent_session_log = session_log
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

  registered_outcome_only_failure = build_tool_call_complete_event(
    tool_call_id="toolu_3",
    tool_name="lookup",
    result={"status": "success"},
    error=None,
    duration_ms=3,
    server="research",
    dispatch={
      "outcome": "error_semantic",
      "attempts": 1,
      "route_id": "mcp:research/lookup",
      "sources": (),
    },
  )
  assert registered_outcome_only_failure["is_error"] is True

  registered_success = build_tool_call_complete_event(
    tool_call_id="toolu_4",
    tool_name="lookup",
    result={"status": "success"},
    error=None,
    duration_ms=3,
    server="research",
    dispatch={
      "outcome": "ok",
      "attempts": 1,
      "route_id": "mcp:research/lookup",
      "sources": (),
    },
  )
  assert registered_success["is_error"] is False


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
  turn_usage = {"input_tokens": 1}

  assert build_turn_complete_event(turn=2, usage=turn_usage) == {
    "type": "turn_complete",
    "turn": 2,
    "usage": {"input_tokens": 1},
  }
  turn_usage["input_tokens"] = 9
  assert build_soft_budget_steer_event("turns", 3, 3) == {
    "type": "soft_budget_steer",
    "kind": "turns",
    "observed": 3,
    "threshold": 3,
  }
  assert build_soft_budget_final_event("cost", 1.6, 1.0) == {
    "type": "soft_budget_final",
    "kind": "cost",
    "observed": 1.6,
    "threshold": 1.0,
  }
  assert build_runtime_guard_event(guard="final_answer", message="continue") == {
    "type": "runtime_guard",
    "guard": "final_answer",
    "message": "continue",
  }


def test_stream_complete_event_includes_static_token_estimates_with_provider_usage() -> None:
  usage = {
    "input_tokens": 10,
    "output_tokens": 20,
    "cache_creation_input_tokens": 3,
    "cache_read_input_tokens": 4,
    "reasoning_tokens_observed": 6,
    "provider_wait_ms": 250,
    "thinking_chars": 1800,
  }

  assert build_stream_complete_event(
    usage_totals=usage,
    estimated_cost=0.12345,
    est_system_tokens=1200,
    est_tools_tokens=3400,
  ) == {
    "type": "stream_complete",
    "terminal_disposition": "completed",
    "usage": {
      "input_tokens": 10,
      "output_tokens": 20,
      "cache_creation_input_tokens": 3,
      "cache_read_input_tokens": 4,
      "reasoning_tokens_observed": 6,
      "provider_wait_ms": 250,
      "thinking_chars": 1800,
      "estimated_cost": 0.1235,
      "est_system_tokens": 1200,
      "est_tools_tokens": 3400,
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


def test_terminal_closure_decision_preserves_failure_and_cancellation_precedence() -> None:
  def decide(**facts: Any):
    return terminal_closure_decision(**{
      "clean_detach_reason": "completed",
      "run_error": None,
      "terminal_event": None,
      "server_terminal_cause": None,
      **facts,
    })

  assert decide().disposition == "success"
  assert decide(clean_detach_reason="operator_pause").reason == "operator_pause"
  cancelled = decide(
    run_error=asyncio.CancelledError("shutdown"), server_terminal_cause="timeout",
  )
  assert (cancelled.disposition, cancelled.reason) == ("cancelled", "timeout")
  failed = decide(clean_detach_reason="operator_pause", run_error=RuntimeError("boom"))
  assert (failed.disposition, failed.reason) == ("error", "error")
  assert decide(terminal_event={"type": "error", "error": "provider failed"}).disposition == "error"
  persistence = decide(persistence_error=OSError("disk unavailable"))
  assert (persistence.disposition, persistence.reason) == ("error", "persistence")


def test_terminal_closure_decision_supplies_unprepared_exception_terminal() -> None:
  error = RuntimeError("incomplete chunked read")
  decision = terminal_closure_decision(
    clean_detach_reason="completed",
    run_error=error,
    terminal_event=None,
    server_terminal_cause=None,
  )

  assert (decision.disposition, decision.reason) == ("error", "error")
  assert decision.cause is error
  assert decision.terminal == build_error_event(str(error))


@pytest.mark.parametrize(
  "terminal",
  [
    {"type": "error", "error": "first provider failure"},
    {
      "type": "stream_complete",
      "terminal_disposition": "interrupted",
      "reason": "timeout",
    },
  ],
)
def test_escaping_exception_preserves_prepared_terminal(terminal: dict[str, Any]) -> None:
  error = RuntimeError("later incomplete chunked read")
  decision = terminal_closure_decision(
    clean_detach_reason="completed",
    run_error=error,
    terminal_event=terminal,
    server_terminal_cause=None,
  )

  assert decision.cause is error
  assert decision.terminal is terminal


def test_unprepared_exception_terminal_preserves_cancellation_and_persistence() -> None:
  cancellation = asyncio.CancelledError("shutdown")
  cancelled = terminal_closure_decision(
    clean_detach_reason="completed",
    run_error=cancellation,
    terminal_event=None,
    server_terminal_cause="timeout",
  )
  assert (cancelled.disposition, cancelled.reason) == ("cancelled", "timeout")
  assert cancelled.cause is cancellation
  assert cancelled.terminal is None

  persistence_error = OSError("disk unavailable")
  failed = terminal_closure_decision(
    clean_detach_reason="completed",
    run_error=RuntimeError("incomplete chunked read"),
    terminal_event=None,
    server_terminal_cause="timeout",
    persistence_error=persistence_error,
  )
  assert (failed.disposition, failed.reason) == ("error", "persistence")
  assert failed.cause is persistence_error
  assert failed.terminal == {
    **build_error_event(
      "terminal_persistence_failed: terminal persistence failed "
      "(OSError: disk unavailable)."
    ),
    "reason": "persistence",
    "server_terminal_cause": "timeout",
  }


@pytest.mark.parametrize(
  "skill_result",
  [
    {"exit_code": 1, "outcome": "post_run_guard_failed"},
    {"exit_code": 0, "outcome": "blocked", "error": "terminal door never called"},
    {"exit_code": 0, "outcome": "error"},
  ],
)
def test_success_disposition_with_failed_skill_result_yields_error_terminal(
  skill_result: dict[str, Any],
) -> None:
  lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id="guard-failed-skill", skill="review", scope="ticker", ticker="MSFT",
  )
  terminal = build_stream_complete_event(
    usage_totals={
      "input_tokens": 1, "output_tokens": 1,
      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
    },
    estimated_cost=0.01,
    est_system_tokens=1,
    est_tools_tokens=1,
  )
  decision = terminal_closure_decision(
    clean_detach_reason="completed",
    run_error=None,
    terminal_event=terminal,
    server_terminal_cause=None,
    skill_lifecycle=lifecycle,
    skill_result={"type": "skill_result_captured", **skill_result},
  )
  assert decision.terminal is not None
  assert decision.terminal["type"] == "error"
  assert "terminal_disposition" not in decision.terminal
  assert decision.terminal["reason"] == skill_result["outcome"]
  assert decision.terminal["error"] == skill_result.get("error", skill_result["outcome"])


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


def test_terminal_closure_decision_carries_the_interruption_the_writer_emits() -> None:
  shutdown_extra_fields = {"shutdown": {"signal_name": "SIGTERM"}}
  staged = ("operator_pause", {"safe_boundary": "before_turn"})
  error = RuntimeError("boom")

  failed = terminal_closure_decision(
    clean_detach_reason="completed",
    run_error=error,
    terminal_event=None,
    server_terminal_cause=None,
    role="writer",
    shutdown_reason="signal_SIGTERM",
    shutdown_extra_fields=shutdown_extra_fields,
    staged_interruption=staged,
  )
  assert failed.interruption == ("error", shutdown_extra_fields)

  cancelled = terminal_closure_decision(
    clean_detach_reason="completed",
    run_error=asyncio.CancelledError(),
    terminal_event=None,
    server_terminal_cause=None,
    role="sub_agent",
    shutdown_reason="signal_SIGTERM",
    shutdown_extra_fields=shutdown_extra_fields,
  )
  assert cancelled.interruption == ("sub_agent_cancelled", {})

  caused = terminal_closure_decision(
    clean_detach_reason="completed",
    run_error=error,
    terminal_event=None,
    server_terminal_cause="timeout",
    role="writer",
    shutdown_reason="signal_SIGTERM",
    shutdown_extra_fields=shutdown_extra_fields,
  )
  assert caused.interruption == ("error", {"server_terminal_cause": "timeout"})

  paused = terminal_closure_decision(
    clean_detach_reason="operator_pause",
    run_error=None,
    terminal_event=None,
    server_terminal_cause=None,
    staged_interruption=staged,
  )
  assert paused.interruption == staged

  persisted = terminal_closure_decision(
    clean_detach_reason="completed",
    run_error=error,
    terminal_event=None,
    server_terminal_cause="timeout",
    persistence_error=OSError("disk unavailable"),
    shutdown_reason="signal_SIGTERM",
    shutdown_extra_fields=shutdown_extra_fields,
  )
  assert persisted.interruption == ("persistence", None)


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

  assert [event["tool_call_id"] for event in synthetic] == ["orphan"]
  interrupted = synthetic[0]
  assert interrupted["original_started_at"] == 12.5
  assert interrupted["tool_risk"] == "risk:write_tool"
  assert interrupted["runner_id"] == "runner-old"
  assert interrupted["sub_agent_id"] == "sub-1"
  result = interrupted["final_tool_result_blocks"][0]
  assert result["tool_use_id"] == "orphan"
  assert result["is_error"] is True


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
  runner = object.__new__(AgentRunner)
  runner._agent_session_log = AgentSessionLog(path=tmp_path / "runner.jsonl")
  runner._runner_id = "runner-1"
  runner._role = "writer"
  runner._sub_agent_id = None
  runner._last_durable_seq = 0

  entry = _run(AgentRunner._append_durable_event(runner, {"type": "custom"}))
  entries, _ = _run(runner._agent_session_log.query(order="asc"))

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
  runner._provider = _StubProvider()
  appended: list[dict[str, Any]] = []
  sleep_delays: list[float] = []

  def _append(event: dict[str, Any]) -> None:
    appended.append(event)

  runner._append = _append

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


@pytest.mark.parametrize("summary_present", [True, False])
@pytest.mark.parametrize("failure_stage", [None, "append", "readback", "detach"])
def test_approval_park_closure_preserves_attempt_usage_without_final_skill_effect(
  tmp_path: Path,
  monkeypatch: Any,
  summary_present: bool,
  failure_stage: str | None,
) -> None:
  park = ApprovalPark(
    job_id="admitted-original-job",
    original_tool_call_id="original-call",
    approval_id="approval-public-id",
    nonce="private-nonce",
    root_turn_id="admitted-root-turn" if summary_present else None,
  )
  terminal = build_stream_complete_event(
    usage_totals={
      "input_tokens": 31,
      "output_tokens": 7,
      "cache_creation_input_tokens": 3,
      "cache_read_input_tokens": 11,
    },
    estimated_cost=0.03123,
    est_system_tokens=101,
    est_tools_tokens=203,
    terminal_disposition="awaiting_approval",
  )
  lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id="still-pending-skill",
    skill="review",
    scope="ticker",
    ticker="MSFT",
  )
  facts: dict[str, Any] = {
    "clean_detach_reason": "awaiting_approval",
    "run_error": None,
    "terminal_event": terminal,
    "server_terminal_cause": None,
    "approval_park": park,
  }
  pending = terminal_closure_decision(**facts, skill_lifecycle=lifecycle)
  assert (pending.disposition, pending.reason) == (
    "awaiting_approval", "awaiting_approval",
  )
  assert pending.skill_result is None and pending.skill_effect is None
  run_failure = RuntimeError("provider failed after capture")
  cancellation = asyncio.CancelledError("caller disconnected")
  persistence_failure = OSError("journal unavailable")
  for overrides, disposition, reason, cause in (
    ({"run_error": run_failure}, "error", "error", run_failure),
    ({"run_error": cancellation}, "cancelled", "cancelled", cancellation),
    ({"server_terminal_cause": "shutdown"}, "cancelled", "shutdown", None),
    (
      {"run_error": cancellation, "persistence_error": persistence_failure},
      "error", "persistence", persistence_failure,
    ),
  ):
    decision = terminal_closure_decision(**{**facts, **overrides})
    assert (decision.disposition, decision.reason, decision.cause) == (
      disposition, reason, cause,
    )
    assert decision.terminal is None or (
      decision.terminal.get("terminal_disposition") != "awaiting_approval"
    )

  session_log = AgentSessionLog(tmp_path / "park.jsonl")
  event_log = EventLog()
  for event in (
    {"type": "skill_run_started", **lifecycle.identity_fields()},
    {
      "type": "assistant_message",
      "content_blocks": [
        {"type": "tool_use", "id": park.original_tool_call_id, "name": "write_record"},
      ],
    },
    {
      "type": "tool_call_start",
      "tool_call_id": park.original_tool_call_id,
      "tool_name": "write_record",
      "started_at": 2.0,
    },
  ):
    event_log.append(event)
    session_log.append_sync({**event, "runner_id": "attempt-runner", "role": "writer"})
  summary = SessionUsageSummary(
    user_id="owner",
    session_id="physical-session-not-job",
    request_id="attempt-request",
    input_tokens=41,
    output_tokens=10,
    cache_read_tokens=13,
    cache_creation_tokens=5,
    cost=0.04,
    turns=2,
    channel="web",
    started_at=1.0,
    ended_at=3.0,
    drain_complete=False,
    in_flight_task_count=2,
    usage_event_count=2,
    usage_event_ids=("observed-root-usage", "observed-child-usage"),
    capability_bind=TEST_CAPABILITY_BIND.to_json(),
  )
  runner = object.__new__(AgentRunner)
  runner._agent_session_log = session_log
  runner._log = event_log
  runner._runner_id = "attempt-runner"
  runner._role = "writer"
  runner._sub_agent_id = None
  runner._gateway_session = None
  runner._durable_attach_emitted = True
  runner._last_durable_seq = 3
  runner._approval_park = park
  runner._terminal_closure_event = terminal
  runner._terminal_closure_receipts = None
  runner._terminal_closure_interruption = None
  runner._terminal_recap_payload = None
  runner._emit_session_recap = True
  runner._full_session_id = "physical-session-not-job"
  runner._session_started_at = 1.0
  runner._session_usage_summary = summary if summary_present else None
  runner._top_level_skill_lifecycle = lifecycle
  runner._top_level_skill_result_policy = None
  runner._top_level_skill_started_committed = True
  runner._top_level_skill_result_committed = False
  runner._top_level_skill_terminal_committed = False
  runner._top_level_skill_detach_committed = False
  runner._top_level_skill_result_event = None
  runner._top_level_skill_completion_effect_plan = None
  runner._top_level_skill_result_task = None
  runner._deferred_top_level_terminal_event = None
  effect_path = tmp_path / "final-named-skill-effect"

  async def prepare_final_result(**_kwargs: Any) -> bool:
    effect_path.write_text("incorrect final result", encoding="utf-8")
    raise AssertionError("park entered final named-skill preparation")

  def commit_final_result(**_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    effect_path.write_text("incorrect final effect", encoding="utf-8")
    raise AssertionError("park entered final named-skill completion")

  runner._prepare_top_level_skill_closure = prepare_final_result
  runner._commit_top_level_skill_plan_sync = commit_final_result
  append_sync = session_log.append_sync
  query_sync = session_log.query_sync
  append_pair = session_log.append_terminal_pair_sync

  def append_with_fault(event: dict[str, Any]) -> LogEntry:
    if failure_stage == "append" and event.get("terminal_disposition") == "awaiting_approval":
      raise OSError("park append failed")
    return append_sync(event)

  def readback_with_fault(**kwargs: Any):
    if failure_stage == "readback" and (
      kwargs.get("after_seq") == kwargs.get("before_seq") == 4
    ):
      return [], None
    return query_sync(**kwargs)

  def detach_with_fault(recap: dict[str, Any], detach: dict[str, Any]):
    if failure_stage == "detach" and detach.get("reason") == "awaiting_approval":
      raise OSError("park detach failed")
    return append_pair(recap, detach)

  monkeypatch.setattr(session_log, "append_sync", append_with_fault)
  monkeypatch.setattr(session_log, "query_sync", readback_with_fault)
  monkeypatch.setattr(session_log, "append_terminal_pair_sync", detach_with_fault)
  settle = runner._settle_run_closure(
    clean_detach_reason="awaiting_approval", run_error=None,
  )
  if failure_stage is None:
    assert _run(settle) is True
  else:
    # The failed closure commits first; the park's persistence cause is then raised.
    with pytest.raises((OSError, RuntimeError)):
      _run(settle)
  durable, _ = query_sync(order="asc")
  events = [entry.event for entry in durable]
  if failure_stage is None:
    assert runner._approval_park is park
  else:
    assert runner._approval_park is None
  assert not effect_path.exists()
  assert runner._top_level_skill_result_task is None
  assert runner._top_level_skill_result_event is None
  assert runner._top_level_skill_completion_effect_plan is None
  assert not runner._top_level_skill_result_committed
  assert not runner._top_level_skill_terminal_committed
  assert not any(event["type"] == "skill_result_captured" for event in events)
  assert events[-2]["type"] == "session_recap"
  assert events[-1]["type"] == "detach"
  live_terminal = runner._terminal_closure_event
  assert live_terminal is not None
  if failure_stage is not None:
    assert live_terminal["type"] == "error"
    assert live_terminal["reason"] == "persistence"
    assert events[-1]["reason"] == "error"
    assert events[-2]["failures"]
    assert _chat_turn_state_from_events(
      [entry.event for entry in event_log.entries]
    ) == "failed"
    return

  expected_usage = {
    "input_tokens": 31,
    "output_tokens": 7,
    "cache_creation_input_tokens": 3,
    "cache_read_input_tokens": 11,
    "reasoning_tokens_observed": 0,
    "provider_wait_ms": 0,
    "thinking_chars": 0,
    "estimated_cost": 0.0312,
    "est_system_tokens": 101,
    "est_tools_tokens": 203,
  }
  parked_terminals = [
    event for event in events
    if event["type"] == "stream_complete"
  ]
  assert len(parked_terminals) == 1
  assert parked_terminals[0]["usage"] == live_terminal["usage"] == expected_usage
  assert parked_terminals[0]["terminal_disposition"] == "awaiting_approval"
  assert parked_terminals[0]["job_id"] == park.job_id
  assert parked_terminals[0]["job_id"] != runner._full_session_id
  if park.root_turn_id is not None:
    assert parked_terminals[0]["root_turn_id"] == park.root_turn_id
  else:
    assert "root_turn_id" not in parked_terminals[0]
  assert "nonce" not in parked_terminals[0]
  assert parked_terminals[0]["approval_id"] == park.approval_id
  assert parked_terminals[0]["original_tool_call_id"] == park.original_tool_call_id
  assert "skill_run_id" not in parked_terminals[0]
  live_events = [entry.event for entry in event_log.entries]
  assert live_events[-2]["type"] == "session_recap"
  assert live_events[-1] == live_terminal
  assert live_events[-2]["usage"] == events[-2]["usage"]
  assert _chat_turn_state_from_events(live_events) == "approval_pending"
  assert not any(
    event["type"] in {"error", "interrupted", "tool_call_interrupted", "tool_call_complete"}
    for event in events
  )
  assert events[-1]["reason"] == "awaiting_approval"
  assert events[-2]["failures"] == []
  recap_usage = events[-2]["usage"]
  if summary_present:
    assert recap_usage["input_tokens"] == summary.input_tokens
    assert recap_usage["cost"] == summary.cost
    assert recap_usage["drain_complete"] is False
    assert recap_usage["in_flight_task_count"] == 2
    assert recap_usage["usage_event_ids"] == list(summary.usage_event_ids)
  else:
    assert recap_usage is None
  projected = adapt_event({
    **live_terminal,
    "nonce": park.nonce,
    "prepared_input": {"secret": "must-not-be-public"},
    "credential_generation_ref": "private-generation",
  }, 1)
  assert projected is not None
  assert projected["usage"] == expected_usage
  assert projected["job_id"] == park.job_id
  if park.root_turn_id is not None:
    assert projected["root_turn_id"] == park.root_turn_id
  else:
    assert "root_turn_id" not in projected
  assert projected["approval_id"] == park.approval_id
  assert projected["original_tool_call_id"] == park.original_tool_call_id
  assert "nonce" not in projected
  assert "prepared_input" not in projected
  assert "credential_generation_ref" not in projected
