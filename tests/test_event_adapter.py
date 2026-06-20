from __future__ import annotations

import pytest

from agent_gateway.event_adapter import (
  V1_FIELD_PROJECTION,
  V1_WIRE_EVENT_TYPES,
  V1Adapter,
  adapt_event,
)


def test_v1_wire_projection_covers_every_v1_type() -> None:
  assert len(V1_WIRE_EVENT_TYPES) == 37
  assert V1_WIRE_EVENT_TYPES
  assert set(V1_FIELD_PROJECTION) == set(V1_WIRE_EVENT_TYPES)
  assert all("type" in fields for fields in V1_FIELD_PROJECTION.values())
  assert "budget_exceeded" in V1_WIRE_EVENT_TYPES
  assert "budget_limited" not in V1_WIRE_EVENT_TYPES
  assert "budget_limited" not in V1_FIELD_PROJECTION
  assert "blocked" not in V1_WIRE_EVENT_TYPES
  assert "blocked" not in V1_FIELD_PROJECTION
  assert "remediating" not in V1_WIRE_EVENT_TYPES
  assert "remediating" not in V1_FIELD_PROJECTION


def test_v1_projection_covers_current_emitter_fixture_shapes() -> None:
  fixtures = [
    {"type": "text_delta", "text": "hello"},
    {"type": "thinking_delta", "text": "thinking"},
    {
      "type": "tool_call_start",
      "tool_call_id": "toolu_1",
      "tool_name": "build_model",
      "tool_input": {"ticker": "MSFT"},
      "execution_location": "backend",
      "call_index": 0,
      "server": "portfolio-mcp",
      "started_at": 1000.0,
      "parent_assistant_message_seq": 7,
    },
    {
      "type": "tool_call_complete",
      "tool_call_id": "toolu_1",
      "tool_name": "build_model",
      "result": {"ok": True},
      "error": None,
      "duration_ms": 12,
      "server": "portfolio-mcp",
      "is_error": False,
      "semantic_error": None,
      "execution_location": "backend",
      "final_tool_result_blocks": [{"type": "tool_result"}],
    },
    {
      "type": "tool_call_interrupted",
      "tool_call_id": "toolu_1",
      "tool_name": "build_model",
      "tool_input": {"ticker": "MSFT"},
      "original_started_at": 1000.0,
      "discovered_at": 1010.0,
      "tool_risk": "write",
      "runner_id": "runner-1",
      "role": "writer",
      "sub_agent_id": "sub0:sess_1",
    },
    {"type": "tool_output_chunk", "tool_call_id": "toolu_1", "tool_name": "code_execute", "stream": "stdout", "text": "line", "seq": 1},
    {"type": "tool_execute_request", "tool_call_id": "toolu_1", "nonce": "nonce", "expires_at": 123, "tool_name": "update_model", "tool_input": {}},
    {"type": "interceptor_decision", "tool_call_id": "toolu_1", "tool_name": "bash", "action": "deny", "code": "blocked", "message": "blocked"},
    {"type": "stream_retry", "attempt": 1, "error": "retry"},
    {"type": "credential_refreshed", "provider": "anthropic", "kind": "auth", "status_code": 401},
    {"type": "compaction", "chars": 123},
    {"type": "turn_complete", "turn": 1, "usage": {"input_tokens": 1}},
    {"type": "max_turns_reached", "turn_count": 5, "max_turns": 5},
    {"type": "budget_exceeded", "total_cost": 1.2, "budget": 1.0},
    {"type": "runtime_guard", "guard": "final_answer", "message": "continue"},
    {"type": "operator_pause", "reason": "operator_pause", "safe_boundary": "before_turn"},
    {"type": "citation_validation", "schema_version": 1, "turn": 1, "violations": [], "violation_count": 0, "warning_codes": []},
    {"type": "heartbeat", "timestamp": 1},
    {
      "type": "interrupted",
      "reason": "recovered_on_attach",
      "runner_id": "runner-1",
      "role": "writer",
      "last_completed_seq": 42,
      "recovered_by_runner_id": "runner-2",
      "recovered_at": 1010.0,
      "safe_boundary": "before_turn",
    },
    {"type": "stream_complete", "usage": {"input_tokens": 1}},
    {"type": "stream_error", "error": "serialize"},
    {"type": "error", "error": "failed"},
    {"type": "tool_approval_request", "tool_call_id": "toolu_1", "approval_id": "appr_1", "nonce": "nonce", "tool_name": "bash", "tool_input": {}, "resolved_qualifier": "bash", "reason": "ask", "allow_persistent_approval": True, "ts": 1.0},
    {"type": "tool_approval_decided", "tool_call_id": "toolu_1", "tool_name": "bash", "outcome": "approved", "decision_source": "user_approved", "allow_tool_type_applied": True, "ts": 1.0},
    {"type": "headless_auto_deny", "tool_call_id": "toolu_1", "tool_name": "bash", "reason": "blocked", "source": "static"},
    {"type": "skill_run_started", "skill_run_id": "run-1", "skill": "model-review", "ticker": "MSFT", "ts": 1.0, "scope": "ticker", "portfolio_id": None},
    {
      "type": "skill_result_captured",
      "skill_run_id": "run-1",
      "skill": "sniff-test",
      "ticker": "MSFT",
      "exit_code": 0,
      "outcome": "success",
      "status": "noop",
      "gate_code": "PROCEED",
      "artifact_refs": ["artifacts/MSFT/sniff-test/result.json"],
      "proposal_ids": [],
      "verdict_echo": {"verdict": "PROCEED"},
      "fms_results": [{"status": "noop"}],
      "artifact_events": [{"type": "artifact_ready"}],
      "output_memory_file": "skills/sniff-test/2026-06-11-run.md",
      "cost_usd": 0.01,
      "duration_s": 12.3,
      "error": None,
      "warnings": [],
    },
    {"type": "typed_recommendations_extracted", "skill": "model-review", "workflow_name": "build-model", "scope": "ticker", "ticker": "MSFT", "portfolio_id": None, "recommendations_count": 1, "verdict_code": "MODEL_HAS_ISSUES", "validation_errors": [], "warnings": [], "source_artifact_path": "skills/model-review.md", "ts": 1.0},
    {
      "type": "artifact_ready",
      "skill_run_id": "run-1",
      "ticker": "MSFT",
      "skill": "model-review",
      "subcommand": "report_model_review",
      "mutation_mode": "preview",
      "artifact_ref": "artifact.json",
      "artifact_id": "artifact-1",
      "artifact_path": "artifact.json",
      "binary_artifact_path": None,
      "proposal_id": None,
      "contract_name": "ModelReview",
      "data_source": "live",
      "status": "noop",
      "gate_code": "PROCEED",
      "sidecar_hash": "sha256:" + "ab" * 32,
      "verdict_echo": {"verdict": "PROCEED"},
      "ts": 1.0,
      "scope": "ticker",
      "portfolio_id": None,
    },
    {"type": "artifact_failed", "skill_run_id": "run-1", "ticker": "MSFT", "skill": "model-review", "error_code": "validation", "error_detail": "bad", "source_path": "source.md", "tool_call_id": "toolu_1", "ts": 1.0},
    {"type": "artifact_unavailable", "ticker": "MSFT", "skill": "model-review", "reason": "stale", "affordance": "rerun", "ts": 1.0},
    {"type": "aggregate_ready", "skill_run_id": "run-1", "ticker": "MSFT", "view_model_id": "vm-1", "trigger": {"kind": "artifact_ready", "source": "model-review"}, "sources_complete": True, "ts": 1.0},
    {"type": "artifact_updated", "skill_run_id": "run-1", "ticker": "MSFT", "skill": "model-review", "artifact_id": "artifact-1", "contract_name": "ModelReview", "partial_view_model": {"status": "partial"}, "ts": 1.0},
    {
      "type": "task_registered",
      "task_id": "bg_0",
      "owner_runner_id": "runner-1",
      "owner_role": "writer",
      "sub_agent_id": "sub0:sess_1",
      "parent_turn_id": "turn-1",
      "call_index": 0,
      "task_type": "background",
      "provider_name": "anthropic",
      "model": "claude-sonnet-4-6",
      "original_task_id": "bg_original",
      "agent_name": "researcher",
      "parent_session_id": "sess_1",
      "metadata": {"resumable": True},
      "started_at": 1000.0,
    },
    {
      "type": "task_completed",
      "task_id": "bg_0",
      "owner_runner_id": "runner-1",
      "owner_role": "writer",
      "sub_agent_id": "sub0:sess_1",
      "parent_turn_id": "turn-1",
      "call_index": 0,
      "task_type": "background",
      "provider_name": "anthropic",
      "model": "claude-sonnet-4-6",
      "original_task_id": "bg_original",
      "final_state": "completed",
      "completed_at": 1010.0,
      "result": {"response": "done"},
      "error": None,
    },
    {
      "type": "parent_message_sent",
      "task_id": "bg_0",
      "owner_runner_id": "runner-1",
      "owner_role": "writer",
      "sub_agent_id": "sub0:sess_1",
      "parent_turn_id": "turn-1",
      "call_index": 0,
      "task_type": "background",
      "provider_name": "anthropic",
      "model": "claude-sonnet-4-6",
      "original_task_id": "bg_original",
      "message_id": "msg-1",
      "sender": {"session_id": "sess_1", "user_id": "alice"},
      "sent_at": 1005.0,
      "message": "continue",
    },
    {"type": "session_recap", "session_id": "sess_1", "seq_range": (1, 3), "started_at": 1.0, "ended_at": 2.0, "trigger": "turn_end", "artifacts": [], "verdicts": [], "approvals": [], "tool_calls_summary": {"total_calls": 0, "successes": 0, "errors": 0, "by_tool_name": {}, "by_server": {}}, "failures": [], "usage": None, "ts": 2.0},
  ]

  for event in fixtures:
    allowed = V1_FIELD_PROJECTION[event["type"]]
    assert set(event) <= allowed, event["type"]


def test_v1_adapter_strips_unknown_fields_for_known_type() -> None:
  adapted = V1Adapter().transform(
    {
      "type": "tool_call_start",
      "tool_call_id": "toolu_1",
      "tool_name": "build_model",
      "tool_input": {"ticker": "MSFT"},
      "execution_location": "backend",
      "future_only": "strip-me",
    }
  )

  assert adapted == {
    "type": "tool_call_start",
    "tool_call_id": "toolu_1",
    "tool_name": "build_model",
    "tool_input": {"ticker": "MSFT"},
    "execution_location": "backend",
  }


def test_v1_adapter_preserves_artifact_ready_fms_metadata() -> None:
  event = {
    "type": "artifact_ready",
    "skill_run_id": "run-1",
    "ticker": "MSFT",
    "skill": "sniff-test",
    "subcommand": "report_sniff_test",
    "mutation_mode": "preview",
    "artifact_ref": "artifacts/MSFT/sniff-test/result.json",
    "artifact_id": "result",
    "artifact_path": "artifacts/MSFT/sniff-test/result.json",
    "binary_artifact_path": None,
    "proposal_id": None,
    "contract_name": "sniff-test",
    "data_source": "live",
    "status": "noop",
    "gate_code": "PROCEED",
    "sidecar_hash": "sha256:" + "ab" * 32,
    "verdict_echo": {"verdict": "PROCEED"},
    "ts": 1.0,
    "scope": "ticker",
    "portfolio_id": None,
    "future_only": "strip-me",
  }

  adapted = V1Adapter().transform(event)

  assert adapted == {key: value for key, value in event.items() if key != "future_only"}


def test_v1_adapter_skips_unknown_type() -> None:
  assert V1Adapter().transform({"type": "future_event", "value": 1}) is None


def test_adapt_event_rejects_unsupported_schema_version() -> None:
  with pytest.raises(ValueError, match="schema_version=99"):
    adapt_event({"type": "text_delta", "text": "hello"}, 99)


def test_tool_output_chunk_inner_seq_is_preserved() -> None:
  adapted = adapt_event(
    {
      "type": "tool_output_chunk",
      "tool_call_id": "toolu_1",
      "tool_name": "code_execute",
      "stream": "stdout",
      "text": "line",
      "seq": 5,
      "future_only": "strip-me",
    },
    1,
  )

  assert adapted == {
    "type": "tool_output_chunk",
    "tool_call_id": "toolu_1",
    "tool_name": "code_execute",
    "stream": "stdout",
    "text": "line",
    "seq": 5,
  }


def test_artifact_failed_preserves_tool_call_id() -> None:
  adapted = adapt_event(
    {
      "type": "artifact_failed",
      "skill_run_id": "run-1",
      "ticker": None,
      "skill": "_html",
      "error_code": "tool_write_failed",
      "error_detail": "bad html",
      "source_path": None,
      "tool_call_id": "tool_html",
      "future_only": "strip-me",
      "ts": 1.0,
    },
    1,
  )

  assert adapted == {
    "type": "artifact_failed",
    "skill_run_id": "run-1",
    "ticker": None,
    "skill": "_html",
    "error_code": "tool_write_failed",
    "error_detail": "bad html",
    "source_path": None,
    "tool_call_id": "tool_html",
    "ts": 1.0,
  }


def test_credential_refreshed_preserves_current_runner_fields() -> None:
  adapted = adapt_event(
    {
      "type": "credential_refreshed",
      "provider": "anthropic",
      "kind": "auth",
      "status_code": 401,
      "future_only": "strip-me",
    },
    1,
  )

  assert adapted == {
    "type": "credential_refreshed",
    "provider": "anthropic",
    "kind": "auth",
    "status_code": 401,
  }
