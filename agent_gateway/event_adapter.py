from __future__ import annotations

from typing import Protocol

from .control_run_lifecycle import canonical_control_run_state
from .events import DEFAULT_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS


def _fields(*names: str) -> frozenset[str]:
  return frozenset({"type", "product_id", *names})


V1_WIRE_EVENT_TYPES: frozenset[str] = frozenset(
  {
    "text_delta",
    "thinking_delta",
    "tool_call_start",
    "tool_call_complete",
    "tool_call_interrupted",
    "tool_output_chunk",
    "tool_execute_request",
    "interceptor_decision",
    "stream_retry",
    "credential_refreshed",
    "compaction",
    "turn_complete",
    "max_turns_reached",
    "budget_exceeded",
    "runtime_guard",
    "operator_pause",
    "citation_validation",
    "heartbeat",
    "interrupted",
    "stream_complete",
    "stream_error",
    "error",
    "tool_approval_request",
    "tool_approval_decided",
    "approval_delivery_acknowledged",
    "headless_auto_deny",
    "skill_run_started",
    "skill_result_captured",
    "typed_recommendations_extracted",
    "artifact_ready",
    "ui_blocks_ready",
    "workflow_output_attached",
    "artifact_failed",
    "artifact_unavailable",
    "aggregate_ready",
    "artifact_updated",
    "task_registered",
    "task_completed",
    "agent_completion",
    "parent_message_sent",
    "parent_message_consumed",
    "session_recap",
  }
)

V1_FIELD_PROJECTION: dict[str, frozenset[str]] = {
  "text_delta": _fields("text"),
  "thinking_delta": _fields("text"),
  "tool_call_start": _fields(
    "tool_call_id",
    "tool_name",
    "tool_input",
    "execution_location",
    "call_index",
    "server",
    "started_at",
    "parent_assistant_message_seq",
    "display",
  ),
  "tool_call_complete": _fields(
    "tool_call_id",
    "tool_name",
    "result",
    "error",
    "duration_ms",
    "server",
    "is_error",
    "semantic_error",
    "execution_location",
    "final_tool_result_blocks",
  ),
  "tool_call_interrupted": _fields(
    "tool_call_id",
    "tool_name",
    "tool_input",
    "original_started_at",
    "discovered_at",
    "reason",
    "tool_risk",
    "runner_id",
    "role",
    "sub_agent_id",
  ),
  "tool_output_chunk": _fields("tool_call_id", "tool_name", "stream", "text", "seq"),
  "tool_execute_request": _fields("tool_call_id", "nonce", "expires_at", "tool_name", "tool_input", "display"),
  "interceptor_decision": _fields("tool_call_id", "tool_name", "action", "code", "message"),
  "stream_retry": _fields("attempt", "error"),
  "credential_refreshed": _fields("provider", "kind", "status_code", "auth_mode", "reason"),
  "compaction": _fields("chars"),
  "turn_complete": _fields("turn", "usage"),
  "max_turns_reached": _fields("turn_count", "max_turns"),
  "budget_exceeded": _fields("total_cost", "budget"),
  "runtime_guard": _fields("guard", "message"),
  "operator_pause": _fields("reason", "safe_boundary"),
  "citation_validation": _fields(
    "schema_version",
    "turn",
    "violations",
    "violation_count",
    "validator_error_code",
    "warning_codes",
    "pending_task_count",
    "total_claims_detected",
    "total_sources_in_registry",
    "registry_scope",
    "judge_called",
    "judge_path",
    "duration_ms",
  ),
  "heartbeat": _fields("timestamp"),
  "interrupted": _fields(
    "reason",
    "runner_id",
    "role",
    "last_completed_seq",
    "recovered_by_runner_id",
    "recovered_at",
    "safe_boundary",
  ),
  "stream_complete": _fields(
    "usage",
    "terminal_disposition",
    "reason",
  ),
  "stream_error": _fields("error"),
  "error": _fields("error"),
  "tool_approval_request": _fields(
    "tool_call_id",
    "approval_id",
    "nonce",
    "tool_name",
    "tool_input",
    "resolved_qualifier",
    "reason",
    "allow_persistent_approval",
    "ts",
  ),
  "tool_approval_decided": _fields(
    "tool_call_id",
    "tool_name",
    "outcome",
    "decision_source",
    "allow_tool_type_applied",
    "ts",
  ),
  "approval_delivery_acknowledged": _fields(
    "launch_nonce",
    "delivery_sequence",
    "approval_id",
    "tool_call_id",
    "nonce",
    "task_id",
    "control_run_id",
    "session_id",
    "channel_id",
    "approved",
    "allow_tool_type",
    "decided_at_ns",
  ),
  "headless_auto_deny": _fields("tool_call_id", "tool_name", "reason", "source"),
  "skill_run_started": _fields("skill_run_id", "skill", "ticker", "ts", "scope", "portfolio_id"),
  "skill_result_captured": _fields(
    "skill_run_id",
    "skill",
    "ticker",
    "scope",
    "portfolio_id",
    "exit_code",
    "outcome",
    "status",
    "gate_code",
    "artifact_refs",
    "proposal_ids",
    "verdict_echo",
    "fms_results",
    "artifact_events",
    "output_memory_file",
    "cost_usd",
    "duration_s",
    "compaction_count",
    "error",
    "warnings",
    "approval_outcome",
    "approval_id",
    "approval_tool_name",
  ),
  "typed_recommendations_extracted": _fields(
    "skill",
    "workflow_name",
    "scope",
    "ticker",
    "portfolio_id",
    "recommendations_count",
    "verdict_code",
    "validation_errors",
    "warnings",
    "source_artifact_path",
    "ts",
  ),
  "artifact_ready": _fields(
    "skill_run_id",
    "ticker",
    "skill",
    "subcommand",
    "mutation_mode",
    "artifact_ref",
    "artifact_id",
    "artifact_path",
    "binary_artifact_path",
    "proposal_id",
    "contract_name",
    "data_source",
    "status",
    "gate_code",
    "sidecar_hash",
    "verdict_echo",
    "ts",
    "scope",
    "portfolio_id",
    "origin",
  ),
  "ui_blocks_ready": _fields(
    "session_id",
    "skill_run_id",
    "turn_key",
    "emission_index",
    "ui_blocks_id",
    "contract_version",
    "payload",
    "text_fallback",
    "ts",
  ),
  "workflow_output_attached": _fields(
    "assistant_message_seq",
    "kind",
    "delivery_envelope",
    "read",
  ),
  "artifact_failed": _fields(
    "skill_run_id",
    "ticker",
    "skill",
    "error_code",
    "error_detail",
    "source_path",
    "tool_call_id",
    "ts",
  ),
  "artifact_unavailable": _fields("ticker", "skill", "reason", "affordance", "ts"),
  "aggregate_ready": _fields("skill_run_id", "ticker", "view_model_id", "trigger", "sources_complete", "ts"),
  "artifact_updated": _fields(
    "skill_run_id",
    "ticker",
    "skill",
    "artifact_id",
    "contract_name",
    "partial_view_model",
    "ts",
  ),
  "task_registered": _fields(
    "task_id",
    "owner_runner_id",
    "owner_role",
    "sub_agent_id",
    "parent_turn_id",
    "call_index",
    "task_type",
    "provider_name",
    "model",
    "original_task_id",
    "agent_name",
    "parent_session_id",
    "metadata",
    "started_at",
  ),
  "task_completed": _fields(
    "task_id",
    "owner_runner_id",
    "owner_role",
    "sub_agent_id",
    "parent_turn_id",
    "call_index",
    "task_type",
    "provider_name",
    "model",
    "original_task_id",
    "final_state",
    "completed_at",
    "result",
    "error",
  ),
  "agent_completion": _fields(
    "event_id",
    "fingerprint",
    "task_id",
    "envelope",
    "ts",
  ),
  "parent_message_sent": _fields(
    "task_id",
    "owner_runner_id",
    "owner_role",
    "sub_agent_id",
    "parent_turn_id",
    "call_index",
    "task_type",
    "provider_name",
    "model",
    "original_task_id",
    "message_id",
    "sender",
    "sent_at",
    "message",
  ),
  "parent_message_consumed": _fields(
    "task_id",
    "message_id",
    "parent_message_seq",
    "consumer_turn",
    "assistant_message_seq",
    "consumed_at",
  ),
  "session_recap": _fields(
    "session_id",
    "seq_range",
    "started_at",
    "ended_at",
    "trigger",
    "artifacts",
    "verdicts",
    "approvals",
    "tool_calls_summary",
    "failures",
    "usage",
    "ts",
  ),
}

CONTROL_V1_ATTRIBUTION_FIELDS = frozenset({"run_id", "control_run_id", "sub_agent_id"})
CONTROL_V1_EXTENSION_EVENT_TYPES: frozenset[str] = frozenset(
  {
    "run_state_changed",
    "events_dropped",
    "replay_truncated",
    "malformed_autonomous_event",
  }
)
CONTROL_V1_WIRE_EVENT_TYPES: frozenset[str] = V1_WIRE_EVENT_TYPES | CONTROL_V1_EXTENSION_EVENT_TYPES
CONTROL_V1_FIELD_PROJECTION: dict[str, frozenset[str]] = {
  event_type: fields | CONTROL_V1_ATTRIBUTION_FIELDS
  for event_type, fields in V1_FIELD_PROJECTION.items()
}
CONTROL_V1_FIELD_PROJECTION.update(
  {
    "run_state_changed": _fields("run_id", "control_run_id", "state", "ts"),
    "events_dropped": _fields(
      "run_id",
      "control_run_id",
      "count",
      "oldest_ts",
      "dropped_through_seq",
    ),
    "replay_truncated": _fields("run_id", "control_run_id", "dropped_before_seq"),
    "malformed_autonomous_event": _fields("run_id", "control_run_id", "raw"),
  }
)


class SchemaAdapter(Protocol):
  def transform(self, event: dict) -> dict | None: ...


class V1Adapter:
  def transform(self, event: dict) -> dict | None:
    event_type = event.get("type")
    if event_type not in V1_WIRE_EVENT_TYPES:
      return None
    allowed_keys = V1_FIELD_PROJECTION.get(str(event_type), frozenset())
    return {key: value for key, value in event.items() if key in allowed_keys}


class ControlV1Adapter:
  def transform(self, event: dict) -> dict | None:
    event_type = event.get("type")
    if event_type not in CONTROL_V1_WIRE_EVENT_TYPES:
      return None
    allowed_keys = CONTROL_V1_FIELD_PROJECTION.get(str(event_type), frozenset())
    projected = {key: value for key, value in event.items() if key in allowed_keys}
    if event_type == "run_state_changed" and "state" in projected:
      projected["state"] = canonical_control_run_state(projected["state"])
    return projected


ADAPTERS: dict[int, SchemaAdapter] = {
  DEFAULT_SCHEMA_VERSION: V1Adapter(),
}
CONTROL_ADAPTERS: dict[int, SchemaAdapter] = {
  DEFAULT_SCHEMA_VERSION: ControlV1Adapter(),
}


def adapt_event(event: dict, schema_version: int) -> dict | None:
  adapter = ADAPTERS.get(schema_version)
  if adapter is None:
    raise ValueError(f"No adapter for schema_version={schema_version}")
  return adapter.transform(event)


def adapt_control_event(event: dict, schema_version: int) -> dict | None:
  adapter = CONTROL_ADAPTERS.get(schema_version)
  if adapter is None:
    raise ValueError(f"No control adapter for schema_version={schema_version}")
  return adapter.transform(event)


__all__ = [
  "ADAPTERS",
  "CONTROL_ADAPTERS",
  "CONTROL_V1_EXTENSION_EVENT_TYPES",
  "CONTROL_V1_FIELD_PROJECTION",
  "CONTROL_V1_WIRE_EVENT_TYPES",
  "DEFAULT_SCHEMA_VERSION",
  "SUPPORTED_SCHEMA_VERSIONS",
  "ControlV1Adapter",
  "SchemaAdapter",
  "V1Adapter",
  "V1_FIELD_PROJECTION",
  "V1_WIRE_EVENT_TYPES",
  "adapt_control_event",
  "adapt_event",
]
