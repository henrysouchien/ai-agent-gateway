from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import importlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from agent_workflow_contracts import AgentCompletionEnvelope, TaskResult

from .artifact_readback import readback_artifact_ready_event
from .runner_background_tasks import (
  _BACKGROUND_RESULT_ACK_RESULT_KEY,
)
from .runner_session_events import (
  build_tool_call_complete_event as _build_tool_call_complete_event,
  build_tool_call_start_event as _build_tool_call_start_event,
)
from .runner_session_lifecycle import _runner_attr
from .runner_state import ToolResultContext
from .runner_tool_audit import redact_tool_input_for_event as _redact_tool_input_for_event
from .secret_boundary import (
  sanitize_boundary_value,
  sanitize_tool_event,
  sanitization_failure_tool_input,
)
from .tool_display import resolve_display
from .tool_dispatch_classification import (
  DEFAULT_MAX_TOOL_RETRIES as _MAX_TOOL_DISPATCH_RETRIES,
  RETRYABLE_OUTCOMES as _RETRYABLE_DISPATCH_OUTCOMES,
  build_dispatch_record,
  classify_semantic_tool_error,
  classify_tool_outcome,
  resolve_dispatch_entry,
  retry_backoff_seconds,
  retry_decision,
)
from .workflow_evidence_provenance import (
  WORKFLOW_EVIDENCE_PROJECTION_RESULT_KEY as _WORKFLOW_EVIDENCE_PROJECTION_RESULT_KEY,
)
from .workflow_output_attachment import (
  WorkflowOutputAttachment,
  WorkflowOutputAttachmentError,
  accepted_workflow_continuation_run_id,
  completed_workflow_output_attachment,
  record_workflow_output_attachment,
)


log = logging.getLogger("agent_gateway.runner")
_RUN_AGENT_DISPATCH_TIMEOUT_SECONDS = 2100.0
_ACTIVE_SKILL_ALLOW_RESULT_KEY = "_active_skill_allow"
_ACTIVE_SKILL_DENY_RESULT_KEY = "_active_skill_deny"
_ACTIVE_SKILL_REPORT_DOORS_RESULT_KEY = "_active_skill_report_doors"
_READABLE_RESOURCE_SNAPSHOT_RESULT_KEY = "_readable_resource_snapshot"
_READABLE_RESOURCE_MAX_CONTENT_BYTES = 2_000_000
_REPEATED_TOOL_EXCLUDED_STOP_AFTER_COUNT = 2
_FMS_WRITER_TOOL_FALLBACKS = frozenset({
  "fms_link_thesis",
  "fms_persist_business_model",
  "fms_persist_dcf_relative_valuation",
  "fms_persist_earnings_scenarios",
  "fms_persist_forecast_assumptions",
  "fms_persist_model_update",
  "fms_persist_scenario_multiple_pricing",
  "fms_persist_ticker_triage",
  "fms_persist_valuation_inputs",
  "fms_record_decision_log",
  "fms_report_business_quality_assessment",
  "fms_report_idea_to_thesis",
  "fms_report_thesis_consultation",
  "fms_resolve_outcome_contracts",
})
_FMS_COMMIT_TOOL_ACTION_CODES = {
  "fms_link_thesis": "link_thesis",
  "fms_persist_business_model": "persist_business_model",
  "fms_persist_dcf_relative_valuation": "persist_dcf_relative_valuation",
  "fms_persist_earnings_scenarios": "persist_earnings_scenarios",
  "fms_persist_forecast_assumptions": "persist_forecast_assumptions",
  "fms_persist_model_update": "persist_model_update",
  "fms_persist_scenario_multiple_pricing": "persist_scenario_multiple_pricing",
  "fms_persist_ticker_triage": "persist_ticker_triage",
  "fms_persist_valuation_inputs": "persist_valuation_inputs",
  "fms_record_decision_log": "record_decision_log",
  "fms_report_business_quality_assessment": "report_business_quality_assessment",
  "fms_report_idea_to_thesis": "report_idea_to_thesis",
  "fms_report_thesis_consultation": "report_thesis_consultation",
  "fms_resolve_outcome_contracts": "resolve_outcome_contracts",
}
_FMS_COMMIT_TOOL_STAGES = {
  "fms_link_thesis": "research",
  "fms_persist_business_model": "bm",
  "fms_persist_dcf_relative_valuation": "valuation",
  "fms_persist_earnings_scenarios": "scenarios",
  "fms_persist_forecast_assumptions": "forecast",
  "fms_persist_model_update": "build",
  "fms_persist_scenario_multiple_pricing": "valuation",
  "fms_persist_ticker_triage": "research",
  "fms_persist_valuation_inputs": "valuation",
  "fms_record_decision_log": "review",
  "fms_report_business_quality_assessment": "diligence",
  "fms_report_idea_to_thesis": "research",
  "fms_report_thesis_consultation": "diligence",
  "fms_resolve_outcome_contracts": "review",
}
_OUTPUT_FILE_GATED_TOOL_FALLBACKS = frozenset({
  "analyze_stock",
})
_OUTPUT_FILE_GATED_TOOL_ALTERNATIVES: dict[str, dict[str, Any]] = {
  "analyze_stock": {
    "suggested_tools": ["get_quote", "industry_peer_comparison"],
    "resolution": (
      "Use inline read tools instead: get_quote for price/profile context and "
      "industry_peer_comparison(symbol=...) for peer metrics. For methodology-backed "
      "risk analysis, use the quantifying-risk skill."
    ),
  },
}


def _canonical_agent_result_payload(result: Any) -> tuple[Any, dict[str, Any] | None]:
  """Serialize only the normalized agent-result wire models.

  Dispatcher handlers are allowed to keep the strongly typed value in-process,
  but the durable tool event and model-visible JSON boundary must never fall
  through ``json.dumps(default=str)``.  This is deliberately not a generic
  Pydantic adapter: only the two admitted parent-facing result contracts cross
  this seam.

  ``AgentCompletionEnvelope.child_evidence`` is stripped here rather than
  dumped: it is runtime provenance the parent runtime consumes, not text the
  model should pay tokens to read, and it rides the same private
  ``ToolResultContext.child_evidence`` channel as the workflow projection
  (D-B4-2).  Returns ``(payload, child_evidence)``.
  """

  if isinstance(result, AgentCompletionEnvelope):
    payload = result.model_dump(mode="json")
    child_evidence = payload.pop("child_evidence", None)
    return payload, (child_evidence if isinstance(child_evidence, dict) else None)
  if isinstance(result, TaskResult):
    return result.model_dump(mode="json"), None
  return result, None


def _is_accepted_ui_blocks_result(tool_name: str, result: Any, error: Any) -> bool:
  if tool_name != "emit_ui_blocks" or error is not None or not isinstance(result, dict):
    return False
  accepted = result.get("accepted")
  return isinstance(accepted, dict) and isinstance(accepted.get("ui_blocks_id"), str)


def _record_tool_excluded_attempt(runner: Any, tool_name: str) -> int:
  counts = getattr(runner, "_tool_excluded_attempt_counts", None)
  if not isinstance(counts, dict):
    counts = {}
    setattr(runner, "_tool_excluded_attempt_counts", counts)
  count = int(counts.get(tool_name, 0)) + 1
  counts[tool_name] = count
  return count


def _augment_repeated_tool_excluded_error(
  error: Dict[str, Any],
  *,
  tool_name: str,
  exclusion_count: int,
) -> Dict[str, Any]:
  augmented = dict(error)
  data = dict(augmented.get("data") or {})
  resolution = (
    "Do not retry this excluded tool in the current context. Use an available "
    "tool path, emit the appropriate blocked/partial verdict, or finish with "
    "the durable evidence already produced."
  )
  data.update({
    "blocked_tool": tool_name,
    "exclusion_count": exclusion_count,
    "repeated_tool_excluded": True,
    "stop_after_tool_results": True,
    "resolution": resolution,
  })
  augmented["data"] = data
  augmented["sub_code"] = "repeated_tool_excluded"
  augmented["message"] = (
    f"Tool '{tool_name}' is not available in this context and was retried "
    f"{exclusion_count} times. {resolution}"
  )
  return augmented


def _readable_resource_created_at(timestamp: float) -> str:
  return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _readable_resource_event_from_snapshot(
  runner: Any,
  snapshot: Any,
  *,
  tool_call_id: str,
  tool_name: str,
  timestamp: float,
) -> dict[str, Any] | None:
  if not isinstance(snapshot, dict):
    return None
  content = snapshot.get("content")
  if not isinstance(content, str) or not content.strip():
    return None
  content_bytes_payload = content.encode("utf-8")
  if len(content_bytes_payload) > _READABLE_RESOURCE_MAX_CONTENT_BYTES:
    return None
  content_sha256 = snapshot.get("content_sha256")
  if not isinstance(content_sha256, str) or not content_sha256.strip():
    return None
  normalized_sha256 = content_sha256.lower()
  if hashlib.sha256(content_bytes_payload).hexdigest() != normalized_sha256:
    return None
  source_path = snapshot.get("source_path")
  if not isinstance(source_path, str) or not source_path.strip():
    return None
  contract_name = snapshot.get("contract_name")
  if not isinstance(contract_name, str) or not contract_name.strip():
    return None
  content_type = snapshot.get("content_type")
  if content_type not in {"text/markdown", "text/plain"}:
    return None
  content_class = snapshot.get("content_class")
  if content_class != "human_readable":
    return None
  content_snapshot_id = snapshot.get("content_snapshot_id")
  if not isinstance(content_snapshot_id, str) or not content_snapshot_id.strip():
    return None
  truncated = snapshot.get("truncated")
  if not isinstance(truncated, bool):
    return None
  control_run_id = str(os.getenv("AGENT_AUTONOMOUS_CONTROL_RUN_ID") or getattr(runner, "_full_session_id", "")).strip()
  if not control_run_id:
    return None
  seed = "\0".join([control_run_id, tool_call_id, source_path, normalized_sha256])
  resource_id = f"rr:{hashlib.sha256(seed.encode('utf-8')).hexdigest()}"
  skill_run_id = str(getattr(runner, "_skill_run_id", "") or "").strip() or f"tool:{tool_call_id}"
  content_bytes = snapshot.get("content_bytes")
  if not isinstance(content_bytes, int) or isinstance(content_bytes, bool):
    return None
  if content_bytes != len(content_bytes_payload):
    return None
  event: dict[str, Any] = {
    "type": "readable_resource_ready",
    "resource_id": resource_id,
    "run_id": control_run_id,
    "control_run_id": control_run_id,
    "skill_run_id": skill_run_id,
    "contract_name": contract_name.strip(),
    "content_type": content_type,
    "content_class": content_class,
    "content_snapshot_id": content_snapshot_id.strip(),
    "content_sha256": normalized_sha256,
    "content_bytes": content_bytes,
    "content": content,
    "truncated": truncated,
    "title": str(snapshot.get("title") or source_path),
    "source_path": source_path,
    "tool_name": str(snapshot.get("tool_name") or tool_name),
    "tool_call_id": tool_call_id,
    "created_at": _readable_resource_created_at(timestamp),
    "ts": timestamp,
  }
  for key in ("byte_start", "byte_end"):
    value = snapshot.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
      event[key] = value
  return event


def _fms_commit_tool_names() -> frozenset[str]:
  for module_name in ("agent.shared.server_policies", "api.agent.shared.server_policies"):
    try:
      server_policies = importlib.import_module(module_name)
    except Exception:
      continue
    names: set[str] = set()
    for attr_name in ("FMS_MODEL_WRITER_TOOLS", "FMS_THESIS_WRITER_TOOLS"):
      raw_names = getattr(server_policies, attr_name, frozenset())
      try:
        names.update(str(tool_name) for tool_name in raw_names if str(tool_name or "").strip())
      except TypeError:
        continue
    if names:
      return frozenset(names)
  return _FMS_WRITER_TOOL_FALLBACKS


def _output_file_gated_tool_names() -> frozenset[str]:
  for module_name in ("agent.shared.server_policies", "api.agent.shared.server_policies"):
    try:
      server_policies = importlib.import_module(module_name)
    except Exception:
      continue
    getter = getattr(server_policies, "get_output_file_tools", None)
    if not callable(getter):
      continue
    try:
      names = getter()
    except Exception:
      continue
    try:
      return frozenset(str(tool_name) for tool_name in names if str(tool_name or "").strip())
    except TypeError:
      continue
  return _OUTPUT_FILE_GATED_TOOL_FALLBACKS


def _fms_commit_blocker_error(tool_name: str) -> Dict[str, Any] | None:
  normalized = str(tool_name or "").strip()
  if normalized not in _fms_commit_tool_names():
    return None
  action_code = _FMS_COMMIT_TOOL_ACTION_CODES.get(normalized, normalized)
  stage = _FMS_COMMIT_TOOL_STAGES.get(normalized, "build")
  resolution = (
    "Run this commit tool in an interactive model-writer/thesis-writer "
    "session with operator approval, then retry the blocked workflow."
  )
  message = (
    f"Tool '{normalized}' is a canonical FMS commit tool and requires interactive "
    "approval in this context. Emit BUILD_BLOCKED with error.data.pending_action, "
    "preserve any approval-ready payload, then retry from an interactive session "
    "after operator approval."
  )
  return {
    "code": "tool_excluded",
    "sub_code": "requires_interactive_approval",
    "message": message,
    "data": {
      "blocked_tool": normalized,
      "tool_family": "fms_commit",
      "tool_class": "state_write",
      "requires_interactive_approval": True,
      "recommended_verdict": "BUILD_BLOCKED",
      "resolution": resolution,
      "pending_action": {
        "code": action_code,
        "stage": stage,
        "message": f"Run {normalized} interactively with operator approval, then retry the workflow.",
        "severity": "blocking",
        "target": normalized,
        "source": "runner_tool_exclusion",
        "metadata": {
          "blocked_tool": normalized,
          "requires_interactive_approval": True,
          "tool_class": "state_write",
          "resolution": resolution,
        },
      },
    },
  }


def _output_file_gated_tool_error(tool_name: str) -> Dict[str, Any] | None:
  normalized = str(tool_name or "").strip()
  if normalized not in _output_file_gated_tool_names():
    return None
  alternatives = _OUTPUT_FILE_GATED_TOOL_ALTERNATIVES.get(normalized, {})
  resolution = str(
    alternatives.get("resolution")
    or "Use an available inline read tool, or retry from an approved file-output workflow."
  )
  data: Dict[str, Any] = {
    "blocked_tool": normalized,
    "tool_class": "read",
    "output_file_gated": True,
    "resolution": resolution,
  }
  suggested_tools = alternatives.get("suggested_tools")
  if isinstance(suggested_tools, list) and suggested_tools:
    data["suggested_tools"] = [str(tool) for tool in suggested_tools if str(tool or "").strip()]
  return {
    "code": "tool_excluded",
    "sub_code": "output_file_gated_tool_excluded",
    "message": (
      f"Tool '{normalized}' is output-file gated and is not available in this "
      f"context without an approved output='file' workflow. {resolution}"
    ),
    "data": data,
  }


def _model_error_data(error: Dict[str, Any]) -> Dict[str, Any] | None:
  raw_data = error.get("data")
  data: Dict[str, Any] = dict(raw_data) if isinstance(raw_data, dict) else {}
  hint = error.get("tool_usage_hint")
  if isinstance(hint, str) and hint.strip() and "tool_usage_hint" not in data:
    data["tool_usage_hint"] = hint
  return data or None


def _error_with_model_error_data(error: Dict[str, Any]) -> Dict[str, Any]:
  data = _model_error_data(error)
  if data is None or error.get("data") == data:
    return error
  enriched = dict(error)
  enriched["data"] = data
  return enriched


class RunnerToolExecutionMixin:
  async def _execute_single_tool(
    self,
    tool_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
    base_kwargs: Dict[str, Any],
    call_index: int = 0,
  ) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
    json_module = _runner_attr(self, "json", json)
    logger = _runner_attr(self, "log", log)
    time_module = _runner_attr(self, "time", time)
    asyncio_module = _runner_attr(self, "asyncio", asyncio)
    timeout_error_type = getattr(asyncio_module, "TimeoutError", asyncio.TimeoutError)
    cancelled_error_type = getattr(asyncio_module, "CancelledError", asyncio.CancelledError)

    effective_tool_input = tool_input
    resolve_effective_tool_input = getattr(self._dispatcher, "resolve_effective_tool_input", None)
    if callable(resolve_effective_tool_input):
      try:
        effective_tool_input = resolve_effective_tool_input(tool_name, tool_input)
      except Exception:
        effective_tool_input = tool_input
    tool_input = effective_tool_input

    redacted_tool_input = _runner_attr(self, "_redact_tool_input_for_event", _redact_tool_input_for_event)(
      tool_name,
      tool_input,
    )
    redacted_tool_input = sanitize_boundary_value(
      redacted_tool_input,
      sink="tool_input",
      boundary=getattr(self, "_secret_boundary", None),
    )
    if not isinstance(redacted_tool_input, dict):
      redacted_tool_input = sanitization_failure_tool_input()
    tool_input_preview = json_module.dumps(redacted_tool_input, default=str)[:200]
    logger.info(
      "[%s] Tool call: %s | input=%s",
      self._sid,
      tool_name,
      tool_input_preview,
      extra={
        "data": {
          "event": "tool_call",
          "session_id": self._sid,
          "tool": tool_name,
          "input_preview": tool_input_preview,
        }
      },
    )
    if tool_name == "emit_canvas_artifact":
      from .canvas_kit_contract import limits as canvas_kit_limits

      source_bytes = len((tool_input.get("tsx_source") or "").encode("utf-8"))
      source_cap = canvas_kit_limits()["source_max_bytes"]
      if source_bytes > source_cap:
        return (
          self._make_error_result(
            tool_id,
            "invalid_input",
            f"emit_canvas_artifact: tsx_source payload {source_bytes} bytes exceeds {source_cap} byte limit",
          ),
          tool_name,
          [],
        )
    if tool_name == "emit_dashboard_artifact":
      payload_bytes = len(json_module.dumps(tool_input.get("payload") or {}).encode("utf-8"))
      if payload_bytes > 256 * 1024:
        return (
          self._make_error_result(
            tool_id,
            "invalid_input",
            f"emit_dashboard_artifact: payload {payload_bytes} bytes exceeds 256KB limit",
          ),
          tool_name,
          [],
        )
    tool_t0 = time_module.time()
    server = self._mcp_client.get_server_for_tool(tool_name) if self._mcp_client is not None else None
    get_provider_id = (
      getattr(self._mcp_client, "get_provider_id_for_tool", None)
      if self._mcp_client is not None
      else None
    )
    provider_id = get_provider_id(tool_name) if callable(get_provider_id) else None
    dispatch_entry = resolve_dispatch_entry(
      tool_name,
      server=server,
      provider_id=provider_id,
    )
    display = _runner_attr(self, "resolve_display", resolve_display)(tool_name, redacted_tool_input)
    tool_start_event = _runner_attr(self, "_build_tool_call_start_event", _build_tool_call_start_event)(
      tool_call_id=tool_id,
      tool_name=tool_name,
      tool_input=redacted_tool_input,
      call_index=call_index,
      server=server,
      started_at=tool_t0,
      parent_assistant_message_seq=self._last_assistant_message_seq,
    )
    if display is not None:
      tool_start_event["display"] = display
    await self._append_durable_event(tool_start_event)
    self._append(tool_start_event)
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    semantic_error: Optional[Dict[str, Any]] = None
    cancelled_exc: BaseException | None = None
    dispatch_attempts = 1
    dispatch_retries_exhausted = False
    result_bytes = 0
    duration_ms = 0
    load_servers_signal: Optional[List[str]] = None
    load_local_tools_signal: Optional[List[str]] = None
    readable_resource_snapshot: dict[str, Any] | None = None
    child_evidence: dict[str, Any] | None = None
    background_result_ack: tuple[str, int] | None = None
    workflow_output_attachment: WorkflowOutputAttachment | None = None
    superseded_continuation_run_id: str | None = None

    try:
      if tool_name in self._effective_excluded_tools():
        error = _fms_commit_blocker_error(tool_name) or _output_file_gated_tool_error(tool_name) or {
          "code": "tool_excluded",
          "message": f"Tool '{tool_name}' is not available in this context",
        }
        exclusion_count = _record_tool_excluded_attempt(self, tool_name)
        if exclusion_count >= _REPEATED_TOOL_EXCLUDED_STOP_AFTER_COUNT:
          error = _augment_repeated_tool_excluded_error(
            error,
            tool_name=tool_name,
            exclusion_count=exclusion_count,
          )
          setattr(self, "_stop_after_tool_results_reason", "repeated_tool_excluded")
          setattr(self, "_stop_after_tool_results_tool_name", tool_name)
      else:
        dispatch_kwargs: Dict[str, Any] = {"call_index": call_index}
        if getattr(self, "_dispatcher_accepts_advertised_tool_names", False):
          dispatch_kwargs["advertised_tool_names"] = base_kwargs.get(
            "_request_advertised_tool_names"
          )
        if self._dispatcher_accepts_abort_event:
          dispatch_kwargs["abort_event"] = self._tool_abort_event
        if self._dispatcher_accepts_skill_run_context:
          dispatch_kwargs["skill_run_id"] = self._skill_run_id
          dispatch_kwargs["workspace_dir"] = self._workspace_dir
          if getattr(self, "_batch_id", None) is not None:
            dispatch_kwargs["batch_id"] = self._batch_id
        if getattr(self, "_dispatcher_accepts_readable_resource_snapshot", False) and tool_name == "memory_write":
          dispatch_kwargs["capture_readable_resource_snapshot"] = True
        needs_approval = False
        requires_approval_fn = getattr(self._dispatcher, "requires_approval", None)
        if requires_approval_fn is not None:
          try:
            needs_approval = requires_approval_fn(tool_name, tool_input)
          except Exception:
            pass
        # MCP tools already carry per-server read timeouts in McpClientManager.
        # Applying the runner's generic cap here would mask longer server policy.
        has_mcp_server_timeout = server is not None
        # run_name_pipeline and workflow_run dispatch captured LLM work whose
        # duration is fundamentally unknowable — wall-clock caps on LLM work are the
        # ACUI-25 anti-pattern (a cap terminally killed legitimate slow turns
        # before the stall guard could act). Liveness for LLM work is the
        # event-gap stall watchdog, never a timeout (operator ruling,
        # 2026-08-01); the generic cap must not apply.
        skip_timeout = (
          tool_name == "get_background_result"
          or tool_name == "run_name_pipeline"
          or tool_name == "workflow_run"
          or needs_approval
          or has_mcp_server_timeout
        )
        effective_tool_timeout = self._tool_call_timeout
        if tool_name == "run_agent":
          # Sub-agents legitimately outrun the generic tool cap, but skipping the
          # cap entirely let a wedged run_agent hold the chat turn open forever
          # (ACUI-1). The inner spawn timeout (DEFAULT_SUB_AGENT_TIMEOUT_SECONDS)
          # is the primary bound and returns a clean tool error; this widened cap
          # is the backstop if that inner await never resolves. It applies even
          # when the generic tool_call_timeout is disabled (None) so inline
          # run_agent is never unbounded.
          effective_tool_timeout = max(
            effective_tool_timeout or 0.0,
            _runner_attr(self, "_RUN_AGENT_DISPATCH_TIMEOUT_SECONDS", _RUN_AGENT_DISPATCH_TIMEOUT_SECONDS),
          )
        # B-2: one bounded, jittered retry loop owns transient dispatch
        # failure. Both arms are inside it, because the skip_timeout
        # population (MCP server timeouts) is exactly where 429 and transport
        # failures actually appear. The tool_call_start event was appended
        # before this loop and is never re-emitted; `attempts` carries the
        # multiplicity. Approval-gated calls never retry (no approval
        # re-entry), and the abort event is checked between attempts.
        retry_deadline = (
          tool_t0 + effective_tool_timeout * (1 + _MAX_TOOL_DISPATCH_RETRIES)
          if effective_tool_timeout is not None
          else None
        )
        while True:
          result = None
          error = None
          dispatch_coro = self._dispatcher.dispatch(
            tool_id,
            tool_name,
            tool_input,
            **dispatch_kwargs,
          )
          if effective_tool_timeout is not None and not skip_timeout:
            try:
              result, error = await asyncio_module.wait_for(dispatch_coro, timeout=effective_tool_timeout)
            except timeout_error_type:
              elapsed = time_module.time() - tool_t0
              logger.error(
                "[%s] Tool %s timed out after %.1fs (limit %.0fs)",
                self._sid,
                tool_name,
                elapsed,
                effective_tool_timeout,
              )
              error = {
                "code": "tool_timeout",
                "sub_code": "timeout",
                "message": f"Tool '{tool_name}' timed out after {effective_tool_timeout:.0f}s. The tool call was cancelled. You may retry or skip this tool.",
              }
          else:
            result, error = await dispatch_coro

          attempt_outcome = classify_tool_outcome(dispatch_entry, result, error)
          abort_event = getattr(self, "_tool_abort_event", None)
          aborted = bool(abort_event is not None and abort_event.is_set())
          wall_clock_exhausted = bool(
            retry_deadline is not None and time_module.time() >= retry_deadline
          )
          if (
            retry_decision(
              dispatch_entry,
              attempt_outcome,
              dispatch_attempts,
              needs_approval=bool(needs_approval),
              aborted=aborted,
              wall_clock_exhausted=wall_clock_exhausted,
            )
            != "retry"
          ):
            dispatch_retries_exhausted = (
              dispatch_attempts > 1 and attempt_outcome in _RETRYABLE_DISPATCH_OUTCOMES
            )
            break
          logger.warning(
            "[%s] Tool %s dispatch %s on attempt %d; retrying",
            self._sid,
            tool_name,
            attempt_outcome,
            dispatch_attempts,
          )
          await asyncio_module.sleep(retry_backoff_seconds(dispatch_attempts))
          dispatch_attempts += 1

      if error is None:
        result, child_evidence = _canonical_agent_result_payload(result)

      # Strip private control fields from result before logging, event capture, and
      # model-bound tool_result content. _load_servers is a control signal -- capture
      # it for _refresh_tools (called after finally), then remove from result.
      if error is None and isinstance(result, dict):
        popped_background_result_ack = result.pop(
          _BACKGROUND_RESULT_ACK_RESULT_KEY,
          None,
        )
        requested_background_task_id = tool_input.get("task_id")
        if (
          tool_name == "get_background_result"
          and isinstance(requested_background_task_id, str)
          and requested_background_task_id.strip() != "*"
          and isinstance(popped_background_result_ack, dict)
          and popped_background_result_ack.get("task_id")
          == requested_background_task_id.strip()
          and isinstance(
            popped_background_result_ack.get("notification_generation"),
            int,
          )
          and not isinstance(
            popped_background_result_ack.get("notification_generation"),
            bool,
          )
        ):
          background_result_ack = (
            requested_background_task_id.strip(),
            popped_background_result_ack["notification_generation"],
          )
        popped_snapshot = result.pop(_READABLE_RESOURCE_SNAPSHOT_RESULT_KEY, None)
        if isinstance(popped_snapshot, dict):
          readable_resource_snapshot = popped_snapshot
        popped_workflow_evidence = result.pop(
          _WORKFLOW_EVIDENCE_PROJECTION_RESULT_KEY,
          None,
        )
        if isinstance(popped_workflow_evidence, dict):
          # The private key is popped before the tool-result hooks run, so the
          # citation hook can never see the projection on the result itself.
          # Hand it forward on the context instead, so the parent registry can
          # be seeded from what the children actually read (D-B4-3).
          child_evidence = popped_workflow_evidence
        popped = result.pop("_load_servers", None)
        if isinstance(popped, list):
          load_servers_signal = [str(server_name) for server_name in popped if server_name]
        popped_local = result.pop("_load_local_tools", None)
        if isinstance(popped_local, list):
          load_local_tools_signal = [str(tool_name) for tool_name in popped_local if tool_name]
        report_doors_key = _runner_attr(
          self,
          "_ACTIVE_SKILL_REPORT_DOORS_RESULT_KEY",
          _ACTIVE_SKILL_REPORT_DOORS_RESULT_KEY,
        )
        skill_allow_key = _runner_attr(
          self,
          "_ACTIVE_SKILL_ALLOW_RESULT_KEY",
          _ACTIVE_SKILL_ALLOW_RESULT_KEY,
        )
        skill_deny_key = _runner_attr(self, "_ACTIVE_SKILL_DENY_RESULT_KEY", _ACTIVE_SKILL_DENY_RESULT_KEY)
        self._activate_skill_report_doors(result.pop(report_doors_key, None))
        self._activate_skill_allow(result.pop(skill_allow_key, None), base_kwargs)
        self._activate_skill_deny(result.pop(skill_deny_key, None), base_kwargs)

      tool_elapsed = time_module.time() - tool_t0
      if error is None:
        semantic_error = _runner_attr(
          self,
          "classify_semantic_tool_error",
          classify_semantic_tool_error,
        )(result)
      if error is None and semantic_error is None:
        superseded_continuation_run_id = accepted_workflow_continuation_run_id(
          tool_name,
          result,
        )
        try:
          workflow_output_attachment = completed_workflow_output_attachment(
            tool_name,
            result,
          )
        except WorkflowOutputAttachmentError as exc:
          result = None
          error = {
            "code": "workflow_output_attachment_invalid",
            "message": str(exc),
          }
      if semantic_error is None and _is_accepted_ui_blocks_result(tool_name, result, error):
        setattr(self, "_stop_after_tool_results_reason", "accepted_ui_blocks")
        setattr(self, "_stop_after_tool_results_tool_name", tool_name)
      log_result = sanitize_boundary_value(
        result,
        sink="tool_log",
        boundary=getattr(self, "_secret_boundary", None),
      )
      log_error = sanitize_boundary_value(
        error if error is not None else semantic_error,
        sink="tool_log",
        boundary=getattr(self, "_secret_boundary", None),
      )
      result_json = json_module.dumps(log_result, default=str) if result is not None else ""
      result_bytes = len(result_json)
      result_preview = result_json[:150] if result_json else "null"
      if error or semantic_error:
        error_detail = error if error is not None else semantic_error
        logger.warning(
          "[%s] Tool %s error (%.1fs): %s",
          self._sid,
          tool_name,
          tool_elapsed,
          log_error,
          extra={
            "data": {
              "event": "tool_done",
              "session_id": self._sid,
              "tool": tool_name,
              "elapsed_s": round(tool_elapsed, 1),
              "server": server,
              "error": True,
              "semantic_error": semantic_error is not None and error is None,
              "error_detail": str(log_error)[:200],
              "error_sub_code": log_error.get("sub_code", "") if isinstance(log_error, dict) else "",
            }
          },
        )
      else:
        logger.info(
          "[%s] Tool %s done (%.1fs) | result=%s",
          self._sid,
          tool_name,
          tool_elapsed,
          result_preview,
          extra={
            "data": {
              "event": "tool_done",
              "session_id": self._sid,
              "tool": tool_name,
              "elapsed_s": round(tool_elapsed, 1),
              "server": server,
              "result_bytes": result_bytes,
              "error": False,
            }
          },
        )
    except cancelled_error_type as exc:
      cancelled_exc = exc
      error = {"code": "cancelled", "message": "Task was cancelled"}
    except Exception as exc:
      safe_exc = sanitize_boundary_value(
        str(exc),
        sink="tool_log",
        boundary=getattr(self, "_secret_boundary", None),
      )
      logger.error("[%s] Tool %s unhandled error: %s", self._sid, tool_name, safe_exc)
      error = {"code": "internal_error", "message": str(exc)}
    finally:
      duration_ms = int((time_module.time() - tool_t0) * 1000)
      if isinstance(error, dict):
        error = _error_with_model_error_data(error)
      # Every exit path funnels here — normal, semantic error, exclusion,
      # timeout, cancellation, unhandled exception — so the dispatch record
      # settles unconditionally, and it is on the event before the cancelled
      # arm's early append below.
      dispatch_record = build_dispatch_record(
        entry=dispatch_entry,
        result=result,
        error=error,
        semantic_error=semantic_error,
        attempts=dispatch_attempts,
        retries_exhausted=dispatch_retries_exhausted,
      )
      tool_complete_event = _runner_attr(
        self,
        "_build_tool_call_complete_event",
        _build_tool_call_complete_event,
      )(
        tool_call_id=tool_id,
        tool_name=tool_name,
        result=result,
        error=error,
        duration_ms=duration_ms,
        server=server,
        dispatch=dispatch_record,
        semantic_error=semantic_error,
      )
      if error is None:
        self._clear_active_skill_if_report_door_completed(tool_complete_event, base_kwargs)
      self._call_on_tool_timing(
        tool_name=tool_name,
        server=server,
        duration_ms=duration_ms,
        is_error=tool_complete_event["is_error"],
        result_bytes=result_bytes,
        tool_call_id=tool_id,
        request_id=self._request_id,
      )

    if cancelled_exc is not None:
      result_entry = self._make_error_result(
        tool_id,
        str(error.get("code", "tool_error")) if isinstance(error, dict) else "tool_error",
        str(error.get("message", "Tool failed")) if isinstance(error, dict) else "Tool failed",
        sub_code=str(error.get("sub_code", "")) if isinstance(error, dict) else "",
      )
      tool_complete_event["final_tool_result_blocks"] = [dict(result_entry)]
      await self._append_durable_event(tool_complete_event)
      self._append(tool_complete_event)
      raise cancelled_exc

    if load_servers_signal:
      self._refresh_tools(base_kwargs, load_servers_signal)
      logger.info(
        "[%s] Loaded MCP servers: %s | total tools now: %d",
        self._sid,
        load_servers_signal,
        len(base_kwargs.get("tools") or []),
      )
    if load_local_tools_signal:
      self._rebuild_filtered_tool_definitions(base_kwargs)
      logger.info(
        "[%s] Loaded local tools: %s | total tools now: %d",
        self._sid,
        load_local_tools_signal,
        len(base_kwargs.get("tools") or []),
      )

    model_result = result
    if error is None:
      model_result = self._annotate_result(result, tool_name=tool_name)

    if error is not None:
      error_code = str(error.get("code", "tool_error"))
      if error_code == "approval_timeout":
        # An expired approval is not an answer. Stop after this tool result
        # instead of letting the model retry the same stale approval.
        setattr(self, "_stop_after_tool_results_reason", "approval_timeout")
        setattr(self, "_stop_after_tool_results_tool_name", tool_name)
      result_entry = self._make_error_result(
        tool_id,
        error_code,
        str(error.get("message", "Tool failed")),
        sub_code=str(error.get("sub_code", "")),
        data=_model_error_data(error),
      )
    else:
      result_entry = {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": json_module.dumps(model_result, default=str),
      }
      if semantic_error is not None:
        result_entry["is_error"] = True

    extra_blocks = await self._call_on_tool_result(
      _runner_attr(self, "ToolResultContext", ToolResultContext)(
        tool_name=tool_name,
        tool_input=dict(tool_input),
        result=result,
        error=error,
        duration_ms=duration_ms,
        tool_call_id=tool_id,
        session_id=self._full_session_id,
        server=server,
        result_entry=result_entry,
        provider_id=provider_id,
        skill_run_id=self._skill_run_id,
        workspace_dir=self._workspace_dir,
        batch_id=getattr(self, "_batch_id", None),
        child_evidence=child_evidence,
        dispatch=dispatch_record,
        boundary_sanitizer=lambda value, sink: sanitize_boundary_value(
          value,
          sink=sink,
          boundary=getattr(self, "_secret_boundary", None),
        ),
      )
    )
    result_entry = sanitize_boundary_value(
      result_entry,
      sink="model_tool_result",
      boundary=getattr(self, "_secret_boundary", None),
    )
    if not isinstance(result_entry, dict):
      result_entry = {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": "<secret-sanitization-failed>",
        "is_error": True,
      }
    sanitized_extra_blocks = sanitize_boundary_value(
      extra_blocks,
      sink="model_tool_result",
      boundary=getattr(self, "_secret_boundary", None),
    )
    extra_blocks = (
      [dict(block) for block in sanitized_extra_blocks if isinstance(block, dict)]
      if isinstance(sanitized_extra_blocks, list)
      else []
    )
    live_entry, durable_entry = self._compact_model_tool_result_entry(result_entry, tool_name=tool_name)
    final_tool_result_blocks = [dict(durable_entry)]
    final_tool_result_blocks.extend(dict(block) for block in extra_blocks)
    tool_complete_event["final_tool_result_blocks"] = final_tool_result_blocks
    tool_complete_event = sanitize_tool_event(
      tool_complete_event,
      sink="tool_complete",
      boundary=getattr(self, "_secret_boundary", None),
    )
    await self._append_durable_event(tool_complete_event)
    self._append(tool_complete_event)
    if superseded_continuation_run_id is not None:
      # The continuation is durably accepted (tool_call_complete appended
      # above), so any staged prior-revision attachment for this run is stale
      # and must never reach a later final assistant turn (PN-E2E-03).
      self._pending_workflow_output_attachments.pop(
        superseded_continuation_run_id,
        None,
      )
    if workflow_output_attachment is not None:
      record_workflow_output_attachment(
        self._pending_workflow_output_attachments,
        workflow_output_attachment,
      )
    if background_result_ack is not None:
      self._pending_background_result_acks[tool_id] = (
        background_result_ack
      )
    if error is None:
      # Stored-artifact readbacks surface to the pane as artifact_ready
      # (origin "readback") right behind their tool_call_complete.
      readback_event = readback_artifact_ready_event(tool_name, result, tool_id)
      if readback_event is not None:
        await self._append_durable_event(readback_event)
        self._append(readback_event)
    if error is None and readable_resource_snapshot is not None:
      resource_event = _readable_resource_event_from_snapshot(
        self,
        readable_resource_snapshot,
        tool_call_id=tool_id,
        tool_name=tool_name,
        timestamp=time_module.time(),
      )
      if resource_event is not None:
        await self._append_durable_event(resource_event)
        self._append(resource_event)
    return live_entry, tool_name, extra_blocks
