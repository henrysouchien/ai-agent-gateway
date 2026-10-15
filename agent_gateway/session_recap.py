from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from collections.abc import Iterable
from typing import Any, Protocol

from .event_log import LogEntry, log_has_terminal
from .events import (
  Confidence,
  RecapApproval,
  RecapArtifact,
  RecapFailure,
  RecapTrigger,
  RecapVerdict,
  SessionRecapEvent,
  ToolCallsSummary,
  _approval_decision_source,
  _approval_outcome,
)
from .multi_user.billing import SessionUsageSummary


log = logging.getLogger("agent_gateway.session_recap")

#: Section name -> disclosure-layer group for ``context_chars.<group>``; a name
#: not listed here is ``other``. Measurement labels only.
CONTEXT_SECTION_GROUPS: dict[str, str] = {
  "identity_section": "identity",
  "persistent_memory": "memory_files",
  "memory_section": "memory_db",
  "memory_instructions": "doctrine",
  "citation_rules": "doctrine",
  "column_identification": "doctrine",
  "excel_tool_usage": "doctrine",
  "no_excel": "doctrine",
  "edgar_guidance": "doctrine",
  "methodology": "doctrine",
  "general_rules": "doctrine",
  "tool_validation": "doctrine",
  "active_mcp_tools": "mcp_sections",
  "deferred_mcp_tools": "mcp_sections",
  "tool_packs_section": "packs",
  "skills_catalog": "skills_preamble",
  "interactive_analyst_friction_capture": "friction",
}
CONTEXT_GROUPS = (
  "identity",
  "memory_files",
  "memory_db",
  "doctrine",
  "mcp_sections",
  "packs",
  "skills_preamble",
  "friction",
  "reminders",
  "other",
)
CONTEXT_COMPOSITION_FIELDS = (
  "context_chars.system_total",
  *(f"context_chars.{group}" for group in CONTEXT_GROUPS),
  "context_tools.advertised",
  "context_tools.invoked",
  "context_skills.invoked",
)


def context_chars_by_group(prompt_sections: Any) -> dict[str, int]:
  """Characters per ``CONTEXT_GROUPS`` entry over a prompt's section records
  (``name``/``text`` attributes); empty when the prompt carries none."""

  chars: dict[str, int] = {}
  for section in prompt_sections or ():
    group = CONTEXT_SECTION_GROUPS.get(str(getattr(section, "name", "")), "other")
    key = f"context_chars.{group}"
    chars[key] = chars.get(key, 0) + len(str(getattr(section, "text", "") or ""))
  return chars


def build_context_composition_event(
  *,
  turn: int,
  group_chars: dict[str, int],
  system_total: int,
  reminders: int,
  tools_advertised: int,
  tools_invoked: int,
  skills_invoked: int,
) -> dict[str, Any]:
  """One per provider request; flat int fields named in ``CONTEXT_COMPOSITION_FIELDS``."""

  event: dict[str, Any] = {
    "type": "context_composition",
    "turn": turn,
    **group_chars,
    "context_chars.system_total": system_total,
    "context_chars.reminders": reminders,
    "context_tools.advertised": tools_advertised,
    "context_tools.invoked": tools_invoked,
    "context_skills.invoked": skills_invoked,
  }
  return event


#: Counts other owners report on events this recap folds, by event type. Each
#: replaced a refusal; the recap keeps the largest value per name in
#: ``tool_calls_summary.measurements`` beside the interceptor counts. None is a
#: bound.
RUN_MEASUREMENT_FIELDS: dict[str, tuple[str, ...]] = {
  # autonomous_runner.send_approval_decision
  "approval_decision_sent": ("decisions_sent", "frame_bytes"),
  # fork_task_registry.ForkTaskRegistry.submit
  "learning_fork_launched": (
    "in_flight",
    "day_invocations",
    "day_charged_microusd",
    "retained_bytes",
  ),
  # runner_fork_agents._prepare_fork_child
  "fork_measurements": ("turns", "suffix_tokens"),
  # runner_session_lifecycle._emit_top_level_skill_started (skill_lifecycle admission)
  "skill_run_started": ("lease_fence_bytes",),
  # runner_run_loop: one per provider request; sizes are the run's peak,
  # the ``invoked`` counts are cumulative so the peak is the total
  "context_composition": CONTEXT_COMPOSITION_FIELDS,
}
#: Calls to the model engine's ``model_*`` tools, counted from the tool calls.
MODEL_CALLS_MEASUREMENT = "model_calls"
#: Provider responses that stopped on the output-length cap
#: (``assistant_message.stop_reason == "max_tokens"``): a measurement of the
#: run, never its outcome.
MAX_TOKENS_STOPS_MEASUREMENT = "max_tokens_stops"


class _SessionRecapSource(Protocol):
  """Event-log surface consumed by recap computation."""

  @property
  def entries(self) -> list[LogEntry]: ...

  @property
  def next_seq(self) -> int: ...


class SessionRecapEventLog(_SessionRecapSource, Protocol):
  """Event-log surface consumed by recap and terminal emission."""

  @property
  def has_terminal(self) -> bool: ...

  def append(self, event: dict[str, Any]) -> object | None: ...

def compute_recap(
  event_log: _SessionRecapSource,
  *,
  session_id: str,
  started_at: float,
  trigger: RecapTrigger,
  pending_failure: RecapFailure | None = None,
  usage: SessionUsageSummary | None = None,
) -> SessionRecapEvent:
  return _compute_recap_entries(
    event_log.entries,
    empty_next_seq=event_log.next_seq,
    session_id=session_id,
    started_at=started_at,
    trigger=trigger,
    pending_failure=pending_failure,
    usage=usage,
  )


def compute_recap_from_events(
  events: list[dict[str, Any]],
  *,
  session_id: str,
  started_at: float,
  trigger: RecapTrigger,
  pending_failure: RecapFailure | None = None,
  usage: SessionUsageSummary | None = None,
) -> SessionRecapEvent:
  entries = [
    LogEntry(seq=index, timestamp=_event_ts(event, time.time()), event=dict(event))
    for index, event in enumerate(events, start=1)
  ]
  return _compute_recap_entries(
    entries,
    empty_next_seq=len(entries) + 1,
    session_id=session_id,
    started_at=started_at,
    trigger=trigger,
    pending_failure=pending_failure,
    usage=usage,
  )


def _compute_recap_entries(
  entries: list[LogEntry],
  *,
  empty_next_seq: int,
  session_id: str,
  started_at: float,
  trigger: RecapTrigger,
  pending_failure: RecapFailure | None,
  usage: SessionUsageSummary | None,
) -> SessionRecapEvent:
  first_seq = entries[0].seq if entries else empty_next_seq
  last_seq = entries[-1].seq if entries else empty_next_seq - 1

  artifacts: list[RecapArtifact] = []
  verdicts: list[RecapVerdict] = []
  approvals: list[RecapApproval] = []
  failures: list[RecapFailure] = []
  tool_calls_summary = _ToolCallAccumulator()

  for entry in entries:
    event = entry.event
    event_type = str(event.get("type") or "")
    tool_calls_summary.record_run_measurements(event_type, event)
    if event_type == "artifact_ready":
      artifacts.append(
        RecapArtifact(
          artifact_id=str(event.get("artifact_id") or ""),
          skill=str(event.get("skill") or ""),
          contract_name=str(event.get("contract_name") or ""),
          ticker=_optional_str(event.get("ticker")),
          artifact_path=str(event.get("artifact_path") or ""),
          emitted_at_seq=entry.seq,
          ts=_event_ts(event, entry.timestamp),
        )
      )
    elif event_type == "skill_result_captured":
      verdict = _verdict_summary_from_skill_result(event)
      if verdict is None:
        continue
      verdicts.append(
        RecapVerdict(
          skill_run_id=str(event.get("skill_run_id") or ""),
          skill=str(event.get("skill") or ""),
          ticker=str(event.get("ticker") or ""),
          verdict_token=verdict["verdict_token"],
          confidence=_confidence(verdict.get("confidence")),
          materiality_cushion=_optional_float(verdict.get("materiality_cushion")),
          one_line_summary=str(verdict.get("one_line_summary") or ""),
          emitted_at_seq=entry.seq,
          ts=_event_ts(event, entry.timestamp),
        )
      )
    elif event_type == "tool_approval_decided":
      approvals.append(
        RecapApproval(
          tool_call_id=str(event.get("tool_call_id") or ""),
          tool_name=str(event.get("tool_name") or ""),
          outcome=_approval_outcome(event.get("outcome")),
          decision_source=_approval_decision_source(event.get("decision_source")),
          allow_tool_type_applied=bool(event.get("allow_tool_type_applied", False)),
          emitted_at_seq=entry.seq,
          ts=_event_ts(event, entry.timestamp),
        )
      )
    elif event_type == "tool_call_start":
      tool_calls_summary.record_start(event)
    elif event_type == "tool_call_complete":
      tool_calls_summary.record_complete(event)
    elif event_type == "interceptor_measurement":
      tool_calls_summary.record_measurement(event)
    elif event_type == "error":
      failures.append(
        RecapFailure(
          failure_type="terminal_error",
          detail=str(event.get("error") or "stream failed"),
          emitted_at_seq=entry.seq,
          ts=_event_ts(event, entry.timestamp),
        )
      )
    elif event_type == "artifact_failed":
      failures.append(
        RecapFailure(
          failure_type="artifact_failed",
          detail=_failure_detail(event),
          emitted_at_seq=entry.seq,
          ts=_event_ts(event, entry.timestamp),
        )
      )
    elif event_type == "artifact_unavailable":
      failures.append(
        RecapFailure(
          failure_type="artifact_unavailable",
          detail=_failure_detail(event),
          emitted_at_seq=entry.seq,
          ts=_event_ts(event, entry.timestamp),
        )
      )
    elif event_type == "interrupted" and event.get("reason") == "soft_budget_abort":
      failures.append(
        RecapFailure(
          failure_type="soft_budget_abort",
          detail=_soft_budget_abort_detail(event),
          emitted_at_seq=entry.seq,
          ts=_event_ts(event, entry.timestamp),
        )
      )

  if pending_failure is not None:
    failures.append(pending_failure)
    last_seq = max(last_seq, pending_failure.emitted_at_seq)

  now = time.time()
  return SessionRecapEvent(
    session_id=session_id,
    seq_range=(first_seq, last_seq),
    started_at=float(started_at),
    ended_at=now,
    trigger=trigger,
    artifacts=artifacts,
    verdicts=verdicts,
    approvals=approvals,
    tool_calls_summary=tool_calls_summary.to_summary(),
    failures=failures,
    usage=usage,
    ts=now,
  )


def emit_recap_then_terminal(
  event_log: SessionRecapEventLog,
  terminal_event: dict[str, Any],
  *,
  session_id: str,
  started_at: float,
  emit_recap: bool = True,
  pending_failure: RecapFailure | None = None,
  usage: SessionUsageSummary | None = None,
  recap_payload: dict[str, Any] | None = None,
) -> None:
  if log_has_terminal(event_log):
    return

  terminal_payload = dict(terminal_event)
  if not emit_recap:
    event_log.append(terminal_payload)
    return

  try:
    event_log.append(recap_payload if recap_payload is not None else terminal_recap_payload(
      event_log, terminal_payload, session_id=session_id, started_at=started_at,
      pending_failure=pending_failure, usage=usage,
    ))
  except Exception:
    log.warning("session_recap compute failed; continuing to terminal", exc_info=True)

  event_log.append(terminal_payload)


def terminal_recap_payload(
  event_log: _SessionRecapSource,
  terminal_event: dict[str, Any],
  *,
  session_id: str,
  started_at: float,
  pending_failure: RecapFailure | None = None,
  usage: SessionUsageSummary | None = None,
  pending_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
  entries = event_log.entries
  next_seq = event_log.next_seq
  for event in pending_events or ():
    entries.append(LogEntry(seq=next_seq, timestamp=time.time(), event=event))
    next_seq += 1
  terminal_expected_seq = next_seq + 1
  failure = pending_failure or _terminal_failure(terminal_event, terminal_expected_seq)
  if failure is not None and failure.emitted_at_seq <= 0:
    failure = replace(failure, emitted_at_seq=terminal_expected_seq)
  return _recap_payload(_compute_recap_entries(
    entries, empty_next_seq=next_seq, session_id=session_id,
    started_at=started_at, trigger="turn_end",
    pending_failure=failure, usage=usage,
  ))


def _recap_payload(recap: SessionRecapEvent) -> dict[str, Any]:
  from .events import event_to_dict

  # ``SessionRecapEvent`` keeps domain tuples (``seq_range`` and the usage
  # summary's ``usage_event_ids``), while the autonomous event channel accepts
  # only exact JSON values. Project the whole recap to its wire representation
  # before EventLog delivery so an optional recap can never prevent the final
  # terminal EVENT + END.
  return json.loads(json.dumps(event_to_dict(recap)))


def _terminal_failure(event: dict[str, Any], emitted_at_seq: int) -> RecapFailure | None:
  if event.get("type") != "error":
    return None
  return RecapFailure(
    failure_type="terminal_error",
    detail=str(event.get("error") or "stream failed"),
    emitted_at_seq=emitted_at_seq,
    ts=_event_ts(event, time.time()),
  )


class _ToolCallAccumulator:
  def __init__(self) -> None:
    self.total_calls = 0
    self.successes = 0
    self.errors = 0
    self.by_tool_name: dict[str, int] = {}
    self.by_server: dict[str, int] = {}
    self.measurements: dict[str, int] = {}
    self._seen_starts: set[str] = set()

  def record_start(self, event: dict[str, Any]) -> None:
    tool_call_id = str(event.get("tool_call_id") or "")
    self.total_calls += 1
    if tool_call_id:
      self._seen_starts.add(tool_call_id)
    self._increment_buckets(event)

  def record_complete(self, event: dict[str, Any]) -> None:
    tool_call_id = str(event.get("tool_call_id") or "")
    if tool_call_id and tool_call_id not in self._seen_starts:
      self.total_calls += 1
      self._increment_buckets(event)
    if bool(event.get("is_error")):
      self.errors += 1
    else:
      self.successes += 1

  def record_measurement(self, event: dict[str, Any]) -> None:
    # Producer: tool_dispatcher_helpers.run_interceptors; largest value per name.
    for name, value in event["measurement"].items():
      self.measurements[name] = max(int(value), self.measurements.get(name, 0))

  def record_run_measurements(self, event_type: str, event: dict[str, Any]) -> None:
    if event_type == "assistant_message" and event.get("stop_reason") == "max_tokens":
      self.measurements[MAX_TOKENS_STOPS_MEASUREMENT] = (
        self.measurements.get(MAX_TOKENS_STOPS_MEASUREMENT, 0) + 1
      )
    for name in RUN_MEASUREMENT_FIELDS.get(event_type, ()):
      value = event.get(name)
      if value is not None:
        self.measurements[name] = max(int(value), self.measurements.get(name, 0))

  def to_summary(self) -> ToolCallsSummary:
    return ToolCallsSummary(
      total_calls=self.total_calls,
      successes=self.successes,
      errors=self.errors,
      by_tool_name=dict(self.by_tool_name),
      by_server=dict(self.by_server),
      measurements=dict(self.measurements),
    )

  def _increment_buckets(self, event: dict[str, Any]) -> None:
    tool_name = str(event.get("tool_name") or "unknown")
    server = _server_name(event)
    self.by_tool_name[tool_name] = self.by_tool_name.get(tool_name, 0) + 1
    self.by_server[server] = self.by_server.get(server, 0) + 1
    if _bare_tool_name(tool_name).startswith("model_"):
      self.measurements[MODEL_CALLS_MEASUREMENT] = (
        self.measurements.get(MODEL_CALLS_MEASUREMENT, 0) + 1
      )


def _server_name(event: dict[str, Any]) -> str:
  raw = event.get("server") or event.get("mcp_server") or event.get("server_name")
  if raw:
    return str(raw)
  tool_name = str(event.get("tool_name") or "")
  if tool_name.startswith("mcp__"):
    parts = tool_name.split("__", 2)
    if len(parts) >= 3 and parts[1]:
      return parts[1]
  return "local"


def _bare_tool_name(tool_name: str) -> str:
  if tool_name.startswith("mcp__"):
    return tool_name.split("__", 2)[-1]
  return tool_name


def fold_run_measurements(events: Iterable[dict[str, Any]]) -> dict[str, int]:
  """The ``RUN_MEASUREMENT_FIELDS`` peaks and the ``max_tokens_stops`` count over
  ``events``: the fold the recap runs, for an owner that adds counts from a log
  the recap did not see."""

  accumulator = _ToolCallAccumulator()
  for event in events:
    accumulator.record_run_measurements(str(event.get("type") or ""), event)
  return accumulator.measurements


def _event_ts(event: dict[str, Any], fallback: float) -> float:
  if event.get("ts") is None:
    return float(fallback)
  return float(event["ts"])


def _optional_str(value: Any) -> str | None:
  if value is None:
    return None
  return str(value)


def _optional_float(value: Any) -> float | None:
  if value is None:
    return None
  return float(value)


def _verdict_summary_from_skill_result(event: dict[str, Any]) -> dict[str, Any] | None:
  verdict_echo = event.get("verdict_echo")
  if not isinstance(verdict_echo, dict):
    verdict_echo = None
    fms_results = event.get("fms_results")
    if isinstance(fms_results, list):
      for result in reversed(fms_results):
        if isinstance(result, dict) and isinstance(result.get("verdict_echo"), dict):
          verdict_echo = result["verdict_echo"]
          break
  if not isinstance(verdict_echo, dict):
    return None

  token = verdict_echo.get("verdict_token") or verdict_echo.get("verdict")
  if not token:
    return None
  return {
    "verdict_token": str(token),
    "confidence": verdict_echo.get("confidence"),
    "materiality_cushion": verdict_echo.get("materiality_cushion"),
    "one_line_summary": (
      verdict_echo.get("one_line_summary")
      or verdict_echo.get("summary")
      or verdict_echo.get("message")
      or str(token)
    ),
  }


def _confidence(value: Any) -> Confidence | None:
  if value == "HIGH":
    return "HIGH"
  if value == "MEDIUM":
    return "MEDIUM"
  if value == "LOW":
    return "LOW"
  return None


def _failure_detail(event: dict[str, Any]) -> str:
  for key in ("error_detail", "error", "message", "reason", "detail"):
    value = event.get(key)
    if value:
      return str(value)
  return str(event.get("type") or "failure")


def _soft_budget_abort_detail(event: dict[str, Any]) -> str:
  return " ".join(
    f"{key}={event.get(key)}" for key in ("kind", "observed", "threshold", "grace_turns")
  )


__all__ = [
  "MAX_TOKENS_STOPS_MEASUREMENT",
  "MODEL_CALLS_MEASUREMENT",
  "RUN_MEASUREMENT_FIELDS",
  "SessionRecapEventLog",
  "compute_recap",
  "compute_recap_from_events",
  "emit_recap_then_terminal",
  "fold_run_measurements",
]
