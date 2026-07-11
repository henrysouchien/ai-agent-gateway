from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any

from .event_log import EventLog, LogEntry, log_has_terminal
from .events import (
  RecapApproval,
  RecapArtifact,
  RecapFailure,
  RecapTrigger,
  RecapVerdict,
  SessionRecapEvent,
  ToolCallsSummary,
)
from .multi_user.billing import SessionUsageSummary


log = logging.getLogger("agent_gateway.session_recap")


def compute_recap(
  event_log: EventLog,
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
    elif event_type == "budget_exceeded":
      failures.append(
        RecapFailure(
          failure_type="budget_exceeded",
          detail=_failure_detail(event),
          emitted_at_seq=entry.seq,
          ts=_event_ts(event, entry.timestamp),
        )
      )
    elif event_type == "max_turns_reached":
      failures.append(
        RecapFailure(
          failure_type="max_turns_reached",
          detail=_failure_detail(event),
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
  event_log: EventLog,
  terminal_event: dict[str, Any],
  *,
  session_id: str,
  started_at: float,
  emit_recap: bool = True,
  pending_failure: RecapFailure | None = None,
  usage: SessionUsageSummary | None = None,
) -> None:
  if log_has_terminal(event_log):
    return

  terminal_payload = dict(terminal_event)
  if not emit_recap:
    event_log.append(terminal_payload)
    return

  terminal_expected_seq = event_log.next_seq + 1
  resolved_failure = pending_failure or _terminal_failure(terminal_payload, terminal_expected_seq)
  if resolved_failure is not None and resolved_failure.emitted_at_seq <= 0:
    resolved_failure = replace(resolved_failure, emitted_at_seq=terminal_expected_seq)

  try:
    recap = compute_recap(
      event_log,
      session_id=session_id,
      started_at=started_at,
      trigger="turn_end",
      pending_failure=resolved_failure,
      usage=usage,
    )
    event_log.append(_recap_payload(recap))
  except Exception:
    log.warning("session_recap compute failed; continuing to terminal", exc_info=True)

  event_log.append(terminal_payload)


def _recap_payload(recap: SessionRecapEvent) -> dict[str, Any]:
  from .events import event_to_dict

  return event_to_dict(recap)


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
    if bool(event.get("is_error")) or event.get("error") is not None or bool(event.get("semantic_error")):
      self.errors += 1
    else:
      self.successes += 1

  def to_summary(self) -> ToolCallsSummary:
    return ToolCallsSummary(
      total_calls=self.total_calls,
      successes=self.successes,
      errors=self.errors,
      by_tool_name=dict(self.by_tool_name),
      by_server=dict(self.by_server),
    )

  def _increment_buckets(self, event: dict[str, Any]) -> None:
    tool_name = str(event.get("tool_name") or "unknown")
    server = _server_name(event)
    self.by_tool_name[tool_name] = self.by_tool_name.get(tool_name, 0) + 1
    self.by_server[server] = self.by_server.get(server, 0) + 1


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


def _confidence(value: Any) -> str | None:
  if value is None:
    return None
  if value in {"HIGH", "MEDIUM", "LOW"}:
    return str(value)
  return None


def _approval_outcome(value: Any) -> str:
  if value in {"approved", "denied", "timeout"}:
    return str(value)
  return "timeout"


def _approval_decision_source(value: Any) -> str:
  if value in {
    "user_approved",
    "delegated_auto_approved",
    "user_denied",
    "relay_policy_denied",
    "headless_auto_deny",
    "headless_hook_approved",
    "session_cache_approved",
    "approval_timeout",
  }:
    return str(value)
  return "approval_timeout"


def _failure_detail(event: dict[str, Any]) -> str:
  for key in ("error_detail", "error", "message", "reason", "detail"):
    value = event.get(key)
    if value:
      return str(value)
  return str(event.get("type") or "failure")


__all__ = ["compute_recap", "compute_recap_from_events", "emit_recap_then_terminal"]
