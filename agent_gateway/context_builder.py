from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from .agent_session_log import AgentSessionLog, LogEntry

Message = dict[str, Any]


def _estimate_tokens(value: Any) -> int:
  return max(1, len(json.dumps(value, default=str)) // 4)


@dataclass
class _InterruptedRunContext:
  interruption: LogEntry
  tool_interruptions: list[LogEntry]
  sub_agent_summaries: list[dict[str, Any]]
  covered_tail_seqs: set[int]


@dataclass
class SessionContextBuilder:
  """Build replay context from a durable session log."""

  agent_session_log: AgentSessionLog
  tail_window_seconds: int | None = 7200
  tail_token_budget: int = 20_000

  async def build(self) -> list[Message]:
    summaries, _ = await self.agent_session_log.query(
      event_types={"summary"},
      order="desc",
      limit=1,
    )
    latest_summary = summaries[0] if summaries else None

    state_updates, _ = await self.agent_session_log.query(
      event_types={"state_update"},
      order="desc",
      limit=1,
    )
    latest_state_update = state_updates[0] if state_updates else None
    interrupted_context = await self._latest_interrupted_run_context()

    if latest_summary is not None:
      covers = latest_summary.event.get("covers") or {}
      covers_to_seq = int(covers.get("to_seq", 0) or 0)
      tail_entries, _ = await self.agent_session_log.query(
        after_seq=covers_to_seq + 1,
        order="asc",
      )
    elif self.tail_window_seconds is None:
      tail_entries, _ = await self.agent_session_log.query(order="asc")
    else:
      since_ts = time.time() - self.tail_window_seconds
      tail_entries, _ = await self.agent_session_log.query(
        after_ts=since_ts,
        order="asc",
      )

    tail_entries = [
      entry
      for entry in tail_entries
      if entry.event.get("type") not in {"summary", "state_update"}
    ]
    if interrupted_context is not None:
      tail_entries = [
        entry for entry in tail_entries if entry.seq not in interrupted_context.covered_tail_seqs
      ]
    tail_entries = self._truncate_to_token_budget(tail_entries, self.tail_token_budget)

    messages: list[Message] = []
    if latest_summary is not None:
      messages.append(self._summary_to_message(latest_summary))
    if latest_state_update is not None:
      messages.append(self._state_update_to_message(latest_state_update))
    if interrupted_context is not None:
      messages.append(self._interrupted_run_to_message(interrupted_context, latest_state_update))
    messages.extend(self._entries_to_messages(tail_entries))
    return messages

  async def _latest_interrupted_run_context(self) -> _InterruptedRunContext | None:
    lifecycle_entries, _ = await self.agent_session_log.query(
      event_types={"attach", "detach", "interrupted"},
      role="writer",
      order="desc",
      limit=100,
    )
    if not lifecycle_entries:
      return None

    before_seq: int | None = None
    if lifecycle_entries[0].event.get("type") == "attach":
      before_seq = lifecycle_entries[0].seq - 1

    previous_lifecycle = [
      entry for entry in lifecycle_entries if before_seq is None or entry.seq <= before_seq
    ]
    if not previous_lifecycle:
      return None

    segment_entries: list[LogEntry] = []
    segment_start_seq = 0
    for entry in previous_lifecycle:
      segment_entries.append(entry)
      if entry.event.get("type") == "attach":
        segment_start_seq = entry.seq
        break

    interruption = next(
      (entry for entry in segment_entries if entry.event.get("type") == "interrupted"),
      None,
    )
    if interruption is None:
      return None

    window_end_seq = before_seq if before_seq is not None else await self.agent_session_log.latest_seq()
    window_start_after = segment_start_seq + 1
    tool_interruptions, _ = await self.agent_session_log.query(
      event_types={"tool_call_interrupted"},
      after_seq=window_start_after,
      before_seq=window_end_seq,
      order="asc",
    )
    window_entries, _ = await self.agent_session_log.query(
      after_seq=window_start_after,
      before_seq=window_end_seq,
      order="asc",
    )
    covered_tail_seqs = {interruption.seq}
    covered_tail_seqs.update(entry.seq for entry in tool_interruptions)
    return _InterruptedRunContext(
      interruption=interruption,
      tool_interruptions=tool_interruptions,
      sub_agent_summaries=self._summarize_sub_agent_work(window_entries),
      covered_tail_seqs=covered_tail_seqs,
    )

  def _truncate_to_token_budget(self, entries: list[LogEntry], budget: int) -> list[LogEntry]:
    if budget <= 0:
      return []

    selected: list[LogEntry] = []
    total_tokens = 0
    for entry in reversed(entries):
      entry_tokens = _estimate_tokens(entry.event)
      if selected and total_tokens + entry_tokens > budget:
        break
      selected.append(entry)
      total_tokens += entry_tokens
      if total_tokens >= budget:
        break
    selected.reverse()
    return selected

  def _summary_to_message(self, entry: LogEntry) -> Message:
    text = str(entry.event.get("text") or "").strip()
    if not text:
      text = "(summary unavailable)"
    return {
      "role": "user",
      "content": f"## Prior session summary\n\n{text}",
    }

  def _interrupted_run_to_message(
    self,
    context: _InterruptedRunContext,
    latest_state_update: LogEntry | None,
  ) -> Message:
    event = context.interruption.event
    reason = str(event.get("reason") or "unknown")
    runner_id = str(event.get("runner_id") or "unknown")
    last_completed_seq = event.get("last_completed_seq")

    sections = [
      "## Previous run interrupted",
      "",
      f"Reason: {reason}",
      f"Interrupted at: {self._format_timestamp(context.interruption.timestamp)}",
      f"Runner: {runner_id}",
    ]
    if last_completed_seq is not None:
      sections.append(f"Last completed event: seq {last_completed_seq}")
    recovered_by = event.get("recovered_by_runner_id")
    if recovered_by:
      sections.append(f"Recovered by runner: {recovered_by}")

    if context.tool_interruptions:
      sections.extend(["", "Interrupted tool calls:"])
      for entry in context.tool_interruptions[:5]:
        tool_event = entry.event
        tool_name = str(tool_event.get("tool_name") or "unknown_tool")
        tool_call_id = str(tool_event.get("tool_call_id") or "unknown_id")
        tool_risk = str(tool_event.get("tool_risk") or "side_effecting")
        started_at = tool_event.get("original_started_at")
        line = f"- `{tool_name}` (`{tool_call_id}`), risk {tool_risk}"
        if started_at is not None:
          line += f", started at {started_at}"
        sections.append(line)
      remaining = len(context.tool_interruptions) - 5
      if remaining > 0:
        sections.append(f"- ...and {remaining} more interrupted tool call(s)")

    if context.sub_agent_summaries:
      sections.extend(["", "Sub-agent work available from the interrupted run:"])
      for summary in context.sub_agent_summaries[:5]:
        sections.append(self._format_sub_agent_summary(summary))
      remaining = len(context.sub_agent_summaries) - 5
      if remaining > 0:
        sections.append(f"- ...and {remaining} more sub-agent(s)")

    state_lines = self._interrupted_state_lines(latest_state_update)
    if state_lines:
      sections.extend(["", "Latest state flags:"])
      sections.extend(state_lines)

    sections.extend(
      [
        "",
        (
          "No automatic retry has been performed. Treat this as interrupted context; "
          "verify freshness and external side effects before relying on it."
        ),
      ]
    )
    return {
      "role": "user",
      "content": "\n".join(sections),
    }

  def _state_update_to_message(self, entry: LogEntry) -> Message:
    payload = entry.event.get("payload", {})
    if not isinstance(payload, dict):
      payload = {}

    sections: list[str] = []
    alerts = payload.get("alerts")
    if isinstance(alerts, list) and alerts:
      sections.append("Active alerts:\n" + "\n".join(f"- {item}" for item in alerts))

    data_flags = payload.get("data_flags")
    if isinstance(data_flags, list) and data_flags:
      sections.append("Data flags:\n" + "\n".join(f"- {item}" for item in data_flags))

    active_servers = payload.get("active_servers")
    if isinstance(active_servers, list) and active_servers:
      sections.append("Active MCP servers: " + ", ".join(str(item) for item in active_servers))

    regime = payload.get("regime")
    if isinstance(regime, str) and regime.strip():
      sections.append(f"Regime: {regime.strip()}")

    next_session = payload.get("next_session")
    if isinstance(next_session, list) and next_session:
      sections.append("Next session:\n" + "\n".join(f"- {item}" for item in next_session if str(item).strip()))

    if payload.get("budget_exceeded"):
      sections.append("Note: previous run exceeded budget; review interrupted events for details.")

    error = payload.get("error")
    if isinstance(error, str) and error.strip():
      sections.append(f"Previous run error: {error.strip()}")

    body = "\n\n".join(section for section in sections if section).strip()
    if not body:
      body = "No structured state was recorded."
    return {
      "role": "user",
      "content": f"## Previous run state\n\n{body}",
    }

  def _summarize_sub_agent_work(self, entries: list[LogEntry]) -> list[dict[str, Any]]:
    by_sub_agent: dict[str, dict[str, Any]] = {}
    for entry in entries:
      event = entry.event
      sub_agent_id = event.get("sub_agent_id")
      if not sub_agent_id:
        continue
      sub_agent_key = str(sub_agent_id)
      summary = by_sub_agent.setdefault(
        sub_agent_key,
        {
          "sub_agent_id": sub_agent_key,
          "first_seq": entry.seq,
          "last_seq": entry.seq,
          "event_count": 0,
          "tool_call_count": 0,
          "interrupted_tool_count": 0,
          "task_id": None,
          "agent_name": None,
          "status": None,
        },
      )
      summary["event_count"] += 1
      summary["last_seq"] = entry.seq
      event_type = str(event.get("type") or "")
      if event_type == "tool_call_start":
        summary["tool_call_count"] += 1
      elif event_type == "tool_call_interrupted":
        summary["interrupted_tool_count"] += 1
        summary["status"] = "interrupted"
      elif event_type == "interrupted":
        summary["status"] = f"interrupted: {event.get('reason') or 'unknown'}"
      elif event_type == "detach" and summary.get("status") is None:
        summary["status"] = str(event.get("reason") or "detached")
      elif event_type == "task_registered":
        summary["task_id"] = event.get("task_id")
        summary["agent_name"] = event.get("agent_name")
      elif event_type == "task_completed":
        summary["task_id"] = event.get("task_id") or summary.get("task_id")
        summary["status"] = str(event.get("final_state") or "completed")

    return sorted(by_sub_agent.values(), key=lambda item: int(item["first_seq"]))

  def _format_sub_agent_summary(self, summary: dict[str, Any]) -> str:
    label = f"`{summary['sub_agent_id']}`"
    if summary.get("agent_name"):
      label = f"{summary['agent_name']} ({label})"
    fragments = [
      f"- {label}: {summary['event_count']} event(s)",
      f"{summary['tool_call_count']} tool call(s)",
      f"last seq {summary['last_seq']}",
    ]
    if summary.get("task_id"):
      fragments.append(f"task {summary['task_id']}")
    if summary.get("status"):
      fragments.append(f"status {summary['status']}")
    if summary.get("interrupted_tool_count"):
      fragments.append(f"{summary['interrupted_tool_count']} interrupted tool call(s)")
    return ", ".join(fragments)

  def _interrupted_state_lines(self, latest_state_update: LogEntry | None) -> list[str]:
    if latest_state_update is None:
      return []
    payload = latest_state_update.event.get("payload", {})
    if not isinstance(payload, dict):
      return []

    lines: list[str] = []
    if payload.get("budget_exceeded"):
      lines.append("- budget_exceeded: true")
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
      lines.append(f"- error: {error.strip()}")

    for label, key in (("data flag", "data_flags"), ("alert", "alerts")):
      values = payload.get(key)
      if not isinstance(values, list):
        continue
      for item in values[:5]:
        text = str(item).strip()
        if text:
          lines.append(f"- {label}: {text}")
      remaining = len(values) - 5
      if remaining > 0:
        lines.append(f"- ...and {remaining} more {label}(s)")
    return lines

  def _format_timestamp(self, timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))

  def _entries_to_messages(self, entries: list[LogEntry]) -> list[Message]:
    messages: list[Message] = []
    for entry in entries:
      event = entry.event
      event_type = str(event.get("type") or "")
      rendered = self._entry_to_message(event_type, event)
      if rendered is None:
        continue
      if isinstance(rendered, list):
        messages.extend(rendered)
      else:
        messages.append(rendered)
    return messages

  def _entry_to_message(self, event_type: str, event: dict[str, Any]) -> Message | list[Message] | None:
    if event_type == "user_message":
      return {
        "role": "user",
        "content": event.get("content", ""),
      }

    if event_type == "assistant_message":
      content_blocks = event.get("content_blocks")
      message: Message = {
        "role": "assistant",
        "content": content_blocks if isinstance(content_blocks, list) else event.get("content", ""),
      }
      if event.get("model"):
        message["model"] = event["model"]
      if event.get("stop_reason"):
        message["stop_reason"] = event["stop_reason"]
      return message

    if event_type == "tool_call_complete":
      tool_call_id = str(event.get("tool_call_id") or "")
      if not tool_call_id:
        return None
      error = event.get("error")
      if error is not None:
        content = json.dumps({"error": error}, default=str)
        block: dict[str, Any] = {
          "type": "tool_result",
          "tool_use_id": tool_call_id,
          "content": content,
          "is_error": True,
        }
      else:
        block = {
          "type": "tool_result",
          "tool_use_id": tool_call_id,
          "content": json.dumps(event.get("result"), default=str),
        }
        if event.get("is_error") is True:
          block["is_error"] = True
      return {
        "role": "user",
        "content": [block],
      }

    if event_type == "tool_call_interrupted":
      tool_name = str(event.get("tool_name") or "unknown_tool")
      tool_risk = str(event.get("tool_risk") or "side_effecting")
      started_at = event.get("original_started_at")
      fragments = [
        f"Previous run interrupted tool `{tool_name}`.",
        f"Risk: {tool_risk}.",
      ]
      if started_at is not None:
        fragments.insert(1, f"Original start time: {started_at}.")
      return {
        "role": "user",
        "content": "[Session log] " + " ".join(fragments),
      }

    if event_type == "interrupted":
      reason = str(event.get("reason") or "unknown")
      return {
        "role": "user",
        "content": f"[Session log] Previous run ended with interruption reason: {reason}.",
      }

    return None


__all__ = ["Message", "SessionContextBuilder"]
