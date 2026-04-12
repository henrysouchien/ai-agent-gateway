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
class SessionContextBuilder:
  """Build replay context from a durable session log."""

  agent_session_log: AgentSessionLog
  tail_window_seconds: int = 7200
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

    if latest_summary is not None:
      covers = latest_summary.event.get("covers") or {}
      covers_to_seq = int(covers.get("to_seq", 0) or 0)
      tail_entries, _ = await self.agent_session_log.query(
        after_seq=covers_to_seq + 1,
        order="asc",
      )
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
    tail_entries = self._truncate_to_token_budget(tail_entries, self.tail_token_budget)

    messages: list[Message] = []
    if latest_summary is not None:
      messages.append(self._summary_to_message(latest_summary))
    if latest_state_update is not None:
      messages.append(self._state_update_to_message(latest_state_update))
    messages.extend(self._entries_to_messages(tail_entries))
    return messages

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
