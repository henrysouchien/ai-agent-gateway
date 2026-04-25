from __future__ import annotations

import json
from typing import Any

from .agent_session_log import AgentSessionLog, LogEntry
from .context_builder import Message
from .task_registry import ParentMessage

ToolResultBlock = dict[str, Any]

_INTERRUPTED_TOOL_RESULT_CONTENT = {
  "status": "interrupted",
  "note": "This tool call did not complete before the sub-agent was interrupted. Verify before retrying.",
}


async def _task_registration(log: AgentSessionLog, task_id: str) -> dict[str, Any] | None:
  entries, _ = await log.query(event_types={"task_registered"}, order="asc")
  for entry in entries:
    if entry.event.get("task_id") == task_id:
      return dict(entry.event)
  return None


def _event_ts(entry: LogEntry) -> float:
  raw = entry.event.get("timestamp", entry.timestamp)
  try:
    return float(raw)
  except (TypeError, ValueError):
    return float(entry.timestamp)


def _message_from_user_event(event: dict[str, Any]) -> Message:
  return {
    "role": "user",
    "content": event.get("content", ""),
  }


def _message_from_assistant_event(event: dict[str, Any]) -> Message:
  content_blocks = event.get("content_blocks")
  message: Message = {
    "role": "assistant",
    "content": content_blocks if isinstance(content_blocks, list) else event.get("content", ""),
  }
  if event.get("model"):
    message["model"] = event["model"]
  if event.get("stop_reason"):
    message["stop_reason"] = event["stop_reason"]
  if event.get("provider"):
    message["provider"] = event["provider"]
  return message


def _tool_result_blocks_from_event(event: dict[str, Any]) -> list[ToolResultBlock]:
  final_blocks = event.get("final_tool_result_blocks")
  if isinstance(final_blocks, list):
    return [dict(block) for block in final_blocks if isinstance(block, dict)]

  tool_call_id = str(event.get("tool_call_id") or "")
  if not tool_call_id:
    return []
  error = event.get("error")
  if error is not None:
    return [
      {
        "type": "tool_result",
        "tool_use_id": tool_call_id,
        "content": json.dumps({"error": error}, default=str),
        "is_error": True,
      }
    ]
  return [
    {
      "type": "tool_result",
      "tool_use_id": tool_call_id,
      "content": json.dumps(event.get("result"), default=str),
    }
  ]


async def reconstruct_messages_for_task(log: AgentSessionLog, task_id: str) -> list[Message]:
  """Rebuild the model-facing transcript for a background sub-agent task."""
  registration = await _task_registration(log, task_id)
  if registration is None:
    return []
  sub_agent_id = registration.get("sub_agent_id")
  if not sub_agent_id:
    metadata = registration.get("metadata")
    if isinstance(metadata, dict):
      sub_agent_id = metadata.get("sub_agent_id")
  if not sub_agent_id:
    return []

  entries, _ = await log.query(
    event_types={"user_message", "assistant_message", "tool_call_complete"},
    sub_agent_id=str(sub_agent_id),
    order="asc",
  )
  messages: list[Message] = []
  pending_tool_results: list[ToolResultBlock] = []

  def flush_tool_results() -> None:
    nonlocal pending_tool_results
    if pending_tool_results:
      messages.append({"role": "user", "content": pending_tool_results})
      pending_tool_results = []

  for entry in entries:
    event = entry.event
    event_type = str(event.get("type") or "")
    if event_type == "tool_call_complete":
      pending_tool_results.extend(_tool_result_blocks_from_event(event))
      continue
    flush_tool_results()
    if event_type == "user_message":
      messages.append(_message_from_user_event(event))
    elif event_type == "assistant_message":
      messages.append(_message_from_assistant_event(event))
  flush_tool_results()
  return messages


def _content_blocks(message: Message) -> list[dict[str, Any]]:
  content = message.get("content")
  if isinstance(content, list):
    return [block for block in content if isinstance(block, dict)]
  return []


def _tool_use_ids(message: Message) -> list[str]:
  if message.get("role") != "assistant":
    return []
  ids: list[str] = []
  for block in _content_blocks(message):
    if block.get("type") == "tool_use" and block.get("id"):
      ids.append(str(block["id"]))
  return ids


def _tool_result_ids(message: Message) -> set[str]:
  if message.get("role") != "user":
    return set()
  return {
    str(block["tool_use_id"])
    for block in _content_blocks(message)
    if block.get("type") == "tool_result" and block.get("tool_use_id")
  }


def detect_orphan_tool_uses(messages: list[Message]) -> list[str]:
  """Find trailing assistant tool_use IDs lacking a following tool_result."""
  for index in range(len(messages) - 1, -1, -1):
    ids = _tool_use_ids(messages[index])
    if not ids:
      continue
    next_results = _tool_result_ids(messages[index + 1]) if index + 1 < len(messages) else set()
    missing = [tool_use_id for tool_use_id in ids if tool_use_id not in next_results]
    return missing if index >= len(messages) - 2 else []
  return []


def build_synthetic_tool_results(orphan_ids: list[str]) -> list[ToolResultBlock]:
  return [
    {
      "type": "tool_result",
      "tool_use_id": str(tool_use_id),
      "content": json.dumps(_INTERRUPTED_TOOL_RESULT_CONTENT),
      "is_error": True,
    }
    for tool_use_id in orphan_ids
  ]


async def reconstruct_parent_messages(log: AgentSessionLog, task_id: str, before_ts: float) -> list[ParentMessage]:
  entries, _ = await log.query(
    event_types={"parent_message_sent"},
    order="asc",
  )
  parent_messages: list[ParentMessage] = []
  for entry in entries:
    event = entry.event
    if event.get("task_id") != task_id:
      continue
    message_id = str(event.get("message_id") or "")
    if not message_id:
      continue
    sent_at = float(event.get("sent_at") or _event_ts(entry))
    if sent_at >= before_ts:
      continue
    parent_messages.append(
      ParentMessage(
        message_id=message_id,
        text=str(event.get("message") or ""),
        sent_at=sent_at,
      )
    )
  return parent_messages


def _parent_context_message(parent_messages: list[ParentMessage], additional_context: str | None) -> Message | None:
  lines = [
    f"[Parent message id={message.message_id}]: {message.text}"
    for message in parent_messages
  ]
  if additional_context is not None and additional_context.strip():
    lines.append(f"[Operator continuation note]: {additional_context.strip()}")
  if not lines:
    return None
  return {"role": "user", "content": "\n".join(lines)}


def place_resume_messages(
  transcript: list[Message],
  synthetic_results: list[ToolResultBlock],
  parent_messages: list[ParentMessage],
  additional_context: str | None,
) -> list[Message]:
  messages = [dict(message) for message in transcript]
  if synthetic_results:
    synthetic = [dict(block) for block in synthetic_results]
    if messages and messages[-1].get("role") == "user" and isinstance(messages[-1].get("content"), list):
      existing = [dict(block) if isinstance(block, dict) else block for block in messages[-1]["content"]]
      messages[-1] = {**messages[-1], "content": synthetic + existing}
    else:
      messages.append({"role": "user", "content": synthetic})

  context_message = _parent_context_message(parent_messages, additional_context)
  if context_message is not None:
    messages.append(context_message)
  return messages


__all__ = [
  "ToolResultBlock",
  "build_synthetic_tool_results",
  "detect_orphan_tool_uses",
  "place_resume_messages",
  "reconstruct_messages_for_task",
  "reconstruct_parent_messages",
]
