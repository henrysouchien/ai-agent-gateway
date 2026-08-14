from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .agent_session_log import AgentSessionLog, LogEntry
from .context_builder import Message
from .task_registry import ParentMessage, format_parent_messages_for_model
from .secret_boundary import sanitize_tool_event

ToolResultBlock = dict[str, Any]

_INTERRUPTED_TOOL_RESULT_CONTENT = {
  "status": "interrupted",
  "note": "This tool call did not complete before the sub-agent was interrupted. Verify before retrying.",
}


class TranscriptIntegrityError(RuntimeError):
  """Durable child transcript facts cannot be joined without ambiguity."""


@dataclass(frozen=True)
class ChildRunSegment:
  """Exact durable child-run evidence associated with one task registration."""

  task_id: str
  original_task_id: str | None
  registration: LogEntry
  completion: LogEntry | None
  runner_id: str | None
  entries: tuple[LogEntry, ...]


def _sub_agent_id(event: dict[str, Any]) -> str | None:
  raw_sub_agent_id = event.get("sub_agent_id")
  if not raw_sub_agent_id:
    metadata = event.get("metadata")
    if isinstance(metadata, dict):
      raw_sub_agent_id = metadata.get("sub_agent_id")
  if not isinstance(raw_sub_agent_id, str) or not raw_sub_agent_id:
    return None
  return raw_sub_agent_id


def _original_task_id(event: dict[str, Any]) -> str | None:
  raw_original_task_id = event.get("original_task_id")
  if raw_original_task_id is None:
    metadata = event.get("metadata")
    if isinstance(metadata, dict):
      raw_original_task_id = metadata.get("original_task_id")
  if not isinstance(raw_original_task_id, str) or not raw_original_task_id:
    return None
  return raw_original_task_id


async def _task_registration(log: AgentSessionLog, task_id: str) -> LogEntry | None:
  entries, _ = await log.query(event_types={"task_registered"}, order="asc")
  matches = [entry for entry in entries if entry.event.get("task_id") == task_id]
  return matches[0] if len(matches) == 1 else None


async def _task_completion(
  log: AgentSessionLog,
  task_id: str,
  *,
  after_seq: int,
) -> LogEntry | None:
  entries, _ = await log.query(
    event_types={"task_completed"},
    after_seq=after_seq,
    order="asc",
  )
  return next((entry for entry in entries if entry.event.get("task_id") == task_id), None)


async def _next_sub_agent_registration(
  log: AgentSessionLog,
  sub_agent_id: str,
  *,
  after_seq: int,
) -> LogEntry | None:
  entries, _ = await log.query(
    event_types={"task_registered"},
    after_seq=after_seq,
    order="asc",
  )
  return next((entry for entry in entries if _sub_agent_id(entry.event) == sub_agent_id), None)


async def child_run_segment_for_task(
  log: AgentSessionLog,
  task_id: str,
) -> ChildRunSegment | None:
  """Resolve one task to its exact durable child runner and sequence window."""
  registration = await _task_registration(log, task_id)
  if registration is None:
    return None

  event = registration.event
  original_task_id = _original_task_id(event)
  sub_agent_id = _sub_agent_id(event)
  completion = await _task_completion(
    log,
    task_id,
    after_seq=registration.seq + 1,
  )

  before_seq: int | None = completion.seq if completion is not None else None
  if completion is None and sub_agent_id is not None:
    next_registration = await _next_sub_agent_registration(
      log,
      sub_agent_id,
      after_seq=registration.seq + 1,
    )
    if next_registration is not None:
      before_seq = next_registration.seq - 1

  if sub_agent_id is None:
    return ChildRunSegment(
      task_id=task_id,
      original_task_id=original_task_id,
      registration=registration,
      completion=completion,
      runner_id=None,
      entries=(),
    )

  attach_entries, _ = await log.query(
    event_types={"attach"},
    sub_agent_id=sub_agent_id,
    role="sub_agent",
    after_seq=registration.seq + 1,
    before_seq=before_seq,
    order="asc",
  )
  attach = attach_entries[0] if attach_entries else None
  raw_runner_id = attach.event.get("runner_id") if attach is not None else None
  runner_id = raw_runner_id if isinstance(raw_runner_id, str) and raw_runner_id else None
  if runner_id is None:
    return ChildRunSegment(
      task_id=task_id,
      original_task_id=original_task_id,
      registration=registration,
      completion=completion,
      runner_id=None,
      entries=(),
    )

  entries, _ = await log.query(
    runner_id=runner_id,
    after_seq=registration.seq + 1,
    before_seq=before_seq,
    order="asc",
  )
  return ChildRunSegment(
    task_id=task_id,
    original_task_id=original_task_id,
    registration=registration,
    completion=completion,
    runner_id=runner_id,
    entries=tuple(entries),
  )


async def reconstruct_child_run_lineage(
  log: AgentSessionLog,
  task_id: str,
) -> list[ChildRunSegment]:
  """Return exact child-run segments from the root task to ``task_id``."""
  reverse_lineage: list[ChildRunSegment] = []
  seen_task_ids: set[str] = set()
  current_task_id = task_id
  while current_task_id:
    if current_task_id in seen_task_ids:
      return []
    seen_task_ids.add(current_task_id)
    segment = await child_run_segment_for_task(log, current_task_id)
    if segment is None:
      return []
    reverse_lineage.append(segment)
    current_task_id = segment.original_task_id or ""
  reverse_lineage.reverse()
  return reverse_lineage


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


def _messages_from_runtime_guard_event(event: dict[str, Any]) -> list[Message]:
  if event.get("guard") != "final_answer":
    return []
  messages: list[Message] = []
  draft_content_blocks = event.get("draft_content_blocks")
  if isinstance(draft_content_blocks, list):
    draft_message: Message = {
      "role": "assistant",
      "content": draft_content_blocks,
    }
    if event.get("draft_model"):
      draft_message["model"] = event["draft_model"]
    if event.get("draft_stop_reason"):
      draft_message["stop_reason"] = event["draft_stop_reason"]
    if event.get("draft_provider"):
      draft_message["provider"] = event["draft_provider"]
    messages.append(draft_message)
  guard_message = event.get("message")
  if isinstance(guard_message, str) and guard_message:
    messages.append({"role": "user", "content": guard_message})
  return messages


def _tool_result_blocks_from_event(event: dict[str, Any]) -> list[ToolResultBlock]:
  event = sanitize_tool_event(event, sink="child_replay")
  final_blocks = event.get("final_tool_result_blocks")
  if isinstance(final_blocks, list):
    return [dict(block) for block in final_blocks if isinstance(block, dict) and not block.get("_event_only")]

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
  block = {
    "type": "tool_result",
    "tool_use_id": tool_call_id,
    "content": json.dumps(event.get("result"), default=str),
  }
  if event.get("is_error") is True:
    block["is_error"] = True
  return [block]


async def _consumed_parent_message_projection(
  log: AgentSessionLog,
  lineage: list[ChildRunSegment],
) -> tuple[dict[int, list[ParentMessage]], set[tuple[str, str]]]:
  """Join consumed acknowledgements to sent bodies and child assistants."""

  if not lineage:
    return {}, set()
  lineage_task_ids = {segment.task_id for segment in lineage}
  lineage_entries = {
    entry.seq: entry
    for segment in lineage
    for entry in segment.entries
  }
  entries, _ = await log.query(
    event_types={"parent_message_sent", "parent_message_consumed"},
    order="asc",
  )
  sent_by_identity: dict[tuple[str, str], list[LogEntry]] = {}
  consumed_by_identity: dict[tuple[str, str], list[LogEntry]] = {}
  for entry in entries:
    event = entry.event
    task_id = event.get("task_id")
    message_id = event.get("message_id")
    if task_id not in lineage_task_ids:
      continue
    if type(message_id) is not str or not message_id:
      raise TranscriptIntegrityError(
        "parent message lifecycle event lacks a canonical message_id"
      )
    identity = (task_id, message_id)
    target = (
      sent_by_identity
      if event.get("type") == "parent_message_sent"
      else consumed_by_identity
    )
    target.setdefault(identity, []).append(entry)

  bound_by_identity: dict[
    tuple[str, str],
    tuple[LogEntry, dict[str, Any]],
  ] = {}
  for assistant in lineage_entries.values():
    if assistant.event.get("type") != "assistant_message":
      continue
    raw_bindings = assistant.event.get("parent_message_consumptions", [])
    if not isinstance(raw_bindings, list):
      raise TranscriptIntegrityError(
        "assistant parent-message consumption bindings are invalid"
      )
    for binding in raw_bindings:
      if (
        not isinstance(binding, dict)
        or set(binding) != {
          "task_id",
          "message_id",
          "parent_message_seq",
          "consumer_turn",
        }
        or binding.get("task_id") not in lineage_task_ids
        or type(binding.get("message_id")) is not str
        or not binding["message_id"]
        or type(binding.get("parent_message_seq")) is not int
        or binding["parent_message_seq"] <= 0
        or type(binding.get("consumer_turn")) is not int
        or binding["consumer_turn"] <= 0
      ):
        raise TranscriptIntegrityError(
          "assistant parent-message consumption binding is malformed"
        )
      identity = (binding["task_id"], binding["message_id"])
      if identity in bound_by_identity:
        raise TranscriptIntegrityError(
          "parent message is bound to multiple assistant responses"
        )
      bound_by_identity[identity] = (assistant, binding)

  historical_by_assistant: dict[int, list[ParentMessage]] = {}
  consumed_identities: set[tuple[str, str]] = set()
  if any(len(sent_entries) != 1 for sent_entries in sent_by_identity.values()):
    raise TranscriptIntegrityError(
      "parent message has duplicate durable sent facts"
    )
  for identity, consumed_entries in consumed_by_identity.items():
    if len(consumed_entries) != 1:
      raise TranscriptIntegrityError(
        "parent message has duplicate durable consumption acknowledgements"
      )
    bound = bound_by_identity.get(identity)
    sent_entries = sent_by_identity.get(identity, [])
    if len(sent_entries) != 1:
      raise TranscriptIntegrityError(
        "consumed parent message lacks one exact durable sent fact"
      )
    if bound is None:
      raise TranscriptIntegrityError(
        "consumption audit lacks its atomic assistant binding"
      )
    assistant, binding = bound
    sent = sent_entries[0]
    consumed = consumed_entries[0]
    event = consumed.event
    lineage_consumed = lineage_entries.get(consumed.seq)
    if (
      binding["parent_message_seq"] != sent.seq
      or event.get("parent_message_seq") != sent.seq
      or event.get("assistant_message_seq") != assistant.seq
      or event.get("consumer_turn") != binding["consumer_turn"]
      or not sent.seq < assistant.seq < consumed.seq
      or lineage_consumed is None
      or lineage_consumed.event != consumed.event
      or assistant.event.get("runner_id") != event.get("runner_id")
      or assistant.event.get("role") != event.get("role")
      or assistant.event.get("sub_agent_id") != event.get("sub_agent_id")
      or (
        sent.event.get("sub_agent_id") is not None
        and sent.event.get("sub_agent_id") != event.get("sub_agent_id")
      )
    ):
      raise TranscriptIntegrityError(
        "parent message consumption acknowledgement has invalid child lineage"
      )

  for identity, (assistant, binding) in bound_by_identity.items():
    sent_entries = sent_by_identity.get(identity, [])
    if len(sent_entries) != 1:
      raise TranscriptIntegrityError(
        "assistant-bound parent message lacks one exact durable sent fact"
      )
    sent = sent_entries[0]
    if (
      binding["parent_message_seq"] != sent.seq
      or not sent.seq < assistant.seq
      or assistant.event.get("role") != "sub_agent"
      or assistant.event.get("sub_agent_id")
      != sent.event.get("sub_agent_id")
      or type(assistant.event.get("runner_id")) is not str
    ):
      raise TranscriptIntegrityError(
        "assistant-bound parent message has invalid child lineage"
      )
    raw_text = sent.event.get("message")
    if type(raw_text) is not str:
      raise TranscriptIntegrityError(
        "durable parent message body is not text"
      )
    historical_by_assistant.setdefault(assistant.seq, []).append(
      ParentMessage(
        message_id=identity[1],
        text=raw_text,
        sent_at=float(sent.event.get("sent_at") or _event_ts(sent)),
        task_id=identity[0],
        sent_seq=sent.seq,
      )
    )
    consumed_identities.add(identity)

  for messages in historical_by_assistant.values():
    messages.sort(key=lambda message: message.sent_seq or 0)
  return historical_by_assistant, consumed_identities


async def reconstruct_messages_for_task(log: AgentSessionLog, task_id: str) -> list[Message]:
  """Rebuild the model-facing transcript for a background sub-agent task."""
  lineage = await reconstruct_child_run_lineage(log, task_id)
  if not lineage:
    return []
  historical_parent_messages, _ = await _consumed_parent_message_projection(
    log,
    lineage,
  )

  messages: list[Message] = []
  pending_tool_results: list[ToolResultBlock] = []

  def flush_tool_results() -> None:
    nonlocal pending_tool_results
    if pending_tool_results:
      messages.append({"role": "user", "content": pending_tool_results})
      pending_tool_results = []

  for entry in (
    entry
    for segment in lineage
    for entry in segment.entries
  ):
    event = entry.event
    event_type = str(event.get("type") or "")
    if event_type == "tool_call_complete":
      pending_tool_results.extend(_tool_result_blocks_from_event(event))
      continue
    flush_tool_results()
    if event_type == "user_message":
      messages.append(_message_from_user_event(event))
    elif event_type == "assistant_message":
      consumed_updates = historical_parent_messages.get(entry.seq)
      if consumed_updates:
        messages.append({
          "role": "user",
          "content": format_parent_messages_for_model(consumed_updates),
        })
      messages.append(_message_from_assistant_event(event))
    elif event_type == "runtime_guard":
      messages.extend(_messages_from_runtime_guard_event(event))
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
  lineage = await reconstruct_child_run_lineage(log, task_id)
  _, consumed_identities = await _consumed_parent_message_projection(
    log,
    lineage,
  )
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
    if (task_id, message_id) in consumed_identities:
      continue
    sent_at = float(event.get("sent_at") or _event_ts(entry))
    if sent_at >= before_ts:
      continue
    parent_messages.append(
      ParentMessage(
        message_id=message_id,
        text=str(event.get("message") or ""),
        sent_at=sent_at,
        task_id=task_id,
        sent_seq=entry.seq,
      )
    )
  return parent_messages


def _parent_context_message(parent_messages: list[ParentMessage], additional_context: str | None) -> Message | None:
  lines = [format_parent_messages_for_model(parent_messages)] if parent_messages else []
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
  "ChildRunSegment",
  "TranscriptIntegrityError",
  "ToolResultBlock",
  "build_synthetic_tool_results",
  "child_run_segment_for_task",
  "detect_orphan_tool_uses",
  "place_resume_messages",
  "reconstruct_child_run_lineage",
  "reconstruct_messages_for_task",
  "reconstruct_parent_messages",
]
