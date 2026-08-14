from __future__ import annotations

import datetime
import uuid
from typing import Any

from .runner_background_tasks import task_correlation_payload
from .task_registry import ParentMessage, TaskState


PARENT_MESSAGE_MAX_CHARS = 12_000
PARENT_MESSAGE_MAX_BYTES = 32 * 1024
PARENT_MESSAGE_ID_MAX_CHARS = 512
PARENT_MESSAGE_ID_MAX_BYTES = 512


def _message_result(
  message: ParentMessage,
  *,
  task_id: str,
) -> dict[str, Any]:
  return {
    "status": "accepted" if message.sent_seq is not None else "queued",
    "task_id": task_id,
    "message_id": message.message_id,
  }


async def _durable_parent_message(
  runner: Any,
  *,
  entry: Any,
  message_id: str,
) -> tuple[ParentMessage | None, dict[str, Any] | None]:
  """Resolve one exact acceptance from the existing session journal."""

  durable_log = getattr(runner, "_agent_session_log", None)
  if durable_log is None:
    return None, None
  entries, _ = await durable_log.query(
    event_types={"parent_message_sent"},
    contains_text=message_id,
    order="asc",
  )
  matches = [
    durable_entry
    for durable_entry in entries
    if (
      durable_entry.event.get("task_id") == entry.task_id
      and durable_entry.event.get("message_id") == message_id
    )
  ]
  if len(matches) > 1:
    return None, {
      "code": "message_integrity_error",
      "message": "message_id has duplicate durable acceptance facts",
    }
  if not matches:
    return None, None
  match = matches[0]
  event = match.event
  text = event.get("message")
  sent_at = event.get("sent_at")
  target_sub_agent_id = (
    entry.metadata.get("sub_agent_id")
    if isinstance(entry.metadata, dict)
    else None
  )
  if (
    type(text) is not str
    or not text
    or text != text.strip()
    or len(text) > PARENT_MESSAGE_MAX_CHARS
    or len(text.encode("utf-8")) > PARENT_MESSAGE_MAX_BYTES
    or isinstance(sent_at, bool)
    or not isinstance(sent_at, (int, float))
    or (
      target_sub_agent_id is not None
      and event.get("sub_agent_id") != target_sub_agent_id
    )
  ):
    return None, {
      "code": "message_integrity_error",
      "message": "message_id has an invalid durable acceptance fact",
    }
  return ParentMessage(
    message_id=message_id,
    text=text,
    sent_at=float(sent_at),
    task_id=entry.task_id,
    sent_seq=match.seq,
  ), None


async def enqueue_parent_message(
  runner: Any,
  *,
  task_id: str,
  message: str,
  message_id: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
  """Accept one exact task message, durably when a session log is present.

  ``TaskEntry.finalization_lock`` is shared with completion persistence.  The
  lock makes the durable ordering authoritative: a message that owns the lock
  first is accepted before completion; a completion that owns it first makes
  the message a terminal conflict.
  """

  if type(message) is not str or not message or message != message.strip():
    return None, {
      "code": "invalid_input",
      "message": "message must be canonical non-empty text",
    }
  if (
    len(message) > PARENT_MESSAGE_MAX_CHARS
    or len(message.encode("utf-8")) > PARENT_MESSAGE_MAX_BYTES
  ):
    return None, {"code": "invalid_input", "message": "message is too long"}
  registry = getattr(runner, "_task_registry", None)
  if registry is None:
    return None, {"code": "not_available", "message": "Task registry not configured"}
  entry = registry.get(task_id)
  if entry is None:
    return None, {"code": "not_found", "message": f"No running agent: {task_id}"}

  if message_id is None:
    resolved_message_id = str(uuid.uuid4())
  elif (
    type(message_id) is not str
    or not message_id
    or message_id != message_id.strip()
    or len(message_id) > PARENT_MESSAGE_ID_MAX_CHARS
    or len(message_id.encode("utf-8")) > PARENT_MESSAGE_ID_MAX_BYTES
  ):
    return None, {
      "code": "invalid_input",
      "message": "message_id must be bounded canonical non-empty text",
    }
  else:
    resolved_message_id = message_id
  async with entry.finalization_lock:
    existing = entry.accepted_parent_messages.get(resolved_message_id)
    if existing is None:
      existing, durable_error = await _durable_parent_message(
        runner,
        entry=entry,
        message_id=resolved_message_id,
      )
      if durable_error is not None:
        return None, durable_error
      if existing is not None:
        entry.accepted_parent_messages[resolved_message_id] = existing
    if existing is not None:
      if existing.text != message:
        return None, {
          "code": "message_id_conflict",
          "message": "message_id was already accepted with different content",
        }
      if not entry.completed and resolved_message_id not in entry.delivered_messages:
        await entry.message_inbox.put(existing)
        entry.delivered_messages.add(resolved_message_id)
      return _message_result(existing, task_id=entry.task_id), None
    if resolved_message_id in entry.delivered_messages:
      return None, {
        "code": "message_integrity_error",
        "message": "queued message identity lacks its accepted content",
      }
    if entry.completed:
      return None, {
        "code": "already_completed",
        "message": f"Agent {task_id} already finished",
      }

    sent_at = datetime.datetime.now(datetime.UTC).timestamp()
    sent_seq: int | None = None
    append_durable_event = getattr(runner, "_append_durable_event", None)
    if append_durable_event is not None:
      correlation = task_correlation_payload(
        entry,
        runner_id=getattr(runner, "_runner_id", None),
        role=getattr(runner, "_role", "writer"),
      )
      if correlation.get("capability_bind") is None:
        correlation.pop("capability_bind", None)
      durable_event = {
        **correlation,
        "type": "parent_message_sent",
        "message_id": resolved_message_id,
        "sender": {
          "session_id": getattr(runner, "_full_session_id", None),
          "user_id": getattr(runner, "_usage_user_id", None),
        },
        "sent_at": sent_at,
        "message": message,
      }
      durable_entry = await append_durable_event(durable_event)
      durable_seq = getattr(durable_entry, "seq", None)
      if type(durable_seq) is int and durable_seq > 0:
        sent_seq = durable_seq

    parent_message = ParentMessage(
      message_id=resolved_message_id,
      text=message,
      sent_at=sent_at,
      task_id=entry.task_id,
      sent_seq=sent_seq,
    )
    entry.accepted_parent_messages[resolved_message_id] = parent_message
    try:
      await entry.message_inbox.put(parent_message)
    except BaseException:
      if sent_seq is None:
        entry.accepted_parent_messages.pop(resolved_message_id, None)
      raise
    entry.delivered_messages.add(resolved_message_id)
    return _message_result(parent_message, task_id=entry.task_id), None


def make_send_message_handler(runner_ref: list[Any]):
  """Build send_message handler using runner_ref late binding."""

  async def _handle_send_message(tool_input: dict[str, Any], **kwargs: Any):
    runner = runner_ref[0]
    if runner is None:
      return None, {"code": "internal_error", "message": "Runner not initialized"}
    registry = getattr(runner, "_task_registry", None)
    if registry is None:
      return None, {"code": "not_available", "message": "Task registry not configured"}

    target = tool_input.get("to", "")
    if isinstance(target, str):
      target = target.strip()
    if not target or not isinstance(target, str):
      return None, {"code": "invalid_input", "message": "'to' is required"}

    message = tool_input.get("message", "")
    if isinstance(message, str):
      message = message.strip()
    if not message or not isinstance(message, str):
      return None, {"code": "invalid_input", "message": "'message' is required"}
    if (
      len(message) > PARENT_MESSAGE_MAX_CHARS
      or len(message.encode("utf-8")) > PARENT_MESSAGE_MAX_BYTES
    ):
      return None, {"code": "invalid_input", "message": "'message' is too long"}

    entry = registry.get(target)
    if entry is None:
      matches = [entry for entry in registry.list_tasks(state=TaskState.RUNNING) if entry.agent_name == target]
      if len(matches) > 1:
        ids = ", ".join(entry.task_id for entry in matches)
        return None, {
          "code": "ambiguous_target",
          "message": f"Multiple running agents named '{target}': {ids}. Use task_id instead.",
        }
      entry = matches[0] if matches else None

    if entry is None:
      return None, {"code": "not_found", "message": f"No running agent: {target}"}
    tool_ctx = kwargs.get("tool_ctx")
    trusted_message_id = getattr(tool_ctx, "tool_call_id", None)
    if tool_ctx is not None:
      if (
        type(trusted_message_id) is not str
        or not trusted_message_id
        or trusted_message_id != trusted_message_id.strip()
        or len(trusted_message_id) > PARENT_MESSAGE_ID_MAX_CHARS
        or len(trusted_message_id.encode("utf-8"))
        > PARENT_MESSAGE_ID_MAX_BYTES
      ):
        return None, {
          "code": "message_control_identity_invalid",
          "message": "send_message requires bounded trusted tool-call identity",
        }
    return await enqueue_parent_message(
      runner,
      task_id=entry.task_id,
      message=message,
      message_id=(
        trusted_message_id
        if trusted_message_id is not None
        else tool_input.get("message_id")
      ),
    )

  return _handle_send_message


__all__ = [
  "PARENT_MESSAGE_MAX_BYTES",
  "PARENT_MESSAGE_MAX_CHARS",
  "PARENT_MESSAGE_ID_MAX_BYTES",
  "PARENT_MESSAGE_ID_MAX_CHARS",
  "enqueue_parent_message",
  "make_send_message_handler",
]
