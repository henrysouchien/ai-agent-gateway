from __future__ import annotations

import datetime
import uuid
from typing import Any

from .task_registry import ParentMessage, TaskState


def make_send_message_handler(runner_ref: list[Any]):
  """Build send_message handler using runner_ref late binding."""

  async def _handle_send_message(tool_input: dict[str, Any], **kwargs: Any):
    _ = kwargs
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
    if entry.completed:
      return None, {"code": "already_completed", "message": f"Agent {target} already finished"}

    message_id = str(tool_input.get("message_id") or uuid.uuid4())
    if message_id in entry.delivered_messages:
      return {"status": "delivered", "task_id": entry.task_id}, None

    sent_at = datetime.datetime.now(datetime.UTC).timestamp()
    append_durable_event = getattr(runner, "_append_durable_event", None)
    if append_durable_event is not None:
      metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
      await append_durable_event(
        {
          "type": "parent_message_sent",
          "task_id": entry.task_id,
          "owner_runner_id": metadata.get("owner_runner_id", getattr(runner, "_runner_id", None)),
          "owner_role": metadata.get("owner_role", getattr(runner, "_role", None)),
          "sub_agent_id": metadata.get("sub_agent_id"),
          "parent_turn_id": metadata.get("parent_turn_id"),
          "call_index": metadata.get("call_index"),
          "task_type": metadata.get("task_type", "background"),
          "provider_name": metadata.get("provider_name", entry.provider_name),
          "model": metadata.get("model", entry.model),
          "message_id": message_id,
          "sender": {
            "session_id": getattr(runner, "_full_session_id", None),
            "user_id": getattr(runner, "_usage_user_id", None),
          },
          "sent_at": sent_at,
          "message": message,
        }
      )
    await entry.message_inbox.put(ParentMessage(message_id=message_id, text=message, sent_at=sent_at))
    entry.delivered_messages.add(message_id)
    return {"status": "delivered", "task_id": entry.task_id}, None

  return _handle_send_message


__all__ = ["make_send_message_handler"]
