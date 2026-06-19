from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union


def build_notification_reminder(notification_queue: Any, *, max_count: int) -> str:
  """Peek notifications into system prompt text. Non-destructive."""
  if notification_queue.pending_count == 0:
    return ""
  notifications = notification_queue.peek(max_count=max_count)
  parts = [notification.format_xml() for notification in notifications]
  remaining = notification_queue.pending_count - len(notifications)
  if remaining > 0:
    parts.append(f"[{remaining} more task notification(s) pending]")
  return "\n".join(parts)


def consume_notifications(notification_queue: Any, *, max_count: int) -> int:
  """Drain up to max_count notifications. Returns count consumed."""
  return len(notification_queue.drain(max_count=max_count))


def inject_system_prompt_reminder(
  system_prompt: Optional[Union[str, List[Tuple[str, bool]]]],
  reminder: str,
) -> Optional[Union[str, List[Tuple[str, bool]]]]:
  if not reminder:
    return system_prompt
  if isinstance(system_prompt, list):
    return [*system_prompt, (reminder, False)]
  base = system_prompt or ""
  if base:
    return f"{base}\n\n{reminder}"
  return reminder
