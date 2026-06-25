from __future__ import annotations

import json
from collections.abc import Collection
from typing import Any, Callable

from .autonomous_runner_state import (
  _REHYDRATED_ACTIVE_STATES,
  _TERMINAL_AUTONOMOUS_STATES,
  AutonomousTask,
)


def event_for_record(record: AutonomousTask, event: dict[str, Any]) -> dict[str, Any]:
  event_copy = dict(event)
  event_copy.setdefault("run_id", record.control_run_id)
  event_copy.setdefault("control_run_id", record.control_run_id)
  return event_copy


def replay_seed_events_for_record(
  record: AutonomousTask,
  *,
  event_for_record_func: Callable[[AutonomousTask, dict[str, Any]], dict[str, Any]] = event_for_record,
) -> list[dict[str, Any]]:
  return [
    event_for_record_func(record, event)
    for event in record.event_lines or []
    if isinstance(event, dict)
  ]


def record_replay_buffer_terminated(
  record: AutonomousTask,
  *,
  terminal_states: Collection[str] = _TERMINAL_AUTONOMOUS_STATES,
  rehydrated_active_states: Collection[str] = _REHYDRATED_ACTIVE_STATES,
) -> bool:
  if record.state in terminal_states or record.state == "finished":
    return True
  if record.proc is not None and record.proc.returncode is None:
    return False
  return record.state not in rehydrated_active_states and record.state != "starting"


def event_duplicate_key(event: dict[str, Any]) -> tuple[str, str] | None:
  if event.get("type") != "parent_message_sent":
    return None
  message_id = event.get("message_id")
  if not isinstance(message_id, str) or not message_id.strip():
    return None
  scope = "|".join(
    str(event.get(key) or "")
    for key in ("task_type", "task_id", "run_id", "control_run_id")
  )
  return ("parent_message_sent", f"{scope}|{message_id.strip()}")


def event_already_recorded(
  record: AutonomousTask,
  event: dict[str, Any],
  *,
  event_duplicate_key_func: Callable[[dict[str, Any]], tuple[str, str] | None] = event_duplicate_key,
) -> bool:
  duplicate_key = event_duplicate_key_func(event)
  if duplicate_key is None:
    return False
  for existing in record.event_lines or ():
    if event_duplicate_key_func(existing) == duplicate_key:
      return True
  return False


def event_file_already_recorded(
  record: AutonomousTask,
  event: dict[str, Any],
  *,
  event_duplicate_key_func: Callable[[dict[str, Any]], tuple[str, str] | None] = event_duplicate_key,
) -> bool:
  duplicate_key = event_duplicate_key_func(event)
  if duplicate_key is None or record.events_path is None:
    return False
  try:
    with record.events_path.open("r", encoding="utf-8", errors="replace") as handle:
      for line in handle:
        try:
          existing = json.loads(line)
        except json.JSONDecodeError:
          continue
        if isinstance(existing, dict) and event_duplicate_key_func(existing) == duplicate_key:
          return True
  except FileNotFoundError:
    return False
  return False


def append_event_to_events_file(
  record: AutonomousTask,
  event: dict[str, Any],
  *,
  event_file_already_recorded_func: Callable[[AutonomousTask, dict[str, Any]], bool] = event_file_already_recorded,
) -> None:
  if record.events_path is None:
    return
  if event_file_already_recorded_func(record, event):
    return
  record.events_path.parent.mkdir(parents=True, exist_ok=True)
  with record.events_path.open("a", encoding="utf-8", buffering=1) as handle:
    handle.write(json.dumps(event, default=str) + "\n")


def operator_inbox_record_for_message_id(
  record: AutonomousTask,
  message_id: str,
) -> dict[str, Any] | None:
  if record.operator_inbox_path is None:
    return None
  try:
    with record.operator_inbox_path.open("r", encoding="utf-8", errors="replace") as handle:
      for line in handle:
        try:
          payload = json.loads(line)
        except json.JSONDecodeError:
          continue
        if isinstance(payload, dict) and payload.get("message_id") == message_id:
          return payload
  except FileNotFoundError:
    return None
  return None


def parent_message_event(
  record: AutonomousTask,
  *,
  message_id: str,
  text: str,
  user_id: str,
  sent_at: float,
  operator_inbox_record_for_message_id_func: Callable[
    [AutonomousTask, str],
    dict[str, Any] | None,
  ] = operator_inbox_record_for_message_id,
  event_for_record_func: Callable[[AutonomousTask, dict[str, Any]], dict[str, Any]] = event_for_record,
) -> dict[str, Any]:
  inbox_record = operator_inbox_record_for_message_id_func(record, message_id)
  event_text = text
  event_sent_at: float | str = sent_at
  sender: dict[str, Any] = {"user_id": user_id}
  if inbox_record is not None:
    inbox_text = inbox_record.get("message") or inbox_record.get("text")
    if isinstance(inbox_text, str) and inbox_text:
      event_text = inbox_text
    inbox_sent_at = inbox_record.get("sent_at")
    if isinstance(inbox_sent_at, (int, float, str)):
      event_sent_at = inbox_sent_at
    inbox_sender = inbox_record.get("sender")
    if isinstance(inbox_sender, dict):
      sender = dict(inbox_sender)
  return event_for_record_func(
    record,
    {
      "type": "parent_message_sent",
      "task_id": record.task_id,
      "task_type": "autonomous",
      "profile": record.profile,
      "mode": record.mode,
      "message_id": message_id,
      "sender": sender,
      "sent_at": event_sent_at,
      "message": event_text,
    },
  )


__all__ = [
  "append_event_to_events_file",
  "event_already_recorded",
  "event_duplicate_key",
  "event_file_already_recorded",
  "event_for_record",
  "operator_inbox_record_for_message_id",
  "parent_message_event",
  "record_replay_buffer_terminated",
  "replay_seed_events_for_record",
]
