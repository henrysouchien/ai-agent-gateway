from __future__ import annotations

from collections.abc import Collection
from typing import Any, Callable

from .autonomous_runner_state import (
  _REHYDRATED_ACTIVE_STATES,
  _TERMINAL_AUTONOMOUS_STATES,
  AutonomousTask,
)
from .autonomous_control_files import find_closed_json_record
from .autonomous_control_contract import (
  AUTONOMOUS_OPERATOR_RECORD_FIELDS,
)
from .autonomous_launch_envelope import AutonomousControlAuthority


_OPERATOR_RECORD_FIELDS = AUTONOMOUS_OPERATOR_RECORD_FIELDS


def event_for_record(record: AutonomousTask, event: dict[str, Any]) -> dict[str, Any]:
  event_copy = dict(event)
  event_copy["run_id"] = record.control_run_id
  event_copy["control_run_id"] = record.control_run_id
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


def operator_inbox_record_for_message_id(
  record: AutonomousTask,
  message_id: str,
) -> dict[str, Any] | None:
  if record.operator_inbox_path is None:
    return None
  authority = record.control_authority
  if (
    type(authority) is not AutonomousControlAuthority
    or authority.operator_inbox_path != str(record.operator_inbox_path)
    or authority.operator_inbox_device is None
    or authority.operator_inbox_inode is None
  ):
    raise RuntimeError(
      "autonomous operator inbox does not match signed authority"
    )

  def matches(payload: dict[str, Any]) -> bool:
    if (
      set(payload) != _OPERATOR_RECORD_FIELDS
      or payload.get("version") != 1
      or payload.get("kind") != "operator_message"
      or payload.get("task_id") != record.task_id
      or payload.get("control_run_id") != record.control_run_id
      or payload.get("session_id") != record.session_id
      or payload.get("channel_id") != record.channel_id
    ):
      raise RuntimeError(
        "autonomous operator record violates its closed contract"
      )
    return payload.get("message_id") == message_id

  return find_closed_json_record(
    record.operator_inbox_path,
    expected_device=authority.operator_inbox_device,
    expected_inode=authority.operator_inbox_inode,
    kind="operator_message",
    fields=_OPERATOR_RECORD_FIELDS,
    predicate=matches,
  )


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
    inbox_text = inbox_record.get("text")
    if isinstance(inbox_text, str) and inbox_text:
      event_text = inbox_text
    inbox_sent_at_ns = inbox_record.get("sent_at_ns")
    if isinstance(inbox_sent_at_ns, int) and not isinstance(
      inbox_sent_at_ns,
      bool,
    ):
      event_sent_at = inbox_sent_at_ns / 1_000_000_000
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
  "event_already_recorded",
  "event_duplicate_key",
  "event_for_record",
  "operator_inbox_record_for_message_id",
  "parent_message_event",
  "record_replay_buffer_terminated",
  "replay_seed_events_for_record",
]
