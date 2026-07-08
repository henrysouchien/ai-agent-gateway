from __future__ import annotations

import asyncio
import json
import logging
import os  # noqa: F401 - compatibility alias for autonomous_runner_state
import secrets
import sys
import time
from pathlib import Path
from typing import Any

from .fixture_gate import is_fixture_profile_name, is_fixture_skill_name, require_fixture_provider_available
from .autonomous_runner_claims import (
  _AGENT_API_CLAIM_AUDIENCE as _AGENT_API_CLAIM_AUDIENCE,
  _AGENT_API_CLAIM_ENV_VARS as _AGENT_API_CLAIM_ENV_VARS,
  _AGENT_API_CLAIM_TTL_SECONDS_DEFAULT as _AGENT_API_CLAIM_TTL_SECONDS_DEFAULT,
  get_agent_api_claim_ttl_seconds as get_agent_api_claim_ttl_seconds,  # noqa: F401 - compatibility alias
  sign_user_claim as sign_user_claim,  # noqa: F401 - compatibility alias
)
from .autonomous_runner_state import (
  _ACTIVE_AUTONOMOUS_PROCESS_STATES,
  _AUTONOMOUS_MANIFEST_FILE_RE as _AUTONOMOUS_MANIFEST_FILE_RE,
  _AUTONOMOUS_RUN_FILE_RE as _AUTONOMOUS_RUN_FILE_RE,
  _AUTONOMOUS_TASK_ID_RE as _AUTONOMOUS_TASK_ID_RE,
  _ManifestTrackedList as _ManifestTrackedList,
  _REHYDRATE_EVENTS_SIZE_CAP_BYTES as _REHYDRATE_EVENTS_SIZE_CAP_BYTES,
  _REHYDRATE_EVENTS_TAIL_LINES as _REHYDRATE_EVENTS_TAIL_LINES,
  _REHYDRATED_ACTIVE_STATES,
  _REHYDRATION_INTERRUPTED_ERROR as _REHYDRATION_INTERRUPTED_ERROR,
  _RUN_RETENTION_DAYS_ENV as _RUN_RETENTION_DAYS_ENV,
  _RUN_RETENTION_SECONDS_PER_DAY as _RUN_RETENTION_SECONDS_PER_DAY,
  _RUN_SEQUENCE_CURSOR_FILE as _RUN_SEQUENCE_CURSOR_FILE,
  _TASK_MANIFEST_VERSION as _TASK_MANIFEST_VERSION,
  _TERMINAL_AUTONOMOUS_STATES,
  AutonomousRegistryStateMixin,
  AutonomousTask,
)
from . import autonomous_runner_events as _runner_events
from . import autonomous_runner_status as _runner_status
from . import autonomous_runner_commands as _runner_commands
from .autonomous_runner_start import (
  _SPAWN_CLEANUP_GRACE_SEC as _SPAWN_CLEANUP_GRACE_SEC,
  AutonomousRegistryStartMixin,
)

_STATUS_TAIL_LINES = 40
_AUTONOMOUS_PROFILE_NAME_RE = _runner_commands._AUTONOMOUS_PROFILE_NAME_RE
_LOGGER = logging.getLogger(__name__)
_APPROVAL_DECISION_AUTONOMOUS_STATES = {"running", "approval_pending", "remediating"}


def normalize_autonomous_profile(profile: str) -> str:
  return _runner_commands.normalize_autonomous_profile(
    profile,
    is_fixture_profile_name_func=is_fixture_profile_name,
    profile_name_re=_AUTONOMOUS_PROFILE_NAME_RE,
  )


class AutonomousRegistry(AutonomousRegistryStartMixin, AutonomousRegistryStateMixin):
  def __init__(
    self,
    *,
    api_dir: Path,
    python_executable: str | None = None,
    log_dir: Path | None = None,
    max_running: int = 2,
    user_event_bus: Any | None = None,
    approval_db_path: Path | str | None = None,
  ) -> None:
    self._api_dir = Path(api_dir)
    self._python = python_executable or sys.executable
    self._log_dir = (log_dir or Path("~/.cache/agent-gateway/autonomous").expanduser()).expanduser()
    self._max_running = max_running
    self._user_event_bus = user_event_bus
    self._approval_db_path = (
      Path(approval_db_path).expanduser().resolve() if approval_db_path is not None else None
    )
    self._tasks: dict[str, AutonomousTask] = {}
    self._seq = self._initial_task_seq()
    self._slot_lock = asyncio.Lock()
    self._reserved_slots = 0
    self._apply_run_file_retention()
    self.rehydrate()

  def set_user_event_bus(self, user_event_bus: Any | None) -> None:
    self._user_event_bus = user_event_bus

  def _next_task_id(self) -> str:
    task_id = f"bg_{self._seq}"
    self._seq += 1
    self._write_sequence_cursor()
    return task_id

  def _build_cmd(
    self,
    *,
    profile: str,
    mode: str,
    task: str | None,
    skill: str | None,
    context: str | None,
    ticker: str | None = None,
    dev_mode: bool = False,
  ) -> list[str]:
    return _runner_commands.build_autonomous_cmd(
      python_executable=self._python,
      profile=profile,
      mode=mode,
      task=task,
      skill=skill,
      context=context,
      ticker=ticker,
      dev_mode=dev_mode,
      normalize_autonomous_profile_func=normalize_autonomous_profile,
      is_fixture_profile_name_func=is_fixture_profile_name,
      is_fixture_skill_name_func=is_fixture_skill_name,
      require_fixture_provider_available_func=require_fixture_provider_available,
    )

  def _start_payload(self, record: AutonomousTask) -> dict[str, Any]:
    return {
      "task_id": record.task_id,
      "run_id": record.control_run_id,
      "log_path": str(record.log_path),
      "started_at": int(record.started_at),
      "cmd": list(record.cmd),
    }

  def _event_for_record(self, record: AutonomousTask, event: dict[str, Any]) -> dict[str, Any]:
    return _runner_events.event_for_record(record, event)

  def _replay_seed_events_for_record(self, record: AutonomousTask) -> list[dict[str, Any]]:
    return _runner_events.replay_seed_events_for_record(
      record,
      event_for_record_func=self._event_for_record,
    )

  def _record_replay_buffer_terminated(self, record: AutonomousTask) -> bool:
    return _runner_events.record_replay_buffer_terminated(
      record,
      terminal_states=_TERMINAL_AUTONOMOUS_STATES,
      rehydrated_active_states=_REHYDRATED_ACTIVE_STATES,
    )

  async def _seed_replay_buffer_for_record(self, record: AutonomousTask) -> None:
    if self._user_event_bus is None:
      return
    seed = getattr(self._user_event_bus, "seed_replay_buffer", None)
    if not callable(seed):
      return
    try:
      await seed(
        record.user_id,
        record.control_run_id,
        self._replay_seed_events_for_record(record),
        terminated=self._record_replay_buffer_terminated(record),
      )
    except Exception:
      pass

  def _event_duplicate_key(self, event: dict[str, Any]) -> tuple[str, str] | None:
    return _runner_events.event_duplicate_key(event)

  def _event_already_recorded(self, record: AutonomousTask, event: dict[str, Any]) -> bool:
    return _runner_events.event_already_recorded(
      record,
      event,
      event_duplicate_key_func=self._event_duplicate_key,
    )

  def _event_file_already_recorded(self, record: AutonomousTask, event: dict[str, Any]) -> bool:
    return _runner_events.event_file_already_recorded(
      record,
      event,
      event_duplicate_key_func=self._event_duplicate_key,
    )

  def _append_event_to_events_file(self, record: AutonomousTask, event: dict[str, Any]) -> None:
    _runner_events.append_event_to_events_file(
      record,
      event,
      event_file_already_recorded_func=self._event_file_already_recorded,
    )

  def _operator_inbox_record_for_message_id(
    self,
    record: AutonomousTask,
    message_id: str,
  ) -> dict[str, Any] | None:
    return _runner_events.operator_inbox_record_for_message_id(record, message_id)

  def _parent_message_event(
    self,
    record: AutonomousTask,
    *,
    message_id: str,
    text: str,
    user_id: str,
    sent_at: float,
  ) -> dict[str, Any]:
    return _runner_events.parent_message_event(
      record,
      message_id=message_id,
      text=text,
      user_id=user_id,
      sent_at=sent_at,
      operator_inbox_record_for_message_id_func=self._operator_inbox_record_for_message_id,
      event_for_record_func=self._event_for_record,
    )

  async def _persist_and_publish_parent_message_event(
    self,
    record: AutonomousTask,
    *,
    message_id: str,
    text: str,
    user_id: str,
    sent_at: float,
  ) -> None:
    event = self._parent_message_event(
      record,
      message_id=message_id,
      text=text,
      user_id=user_id,
      sent_at=sent_at,
    )
    self._append_event_to_events_file(record, event)
    await self._record_and_publish_event(record, event)

  async def _record_and_publish_event(self, record: AutonomousTask, event: dict[str, Any]) -> None:
    event_copy = self._event_for_record(record, event)
    if self._event_already_recorded(record, event_copy):
      return
    if record.event_lines is None:
      record.event_lines = []
    if self._user_event_bus is None:
      record.event_lines.append(event_copy)
      return
    await self._seed_replay_buffer_for_record(record)
    record.event_lines.append(event_copy)
    try:
      await self._user_event_bus.publish(
        user_id=record.user_id,
        control_run_id=record.control_run_id,
        event=event_copy,
      )
    except Exception:
      pass

  async def _publish_run_state(self, record: AutonomousTask, state: str) -> None:
    await self._record_and_publish_event(
      record,
      {
        "type": "run_state_changed",
        "run_id": record.control_run_id,
        "control_run_id": record.control_run_id,
        "state": state,
        "ts": int(time.time()),
      },
    )

  async def _cleanup_run_buffer(self, record: AutonomousTask) -> None:
    if self._user_event_bus is None:
      return
    try:
      await self._user_event_bus.cleanup_run(record.user_id, record.control_run_id)
    except Exception:
      pass

  def _terminal_state_for_record(self, record: AutonomousTask) -> str:
    if record.state == "killed":
      return "cancelled"
    if record.state in {"budget_limited", "budget_exceeded"} or self._record_has_budget_exceeded(record):
      return "budget_limited"
    if record.state == "blocked":
      return "blocked"
    if record.state in {"completed", "finished"}:
      return "completed"
    if record.state == "failed":
      return "failed"
    return "running"

  def _is_active_process_state(self, record: AutonomousTask) -> bool:
    return record.state in _ACTIVE_AUTONOMOUS_PROCESS_STATES

  def _has_terminal_run_state(self, record: AutonomousTask, state: str) -> bool:
    for event in record.event_lines or ():
      if event.get("type") == "run_state_changed" and event.get("state") == state:
        return True
    return False

  def _tail_lines(self, log_path: Path, line_count: int) -> tuple[list[str], int]:
    return _runner_status.tail_lines(log_path, line_count)

  def _status_payload(self, record: AutonomousTask) -> dict[str, Any]:
    return _runner_status.status_payload(
      record,
      tail_lines_func=self._tail_lines,
      status_tail_lines=_STATUS_TAIL_LINES,
    )

  def _get(self, task_id: str) -> AutonomousTask:
    record = self._tasks.get(task_id)
    if record is None:
      raise ValueError(f"Unknown task_id: {task_id}")
    return record

  def _find_by_control_run_id(self, control_run_id: str) -> AutonomousTask | None:
    record = self._tasks.get(control_run_id)
    if record is not None:
      return record
    return next(
      (task for task in self._tasks.values() if task.control_run_id == control_run_id),
      None,
    )

  def live_process_count(self) -> int:
    return sum(
      1
      for record in self._tasks.values()
      if record.proc is not None and record.proc.returncode is None
    )

  async def _tail_events_file(self, task_id: str) -> None:
    record = self._tasks.get(task_id)
    if record is None or record.events_path is None:
      return

    offset = 0
    while True:
      try:
        if record.events_path.exists():
          with record.events_path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            while True:
              line = handle.readline()
              if not line:
                break
              offset = handle.tell()
              stripped = line.strip()
              if not stripped:
                continue
              try:
                event = json.loads(stripped)
              except json.JSONDecodeError:
                event = {"type": "malformed_autonomous_event", "raw": stripped}
              if isinstance(event, dict):
                await self._record_and_publish_event(record, event)
      except FileNotFoundError:
        pass

      if record.completed_at is not None or record.state == "finished" or record.state in _TERMINAL_AUTONOMOUS_STATES:
        return
      await asyncio.sleep(0.1)

  async def _finish_events_tail(self, record: AutonomousTask) -> None:
    if record.events_tail_task is None:
      return
    try:
      await asyncio.wait_for(asyncio.shield(record.events_tail_task), timeout=1.0)
    except asyncio.TimeoutError:
      record.events_tail_task.cancel()
      await asyncio.gather(record.events_tail_task, return_exceptions=True)

  async def _reap(self, task_id: str) -> None:
    record = self._tasks.get(task_id)
    if record is None or record.proc is None:
      return
    try:
      exit_code = await record.proc.wait()
    except Exception as exc:
      if self._is_active_process_state(record):
        record.state = "failed"
        record.error = f"reaper failed: {exc}"
      record.completed_at = time.time()
      self._write_task_manifest(record)
      if record.log_handle is not None:
        record.log_handle.close()
        record.log_handle = None
      await self._release_slot(record)
      await self._finish_events_tail(record)
      if self._apply_budget_limited_terminal_state(record):
        self._write_task_manifest(record)
      terminal_state = self._terminal_state_for_record(record)
      if terminal_state != "running" and not self._has_terminal_run_state(record, terminal_state):
        await self._publish_run_state(record, terminal_state)
      if terminal_state != "running":
        await self._cleanup_run_buffer(record)
      return

    record.exit_code = exit_code
    if self._is_active_process_state(record):
      if exit_code == 0:
        record.state = "completed"
      else:
        record.state = "failed"
        record.error = record.error or f"Process exited with code {exit_code}"
      record.completed_at = time.time()
    else:
      record.completed_at = record.completed_at or time.time()
    self._write_task_manifest(record)

    if record.log_handle is not None:
      record.log_handle.close()
      record.log_handle = None

    await self._release_slot(record)
    await self._finish_events_tail(record)
    if self._apply_budget_limited_terminal_state(record):
      self._write_task_manifest(record)
    terminal_state = self._terminal_state_for_record(record)
    if terminal_state != "running" and not self._has_terminal_run_state(record, terminal_state):
      await self._publish_run_state(record, terminal_state)
    if terminal_state != "running":
      await self._cleanup_run_buffer(record)

  def status(self, task_id: str) -> dict[str, Any]:
    return self._status_payload(self._get(task_id))

  async def wait(self, task_id: str, *, timeout_sec: int = 600) -> dict[str, Any]:
    record = self._get(task_id)
    if self._is_active_process_state(record) and record.reaper_task is not None:
      try:
        await asyncio.wait_for(asyncio.shield(record.reaper_task), timeout=float(timeout_sec))
      except asyncio.TimeoutError:
        pass
    return self._status_payload(record)

  def logs(self, task_id: str, *, tail: int = 200) -> dict[str, Any]:
    record = self._get(task_id)
    lines, total_lines = self._tail_lines(record.log_path, int(tail))
    return {
      "task_id": record.task_id,
      "log_path": str(record.log_path),
      "lines": lines,
      "total_lines": total_lines,
    }

  async def send_operator_message(
    self,
    control_run_id: str,
    *,
    user_id: str,
    channel: str | None = None,
    message: str,
    message_id: str | None = None,
  ) -> dict[str, Any]:
    record = self._find_by_control_run_id(control_run_id)
    if record is None:
      raise ValueError(f"Unknown control_run_id: {control_run_id}")
    if (record.owner_user_id or record.user_id) != user_id:
      raise PermissionError("Run not found")

    normalized_channel = channel.strip().lower() if isinstance(channel, str) and channel.strip() else None
    if record.channel is not None and normalized_channel != record.channel:
      raise PermissionError("Run not found")

    if record.state not in {"running", "waiting", "approval_pending"} or (
      record.proc is not None and record.proc.returncode is not None
    ):
      raise RuntimeError("Autonomous run is not accepting messages")
    if record.event_lines is not None and any(
      event.get("type") == "stream_complete"
      for event in record.event_lines
    ):
      raise RuntimeError("Autonomous run is no longer accepting messages")

    text = message.strip() if isinstance(message, str) else ""
    if not text:
      raise ValueError("message is required")

    if record.operator_inbox_path is None:
      raise RuntimeError("Autonomous operator inbox unavailable")

    async with record.operator_message_lock:
      resolved_message_id = message_id.strip() if isinstance(message_id, str) and message_id.strip() else None
      resolved_message_id = resolved_message_id or f"op_{secrets.token_hex(8)}"
      if resolved_message_id in record.delivered_messages:
        await self._persist_and_publish_parent_message_event(
          record,
          message_id=resolved_message_id,
          text=text,
          user_id=user_id,
          sent_at=time.time(),
        )
        return {
          "task_id": record.task_id,
          "run_id": record.control_run_id,
          "message_id": resolved_message_id,
          "delivery_status": "duplicate",
        }

      existing_inbox_record = self._operator_inbox_record_for_message_id(record, resolved_message_id)
      if existing_inbox_record is not None:
        await self._persist_and_publish_parent_message_event(
          record,
          message_id=resolved_message_id,
          text=text,
          user_id=user_id,
          sent_at=time.time(),
        )
        record.delivered_messages.add(resolved_message_id)
        return {
          "task_id": record.task_id,
          "run_id": record.control_run_id,
          "message_id": resolved_message_id,
          "delivery_status": "duplicate",
        }

      sent_at = time.time()
      inbox_record = {
        "message_id": resolved_message_id,
        "text": text,
        "message": text,
        "sent_at": sent_at,
        "sender": {
          "user_id": user_id,
        },
        "channel": normalized_channel,
      }
      record.operator_inbox_path.parent.mkdir(parents=True, exist_ok=True)
      with record.operator_inbox_path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(inbox_record, default=str) + "\n")

      await self._persist_and_publish_parent_message_event(
        record,
        message_id=resolved_message_id,
        text=text,
        user_id=user_id,
        sent_at=sent_at,
      )
      record.delivered_messages.add(resolved_message_id)
      return {
        "task_id": record.task_id,
        "run_id": record.control_run_id,
        "message_id": resolved_message_id,
        "delivery_status": "delivered",
      }

  async def send_approval_decision(
    self,
    control_run_id: str,
    *,
    user_id: str,
    channel: str | None = None,
    approval_id: str,
    tool_call_id: str,
    nonce: str,
    approved: bool,
    allow_tool_type: bool = False,
    reason: str | None = None,
  ) -> dict[str, Any]:
    record = self._find_by_control_run_id(control_run_id)
    if record is None:
      raise ValueError(f"Unknown control_run_id: {control_run_id}")
    if (record.owner_user_id or record.user_id) != user_id:
      raise PermissionError("Run not found")

    normalized_channel = channel.strip().lower() if isinstance(channel, str) and channel.strip() else None
    if record.channel is not None and normalized_channel != record.channel:
      raise PermissionError("Run not found")

    if record.state not in _APPROVAL_DECISION_AUTONOMOUS_STATES or (
      record.proc is not None and record.proc.returncode is not None
    ):
      raise RuntimeError("Autonomous run is not running")
    if record.approval_decisions_path is None:
      raise RuntimeError("Autonomous approval inbox unavailable")

    decision_record = {
      "approval_id": approval_id,
      "tool_call_id": tool_call_id,
      "nonce": nonce,
      "approved": bool(approved),
      "allow_tool_type": bool(allow_tool_type),
      "reason": reason,
      "decider": {
        "user_id": user_id,
      },
      "channel": normalized_channel,
      "decided_at": time.time(),
    }
    record.approval_decisions_path.parent.mkdir(parents=True, exist_ok=True)
    with record.approval_decisions_path.open("a", encoding="utf-8", buffering=1) as handle:
      handle.write(json.dumps(decision_record, default=str) + "\n")

    await self._record_and_publish_event(
      record,
      {
        "type": "approval_decision_sent",
        "task_id": record.task_id,
        "run_id": record.control_run_id,
        "control_run_id": record.control_run_id,
        "approval_id": approval_id,
        "tool_call_id": tool_call_id,
        "approved": bool(approved),
        "allow_tool_type": bool(allow_tool_type),
        "decider": {
          "user_id": user_id,
        },
        "sent_at": decision_record["decided_at"],
      },
    )
    return {
      "task_id": record.task_id,
      "run_id": record.control_run_id,
      "approval_id": approval_id,
      "tool_call_id": tool_call_id,
    }

  async def cancel(self, task_id: str) -> dict[str, Any]:
    record = self._get(task_id)
    if self._is_active_process_state(record):
      if record.proc is not None and record.proc.returncode is None:
        try:
          record.proc.terminate()
        except ProcessLookupError:
          pass
      record.state = "killed"
      record.completed_at = time.time()
      record.error = record.error or "Process terminated by user"
      self._write_task_manifest(record)
      await self._publish_run_state(record, "cancelled")
      await self._cleanup_run_buffer(record)
    return self._status_payload(record)

  async def shutdown(self, *, grace_sec: float = 10.0) -> None:
    live_records = [
      record
      for record in self._tasks.values()
      if record.proc is not None and record.proc.returncode is None
    ]

    for record in live_records:
      if self._is_active_process_state(record):
        record.state = "killed"
        record.completed_at = time.time()
        record.error = record.error or "Process terminated during gateway shutdown"
        self._write_task_manifest(record)
      try:
        record.proc.terminate()
      except ProcessLookupError:
        pass

    waiters = [record.reaper_task for record in live_records if record.reaper_task is not None]
    if waiters:
      done, pending = await asyncio.wait(waiters, timeout=grace_sec)
      if pending:
        for record in live_records:
          if record.proc is not None and record.proc.returncode is None:
            try:
              record.proc.kill()
            except ProcessLookupError:
              pass
        await asyncio.gather(*pending, return_exceptions=True)
      else:
        await asyncio.gather(*done, return_exceptions=True)

    for record in self._tasks.values():
      if record.log_handle is not None:
        record.log_handle.close()
        record.log_handle = None
      if record.events_tail_task is not None and not record.events_tail_task.done():
        record.events_tail_task.cancel()
        await asyncio.gather(record.events_tail_task, return_exceptions=True)


__all__ = ["AutonomousRegistry", "AutonomousTask", "normalize_autonomous_profile"]
