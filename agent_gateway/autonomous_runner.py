from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from .fixture_gate import is_fixture_profile_name, is_fixture_skill_name, require_fixture_provider_available
from .autonomous_runner_claims import (
  _AGENT_API_CLAIM_AUDIENCE as _AGENT_API_CLAIM_AUDIENCE,
  _AGENT_API_CLAIM_ENV_VARS as _AGENT_API_CLAIM_ENV_VARS,
  _AGENT_API_CLAIM_TTL_SECONDS_DEFAULT as _AGENT_API_CLAIM_TTL_SECONDS_DEFAULT,
  get_agent_api_claim_ttl_seconds,
  sign_user_claim,
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

_STATUS_TAIL_LINES = 40
_SPAWN_CLEANUP_GRACE_SEC = 1.0
_AUTONOMOUS_PROFILE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_LOGGER = logging.getLogger(__name__)


def normalize_autonomous_profile(profile: str) -> str:
  normalized_profile = str(profile or "").strip().lower()
  if not normalized_profile:
    raise ValueError("profile is required")
  if is_fixture_profile_name(normalized_profile):
    return normalized_profile
  if not _AUTONOMOUS_PROFILE_NAME_RE.fullmatch(normalized_profile):
    raise ValueError("profile must be a Python module-safe name using letters, numbers, and underscores")
  return normalized_profile


class AutonomousRegistry(AutonomousRegistryStateMixin):
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
    normalized_profile = normalize_autonomous_profile(profile)
    if is_fixture_profile_name(normalized_profile):
      require_fixture_provider_available("fixture profile dispatch", error_type=ValueError)

    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"once", "task", "skill"}:
      raise ValueError("mode must be once, task, or skill")

    if dev_mode and normalized_mode == "task":
      raise ValueError("dev_mode is implicit for mode='task'; do not pass dev_mode=True")
    if dev_mode and normalized_mode == "once":
      raise ValueError("dev_mode requires mode='skill'; use mode='task' for dev tasks instead")

    cmd = [self._python, "-m", "agent.autonomous", "--profile", normalized_profile]
    if dev_mode:
      cmd.append("--dev")

    if normalized_mode == "once":
      if task or skill or context:
        raise ValueError("mode='once' does not accept task, skill, or context")
      return cmd

    if normalized_mode == "task":
      if not task or not task.strip():
        raise ValueError("task is required when mode='task'")
      if skill or context:
        raise ValueError("mode='task' only accepts the task parameter")
      cmd.extend(["--task", task.strip()])
      return cmd

    if not skill or not skill.strip():
      raise ValueError("skill is required when mode='skill'")
    if is_fixture_skill_name(skill):
      require_fixture_provider_available("fixture skill dispatch", error_type=ValueError)
    if task:
      raise ValueError("mode='skill' does not accept task")
    cmd.extend(["--skill", skill.strip()])
    if ticker and ticker.strip():
      cmd.extend(["--ticker", ticker.strip().upper()])
    if context and context.strip():
      cmd.extend(["--context", context.strip()])
    return cmd

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
    if not log_path.exists():
      return [], 0
    if line_count <= 0:
      total_lines = 0
      with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for total_lines, _ in enumerate(handle, start=1):
          pass
      return [], total_lines

    total_lines = 0
    recent = deque(maxlen=line_count)
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
      for total_lines, line in enumerate(handle, start=1):
        recent.append(line.rstrip("\n"))
    return list(recent), total_lines

  def _status_payload(self, record: AutonomousTask) -> dict[str, Any]:
    payload: dict[str, Any] = {
      "state": record.state,
      "elapsed_sec": record.elapsed_sec,
    }
    if record.exit_code is not None:
      payload["exit_code"] = record.exit_code
    if record.error:
      payload["error"] = record.error
    lines, _total = self._tail_lines(record.log_path, _STATUS_TAIL_LINES)
    if lines:
      payload["log_tail"] = "\n".join(lines)
    return payload

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

  async def _reserve_slot(self) -> None:
    async with self._slot_lock:
      if self._reserved_slots >= self._max_running:
        raise RuntimeError(f"Autonomous concurrency limit reached ({self._max_running})")
      self._reserved_slots += 1

  async def _release_slot(self, record: AutonomousTask | None = None) -> None:
    async with self._slot_lock:
      if record is None:
        self._reserved_slots = max(0, self._reserved_slots - 1)
        return
      if not record.slot_reserved:
        return
      record.slot_reserved = False
      self._reserved_slots = max(0, self._reserved_slots - 1)

  async def _await_cleanup(self, cleanup_coro) -> None:
    cleanup_task = asyncio.create_task(cleanup_coro)
    try:
      await asyncio.shield(cleanup_task)
    except asyncio.CancelledError:
      await cleanup_task
      raise

  async def _terminate_unowned_process(self, record: AutonomousTask | None) -> None:
    proc = None if record is None else record.proc
    if proc is None or proc.returncode is not None:
      return
    try:
      proc.terminate()
    except ProcessLookupError:
      return
    try:
      await asyncio.wait_for(proc.wait(), timeout=_SPAWN_CLEANUP_GRACE_SEC)
    except asyncio.TimeoutError:
      try:
        proc.kill()
      except ProcessLookupError:
        pass
      await proc.wait()

  async def _cleanup_uncommitted_start(
    self,
    *,
    task_id: str,
    record: AutonomousTask | None,
    log_handle: Any | None,
  ) -> None:
    self._tasks.pop(task_id, None)
    self._delete_task_manifest(task_id)
    await self._terminate_unowned_process(record)
    if log_handle is not None:
      log_handle.close()
    if record is not None:
      record.log_handle = None
      if record.events_tail_task is not None and not record.events_tail_task.done():
        record.events_tail_task.cancel()
        await asyncio.gather(record.events_tail_task, return_exceptions=True)
      await self._release_slot(record)
    else:
      await self._release_slot()

  async def start(
    self,
    *,
    profile: str,
    mode: str,
    user_id: str,
    user_email: str | None,
    control_run_id: str | None = None,
    task: str | None = None,
    skill: str | None = None,
    context: str | None = None,
    ticker: str | None = None,
    channel: str | None = None,
    dev_mode: bool = False,
    resumed_from: str | None = None,
  ) -> dict[str, Any]:
    await self._reserve_slot()
    task_id = self._next_task_id()
    control_run_id = control_run_id or task_id
    log_handle = None
    record: AutonomousTask | None = None
    ownership_transferred = False
    try:
      cmd = self._build_cmd(
        profile=profile,
        mode=mode,
        task=task,
        skill=skill,
        context=context,
        ticker=ticker,
        dev_mode=dev_mode,
      )
      normalized_mode = mode.strip().lower()
      effective_dev_mode = bool(dev_mode or normalized_mode == "task")
      log_path = self._log_dir / f"{task_id}.log"
      events_path = self._log_dir / f"{task_id}.events.jsonl"
      operator_inbox_path = self._log_dir / f"{task_id}.operator-messages.jsonl"
      approval_decisions_path = self._log_dir / f"{task_id}.approval-decisions.jsonl"
      self._log_dir.mkdir(parents=True, exist_ok=True)
      events_path.write_text("", encoding="utf-8")
      operator_inbox_path.write_text("", encoding="utf-8")
      approval_decisions_path.write_text("", encoding="utf-8")
      log_handle = log_path.open("wb")
      record = AutonomousTask(
        task_id=task_id,
        control_run_id=control_run_id,
        user_id=user_id,
        user_email=user_email,
        profile=normalize_autonomous_profile(profile),
        mode=mode.strip().lower(),
        task=task.strip() if isinstance(task, str) and task.strip() else None,
        skill=skill.strip() if isinstance(skill, str) and skill.strip() else None,
        context=context.strip() if isinstance(context, str) and context.strip() else None,
        ticker=ticker.strip().upper() if isinstance(ticker, str) and ticker.strip() else None,
        channel=channel.strip().lower() if isinstance(channel, str) and channel.strip() else None,
        dev_mode=effective_dev_mode,
        cmd=cmd,
        log_path=log_path,
        events_path=events_path,
        operator_inbox_path=operator_inbox_path,
        approval_decisions_path=approval_decisions_path,
        started_at=time.time(),
        log_handle=log_handle,
        slot_reserved=True,
        event_lines=[],
        resumed_from=resumed_from.strip() if isinstance(resumed_from, str) and resumed_from.strip() else None,
      )
      self._attach_manifest_tracking(record)
      self._tasks[task_id] = record
      record.events_tail_task = asyncio.create_task(self._tail_events_file(task_id))

      env = dict(os.environ)
      env["PYTHONUNBUFFERED"] = "1"
      hmac_key = os.getenv("AGENT_API_USER_CLAIM_HMAC_KEY", "").strip()
      if not hmac_key:
        raise RuntimeError(
          "AGENT_API_USER_CLAIM_HMAC_KEY required for autonomous dispatch. "
          "Set it in the gateway env (.env or process env)."
        )
      claim_env = sign_user_claim(
        hmac_key,
        user_id=user_id,
        user_email=user_email,
        ttl_seconds=get_agent_api_claim_ttl_seconds(),
      )
      env.update(claim_env)
      env["AUTONOMOUS_USER_ID"] = user_id
      env["AUTONOMOUS_USER_EMAIL"] = user_email or ""
      env["AGENT_AUTONOMOUS_EVENTS_PATH"] = str(events_path)
      env["AGENT_AUTONOMOUS_OPERATOR_INBOX_PATH"] = str(operator_inbox_path)
      env["AGENT_AUTONOMOUS_APPROVAL_DECISIONS_PATH"] = str(approval_decisions_path)
      env["AGENT_AUTONOMOUS_GATEWAY_SESSION_ID"] = (
        f"agent-control:{control_run_id}:{int(record.started_at)}"
      )
      env["AGENT_AUTONOMOUS_CONTROL_RUN_ID"] = control_run_id
      env["AGENT_AUTONOMOUS_CONTROL_CHANNEL"] = record.channel or ""
      if record.dev_mode:
        env[f"{record.profile.upper().replace('-', '_')}_DEV_MODE"] = "true"
      if self._approval_db_path is not None:
        env["AGENT_AUTONOMOUS_APPROVALS_DB_PATH"] = str(self._approval_db_path)
      record.proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(self._api_dir),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=log_handle,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
      )

      assert record is not None
      record.reaper_task = asyncio.create_task(self._reap(task_id))
      await self._publish_run_state(record, "running")
      ownership_transferred = True
      self._write_task_manifest(record)
      return self._start_payload(record)
    except OSError as exc:
      raise RuntimeError(f"spawn failed: {exc}") from exc
    finally:
      if not ownership_transferred:
        await self._await_cleanup(
          self._cleanup_uncommitted_start(
            task_id=task_id,
            record=record,
            log_handle=log_handle,
          )
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
    if record.user_id != user_id:
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
    if record.user_id != user_id:
      raise PermissionError("Run not found")

    normalized_channel = channel.strip().lower() if isinstance(channel, str) and channel.strip() else None
    if record.channel is not None and normalized_channel != record.channel:
      raise PermissionError("Run not found")

    if not self._is_active_process_state(record) or (
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
