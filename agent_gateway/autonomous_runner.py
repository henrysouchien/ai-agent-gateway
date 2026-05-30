from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_AGENT_API_CLAIM_AUDIENCE = "agent_api_v1"
_AGENT_API_CLAIM_TTL_SECONDS_DEFAULT = 300
_AGENT_API_CLAIM_ENV_VARS = {
  "audience": "AGENT_API_CLAIM_AUDIENCE",
  "issued_at": "AGENT_API_CLAIM_ISSUED_AT",
  "expiry": "AGENT_API_CLAIM_EXPIRY",
  "user_id": "AGENT_API_CLAIM_USER_ID",
  "user_email": "AGENT_API_CLAIM_USER_EMAIL",
  "nonce": "AGENT_API_CLAIM_NONCE",
  "signature": "AGENT_API_CLAIM_SIGNATURE",
}
_STATUS_TAIL_LINES = 40
_SPAWN_CLEANUP_GRACE_SEC = 1.0
AUTONOMOUS_ALLOWED_PROFILES = frozenset({"analyst", "advisor", "research_producer", "tutor"})
_AUTONOMOUS_ALLOWED_PROFILE_DISPLAY = "analyst, advisor, research_producer, or tutor"


def get_agent_api_claim_ttl_seconds() -> int:
  raw = os.getenv("AGENT_API_CLAIM_TTL_SECONDS", "").strip()
  if not raw:
    return _AGENT_API_CLAIM_TTL_SECONDS_DEFAULT
  try:
    ttl_seconds = int(raw)
  except ValueError:
    return _AGENT_API_CLAIM_TTL_SECONDS_DEFAULT
  return ttl_seconds if ttl_seconds > 0 else _AGENT_API_CLAIM_TTL_SECONDS_DEFAULT


def sign_user_claim(
  hmac_key: str,
  *,
  user_id: str,
  user_email: str | None,
  ttl_seconds: int,
) -> dict[str, str]:
  if ttl_seconds <= 0:
    raise ValueError("ttl_seconds must be positive")

  issued_at = int(time.time())
  expiry = issued_at + ttl_seconds
  nonce = secrets.token_hex(16)
  normalized_email = user_email or ""
  canonical = f"{_AGENT_API_CLAIM_AUDIENCE}\n{issued_at}\n{expiry}\n{user_id}\n{normalized_email}\n{nonce}".encode(
    "utf-8"
  )
  signature = hmac.new(hmac_key.encode("utf-8"), canonical, hashlib.sha256).hexdigest()

  return {
    _AGENT_API_CLAIM_ENV_VARS["audience"]: _AGENT_API_CLAIM_AUDIENCE,
    _AGENT_API_CLAIM_ENV_VARS["issued_at"]: str(issued_at),
    _AGENT_API_CLAIM_ENV_VARS["expiry"]: str(expiry),
    _AGENT_API_CLAIM_ENV_VARS["user_id"]: user_id,
    _AGENT_API_CLAIM_ENV_VARS["user_email"]: normalized_email,
    _AGENT_API_CLAIM_ENV_VARS["nonce"]: nonce,
    _AGENT_API_CLAIM_ENV_VARS["signature"]: signature,
  }


@dataclass
class AutonomousTask:
  task_id: str
  control_run_id: str
  user_id: str
  user_email: str | None
  profile: str
  mode: str
  task: str | None
  skill: str | None
  context: str | None
  ticker: str | None
  channel: str | None
  dev_mode: bool
  cmd: list[str]
  log_path: Path
  events_path: Path | None
  started_at: float
  state: str = "running"
  exit_code: int | None = None
  error: str | None = None
  proc: asyncio.subprocess.Process | None = None
  reaper_task: asyncio.Task[None] | None = None
  events_tail_task: asyncio.Task[None] | None = None
  completed_at: float | None = None
  log_handle: Any | None = None
  slot_reserved: bool = False
  event_lines: list[dict[str, Any]] | None = None

  @property
  def elapsed_sec(self) -> int:
    end_time = self.completed_at if self.completed_at is not None else time.time()
    return max(0, int(end_time - self.started_at))


class AutonomousRegistry:
  def __init__(
    self,
    *,
    api_dir: Path,
    python_executable: str | None = None,
    log_dir: Path | None = None,
    max_running: int = 2,
    user_event_bus: Any | None = None,
  ) -> None:
    self._api_dir = Path(api_dir)
    self._python = python_executable or sys.executable
    self._log_dir = (log_dir or Path("~/.cache/agent-gateway/autonomous").expanduser()).expanduser()
    self._max_running = max_running
    self._user_event_bus = user_event_bus
    self._tasks: dict[str, AutonomousTask] = {}
    self._seq = 0
    self._slot_lock = asyncio.Lock()
    self._reserved_slots = 0

  def set_user_event_bus(self, user_event_bus: Any | None) -> None:
    self._user_event_bus = user_event_bus

  def _next_task_id(self) -> str:
    task_id = f"bg_{self._seq}"
    self._seq += 1
    return task_id

  def _build_cmd(
    self,
    *,
    profile: str,
    mode: str,
    task: str | None,
    skill: str | None,
    context: str | None,
    dev_mode: bool = False,
  ) -> list[str]:
    normalized_profile = profile.strip().lower()
    if normalized_profile not in AUTONOMOUS_ALLOWED_PROFILES:
      raise ValueError(f"profile must be {_AUTONOMOUS_ALLOWED_PROFILE_DISPLAY}")

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
    if task:
      raise ValueError("mode='skill' does not accept task")
    cmd.extend(["--skill", skill.strip()])
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
    event_copy = dict(event)
    event_copy.setdefault("run_id", record.control_run_id)
    event_copy.setdefault("control_run_id", record.control_run_id)
    return event_copy

  async def _record_and_publish_event(self, record: AutonomousTask, event: dict[str, Any]) -> None:
    event_copy = self._event_for_record(record, event)
    if record.event_lines is None:
      record.event_lines = []
    record.event_lines.append(event_copy)
    if self._user_event_bus is None:
      return
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
    if record.state in {"completed", "finished"}:
      return "completed"
    if record.state == "failed":
      return "failed"
    return "running"

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
  ) -> dict[str, Any]:
    cmd = self._build_cmd(profile=profile, mode=mode, task=task, skill=skill, context=context, dev_mode=dev_mode)
    await self._reserve_slot()
    task_id = self._next_task_id()
    control_run_id = control_run_id or task_id
    log_path = self._log_dir / f"{task_id}.log"
    events_path = self._log_dir / f"{task_id}.events.jsonl"
    log_handle = None
    record: AutonomousTask | None = None
    ownership_transferred = False
    try:
      self._log_dir.mkdir(parents=True, exist_ok=True)
      events_path.touch(exist_ok=True)
      log_handle = log_path.open("ab")
      record = AutonomousTask(
        task_id=task_id,
        control_run_id=control_run_id,
        user_id=user_id,
        user_email=user_email,
        profile=profile.strip().lower(),
        mode=mode.strip().lower(),
        task=task.strip() if isinstance(task, str) and task.strip() else None,
        skill=skill.strip() if isinstance(skill, str) and skill.strip() else None,
        context=context.strip() if isinstance(context, str) and context.strip() else None,
        ticker=ticker.strip().upper() if isinstance(ticker, str) and ticker.strip() else None,
        channel=channel.strip().lower() if isinstance(channel, str) and channel.strip() else None,
        dev_mode=bool(dev_mode),
        cmd=cmd,
        log_path=log_path,
        events_path=events_path,
        started_at=time.time(),
        log_handle=log_handle,
        slot_reserved=True,
        event_lines=[],
      )
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

      if record.completed_at is not None or record.state in {"completed", "finished", "failed", "killed"}:
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
      if record.state == "running":
        record.state = "failed"
        record.error = f"reaper failed: {exc}"
      record.completed_at = time.time()
      if record.log_handle is not None:
        record.log_handle.close()
        record.log_handle = None
      await self._release_slot(record)
      await self._finish_events_tail(record)
      terminal_state = self._terminal_state_for_record(record)
      if terminal_state != "running" and not self._has_terminal_run_state(record, terminal_state):
        await self._publish_run_state(record, terminal_state)
      if terminal_state != "running":
        await self._cleanup_run_buffer(record)
      return

    record.exit_code = exit_code
    if record.state == "running":
      if exit_code == 0:
        record.state = "completed"
      else:
        record.state = "failed"
        record.error = record.error or f"Process exited with code {exit_code}"
      record.completed_at = time.time()
    else:
      record.completed_at = record.completed_at or time.time()

    if record.log_handle is not None:
      record.log_handle.close()
      record.log_handle = None

    await self._release_slot(record)
    await self._finish_events_tail(record)
    terminal_state = self._terminal_state_for_record(record)
    if terminal_state != "running" and not self._has_terminal_run_state(record, terminal_state):
      await self._publish_run_state(record, terminal_state)
    if terminal_state != "running":
      await self._cleanup_run_buffer(record)

  def status(self, task_id: str) -> dict[str, Any]:
    return self._status_payload(self._get(task_id))

  async def wait(self, task_id: str, *, timeout_sec: int = 600) -> dict[str, Any]:
    record = self._get(task_id)
    if record.state == "running" and record.reaper_task is not None:
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

  async def cancel(self, task_id: str) -> dict[str, Any]:
    record = self._get(task_id)
    if record.state == "running":
      if record.proc is not None and record.proc.returncode is None:
        try:
          record.proc.terminate()
        except ProcessLookupError:
          pass
      record.state = "killed"
      record.completed_at = time.time()
      record.error = record.error or "Process terminated by user"
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
      if record.state == "running":
        record.state = "killed"
        record.completed_at = time.time()
        record.error = record.error or "Process terminated during gateway shutdown"
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


__all__ = ["AUTONOMOUS_ALLOWED_PROFILES", "AutonomousRegistry", "AutonomousTask"]
