from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .fixture_gate import is_fixture_profile_name, is_fixture_skill_name, require_fixture_provider_available

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
_AUTONOMOUS_PROFILE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_AUTONOMOUS_TASK_ID_RE = re.compile(r"^bg_\d+$")
_AUTONOMOUS_RUN_FILE_RE = re.compile(r"^bg_(\d+)\..+")
_AUTONOMOUS_MANIFEST_FILE_RE = re.compile(r"^bg_(\d+)\.task\.json$")
_ACTIVE_AUTONOMOUS_PROCESS_STATES = {"running", "approval_pending", "remediating"}
_REHYDRATED_ACTIVE_STATES = {"running", "approval_pending", "queued", "waiting", "remediating"}
_TERMINAL_AUTONOMOUS_STATES = {"completed", "failed", "killed", "interrupted", "budget_limited", "budget_exceeded", "blocked"}
_REHYDRATION_INTERRUPTED_ERROR = "gateway restarted while run was active"
_REHYDRATE_EVENTS_SIZE_CAP_BYTES = 5 * 1024 * 1024
_REHYDRATE_EVENTS_TAIL_LINES = 2000
_TASK_MANIFEST_VERSION = 1
_RUN_SEQUENCE_CURSOR_FILE = ".autonomous-sequence.json"
_RUN_RETENTION_DAYS_ENV = "AGENT_AUTONOMOUS_RUN_RETENTION_DAYS"
_RUN_RETENTION_SECONDS_PER_DAY = 86400.0
_LOGGER = logging.getLogger(__name__)


class _ManifestTrackedList(list[str]):
  def __init__(self, values: list[str], on_change) -> None:
    super().__init__(values)
    self._on_change = on_change

  def _changed(self) -> None:
    self._on_change()

  def append(self, item: str) -> None:
    super().append(item)
    self._changed()

  def extend(self, values) -> None:
    super().extend(values)
    self._changed()

  def insert(self, index: int, item: str) -> None:
    super().insert(index, item)
    self._changed()

  def remove(self, item: str) -> None:
    super().remove(item)
    self._changed()

  def pop(self, index: int = -1) -> str:
    item = super().pop(index)
    self._changed()
    return item

  def clear(self) -> None:
    super().clear()
    self._changed()

  def sort(self, *args, **kwargs) -> None:
    super().sort(*args, **kwargs)
    self._changed()

  def reverse(self) -> None:
    super().reverse()
    self._changed()

  def __setitem__(self, index, value) -> None:
    super().__setitem__(index, value)
    self._changed()

  def __delitem__(self, index) -> None:
    super().__delitem__(index)
    self._changed()

  def __iadd__(self, values):
    result = super().__iadd__(values)
    self._changed()
    return result


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


def normalize_autonomous_profile(profile: str) -> str:
  normalized_profile = str(profile or "").strip().lower()
  if not normalized_profile:
    raise ValueError("profile is required")
  if is_fixture_profile_name(normalized_profile):
    return normalized_profile
  if not _AUTONOMOUS_PROFILE_NAME_RE.fullmatch(normalized_profile):
    raise ValueError("profile must be a Python module-safe name using letters, numbers, and underscores")
  return normalized_profile


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
  operator_inbox_path: Path | None
  approval_decisions_path: Path | None
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
  delivered_messages: set[str] = field(default_factory=set)
  operator_message_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
  resume_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
  resumed_from: str | None = None
  resumed_as: list[str] = field(default_factory=list)

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

  def _initial_task_seq(self) -> int:
    if not self._log_dir.exists():
      return 0
    max_existing = -1
    try:
      for path in self._log_dir.iterdir():
        if not path.is_file():
          continue
        match = _AUTONOMOUS_RUN_FILE_RE.match(path.name)
        if match is None:
          continue
        max_existing = max(max_existing, int(match.group(1)))
    except OSError:
      _LOGGER.warning(
        "Failed to scan autonomous log directory for existing run ids: %s",
        self._log_dir,
        exc_info=True,
      )
      return 0
    return max(max_existing + 1, self._read_sequence_cursor())

  def _sequence_cursor_path(self) -> Path:
    return self._log_dir / _RUN_SEQUENCE_CURSOR_FILE

  def _read_sequence_cursor(self) -> int:
    path = self._sequence_cursor_path()
    try:
      payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
      return 0
    except (OSError, json.JSONDecodeError):
      _LOGGER.warning("Failed to read autonomous run sequence cursor: %s", path, exc_info=True)
      return 0
    if not isinstance(payload, dict):
      return 0
    next_seq = payload.get("next_seq")
    return int(next_seq) if isinstance(next_seq, int) and next_seq > 0 else 0

  def _write_sequence_cursor(self) -> bool:
    path = self._sequence_cursor_path()
    tmp_path = path.with_name(f"{path.name}.{secrets.token_hex(8)}.tmp")
    try:
      path.parent.mkdir(parents=True, exist_ok=True)
      tmp_path.write_text(json.dumps({"next_seq": self._seq}, sort_keys=True) + "\n", encoding="utf-8")
      os.replace(tmp_path, path)
      return True
    except Exception:
      try:
        tmp_path.unlink()
      except OSError:
        pass
      _LOGGER.warning("Failed to write autonomous run sequence cursor: %s", path, exc_info=True)
      return False

  def _configured_run_retention_days(self) -> float | None:
    raw = os.getenv(_RUN_RETENTION_DAYS_ENV, "").strip()
    if not raw:
      return None
    try:
      days = float(raw)
    except ValueError:
      _LOGGER.warning("Ignoring invalid %s=%r; expected positive day count", _RUN_RETENTION_DAYS_ENV, raw)
      return None
    return days if days > 0 else None

  def _run_file_group_paths(self, task_id: str, manifest_path: Path, manifest: dict[str, Any]) -> list[Path]:
    """Return the manifest plus every durable evidence file needed for resume."""

    paths: set[Path] = set()
    try:
      paths.update(path for path in self._log_dir.glob(f"{task_id}.*") if path.is_file())
    except OSError:
      _LOGGER.warning("Failed to scan autonomous run files for retention: %s", task_id, exc_info=True)
    paths.add(manifest_path)
    for field_name in ("log_path", "events_path", "operator_inbox_path", "approval_decisions_path"):
      path = self._path_from_manifest(manifest, field_name)
      if path is not None and self._is_log_dir_child(path):
        paths.add(path)
    return sorted(paths, key=lambda path: path.name)

  def _is_log_dir_child(self, path: Path) -> bool:
    try:
      log_dir = self._log_dir.resolve()
      resolved = path.expanduser().resolve()
    except OSError:
      return False
    return resolved == log_dir or log_dir in resolved.parents

  def _manifest_retention_timestamp(self, manifest_path: Path, manifest: dict[str, Any]) -> float | None:
    for field_name in ("completed_at", "started_at"):
      value = manifest.get(field_name)
      if isinstance(value, (int, float)):
        return float(value)
    try:
      return manifest_path.stat().st_mtime
    except OSError:
      return None

  def _should_prune_run_manifest(
    self,
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    cutoff_ts: float,
  ) -> bool:
    if manifest.get("manifest_version") != _TASK_MANIFEST_VERSION:
      return False
    state = str(manifest.get("state") or "").strip().lower()
    if state not in {"completed", "finished"}:
      return False
    if manifest.get("resumed_from"):
      return False
    resumed_as = manifest.get("resumed_as")
    if isinstance(resumed_as, list) and resumed_as:
      return False
    completed_ts = self._manifest_retention_timestamp(manifest_path, manifest)
    return completed_ts is not None and completed_ts < cutoff_ts

  def _apply_run_file_retention(self) -> None:
    retention_days = self._configured_run_retention_days()
    if retention_days is None or not self._log_dir.exists():
      return

    cutoff_ts = time.time() - (retention_days * _RUN_RETENTION_SECONDS_PER_DAY)
    pruned_runs = 0
    pruned_files = 0
    if not self._write_sequence_cursor():
      return
    for manifest_path in self._manifest_paths():
      try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      except (OSError, json.JSONDecodeError):
        continue
      if not isinstance(manifest, dict):
        continue
      if not self._should_prune_run_manifest(manifest_path, manifest, cutoff_ts=cutoff_ts):
        continue
      task_id = self._coerce_manifest_str(manifest, "task_id")
      if not task_id or _AUTONOMOUS_TASK_ID_RE.fullmatch(task_id) is None:
        continue
      deleted_any = False
      for path in self._run_file_group_paths(task_id, manifest_path, manifest):
        try:
          path.unlink()
          pruned_files += 1
          deleted_any = True
        except FileNotFoundError:
          pass
        except OSError:
          _LOGGER.warning("Failed to prune autonomous run file: %s", path, exc_info=True)
      if deleted_any:
        self._tasks.pop(task_id, None)
        pruned_runs += 1
    if pruned_runs:
      _LOGGER.info(
        "Pruned %d autonomous run(s), %d file(s), older than %.3g day(s)",
        pruned_runs,
        pruned_files,
        retention_days,
      )

  def _manifest_path(self, task_id: str) -> Path:
    return self._log_dir / f"{task_id}.task.json"

  def _manifest_payload(self, record: AutonomousTask) -> dict[str, Any]:
    return {
      "manifest_version": _TASK_MANIFEST_VERSION,
      "task_id": record.task_id,
      "control_run_id": record.control_run_id,
      "user_id": record.user_id,
      "user_email": record.user_email,
      "profile": record.profile,
      "mode": record.mode,
      "task": record.task,
      "skill": record.skill,
      "context": record.context,
      "ticker": record.ticker,
      "channel": record.channel,
      "dev_mode": record.dev_mode,
      "cmd": list(record.cmd),
      "log_path": str(record.log_path),
      "events_path": str(record.events_path) if record.events_path is not None else None,
      "operator_inbox_path": (
        str(record.operator_inbox_path) if record.operator_inbox_path is not None else None
      ),
      "approval_decisions_path": (
        str(record.approval_decisions_path) if record.approval_decisions_path is not None else None
      ),
      "started_at": record.started_at,
      "state": record.state,
      "exit_code": record.exit_code,
      "error": record.error,
      "completed_at": record.completed_at,
      "resumed_from": record.resumed_from,
      "resumed_as": list(record.resumed_as),
    }

  def _attach_manifest_tracking(self, record: AutonomousTask) -> None:
    if isinstance(record.resumed_as, _ManifestTrackedList):
      return
    record.resumed_as = _ManifestTrackedList(
      list(record.resumed_as),
      lambda: self._write_task_manifest(record),
    )

  def _write_task_manifest(self, record: AutonomousTask) -> None:
    manifest_path = self._manifest_path(record.task_id)
    tmp_path = manifest_path.with_name(f"{manifest_path.name}.{secrets.token_hex(8)}.tmp")
    try:
      manifest_path.parent.mkdir(parents=True, exist_ok=True)
      tmp_path.write_text(
        json.dumps(self._manifest_payload(record), sort_keys=True) + "\n",
        encoding="utf-8",
      )
      os.replace(tmp_path, manifest_path)
    except Exception:
      try:
        tmp_path.unlink()
      except OSError:
        pass
      _LOGGER.warning(
        "Failed to write autonomous task manifest for %s",
        record.task_id,
        exc_info=True,
      )

  def _delete_task_manifest(self, task_id: str) -> None:
    try:
      self._manifest_path(task_id).unlink()
    except FileNotFoundError:
      pass
    except OSError:
      _LOGGER.warning(
        "Failed to delete autonomous task manifest for uncommitted run %s",
        task_id,
        exc_info=True,
      )

  def _manifest_paths(self) -> list[Path]:
    if not self._log_dir.exists():
      return []
    try:
      return sorted(
        (
          path
          for path in self._log_dir.iterdir()
          if path.is_file() and _AUTONOMOUS_MANIFEST_FILE_RE.match(path.name)
        ),
        key=lambda path: path.name,
      )
    except OSError:
      _LOGGER.warning(
        "Failed to scan autonomous log directory for manifests: %s",
        self._log_dir,
        exc_info=True,
      )
      return []

  def _path_from_manifest(
    self,
    manifest: dict[str, Any],
    field_name: str,
    *,
    fallback: Path | None = None,
  ) -> Path | None:
    raw = manifest.get(field_name)
    if isinstance(raw, str) and raw.strip():
      return Path(raw).expanduser()
    return fallback

  def _event_timestamp(self, event: dict[str, Any]) -> float | None:
    for key in ("ts", "sent_at", "decided_at", "timestamp", "created_at"):
      raw = event.get(key)
      if isinstance(raw, (int, float)):
        return float(raw)
      if isinstance(raw, str):
        try:
          return float(raw)
        except ValueError:
          continue
    return None

  def _last_event_timestamp(self, events: list[dict[str, Any]]) -> float | None:
    for event in reversed(events):
      ts = self._event_timestamp(event)
      if ts is not None:
        return ts
    return None

  def _parse_event_lines(self, lines: list[str], *, path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in lines:
      stripped = line.strip()
      if not stripped:
        continue
      try:
        event = json.loads(stripped)
      except json.JSONDecodeError:
        _LOGGER.warning("Skipping malformed autonomous event line while rehydrating %s", path)
        continue
      if isinstance(event, dict):
        events.append(event)
    return events

  def _load_rehydrated_events(self, events_path: Path | None) -> list[dict[str, Any]]:
    if events_path is None:
      return []
    try:
      stat = events_path.stat()
    except FileNotFoundError:
      return []
    except OSError:
      _LOGGER.warning("Failed to stat autonomous events file for rehydration: %s", events_path, exc_info=True)
      return []

    try:
      if stat.st_size <= _REHYDRATE_EVENTS_SIZE_CAP_BYTES:
        with events_path.open("r", encoding="utf-8", errors="replace") as handle:
          return self._parse_event_lines(handle.readlines(), path=events_path)

      _LOGGER.warning(
        "Autonomous events file exceeds rehydrate cap; loading tail only: %s",
        events_path,
      )
      recent = deque(maxlen=_REHYDRATE_EVENTS_TAIL_LINES)
      with events_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
          recent.append(line)
      return self._parse_event_lines(list(recent), path=events_path)
    except OSError:
      _LOGGER.warning("Failed to load autonomous events for rehydration: %s", events_path, exc_info=True)
      return []

  @staticmethod
  def _events_include_budget_exceeded(events: list[dict[str, Any]] | None) -> bool:
    return any(
      isinstance(event, dict) and event.get("type") == "budget_exceeded"
      for event in events or []
    )

  def _record_has_budget_exceeded(self, record: AutonomousTask) -> bool:
    return self._events_include_budget_exceeded(record.event_lines)

  def _canonical_terminal_state(self, state: str, events: list[dict[str, Any]] | None) -> str:
    if state in {"budget_limited", "budget_exceeded"}:
      return "budget_limited"
    if state in {"completed", "finished", "failed"} and self._events_include_budget_exceeded(events):
      return "budget_limited"
    return state

  def _apply_budget_limited_terminal_state(self, record: AutonomousTask) -> bool:
    if record.state not in {"completed", "finished", "failed", "budget_exceeded"}:
      return False
    if not self._record_has_budget_exceeded(record):
      return False
    previous_state = record.state
    record.state = "budget_limited"
    if record.error and record.error.startswith("Process exited with code "):
      record.error = None
    return record.state != previous_state

  def _coerce_manifest_str(self, manifest: dict[str, Any], field_name: str) -> str | None:
    value = manifest.get(field_name)
    return value if isinstance(value, str) else None

  def _task_from_manifest(
    self,
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    rehydrate_time: float,
  ) -> AutonomousTask | None:
    if manifest.get("manifest_version") != _TASK_MANIFEST_VERSION:
      _LOGGER.warning("Skipping autonomous manifest with unsupported version: %s", manifest_path)
      return None

    task_id = self._coerce_manifest_str(manifest, "task_id")
    control_run_id = self._coerce_manifest_str(manifest, "control_run_id")
    user_id = self._coerce_manifest_str(manifest, "user_id")
    profile = self._coerce_manifest_str(manifest, "profile")
    mode = self._coerce_manifest_str(manifest, "mode")
    if (
      not task_id
      or _AUTONOMOUS_TASK_ID_RE.fullmatch(task_id) is None
      or not control_run_id
      or not user_id
      or not profile
      or not mode
    ):
      _LOGGER.warning("Skipping autonomous manifest missing required fields: %s", manifest_path)
      return None

    events_path = self._path_from_manifest(
      manifest,
      "events_path",
      fallback=manifest_path.with_name(f"{task_id}.events.jsonl"),
    )
    events = self._load_rehydrated_events(events_path)

    raw_manifest_state = self._coerce_manifest_str(manifest, "state") or "running"
    raw_state = self._canonical_terminal_state(raw_manifest_state, events)
    was_interrupted = False
    completed_at = manifest.get("completed_at")
    completed_at = float(completed_at) if isinstance(completed_at, (int, float)) else None
    error = self._coerce_manifest_str(manifest, "error")
    if raw_state == "budget_limited" and error and error.startswith("Process exited with code "):
      error = None
    if raw_state in _REHYDRATED_ACTIVE_STATES or raw_state not in _TERMINAL_AUTONOMOUS_STATES:
      raw_state = "interrupted"
      was_interrupted = True
      completed_at = self._last_event_timestamp(events) or rehydrate_time
      error = _REHYDRATION_INTERRUPTED_ERROR

    cmd = manifest.get("cmd")
    resumed_as = manifest.get("resumed_as")
    user_email = manifest.get("user_email")
    exit_code = manifest.get("exit_code")
    started_at = manifest.get("started_at")

    record = AutonomousTask(
      task_id=task_id,
      control_run_id=control_run_id,
      user_id=user_id,
      user_email=user_email if isinstance(user_email, str) else None,
      profile=profile,
      mode=mode,
      task=self._coerce_manifest_str(manifest, "task"),
      skill=self._coerce_manifest_str(manifest, "skill"),
      context=self._coerce_manifest_str(manifest, "context"),
      ticker=self._coerce_manifest_str(manifest, "ticker"),
      channel=self._coerce_manifest_str(manifest, "channel"),
      dev_mode=bool(manifest.get("dev_mode", False)),
      cmd=[str(part) for part in cmd] if isinstance(cmd, list) else [],
      log_path=self._path_from_manifest(
        manifest,
        "log_path",
        fallback=manifest_path.with_name(f"{task_id}.log"),
      )
      or manifest_path.with_name(f"{task_id}.log"),
      events_path=events_path,
      operator_inbox_path=self._path_from_manifest(
        manifest,
        "operator_inbox_path",
        fallback=manifest_path.with_name(f"{task_id}.operator-messages.jsonl"),
      ),
      approval_decisions_path=self._path_from_manifest(
        manifest,
        "approval_decisions_path",
        fallback=manifest_path.with_name(f"{task_id}.approval-decisions.jsonl"),
      ),
      started_at=float(started_at) if isinstance(started_at, (int, float)) else rehydrate_time,
      state=raw_state,
      exit_code=int(exit_code) if isinstance(exit_code, int) else None,
      error=error,
      proc=None,
      reaper_task=None,
      events_tail_task=None,
      completed_at=completed_at,
      log_handle=None,
      slot_reserved=False,
      event_lines=events,
      resumed_from=self._coerce_manifest_str(manifest, "resumed_from"),
      resumed_as=[str(item) for item in resumed_as] if isinstance(resumed_as, list) else [],
    )
    self._attach_manifest_tracking(record)
    if was_interrupted or record.state != raw_manifest_state:
      self._write_task_manifest(record)
    return record

  def rehydrate(self) -> None:
    rehydrate_time = time.time()
    for manifest_path in self._manifest_paths():
      try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      except (OSError, json.JSONDecodeError):
        _LOGGER.warning("Skipping unreadable autonomous task manifest: %s", manifest_path, exc_info=True)
        continue
      if not isinstance(manifest, dict):
        _LOGGER.warning("Skipping malformed autonomous task manifest: %s", manifest_path)
        continue

      record = self._task_from_manifest(
        manifest,
        manifest_path=manifest_path,
        rehydrate_time=rehydrate_time,
      )
      if record is None:
        continue
      if record.task_id in self._tasks:
        _LOGGER.warning("Skipping duplicate autonomous task manifest for %s", record.task_id)
        continue
      self._tasks[record.task_id] = record

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
    event_copy = dict(event)
    event_copy.setdefault("run_id", record.control_run_id)
    event_copy.setdefault("control_run_id", record.control_run_id)
    return event_copy

  def _replay_seed_events_for_record(self, record: AutonomousTask) -> list[dict[str, Any]]:
    return [
      self._event_for_record(record, event)
      for event in record.event_lines or []
      if isinstance(event, dict)
    ]

  def _record_replay_buffer_terminated(self, record: AutonomousTask) -> bool:
    if record.state in _TERMINAL_AUTONOMOUS_STATES or record.state == "finished":
      return True
    if record.proc is not None and record.proc.returncode is None:
      return False
    return record.state not in _REHYDRATED_ACTIVE_STATES and record.state != "starting"

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

  def _event_already_recorded(self, record: AutonomousTask, event: dict[str, Any]) -> bool:
    duplicate_key = self._event_duplicate_key(event)
    if duplicate_key is None:
      return False
    for existing in record.event_lines or ():
      if self._event_duplicate_key(existing) == duplicate_key:
        return True
    return False

  def _event_file_already_recorded(self, record: AutonomousTask, event: dict[str, Any]) -> bool:
    duplicate_key = self._event_duplicate_key(event)
    if duplicate_key is None or record.events_path is None:
      return False
    try:
      with record.events_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
          try:
            existing = json.loads(line)
          except json.JSONDecodeError:
            continue
          if isinstance(existing, dict) and self._event_duplicate_key(existing) == duplicate_key:
            return True
    except FileNotFoundError:
      return False
    return False

  def _append_event_to_events_file(self, record: AutonomousTask, event: dict[str, Any]) -> None:
    if record.events_path is None:
      return
    if self._event_file_already_recorded(record, event):
      return
    record.events_path.parent.mkdir(parents=True, exist_ok=True)
    with record.events_path.open("a", encoding="utf-8", buffering=1) as handle:
      handle.write(json.dumps(event, default=str) + "\n")

  def _operator_inbox_record_for_message_id(
    self,
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

  def _parent_message_event(
    self,
    record: AutonomousTask,
    *,
    message_id: str,
    text: str,
    user_id: str,
    sent_at: float,
  ) -> dict[str, Any]:
    inbox_record = self._operator_inbox_record_for_message_id(record, message_id)
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
    return self._event_for_record(
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
