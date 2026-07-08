from __future__ import annotations

import asyncio
import importlib
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

_AUTONOMOUS_TASK_ID_RE = re.compile(r"^bg_\d+$")
_AUTONOMOUS_RUN_FILE_RE = re.compile(r"^bg_(\d+)\..+")
_AUTONOMOUS_MANIFEST_FILE_RE = re.compile(r"^bg_(\d+)\.task\.json$")
_ACTIVE_AUTONOMOUS_PROCESS_STATES = {"starting", "queued", "waiting", "running", "approval_pending", "remediating"}
_REHYDRATED_ACTIVE_STATES = {"running", "approval_pending", "queued", "waiting", "remediating"}
_TERMINAL_AUTONOMOUS_STATES = {
  "completed",
  "failed",
  "killed",
  "interrupted",
  "budget_limited",
  "budget_exceeded",
  "blocked",
}
_REHYDRATION_INTERRUPTED_ERROR = "gateway restarted while run was active"
_REHYDRATE_EVENTS_SIZE_CAP_BYTES = 5 * 1024 * 1024
_REHYDRATE_EVENTS_TAIL_LINES = 2000
_TASK_MANIFEST_VERSION = 2
_RUN_SEQUENCE_CURSOR_FILE = ".autonomous-sequence.json"
_RUN_RETENTION_DAYS_ENV = "AGENT_AUTONOMOUS_RUN_RETENTION_DAYS"
_RUN_RETENTION_SECONDS_PER_DAY = 86400.0
_LOGGER = logging.getLogger("agent_gateway.autonomous_runner")
_DISPATCH_SCOPE_KEYS = frozenset({"kind", "source", "portfolio_name", "portfolio_id", "display_name"})


def _runtime_module() -> Any:
  for module_name in ("agent_gateway.autonomous_runner", "autonomous_runner"):
    module = sys.modules.get(module_name)
    if module is not None:
      return module
  return sys.modules[__name__]


def _runtime_attr(name: str, default: Any) -> Any:
  return getattr(_runtime_module(), name, default)


def _time_time() -> float:
  return _runtime_attr("time", time).time()


def _os_replace(src: Path, dst: Path) -> None:
  _runtime_attr("os", os).replace(src, dst)


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
  dispatch_scope: dict[str, Any] | None
  cmd: list[str]
  log_path: Path
  events_path: Path | None
  operator_inbox_path: Path | None
  approval_decisions_path: Path | None
  started_at: float
  state: str = "running"
  exit_code: int | None = None
  error: str | None = None
  proc: Any | None = None
  reaper_task: Any | None = None
  events_tail_task: Any | None = None
  completed_at: float | None = None
  log_handle: Any | None = None
  slot_reserved: bool = False
  event_lines: list[dict[str, Any]] | None = None
  delivered_messages: set[str] = field(default_factory=set)
  owner_user_id: str | None = None
  raw_user_id: str | None = None
  user_slug: str | None = None
  risk_user_id: int = 0
  user_aliases: list[str] = field(default_factory=list)
  identity_status: str = "legacy_user_id_fallback"
  operator_message_lock: Any = field(default_factory=lambda: _runtime_attr("asyncio", asyncio).Lock())
  resume_lock: Any = field(default_factory=lambda: _runtime_attr("asyncio", asyncio).Lock())
  resumed_from: str | None = None
  resumed_as: list[str] = field(default_factory=list)
  schedule_id: str | None = None
  schedule_name: str | None = None

  def __post_init__(self) -> None:
    if self.owner_user_id is None:
      self.owner_user_id = self.user_id
    if self.raw_user_id is None:
      self.raw_user_id = self.user_id
    if not self.user_aliases:
      self.user_aliases = _normalize_identity_aliases(
        self.owner_user_id,
        self.raw_user_id,
        self.user_slug,
        self.user_email,
      )

  @property
  def elapsed_sec(self) -> int:
    end_time = self.completed_at if self.completed_at is not None else _time_time()
    return max(0, int(end_time - self.started_at))


def _positive_int(value: Any) -> int | None:
  if isinstance(value, bool):
    return None
  if isinstance(value, int):
    return value if value > 0 else None
  if isinstance(value, str):
    cleaned = value.strip()
    if cleaned.isdecimal():
      parsed = int(cleaned)
      return parsed if parsed > 0 else None
  return None


def _normalize_identity_str(value: Any) -> str | None:
  if isinstance(value, bool):
    return None
  cleaned = str(value or "").strip()
  return cleaned or None


def _normalize_identity_aliases(*values: Any) -> list[str]:
  aliases: list[str] = []
  for value in values:
    if isinstance(value, (list, tuple)):
      for item in value:
        normalized_item = _normalize_identity_str(item)
        if normalized_item is not None and normalized_item not in aliases:
          aliases.append(normalized_item)
      continue
    normalized = _normalize_identity_str(value)
    if normalized is not None and normalized not in aliases:
      aliases.append(normalized)
  return aliases


def _normalize_dispatch_scope(value: Any) -> dict[str, Any] | None:
  if value is None:
    return None
  if not isinstance(value, dict):
    return None
  if set(value) - _DISPATCH_SCOPE_KEYS:
    return None
  if value.get("kind") != "portfolio":
    return None
  if value.get("source") not in {"active_default", "user_selected"}:
    return None
  portfolio_name = value.get("portfolio_name")
  if not isinstance(portfolio_name, str) or not portfolio_name.strip() or len(portfolio_name) > 256:
    return None
  normalized: dict[str, Any] = {
    "kind": "portfolio",
    "source": value["source"],
    "portfolio_name": portfolio_name,
    "portfolio_id": None,
    "display_name": None,
  }
  for key in ("portfolio_id", "display_name"):
    raw = value.get(key)
    if raw is None:
      continue
    if not isinstance(raw, str) or not raw.strip() or len(raw) > 256:
      return None
    normalized[key] = raw
  return normalized


def _user_identity_api() -> Any | None:
  api_dir = Path(__file__).resolve().parents[3] / "api"
  if api_dir.exists() and str(api_dir) not in sys.path:
    sys.path.insert(0, str(api_dir))
  try:
    return importlib.import_module("user_identity")
  except ModuleNotFoundError as exc:
    if exc.name not in {"user_identity", "api"}:
      raise
    try:
      return importlib.import_module("api.user_identity")
    except ModuleNotFoundError as nested_exc:
      if nested_exc.name not in {"user_identity", "api"}:
        raise
      return None


def _fallback_identity_payload(
  *,
  user_id: str,
  user_email: str | None,
  identity_status: str,
  risk_user_id: Any = None,
  owner_user_id: Any = None,
  raw_user_id: Any = None,
  user_slug: Any = None,
  user_aliases: Any = None,
) -> dict[str, Any]:
  normalized_owner = _normalize_identity_str(owner_user_id) or _normalize_identity_str(user_id) or ""
  normalized_raw = _normalize_identity_str(raw_user_id) or _normalize_identity_str(user_id)
  normalized_slug = _normalize_identity_str(user_slug)
  normalized_risk = _positive_int(risk_user_id) or _positive_int(normalized_owner) or 0
  aliases = _normalize_identity_aliases(
    normalized_owner,
    normalized_raw,
    normalized_slug,
    user_email,
    user_aliases if isinstance(user_aliases, list) else None,
  )
  return {
    "owner_user_id": normalized_owner,
    "raw_user_id": normalized_raw,
    "user_slug": normalized_slug,
    "risk_user_id": normalized_risk,
    "user_aliases": aliases,
    "identity_status": identity_status,
  }


def _manifest_identity_payload(manifest: dict[str, Any], *, user_id: str, user_email: str | None) -> dict[str, Any]:
  manifest_version = manifest.get("manifest_version")
  current_version = _runtime_attr("_TASK_MANIFEST_VERSION", _TASK_MANIFEST_VERSION)
  if manifest_version == current_version:
    owner_user_id = _normalize_identity_str(manifest.get("owner_user_id")) or user_id
    return _fallback_identity_payload(
      user_id=owner_user_id,
      user_email=user_email,
      identity_status=_normalize_identity_str(manifest.get("identity_status")) or "manifest_v2",
      risk_user_id=manifest.get("risk_user_id"),
      owner_user_id=owner_user_id,
      raw_user_id=manifest.get("raw_user_id") or user_id,
      user_slug=manifest.get("user_slug"),
      user_aliases=manifest.get("user_aliases"),
    )

  api = _user_identity_api()
  resolver = getattr(api, "resolve_canonical_user_identity", None) if api is not None else None
  if callable(resolver):
    try:
      identity = resolver(
        user_id,
        risk_user_id=manifest.get("risk_user_id"),
        user_email=user_email,
        mapped_slug=manifest.get("user_slug"),
        allow_legacy_fallback=True,
      )
      status = str(identity.identity_status)
      if status == "legacy_user_id_fallback" and _positive_int(user_id) is None:
        status = "legacy_unresolved"
      return {
        "owner_user_id": str(identity.owner_user_id),
        "raw_user_id": user_id,
        "user_slug": identity.user_slug,
        "risk_user_id": int(identity.risk_user_id),
        "user_aliases": [str(alias) for alias in identity.aliases],
        "identity_status": status,
      }
    except ValueError:
      pass

  raw_numeric = _positive_int(user_id)
  if raw_numeric is not None:
    return _fallback_identity_payload(
      user_id=str(raw_numeric),
      user_email=user_email,
      identity_status="numeric_user_id",
      risk_user_id=raw_numeric,
      owner_user_id=str(raw_numeric),
      raw_user_id=user_id,
    )
  return _fallback_identity_payload(
    user_id=user_id,
    user_email=user_email,
    identity_status="legacy_unresolved",
    raw_user_id=user_id,
    user_slug=user_id,
  )


class AutonomousRegistryStateMixin:
  def _initial_task_seq(self) -> int:
    if not self._log_dir.exists():
      return 0
    max_existing = -1
    try:
      for path in self._log_dir.iterdir():
        if not path.is_file():
          continue
        match = _runtime_attr("_AUTONOMOUS_RUN_FILE_RE", _AUTONOMOUS_RUN_FILE_RE).match(path.name)
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
    return self._log_dir / _runtime_attr("_RUN_SEQUENCE_CURSOR_FILE", _RUN_SEQUENCE_CURSOR_FILE)

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
      _os_replace(tmp_path, path)
      return True
    except Exception:
      try:
        tmp_path.unlink()
      except OSError:
        pass
      _LOGGER.warning("Failed to write autonomous run sequence cursor: %s", path, exc_info=True)
      return False

  def _configured_run_retention_days(self) -> float | None:
    env_name = _runtime_attr("_RUN_RETENTION_DAYS_ENV", _RUN_RETENTION_DAYS_ENV)
    raw = os.getenv(env_name, "").strip()
    if not raw:
      return None
    try:
      days = float(raw)
    except ValueError:
      _LOGGER.warning("Ignoring invalid %s=%r; expected positive day count", env_name, raw)
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
    if manifest.get("manifest_version") not in {1, _runtime_attr("_TASK_MANIFEST_VERSION", _TASK_MANIFEST_VERSION)}:
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

    cutoff_ts = _time_time() - (
      retention_days * _runtime_attr("_RUN_RETENTION_SECONDS_PER_DAY", _RUN_RETENTION_SECONDS_PER_DAY)
    )
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
      if not task_id or _runtime_attr("_AUTONOMOUS_TASK_ID_RE", _AUTONOMOUS_TASK_ID_RE).fullmatch(task_id) is None:
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
    owner_user_id = _normalize_identity_str(record.owner_user_id) or record.user_id
    raw_user_id = _normalize_identity_str(record.raw_user_id) or record.user_id
    return {
      "manifest_version": _runtime_attr("_TASK_MANIFEST_VERSION", _TASK_MANIFEST_VERSION),
      "task_id": record.task_id,
      "control_run_id": record.control_run_id,
      "owner_user_id": owner_user_id,
      "user_id": owner_user_id,
      "raw_user_id": raw_user_id,
      "user_slug": record.user_slug,
      "risk_user_id": record.risk_user_id,
      "user_email": record.user_email,
      "user_aliases": list(record.user_aliases),
      "identity_status": record.identity_status,
      "profile": record.profile,
      "mode": record.mode,
      "task": record.task,
      "skill": record.skill,
      "context": record.context,
      "ticker": record.ticker,
      "channel": record.channel,
      "dev_mode": record.dev_mode,
      "dispatch_scope": _normalize_dispatch_scope(record.dispatch_scope),
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
      "schedule_id": record.schedule_id,
      "schedule_name": record.schedule_name,
    }

  def _attach_manifest_tracking(self, record: AutonomousTask) -> None:
    if isinstance(record.resumed_as, _runtime_attr("_ManifestTrackedList", _ManifestTrackedList)):
      return
    record.resumed_as = _runtime_attr("_ManifestTrackedList", _ManifestTrackedList)(
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
      _os_replace(tmp_path, manifest_path)
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
          if path.is_file() and _runtime_attr("_AUTONOMOUS_MANIFEST_FILE_RE", _AUTONOMOUS_MANIFEST_FILE_RE).match(path.name)
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
      if stat.st_size <= _runtime_attr("_REHYDRATE_EVENTS_SIZE_CAP_BYTES", _REHYDRATE_EVENTS_SIZE_CAP_BYTES):
        with events_path.open("r", encoding="utf-8", errors="replace") as handle:
          return self._parse_event_lines(handle.readlines(), path=events_path)

      _LOGGER.warning(
        "Autonomous events file exceeds rehydrate cap; loading tail only: %s",
        events_path,
      )
      recent = deque(maxlen=_runtime_attr("_REHYDRATE_EVENTS_TAIL_LINES", _REHYDRATE_EVENTS_TAIL_LINES))
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
    manifest_version = manifest.get("manifest_version")
    if manifest_version not in {1, _runtime_attr("_TASK_MANIFEST_VERSION", _TASK_MANIFEST_VERSION)}:
      _LOGGER.warning("Skipping autonomous manifest with unsupported version: %s", manifest_path)
      return None

    task_id = self._coerce_manifest_str(manifest, "task_id")
    control_run_id = self._coerce_manifest_str(manifest, "control_run_id")
    user_id = self._coerce_manifest_str(manifest, "user_id")
    profile = self._coerce_manifest_str(manifest, "profile")
    mode = self._coerce_manifest_str(manifest, "mode")
    if (
      not task_id
      or _runtime_attr("_AUTONOMOUS_TASK_ID_RE", _AUTONOMOUS_TASK_ID_RE).fullmatch(task_id) is None
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
    terminal_states = _runtime_attr("_TERMINAL_AUTONOMOUS_STATES", _TERMINAL_AUTONOMOUS_STATES)
    rehydrated_active_states = _runtime_attr("_REHYDRATED_ACTIVE_STATES", _REHYDRATED_ACTIVE_STATES)
    if raw_state in rehydrated_active_states or raw_state not in terminal_states:
      raw_state = "interrupted"
      was_interrupted = True
      completed_at = self._last_event_timestamp(events) or rehydrate_time
      error = _runtime_attr("_REHYDRATION_INTERRUPTED_ERROR", _REHYDRATION_INTERRUPTED_ERROR)

    cmd = manifest.get("cmd")
    resumed_as = manifest.get("resumed_as")
    user_email = manifest.get("user_email")
    user_email = user_email if isinstance(user_email, str) else None
    identity = _manifest_identity_payload(manifest, user_id=user_id, user_email=user_email)
    exit_code = manifest.get("exit_code")
    started_at = manifest.get("started_at")

    record = AutonomousTask(
      task_id=task_id,
      control_run_id=control_run_id,
      user_id=str(identity["owner_user_id"]),
      user_email=user_email,
      profile=profile,
      mode=mode,
      task=self._coerce_manifest_str(manifest, "task"),
      skill=self._coerce_manifest_str(manifest, "skill"),
      context=self._coerce_manifest_str(manifest, "context"),
      ticker=self._coerce_manifest_str(manifest, "ticker"),
      channel=self._coerce_manifest_str(manifest, "channel"),
      dev_mode=bool(manifest.get("dev_mode", False)),
      dispatch_scope=_normalize_dispatch_scope(manifest.get("dispatch_scope")),
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
      owner_user_id=str(identity["owner_user_id"]),
      raw_user_id=identity["raw_user_id"],
      user_slug=identity["user_slug"],
      risk_user_id=int(identity["risk_user_id"]),
      user_aliases=list(identity["user_aliases"]),
      identity_status=str(identity["identity_status"]),
      resumed_from=self._coerce_manifest_str(manifest, "resumed_from"),
      resumed_as=[str(item) for item in resumed_as] if isinstance(resumed_as, list) else [],
      schedule_id=self._coerce_manifest_str(manifest, "schedule_id"),
      schedule_name=self._coerce_manifest_str(manifest, "schedule_name"),
    )
    self._attach_manifest_tracking(record)
    if (
      was_interrupted
      or record.state != raw_manifest_state
      or manifest_version != _runtime_attr("_TASK_MANIFEST_VERSION", _TASK_MANIFEST_VERSION)
    ):
      self._write_task_manifest(record)
    return record

  def rehydrate(self) -> None:
    rehydrate_time = _time_time()
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


__all__ = [
  "AutonomousRegistryStateMixin",
  "AutonomousTask",
]
