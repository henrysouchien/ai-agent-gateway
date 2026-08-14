from __future__ import annotations

import asyncio
import errno
import fcntl
import importlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import stat
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .autonomous_event_channel import (
  AutonomousEventAcknowledgement,
  AutonomousEventChannelParent,
  AutonomousEventRecord,
  ReceivedAutonomousEventStream,
)
from .autonomous_approval_channel import (
  AutonomousApprovalChannelParent,
)
from .autonomous_launch_envelope import AutonomousControlAuthority
from .autonomous_claim_broker import AutonomousClaimBroker
from .capability_binding import CapabilityBind
from .role_validation import require_exact_role

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
_TASK_MANIFEST_VERSION = 7
_RUN_SEQUENCE_CURSOR_FILE = ".autonomous-sequence.json"
_LOGGER = logging.getLogger("agent_gateway.autonomous_runner")
_DISPATCH_SCOPE_KEYS = frozenset({"kind", "source", "portfolio_name", "portfolio_id", "display_name"})
_ROOT_TERMINAL_EVENT_TYPES = frozenset({"error", "stream_error", "stream_complete"})
_AUTONOMOUS_TERMINAL_REASONS = frozenset({"writer_lease_already_held"})


def is_root_run_event(event: Any) -> bool:
  """Return whether an event belongs to the parent run rather than a child."""
  return isinstance(event, dict) and event.get("sub_agent_id") is None


def is_root_terminal_event(event: Any) -> bool:
  """Return whether an event may settle the parent autonomous run."""
  return (
    is_root_run_event(event)
    and event.get("type") in _ROOT_TERMINAL_EVENT_TYPES
  )


def _runtime_module() -> Any:
  for module_name in ("agent_gateway.autonomous_runner", "autonomous_runner"):
    module = sys.modules.get(module_name)
    if module is not None:
      return module
  return sys.modules[__name__]


@dataclass
class _SpillCleanupLease:
  directory: Path
  fd: int
  directory_fd: int

  def is_current(self) -> bool:
    try:
      held_directory = os.fstat(self.directory_fd)
      visible_directory = os.lstat(self.directory)
      held_lease = os.fstat(self.fd)
      visible_lease = os.stat(".lease", dir_fd=self.directory_fd, follow_symlinks=False)
    except OSError:
      return False
    return (
      stat.S_ISDIR(held_directory.st_mode)
      and stat.S_ISDIR(visible_directory.st_mode)
      and (held_directory.st_dev, held_directory.st_ino)
      == (visible_directory.st_dev, visible_directory.st_ino)
      and stat.S_ISREG(held_lease.st_mode)
      and stat.S_ISREG(visible_lease.st_mode)
      and (held_lease.st_dev, held_lease.st_ino) == (visible_lease.st_dev, visible_lease.st_ino)
    )

  def close(self) -> None:
    try:
      fcntl.flock(self.fd, fcntl.LOCK_UN)
    except OSError:
      pass
    try:
      os.close(self.fd)
    finally:
      os.close(self.directory_fd)


def autonomous_owner_lease_is_released(
  path: Path,
  *,
  expected_device: int,
  expected_inode: int,
) -> bool:
  """Return whether the exact sentinel-held run ownership lease is free."""
  flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
  if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
  try:
    fd = os.open(path, flags)
  except OSError as exc:
    raise RuntimeError(
      "autonomous owner lease is unavailable"
    ) from exc
  try:
    held = os.fstat(fd)
    visible = os.lstat(path)
    if (
      not stat.S_ISREG(held.st_mode)
      or not stat.S_ISREG(visible.st_mode)
      or stat.S_IMODE(held.st_mode) != 0o600
      or stat.S_IMODE(visible.st_mode) != 0o600
      or held.st_nlink != 1
      or visible.st_nlink != 1
      or held.st_uid != os.geteuid()
      or visible.st_uid != os.geteuid()
      or (held.st_dev, held.st_ino)
      != (expected_device, expected_inode)
      or (visible.st_dev, visible.st_ino)
      != (expected_device, expected_inode)
    ):
      raise RuntimeError(
        "autonomous owner lease identity changed"
      )
    try:
      fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
      if exc.errno in {errno.EACCES, errno.EAGAIN}:
        return False
      raise RuntimeError(
        "autonomous owner lease could not be inspected"
      ) from exc
    return True
  finally:
    os.close(fd)


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
  session_id: str
  channel_id: str
  user_id: str
  user_email: str | None
  profile: str
  mode: str
  task: str | None
  skill: str | None
  pack: str | None
  deliver: bool
  context: str | None
  ticker: str | None
  channel: str | None
  dev_mode: bool
  dispatch_scope: dict[str, Any] | None
  cmd: list[str]
  log_path: Path
  operator_inbox_path: Path | None
  approval_decisions_path: Path | None
  control_authority: AutonomousControlAuthority
  owner_lease_path: Path
  owner_lease_device: int
  owner_lease_inode: int
  started_at: float
  events_path: Path | None = None
  events_device: int | None = None
  events_inode: int | None = None
  events_evidence_status: str = "missing"
  manifest_version: int = _TASK_MANIFEST_VERSION
  max_budget_usd: float | None = None
  state: str = "running"
  exit_code: int | None = None
  error: str | None = None
  terminal_reason: str | None = None
  proc: Any | None = None
  reaper_task: Any | None = None
  sentinel_status_task: Any | None = None
  event_channel: AutonomousEventChannelParent | None = None
  event_channel_task: Any | None = None
  event_channel_records: list[AutonomousEventRecord] = field(default_factory=list)
  event_channel_projected_events: list[dict[str, Any]] = field(default_factory=list)
  event_channel_stream: ReceivedAutonomousEventStream | None = None
  event_channel_acknowledgement: AutonomousEventAcknowledgement | None = None
  event_channel_ack_started: bool = False
  cancellation_requested: bool = False
  process_group_pid: int | None = None
  process_group_id: int | None = None
  owner_lifeline_fd: int | None = None
  completed_at: float | None = None
  terminal_manifest_committed: bool = False
  approval_delivery_quarantined: bool = False
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
  role: str = field(kw_only=True)
  operator_message_lock: Any = field(default_factory=lambda: _runtime_attr("asyncio", asyncio).Lock())
  event_record_lock: Any = field(default_factory=lambda: _runtime_attr("asyncio", asyncio).Lock())
  approval_decision_lock: Any = field(default_factory=lambda: _runtime_attr("asyncio", asyncio).Lock())
  resume_lock: Any = field(default_factory=lambda: _runtime_attr("asyncio", asyncio).Lock())
  resumed_from: str | None = None
  resumed_as: list[str] = field(default_factory=list)
  schedule_id: str | None = None
  schedule_name: str | None = None
  tool_result_spill_dir: Path | None = None
  capability_bind: CapabilityBind | None = None
  claim_broker: AutonomousClaimBroker | None = None
  launch_nonce: str | None = None
  approval_channel: AutonomousApprovalChannelParent | None = None

  def __post_init__(self) -> None:
    if self.mode not in {"once", "task", "skill", "pack"}:
      raise ValueError("autonomous task mode must be once, task, skill, or pack")
    if (
      type(self.control_run_id) is not str
      or not self.control_run_id
      or self.control_run_id != self.control_run_id.strip()
      or len(self.control_run_id) > 512
      or any(
        ord(character) < 0x20
        for character in self.control_run_id
      )
    ):
      raise ValueError(
        "autonomous task control_run_id must be canonical"
      )
    if (
      not isinstance(self.session_id, str)
      or not self.session_id
      or self.session_id != self.session_id.strip()
      or self.session_id != self.task_id
    ):
      raise ValueError("autonomous task session_id must be canonical")
    if (
      not isinstance(self.channel_id, str)
      or re.fullmatch(r"[0-9a-f]{64}", self.channel_id) is None
    ):
      raise ValueError("autonomous task channel_id is invalid")
    if type(self.control_authority) is not AutonomousControlAuthority:
      raise ValueError(
        "autonomous task requires exact signed control authority"
      )
    if self.control_authority.control_mode != "file":
      raise ValueError(
        "ordinary autonomous task requires file control authority"
      )
    if (
      not isinstance(self.owner_lease_path, Path)
      or not self.owner_lease_path.is_absolute()
      or isinstance(self.owner_lease_device, bool)
      or not isinstance(self.owner_lease_device, int)
      or self.owner_lease_device < 0
      or isinstance(self.owner_lease_inode, bool)
      or not isinstance(self.owner_lease_inode, int)
      or self.owner_lease_inode <= 0
    ):
      raise ValueError(
        "autonomous task requires exact owner lease authority"
      )
    if (
      self.operator_inbox_path is None
      or str(self.operator_inbox_path)
      != self.control_authority.operator_inbox_path
      or (
        str(self.approval_decisions_path)
        if self.approval_decisions_path is not None
        else None
      )
      != self.control_authority.approval_decisions_path
    ):
      raise ValueError(
        "autonomous task control paths do not match signed authority"
      )
    if self.launch_nonce is not None and (
      type(self.launch_nonce) is not str
      or re.fullmatch(r"[0-9a-f]{32}", self.launch_nonce) is None
    ):
      raise ValueError("autonomous task launch_nonce is invalid")
    if (
      self.approval_channel is not None
      and type(self.approval_channel)
      is not AutonomousApprovalChannelParent
    ):
      raise ValueError(
        "autonomous task approval channel must be exact or None"
      )
    if (
      self.approval_channel is not None
      and self.launch_nonce is None
    ):
      raise ValueError(
        "autonomous task approval channel requires launch_nonce"
      )
    if not isinstance(self.deliver, bool):
      raise ValueError("autonomous task deliver must be a bool")
    if self.pack is not None and (
      not isinstance(self.pack, str)
      or not self.pack
      or self.pack != self.pack.strip()
    ):
      raise ValueError("autonomous task pack must be a normalized non-empty string")
    if self.mode == "pack":
      if self.pack is None:
        raise ValueError("autonomous pack task requires pack")
      if (
        self.task is not None
        or self.skill is not None
        or self.context is not None
        or self.ticker is not None
        or self.dev_mode
        or self.max_budget_usd is not None
      ):
        raise ValueError("autonomous pack task has incompatible launch fields")
    elif self.pack is not None:
      raise ValueError("autonomous task pack is only valid for mode='pack'")
    if not self.deliver and self.mode != "skill":
      raise ValueError("autonomous task deliver=False is only valid for mode='skill'")
    if self.owner_user_id is None:
      self.owner_user_id = self.user_id
    self.role = require_exact_role(self.role)
    if self.raw_user_id is None:
      self.raw_user_id = self.user_id
    if not self.user_aliases:
      self.user_aliases = _normalize_identity_aliases(
        self.owner_user_id,
        self.raw_user_id,
        self.user_slug,
        self.user_email,
      )
    if self.capability_bind is not None:
      if self.capability_bind.capability_id != "session.driver":
        raise ValueError("autonomous task requires a session.driver capability bind")
      if self.capability_bind.run_mode not in {"autonomous", "cron"}:
        raise ValueError("autonomous task bind must use autonomous or cron run mode")
    if (
      isinstance(self.manifest_version, bool)
      or not isinstance(self.manifest_version, int)
      or self.manifest_version != _TASK_MANIFEST_VERSION
    ):
      raise ValueError("autonomous task manifest_version is unsupported")
    if self.capability_bind is None:
      raise ValueError(
        "v7 autonomous tasks require a complete capability bind"
      )
    if (
      self.terminal_reason is not None
      and (
        not isinstance(self.terminal_reason, str)
        or self.terminal_reason not in _AUTONOMOUS_TERMINAL_REASONS
      )
    ):
      raise ValueError("autonomous task terminal_reason is invalid")
    if self.terminal_reason is not None and (
      self.state not in {"completed", "finished"}
      or self.exit_code != 0
      or self.error is not None
    ):
      raise ValueError(
        "autonomous task terminal_reason requires successful completion"
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


def _positive_finite_float(value: Any) -> float | None:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return None
  normalized = float(value)
  return normalized if math.isfinite(normalized) and normalized > 0 else None


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


def _user_identity_api(*, api_dir: Path | None = None) -> Any | None:
  explicit_api_dir = api_dir is not None
  resolved_api_dir = (
    api_dir.resolve()
    if api_dir is not None
    else Path(__file__).resolve().parents[3] / "api"
  )
  expected_module_path = resolved_api_dir / "user_identity.py"
  if explicit_api_dir and not expected_module_path.is_file():
    return None
  if resolved_api_dir.exists() and str(resolved_api_dir) not in sys.path:
    sys.path.insert(0, str(resolved_api_dir))
  try:
    module = importlib.import_module("user_identity")
  except ModuleNotFoundError as exc:
    if exc.name not in {"user_identity", "api"}:
      raise
    if explicit_api_dir:
      return None
    try:
      module = importlib.import_module("api.user_identity")
    except ModuleNotFoundError as nested_exc:
      if nested_exc.name not in {"user_identity", "api"}:
        raise
      return None
  if explicit_api_dir:
    module_origin = getattr(getattr(module, "__spec__", None), "origin", None)
    if not isinstance(module_origin, str):
      module_origin = getattr(module, "__file__", None)
    if not isinstance(module_origin, str):
      return None
    try:
      if Path(module_origin).resolve() != expected_module_path.resolve():
        return None
    except OSError:
      return None
  return module


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
  owner_user_id = _normalize_identity_str(manifest.get("owner_user_id")) or user_id
  return _fallback_identity_payload(
    user_id=owner_user_id,
    user_email=user_email,
    identity_status=(
      _normalize_identity_str(manifest.get("identity_status"))
      or "manifest_v5"
    ),
    risk_user_id=manifest.get("risk_user_id"),
    owner_user_id=owner_user_id,
    raw_user_id=manifest.get("raw_user_id") or user_id,
    user_slug=manifest.get("user_slug"),
    user_aliases=manifest.get("user_aliases"),
  )


class AutonomousRegistryStateMixin:
  _skip_warned: set[str]

  def _warn_once(self, manifest_path: Path, message: str) -> None:
    """Log a manifest-skip once per path, not once per rehydrate cycle.

    Rehydrate runs repeatedly; an unreadable record is a standing condition,
    not a new event each pass. Re-logging it every cycle produced a 353 MB
    error log and buried real failures.
    """

    seen = getattr(self, "_skip_warned", None)
    if seen is None:
      seen = set()
      self._skip_warned = seen
    key = f"{message}\0{manifest_path}"
    if key in seen:
      return
    seen.add(key)
    _LOGGER.warning(message, manifest_path)

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

  def _expected_tool_result_spill_dir(self, task_id: str) -> Path:
    return self._log_dir.expanduser().resolve() / f"{task_id}.tool_result_spill"

  def _registered_tool_result_spill_dir(
    self,
    task_id: str,
    raw_path: Any,
    *,
    require_exists: bool,
  ) -> Path | None:
    if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
      return None
    path = Path(raw_path).expanduser()
    expected = self._expected_tool_result_spill_dir(task_id)
    try:
      if not path.is_absolute() or path != expected or path.resolve(strict=False) != expected:
        return None
      if require_exists:
        info = path.lstat()
        if path.is_symlink() or not stat.S_ISDIR(info.st_mode):
          return None
    except OSError:
      return None
    return path

  def _acquire_tool_result_spill_cleanup_lease(
    self,
    spill_dir: Path,
  ) -> _SpillCleanupLease | None:
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
      directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
      directory_flags |= os.O_NOFOLLOW
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
      flags |= os.O_NOFOLLOW
    directory_fd: int | None = None
    fd: int | None = None
    try:
      directory_fd = os.open(spill_dir, directory_flags)
      held_directory = os.fstat(directory_fd)
      visible_directory = os.lstat(spill_dir)
      if (
        not stat.S_ISDIR(held_directory.st_mode)
        or not stat.S_ISDIR(visible_directory.st_mode)
        or (held_directory.st_dev, held_directory.st_ino)
        != (visible_directory.st_dev, visible_directory.st_ino)
      ):
        raise OSError("spill directory identity mismatch")
      fd = os.open(".lease", flags, 0o600, dir_fd=directory_fd)
      fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
      held = os.fstat(fd)
      visible = os.stat(".lease", dir_fd=directory_fd, follow_symlinks=False)
      if (
        not stat.S_ISREG(held.st_mode)
        or not stat.S_ISREG(visible.st_mode)
        or (held.st_dev, held.st_ino) != (visible.st_dev, visible.st_ino)
      ):
        raise OSError("spill lease identity mismatch")
      lease = _SpillCleanupLease(
        directory=spill_dir,
        fd=fd,
        directory_fd=directory_fd,
      )
      if not lease.is_current():
        raise OSError("spill cleanup directory was replaced")
      return lease
    except OSError:
      if fd is not None:
        try:
          os.close(fd)
        except OSError:
          pass
      if directory_fd is not None:
        try:
          os.close(directory_fd)
        except OSError:
          pass
      return None

  def _remove_registered_tool_result_spill_dir(
    self,
    task_id: str,
    raw_path: Any,
    *,
    require_starting_manifest: bool = False,
  ) -> bool:
    spill_dir = self._registered_tool_result_spill_dir(
      task_id,
      raw_path,
      require_exists=True,
    )
    if spill_dir is None:
      return not Path(str(raw_path or "")).exists()
    lease = self._acquire_tool_result_spill_cleanup_lease(spill_dir)
    if lease is None:
      return False
    try:
      if require_starting_manifest:
        try:
          manifest = json.loads(self._manifest_path(task_id).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
          return False
        if (
          not isinstance(manifest, dict)
          or manifest.get("manifest_version") != _TASK_MANIFEST_VERSION
          or manifest.get("state") != "starting"
        ):
          return False
      if not lease.is_current():
        return False
      shutil.rmtree(spill_dir)
      return True
    except OSError:
      _LOGGER.warning("Failed to remove autonomous tool-result spill dir: %s", spill_dir, exc_info=True)
      return False
    finally:
      lease.close()

  def _cleanup_uncommitted_spill_starts(self) -> None:
    self._spill_start_cleanup_skipped: set[str] = set()
    for manifest_path in self._manifest_paths():
      try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      except (OSError, json.JSONDecodeError):
        continue
      if (
        not isinstance(manifest, dict)
        or manifest.get("manifest_version") != _TASK_MANIFEST_VERSION
        or manifest.get("state") != "starting"
      ):
        continue
      task_id = self._coerce_manifest_str(manifest, "task_id")
      if not task_id or _runtime_attr("_AUTONOMOUS_TASK_ID_RE", _AUTONOMOUS_TASK_ID_RE).fullmatch(task_id) is None:
        continue
      raw_path = manifest.get("tool_result_spill_dir")
      spill_dir = self._registered_tool_result_spill_dir(task_id, raw_path, require_exists=False)
      if spill_dir is None or not spill_dir.exists():
        continue
      if not self._remove_registered_tool_result_spill_dir(
        task_id,
        raw_path,
        require_starting_manifest=True,
      ):
        self._spill_start_cleanup_skipped.add(task_id)

  def _is_log_dir_child(self, path: Path) -> bool:
    try:
      log_dir = self._log_dir.resolve()
      resolved = path.expanduser().resolve()
    except OSError:
      return False
    return resolved == log_dir or log_dir in resolved.parents

  def _manifest_path(self, task_id: str) -> Path:
    return self._log_dir / f"{task_id}.task.json"

  def _manifest_payload(self, record: AutonomousTask) -> dict[str, Any]:
    owner_user_id = _normalize_identity_str(record.owner_user_id) or record.user_id
    raw_user_id = _normalize_identity_str(record.raw_user_id) or record.user_id
    payload = {
      "manifest_version": record.manifest_version,
      "task_id": record.task_id,
      "control_run_id": record.control_run_id,
      "session_id": record.session_id,
      "channel_id": record.channel_id,
      "owner_user_id": owner_user_id,
      "user_id": owner_user_id,
      "raw_user_id": raw_user_id,
      "user_slug": record.user_slug,
      "risk_user_id": record.risk_user_id,
      "user_email": record.user_email,
      "user_aliases": list(record.user_aliases),
      "identity_status": record.identity_status,
      "role": record.role,
      "profile": record.profile,
      "mode": record.mode,
      "task": record.task,
      "skill": record.skill,
      "pack": record.pack,
      "deliver": record.deliver,
      "context": record.context,
      "ticker": record.ticker,
      "channel": record.channel,
      "dev_mode": record.dev_mode,
      "max_budget_usd": record.max_budget_usd,
      "dispatch_scope": _normalize_dispatch_scope(record.dispatch_scope),
      "containment_expectation": {
        "platform": sys.platform,
        "expected_backend": (
          "landlock-v6"
          if sys.platform == "linux"
          else "darwin-degraded"
        ),
        "expected_degraded": sys.platform == "darwin",
      },
      "cmd": list(record.cmd),
      "log_path": str(record.log_path),
      "events_path": (
        str(record.events_path) if record.events_path is not None else None
      ),
      "operator_inbox_path": (
        str(record.operator_inbox_path) if record.operator_inbox_path is not None else None
      ),
      "approval_decisions_path": (
        str(record.approval_decisions_path) if record.approval_decisions_path is not None else None
      ),
      "owner_lease_path": str(record.owner_lease_path),
      "owner_lease_device": record.owner_lease_device,
      "owner_lease_inode": record.owner_lease_inode,
      "control_authority": record.control_authority.receipt(),
      "started_at": record.started_at,
      "state": record.state,
      "exit_code": record.exit_code,
      "error": record.error,
      "terminal_reason": record.terminal_reason,
      "completed_at": record.completed_at,
      "resumed_from": record.resumed_from,
      "resumed_as": list(record.resumed_as),
      "schedule_id": record.schedule_id,
      "schedule_name": record.schedule_name,
      "tool_result_spill_dir": (
        str(record.tool_result_spill_dir) if record.tool_result_spill_dir is not None else None
      ),
    }
    if record.manifest_version == _runtime_attr(
      "_TASK_MANIFEST_VERSION",
      _TASK_MANIFEST_VERSION,
    ):
      payload["capability_bind"] = (
        record.capability_bind.receipt()
        if record.capability_bind is not None
        else None
      )
    return payload

  def _attach_manifest_tracking(self, record: AutonomousTask) -> None:
    if isinstance(record.resumed_as, _runtime_attr("_ManifestTrackedList", _ManifestTrackedList)):
      return
    record.resumed_as = _runtime_attr("_ManifestTrackedList", _ManifestTrackedList)(
      list(record.resumed_as),
      lambda: self._write_task_manifest(record),
    )

  def _write_task_manifest(self, record: AutonomousTask, *, checked: bool = False) -> bool:
    manifest_path = self._manifest_path(record.task_id)
    tmp_path = manifest_path.with_name(f"{manifest_path.name}.{secrets.token_hex(8)}.tmp")
    tmp_created = False
    tmp_fd = -1
    directory_fd = -1
    try:
      manifest_path.parent.mkdir(parents=True, exist_ok=True)
      directory_fd = os.open(
        manifest_path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
      )
      tmp_fd = os.open(
        tmp_path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
      )
      tmp_created = True
      os.fchmod(tmp_fd, 0o600)
      tmp_stat = os.fstat(tmp_fd)
      if (
        not stat.S_ISREG(tmp_stat.st_mode)
        or stat.S_IMODE(tmp_stat.st_mode) != 0o600
        or tmp_stat.st_nlink != 1
        or tmp_stat.st_uid != os.geteuid()
      ):
        raise RuntimeError("autonomous task manifest temporary file is unsafe")
      encoded = (
        json.dumps(self._manifest_payload(record), sort_keys=True) + "\n"
      ).encode("utf-8")
      offset = 0
      while offset < len(encoded):
        written = os.write(tmp_fd, encoded[offset:])
        if written <= 0:
          raise RuntimeError("autonomous task manifest write was incomplete")
        offset += written
      os.fsync(tmp_fd)
      os.close(tmp_fd)
      tmp_fd = -1
      _os_replace(tmp_path, manifest_path)
      tmp_created = False
      os.fsync(directory_fd)
      return True
    except Exception:
      if tmp_fd >= 0:
        try:
          os.close(tmp_fd)
        except OSError:
          pass
      if tmp_created:
        try:
          tmp_path.unlink()
        except OSError:
          pass
      _LOGGER.warning(
        "Failed to write autonomous task manifest for %s",
        record.task_id,
        exc_info=True,
      )
      if checked:
        return False
      return False
    finally:
      if directory_fd >= 0:
        try:
          os.close(directory_fd)
        except OSError:
          pass

  def _delete_task_manifest(self, task_id: str) -> bool:
    manifest_path = self._manifest_path(task_id)
    directory_fd = -1
    try:
      directory_fd = os.open(
        manifest_path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
      )
      try:
        os.unlink(
          manifest_path.name,
          dir_fd=directory_fd,
        )
      except FileNotFoundError:
        # Still synchronize the directory. A caller may be retrying after an
        # earlier unlink succeeded but its directory fsync failed.
        pass
      os.fsync(directory_fd)
      return True
    except Exception:
      _LOGGER.warning(
        "Failed to durably delete autonomous task manifest for %s",
        task_id,
        exc_info=True,
      )
      return False
    finally:
      if directory_fd >= 0:
        try:
          os.close(directory_fd)
        except OSError:
          pass

  def _quarantine_task_manifest(self, task_id: str) -> bool:
    manifest_path = self._manifest_path(task_id)
    quarantine_path = manifest_path.with_name(
      f".{manifest_path.name}.{secrets.token_hex(8)}.quarantined"
    )
    directory_fd = -1
    try:
      directory_fd = os.open(
        manifest_path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
      )
      try:
        os.replace(
          manifest_path.name,
          quarantine_path.name,
          src_dir_fd=directory_fd,
          dst_dir_fd=directory_fd,
        )
      except FileNotFoundError:
        # As with deletion, a retry after a failed directory fsync must still
        # synchronize the already-completed namespace change.
        pass
      os.fsync(directory_fd)
      return True
    except Exception:
      _LOGGER.critical(
        "Failed to quarantine autonomous task manifest for fenced run %s",
        task_id,
        exc_info=True,
      )
      return False
    finally:
      if directory_fd >= 0:
        try:
          os.close(directory_fd)
        except OSError:
          pass

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

  @staticmethod
  def _event_timestamp(event: dict[str, Any]) -> float | None:
    for key in ("ts", "sent_at", "decided_at", "timestamp", "created_at"):
      raw = event.get(key)
      if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
      if isinstance(raw, str):
        try:
          return float(raw)
        except ValueError:
          continue
    return None

  def _last_event_timestamp(self, events: list[dict[str, Any]]) -> float | None:
    for event in reversed(events):
      timestamp = self._event_timestamp(event)
      if timestamp is not None:
        return timestamp
    return None

  def _parse_event_lines(
    self,
    lines: list[str],
    *,
    path: Path,
  ) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    skipped = 0
    for line in lines:
      stripped = line.strip()
      if not stripped:
        continue
      try:
        event = json.loads(stripped)
      except json.JSONDecodeError:
        skipped += 1
        continue
      if isinstance(event, dict):
        events.append(event)
      else:
        skipped += 1
    if skipped:
      _LOGGER.warning(
        "Skipped %d malformed autonomous event line(s) while rehydrating %s",
        skipped,
        path,
      )
    return events, skipped

  def _load_rehydrated_events(
    self,
    events_path: Path,
  ) -> tuple[list[dict[str, Any]], str]:
    flags = (
      os.O_RDONLY
      | getattr(os, "O_CLOEXEC", 0)
      | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
      fd = os.open(events_path, flags)
    except FileNotFoundError:
      return [], "missing"
    except OSError:
      _LOGGER.warning(
        "Failed to open autonomous events file for rehydration: %s",
        events_path,
        exc_info=True,
      )
      return [], "unreadable"
    try:
      file_stat = os.fstat(fd)
      if (
        not stat.S_ISREG(file_stat.st_mode)
        or file_stat.st_uid != os.geteuid()
        or file_stat.st_nlink != 1
      ):
        _LOGGER.warning(
          "Autonomous events file has unsafe read identity: %s",
          events_path,
        )
        return [], "unreadable"
      oversized = file_stat.st_size > _runtime_attr(
        "_REHYDRATE_EVENTS_SIZE_CAP_BYTES",
        _REHYDRATE_EVENTS_SIZE_CAP_BYTES,
      )
      with os.fdopen(fd, "r", encoding="utf-8", errors="replace") as handle:
        fd = -1
        if oversized:
          _LOGGER.warning(
            "Autonomous events file exceeds rehydrate cap; loading tail only: %s",
            events_path,
          )
          recent = deque(
            maxlen=_runtime_attr(
              "_REHYDRATE_EVENTS_TAIL_LINES",
              _REHYDRATE_EVENTS_TAIL_LINES,
            )
          )
          for line in handle:
            recent.append(line)
          lines = list(recent)
        else:
          lines = handle.readlines()
      events, skipped = self._parse_event_lines(lines, path=events_path)
      if oversized:
        return events, "tail_truncated"
      if skipped:
        return events, "partial_malformed"
      return events, "complete"
    except OSError:
      _LOGGER.warning(
        "Failed to load autonomous events for rehydration: %s",
        events_path,
        exc_info=True,
      )
      return [], "unreadable"
    finally:
      if fd >= 0:
        os.close(fd)

  @staticmethod
  def _terminal_event_outcome(
    events: list[dict[str, Any]] | None,
  ) -> tuple[str | None, str | None, str | None]:
    for event in events or []:
      if not is_root_terminal_event(event):
        continue
      event_type = event.get("type")
      terminal_reason = event.get("terminal_reason")
      if (
        terminal_reason is not None
        and (
          not isinstance(terminal_reason, str)
          or terminal_reason not in _AUTONOMOUS_TERMINAL_REASONS
        )
      ):
        return (
          "failed",
          f"Invalid autonomous terminal_reason: {terminal_reason!r}",
          None,
        )
      if event_type in {"error", "stream_error"}:
        if terminal_reason is not None:
          return (
            "failed",
            "Error terminal cannot carry autonomous terminal_reason",
            None,
          )
        return (
          "failed",
          str(event.get("error") or event.get("message") or event_type),
          None,
        )
      if event_type != "stream_complete":
        continue
      disposition = event.get("terminal_disposition")
      if disposition == "completed":
        return "completed", None, terminal_reason
      if disposition == "interrupted":
        if terminal_reason is not None:
          return (
            "failed",
            "Interrupted terminal cannot carry autonomous terminal_reason",
            None,
          )
        if event.get("reason") == "budget_exceeded":
          return "budget_limited", None, None
        return "interrupted", None, None
      return (
        "failed",
        f"Invalid stream_complete terminal_disposition: {disposition!r}",
        None,
      )
    return None, "Process exited without a terminal stream event", None

  def _apply_terminal_event_state(self, record: AutonomousTask) -> bool:
    if record.state not in {"completed", "finished", "failed"}:
      return False

    prior = (record.state, record.error, record.terminal_reason)
    outcome, terminal_error, terminal_reason = self._terminal_event_outcome(
      record.event_channel_projected_events
    )
    if outcome == "budget_limited":
      record.state = "budget_limited"
      record.error = None
      record.terminal_reason = None
    elif outcome == "interrupted":
      record.state = "interrupted"
      record.error = None
      record.terminal_reason = None
    elif outcome == "completed" and record.exit_code == 0:
      record.state = "completed"
      record.error = None
      record.terminal_reason = terminal_reason
    else:
      record.state = "failed"
      record.terminal_reason = None
      if outcome == "failed" and terminal_error is not None:
        record.error = terminal_error
      elif record.error is None:
        record.error = terminal_error or f"Process exited with code {record.exit_code}"
    return prior != (record.state, record.error, record.terminal_reason)

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
    current_manifest_version = _runtime_attr(
      "_TASK_MANIFEST_VERSION",
      _TASK_MANIFEST_VERSION,
    )
    if (
      isinstance(manifest_version, bool)
      or not isinstance(manifest_version, int)
      or manifest_version != current_manifest_version
    ):
      self._warn_once(manifest_path, "Skipping autonomous manifest with unsupported version: %s")
      return None
    retired_binding_fields = {
      "execution_transport",
      "capability_credential_handle_id",
    }.intersection(manifest)
    if retired_binding_fields:
      self._warn_once(
        manifest_path,
        "Skipping autonomous manifest with retired parallel binding fields: %s",
      )
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
      self._warn_once(manifest_path, "Skipping autonomous manifest missing required fields: %s")
      return None
    if "pack" not in manifest or "deliver" not in manifest:
      self._warn_once(manifest_path, "Skipping v7 autonomous manifest missing pack/deliver fields: %s")
      return None
    canonical_events_path = manifest_path.with_name(
      f"{task_id}.events.jsonl"
    )
    events_path_status: str | None = None
    raw_events_path = manifest.get("events_path")
    if "events_path" in manifest and raw_events_path is not None:
      if not isinstance(raw_events_path, str) or not raw_events_path.strip():
        events_path_status = "path_mismatch"
      else:
        try:
          supplied_events_path = Path(raw_events_path).expanduser().resolve()
          resolved_canonical_events_path = canonical_events_path.resolve()
        except OSError:
          events_path_status = "path_mismatch"
        else:
          if supplied_events_path != resolved_canonical_events_path:
            events_path_status = "path_mismatch"
    if events_path_status == "path_mismatch":
      _LOGGER.warning(
        "Autonomous manifest events_path does not match the canonical path: %s",
        manifest_path,
      )
      events: list[dict[str, Any]] = []
      events_evidence_status = "path_mismatch"
    else:
      events, events_evidence_status = self._load_rehydrated_events(
        canonical_events_path
      )
    session_id = self._coerce_manifest_str(manifest, "session_id")
    channel_id = self._coerce_manifest_str(manifest, "channel_id")
    owner_lease_path = self._path_from_manifest(
      manifest,
      "owner_lease_path",
    )
    owner_lease_device = manifest.get("owner_lease_device")
    owner_lease_inode = manifest.get("owner_lease_inode")
    try:
      control_authority = AutonomousControlAuthority.from_receipt(
        manifest.get("control_authority")
      )
    except (TypeError, ValueError):
      _LOGGER.warning(
        "Skipping v7 autonomous manifest with invalid control authority: %s",
        manifest_path,
        exc_info=True,
      )
      return None
    if (
      session_id is None
      or channel_id is None
      or re.fullmatch(r"[0-9a-f]{64}", channel_id) is None
      or owner_lease_path is None
      or not owner_lease_path.is_absolute()
      or isinstance(owner_lease_device, bool)
      or not isinstance(owner_lease_device, int)
      or owner_lease_device < 0
      or isinstance(owner_lease_inode, bool)
      or not isinstance(owner_lease_inode, int)
      or owner_lease_inode <= 0
    ):
      self._warn_once(manifest_path, "Skipping v7 autonomous manifest with invalid session/channel/owner authority: %s")
      return None
    raw_pack = manifest.get("pack")
    deliver = manifest.get("deliver")
    if (
      mode not in {"once", "task", "skill", "pack"}
      or (
        raw_pack is not None
        and (
          not isinstance(raw_pack, str)
          or not raw_pack
          or raw_pack != raw_pack.strip()
        )
      )
      or not isinstance(deliver, bool)
      or (mode == "pack" and raw_pack is None)
      or (mode != "pack" and raw_pack is not None)
      or (not deliver and mode != "skill")
      or (
        mode == "pack"
        and (
          any(
            manifest.get(field_name) is not None
            for field_name in (
              "task",
              "skill",
              "context",
              "ticker",
              "max_budget_usd",
            )
          )
          or manifest.get("dev_mode") is not False
        )
      )
    ):
      self._warn_once(manifest_path, "Skipping autonomous manifest with invalid pack/deliver contract: %s")
      return None

    raw_manifest_state = self._coerce_manifest_str(manifest, "state") or "running"
    raw_exit_code = manifest.get("exit_code")
    exit_code = (
      int(raw_exit_code)
      if isinstance(raw_exit_code, int) and not isinstance(raw_exit_code, bool)
      else None
    )
    raw_state = raw_manifest_state
    was_interrupted = False
    completed_at = manifest.get("completed_at")
    completed_at = float(completed_at) if isinstance(completed_at, (int, float)) else None
    error = self._coerce_manifest_str(manifest, "error")
    terminal_reason = self._coerce_manifest_str(manifest, "terminal_reason")
    if (
      terminal_reason is not None
      and terminal_reason not in _AUTONOMOUS_TERMINAL_REASONS
    ):
      self._warn_once(manifest_path, "Skipping autonomous manifest with invalid terminal_reason: %s")
      return None
    terminal_states = _runtime_attr("_TERMINAL_AUTONOMOUS_STATES", _TERMINAL_AUTONOMOUS_STATES)
    rehydrated_active_states = _runtime_attr("_REHYDRATED_ACTIVE_STATES", _REHYDRATED_ACTIVE_STATES)
    terminal_outcome, terminal_error, event_terminal_reason = (
      self._terminal_event_outcome(events)
    )
    has_budget_exceeded = any(
      is_root_run_event(event) and event.get("type") == "budget_exceeded"
      for event in events
    )
    preserved_states = {
      "killed",
      "blocked",
      "budget_limited",
      "budget_exceeded",
      "interrupted",
    }
    is_active_or_unknown = (
      raw_manifest_state in rehydrated_active_states
      or raw_manifest_state not in terminal_states
    )
    if raw_manifest_state == "failed" and error is not None:
      pass
    elif raw_manifest_state in preserved_states:
      if raw_manifest_state in {"budget_limited", "budget_exceeded"}:
        raw_state = "budget_limited"
        error = None
        terminal_reason = None
    elif raw_manifest_state in {"completed", "failed"}:
      if terminal_outcome == "budget_limited":
        raw_state = "budget_limited"
        error = None
        terminal_reason = None
      elif terminal_outcome == "interrupted":
        raw_state = "interrupted"
        error = None
        terminal_reason = None
      elif terminal_outcome == "completed":
        if exit_code == 0:
          raw_state = "completed"
          error = None
          terminal_reason = event_terminal_reason
        else:
          raw_state = "failed"
          error = f"Process exited with code {exit_code}"
          terminal_reason = None
      elif terminal_outcome == "failed":
        raw_state = "failed"
        error = terminal_error
        terminal_reason = None
      elif has_budget_exceeded:
        raw_state = "budget_limited"
        error = None
        terminal_reason = None
    elif is_active_or_unknown:
      completed_at = self._last_event_timestamp(events) or rehydrate_time
      if terminal_outcome == "budget_limited":
        raw_state = "budget_limited"
        error = None
        terminal_reason = None
      elif terminal_outcome == "interrupted":
        raw_state = "interrupted"
        error = None
        terminal_reason = None
      elif terminal_outcome == "completed":
        if exit_code == 0:
          raw_state = "completed"
          error = None
          terminal_reason = event_terminal_reason
        else:
          raw_state = "failed"
          error = (
            "completed stream without a committed process exit"
            if exit_code is None
            else f"Process exited with code {exit_code}"
          )
          terminal_reason = None
      elif terminal_outcome == "failed":
        raw_state = "failed"
        error = terminal_error
        terminal_reason = None
      else:
        raw_state = "interrupted"
        was_interrupted = True
        error = _runtime_attr(
          "_REHYDRATION_INTERRUPTED_ERROR",
          _REHYDRATION_INTERRUPTED_ERROR,
        )
        terminal_reason = None

    cmd = manifest.get("cmd")
    resumed_as = manifest.get("resumed_as")
    user_email = manifest.get("user_email")
    user_email = user_email if isinstance(user_email, str) else None
    identity = _manifest_identity_payload(manifest, user_id=user_id, user_email=user_email)
    started_at = manifest.get("started_at")
    raw_capability_bind = manifest.get("capability_bind")
    capability_bind: CapabilityBind | None = None
    if raw_capability_bind is None:
      self._warn_once(manifest_path, "Skipping v7 autonomous manifest without a capability bind: %s")
      return None
    try:
      capability_bind = CapabilityBind.from_receipt(raw_capability_bind)
    except (TypeError, ValueError):
      _LOGGER.warning(
        "Skipping autonomous manifest with invalid capability bind: %s",
        manifest_path,
        exc_info=True,
      )
      return None
    if capability_bind.capability_id != "session.driver":
      self._warn_once(manifest_path, "Skipping autonomous manifest with non-session capability bind: %s")
      return None
    if capability_bind.run_mode not in {"autonomous", "cron"}:
      self._warn_once(manifest_path, "Skipping autonomous manifest with non-autonomous capability bind: %s")
      return None

    record_kwargs = dict(
      task_id=task_id,
      control_run_id=control_run_id,
      session_id=session_id,
      channel_id=channel_id,
      user_id=str(identity["owner_user_id"]),
      user_email=user_email,
      role=manifest.get("role"),
      profile=profile,
      mode=mode,
      task=self._coerce_manifest_str(manifest, "task"),
      skill=self._coerce_manifest_str(manifest, "skill"),
      pack=raw_pack,
      deliver=deliver,
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
      events_path=canonical_events_path,
      events_device=None,
      events_inode=None,
      events_evidence_status=events_evidence_status,
      operator_inbox_path=self._path_from_manifest(
        manifest,
        "operator_inbox_path",
        fallback=manifest_path.with_name(f"{task_id}.operator-messages.jsonl"),
      ),
      approval_decisions_path=self._path_from_manifest(
        manifest,
        "approval_decisions_path",
        fallback=None,
      ),
      control_authority=control_authority,
      owner_lease_path=owner_lease_path,
      owner_lease_device=owner_lease_device,
      owner_lease_inode=owner_lease_inode,
      started_at=float(started_at) if isinstance(started_at, (int, float)) else rehydrate_time,
      manifest_version=manifest_version,
      max_budget_usd=_positive_finite_float(manifest.get("max_budget_usd")),
      state=raw_state,
      exit_code=exit_code,
      error=error,
      terminal_reason=terminal_reason,
      proc=None,
      reaper_task=None,
      completed_at=completed_at,
      terminal_manifest_committed=(
        not was_interrupted
        and raw_state in terminal_states
      ),
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
      capability_bind=capability_bind,
      tool_result_spill_dir=self._registered_tool_result_spill_dir(
        task_id,
        manifest.get("tool_result_spill_dir"),
        require_exists=False,
      ),
    )
    try:
      record = AutonomousTask(**record_kwargs)
    except (TypeError, ValueError):
      _LOGGER.warning(
        "Skipping autonomous manifest with invalid task invariants: %s",
        manifest_path,
        exc_info=True,
      )
      return None
    self._attach_manifest_tracking(record)
    try:
      owner_cleanup_active = (
        was_interrupted
        and not autonomous_owner_lease_is_released(
          record.owner_lease_path,
          expected_device=record.owner_lease_device,
          expected_inode=record.owner_lease_inode,
        )
      )
    except RuntimeError:
      _LOGGER.warning(
        "Skipping autonomous manifest with invalid owner lease: %s",
        manifest_path,
        exc_info=True,
      )
      return None
    if (
      (was_interrupted and not owner_cleanup_active)
      or record.state != raw_manifest_state and not owner_cleanup_active
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
      task_id = self._coerce_manifest_str(manifest, "task_id")
      if task_id and task_id in getattr(self, "_spill_start_cleanup_skipped", set()):
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
