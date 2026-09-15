from __future__ import annotations

import errno
import fcntl
import hashlib
import logging
import os
import re
import shutil
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_gateway.tool_result_spill import SpillBudget, SpillCapabilities, SpillSink


AUTONOMOUS_SPILL_DIR_ENV = "AGENT_AUTONOMOUS_TOOL_RESULT_SPILL_DIR"
AUTONOMOUS_SPILL_TTL_HOURS_ENV = "AGENT_AUTONOMOUS_SPILL_TTL_HOURS"
AUTONOMOUS_SPILL_TTL_HOURS_DEFAULT = 72.0
AUTONOMOUS_SPILL_FILE_MAX_BYTES = 64 * 1024 * 1024
AUTONOMOUS_SPILL_RUN_MAX_BYTES = 256 * 1024 * 1024
_RUN_ROOT_NAME = ".agent-runs"
_SPILL_DIR_NAME = "tool_result_spill"
_LEASE_NAME = ".lease"
_FRESHNESS_NAME = ".spill_fresh"
_PENDING_RUN_PREFIX = ".pending-"
_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
_RUN_ROOT_README = """\
This tree is scratch space for oversized tool results (spill); entries expire after about 72 hours.
It is NOT the run record store.
Autonomous run records live at {autonomous_log_dir}/bg_N.* and transcripts live at api/sessions/.
See docs/reference/agent-run-evaluation.md.
"""
_LOGGER = logging.getLogger(__name__)


@dataclass
class _DirectoryLease:
  directory: Path
  fd: int
  directory_fd: int

  def is_current(self) -> bool:
    if self.fd < 0 or self.directory_fd < 0:
      return False
    try:
      held_directory = os.fstat(self.directory_fd)
      visible_directory = os.lstat(self.directory)
      held_lease = os.fstat(self.fd)
      visible_lease = os.stat(_LEASE_NAME, dir_fd=self.directory_fd, follow_symlinks=False)
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
    if self.fd < 0 and self.directory_fd < 0:
      return
    fd = self.fd
    directory_fd = self.directory_fd
    self.fd = -1
    self.directory_fd = -1
    if fd >= 0:
      try:
        fcntl.flock(fd, fcntl.LOCK_UN)
      finally:
        os.close(fd)
    if directory_fd >= 0:
      os.close(directory_fd)

  def __del__(self) -> None:
    try:
      self.close()
    except Exception:
      pass


def build_spill_sink(
  ctx: Any,
  *,
  run_id: str | None,
  available_tool_names: set[str],
  logger: Any = _LOGGER,
) -> SpillSink | None:
  existing = getattr(ctx, "tool_result_spill_sink", None)
  if isinstance(existing, SpillSink):
    return existing

  capabilities = SpillCapabilities(
    code_execute="code_execute" in available_tool_names,
    file_read="file_read" in available_tool_names,
    file_grep="file_grep" in available_tool_names,
    spill_read="tool_result_read" in available_tool_names,
  )
  artifact_raw = os.getenv(AUTONOMOUS_SPILL_DIR_ENV, "").strip()
  if artifact_raw:
    sink = _build_artifact_sink(Path(artifact_raw), capabilities=capabilities, logger=logger)
  else:
    workspace = getattr(ctx, "workspace", None)
    if workspace is None:
      return None
    chosen_run_id = str(run_id or getattr(ctx, "tool_result_spill_run_id", None) or uuid.uuid4().hex)
    setattr(ctx, "tool_result_spill_run_id", chosen_run_id)
    sink = _build_direct_sink(
      Path(workspace),
      chosen_run_id,
      capabilities=capabilities,
      logger=logger,
    )
  if sink is not None:
    setattr(ctx, "tool_result_spill_sink", sink)
  return sink


def sweep_direct_spill_roots(
  workspace: Path,
  *,
  ttl_hours: float | None = None,
  now: float | None = None,
  logger: Any = _LOGGER,
) -> int:
  workspace_root = _validated_workspace(workspace)
  runs_root = workspace_root / _RUN_ROOT_NAME
  if not runs_root.exists():
    return 0
  if runs_root.is_symlink() or not runs_root.is_dir():
    logger.warning("Autonomous spill TTL sweep skipped unsafe run root: %s", runs_root)
    return 0
  ttl = _configured_ttl_hours() if ttl_hours is None else max(0.0, float(ttl_hours))
  cutoff = (time.time() if now is None else float(now)) - (ttl * 3600.0)
  removed = 0
  for run_dir in list(runs_root.iterdir()):
    if (
      run_dir.name.startswith(_PENDING_RUN_PREFIX)
      or _SAFE_RUN_ID_RE.fullmatch(run_dir.name) is None
      or run_dir.is_symlink()
      or not run_dir.is_dir()
      or run_dir.parent != runs_root
    ):
      continue
    lease = _try_acquire_lease(run_dir, create=True)
    if lease is None:
      continue
    try:
      marker = run_dir / _FRESHNESS_NAME
      try:
        marker_mtime = marker.lstat().st_mtime
      except FileNotFoundError:
        marker_mtime = float("-inf")
      except OSError:
        continue
      if marker.is_symlink() or marker_mtime >= cutoff:
        continue
      if run_dir.parent.resolve() != runs_root.resolve():
        continue
      shutil.rmtree(run_dir)
      removed += 1
    except OSError:
      logger.warning("Failed to prune autonomous spill root: %s", run_dir, exc_info=True)
    finally:
      lease.close()
  return removed


def _build_artifact_sink(
  spill_dir: Path,
  *,
  capabilities: SpillCapabilities,
  logger: Any,
) -> SpillSink | None:
  try:
    if not spill_dir.is_absolute() or spill_dir.is_symlink() or not spill_dir.is_dir():
      raise ValueError("artifact spill directory must be an existing absolute non-symlink directory")
    resolved = spill_dir.resolve(strict=True)
    lease = _try_acquire_lease(resolved, create=True)
    if lease is None:
      raise RuntimeError("artifact spill lease unavailable")
  except Exception as exc:
    logger.warning("Autonomous artifact spill disabled: %s", exc)
    return None

  def _provider() -> str:
    if not lease.is_current() or resolved.is_symlink() or not resolved.is_dir():
      raise RuntimeError("artifact spill root or lease was lost")
    return str(resolved)

  sink = SpillSink(
    root_provider=_provider,
    capabilities=capabilities,
    budget=SpillBudget(AUTONOMOUS_SPILL_RUN_MAX_BYTES),
    max_file_bytes=AUTONOMOUS_SPILL_FILE_MAX_BYTES,
  )
  setattr(sink, "_autonomous_spill_lease", lease)
  setattr(sink, "_autonomous_spill_lane", "artifact")
  return sink


def _build_direct_sink(
  workspace: Path,
  run_id: str,
  *,
  capabilities: SpillCapabilities,
  logger: Any,
) -> SpillSink | None:
  try:
    workspace_root = _validated_workspace(workspace)
    sweep_direct_spill_roots(workspace_root, logger=logger)
    runs_root = workspace_root / _RUN_ROOT_NAME
    safe_id = _safe_run_id(run_id)
    run_dir = runs_root / safe_id
    spill_dir = run_dir / _SPILL_DIR_NAME
  except Exception as exc:
    logger.warning("Autonomous direct spill disabled: %s", exc)
    return None

  establish_lock = threading.RLock()
  lease: _DirectoryLease | None = None

  def _ensure_established() -> _DirectoryLease:
    nonlocal lease
    with establish_lock:
      if lease is not None and lease.fd >= 0:
        if not lease.is_current():
          raise RuntimeError("direct spill root or lease was replaced")
        _require_direct_containment(workspace_root, run_dir, spill_dir)
        return lease
      _mkdir_non_symlink(runs_root)
      established = _establish_or_reuse_direct_root(runs_root, run_dir)
      try:
        _require_direct_containment(workspace_root, run_dir, spill_dir)
      except Exception:
        established.close()
        raise
      lease = established
      setattr(sink, "_autonomous_spill_lease", established)
      return established

  def _provider() -> str:
    active_lease = _ensure_established()
    if not active_lease.is_current():
      raise RuntimeError("direct spill lease was lost")
    _require_direct_containment(workspace_root, run_dir, spill_dir)
    return str(spill_dir.resolve(strict=True))

  def _touch_freshness() -> None:
    _ensure_established()
    _touch_freshness_marker(run_dir)

  sink = SpillSink(
    root_provider=_provider,
    capabilities=capabilities,
    budget=SpillBudget(AUTONOMOUS_SPILL_RUN_MAX_BYTES),
    max_file_bytes=AUTONOMOUS_SPILL_FILE_MAX_BYTES,
    after_commit=_touch_freshness,
  )
  setattr(sink, "_autonomous_spill_lease", None)
  setattr(sink, "_autonomous_spill_lane", "direct")
  setattr(sink, "_autonomous_spill_run_dir", run_dir)
  return sink


def _establish_or_reuse_direct_root(runs_root: Path, run_dir: Path) -> _DirectoryLease:
  readme = runs_root / "README.md"
  try:
    with readme.open("x", encoding="utf-8") as handle:
      handle.write(_RUN_ROOT_README)
  except OSError:
    pass
  if run_dir.exists():
    if run_dir.is_symlink() or not run_dir.is_dir():
      raise RuntimeError("direct spill run root is unsafe")
    lease = _try_acquire_lease(run_dir, create=True)
    if lease is None:
      raise RuntimeError("direct spill run is already active")
    spill_dir = run_dir / _SPILL_DIR_NAME
    try:
      _mkdir_non_symlink(spill_dir)
      _touch_freshness_marker(run_dir)
      return lease
    except Exception:
      lease.close()
      raise

  pending = runs_root / f"{_PENDING_RUN_PREFIX}{uuid.uuid4().hex}"
  pending.mkdir(mode=0o700)
  lease: _DirectoryLease | None = None
  try:
    (pending / _SPILL_DIR_NAME).mkdir(mode=0o700)
    lease = _try_acquire_lease(pending, create=True)
    if lease is None:
      raise RuntimeError("failed to acquire new direct spill lease")
    _touch_freshness_marker(pending, create_exclusive=True)
    try:
      pending.rename(run_dir)
    except OSError as exc:
      if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
        raise
      lease.close()
      lease = None
      shutil.rmtree(pending)
      return _establish_or_reuse_direct_root(runs_root, run_dir)
    lease.directory = run_dir
    return lease
  except Exception:
    if lease is not None:
      lease.close()
    if pending.exists() and not pending.is_symlink():
      shutil.rmtree(pending, ignore_errors=True)
    raise


def _try_acquire_lease(directory: Path, *, create: bool) -> _DirectoryLease | None:
  directory_flags = os.O_RDONLY
  if hasattr(os, "O_DIRECTORY"):
    directory_flags |= os.O_DIRECTORY
  if hasattr(os, "O_NOFOLLOW"):
    directory_flags |= os.O_NOFOLLOW
  flags = os.O_RDWR
  if create:
    flags |= os.O_CREAT
  if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
  directory_fd: int | None = None
  fd: int | None = None
  try:
    directory_fd = os.open(directory, directory_flags)
    held_directory = os.fstat(directory_fd)
    visible_directory = os.lstat(directory)
    if (
      not stat.S_ISDIR(held_directory.st_mode)
      or not stat.S_ISDIR(visible_directory.st_mode)
      or (held_directory.st_dev, held_directory.st_ino)
      != (visible_directory.st_dev, visible_directory.st_ino)
    ):
      raise RuntimeError("spill directory identity mismatch")
    fd = os.open(_LEASE_NAME, flags, 0o600, dir_fd=directory_fd)
  except OSError:
    if fd is not None:
      os.close(fd)
    if directory_fd is not None:
      os.close(directory_fd)
    return None
  except Exception:
    if fd is not None:
      os.close(fd)
    if directory_fd is not None:
      os.close(directory_fd)
    return None
  try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    held = os.fstat(fd)
    visible = os.stat(_LEASE_NAME, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(held.st_mode) or not stat.S_ISREG(visible.st_mode):
      raise RuntimeError("spill lease is not a regular file")
    if (held.st_dev, held.st_ino) != (visible.st_dev, visible.st_ino):
      raise RuntimeError("spill lease path was replaced")
    lease = _DirectoryLease(directory=directory, fd=fd, directory_fd=directory_fd)
    if not lease.is_current():
      raise RuntimeError("spill lease directory was replaced")
    return lease
  except Exception:
    try:
      fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
      pass
    os.close(fd)
    os.close(directory_fd)
    return None


def _validated_workspace(workspace: Path) -> Path:
  expanded = workspace.expanduser()
  if expanded.is_symlink() or not expanded.is_dir():
    raise RuntimeError("autonomous workspace must be an existing non-symlink directory")
  return expanded.resolve(strict=True)


def _mkdir_non_symlink(path: Path) -> None:
  try:
    path.mkdir(mode=0o700)
  except FileExistsError:
    pass
  if path.is_symlink() or not path.is_dir():
    raise RuntimeError(f"unsafe spill path component: {path}")


def _require_direct_containment(workspace: Path, run_dir: Path, spill_dir: Path) -> None:
  runs_root = workspace / _RUN_ROOT_NAME
  for path in (runs_root, run_dir, spill_dir):
    if path.is_symlink() or not path.is_dir():
      raise RuntimeError(f"unsafe spill path component: {path}")
  resolved_workspace = workspace.resolve(strict=True)
  resolved_run = run_dir.resolve(strict=True)
  resolved_spill = spill_dir.resolve(strict=True)
  if resolved_workspace not in resolved_run.parents or resolved_run not in resolved_spill.parents:
    raise RuntimeError("direct spill root escapes workspace")


def _safe_run_id(run_id: str) -> str:
  cleaned = str(run_id).strip()
  if (
    _SAFE_RUN_ID_RE.fullmatch(cleaned)
    and cleaned not in {".", ".."}
    and not cleaned.startswith(_PENDING_RUN_PREFIX)
  ):
    return cleaned
  normalized = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)[:80].strip(".") or "run"
  digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12]
  return f"{normalized}-{digest}"


def _touch_freshness_marker(run_dir: Path, *, create_exclusive: bool = False) -> None:
  marker = run_dir / _FRESHNESS_NAME
  flags = os.O_WRONLY | os.O_CREAT
  if create_exclusive:
    flags |= os.O_EXCL
  if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
  fd = os.open(marker, flags, 0o600)
  try:
    held = os.fstat(fd)
    visible = os.lstat(marker)
    if (
      not stat.S_ISREG(held.st_mode)
      or not stat.S_ISREG(visible.st_mode)
      or (held.st_dev, held.st_ino) != (visible.st_dev, visible.st_ino)
    ):
      raise RuntimeError("direct spill freshness marker identity mismatch")
    os.utime(fd, None)
  finally:
    os.close(fd)


def _configured_ttl_hours() -> float:
  raw = os.getenv(AUTONOMOUS_SPILL_TTL_HOURS_ENV, "").strip()
  if not raw:
    return AUTONOMOUS_SPILL_TTL_HOURS_DEFAULT
  try:
    value = float(raw)
  except ValueError:
    return AUTONOMOUS_SPILL_TTL_HOURS_DEFAULT
  return value if value > 0 else AUTONOMOUS_SPILL_TTL_HOURS_DEFAULT


__all__ = [
  "AUTONOMOUS_SPILL_DIR_ENV",
  "AUTONOMOUS_SPILL_FILE_MAX_BYTES",
  "AUTONOMOUS_SPILL_RUN_MAX_BYTES",
  "AUTONOMOUS_SPILL_TTL_HOURS_DEFAULT",
  "AUTONOMOUS_SPILL_TTL_HOURS_ENV",
  "build_spill_sink",
  "sweep_direct_spill_roots",
]
