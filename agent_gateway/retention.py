"""Generic durable-artifact retention policy and sweep machinery.

This module intentionally contains no application paths.  Package consumers
register adapters explicitly; importing :mod:`agent_gateway` never starts a
sweep.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import logging
import math
import os
import stat
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Protocol


SweepMode = Literal["dry_run", "enforce"]
RetentionStrategy = Literal[
  "age",
  "total_bytes",
  "keep_n",
  "age_or_keep_n",
  "truncate",
  "rotate",
  "keep_forever",
]

log = logging.getLogger(__name__)


class RetentionSafetyError(ValueError):
  """A configured root or candidate fails a destructive-operation guard."""


class RetentionLockBusy(RuntimeError):
  """Another process owns the installation-wide retention sweep lock."""


@dataclass(frozen=True)
class RetentionPolicy:
  """A resolved policy.  Absence is never interpreted as keep-forever."""

  strategy: RetentionStrategy
  owner: str
  reason: str
  max_age_days: float | None = None
  max_total_bytes: int | None = None
  keep_n: int | None = None

  def __post_init__(self) -> None:
    if not self.owner.strip():
      raise ValueError("retention policy owner is required")
    if not self.reason.strip():
      raise ValueError("retention policy reason is required")
    if self.max_age_days is not None and self.max_age_days <= 0:
      raise ValueError("max_age_days must be positive")
    if self.max_total_bytes is not None and self.max_total_bytes <= 0:
      raise ValueError("max_total_bytes must be positive")
    if self.keep_n is not None and self.keep_n < 0:
      raise ValueError("keep_n cannot be negative")
    required = {
      "age": self.max_age_days is not None,
      "total_bytes": self.max_total_bytes is not None,
      "keep_n": self.keep_n is not None,
      "age_or_keep_n": self.max_age_days is not None and self.keep_n is not None,
      "truncate": self.max_total_bytes is not None,
      "rotate": self.max_total_bytes is not None and self.keep_n is not None,
      "keep_forever": all(
        value is None for value in (self.max_age_days, self.max_total_bytes, self.keep_n)
      ),
    }[self.strategy]
    if not required:
      raise ValueError(f"strategy {self.strategy!r} is missing or conflicts with its bound")

  @classmethod
  def keep_forever(cls, reason: str, owner: str) -> "RetentionPolicy":
    return cls(strategy="keep_forever", reason=reason, owner=owner)

  @property
  def is_keep_forever(self) -> bool:
    return self.strategy == "keep_forever"


@dataclass(frozen=True)
class RetentionSweepReport:
  key: str
  mode: SweepMode
  would_delete_count: int = 0
  would_delete_bytes: int = 0
  deleted_count: int = 0
  deleted_bytes: int = 0
  skipped_count: int = 0
  errors: tuple[str, ...] = ()
  details: tuple[str, ...] = ()
  started_at: str | None = None
  completed_at: str | None = None

  def merged(self, other: "RetentionSweepReport") -> "RetentionSweepReport":
    return RetentionSweepReport(
      key=self.key,
      mode=self.mode,
      would_delete_count=self.would_delete_count + other.would_delete_count,
      would_delete_bytes=self.would_delete_bytes + other.would_delete_bytes,
      deleted_count=self.deleted_count + other.deleted_count,
      deleted_bytes=self.deleted_bytes + other.deleted_bytes,
      skipped_count=self.skipped_count + other.skipped_count,
      errors=self.errors + other.errors,
      details=self.details + other.details,
      started_at=self.started_at or other.started_at,
      completed_at=other.completed_at or self.completed_at,
    )

  def as_dict(self) -> dict[str, Any]:
    return {
      "key": self.key,
      "mode": self.mode,
      "would_delete_count": self.would_delete_count,
      "would_delete_bytes": self.would_delete_bytes,
      "deleted_count": self.deleted_count,
      "deleted_bytes": self.deleted_bytes,
      "skipped_count": self.skipped_count,
      "errors": list(self.errors),
      "details": list(self.details),
      "started_at": self.started_at,
      "completed_at": self.completed_at,
    }


@dataclass(frozen=True)
class RetentionSweepContext:
  mode: SweepMode
  now: datetime
  policy: RetentionPolicy
  authorized_roots: tuple[Path, ...]
  install_id: str | None = None
  migration_authorization: Mapping[str, Any] = field(
    default_factory=lambda: MappingProxyType({})
  )
  deadline_monotonic: float | None = None
  cancelled: Callable[[], bool] = lambda: False
  report_sink: Callable[[RetentionSweepReport], None] | None = None

  @property
  def enforce(self) -> bool:
    return self.mode == "enforce"

  @property
  def now_ts(self) -> float:
    return self.now.timestamp()

  def cutoff_ts(self, max_age_days: float | None = None) -> float:
    days = self.policy.max_age_days if max_age_days is None else max_age_days
    if days is None:
      raise ValueError("policy has no age cutoff")
    return self.now_ts - days * 86_400


class RetentionAdapter(Protocol):
  def sweep(self, context: RetentionSweepContext) -> RetentionSweepReport: ...


@dataclass(frozen=True)
class RetentionCatalogEntry:
  key: str
  policy: RetentionPolicy
  adapter: RetentionAdapter
  roots: tuple[Path, ...]
  triage_input: bool = False
  invariants: tuple[str, ...] = ()
  deferred: bool = False
  # Set ONLY for entries whose adapter delegates deletion to a component that
  # enforces its own keep-set and re-verifies afterwards (today: runtime_versions
  # -> local_gateway_runtime.gc_runtime). For those, `roots` is descriptive and the
  # adapter never reads `authorized_roots`, so running the generic repo-containment
  # check over it is both useless and actively harmful: a gateway running out of a
  # runtime version has repo_root = <RUNTIME_ROOT>/versions/<VER>/ai-excel-addin, so
  # the runtime-GC root is structurally a PARENT of repo_root and _is_broad_root
  # rejects it on every sweep. This flag skips that pre-computation for such entries.
  # It does NOT relax _is_broad_root, and it does NOT change any other entry.
  adapter_owns_root_validation: bool = False

  def __post_init__(self) -> None:
    if not self.key.strip():
      raise ValueError("catalog key is required")
    if not self.roots:
      raise ValueError(f"catalog entry {self.key!r} has no authorized roots")
    if self.triage_input:
      days = self.policy.max_age_days
      if days is None or days * 24 < 72:
        raise ValueError(f"triage input {self.key!r} must retain at least 72 hours")
    if self.policy.is_keep_forever:
      log.info(
        "retention keep_forever registered: key=%s owner=%s reason=%s",
        self.key,
        self.policy.owner,
        self.policy.reason,
      )


@dataclass(frozen=True)
class RetentionCatalog:
  entries: tuple[RetentionCatalogEntry, ...]
  state_root: Path
  repo_root: Path | None = None
  sweep_interval_seconds: float = 6 * 3600

  def __post_init__(self) -> None:
    keys = [entry.key for entry in self.entries]
    if len(keys) != len(set(keys)):
      raise ValueError("retention catalog keys must be unique")
    resolve_safe_root(self.state_root, repo_root=self.repo_root)

  def dump(self) -> list[dict[str, Any]]:
    return [
      {
        "key": entry.key,
        "strategy": entry.policy.strategy,
        "max_age_days": entry.policy.max_age_days,
        "max_total_bytes": entry.policy.max_total_bytes,
        "keep_n": entry.policy.keep_n,
        "owner": entry.policy.owner,
        "reason": entry.policy.reason,
        "roots": [str(path) for path in entry.roots],
        "invariants": list(entry.invariants),
        "deferred": entry.deferred,
      }
      for entry in self.entries
    ]


def _is_broad_root(path: Path, *, repo_root: Path | None = None) -> bool:
  """Mirror local_tools' broad-root guard for retention roots."""

  resolved = path.expanduser().resolve(strict=False)
  home = Path.home().resolve(strict=False)
  if resolved in {Path("/"), home}:
    return True
  if repo_root is not None:
    project = repo_root.expanduser().resolve(strict=False)
    if resolved == project or resolved in project.parents:
      return True
  return False


def _has_symlink_component(path: Path) -> bool:
  absolute = path.absolute()
  current = Path(absolute.anchor)
  for part in absolute.parts[1:]:
    current /= part
    try:
      if stat.S_ISLNK(current.lstat().st_mode):
        return True
    except FileNotFoundError:
      return False
  return False


def resolve_safe_root(path: Path | str, *, repo_root: Path | None = None) -> Path:
  raw = Path(path).expanduser()
  if raw.is_symlink() or _has_symlink_component(raw):
    raise RetentionSafetyError(f"retention root must not be a symlink: {raw}")
  resolved = raw.resolve(strict=False)
  if _is_broad_root(resolved, repo_root=repo_root):
    raise RetentionSafetyError(f"refusing broad retention root: {resolved}")
  return resolved


def resolve_contained_path(
  path: Path | str,
  root: Path | str,
  *,
  repo_root: Path | None = None,
  require_exists: bool = True,
) -> Path:
  safe_root = resolve_safe_root(root, repo_root=repo_root)
  raw = Path(path).expanduser()
  if _has_symlink_component(raw):
    raise RetentionSafetyError(f"retention candidate contains a symlink component: {raw}")
  try:
    info = raw.lstat()
  except FileNotFoundError:
    if require_exists:
      raise RetentionSafetyError(f"retention candidate is missing: {raw}") from None
  else:
    if stat.S_ISLNK(info.st_mode):
      raise RetentionSafetyError(f"retention candidate must not be a symlink: {raw}")
  resolved = raw.resolve(strict=False)
  if resolved == safe_root or safe_root not in resolved.parents:
    raise RetentionSafetyError(f"retention candidate escapes root {safe_root}: {raw}")
  return resolved


@contextlib.contextmanager
def sweep_lock(lock_path: Path, *, blocking: bool = False):
  """Take the one process-wide lock shared by CLI and server entry points."""

  # A state-root directory lock avoids creating a third dry-run artifact.  A
  # caller may still supply a regular lock pathname when write purity is not a
  # concern.
  if lock_path.is_dir():
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
      flags |= os.O_DIRECTORY
    fd = os.open(lock_path, flags)
  else:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
      flags |= os.O_NOFOLLOW
    fd = os.open(lock_path, flags, 0o600)
  try:
    operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
      fcntl.flock(fd, operation)
    except OSError as exc:
      if exc.errno in {errno.EACCES, errno.EAGAIN}:
        raise RetentionLockBusy(f"retention sweep already running: {lock_path}") from exc
      raise
    yield
  finally:
    try:
      fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
      os.close(fd)


class KeepForeverAdapter:
  def __init__(self, key: str):
    self.key = key

  def sweep(self, context: RetentionSweepContext) -> RetentionSweepReport:
    return RetentionSweepReport(
      key=self.key,
      mode=context.mode,
      details=(
        f"keep_forever owner={context.policy.owner} reason={context.policy.reason}",
      ),
    )


class FileAgeAdapter:
  """Reusable age deletion for roots with no higher-level file semantics."""

  def __init__(
    self,
    key: str,
    root: Path,
    *,
    pattern: str = "*",
    recursive: bool = False,
    repo_root: Path | None = None,
  ) -> None:
    self.key = key
    self.root = Path(root)
    self.pattern = pattern
    self.recursive = recursive
    self.repo_root = repo_root

  def sweep(self, context: RetentionSweepContext) -> RetentionSweepReport:
    root = resolve_safe_root(self.root, repo_root=self.repo_root)
    if not root.exists():
      return RetentionSweepReport(key=self.key, mode=context.mode)
    if root.is_symlink() or not root.is_dir():
      raise RetentionSafetyError(f"file-age root is not a regular directory: {root}")
    cutoff = context.cutoff_ts()
    candidates = root.rglob(self.pattern) if self.recursive else root.glob(self.pattern)
    count = size = deleted = deleted_bytes = skipped = 0
    errors: list[str] = []
    for candidate in candidates:
      try:
        resolved = resolve_contained_path(candidate, root, repo_root=self.repo_root)
        info = candidate.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_mtime >= cutoff:
          continue
        count += 1
        size += info.st_size
        if context.enforce:
          resolved.unlink()
          deleted += 1
          deleted_bytes += info.st_size
      except RetentionSafetyError as exc:
        skipped += 1
        errors.append(str(exc))
      except FileNotFoundError:
        continue
      except OSError as exc:
        errors.append(f"{candidate}: {exc}")
    return RetentionSweepReport(
      key=self.key,
      mode=context.mode,
      would_delete_count=count,
      would_delete_bytes=size,
      deleted_count=deleted,
      deleted_bytes=deleted_bytes,
      skipped_count=skipped,
      errors=tuple(errors),
    )


class UiBlocksEnvelopeAgeAdapter:
  """Age multi-user UI-block envelopes by their chat timestamp."""

  def __init__(self, key: str, users_root: Path, *, repo_root: Path | None = None) -> None:
    self.key = key
    self.root = Path(users_root)
    self.repo_root = repo_root

  def sweep(self, context: RetentionSweepContext) -> RetentionSweepReport:
    root = resolve_safe_root(self.root, repo_root=self.repo_root)
    if not root.exists():
      return RetentionSweepReport(key=self.key, mode=context.mode)
    if root.is_symlink() or not root.is_dir():
      raise RetentionSafetyError(f"ui-blocks root is not a regular directory: {root}")
    cutoff = context.cutoff_ts()
    count = size = deleted = deleted_bytes = skipped = 0
    errors: list[str] = []
    for candidate in root.glob("*/workspace/artifacts/_ui_blocks/*.json"):
      try:
        resolved = resolve_contained_path(candidate, root, repo_root=self.repo_root)
        info = candidate.lstat()
        if not stat.S_ISREG(info.st_mode):
          continue
        envelope_ts = _ui_blocks_envelope_ts(candidate, fallback=info.st_mtime)
        if envelope_ts >= cutoff:
          continue
        count += 1
        size += info.st_size
        if context.enforce:
          resolved.unlink()
          deleted += 1
          deleted_bytes += info.st_size
      except RetentionSafetyError as exc:
        skipped += 1
        errors.append(str(exc))
      except FileNotFoundError:
        continue
      except OSError as exc:
        errors.append(f"{candidate}: {exc}")
    if deleted:
      try:
        from .artifact_sidecar_index import reconcile_ui_blocks_index

        reconcile_ui_blocks_index(root)
      except Exception as exc:
        errors.append(f"ui blocks index reconciliation failed: {exc}")
    return RetentionSweepReport(
      key=self.key,
      mode=context.mode,
      would_delete_count=count,
      would_delete_bytes=size,
      deleted_count=deleted,
      deleted_bytes=deleted_bytes,
      skipped_count=skipped,
      errors=tuple(errors),
    )


def _ui_blocks_envelope_ts(path: Path, *, fallback: float) -> float:
  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or isinstance(payload.get("ts"), bool):
      raise ValueError("invalid envelope timestamp")
    value = float(payload["ts"])
    if not math.isfinite(value):
      raise ValueError("invalid envelope timestamp")
    return value
  except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    return fallback


class RetentionSweeper:
  def __init__(self, catalog: RetentionCatalog, *, logger: logging.Logger | None = None):
    self.catalog = catalog
    self.log = logger or log

  def sweep(
    self,
    mode: SweepMode,
    *,
    now: datetime | None = None,
    install_id: str | None = None,
    migration_authorization: Mapping[str, Any] | None = None,
    enforce_allowlist: frozenset[str] | None = None,
    blocking_lock: bool = False,
    report_sink: Callable[[RetentionSweepReport], None] | None = None,
  ) -> tuple[RetentionSweepReport, ...]:
    snapshot = now or datetime.now(timezone.utc)
    if snapshot.tzinfo is None or snapshot.utcoffset() is None:
      raise ValueError("retention sweep clock must have an explicit UTC offset")
    lock_path = self.catalog.state_root
    reports: list[RetentionSweepReport] = []
    with sweep_lock(lock_path, blocking=blocking_lock):
      for entry in self.catalog.entries:
        started = datetime.now(timezone.utc).isoformat()
        effective_mode: SweepMode = mode
        pending_reacceptance = (
          mode == "enforce"
          and enforce_allowlist is not None
          and entry.key not in enforce_allowlist
        )
        if pending_reacceptance:
          effective_mode = "dry_run"
        try:
          context = RetentionSweepContext(
            mode=effective_mode,
            now=snapshot,
            policy=entry.policy,
            authorized_roots=(
              ()
              if entry.adapter_owns_root_validation
              else tuple(
                resolve_safe_root(root, repo_root=self.catalog.repo_root) for root in entry.roots
              )
            ),
            install_id=install_id,
            migration_authorization=MappingProxyType(dict(migration_authorization or {})),
            report_sink=report_sink,
          )
          report = entry.adapter.sweep(context)
        except Exception as exc:  # per-adapter isolation is a driver invariant
          self.log.exception("retention adapter failed: %s", entry.key)
          report = RetentionSweepReport(
            key=entry.key,
            mode=effective_mode,
            errors=(f"{type(exc).__name__}: {exc}",),
          )
        details = report.details
        if pending_reacceptance:
          details += ("pending migration re-acceptance",)
        completed = replace(
          report,
          key=entry.key,
          mode=effective_mode,
          details=details,
          started_at=report.started_at or started,
          completed_at=report.completed_at or datetime.now(timezone.utc).isoformat(),
        )
        reports.append(completed)
        if report_sink is not None:
          report_sink(completed)
    return tuple(reports)


def sweep_retention(
  catalog: RetentionCatalog,
  mode: SweepMode,
  **kwargs: Any,
) -> tuple[RetentionSweepReport, ...]:
  return RetentionSweeper(catalog).sweep(mode, **kwargs)


__all__ = [
  "FileAgeAdapter",
  "KeepForeverAdapter",
  "RetentionAdapter",
  "RetentionCatalog",
  "RetentionCatalogEntry",
  "RetentionLockBusy",
  "RetentionPolicy",
  "RetentionSafetyError",
  "RetentionSweepContext",
  "RetentionSweepReport",
  "RetentionSweeper",
  "resolve_contained_path",
  "resolve_safe_root",
  "sweep_lock",
  "sweep_retention",
]
