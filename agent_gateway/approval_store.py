from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
import hashlib
import logging
import os
import secrets
import sqlite3
import stat
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

from . import approval_store_rows as _rows
from . import prepared_business_model_store as _prepared_bm
from . import raw_patch_authorization_store as _raw_patch_auth
from .approval_notifications import (
  ApprovalNotificationDestinationResolver,
  ApprovalNotificationSender,
  approval_notification_policy_for_request,
  maybe_await,
  normalize_approval_notification_destinations,
  render_approval_notification_message,
)
from .approval_policy import (
  ApprovalRequest,
  ApprovalState,
  ApprovalVote,
  DelegationGrant,
  PersistentGrant,
  revalidate_approval_request,
  utc_now,
)


TERMINAL_STATES = frozenset({"auto_approved", "auto_denied", "approved", "denied", "expired"})
APPROVAL_DB_PATH_ENV = "GATEWAY_APPROVAL_DB_PATH"
_AUTONOMOUS_APPROVAL_DELIVERY_OUTBOX_COLUMNS = frozenset({
  "delivery_sequence",
  "approval_id",
  "tool_call_id",
  "nonce",
  "task_id",
  "control_run_id",
  "session_id",
  "channel_id",
  "approved",
  "allow_tool_type",
  "decided_at_ns",
  "retry_deadline_ns",
  "next_attempt_ns",
  "last_attempt_ns",
  "state",
  "audit_state",
  "triggering_vote_id",
  "vote_audit_entry_id",
  "terminal_audit_entry_id",
  "attempt_count",
  "last_error",
  "created_at",
  "updated_at",
  "audit_ready_at",
  "published_at",
  "acknowledged_at",
  "quarantined_at",
})
USER_DATA_DIR_ENV = "USER_DATA_DIR"
log = logging.getLogger("agent_gateway.approval_store")
_AUTONOMOUS_APPROVAL_DELIVERY_CONTEXT_FIELDS = frozenset({
  "task_id",
  "control_run_id",
  "session_id",
  "channel_id",
  "tool_call_id",
  "nonce",
})
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm")
AUTONOMOUS_APPROVAL_DELIVERY_MAX_ATTEMPTS = 5
AUTONOMOUS_APPROVAL_DELIVERY_RETRY_WINDOW_NS = (
  5 * 60 * 1_000_000_000
)
AUTONOMOUS_APPROVAL_DELIVERY_RETRY_BASE_NS = 1_000_000_000
AUTONOMOUS_APPROVAL_DELIVERY_RETRY_MAX_NS = 16_000_000_000
_SQLITE_MAX_INTEGER = (1 << 63) - 1


def _require_secure_sqlite_parent(path: Path) -> os.stat_result:
  try:
    parent_stat = os.lstat(path.parent)
  except OSError as exc:
    raise RuntimeError(
      f"approval store parent is unavailable: {path.parent}"
    ) from exc
  if (
    not stat.S_ISDIR(parent_stat.st_mode)
    or parent_stat.st_uid != os.geteuid()
    or stat.S_IMODE(parent_stat.st_mode) & 0o022
  ):
    raise RuntimeError(
      f"approval store parent has unsafe file identity: {path.parent}"
    )
  return parent_stat


def _open_secure_sqlite_parent(
  path: Path,
) -> tuple[int, os.stat_result]:
  parent_stat = _require_secure_sqlite_parent(path)
  parent_flags = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
  )
  try:
    parent_fd = os.open(path.parent, parent_flags)
  except OSError as exc:
    raise RuntimeError(
      f"approval store parent is unavailable: {path.parent}"
    ) from exc
  try:
    bound_parent_stat = os.fstat(parent_fd)
    visible_parent_stat = os.lstat(path.parent)
    if (
      not stat.S_ISDIR(bound_parent_stat.st_mode)
      or bound_parent_stat.st_dev != parent_stat.st_dev
      or bound_parent_stat.st_ino != parent_stat.st_ino
      or bound_parent_stat.st_uid != os.geteuid()
      or stat.S_IMODE(bound_parent_stat.st_mode) & 0o022
      or visible_parent_stat.st_dev != bound_parent_stat.st_dev
      or visible_parent_stat.st_ino != bound_parent_stat.st_ino
      or visible_parent_stat.st_uid != os.geteuid()
      or stat.S_IMODE(visible_parent_stat.st_mode) & 0o022
    ):
      raise RuntimeError(
        f"approval store parent identity changed: {path.parent}"
      )
    return parent_fd, bound_parent_stat
  except BaseException:
    os.close(parent_fd)
    raise


def _require_matching_sqlite_identity(
  path: Path,
  *,
  parent_fd: int,
  bound_parent_stat: os.stat_result,
  file_stat: os.stat_result,
) -> None:
  relative_stat = os.stat(
    path.name,
    dir_fd=parent_fd,
    follow_symlinks=False,
  )
  visible_stat = os.lstat(path)
  visible_parent_stat = os.lstat(path.parent)
  if (
    visible_parent_stat.st_dev != bound_parent_stat.st_dev
    or visible_parent_stat.st_ino != bound_parent_stat.st_ino
    or visible_parent_stat.st_uid != os.geteuid()
    or stat.S_IMODE(visible_parent_stat.st_mode) & 0o022
    or any(
      (
        candidate.st_dev != file_stat.st_dev
        or candidate.st_ino != file_stat.st_ino
        or not stat.S_ISREG(candidate.st_mode)
        or stat.S_IMODE(candidate.st_mode) != 0o600
        or candidate.st_nlink != 1
        or candidate.st_uid != os.geteuid()
      )
      for candidate in (file_stat, relative_stat, visible_stat)
    )
  ):
    raise RuntimeError(
      f"approval store file identity changed: {path}"
    )


def _secure_create_sqlite_file(path: Path) -> os.stat_result:
  parent_fd, bound_parent_stat = _open_secure_sqlite_parent(path)
  fd = -1
  created = False
  try:
    flags = (
      os.O_RDWR
      | os.O_CREAT
      | os.O_EXCL
      | getattr(os, "O_CLOEXEC", 0)
      | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(
      path.name,
      flags,
      0o600,
      dir_fd=parent_fd,
    )
    created = True
    os.fchmod(fd, 0o600)
    os.fsync(fd)
    file_stat = os.fstat(fd)
    _require_matching_sqlite_identity(
      path,
      parent_fd=parent_fd,
      bound_parent_stat=bound_parent_stat,
      file_stat=file_stat,
    )
    os.fsync(parent_fd)
    _require_matching_sqlite_identity(
      path,
      parent_fd=parent_fd,
      bound_parent_stat=bound_parent_stat,
      file_stat=os.fstat(fd),
    )
    return file_stat
  except BaseException as exc:
    if created:
      rollback_errors: list[BaseException] = []
      try:
        opened_stat = os.fstat(fd)
        relative_stat = os.stat(
          path.name,
          dir_fd=parent_fd,
          follow_symlinks=False,
        )
        if (
          relative_stat.st_dev != opened_stat.st_dev
          or relative_stat.st_ino != opened_stat.st_ino
        ):
          raise RuntimeError(
            "approval store create rollback identity changed"
          )
        os.unlink(path.name, dir_fd=parent_fd)
      except BaseException as rollback_exc:
        rollback_errors.append(rollback_exc)
      try:
        os.fsync(parent_fd)
      except BaseException as rollback_exc:
        rollback_errors.append(rollback_exc)
      if rollback_errors:
        exc.add_note(
          "approval store create rollback failed: "
          + "; ".join(
            f"{type(error).__name__}: {error}"
            for error in rollback_errors
          )
        )
    raise
  finally:
    if fd >= 0:
      os.close(fd)
    os.close(parent_fd)


def _require_secure_sqlite_file(
  path: Path,
  *,
  expected_device: int | None = None,
  expected_inode: int | None = None,
  repair_permissions: bool = False,
) -> os.stat_result:
  parent_fd, bound_parent_stat = _open_secure_sqlite_parent(path)
  flags = (
    os.O_RDWR
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
  )
  try:
    fd = os.open(path.name, flags, dir_fd=parent_fd)
  except OSError as exc:
    os.close(parent_fd)
    raise RuntimeError(
      f"approval store file is unavailable: {path}"
    ) from exc
  try:
    file_stat = os.fstat(fd)
    if (
      not stat.S_ISREG(file_stat.st_mode)
      or file_stat.st_nlink != 1
      or file_stat.st_uid != os.geteuid()
      or (
        expected_device is not None
        and file_stat.st_dev != expected_device
      )
      or (
        expected_inode is not None
        and file_stat.st_ino != expected_inode
      )
    ):
      raise RuntimeError(
        f"approval store file has unsafe identity: {path}"
      )
    if stat.S_IMODE(file_stat.st_mode) != 0o600:
      if not repair_permissions:
        raise RuntimeError(
          f"approval store file has unsafe permissions: {path}"
        )
      os.fchmod(fd, 0o600)
      os.fsync(fd)
      file_stat = os.fstat(fd)
    _require_matching_sqlite_identity(
      path,
      parent_fd=parent_fd,
      bound_parent_stat=bound_parent_stat,
      file_stat=file_stat,
    )
    os.fsync(parent_fd)
    file_stat = os.fstat(fd)
    _require_matching_sqlite_identity(
      path,
      parent_fd=parent_fd,
      bound_parent_stat=bound_parent_stat,
      file_stat=file_stat,
    )
    return file_stat
  finally:
    os.close(fd)
    os.close(parent_fd)


def _prepare_secure_sqlite_sidecar(path: Path) -> None:
  try:
    _secure_create_sqlite_file(path)
  except FileExistsError:
    _require_secure_sqlite_file(path)


def _canonical_delivery_text(
  value: Any,
  *,
  field_name: str,
  max_length: int = 512,
) -> str:
  if (
    type(value) is not str
    or not value
    or value != value.strip()
    or len(value) > max_length
    or "\x00" in value
  ):
    raise ValueError(
      f"autonomous approval delivery {field_name} is invalid"
    )
  return value


def _normalize_autonomous_approval_delivery_context(
  value: Mapping[str, Any],
) -> dict[str, str]:
  if not isinstance(value, Mapping) or set(value) != (
    _AUTONOMOUS_APPROVAL_DELIVERY_CONTEXT_FIELDS
  ):
    raise ValueError(
      "autonomous approval delivery context fields are invalid"
    )
  normalized = {
    field_name: _canonical_delivery_text(
      value[field_name],
      field_name=field_name,
    )
    for field_name in _AUTONOMOUS_APPROVAL_DELIVERY_CONTEXT_FIELDS
  }
  channel_id = normalized["channel_id"]
  if (
    len(channel_id) != 64
    or any(character not in "0123456789abcdef" for character in channel_id)
  ):
    raise ValueError(
      "autonomous approval delivery channel_id is invalid"
    )
  return normalized


def _datetime_to_epoch_ns(value: datetime) -> int:
  normalized = value.astimezone(UTC)
  epoch = datetime(1970, 1, 1, tzinfo=UTC)
  delta = normalized - epoch
  return (
    ((delta.days * 86_400) + delta.seconds) * 1_000_000_000
    + delta.microseconds * 1_000
  )


def _trusted_utc_now(
  conn: sqlite3.Connection,
) -> tuple[int, datetime]:
  """Advance the durable server-clock high-water or refuse rollback."""
  now_ns = time.time_ns()
  prior_row = conn.execute(
    """
    SELECT observed_at_ns FROM approval_clock_high_water
    WHERE clock_id = 1
    """
  ).fetchone()
  if (
    prior_row is not None
    and now_ns < int(prior_row["observed_at_ns"])
  ):
    raise RuntimeError(
      "approval clock rollback detected"
    )
  conn.execute(
    """
    INSERT INTO approval_clock_high_water (clock_id, observed_at_ns)
    VALUES (1, ?)
    ON CONFLICT(clock_id) DO UPDATE SET
      observed_at_ns = excluded.observed_at_ns
    """,
    (now_ns,),
  )
  seconds, nanoseconds = divmod(now_ns, 1_000_000_000)
  now = datetime.fromtimestamp(seconds, tz=UTC).replace(
    microsecond=nanoseconds // 1_000
  )
  return now_ns, now


def _autonomous_approval_delivery_projection(
  row: sqlite3.Row,
) -> dict[str, Any]:
  approved = int(row["approved"])
  allow_tool_type = int(row["allow_tool_type"])
  delivery_sequence = int(row["delivery_sequence"])
  if (
    approved not in {0, 1}
    or allow_tool_type != 0
    or delivery_sequence < 1
  ):
    raise RuntimeError(
      "autonomous approval delivery outbox contains invalid values"
    )
  return {
    "delivery_sequence": delivery_sequence,
    "approval_id": str(row["approval_id"]),
    "tool_call_id": str(row["tool_call_id"]),
    "nonce": str(row["nonce"]),
    "task_id": str(row["task_id"]),
    "control_run_id": str(row["control_run_id"]),
    "session_id": str(row["session_id"]),
    "channel_id": str(row["channel_id"]),
    "approved": bool(approved),
    "allow_tool_type": False,
    "decided_at_ns": int(row["decided_at_ns"]),
    "retry_deadline_ns": int(row["retry_deadline_ns"]),
    "next_attempt_ns": int(row["next_attempt_ns"]),
    "last_attempt_ns": (
      int(row["last_attempt_ns"])
      if row["last_attempt_ns"] is not None
      else None
    ),
    "state": str(row["state"]),
    "audit_state": str(row["audit_state"]),
    "triggering_vote_id": (
      str(row["triggering_vote_id"])
      if row["triggering_vote_id"] is not None
      else None
    ),
    "vote_audit_entry_id": (
      str(row["vote_audit_entry_id"])
      if row["vote_audit_entry_id"] is not None
      else None
    ),
    "terminal_audit_entry_id": str(
      row["terminal_audit_entry_id"]
    ),
    "attempt_count": int(row["attempt_count"]),
    "last_error": (
      str(row["last_error"])
      if row["last_error"] is not None
      else None
    ),
    "created_at": str(row["created_at"]),
    "updated_at": str(row["updated_at"]),
    "audit_ready_at": (
      str(row["audit_ready_at"])
      if row["audit_ready_at"] is not None
      else None
    ),
    "published_at": (
      str(row["published_at"])
      if row["published_at"] is not None
      else None
    ),
    "acknowledged_at": (
      str(row["acknowledged_at"])
      if row["acknowledged_at"] is not None
      else None
    ),
    "quarantined_at": (
      str(row["quarantined_at"])
      if row["quarantined_at"] is not None
      else None
    ),
  }


def _autonomous_approval_audit_entry_id(
  *,
  event_type: str,
  approval_id: str,
  tool_call_id: str,
  nonce: str,
  source_id: str,
  event_at_ns: int,
) -> str:
  identity = "\0".join((
    "autonomous-approval-delivery-audit-v1",
    event_type,
    approval_id,
    tool_call_id,
    nonce,
    source_id,
    str(event_at_ns),
  )).encode("utf-8")
  return (
    "autonomous-approval-delivery-v1:"
    f"{hashlib.sha256(identity).hexdigest()}"
  )


def _approval_vote_from_row(row: sqlite3.Row) -> ApprovalVote:
  decision = str(row["decision"])
  decided_at = _dt_from_text(str(row["decided_at"]))
  if decision not in {"approved", "denied"} or decided_at is None:
    raise RuntimeError(
      "autonomous approval delivery vote row is invalid"
    )
  return ApprovalVote(
    vote_id=str(row["vote_id"]),
    approval_id=str(row["approval_id"]),
    decider_id=str(row["decider_id"]),
    decider_role=(
      str(row["decider_role"])
      if row["decider_role"] is not None
      else None
    ),
    decision=decision,
    decision_reason=(
      str(row["decision_reason"])
      if row["decision_reason"] is not None
      else None
    ),
    decided_at=decided_at,
  )


def _insert_autonomous_approval_delivery_in_transaction(
  conn: sqlite3.Connection,
  *,
  request: ApprovalRequest,
  delivery: Mapping[str, str],
  approved: bool,
  vote: ApprovalVote | None,
) -> dict[str, Any]:
  expected_state = "approved" if approved else "denied"
  if request.state != expected_state:
    raise ValueError(
      "autonomous approval delivery disagrees with durable state"
    )
  if request.tool_call_id != delivery["tool_call_id"]:
    raise ValueError(
      "autonomous approval delivery tool-call identity changed"
    )
  if (
    request.request_id != delivery["control_run_id"]
    or request.run_id != delivery["control_run_id"]
    or request.session_id != delivery["session_id"]
  ):
    raise ValueError(
      "autonomous approval delivery run identity changed"
    )
  if request.decided_at is None:
    raise RuntimeError(
      "terminal autonomous approval is missing decided_at"
    )
  if vote is not None and (
    vote.approval_id != request.approval_id
    or vote.decision != expected_state
    or _datetime_to_epoch_ns(vote.decided_at)
    != _datetime_to_epoch_ns(request.decided_at)
  ):
    raise ValueError(
      "autonomous approval delivery vote identity changed"
    )
  decided_at_ns = _datetime_to_epoch_ns(request.decided_at)
  vote_audit_entry_id = (
    _autonomous_approval_audit_entry_id(
      event_type="vote_recorded",
      approval_id=request.approval_id,
      tool_call_id=request.tool_call_id,
      nonce=delivery["nonce"],
      source_id=vote.vote_id,
      event_at_ns=_datetime_to_epoch_ns(vote.decided_at),
    )
    if vote is not None
    else None
  )
  terminal_audit_entry_id = _autonomous_approval_audit_entry_id(
    event_type=expected_state,
    approval_id=request.approval_id,
    tool_call_id=request.tool_call_id,
    nonce=delivery["nonce"],
    source_id=vote.vote_id if vote is not None else "cancellation",
    event_at_ns=decided_at_ns,
  )
  delivery_now_ns, delivery_now_dt = _trusted_utc_now(conn)
  if (
    delivery_now_ns
    > _SQLITE_MAX_INTEGER
    - AUTONOMOUS_APPROVAL_DELIVERY_RETRY_WINDOW_NS
  ):
    raise RuntimeError(
      "autonomous approval delivery retry deadline overflow"
    )
  retry_deadline_ns = (
    delivery_now_ns
    + AUTONOMOUS_APPROVAL_DELIVERY_RETRY_WINDOW_NS
  )
  delivery_now = _dt_to_text(delivery_now_dt)
  sequence_row = conn.execute(
    """
    SELECT next_value
    FROM autonomous_approval_delivery_sequence
    WHERE sequence_id = 1
    """
  ).fetchone()
  if sequence_row is None:
    delivery_sequence = 1
    conn.execute(
      """
      INSERT INTO autonomous_approval_delivery_sequence (
        sequence_id, next_value
      ) VALUES (1, 2)
      """
    )
  else:
    delivery_sequence = int(sequence_row["next_value"])
    if not 1 <= delivery_sequence < _SQLITE_MAX_INTEGER:
      raise RuntimeError(
        "autonomous approval delivery sequence is invalid"
      )
    cursor = conn.execute(
      """
      UPDATE autonomous_approval_delivery_sequence
      SET next_value = ?
      WHERE sequence_id = 1 AND next_value = ?
      """,
      (delivery_sequence + 1, delivery_sequence),
    )
    if cursor.rowcount != 1:
      raise RuntimeError(
        "autonomous approval delivery sequence changed"
      )
  conn.execute(
    """
    INSERT INTO autonomous_approval_delivery_outbox (
      delivery_sequence, approval_id, tool_call_id, nonce,
      task_id, control_run_id,
      session_id, channel_id, approved, allow_tool_type,
      decided_at_ns, retry_deadline_ns, next_attempt_ns,
      last_attempt_ns, state, audit_state,
      triggering_vote_id,
      vote_audit_entry_id, terminal_audit_entry_id,
      attempt_count, last_error, created_at, updated_at,
      audit_ready_at, published_at, acknowledged_at, quarantined_at
    ) VALUES (
      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
    )
    """,
    (
      delivery_sequence,
      request.approval_id,
      request.tool_call_id,
      delivery["nonce"],
      delivery["task_id"],
      delivery["control_run_id"],
      delivery["session_id"],
      delivery["channel_id"],
      1 if approved else 0,
      0,
      decided_at_ns,
      retry_deadline_ns,
      delivery_now_ns,
      None,
      "pending",
      "pending",
      vote.vote_id if vote is not None else None,
      vote_audit_entry_id,
      terminal_audit_entry_id,
      0,
      None,
      delivery_now,
      delivery_now,
      None,
      None,
      None,
      None,
    ),
  )
  delivery_row = conn.execute(
    """
    SELECT * FROM autonomous_approval_delivery_outbox
    WHERE approval_id = ? AND tool_call_id = ? AND nonce = ?
    """,
    (
      request.approval_id,
      request.tool_call_id,
      delivery["nonce"],
    ),
  ).fetchone()
  if delivery_row is None:
    raise RuntimeError(
      "autonomous approval delivery outbox insert failed"
    )
  stored_delivery = _autonomous_approval_delivery_projection(
    delivery_row
  )
  expected_delivery = {
    "approval_id": request.approval_id,
    "tool_call_id": request.tool_call_id,
    "nonce": delivery["nonce"],
    "task_id": delivery["task_id"],
    "control_run_id": delivery["control_run_id"],
    "session_id": delivery["session_id"],
    "channel_id": delivery["channel_id"],
    "approved": approved,
    "allow_tool_type": False,
    "decided_at_ns": decided_at_ns,
    "triggering_vote_id": (
      vote.vote_id if vote is not None else None
    ),
    "vote_audit_entry_id": vote_audit_entry_id,
    "terminal_audit_entry_id": terminal_audit_entry_id,
  }
  if any(
    stored_delivery[field_name] != expected_value
    for field_name, expected_value in expected_delivery.items()
  ):
    raise ValueError(
      "autonomous approval delivery id was reused with different content"
    )
  return stored_delivery


class PersistentGrantCancellationFenced(RuntimeError):
  """A grant cannot become visible after its approval entered cancellation."""


class PreparedReconciliationConflict(StrEnum):
  MISSING_APPROVAL = "missing_approval"
  UNKNOWN_APPROVAL_STATE = "unknown_approval_state"
  LINEAGE_CONFLICT = "lineage_conflict"
  CAS_CONFLICT = "cas_conflict"


@dataclass(frozen=True)
class PreparedReconciliationCursor:
  caller_kind: str
  user_scope: str
  idempotency_locator: str

  @property
  def log_token(self) -> str:
    identity = "\0".join(
      (self.caller_kind, self.user_scope, self.idempotency_locator)
    ).encode("utf-8")
    return hashlib.sha256(identity).hexdigest()[:16]


@dataclass(frozen=True)
class PreparedReconciliationResult:
  scanned: int = 0
  authorized: int = 0
  denied: int = 0
  expired: int = 0
  missing_approval: int = 0
  unknown_approval_state: int = 0
  lineage_conflict: int = 0
  cas_conflict: int = 0
  cursor: PreparedReconciliationCursor | None = None
  wrapped: bool = False

  @property
  def conflict_count(self) -> int:
    return (
      self.missing_approval
      + self.unknown_approval_state
      + self.lineage_conflict
      + self.cas_conflict
    )


@dataclass(frozen=True)
class TargetedPreparedReconciliationResult:
  record: _prepared_bm.PreparedBusinessModelChange | None
  transitioned: bool = False
  conflict: PreparedReconciliationConflict | None = None


@dataclass(frozen=True)
class ApprovalMaintenanceResult:
  approvals_expired: int
  prepared: PreparedReconciliationResult


def resolve_approval_db_path(
  *,
  env_get: Any = os.getenv,
) -> Path:
  user_data_dir = str(
    env_get(USER_DATA_DIR_ENV, "") or ""
  ).strip()
  if not user_data_dir:
    raise RuntimeError(
      "approval database requires USER_DATA_DIR"
    )
  user_data_path = Path(user_data_dir).expanduser()
  if not user_data_path.is_absolute():
    raise ValueError(
      f"{USER_DATA_DIR_ENV} must be an absolute path"
    )
  canonical_path = (
    user_data_path / "gateway" / "approvals.sqlite3"
  ).resolve(strict=False)

  configured = str(env_get(APPROVAL_DB_PATH_ENV, "") or "").strip()
  if configured:
    path = Path(configured).expanduser()
    if not path.is_absolute():
      raise ValueError(f"{APPROVAL_DB_PATH_ENV} must be an absolute path")
    configured_path = path.resolve(strict=False)
    if configured_path != canonical_path:
      raise ValueError(
        f"{APPROVAL_DB_PATH_ENV} must equal "
        f"{USER_DATA_DIR_ENV}/gateway/approvals.sqlite3"
      )
  return canonical_path


_dt_to_text = _rows.dt_to_text
_dt_from_text = _rows.dt_from_text
_json_dumps = _rows.json_dumps
_json_loads = _rows.json_loads


class ApprovalRequestStore(Protocol):
  async def create(self, request: ApprovalRequest) -> ApprovalRequest: ...
  async def create_or_get_by_tool_call_id(
    self,
    request: ApprovalRequest,
  ) -> tuple[ApprovalRequest, bool]: ...
  async def get(self, approval_id: str) -> ApprovalRequest | None: ...
  async def get_by_tool_call_id(self, tool_call_id: str) -> ApprovalRequest | None: ...
  async def update_request(self, request: ApprovalRequest) -> ApprovalRequest: ...
  async def transition_state(
    self,
    approval_id: str,
    state: ApprovalState,
    *,
    expected_state_version: int | None = None,
    expires_at: datetime | None = None,
    decider_id: str | None = None,
    decider_role: str | None = None,
    decision: str | None = None,
    decision_reason: str | None = None,
  ) -> ApprovalRequest: ...
  async def force_deny_pending(
    self,
    approval_id: str,
    *,
    decider_id: str,
    decider_role: str | None = None,
    decision_reason: str | None = None,
  ) -> tuple[ApprovalRequest, bool]: ...
  async def terminalize_pending_for_cancellation(
    self,
    approval_id: str,
    *,
    expected_tool_call_id: str,
    expected_user_id: str,
    expected_request_id: str,
    expected_run_id: str,
    expected_session_id: str,
    expected_channel: str | None,
    decider_id: str | None = None,
    decider_role: str | None = None,
    decision_reason: str,
    autonomous_delivery: Mapping[str, Any] | None = None,
  ) -> tuple[ApprovalRequest, bool, bool]: ...
  async def abort_unpublished_approval(
    self,
    approval_id: str,
    *,
    expected_tool_call_id: str,
    expected_user_id: str,
    expected_request_id: str,
    expected_run_id: str,
    expected_session_id: str,
    expected_channel: str | None,
    decision_reason: str,
  ) -> tuple[ApprovalRequest, bool, bool]: ...
  async def record_vote(self, approval_id: str, vote: ApprovalVote) -> ApprovalRequest: ...
  async def create_persistent_grant(self, grant: PersistentGrant) -> PersistentGrant: ...
  async def find_persistent_grant(
    self,
    *,
    user_id: str,
    tool_name: str,
    scope_hint: str,
    now: datetime | None = None,
    approval_constraint: str = "standard",
  ) -> PersistentGrant | None: ...
  async def revoke_persistent_grant(self, grant_id: str, *, revoked_at: datetime | None = None) -> None: ...
  async def revoke_persistent_grants_for_approval(
    self,
    approval_id: str,
    *,
    revoked_at: datetime | None = None,
  ) -> int: ...
  async def fence_persistent_grants_for_cancellation(
    self,
    approval_id: str,
    *,
    expected_tool_call_id: str,
    expected_user_id: str,
    expected_request_id: str,
    expected_run_id: str,
    expected_session_id: str,
    expected_channel: str | None,
  ) -> tuple[ApprovalRequest, bool]: ...
  async def create_delegation_grant(self, grant: DelegationGrant) -> DelegationGrant: ...
  async def get_delegation_grant(self, delegation_id: str) -> DelegationGrant | None: ...
  async def claim_delegation_grant(
    self,
    *,
    delegation_id: str,
    bound_relay_request_id: str,
    bound_excel_session_id: str,
    now: datetime | None = None,
  ) -> DelegationGrant | None: ...
  async def revoke_delegation_grant(self, delegation_id: str, *, revoked_at: datetime | None = None) -> None: ...
  async def expire_pending(self, *, now: datetime | None = None) -> int: ...
  async def maintain_pending(
    self,
    *,
    now: datetime | None = None,
    prepared_cursor: PreparedReconciliationCursor | None = None,
    prepared_page_size: int = 100,
  ) -> ApprovalMaintenanceResult: ...
  async def reconcile_prepared_business_model_change(
    self,
    *,
    caller_kind: str,
    user_scope: str,
    idempotency_locator: str,
    now: datetime | None = None,
  ) -> TargetedPreparedReconciliationResult: ...


class SQLiteApprovalStore:
  """SQLite-backed approval store with atomic vote recording."""

  def __init__(
    self,
    path: str | os.PathLike[str],
    *,
    audit_emitter: Any | None = None,
    notification_destination_resolver: ApprovalNotificationDestinationResolver | None = None,
    notification_sender: ApprovalNotificationSender | None = None,
    expected_device: int | None = None,
    expected_inode: int | None = None,
  ) -> None:
    if (expected_device is None) != (expected_inode is None):
      raise ValueError(
        "approval store expected device and inode must be provided together"
      )
    if expected_device is not None and (
      type(expected_device) is not int
      or expected_device < 0
      or type(expected_inode) is not int
      or expected_inode < 1
    ):
      raise ValueError("approval store expected identity is invalid")
    self.path = Path(os.path.abspath(os.fspath(path)))
    _require_secure_sqlite_parent(self.path)
    try:
      initial_stat = _secure_create_sqlite_file(self.path)
    except FileExistsError:
      initial_stat = _require_secure_sqlite_file(
        self.path,
        expected_device=expected_device,
        expected_inode=expected_inode,
        repair_permissions=expected_device is None,
      )
    if (
      expected_device is not None
      and (
        initial_stat.st_dev != expected_device
        or initial_stat.st_ino != expected_inode
      )
    ):
      raise RuntimeError("approval store file identity changed")
    self._device = initial_stat.st_dev
    self._inode = initial_stat.st_ino
    self._audit_emitter = audit_emitter
    self._notification_destination_resolver = notification_destination_resolver
    self._notification_sender = notification_sender
    self._notification_delivery_task: asyncio.Task | None = None
    self._lock = asyncio.Lock()
    self._init_schema()

  @property
  def audit_emitter(self) -> Any | None:
    return self._audit_emitter

  def _require_bound_file(self) -> os.stat_result:
    _require_secure_sqlite_parent(self.path)
    return _require_secure_sqlite_file(
      self.path,
      expected_device=self._device,
      expected_inode=self._inode,
    )

  def _prepare_sidecars(self) -> None:
    for suffix in _SQLITE_SIDECAR_SUFFIXES:
      _prepare_secure_sqlite_sidecar(
        Path(f"{self.path}{suffix}")
      )

  def _require_sidecars(self) -> None:
    for suffix in _SQLITE_SIDECAR_SUFFIXES:
      _require_secure_sqlite_file(
        Path(f"{self.path}{suffix}")
      )

  def _connect(self) -> sqlite3.Connection:
    self._require_bound_file()
    self._prepare_sidecars()
    database_uri = f"{self.path.as_uri()}?mode=rw"
    conn = sqlite3.connect(
      database_uri,
      isolation_level=None,
      uri=True,
    )
    try:
      self._require_bound_file()
      conn.row_factory = sqlite3.Row
      conn.execute("PRAGMA foreign_keys=ON")
      journal_mode = str(
        conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
      ).lower()
      if journal_mode != "wal":
        raise RuntimeError(
          "approval store failed to enter canonical WAL mode"
        )
      conn.execute("PRAGMA synchronous=FULL")
      self._require_bound_file()
      self._require_sidecars()
      return conn
    except BaseException:
      conn.close()
      raise

  @contextmanager
  def _connection(self) -> Iterator[sqlite3.Connection]:
    conn = self._connect()
    try:
      with conn:
        yield conn
    finally:
      conn.close()

  def _init_schema(self) -> None:
    with self._connection() as conn:
      existing_outbox_columns = {
        str(row["name"])
        for row in conn.execute(
          "PRAGMA table_info(autonomous_approval_delivery_outbox)"
        ).fetchall()
      }
      if (
        existing_outbox_columns
        and existing_outbox_columns
        != _AUTONOMOUS_APPROVAL_DELIVERY_OUTBOX_COLUMNS
      ):
        if conn.execute(
          "SELECT 1 FROM autonomous_approval_delivery_outbox LIMIT 1"
        ).fetchone() is not None:
          raise RuntimeError(
            "noncanonical autonomous approval delivery outbox contains rows"
          )
        conn.execute("DROP TABLE autonomous_approval_delivery_outbox")
      conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS approval_requests (
          approval_id TEXT PRIMARY KEY,
          tool_call_id TEXT NOT NULL,
          parent_approval_id TEXT,
          approval_chain_id TEXT NOT NULL,
          delegation_id TEXT,
          request_id TEXT NOT NULL,
          session_id TEXT,
          run_id TEXT,
          user_id TEXT NOT NULL,
          profile TEXT NOT NULL,
          channel TEXT NOT NULL,
          tool_name TEXT NOT NULL,
          tool_class TEXT NOT NULL,
          tool_args_redacted TEXT NOT NULL,
          args_hash TEXT NOT NULL,
          reason TEXT,
          blast_radius_summary TEXT NOT NULL,
          state TEXT NOT NULL,
          state_version INTEGER NOT NULL DEFAULT 0,
          requested_at TEXT NOT NULL,
          decided_at TEXT,
          expires_at TEXT,
          decider_id TEXT,
          decider_role TEXT,
          decision TEXT,
          decision_reason TEXT,
          args_predicate TEXT,
          policy_id TEXT NOT NULL,
          policy_version TEXT NOT NULL,
          policy_bundle_hash TEXT NOT NULL,
          persistent_grant_scope TEXT,
          tenant_id TEXT,
          skill TEXT,
          notification_policy TEXT NOT NULL DEFAULT 'auto',
          approval_constraint TEXT NOT NULL DEFAULT 'legacy_unknown'
            CHECK (approval_constraint IN ('standard', 'fresh_human_owner', 'legacy_unknown')),
          required_owner_user_id TEXT,
          identity_source TEXT,
          change_set_id TEXT,
          change_hash TEXT,
          base_vector_hash TEXT,
          reviewed_change_binding_digest TEXT,
          review_reference_json TEXT,
          execution_semantics_digest TEXT,
          authorization_mode TEXT NOT NULL DEFAULT 'HUMAN',
          grant_reference TEXT,
          cache_reference TEXT,
          CHECK (
            (approval_constraint = 'fresh_human_owner'
              AND required_owner_user_id IS NOT NULL
              AND length(trim(required_owner_user_id)) > 0)
            OR
            (approval_constraint IN ('standard', 'legacy_unknown')
              AND required_owner_user_id IS NULL)
          )
        );
        CREATE INDEX IF NOT EXISTS idx_approval_requests_tool_call_id
          ON approval_requests(tool_call_id);
        CREATE INDEX IF NOT EXISTS idx_approval_requests_state_expires
          ON approval_requests(state, expires_at);

        CREATE TABLE IF NOT EXISTS approval_votes (
          vote_id TEXT PRIMARY KEY,
          approval_id TEXT NOT NULL REFERENCES approval_requests(approval_id) ON DELETE CASCADE,
          decider_id TEXT NOT NULL,
          decider_role TEXT,
          decision TEXT NOT NULL,
          decision_reason TEXT,
          decided_at TEXT NOT NULL,
          UNIQUE(approval_id, decider_id)
        );

        CREATE TABLE IF NOT EXISTS persistent_grants (
          grant_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          tool_name TEXT NOT NULL,
          scope_hint TEXT NOT NULL,
          args_predicate TEXT,
          granted_at TEXT NOT NULL,
          expires_at TEXT,
          revoked_at TEXT,
          granted_via_approval_id TEXT NOT NULL,
          policy_id TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_persistent_grants_lookup
          ON persistent_grants(user_id, tool_name, scope_hint, revoked_at, expires_at);

        CREATE TABLE IF NOT EXISTS persistent_grant_cancellation_fences (
          approval_id TEXT PRIMARY KEY
            REFERENCES approval_requests(approval_id) ON DELETE CASCADE,
          fenced_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS delegation_grants (
          delegation_id TEXT PRIMARY KEY,
          delegator_user_id TEXT NOT NULL,
          delegator_run_id TEXT,
          delegator_session_id TEXT,
          delegator_profile TEXT NOT NULL,
          delegator_channel TEXT NOT NULL,
          bound_excel_session_id TEXT NOT NULL,
          bound_relay_request_id TEXT NOT NULL,
          bound_workbook TEXT,
          tool_class_ceiling TEXT NOT NULL,
          args_predicate TEXT,
          window_seconds INTEGER NOT NULL,
          exclude_external_write_bypass INTEGER NOT NULL DEFAULT 1,
          created_at TEXT NOT NULL,
          expires_at TEXT,
          revoked_at TEXT,
          consumed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_delegation_grants_lookup
          ON delegation_grants(delegator_user_id, bound_excel_session_id, bound_relay_request_id);

        CREATE TABLE IF NOT EXISTS approval_notification_outbox (
          approval_id TEXT NOT NULL REFERENCES approval_requests(approval_id) ON DELETE CASCADE,
          channel TEXT NOT NULL,
          destination TEXT NOT NULL,
          state TEXT NOT NULL,
          message TEXT NOT NULL,
          dedupe_key TEXT NOT NULL,
          attempt_count INTEGER NOT NULL DEFAULT 0,
          last_error TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          sent_at TEXT,
          PRIMARY KEY (approval_id, channel, destination)
        );
        CREATE INDEX IF NOT EXISTS idx_approval_notification_outbox_state
          ON approval_notification_outbox(state, updated_at);

        CREATE TABLE IF NOT EXISTS approval_clock_high_water (
          clock_id INTEGER PRIMARY KEY CHECK (clock_id = 1),
          observed_at_ns INTEGER NOT NULL CHECK (observed_at_ns > 0)
        );

        CREATE TABLE IF NOT EXISTS autonomous_approval_delivery_sequence (
          sequence_id INTEGER PRIMARY KEY CHECK (sequence_id = 1),
          next_value INTEGER NOT NULL CHECK (next_value > 0)
        );

        CREATE TABLE IF NOT EXISTS autonomous_approval_delivery_outbox (
          delivery_sequence INTEGER NOT NULL UNIQUE
            CHECK (delivery_sequence > 0),
          approval_id TEXT NOT NULL
            REFERENCES approval_requests(approval_id) ON DELETE CASCADE,
          tool_call_id TEXT NOT NULL,
          nonce TEXT NOT NULL,
          task_id TEXT NOT NULL,
          control_run_id TEXT NOT NULL,
          session_id TEXT NOT NULL,
          channel_id TEXT NOT NULL,
          approved INTEGER NOT NULL CHECK (approved IN (0, 1)),
          allow_tool_type INTEGER NOT NULL DEFAULT 0
            CHECK (allow_tool_type = 0),
          decided_at_ns INTEGER NOT NULL CHECK (decided_at_ns > 0),
          retry_deadline_ns INTEGER NOT NULL
            CHECK (retry_deadline_ns > 0),
          next_attempt_ns INTEGER NOT NULL
            CHECK (next_attempt_ns > 0),
          last_attempt_ns INTEGER
            CHECK (last_attempt_ns IS NULL OR last_attempt_ns > 0),
          state TEXT NOT NULL DEFAULT 'pending'
            CHECK (state IN (
              'pending', 'published', 'acknowledged', 'quarantined'
            )),
          audit_state TEXT NOT NULL DEFAULT 'pending'
            CHECK (audit_state IN ('pending', 'ready')),
          triggering_vote_id TEXT
            REFERENCES approval_votes(vote_id),
          vote_audit_entry_id TEXT,
          terminal_audit_entry_id TEXT NOT NULL,
          attempt_count INTEGER NOT NULL DEFAULT 0
            CHECK (attempt_count >= 0),
          last_error TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          audit_ready_at TEXT,
          published_at TEXT,
          acknowledged_at TEXT,
          quarantined_at TEXT,
          PRIMARY KEY (approval_id, tool_call_id, nonce),
          UNIQUE (approval_id),
          CHECK (
            (triggering_vote_id IS NULL
              AND vote_audit_entry_id IS NULL)
            OR
            (triggering_vote_id IS NOT NULL
              AND vote_audit_entry_id IS NOT NULL)
          ),
          CHECK (
            (audit_state = 'pending' AND audit_ready_at IS NULL)
            OR
            (audit_state = 'ready' AND audit_ready_at IS NOT NULL)
          ),
          CHECK (
            state NOT IN ('published', 'acknowledged')
            OR audit_state = 'ready'
          ),
          CHECK (
            (state = 'pending'
              AND published_at IS NULL
              AND acknowledged_at IS NULL
              AND quarantined_at IS NULL)
            OR
            (state = 'published'
              AND published_at IS NOT NULL
              AND acknowledged_at IS NULL
              AND quarantined_at IS NULL)
            OR
            (state = 'acknowledged'
              AND published_at IS NOT NULL
              AND acknowledged_at IS NOT NULL
              AND quarantined_at IS NULL)
            OR
            (state = 'quarantined'
              AND published_at IS NULL
              AND acknowledged_at IS NULL
              AND quarantined_at IS NOT NULL)
          )
        );
        CREATE INDEX IF NOT EXISTS idx_autonomous_approval_delivery_state
          ON autonomous_approval_delivery_outbox(
            state, audit_state, updated_at
          );
        """
      )
      outbox_columns = {
        str(row["name"])
        for row in conn.execute(
          "PRAGMA table_info(autonomous_approval_delivery_outbox)"
        ).fetchall()
      }
      if (
        outbox_columns
        != _AUTONOMOUS_APPROVAL_DELIVERY_OUTBOX_COLUMNS
      ):
        raise RuntimeError(
          "autonomous approval delivery outbox schema is not canonical"
        )
      sequence_row = conn.execute(
        """
        SELECT next_value
        FROM autonomous_approval_delivery_sequence
        WHERE sequence_id = 1
        """
      ).fetchone()
      high_water_row = conn.execute(
        """
        SELECT COALESCE(MAX(delivery_sequence), 0) AS high_water
        FROM autonomous_approval_delivery_outbox
        """
      ).fetchone()
      if high_water_row is None:
        raise RuntimeError(
          "autonomous approval delivery sequence query failed"
        )
      high_water = int(high_water_row["high_water"])
      if (
        (high_water > 0 and sequence_row is None)
        or (
          sequence_row is not None
          and int(sequence_row["next_value"]) <= high_water
        )
      ):
        raise RuntimeError(
          "autonomous approval delivery sequence is inconsistent"
        )
      _trusted_utc_now(conn)
      columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(approval_requests)").fetchall()}
      if "skill" not in columns:
        conn.execute("ALTER TABLE approval_requests ADD COLUMN skill TEXT")
      if "delegation_id" not in columns:
        conn.execute("ALTER TABLE approval_requests ADD COLUMN delegation_id TEXT")
      if "notification_policy" not in columns:
        conn.execute("ALTER TABLE approval_requests ADD COLUMN notification_policy TEXT NOT NULL DEFAULT 'auto'")
      identity_columns = {
        "approval_constraint": (
          "TEXT NOT NULL DEFAULT 'legacy_unknown' "
          "CHECK (approval_constraint IN "
          "('standard', 'fresh_human_owner', 'legacy_unknown'))"
        ),
        "required_owner_user_id": "TEXT",
        "identity_source": "TEXT",
        "change_set_id": "TEXT",
        "change_hash": "TEXT",
        "base_vector_hash": "TEXT",
        "reviewed_change_binding_digest": "TEXT",
        "review_reference_json": "TEXT",
        "execution_semantics_digest": "TEXT",
        "authorization_mode": "TEXT NOT NULL DEFAULT 'HUMAN'",
        "grant_reference": "TEXT",
        "cache_reference": "TEXT",
      }
      for column_name, column_type in identity_columns.items():
        if column_name not in columns:
          conn.execute(
            f"ALTER TABLE approval_requests ADD COLUMN {column_name} {column_type}"
          )
      _prepared_bm.ensure_schema(conn)
      _raw_patch_auth.ensure_schema(conn)

  def _insert_request(
    self,
    conn: sqlite3.Connection,
    request: ApprovalRequest,
  ) -> None:
    conn.execute(
      """
          INSERT INTO approval_requests (
            approval_id, tool_call_id, parent_approval_id, approval_chain_id,
            delegation_id, request_id, session_id, run_id, user_id, profile, channel,
            tool_name, tool_class, tool_args_redacted, args_hash, reason,
            blast_radius_summary, state, state_version, requested_at, decided_at,
            expires_at, decider_id, decider_role, decision, decision_reason,
            args_predicate, policy_id, policy_version, policy_bundle_hash,
            persistent_grant_scope, tenant_id, skill,
            notification_policy, approval_constraint, required_owner_user_id,
            identity_source, change_set_id, change_hash,
            base_vector_hash, reviewed_change_binding_digest,
            review_reference_json, execution_semantics_digest,
            authorization_mode, grant_reference, cache_reference
          ) VALUES (
            :approval_id, :tool_call_id, :parent_approval_id, :approval_chain_id,
            :delegation_id, :request_id, :session_id, :run_id, :user_id, :profile, :channel,
            :tool_name, :tool_class, :tool_args_redacted, :args_hash, :reason,
            :blast_radius_summary, :state, :state_version, :requested_at, :decided_at,
            :expires_at, :decider_id, :decider_role, :decision, :decision_reason,
            :args_predicate, :policy_id, :policy_version, :policy_bundle_hash,
            :persistent_grant_scope, :tenant_id, :skill,
            :notification_policy, :approval_constraint, :required_owner_user_id,
            :identity_source, :change_set_id, :change_hash,
            :base_vector_hash, :reviewed_change_binding_digest,
            :review_reference_json, :execution_semantics_digest,
            :authorization_mode, :grant_reference, :cache_reference
          )
      """,
      self._request_to_row(request),
    )

  async def create(self, request: ApprovalRequest) -> ApprovalRequest:
    request = revalidate_approval_request(request)
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        self._insert_request(conn, request)
        conn.commit()
    await self._emit("request_created", request)
    return request

  async def create_raw_patch_authorization(
    self,
    request: ApprovalRequest,
    *,
    prepared_payload: bytes,
  ) -> ApprovalRequest:
    """Atomically create an approval and bind its exact reviewed patch bytes."""

    request = revalidate_approval_request(request)
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        self._insert_request(conn, request)
        _raw_patch_auth.bind_prepared(
          conn,
          approval_id=request.approval_id,
          prepared_payload=prepared_payload,
        )
        conn.commit()
    await self._emit("request_created", request)
    return request

  async def create_or_get_by_tool_call_id(
    self,
    request: ApprovalRequest,
  ) -> tuple[ApprovalRequest, bool]:
    """Atomically reuse one durable request across processes and store instances."""

    request = revalidate_approval_request(request)
    created = False
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
          """
          SELECT * FROM approval_requests
          WHERE tool_call_id = ?
          ORDER BY requested_at DESC
          LIMIT 1
          """,
          (request.tool_call_id,),
        ).fetchone()
        if row is None:
          self._insert_request(conn, request)
          stored = request
          created = True
        else:
          stored = self._row_to_request_with_projection(row)
          if (
            stored.approval_constraint != request.approval_constraint
            or stored.required_owner_user_id != request.required_owner_user_id
          ):
            raise ValueError(
              "approval constraint conflicts with existing tool-call identity"
            )
        conn.commit()
    if created:
      await self._emit("request_created", stored)
    return stored, created

  async def create_or_get_with_prepared_business_model_change(
    self,
    request: ApprovalRequest,
    record: _prepared_bm.PreparedBusinessModelChange,
  ) -> tuple[
    ApprovalRequest,
    _prepared_bm.PreparedBusinessModelChange,
    bool,
  ]:
    """Atomically bind one approval request to its immutable prepared plan."""

    request = revalidate_approval_request(request)
    created = False
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
          row = conn.execute(
            """
            SELECT * FROM approval_requests
            WHERE tool_call_id = ?
            ORDER BY requested_at DESC
            LIMIT 1
            """,
            (request.tool_call_id,),
          ).fetchone()
          if row is None:
            self._insert_request(conn, request)
            stored_request = request
            created = True
          else:
            stored_request = self._row_to_request_with_projection(row)
            if (
              stored_request.tool_name != request.tool_name
              or stored_request.args_hash != request.args_hash
              or stored_request.user_id != request.user_id
              or stored_request.approval_constraint
              != request.approval_constraint
              or stored_request.required_owner_user_id
              != request.required_owner_user_id
            ):
              raise _prepared_bm.PreparedBusinessModelError(
                "approval tool-call identity conflicts with prepared BusinessModel intent"
              )
          normalized_record = replace(
            record,
            created_at=stored_request.requested_at.astimezone(UTC).isoformat(),
            expires_at=(
              stored_request.expires_at.astimezone(UTC).isoformat()
              if stored_request.expires_at is not None
              else None
            ),
            approval_id=stored_request.approval_id,
            approval_chain_id=stored_request.approval_chain_id,
          )
          stored_record, _ = _prepared_bm.insert_or_verify(
            conn,
            normalized_record,
          )
          conn.commit()
        except BaseException:
          conn.rollback()
          raise
    if created:
      await self._emit("request_created", stored_request)
    return stored_request, stored_record, created

  async def get(self, approval_id: str) -> ApprovalRequest | None:
    with self._connection() as conn:
      row = conn.execute(
        "SELECT * FROM approval_requests WHERE approval_id = ?",
        (approval_id,),
      ).fetchone()
    return self._row_to_request_with_projection(row) if row is not None else None

  async def get_by_tool_call_id(self, tool_call_id: str) -> ApprovalRequest | None:
    with self._connection() as conn:
      row = conn.execute(
        "SELECT * FROM approval_requests WHERE tool_call_id = ? ORDER BY requested_at DESC LIMIT 1",
        (tool_call_id,),
      ).fetchone()
    return self._row_to_request_with_projection(row) if row is not None else None

  async def update_request(self, request: ApprovalRequest) -> ApprovalRequest:
    request = revalidate_approval_request(request)
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current_row = conn.execute(
          "SELECT * FROM approval_requests WHERE approval_id = ?",
          (request.approval_id,),
        ).fetchone()
        if current_row is None:
          raise KeyError(f"approval request not found: {request.approval_id}")
        if _raw_patch_auth.get_claim(
          conn,
          approval_id=request.approval_id,
        ) is not None:
          raise _raw_patch_auth.RawPatchAuthorizationError(
            "claimed raw patch approval cannot be modified"
          )
        current = self._row_to_request(current_row)
        identity_fields = (
          "approval_constraint",
          "required_owner_user_id",
          "identity_source",
          "change_set_id",
          "change_hash",
          "base_vector_hash",
          "reviewed_change_binding_digest",
          "review_reference",
          "execution_semantics_digest",
        )
        if any(
          getattr(current, field_name) != getattr(request, field_name)
          for field_name in identity_fields
        ):
          raise ValueError("approval request identity is immutable")
        conn.execute(
          """
          UPDATE approval_requests SET
            tool_args_redacted = :tool_args_redacted,
            args_hash = :args_hash,
            reason = :reason,
            blast_radius_summary = :blast_radius_summary,
            args_predicate = :args_predicate,
            policy_id = :policy_id,
            policy_version = :policy_version,
            policy_bundle_hash = :policy_bundle_hash,
            persistent_grant_scope = :persistent_grant_scope,
            notification_policy = :notification_policy,
            authorization_mode = :authorization_mode,
            grant_reference = :grant_reference,
            cache_reference = :cache_reference
          WHERE approval_id = :approval_id
          """,
          self._request_to_row(request),
        )
        conn.commit()
    return request

  async def transition_state(
    self,
    approval_id: str,
    state: ApprovalState,
    *,
    expected_state_version: int | None = None,
    expires_at: datetime | None = None,
    decider_id: str | None = None,
    decider_role: str | None = None,
    decision: str | None = None,
    decision_reason: str | None = None,
  ) -> ApprovalRequest:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current_row = conn.execute(
          "SELECT * FROM approval_requests WHERE approval_id = ?",
          (approval_id,),
        ).fetchone()
        if current_row is None:
          raise KeyError(f"approval request not found: {approval_id}")
        if _raw_patch_auth.get_claim(conn, approval_id=approval_id) is not None:
          raise _raw_patch_auth.RawPatchAuthorizationError(
            "claimed raw patch approval cannot transition"
          )
        current = self._row_to_request(current_row)
        if expected_state_version is not None and current.state_version != expected_state_version:
          raise RuntimeError("approval request state_version changed")
        if state in {"approved", "auto_approved"}:
          if current.approval_constraint == "legacy_unknown":
            raise ValueError(
              "approval constraint is unknown; replan and reauthorize"
            )
          if current.approval_constraint == "fresh_human_owner":
            raise ValueError(
              "fresh owner approval constraint can only be approved by an owner vote"
            )
        decided_at = utc_now() if state in TERMINAL_STATES else current.decided_at
        terminal_decision = decision
        if terminal_decision is None and state in {"approved", "denied", "auto_approved", "auto_denied", "expired"}:
          terminal_decision = state
        updated = replace(
          current,
          state=state,
          state_version=current.state_version + 1,
          expires_at=expires_at if expires_at is not None else current.expires_at,
          decided_at=decided_at,
          decider_id=decider_id if decider_id is not None else current.decider_id,
          decider_role=decider_role if decider_role is not None else current.decider_role,
          decision=terminal_decision,  # type: ignore[arg-type]
          decision_reason=decision_reason if decision_reason is not None else current.decision_reason,
        )
        conn.execute(
          """
          UPDATE approval_requests SET
            state = :state,
            state_version = :state_version,
            expires_at = :expires_at,
            decided_at = :decided_at,
            decider_id = :decider_id,
            decider_role = :decider_role,
            decision = :decision,
            decision_reason = :decision_reason
          WHERE approval_id = :approval_id
          """,
          self._request_to_row(updated),
        )
        conn.commit()
    await self._emit(self._event_type_for_state(updated.state), updated)
    return updated

  async def force_deny_pending(
    self,
    approval_id: str,
    *,
    decider_id: str,
    decider_role: str | None = None,
    decision_reason: str | None = None,
  ) -> tuple[ApprovalRequest, bool]:
    """Atomically deny a still-pending request for terminal runtime teardown.

    This is deliberately distinct from a vote: cancelling the runtime that owns
    an approval must close the durable request regardless of its quorum size.
    A terminal decision that wins the store lock is never overwritten.
    """
    transitioned = False
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current_row = conn.execute(
          "SELECT * FROM approval_requests WHERE approval_id = ?",
          (approval_id,),
        ).fetchone()
        if current_row is None:
          raise KeyError(f"approval request not found: {approval_id}")
        current = self._row_to_request(current_row)
        if current.state in TERMINAL_STATES:
          conn.commit()
          return current, False
        if current.state != "pending_user":
          raise RuntimeError(
            f"approval request is not pending_user: {approval_id} ({current.state})"
          )
        if _raw_patch_auth.get_claim(conn, approval_id=approval_id) is not None:
          raise _raw_patch_auth.RawPatchAuthorizationError(
            "claimed raw patch approval cannot transition"
          )
        updated = replace(
          current,
          state="denied",
          state_version=current.state_version + 1,
          decided_at=utc_now(),
          decider_id=decider_id,
          decider_role=decider_role,
          decision="denied",
          decision_reason=decision_reason,
        )
        conn.execute(
          """
          UPDATE approval_requests SET
            state = :state,
            state_version = :state_version,
            decided_at = :decided_at,
            decider_id = :decider_id,
            decider_role = :decider_role,
            decision = :decision,
            decision_reason = :decision_reason
          WHERE approval_id = :approval_id
          """,
          self._request_to_row(updated),
        )
        conn.commit()
        transitioned = True
    if transitioned:
      await self._emit(self._event_type_for_state(updated.state), updated)
    return updated, transitioned

  async def terminalize_pending_for_cancellation(
    self,
    approval_id: str,
    *,
    expected_tool_call_id: str,
    expected_user_id: str,
    expected_request_id: str,
    expected_run_id: str,
    expected_session_id: str,
    expected_channel: str | None,
    decider_id: str | None = None,
    decider_role: str | None = None,
    decision_reason: str,
    autonomous_delivery: Mapping[str, Any] | None = None,
  ) -> tuple[ApprovalRequest, bool, bool]:
    """Atomically deny a still-pending approval without quorum semantics.

    Cancellation is a lifecycle boundary, not an approval vote. Serializing
    the full durable-identity join, pending-state check, and terminal transition
    in one SQLite write transaction prevents cross-run mutation and release
    after the owning run has begun teardown.
    """

    normalized_delivery = (
      _normalize_autonomous_approval_delivery_context(
        autonomous_delivery
      )
      if autonomous_delivery is not None
      else None
    )
    transitioned = False
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current_row = conn.execute(
          "SELECT * FROM approval_requests WHERE approval_id = ?",
          (approval_id,),
        ).fetchone()
        if current_row is None:
          conn.rollback()
          raise KeyError(f"approval request not found: {approval_id}")
        current = self._row_to_request(current_row)
        normalized_expected_channel = str(expected_channel or "").strip().lower()
        identity_matches = all((
          current.tool_call_id == expected_tool_call_id,
          current.user_id == expected_user_id,
          current.request_id == expected_request_id,
          current.run_id == expected_run_id,
          current.session_id == expected_session_id,
          str(current.channel or "").strip().lower() == normalized_expected_channel,
        ))
        if not identity_matches:
          conn.commit()
          return current, False, False
        if current.state != "pending_user":
          conn.commit()
          return current, False, True
        if _raw_patch_auth.get_claim(conn, approval_id=approval_id) is not None:
          conn.rollback()
          raise _raw_patch_auth.RawPatchAuthorizationError(
            "claimed raw patch approval cannot be cancelled"
          )
        updated = replace(
          current,
          state="denied",
          state_version=current.state_version + 1,
          decided_at=utc_now(),
          decider_id=decider_id if decider_id is not None else current.decider_id,
          decider_role=decider_role if decider_role is not None else current.decider_role,
          decision="denied",
          decision_reason=decision_reason,
        )
        cursor = conn.execute(
          """
          UPDATE approval_requests SET
            state = :state,
            state_version = :state_version,
            decided_at = :decided_at,
            decider_id = :decider_id,
            decider_role = :decider_role,
            decision = :decision,
            decision_reason = :decision_reason
          WHERE approval_id = :approval_id
            AND state = 'pending_user'
            AND state_version = :expected_state_version
            AND tool_call_id = :expected_tool_call_id
            AND user_id = :expected_user_id
            AND request_id = :expected_request_id
            AND run_id = :expected_run_id
            AND session_id = :expected_session_id
            AND LOWER(TRIM(channel)) = :expected_channel
          """,
          {
            **self._request_to_row(updated),
            "expected_state_version": current.state_version,
            "expected_tool_call_id": expected_tool_call_id,
            "expected_user_id": expected_user_id,
            "expected_request_id": expected_request_id,
            "expected_run_id": expected_run_id,
            "expected_session_id": expected_session_id,
            "expected_channel": normalized_expected_channel,
          },
        )
        if cursor.rowcount != 1:
          conn.rollback()
          raise RuntimeError("approval cancellation compare-and-set failed")
        if normalized_delivery is not None:
          _insert_autonomous_approval_delivery_in_transaction(
            conn,
            request=updated,
            delivery=normalized_delivery,
            approved=False,
            vote=None,
          )
        conn.commit()
        transitioned = True
    if transitioned:
      if normalized_delivery is None:
        await self._emit("denied", updated)
      elif callable(getattr(
        self._audit_emitter,
        "emit_audit_for_lifecycle_event",
        None,
      )):
        await self.ensure_autonomous_approval_delivery_audited(
          updated.approval_id,
          tool_call_id=normalized_delivery["tool_call_id"],
          nonce=normalized_delivery["nonce"],
        )
    return updated, transitioned, True

  async def abort_unpublished_approval(
    self,
    approval_id: str,
    *,
    expected_tool_call_id: str,
    expected_user_id: str,
    expected_request_id: str,
    expected_run_id: str,
    expected_session_id: str,
    expected_channel: str | None,
    decision_reason: str,
  ) -> tuple[ApprovalRequest, bool, bool]:
    """Deny an admitted request that unwound before live publication."""
    transitioned = False
    revoked_grant_rows: list[sqlite3.Row] = []
    now = utc_now()
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current_row = conn.execute(
          "SELECT * FROM approval_requests WHERE approval_id = ?",
          (approval_id,),
        ).fetchone()
        if current_row is None:
          conn.commit()
          raise KeyError(f"approval request not found: {approval_id}")
        current = self._row_to_request(current_row)
        normalized_expected_channel = str(expected_channel or "").strip().lower()
        identity_matches = all((
          current.tool_call_id == expected_tool_call_id,
          current.user_id == expected_user_id,
          current.request_id == expected_request_id,
          current.run_id == expected_run_id,
          current.session_id == expected_session_id,
          str(current.channel or "").strip().lower() == normalized_expected_channel,
        ))
        if not identity_matches:
          conn.commit()
          return current, False, False
        if _raw_patch_auth.get_claim(conn, approval_id=approval_id) is not None:
          conn.rollback()
          raise _raw_patch_auth.RawPatchAuthorizationError(
            "claimed raw patch approval cannot be aborted"
          )
        updated = current
        if current.state not in TERMINAL_STATES:
          updated = replace(
            current,
            state="denied",
            state_version=current.state_version + 1,
            decided_at=now,
            decision="denied",
            decision_reason=decision_reason,
          )
          conn.execute(
            """
            UPDATE approval_requests SET
              state = :state,
              state_version = :state_version,
              decided_at = :decided_at,
              decision = :decision,
              decision_reason = :decision_reason
            WHERE approval_id = :approval_id
            """,
            self._request_to_row(updated),
          )
          transitioned = True
        revoked_grant_rows = conn.execute(
          """
          SELECT * FROM persistent_grants
          WHERE granted_via_approval_id = ? AND revoked_at IS NULL
          """,
          (approval_id,),
        ).fetchall()
        conn.execute(
          """
          UPDATE persistent_grants SET revoked_at = ?
          WHERE granted_via_approval_id = ? AND revoked_at IS NULL
          """,
          (_dt_to_text(now), approval_id),
        )
        conn.execute(
          "DELETE FROM approval_notification_outbox WHERE approval_id = ?",
          (approval_id,),
        )
        conn.commit()
    if transitioned:
      await self._emit("denied", updated)
    for row in revoked_grant_rows:
      await self._emit_grant(
        "persistent_grant_revoked",
        replace(self._row_to_grant(row), revoked_at=now),
        request=updated,
      )
    return updated, transitioned, True

  async def record_vote(self, approval_id: str, vote: ApprovalVote) -> ApprovalRequest:
    return await self._record_vote(
      approval_id,
      vote,
      autonomous_delivery=None,
    )

  async def record_vote_with_autonomous_delivery(
    self,
    approval_id: str,
    vote: ApprovalVote,
    *,
    delivery: Mapping[str, Any],
  ) -> ApprovalRequest:
    return await self._record_vote(
      approval_id,
      vote,
      autonomous_delivery=delivery,
    )

  async def _record_vote(
    self,
    approval_id: str,
    vote: ApprovalVote,
    *,
    autonomous_delivery: Mapping[str, Any] | None,
  ) -> ApprovalRequest:
    if vote.approval_id != approval_id:
      raise ValueError("approval vote identity does not match request")
    normalized_delivery = (
      _normalize_autonomous_approval_delivery_context(
        autonomous_delivery
      )
      if autonomous_delivery is not None
      else None
    )
    if normalized_delivery is not None:
      existing_delivery = (
        await self.get_autonomous_approval_delivery(
          approval_id,
          tool_call_id=normalized_delivery["tool_call_id"],
          nonce=normalized_delivery["nonce"],
        )
      )
      if existing_delivery is not None:
        expected_decision = vote.decision == "approved"
        expected_delivery = {
          **normalized_delivery,
          "approval_id": approval_id,
          "approved": expected_decision,
          "allow_tool_type": False,
        }
        if any(
          existing_delivery[field_name] != expected_value
          for field_name, expected_value
          in expected_delivery.items()
        ):
          raise ValueError(
            "autonomous approval delivery retry identity changed"
          )
        current = await self.get(approval_id)
        expected_state = (
          "approved" if expected_decision else "denied"
        )
        if (
          current is None
          or current.state != expected_state
          or current.decision != expected_state
          or current.decider_id != vote.decider_id
        ):
          raise ValueError(
            "autonomous approval delivery retry decision changed"
          )
        await self.ensure_autonomous_approval_delivery_audited(
          approval_id,
          tool_call_id=normalized_delivery["tool_call_id"],
          nonce=normalized_delivery["nonce"],
        )
        return current
    emitted_vote = False
    terminal_event: str | None = None
    updated: ApprovalRequest
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current_row = conn.execute(
          "SELECT * FROM approval_requests WHERE approval_id = ?",
          (approval_id,),
        ).fetchone()
        if current_row is None:
          raise KeyError(f"approval request not found: {approval_id}")
        current = self._row_to_request(current_row)
        if (
          normalized_delivery is not None
          and current.tool_call_id != normalized_delivery["tool_call_id"]
        ):
          raise ValueError(
            "autonomous approval delivery tool-call identity changed"
          )
        if current.state in TERMINAL_STATES:
          conn.commit()
          return current

        trusted_now_ns, trusted_now = _trusted_utc_now(conn)
        if (
          current.expires_at is not None
          and trusted_now_ns >= _datetime_to_epoch_ns(current.expires_at)
        ):
          updated = replace(
            current,
            state="expired",
            state_version=current.state_version + 1,
            decided_at=trusted_now,
            decision="expired",
          )
          cursor = conn.execute(
            """
            UPDATE approval_requests SET
              state = :state,
              state_version = :state_version,
              decided_at = :decided_at,
              decision = :decision
            WHERE approval_id = :approval_id
              AND state_version = :expected_state_version
              AND state NOT IN (
                'auto_approved', 'auto_denied', 'approved',
                'denied', 'expired'
              )
            """,
            {
              **self._request_to_row(updated),
              "expected_state_version": current.state_version,
            },
          )
          if cursor.rowcount != 1:
            conn.rollback()
            raise RuntimeError(
              "approval expiration compare-and-set failed"
            )
          terminal_event = "expired"
        else:
          if vote.decision == "approved":
            if current.approval_constraint == "legacy_unknown":
              raise ValueError(
                "approval constraint is unknown; replan and reauthorize"
              )
            if current.approval_constraint == "fresh_human_owner" and (
              current.authorization_mode not in {
                "HUMAN",
                "OWNER_CONTROL_PLANE",
              }
              or vote.decider_id != current.required_owner_user_id
              or vote.decider_role != "owner"
            ):
              raise ValueError(
                "fresh owner approval constraint requires its exact owner vote"
              )

          existing = conn.execute(
            "SELECT * FROM approval_votes WHERE approval_id = ? AND decider_id = ?",
            (approval_id, vote.decider_id),
          ).fetchone()
          if existing is None:
            conn.execute(
              """
              INSERT INTO approval_votes (
                vote_id, approval_id, decider_id, decider_role, decision,
                decision_reason, decided_at
              ) VALUES (?, ?, ?, ?, ?, ?, ?)
              """,
              (
                vote.vote_id,
                vote.approval_id,
                vote.decider_id,
                vote.decider_role,
                vote.decision,
                vote.decision_reason,
                _dt_to_text(vote.decided_at),
              ),
            )
            emitted_vote = True
          elif (
            str(existing["vote_id"]) != vote.vote_id
            or str(existing["decision"]) != vote.decision
          ):
            raise ValueError("approval already has a different owner decision")

          terminal_state: ApprovalState = vote.decision
          updated = replace(
            current,
            state=terminal_state,
            state_version=current.state_version + 1,
            decided_at=vote.decided_at,
            decider_id=vote.decider_id,
            decider_role=vote.decider_role,
            decision=terminal_state,
            decision_reason=vote.decision_reason,
          )
          terminal_event = terminal_state

          conn.execute(
            """
            UPDATE approval_requests SET
              state = :state,
              state_version = :state_version,
              decided_at = :decided_at,
              decider_id = :decider_id,
              decider_role = :decider_role,
              decision = :decision,
              decision_reason = :decision_reason
            WHERE approval_id = :approval_id
            """,
            self._request_to_row(updated),
          )
          if (
            normalized_delivery is not None
            and updated.state in {"approved", "denied"}
          ):
            _insert_autonomous_approval_delivery_in_transaction(
              conn,
              request=updated,
              delivery=normalized_delivery,
              approved=updated.state == "approved",
              vote=vote,
            )
        conn.commit()
    if (
      normalized_delivery is not None
      and updated.state in {"approved", "denied"}
    ):
      if callable(getattr(
        self._audit_emitter,
        "emit_audit_for_lifecycle_event",
        None,
      )):
        await self.ensure_autonomous_approval_delivery_audited(
          updated.approval_id,
          tool_call_id=normalized_delivery["tool_call_id"],
          nonce=normalized_delivery["nonce"],
        )
      return updated
    if emitted_vote:
      await self._emit("vote_recorded", updated, vote=vote)
    if terminal_event is not None:
      if terminal_event == "expired":
        await self._emit(terminal_event, updated)
      else:
        await self._emit(terminal_event, updated, vote=vote)
    return updated

  async def ensure_autonomous_approval_delivery_audited(
    self,
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
  ) -> dict[str, Any]:
    """Durably audit an outbox decision before it may be published."""
    normalized_approval_id = _canonical_delivery_text(
      approval_id,
      field_name="approval_id",
    )
    normalized_tool_call_id = _canonical_delivery_text(
      tool_call_id,
      field_name="tool_call_id",
    )
    normalized_nonce = _canonical_delivery_text(
      nonce,
      field_name="nonce",
    )
    async with self._lock:
      with self._connection() as conn:
        delivery_row = conn.execute(
          """
          SELECT * FROM autonomous_approval_delivery_outbox
          WHERE approval_id = ? AND tool_call_id = ? AND nonce = ?
          """,
          (
            normalized_approval_id,
            normalized_tool_call_id,
            normalized_nonce,
          ),
        ).fetchone()
        if delivery_row is None:
          raise KeyError(
            "autonomous approval delivery outbox record not found"
          )
        delivery = _autonomous_approval_delivery_projection(
          delivery_row
        )
        if delivery["audit_state"] == "ready":
          return delivery
        if delivery["state"] != "pending":
          raise RuntimeError(
            "unaudited autonomous approval delivery is not pending"
          )
        request_row = conn.execute(
          "SELECT * FROM approval_requests WHERE approval_id = ?",
          (normalized_approval_id,),
        ).fetchone()
        if request_row is None:
          raise RuntimeError(
            "autonomous approval delivery request is missing"
          )
        request = self._row_to_request(request_row)
        expected_state = (
          "approved" if delivery["approved"] else "denied"
        )
        if (
          request.state != expected_state
          or request.decision != expected_state
          or request.approval_id != delivery["approval_id"]
          or request.tool_call_id != delivery["tool_call_id"]
          or request.request_id != delivery["control_run_id"]
          or request.run_id != delivery["control_run_id"]
          or request.session_id != delivery["session_id"]
          or request.decided_at is None
          or _datetime_to_epoch_ns(request.decided_at)
          != delivery["decided_at_ns"]
        ):
          raise RuntimeError(
            "autonomous approval delivery durable identity mismatch"
          )

        vote: ApprovalVote | None = None
        triggering_vote_id = delivery["triggering_vote_id"]
        if triggering_vote_id is not None:
          vote_row = conn.execute(
            "SELECT * FROM approval_votes WHERE vote_id = ?",
            (triggering_vote_id,),
          ).fetchone()
          if vote_row is None:
            raise RuntimeError(
              "autonomous approval delivery triggering vote is missing"
            )
          vote = _approval_vote_from_row(vote_row)
          if (
            vote.approval_id != normalized_approval_id
            or vote.decision != expected_state
            or _datetime_to_epoch_ns(vote.decided_at)
            != delivery["decided_at_ns"]
          ):
            raise RuntimeError(
              "autonomous approval delivery vote identity mismatch"
            )

        expected_vote_entry_id = (
          _autonomous_approval_audit_entry_id(
            event_type="vote_recorded",
            approval_id=normalized_approval_id,
            tool_call_id=normalized_tool_call_id,
            nonce=normalized_nonce,
            source_id=vote.vote_id,
            event_at_ns=_datetime_to_epoch_ns(vote.decided_at),
          )
          if vote is not None
          else None
        )
        expected_terminal_entry_id = (
          _autonomous_approval_audit_entry_id(
            event_type=expected_state,
            approval_id=normalized_approval_id,
            tool_call_id=normalized_tool_call_id,
            nonce=normalized_nonce,
            source_id=(
              vote.vote_id if vote is not None else "cancellation"
            ),
            event_at_ns=delivery["decided_at_ns"],
          )
        )
        if (
          delivery["vote_audit_entry_id"]
          != expected_vote_entry_id
          or delivery["terminal_audit_entry_id"]
          != expected_terminal_entry_id
        ):
          raise RuntimeError(
            "autonomous approval delivery audit receipt mismatch"
          )

    emit = getattr(
      self._audit_emitter,
      "emit_audit_for_lifecycle_event",
      None,
    )
    if not callable(emit):
      raise RuntimeError(
        "autonomous approval delivery durable audit is unavailable"
      )
    if vote is not None:
      await emit(
        event_type="vote_recorded",
        request=request,
        raw_tool_args={},
        vote=vote,
        pending_tools_nonce=normalized_nonce,
        skill=request.skill,
        entry_id=expected_vote_entry_id,
        event_ts=vote.decided_at,
      )
    await emit(
      event_type=expected_state,
      request=request,
      raw_tool_args={},
      vote=vote,
      pending_tools_nonce=normalized_nonce,
      skill=request.skill,
      entry_id=expected_terminal_entry_id,
      event_ts=request.decided_at,
    )

    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _ready_at_ns, ready_at_value = _trusted_utc_now(conn)
        ready_at = _dt_to_text(ready_at_value)
        current_row = conn.execute(
          """
          SELECT * FROM autonomous_approval_delivery_outbox
          WHERE approval_id = ? AND tool_call_id = ? AND nonce = ?
          """,
          (
            normalized_approval_id,
            normalized_tool_call_id,
            normalized_nonce,
          ),
        ).fetchone()
        if current_row is None:
          conn.rollback()
          raise RuntimeError(
            "autonomous approval delivery outbox disappeared"
          )
        current = _autonomous_approval_delivery_projection(
          current_row
        )
        if current["audit_state"] == "ready":
          conn.commit()
          return current
        cursor = conn.execute(
          """
          UPDATE autonomous_approval_delivery_outbox SET
            audit_state = 'ready',
            audit_ready_at = ?,
            updated_at = ?
          WHERE approval_id = ? AND tool_call_id = ? AND nonce = ?
            AND state = 'pending'
            AND audit_state = 'pending'
            AND terminal_audit_entry_id = ?
            AND (
              (vote_audit_entry_id IS NULL AND ? IS NULL)
              OR vote_audit_entry_id = ?
            )
          """,
          (
            ready_at,
            ready_at,
            normalized_approval_id,
            normalized_tool_call_id,
            normalized_nonce,
            expected_terminal_entry_id,
            expected_vote_entry_id,
            expected_vote_entry_id,
          ),
        )
        if cursor.rowcount != 1:
          conn.rollback()
          raise RuntimeError(
            "autonomous approval delivery audit receipt compare-and-set failed"
          )
        stored = conn.execute(
          """
          SELECT * FROM autonomous_approval_delivery_outbox
          WHERE approval_id = ? AND tool_call_id = ? AND nonce = ?
          """,
          (
            normalized_approval_id,
            normalized_tool_call_id,
            normalized_nonce,
          ),
        ).fetchone()
        conn.commit()
    if stored is None:
      raise RuntimeError(
        "autonomous approval delivery outbox disappeared"
      )
    return _autonomous_approval_delivery_projection(stored)

  async def get_autonomous_approval_delivery(
    self,
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
  ) -> dict[str, Any] | None:
    normalized_approval_id = _canonical_delivery_text(
      approval_id,
      field_name="approval_id",
    )
    normalized_tool_call_id = _canonical_delivery_text(
      tool_call_id,
      field_name="tool_call_id",
    )
    normalized_nonce = _canonical_delivery_text(
      nonce,
      field_name="nonce",
    )
    async with self._lock:
      with self._connection() as conn:
        row = conn.execute(
          """
          SELECT * FROM autonomous_approval_delivery_outbox
          WHERE approval_id = ? AND tool_call_id = ? AND nonce = ?
          """,
          (
            normalized_approval_id,
            normalized_tool_call_id,
            normalized_nonce,
          ),
        ).fetchone()
    return (
      _autonomous_approval_delivery_projection(row)
      if row is not None
      else None
    )

  @contextmanager
  def autonomous_approval_delivery_append_transaction(
    self,
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
    approved: bool,
  ) -> Iterator[None]:
    """Serialize one bounded child-inbox append against cancellation."""
    normalized_approval_id = _canonical_delivery_text(
      approval_id,
      field_name="approval_id",
    )
    normalized_tool_call_id = _canonical_delivery_text(
      tool_call_id,
      field_name="tool_call_id",
    )
    normalized_nonce = _canonical_delivery_text(
      nonce,
      field_name="nonce",
    )
    if type(approved) is not bool:
      raise ValueError(
        "autonomous approval delivery approved is invalid"
      )
    with self._connection() as conn:
      conn.execute("BEGIN IMMEDIATE")
      row = conn.execute(
        """
        SELECT * FROM autonomous_approval_delivery_outbox
        WHERE approval_id = ? AND tool_call_id = ? AND nonce = ?
        """,
        (
          normalized_approval_id,
          normalized_tool_call_id,
          normalized_nonce,
        ),
      ).fetchone()
      if row is None:
        conn.rollback()
        raise KeyError(
          "autonomous approval delivery outbox record not found"
        )
      delivery = _autonomous_approval_delivery_projection(row)
      if delivery["approved"] is not approved:
        conn.rollback()
        raise ValueError(
          "autonomous approval delivery decision changed"
        )
      if delivery["audit_state"] != "ready":
        conn.rollback()
        raise RuntimeError(
          "autonomous approval delivery audit receipt is not ready"
        )
      if delivery["state"] != "pending":
        conn.rollback()
        raise RuntimeError(
          "autonomous approval delivery is not pending"
        )
      if approved:
        cancellation_fence = conn.execute(
          """
          SELECT 1 FROM persistent_grant_cancellation_fences
          WHERE approval_id = ?
          """,
          (normalized_approval_id,),
        ).fetchone()
        if cancellation_fence is not None:
          conn.rollback()
          raise PersistentGrantCancellationFenced(
            "approved autonomous delivery is fenced for cancellation"
          )
      try:
        yield
      except BaseException:
        conn.rollback()
        raise
      else:
        _published_at_ns, published_at_value = _trusted_utc_now(
          conn
        )
        published_at = _dt_to_text(published_at_value)
        cursor = conn.execute(
          """
          UPDATE autonomous_approval_delivery_outbox SET
            state = 'published',
            attempt_count = attempt_count + 1,
            last_error = NULL,
            updated_at = ?,
            published_at = ?
          WHERE approval_id = ? AND tool_call_id = ? AND nonce = ?
            AND state = 'pending'
            AND audit_state = 'ready'
          """,
          (
            published_at,
            published_at,
            normalized_approval_id,
            normalized_tool_call_id,
            normalized_nonce,
          ),
        )
        if cursor.rowcount != 1:
          conn.rollback()
          raise RuntimeError(
            "autonomous approval publication compare-and-set failed"
          )
        conn.commit()

  def reconcile_autonomous_approval_delivery_duplicate(
    self,
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
    approved: bool,
  ) -> dict[str, Any]:
    """Reconcile an exact fsynced child record after a parent crash."""
    normalized_approval_id = _canonical_delivery_text(
      approval_id,
      field_name="approval_id",
    )
    normalized_tool_call_id = _canonical_delivery_text(
      tool_call_id,
      field_name="tool_call_id",
    )
    normalized_nonce = _canonical_delivery_text(
      nonce,
      field_name="nonce",
    )
    if type(approved) is not bool:
      raise ValueError(
        "autonomous approval delivery approved is invalid"
      )
    with self._connection() as conn:
      conn.execute("BEGIN IMMEDIATE")
      row = conn.execute(
        """
        SELECT * FROM autonomous_approval_delivery_outbox
        WHERE approval_id = ? AND tool_call_id = ? AND nonce = ?
        """,
        (
          normalized_approval_id,
          normalized_tool_call_id,
          normalized_nonce,
        ),
      ).fetchone()
      if row is None:
        conn.rollback()
        raise KeyError(
          "autonomous approval delivery outbox record not found"
        )
      delivery = _autonomous_approval_delivery_projection(row)
      if delivery["approved"] is not approved:
        conn.rollback()
        raise ValueError(
          "autonomous approval delivery decision changed"
        )
      if delivery["audit_state"] != "ready":
        conn.rollback()
        raise RuntimeError(
          "autonomous approval delivery audit receipt is not ready"
        )
      if delivery["state"] in {"published", "acknowledged"}:
        conn.commit()
        return delivery
      if delivery["state"] != "pending":
        conn.rollback()
        raise RuntimeError(
          "autonomous approval delivery cannot be reconciled"
        )
      if approved:
        cancellation_fence = conn.execute(
          """
          SELECT 1 FROM persistent_grant_cancellation_fences
          WHERE approval_id = ?
          """,
          (normalized_approval_id,),
        ).fetchone()
        if cancellation_fence is not None:
          conn.rollback()
          raise PersistentGrantCancellationFenced(
            "approved autonomous delivery is fenced for cancellation"
          )
      _published_at_ns, published_at_value = _trusted_utc_now(
        conn
      )
      published_at = _dt_to_text(published_at_value)
      cursor = conn.execute(
        """
        UPDATE autonomous_approval_delivery_outbox SET
          state = 'published',
          attempt_count = attempt_count + 1,
          last_error = NULL,
          updated_at = ?,
          published_at = ?
        WHERE approval_id = ? AND tool_call_id = ? AND nonce = ?
          AND state = 'pending'
          AND audit_state = 'ready'
        """,
        (
          published_at,
          published_at,
          normalized_approval_id,
          normalized_tool_call_id,
          normalized_nonce,
        ),
      )
      if cursor.rowcount != 1:
        conn.rollback()
        raise RuntimeError(
          "autonomous approval duplicate reconciliation failed"
        )
      stored = conn.execute(
        """
        SELECT * FROM autonomous_approval_delivery_outbox
        WHERE approval_id = ? AND tool_call_id = ? AND nonce = ?
        """,
        (
          normalized_approval_id,
          normalized_tool_call_id,
          normalized_nonce,
        ),
      ).fetchone()
      conn.commit()
    if stored is None:
      raise RuntimeError(
        "autonomous approval delivery outbox disappeared"
      )
    return _autonomous_approval_delivery_projection(stored)

  @contextmanager
  def autonomous_approval_delivery_duplicate_transaction(
    self,
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
    approved: bool,
  ) -> Iterator[None]:
    self.reconcile_autonomous_approval_delivery_duplicate(
      approval_id,
      tool_call_id=tool_call_id,
      nonce=nonce,
      approved=approved,
    )
    yield

  async def acknowledge_autonomous_approval_delivery(
    self,
    approval_id: str,
    *,
    task_id: str,
    control_run_id: str,
    session_id: str,
    channel_id: str,
    tool_call_id: str,
    nonce: str,
    approved: bool,
    decided_at_ns: int,
  ) -> dict[str, Any]:
    normalized_approval_id = _canonical_delivery_text(
      approval_id,
      field_name="approval_id",
    )
    authority = _normalize_autonomous_approval_delivery_context(
      {
        "task_id": task_id,
        "control_run_id": control_run_id,
        "session_id": session_id,
        "channel_id": channel_id,
        "tool_call_id": tool_call_id,
        "nonce": nonce,
      }
    )
    if type(approved) is not bool:
      raise ValueError(
        "autonomous approval acknowledgment decision is invalid"
      )
    if (
      type(decided_at_ns) is not int
      or decided_at_ns < 1
    ):
      raise ValueError(
        "autonomous approval acknowledgment decided_at_ns is invalid"
      )
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _acknowledged_at_ns, acknowledged_at = (
          _trusted_utc_now(conn)
        )
        now_text = _dt_to_text(acknowledged_at)
        row = conn.execute(
          """
          SELECT * FROM autonomous_approval_delivery_outbox
          WHERE approval_id = ? AND tool_call_id = ? AND nonce = ?
          """,
          (
            normalized_approval_id,
            authority["tool_call_id"],
            authority["nonce"],
          ),
        ).fetchone()
        if row is None:
          raise KeyError(
            "autonomous approval delivery outbox record not found"
          )
        current = _autonomous_approval_delivery_projection(row)
        expected_authority: dict[str, Any] = {
          "approval_id": normalized_approval_id,
          **authority,
          "approved": approved,
          "decided_at_ns": decided_at_ns,
        }
        if any(
          current[field_name] != expected_value
          for field_name, expected_value
          in expected_authority.items()
        ):
          conn.rollback()
          raise ValueError(
            "autonomous approval acknowledgment authority mismatch"
          )
        if current["audit_state"] != "ready":
          conn.rollback()
          raise RuntimeError(
            "autonomous approval acknowledgment audit is not ready"
          )
        if current["state"] == "acknowledged":
          conn.commit()
          return current
        if current["state"] != "published":
          conn.rollback()
          raise RuntimeError(
            "autonomous approval delivery was not published"
          )
        cursor = conn.execute(
          """
          UPDATE autonomous_approval_delivery_outbox SET
            state = 'acknowledged',
            last_error = NULL,
            updated_at = ?,
            acknowledged_at = ?
          WHERE approval_id = ?
            AND task_id = ?
            AND control_run_id = ?
            AND session_id = ?
            AND channel_id = ?
            AND tool_call_id = ?
            AND nonce = ?
            AND approved = ?
            AND allow_tool_type = 0
            AND decided_at_ns = ?
            AND state = 'published'
            AND audit_state = 'ready'
          """,
          (
            now_text,
            now_text,
            normalized_approval_id,
            authority["task_id"],
            authority["control_run_id"],
            authority["session_id"],
            authority["channel_id"],
            authority["tool_call_id"],
            authority["nonce"],
            1 if approved else 0,
            decided_at_ns,
          ),
        )
        if cursor.rowcount != 1:
          conn.rollback()
          raise RuntimeError(
            "autonomous approval acknowledgment compare-and-set failed"
          )
        stored = conn.execute(
          """
          SELECT * FROM autonomous_approval_delivery_outbox
          WHERE approval_id = ? AND tool_call_id = ? AND nonce = ?
          """,
          (
            normalized_approval_id,
            authority["tool_call_id"],
            authority["nonce"],
          ),
        ).fetchone()
        conn.commit()
    if stored is None:
      raise RuntimeError(
        "autonomous approval delivery outbox disappeared"
      )
    projection = _autonomous_approval_delivery_projection(stored)
    if projection["state"] != "acknowledged":
      raise RuntimeError(
        "autonomous approval delivery was not acknowledged"
      )
    return projection

  async def list_pending_autonomous_approval_deliveries(
    self,
    *,
    limit: int = 64,
    after_sequence: int = 0,
    through_sequence: int | None = None,
  ) -> list[dict[str, Any]]:
    if type(limit) is not int or not 1 <= limit <= 256:
      raise ValueError(
        "autonomous approval delivery limit must be 1..256"
      )
    if type(after_sequence) is not int or after_sequence < 0:
      raise ValueError(
        "autonomous approval delivery after_sequence is invalid"
      )
    if (
      through_sequence is not None
      and (
        type(through_sequence) is not int
        or through_sequence < after_sequence
      )
    ):
      raise ValueError(
        "autonomous approval delivery through_sequence is invalid"
      )
    async with self._lock:
      with self._connection() as conn:
        rows = conn.execute(
          """
          SELECT * FROM autonomous_approval_delivery_outbox
          WHERE state = 'pending'
            AND delivery_sequence > ?
            AND (? IS NULL OR delivery_sequence <= ?)
          ORDER BY delivery_sequence ASC
          LIMIT ?
          """,
          (
            after_sequence,
            through_sequence,
            through_sequence,
            limit,
          ),
        ).fetchall()
    return [
      _autonomous_approval_delivery_projection(row)
      for row in rows
    ]

  async def autonomous_approval_delivery_high_water(self) -> int:
    window = await self.autonomous_approval_delivery_recovery_window()
    return int(window["high_water"])

  async def autonomous_approval_delivery_recovery_window(
    self,
  ) -> dict[str, int]:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        observed_at_ns, _now = _trusted_utc_now(conn)
        row = conn.execute(
          """
          SELECT COALESCE(MAX(delivery_sequence), 0) AS high_water
          FROM autonomous_approval_delivery_outbox
          """
        ).fetchone()
        conn.commit()
    if row is None:
      raise RuntimeError(
        "autonomous approval delivery high-water query failed"
      )
    high_water = int(row["high_water"])
    if high_water < 0:
      raise RuntimeError(
        "autonomous approval delivery high-water is invalid"
      )
    return {
      "high_water": high_water,
      "observed_at_ns": observed_at_ns,
    }

  async def record_autonomous_approval_delivery_failure(
    self,
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
    error: str,
  ) -> dict[str, Any]:
    normalized_approval_id = _canonical_delivery_text(
      approval_id,
      field_name="approval_id",
    )
    normalized_tool_call_id = _canonical_delivery_text(
      tool_call_id,
      field_name="tool_call_id",
    )
    normalized_nonce = _canonical_delivery_text(
      nonce,
      field_name="nonce",
    )
    normalized_error = _canonical_delivery_text(
      error,
      field_name="last_error",
      max_length=512,
    )
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
          """
          SELECT * FROM autonomous_approval_delivery_outbox
          WHERE approval_id = ? AND tool_call_id = ? AND nonce = ?
          """,
          (
            normalized_approval_id,
            normalized_tool_call_id,
            normalized_nonce,
          ),
        ).fetchone()
        if row is None:
          conn.rollback()
          raise KeyError(
            "autonomous approval delivery outbox record not found"
          )
        current = _autonomous_approval_delivery_projection(row)
        if current["state"] in {"acknowledged", "quarantined"}:
          conn.commit()
          return current
        now_ns, now = _trusted_utc_now(conn)
        now_text = _dt_to_text(now)
        next_attempt_count = int(current["attempt_count"]) + 1
        quarantined = (
          current["state"] == "pending"
          and (
            next_attempt_count
            >= AUTONOMOUS_APPROVAL_DELIVERY_MAX_ATTEMPTS
            or now_ns >= int(current["retry_deadline_ns"])
          )
        )
        next_state = (
          "quarantined" if quarantined else current["state"]
        )
        retry_delay_ns = min(
          AUTONOMOUS_APPROVAL_DELIVERY_RETRY_BASE_NS
          * (1 << min(max(next_attempt_count - 1, 0), 30)),
          AUTONOMOUS_APPROVAL_DELIVERY_RETRY_MAX_NS,
        )
        if now_ns > _SQLITE_MAX_INTEGER - retry_delay_ns:
          conn.rollback()
          raise RuntimeError(
            "autonomous approval delivery retry time overflow"
          )
        next_attempt_ns = (
          now_ns
          if quarantined or current["state"] == "published"
          else now_ns + retry_delay_ns
        )
        cursor = conn.execute(
          """
          UPDATE autonomous_approval_delivery_outbox SET
            state = ?,
            attempt_count = ?,
            last_error = ?,
            updated_at = ?,
            next_attempt_ns = ?,
            last_attempt_ns = ?,
            quarantined_at = ?
          WHERE approval_id = ? AND tool_call_id = ? AND nonce = ?
            AND state = ?
            AND attempt_count = ?
          """,
          (
            next_state,
            next_attempt_count,
            normalized_error,
            now_text,
            next_attempt_ns,
            now_ns,
            now_text if quarantined else None,
            normalized_approval_id,
            normalized_tool_call_id,
            normalized_nonce,
            current["state"],
            current["attempt_count"],
          ),
        )
        stored = conn.execute(
          """
          SELECT * FROM autonomous_approval_delivery_outbox
          WHERE approval_id = ? AND tool_call_id = ? AND nonce = ?
          """,
          (
            normalized_approval_id,
            normalized_tool_call_id,
            normalized_nonce,
          ),
        ).fetchone()
        if stored is None:
          conn.rollback()
          raise KeyError(
            "autonomous approval delivery outbox record not found"
          )
        if cursor.rowcount != 1:
          conn.rollback()
          raise RuntimeError(
            "autonomous approval delivery failure compare-and-set failed"
          )
        conn.commit()
    if stored is None:
      raise RuntimeError(
        "autonomous approval delivery outbox disappeared"
      )
    return _autonomous_approval_delivery_projection(stored)

  async def quarantine_autonomous_approval_delivery(
    self,
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
    error: str,
  ) -> dict[str, Any]:
    normalized_approval_id = _canonical_delivery_text(
      approval_id,
      field_name="approval_id",
    )
    normalized_tool_call_id = _canonical_delivery_text(
      tool_call_id,
      field_name="tool_call_id",
    )
    normalized_nonce = _canonical_delivery_text(
      nonce,
      field_name="nonce",
    )
    normalized_error = _canonical_delivery_text(
      error,
      field_name="last_error",
      max_length=512,
    )
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
          """
          SELECT * FROM autonomous_approval_delivery_outbox
          WHERE approval_id = ? AND tool_call_id = ? AND nonce = ?
          """,
          (
            normalized_approval_id,
            normalized_tool_call_id,
            normalized_nonce,
          ),
        ).fetchone()
        if row is None:
          conn.rollback()
          raise KeyError(
            "autonomous approval delivery outbox record not found"
          )
        current = _autonomous_approval_delivery_projection(row)
        if current["state"] == "quarantined":
          conn.commit()
          return current
        if current["state"] != "pending":
          conn.rollback()
          raise RuntimeError(
            "only pending autonomous approval delivery can be quarantined"
          )
        _now_ns, now = _trusted_utc_now(conn)
        now_text = _dt_to_text(now)
        cursor = conn.execute(
          """
          UPDATE autonomous_approval_delivery_outbox SET
            state = 'quarantined',
            attempt_count = attempt_count + 1,
            last_error = ?,
            updated_at = ?,
            next_attempt_ns = ?,
            last_attempt_ns = ?,
            quarantined_at = ?
          WHERE approval_id = ? AND tool_call_id = ? AND nonce = ?
            AND state = 'pending'
            AND attempt_count = ?
          """,
          (
            normalized_error,
            now_text,
            _now_ns,
            _now_ns,
            now_text,
            normalized_approval_id,
            normalized_tool_call_id,
            normalized_nonce,
            current["attempt_count"],
          ),
        )
        if cursor.rowcount != 1:
          conn.rollback()
          raise RuntimeError(
            "autonomous approval quarantine compare-and-set failed"
          )
        stored = conn.execute(
          """
          SELECT * FROM autonomous_approval_delivery_outbox
          WHERE approval_id = ? AND tool_call_id = ? AND nonce = ?
          """,
          (
            normalized_approval_id,
            normalized_tool_call_id,
            normalized_nonce,
          ),
        ).fetchone()
        conn.commit()
    if stored is None:
      raise RuntimeError(
        "autonomous approval delivery outbox disappeared"
      )
    return _autonomous_approval_delivery_projection(stored)

  async def create_persistent_grant(self, grant: PersistentGrant) -> PersistentGrant:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cancellation_fence = conn.execute(
          """
          SELECT 1 FROM persistent_grant_cancellation_fences
          WHERE approval_id = ?
          """,
          (grant.granted_via_approval_id,),
        ).fetchone()
        if cancellation_fence is not None:
          conn.rollback()
          raise PersistentGrantCancellationFenced(
            "persistent grant approval is fenced for cancellation"
          )
        source_row = conn.execute(
          "SELECT * FROM approval_requests WHERE approval_id = ?",
          (grant.granted_via_approval_id,),
        ).fetchone()
        if source_row is None:
          raise ValueError("approval constraint source for persistent grant is missing")
        source = self._row_to_request(source_row)
        if (
          source.approval_constraint != "standard"
          or source.state != "approved"
          or source.decision != "approved"
          or source.user_id != grant.user_id
          or source.tool_name != grant.tool_name
          or source.persistent_grant_scope != grant.scope_hint
        ):
          raise ValueError(
            "approval constraint does not permit persistent grant minting"
          )
        conn.execute(
          """
          INSERT INTO persistent_grants (
            grant_id, user_id, tool_name, scope_hint, args_predicate,
            granted_at, expires_at, revoked_at, granted_via_approval_id, policy_id
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          """,
          (
            grant.grant_id,
            grant.user_id,
            grant.tool_name,
            grant.scope_hint,
            _json_dumps(grant.args_predicate) if grant.args_predicate is not None else None,
            _dt_to_text(grant.granted_at),
            _dt_to_text(grant.expires_at),
            _dt_to_text(grant.revoked_at),
            grant.granted_via_approval_id,
            grant.policy_id,
          ),
        )
        conn.commit()
    await self._emit_grant(
      "persistent_grant_created",
      grant,
      request=await self.get(grant.granted_via_approval_id),
    )
    return grant

  async def find_persistent_grant(
    self,
    *,
    user_id: str,
    tool_name: str,
    scope_hint: str,
    now: datetime | None = None,
    approval_constraint: str = "standard",
  ) -> PersistentGrant | None:
    if approval_constraint != "standard":
      return None
    now_text = _dt_to_text(now or utc_now())
    async with self._lock:
      with self._connection() as conn:
        row = conn.execute(
          """
          SELECT grants.* FROM persistent_grants AS grants
          WHERE grants.user_id = ?
            AND grants.tool_name = ?
            AND grants.scope_hint = ?
            AND grants.revoked_at IS NULL
            AND (grants.expires_at IS NULL OR grants.expires_at > ?)
            AND NOT EXISTS (
              SELECT 1 FROM persistent_grant_cancellation_fences AS fences
              WHERE fences.approval_id = grants.granted_via_approval_id
            )
          ORDER BY grants.granted_at DESC
          LIMIT 1
          """,
          (user_id, tool_name, scope_hint, now_text),
        ).fetchone()
    return self._row_to_grant(row) if row is not None else None

  async def revoke_persistent_grant(self, grant_id: str, *, revoked_at: datetime | None = None) -> None:
    when = revoked_at or utc_now()
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
          "SELECT * FROM persistent_grants WHERE grant_id = ?",
          (grant_id,),
        ).fetchone()
        conn.execute(
          "UPDATE persistent_grants SET revoked_at = ? WHERE grant_id = ?",
          (_dt_to_text(when), grant_id),
        )
        conn.commit()
    if row is not None:
      grant = self._row_to_grant(row)
      await self._emit_grant(
        "persistent_grant_revoked",
        grant,
        request=await self.get(grant.granted_via_approval_id),
      )

  async def revoke_persistent_grants_for_approval(
    self,
    approval_id: str,
    *,
    revoked_at: datetime | None = None,
  ) -> int:
    """Atomically revoke every active grant produced by one approval."""
    when = revoked_at or utc_now()
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
          """
          SELECT * FROM persistent_grants
          WHERE granted_via_approval_id = ? AND revoked_at IS NULL
          """,
          (approval_id,),
        ).fetchall()
        conn.execute(
          """
          UPDATE persistent_grants SET revoked_at = ?
          WHERE granted_via_approval_id = ? AND revoked_at IS NULL
          """,
          (_dt_to_text(when), approval_id),
        )
        conn.commit()
    request = await self.get(approval_id)
    for row in rows:
      grant = replace(self._row_to_grant(row), revoked_at=when)
      await self._emit_grant(
        "persistent_grant_revoked",
        grant,
        request=request,
      )
    return len(rows)

  async def fence_persistent_grants_for_cancellation(
    self,
    approval_id: str,
    *,
    expected_tool_call_id: str,
    expected_user_id: str,
    expected_request_id: str,
    expected_run_id: str,
    expected_session_id: str,
    expected_channel: str | None,
  ) -> tuple[ApprovalRequest, bool]:
    """Hide and revoke grants at the durable cancellation boundary."""
    now = utc_now()
    revoked_grant_rows: list[sqlite3.Row] = []
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current_row = conn.execute(
          "SELECT * FROM approval_requests WHERE approval_id = ?",
          (approval_id,),
        ).fetchone()
        if current_row is None:
          conn.rollback()
          raise KeyError(f"approval request not found: {approval_id}")
        current = self._row_to_request(current_row)
        normalized_expected_channel = str(expected_channel or "").strip().lower()
        identity_matches = all((
          current.tool_call_id == expected_tool_call_id,
          current.user_id == expected_user_id,
          current.request_id == expected_request_id,
          current.run_id == expected_run_id,
          current.session_id == expected_session_id,
          str(current.channel or "").strip().lower() == normalized_expected_channel,
        ))
        if not identity_matches:
          conn.commit()
          return current, False
        conn.execute(
          """
          INSERT OR IGNORE INTO persistent_grant_cancellation_fences (
            approval_id, fenced_at
          ) VALUES (?, ?)
          """,
          (approval_id, _dt_to_text(now)),
        )
        revoked_grant_rows = conn.execute(
          """
          SELECT * FROM persistent_grants
          WHERE granted_via_approval_id = ? AND revoked_at IS NULL
          """,
          (approval_id,),
        ).fetchall()
        conn.execute(
          """
          UPDATE persistent_grants SET revoked_at = ?
          WHERE granted_via_approval_id = ? AND revoked_at IS NULL
          """,
          (_dt_to_text(now), approval_id),
        )
        conn.commit()
    for row in revoked_grant_rows:
      grant = replace(self._row_to_grant(row), revoked_at=now)
      try:
        await self._emit_grant(
          "persistent_grant_revoked",
          grant,
          request=current,
        )
      except asyncio.CancelledError:
        await self._emit_grant(
          "persistent_grant_revoked",
          grant,
          request=current,
        )
    return current, True

  async def create_delegation_grant(self, grant: DelegationGrant) -> DelegationGrant:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
          """
          INSERT INTO delegation_grants (
            delegation_id, delegator_user_id, delegator_run_id, delegator_session_id,
            delegator_profile, delegator_channel, bound_excel_session_id,
            bound_relay_request_id, bound_workbook, tool_class_ceiling,
            args_predicate, window_seconds, exclude_external_write_bypass,
            created_at, expires_at, revoked_at, consumed_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          """,
          (
            grant.delegation_id,
            grant.delegator_user_id,
            grant.delegator_run_id,
            grant.delegator_session_id,
            grant.delegator_profile,
            grant.delegator_channel,
            grant.bound_excel_session_id,
            grant.bound_relay_request_id,
            grant.bound_workbook,
            _json_dumps(sorted(grant.tool_class_ceiling)),
            _json_dumps(grant.args_predicate) if grant.args_predicate is not None else None,
            grant.window_seconds,
            1 if grant.exclude_external_write_bypass else 0,
            _dt_to_text(grant.created_at),
            _dt_to_text(grant.expires_at),
            _dt_to_text(grant.revoked_at),
            _dt_to_text(grant.consumed_at),
          ),
        )
        conn.commit()
    return grant

  async def get_delegation_grant(self, delegation_id: str) -> DelegationGrant | None:
    with self._connection() as conn:
      row = conn.execute(
        "SELECT * FROM delegation_grants WHERE delegation_id = ?",
        (delegation_id,),
      ).fetchone()
    return self._row_to_delegation_grant(row) if row is not None else None

  async def claim_delegation_grant(
    self,
    *,
    delegation_id: str,
    bound_relay_request_id: str,
    bound_excel_session_id: str,
    now: datetime | None = None,
  ) -> DelegationGrant | None:
    now_value = now or utc_now()
    now_text = _dt_to_text(now_value)
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
          """
          UPDATE delegation_grants
          SET consumed_at = :now
          WHERE delegation_id = :delegation_id
            AND bound_relay_request_id = :rid
            AND bound_excel_session_id = :sid
            AND consumed_at IS NULL
            AND revoked_at IS NULL
            AND (expires_at IS NULL OR expires_at > :now)
          """,
          {
            "now": now_text,
            "delegation_id": delegation_id,
            "rid": bound_relay_request_id,
            "sid": bound_excel_session_id,
          },
        )
        if cursor.rowcount == 1:
          row = conn.execute(
            "SELECT * FROM delegation_grants WHERE delegation_id = ?",
            (delegation_id,),
          ).fetchone()
          conn.commit()
          return self._row_to_delegation_grant(row) if row is not None else None
        conn.commit()
    return None

  async def revoke_delegation_grant(self, delegation_id: str, *, revoked_at: datetime | None = None) -> None:
    when = revoked_at or utc_now()
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
          "UPDATE delegation_grants SET revoked_at = ? WHERE delegation_id = ?",
          (_dt_to_text(when), delegation_id),
        )
        conn.commit()

  def _expire_pending_requests_in_transaction(
    self,
    conn: sqlite3.Connection,
    *,
    now: datetime,
  ) -> list[ApprovalRequest]:
    rows = conn.execute(
      """
      SELECT * FROM approval_requests
      WHERE state = 'pending_user'
        AND expires_at IS NOT NULL
        AND expires_at <= ?
      """,
      (_dt_to_text(now),),
    ).fetchall()
    expired: list[ApprovalRequest] = []
    for row in rows:
      current = self._row_to_request(row)
      updated = replace(
        current,
        state="expired",
        state_version=current.state_version + 1,
        decided_at=now,
        decision="expired",
      )
      cursor = conn.execute(
        """
        UPDATE approval_requests SET
          state = :state,
          state_version = :state_version,
          decided_at = :decided_at,
          decision = :decision
        WHERE approval_id = :approval_id
          AND state = 'pending_user'
          AND state_version = :expected_state_version
        """,
        {
          **self._request_to_row(updated),
          "expected_state_version": current.state_version,
        },
      )
      if cursor.rowcount == 1:
        expired.append(updated)
    return expired

  @staticmethod
  def _prepared_cursor_for_record(
    record: _prepared_bm.PreparedBusinessModelChange,
  ) -> PreparedReconciliationCursor:
    return PreparedReconciliationCursor(
      caller_kind=record.caller_kind,
      user_scope=record.user_scope,
      idempotency_locator=record.idempotency_locator,
    )

  @staticmethod
  def _pending_prepared_page(
    conn: sqlite3.Connection,
    *,
    cursor: PreparedReconciliationCursor | None,
    page_size: int,
  ) -> list[sqlite3.Row]:
    if cursor is None:
      return conn.execute(
        """
        SELECT * FROM prepared_business_model_change
        WHERE lifecycle = 'PENDING'
        ORDER BY caller_kind, user_scope, idempotency_locator
        LIMIT ?
        """,
        (page_size,),
      ).fetchall()
    return conn.execute(
      """
      SELECT * FROM prepared_business_model_change
      WHERE lifecycle = 'PENDING'
        AND (
          caller_kind > ?
          OR (caller_kind = ? AND user_scope > ?)
          OR (
            caller_kind = ? AND user_scope = ?
            AND idempotency_locator > ?
          )
        )
      ORDER BY caller_kind, user_scope, idempotency_locator
      LIMIT ?
      """,
      (
        cursor.caller_kind,
        cursor.caller_kind,
        cursor.user_scope,
        cursor.caller_kind,
        cursor.user_scope,
        cursor.idempotency_locator,
        page_size,
      ),
    ).fetchall()

  @staticmethod
  def _reconcile_prepared_record_in_transaction(
    conn: sqlite3.Connection,
    *,
    record: _prepared_bm.PreparedBusinessModelChange,
    now: datetime,
  ) -> TargetedPreparedReconciliationResult:
    if record.lifecycle is not _prepared_bm.PreparedBusinessModelLifecycle.PENDING:
      return TargetedPreparedReconciliationResult(record=record)

    if record.expires_at is not None:
      expires_at = _prepared_bm._parse_timestamp(record.expires_at, "expires_at")
      if expires_at <= now.astimezone(UTC):
        try:
          expired = _prepared_bm.transition(
            conn,
            caller_kind=record.caller_kind,
            user_scope=record.user_scope,
            idempotency_locator=record.idempotency_locator,
            expected=_prepared_bm.PreparedBusinessModelLifecycle.PENDING,
            target=_prepared_bm.PreparedBusinessModelLifecycle.EXPIRED,
            approval_id=record.approval_id,
            approval_chain_id=record.approval_chain_id,
          )
        except _prepared_bm.PreparedBusinessModelError:
          current = _prepared_bm.get(
            conn,
            caller_kind=record.caller_kind,
            user_scope=record.user_scope,
            idempotency_locator=record.idempotency_locator,
          )
          return TargetedPreparedReconciliationResult(
            record=current,
            conflict=PreparedReconciliationConflict.CAS_CONFLICT,
          )
        return TargetedPreparedReconciliationResult(
          record=expired,
          transitioned=True,
        )

    approval_row = conn.execute(
      "SELECT approval_id, approval_chain_id, state FROM approval_requests WHERE approval_id = ?",
      (record.approval_id,),
    ).fetchone()
    if approval_row is None:
      return TargetedPreparedReconciliationResult(
        record=record,
        conflict=PreparedReconciliationConflict.MISSING_APPROVAL,
      )
    if str(approval_row["approval_chain_id"]) != record.approval_chain_id:
      return TargetedPreparedReconciliationResult(
        record=record,
        conflict=PreparedReconciliationConflict.LINEAGE_CONFLICT,
      )

    approval_state = str(approval_row["state"])
    target = {
      "approved": _prepared_bm.PreparedBusinessModelLifecycle.AUTHORIZED,
      "auto_approved": _prepared_bm.PreparedBusinessModelLifecycle.AUTHORIZED,
      "denied": _prepared_bm.PreparedBusinessModelLifecycle.DENIED,
      "auto_denied": _prepared_bm.PreparedBusinessModelLifecycle.DENIED,
      "cancelled": _prepared_bm.PreparedBusinessModelLifecycle.DENIED,
      "expired": _prepared_bm.PreparedBusinessModelLifecycle.EXPIRED,
    }.get(approval_state)
    if target is None:
      if approval_state in {"created", "pending_user"}:
        return TargetedPreparedReconciliationResult(record=record)
      return TargetedPreparedReconciliationResult(
        record=record,
        conflict=PreparedReconciliationConflict.UNKNOWN_APPROVAL_STATE,
      )
    try:
      reconciled = _prepared_bm.transition(
        conn,
        caller_kind=record.caller_kind,
        user_scope=record.user_scope,
        idempotency_locator=record.idempotency_locator,
        expected=_prepared_bm.PreparedBusinessModelLifecycle.PENDING,
        target=target,
        approval_id=record.approval_id,
        approval_chain_id=record.approval_chain_id,
      )
    except _prepared_bm.PreparedBusinessModelError:
      current = _prepared_bm.get(
        conn,
        caller_kind=record.caller_kind,
        user_scope=record.user_scope,
        idempotency_locator=record.idempotency_locator,
      )
      return TargetedPreparedReconciliationResult(
        record=current,
        conflict=PreparedReconciliationConflict.CAS_CONFLICT,
      )
    return TargetedPreparedReconciliationResult(
      record=reconciled,
      transitioned=True,
    )

  def _reconcile_prepared_page_in_transaction(
    self,
    conn: sqlite3.Connection,
    *,
    now: datetime,
    cursor: PreparedReconciliationCursor | None,
    page_size: int,
  ) -> PreparedReconciliationResult:
    if type(page_size) is not int or not 1 <= page_size <= 1000:
      raise ValueError("prepared reconciliation page_size must be between 1 and 1000")
    if cursor is not None and type(cursor) is not PreparedReconciliationCursor:
      raise TypeError("prepared reconciliation cursor must be typed")

    wrapped = False
    rows = self._pending_prepared_page(
      conn,
      cursor=cursor,
      page_size=page_size,
    )
    if not rows and cursor is not None:
      wrapped = True
      rows = self._pending_prepared_page(
        conn,
        cursor=None,
        page_size=page_size,
      )

    counts = {
      "authorized": 0,
      "denied": 0,
      "expired": 0,
      "missing_approval": 0,
      "unknown_approval_state": 0,
      "lineage_conflict": 0,
      "cas_conflict": 0,
    }
    next_cursor: PreparedReconciliationCursor | None = None
    for row in rows:
      record = _prepared_bm._from_row(row)
      next_cursor = self._prepared_cursor_for_record(record)
      item = self._reconcile_prepared_record_in_transaction(
        conn,
        record=record,
        now=now,
      )
      if item.conflict is not None:
        counts[item.conflict.value] += 1
      elif item.transitioned and item.record is not None:
        if item.record.lifecycle is _prepared_bm.PreparedBusinessModelLifecycle.AUTHORIZED:
          counts["authorized"] += 1
        elif item.record.lifecycle is _prepared_bm.PreparedBusinessModelLifecycle.DENIED:
          counts["denied"] += 1
        elif item.record.lifecycle is _prepared_bm.PreparedBusinessModelLifecycle.EXPIRED:
          counts["expired"] += 1
    return PreparedReconciliationResult(
      scanned=len(rows),
      cursor=next_cursor,
      wrapped=wrapped,
      **counts,
    )

  async def expire_pending(self, *, now: datetime | None = None) -> int:
    now_value = now or utc_now()
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        expired = self._expire_pending_requests_in_transaction(
          conn,
          now=now_value,
        )
        conn.commit()
    for request in expired:
      await self._emit("expired", request)
    return len(expired)

  async def maintain_pending(
    self,
    *,
    now: datetime | None = None,
    prepared_cursor: PreparedReconciliationCursor | None = None,
    prepared_page_size: int = 100,
  ) -> ApprovalMaintenanceResult:
    now_value = now or utc_now()
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        expired = self._expire_pending_requests_in_transaction(
          conn,
          now=now_value,
        )
        prepared = self._reconcile_prepared_page_in_transaction(
          conn,
          now=now_value,
          cursor=prepared_cursor,
          page_size=prepared_page_size,
        )
        conn.commit()
    for request in expired:
      await self._emit("expired", request)
    return ApprovalMaintenanceResult(
      approvals_expired=len(expired),
      prepared=prepared,
    )

  async def reconcile_prepared_business_model_change(
    self,
    *,
    caller_kind: str,
    user_scope: str,
    idempotency_locator: str,
    now: datetime | None = None,
  ) -> TargetedPreparedReconciliationResult:
    now_value = now or utc_now()
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        record = _prepared_bm.get(
          conn,
          caller_kind=caller_kind,
          user_scope=user_scope,
          idempotency_locator=idempotency_locator,
        )
        if record is None:
          conn.commit()
          return TargetedPreparedReconciliationResult(record=None)
        result = self._reconcile_prepared_record_in_transaction(
          conn,
          record=record,
          now=now_value,
        )
        conn.commit()
        return result

  async def enqueue_pending_approval_notification(self, request: ApprovalRequest) -> dict[str, Any] | None:
    if request.state != "pending_user":
      return None
    now_text = _dt_to_text(utc_now())
    if approval_notification_policy_for_request(request) == "skip":
      async with self._lock:
        with self._connection() as conn:
          conn.execute("BEGIN IMMEDIATE")
          self._insert_notification_row(
            conn,
            approval_id=request.approval_id,
            channel="",
            destination="",
            state="skipped_policy",
            message="",
            now_text=now_text,
          )
          projection = self._notification_projection_for_conn(conn, request.approval_id)
          conn.commit()
      return projection

    destinations = await self._resolve_notification_destinations(request)
    if not destinations:
      async with self._lock:
        with self._connection() as conn:
          conn.execute("BEGIN IMMEDIATE")
          self._insert_notification_row(
            conn,
            approval_id=request.approval_id,
            channel="",
            destination="",
            state="skipped_no_destination",
            message="",
            now_text=now_text,
          )
          projection = self._notification_projection_for_conn(conn, request.approval_id)
          conn.commit()
      return projection

    message = render_approval_notification_message(request)
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for destination in destinations:
          self._insert_notification_row(
            conn,
            approval_id=request.approval_id,
            channel=destination.channel,
            destination=destination.destination,
            state="pending",
            message=message,
            now_text=now_text,
          )
        projection = self._notification_projection_for_conn(conn, request.approval_id)
        conn.commit()
    return projection

  async def deliver_pending_approval_notifications(self, *, limit: int = 50) -> int:
    if self._notification_sender is None:
      return 0
    rows = await self._claim_pending_notification_rows(limit=max(1, int(limit)))
    delivered_or_failed = 0
    for row in rows:
      now_text = _dt_to_text(utc_now())
      try:
        await maybe_await(self._notification_sender(row))
      except Exception as exc:
        await self._mark_notification_failed(row, type(exc).__name__, now_text=now_text)
      else:
        await self._mark_notification_sent(row, now_text=now_text)
      delivered_or_failed += 1
    return delivered_or_failed

  def schedule_approval_notification_delivery(self, *, limit: int = 50) -> bool:
    if self._notification_sender is None:
      return False
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      return False
    if self._notification_delivery_task is not None and not self._notification_delivery_task.done():
      return False
    self._notification_delivery_task = loop.create_task(
      self.deliver_pending_approval_notifications(limit=limit)
    )
    return True

  async def get_approval_notification_projection(self, approval_id: str) -> dict[str, Any] | None:
    with self._connection() as conn:
      return self._notification_projection_for_conn(conn, approval_id)

  async def retry_failed_approval_notifications(self, approval_id: str) -> dict[str, Any]:
    now_text = _dt_to_text(utc_now())
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
          """
          UPDATE approval_notification_outbox
          SET state = 'pending',
              updated_at = ?,
              last_error = NULL
          WHERE approval_id = ?
            AND state = 'failed_retryable'
            AND EXISTS (
              SELECT 1
              FROM approval_requests
              WHERE approval_requests.approval_id = approval_notification_outbox.approval_id
                AND approval_requests.state = 'pending_user'
            )
          """,
          (now_text, approval_id),
        )
        projection = self._notification_projection_for_conn(conn, approval_id)
        conn.commit()
    return {
      "approval_id": approval_id,
      "requeued": int(cursor.rowcount or 0),
      "notification": projection,
    }

  async def list_approval_notification_outbox(self, approval_id: str | None = None) -> list[dict[str, Any]]:
    with self._connection() as conn:
      if approval_id is None:
        rows = conn.execute(
          "SELECT * FROM approval_notification_outbox ORDER BY created_at ASC"
        ).fetchall()
      else:
        rows = conn.execute(
          "SELECT * FROM approval_notification_outbox WHERE approval_id = ? ORDER BY created_at ASC",
          (approval_id,),
        ).fetchall()
    return [dict(row) for row in rows]

  async def _emit(self, event_type: str, request: ApprovalRequest, **kwargs: Any) -> None:
    if self._audit_emitter is None:
      return
    emit = getattr(self._audit_emitter, "emit_audit_for_lifecycle_event", None)
    if emit is None:
      return
    kwargs.setdefault("skill", request.skill)
    await emit(event_type=event_type, request=request, raw_tool_args={}, **kwargs)

  async def _emit_grant(
    self,
    event_type: str,
    grant: PersistentGrant,
    *,
    request: ApprovalRequest | None = None,
  ) -> None:
    if self._audit_emitter is None:
      return
    emit = getattr(self._audit_emitter, "emit_grant_event", None)
    if emit is None:
      return
    await emit(event_type=event_type, grant=grant, request=request)

  @staticmethod
  def _event_type_for_state(state: str) -> str:
    return {
      "auto_approved": "auto_approved",
      "auto_denied": "auto_denied",
      "pending_user": "user_hold_started",
      "approved": "approved",
      "denied": "denied",
      "expired": "expired",
    }.get(state, state)

  async def _resolve_notification_destinations(self, request: ApprovalRequest):
    if self._notification_destination_resolver is None:
      return []
    raw_destinations = await maybe_await(self._notification_destination_resolver(request))
    return normalize_approval_notification_destinations(raw_destinations or [])

  async def _claim_pending_notification_rows(self, *, limit: int) -> list[dict[str, Any]]:
    now_text = _dt_to_text(utc_now())
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = [
          dict(row)
          for row in conn.execute(
            """
            SELECT outbox.*
            FROM approval_notification_outbox AS outbox
            JOIN approval_requests AS approvals
              ON approvals.approval_id = outbox.approval_id
            WHERE outbox.state = 'pending'
              AND approvals.state = 'pending_user'
            ORDER BY outbox.created_at ASC
            LIMIT ?
            """,
            (limit,),
          ).fetchall()
        ]
        for row in rows:
          conn.execute(
            """
            UPDATE approval_notification_outbox
            SET state = 'processing',
                updated_at = ?,
                attempt_count = attempt_count + 1
            WHERE approval_id = ?
              AND channel = ?
              AND destination = ?
              AND state = 'pending'
            """,
            (now_text, row["approval_id"], row["channel"], row["destination"]),
          )
        conn.commit()
    return rows

  @staticmethod
  def _insert_notification_row(
    conn: sqlite3.Connection,
    *,
    approval_id: str,
    channel: str,
    destination: str,
    state: str,
    message: str,
    now_text: str | None,
  ) -> None:
    conn.execute(
      """
      INSERT OR IGNORE INTO approval_notification_outbox (
        approval_id, channel, destination, state, message, dedupe_key,
        attempt_count, last_error, created_at, updated_at, sent_at
      ) VALUES (?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, NULL)
      """,
      (
        approval_id,
        channel,
        destination,
        state,
        message,
        f"approval:{approval_id}:{channel}:{destination}",
        now_text,
        now_text,
      ),
    )

  async def _mark_notification_sent(self, row: dict[str, Any], *, now_text: str | None) -> None:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
          """
          UPDATE approval_notification_outbox
          SET state = 'sent',
              updated_at = ?,
              sent_at = ?,
              last_error = NULL
          WHERE approval_id = ?
            AND channel = ?
            AND destination = ?
            AND state IN ('pending', 'processing')
          """,
          (now_text, now_text, row["approval_id"], row["channel"], row["destination"]),
        )
        conn.commit()

  async def _mark_notification_failed(self, row: dict[str, Any], error: str, *, now_text: str | None) -> None:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
          """
          UPDATE approval_notification_outbox
          SET state = 'failed_retryable',
              updated_at = ?,
              last_error = ?
          WHERE approval_id = ?
            AND channel = ?
            AND destination = ?
            AND state IN ('pending', 'processing')
          """,
          (now_text, error[:500], row["approval_id"], row["channel"], row["destination"]),
        )
        conn.commit()

  def _row_to_request_with_projection(self, row: sqlite3.Row) -> ApprovalRequest:
    request = self._row_to_request(row)
    with self._connection() as conn:
      notification = self._notification_projection_for_conn(conn, request.approval_id)
    if notification is None:
      return request
    return replace(request, notification=notification)

  @staticmethod
  def _notification_projection_for_conn(conn: sqlite3.Connection, approval_id: str) -> dict[str, Any] | None:
    rows = conn.execute(
      """
      SELECT state, channel, sent_at
      FROM approval_notification_outbox
      WHERE approval_id = ?
      """,
      (approval_id,),
    ).fetchall()
    if not rows:
      return None
    states = {"pending" if str(row["state"]) == "processing" else str(row["state"]) for row in rows}
    state_order = [
      "sent",
      "failed_retryable",
      "failed_terminal",
      "pending",
      "skipped_no_destination",
      "skipped_policy",
    ]
    state = next((candidate for candidate in state_order if candidate in states), None)
    if state is None:
      return None
    channels: list[str] = []
    for row in rows:
      channel = str(row["channel"] or "")
      if not channel or channel not in {"telegram", "email", "push"}:
        continue
      row_state = "pending" if row["state"] == "processing" else row["state"]
      if state == "sent" and row_state != "sent":
        continue
      if state != "sent" and row_state != state:
        continue
      if channel not in channels:
        channels.append(channel)
    projection: dict[str, Any] = {"state": state, "channels": channels}
    sent_values = sorted(str(row["sent_at"]) for row in rows if row["sent_at"])
    if sent_values:
      projection["last_sent_at"] = sent_values[-1]
    return projection

  async def insert_or_verify_prepared_business_model_change(
    self,
    record: _prepared_bm.PreparedBusinessModelChange,
  ) -> tuple[_prepared_bm.PreparedBusinessModelChange, bool]:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        result = _prepared_bm.insert_or_verify(conn, record)
        conn.commit()
        return result

  async def get_prepared_business_model_change(
    self,
    *,
    caller_kind: str,
    user_scope: str,
    idempotency_locator: str,
  ) -> _prepared_bm.PreparedBusinessModelChange | None:
    async with self._lock:
      with self._connection() as conn:
        return _prepared_bm.get(
          conn,
          caller_kind=caller_kind,
          user_scope=user_scope,
          idempotency_locator=idempotency_locator,
        )

  async def transition_prepared_business_model_change(
    self,
    *,
    caller_kind: str,
    user_scope: str,
    idempotency_locator: str,
    expected: _prepared_bm.PreparedBusinessModelLifecycle,
    target: _prepared_bm.PreparedBusinessModelLifecycle,
    approval_id: str | None = None,
    approval_chain_id: str | None = None,
    execution_receipt: bytes | None = None,
    restoration_digest: str | None = None,
    checkpoint_id: str | None = None,
    consumed_at: str | None = None,
  ) -> _prepared_bm.PreparedBusinessModelChange:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        record = _prepared_bm.transition(
          conn,
          caller_kind=caller_kind,
          user_scope=user_scope,
          idempotency_locator=idempotency_locator,
          expected=expected,
          target=target,
          approval_id=approval_id,
          approval_chain_id=approval_chain_id,
          execution_receipt=execution_receipt,
          restoration_digest=restoration_digest,
          checkpoint_id=checkpoint_id,
          consumed_at=consumed_at,
        )
        conn.commit()
        return record

  async def expire_prepared_business_model_changes(self, *, now: str) -> int:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        count = _prepared_bm.expire_pending(conn, now=now)
        conn.commit()
        return count

  async def resolve_raw_patch_authorization_reference(
    self,
    reference: str,
  ) -> ApprovalRequest | None:
    """Resolve the opaque route reference without accepting raw approval IDs."""

    tool_call_id = _raw_patch_auth.decode_reference(reference)
    return await self.get_by_tool_call_id(tool_call_id)

  async def consume_raw_patch_authorization(
    self,
    *,
    approval_id: str,
    approval_chain_id: str,
    authorization_mode: str,
    grant_reference: str | None,
    cache_reference: str | None,
    user_id: str,
    tool_name: str,
    args_hash: str,
    change_set_id: str,
    change_hash: str,
    base_vector_hash: str,
    reviewed_change_binding_digest: str,
    execution_semantics_digest: str,
    prepared_payload_digest: str,
    claim_token: str,
    receipt: bytes,
  ) -> tuple[_raw_patch_auth.RawPatchAuthorizationConsumption, bool]:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        self._validate_raw_patch_authorization_locked(
          conn,
          approval_id=approval_id,
          approval_chain_id=approval_chain_id,
          authorization_mode=authorization_mode,
          grant_reference=grant_reference,
          cache_reference=cache_reference,
          user_id=user_id,
          tool_name=tool_name,
          args_hash=args_hash,
          change_set_id=change_set_id,
          change_hash=change_hash,
          base_vector_hash=base_vector_hash,
          reviewed_change_binding_digest=reviewed_change_binding_digest,
          execution_semantics_digest=execution_semantics_digest,
          prepared_payload_digest=prepared_payload_digest,
          require_unexpired=False,
        )
        claim = _raw_patch_auth.get_claim(conn, approval_id=approval_id)
        if (
          claim is None
          or claim.claim_token != claim_token
          or claim.change_set_id != change_set_id
          or claim.change_hash != change_hash
          or claim.prepared_payload_digest != prepared_payload_digest
        ):
          raise _raw_patch_auth.RawPatchAuthorizationError(
            "raw patch authorization claim changed before consumption"
          )
        result = _raw_patch_auth.consume(
          conn,
          approval_id=approval_id,
          change_set_id=change_set_id,
          change_hash=change_hash,
          receipt=receipt,
        )
        _raw_patch_auth.finalize_claim(
          conn,
          approval_id=approval_id,
          claim_token=claim_token,
        )
        conn.commit()
        return result

  async def claim_raw_patch_authorization(
    self,
    *,
    approval_id: str,
    approval_chain_id: str,
    authorization_mode: str,
    grant_reference: str | None,
    cache_reference: str | None,
    user_id: str,
    tool_name: str,
    args_hash: str,
    change_set_id: str,
    change_hash: str,
    base_vector_hash: str,
    reviewed_change_binding_digest: str,
    execution_semantics_digest: str,
    prepared_payload_digest: str,
  ) -> _raw_patch_auth.RawPatchAuthorizationClaim:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        self._validate_raw_patch_authorization_locked(
          conn,
          approval_id=approval_id,
          approval_chain_id=approval_chain_id,
          authorization_mode=authorization_mode,
          grant_reference=grant_reference,
          cache_reference=cache_reference,
          user_id=user_id,
          tool_name=tool_name,
          args_hash=args_hash,
          change_set_id=change_set_id,
          change_hash=change_hash,
          base_vector_hash=base_vector_hash,
          reviewed_change_binding_digest=reviewed_change_binding_digest,
          execution_semantics_digest=execution_semantics_digest,
          prepared_payload_digest=prepared_payload_digest,
          require_unexpired=True,
        )
        claim, _created = _raw_patch_auth.claim(
          conn,
          approval_id=approval_id,
          claim_token=secrets.token_urlsafe(32),
          change_set_id=change_set_id,
          change_hash=change_hash,
          prepared_payload_digest=prepared_payload_digest,
        )
        conn.commit()
        return claim

  async def release_raw_patch_authorization_claim(
    self,
    *,
    approval_id: str,
    approval_chain_id: str,
    authorization_mode: str,
    grant_reference: str | None,
    cache_reference: str | None,
    user_id: str,
    tool_name: str,
    args_hash: str,
    change_set_id: str,
    change_hash: str,
    base_vector_hash: str,
    reviewed_change_binding_digest: str,
    execution_semantics_digest: str,
    prepared_payload_digest: str,
    claim_token: str,
  ) -> None:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        self._validate_raw_patch_authorization_locked(
          conn,
          approval_id=approval_id,
          approval_chain_id=approval_chain_id,
          authorization_mode=authorization_mode,
          grant_reference=grant_reference,
          cache_reference=cache_reference,
          user_id=user_id,
          tool_name=tool_name,
          args_hash=args_hash,
          change_set_id=change_set_id,
          change_hash=change_hash,
          base_vector_hash=base_vector_hash,
          reviewed_change_binding_digest=reviewed_change_binding_digest,
          execution_semantics_digest=execution_semantics_digest,
          prepared_payload_digest=prepared_payload_digest,
          require_unexpired=False,
        )
        _raw_patch_auth.release_claim(
          conn,
          approval_id=approval_id,
          claim_token=claim_token,
        )
        conn.commit()

  async def heartbeat_raw_patch_authorization_claim(
    self,
    *,
    approval_id: str,
    claim_token: str,
    lease_seconds: float = 120.0,
  ) -> _raw_patch_auth.RawPatchAuthorizationClaim:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        claim = _raw_patch_auth.heartbeat_claim(
          conn,
          approval_id=approval_id,
          claim_token=claim_token,
          lease_seconds=lease_seconds,
        )
        conn.commit()
        return claim

  def _validate_raw_patch_authorization_locked(
    self,
    conn: sqlite3.Connection,
    *,
    approval_id: str,
    approval_chain_id: str,
    authorization_mode: str,
    grant_reference: str | None,
    cache_reference: str | None,
    user_id: str,
    tool_name: str,
    args_hash: str,
    change_set_id: str,
    change_hash: str,
    base_vector_hash: str,
    reviewed_change_binding_digest: str,
    execution_semantics_digest: str,
    prepared_payload_digest: str,
    require_unexpired: bool,
  ) -> None:
    row = conn.execute(
      "SELECT * FROM approval_requests WHERE approval_id = ?",
      (approval_id,),
    ).fetchone()
    if row is None:
      raise KeyError(f"approval request not found: {approval_id}")
    request = self._row_to_request(row)
    expected = {
      "approval_chain_id": approval_chain_id,
      "authorization_mode": authorization_mode,
      "grant_reference": grant_reference,
      "cache_reference": cache_reference,
      "user_id": user_id,
      "tool_name": tool_name,
      "tool_class": "state_write",
      "args_hash": args_hash,
      "identity_source": "reviewed_change_binding",
      "change_set_id": change_set_id,
      "change_hash": change_hash,
      "base_vector_hash": base_vector_hash,
      "reviewed_change_binding_digest": reviewed_change_binding_digest,
      "execution_semantics_digest": execution_semantics_digest,
    }
    if any(
      getattr(request, field_name) != expected_value
      for field_name, expected_value in expected.items()
    ):
      raise _raw_patch_auth.RawPatchAuthorizationError(
        "approval identity changed before raw patch execution"
      )
    if (
      request.state not in {"approved", "auto_approved"}
      or request.decision != request.state
    ):
      raise _raw_patch_auth.RawPatchAuthorizationError(
        "approval outcome is not executable for raw patch execution"
      )
    if (
      require_unexpired
      and request.expires_at is not None
      and request.expires_at <= datetime.now(UTC)
    ):
      raise _raw_patch_auth.RawPatchAuthorizationError(
        "approval expired before raw patch execution"
      )
    prepared = _raw_patch_auth.get_prepared(conn, approval_id=approval_id)
    if (
      prepared is None
      or prepared.prepared_payload_digest != prepared_payload_digest
    ):
      raise _raw_patch_auth.RawPatchAuthorizationError(
        "prepared patch identity changed before raw patch execution"
      )

  async def bind_raw_patch_prepared_authorization(
    self,
    *,
    approval_id: str,
    prepared_payload: bytes,
  ) -> tuple[_raw_patch_auth.RawPatchPreparedAuthorization, bool]:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
          "SELECT approval_id FROM approval_requests WHERE approval_id = ?",
          (approval_id,),
        ).fetchone()
        if row is None:
          raise KeyError(f"approval request not found: {approval_id}")
        result = _raw_patch_auth.bind_prepared(
          conn,
          approval_id=approval_id,
          prepared_payload=prepared_payload,
        )
        conn.commit()
        return result

  async def get_raw_patch_prepared_authorization(
    self,
    *,
    approval_id: str,
  ) -> _raw_patch_auth.RawPatchPreparedAuthorization | None:
    async with self._lock:
      with self._connection() as conn:
        return _raw_patch_auth.get_prepared(conn, approval_id=approval_id)

  async def get_raw_patch_authorization_consumption(
    self,
    *,
    approval_id: str,
  ) -> _raw_patch_auth.RawPatchAuthorizationConsumption | None:
    async with self._lock:
      with self._connection() as conn:
        return _raw_patch_auth.get(conn, approval_id=approval_id)

  @staticmethod
  def _request_to_row(request: ApprovalRequest) -> dict[str, Any]:
    return _rows.request_to_row(request)

  @staticmethod
  def _row_to_request(row: sqlite3.Row) -> ApprovalRequest:
    return _rows.row_to_request(row)

  @staticmethod
  def _row_to_grant(row: sqlite3.Row) -> PersistentGrant:
    return _rows.row_to_grant(row)

  @staticmethod
  def _row_to_delegation_grant(row: sqlite3.Row) -> DelegationGrant:
    return _rows.row_to_delegation_grant(row)


async def expire_pending_loop(store: ApprovalRequestStore, *, interval_seconds: float = 30.0) -> None:
  prepared_cursor: PreparedReconciliationCursor | None = None
  while True:
    await asyncio.sleep(interval_seconds)
    maintain = getattr(store, "maintain_pending", None)
    if not callable(maintain):
      await store.expire_pending()
      continue
    result = await maintain(prepared_cursor=prepared_cursor)
    prepared_cursor = result.prepared.cursor
    fields = {
      "approvals_expired": result.approvals_expired,
      "prepared_scanned": result.prepared.scanned,
      "prepared_authorized": result.prepared.authorized,
      "prepared_denied": result.prepared.denied,
      "prepared_expired": result.prepared.expired,
      "prepared_cursor": (
        result.prepared.cursor.log_token
        if result.prepared.cursor is not None
        else None
      ),
      "prepared_cursor_wrapped": result.prepared.wrapped,
      "prepared_missing_approval": result.prepared.missing_approval,
      "prepared_unknown_approval_state": result.prepared.unknown_approval_state,
      "prepared_lineage_conflict": result.prepared.lineage_conflict,
      "prepared_cas_conflict": result.prepared.cas_conflict,
    }
    log.info("approval maintenance completed", extra=fields)
    if result.prepared.conflict_count:
      log.warning("prepared BusinessModel reconciliation conflicts", extra=fields)
