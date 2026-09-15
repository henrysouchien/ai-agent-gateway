from __future__ import annotations

import asyncio
import base64
import fcntl
import hashlib
import json
import os
import stat
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

from .approval_audit import APPROVAL_AUDIT_RETENTION_DAYS, ApprovalAuditEntry
from .artifact_paths import user_data_dir


APPROVAL_AUDIT_MAX_FILE_BYTES = 16 * 1024 * 1024
APPROVAL_AUDIT_MAX_RECORD_BYTES = 512 * 1024
APPROVAL_AUDIT_MAX_RECORDS = 32_768


class AuditWriter(Protocol):
  async def write(self, entry: ApprovalAuditEntry) -> None: ...
  async def flush(self) -> None: ...
  async def query(
    self,
    *,
    approval_id: str | None = None,
    request_id: str | None = None,
    tool_call_id: str | None = None,
    approval_chain_id: str | None = None,
    event_type: str | None = None,
    user_id: str | None = None,
    tool_name: str | None = None,
    tenant_id: str | None = None,
    profile: str | None = None,
    date_range: tuple[datetime, datetime] | None = None,
    legal_hold_only: bool = False,
    limit: int = 100,
    cursor: str | None = None,
    order: Literal["asc", "desc"] = "desc",
  ) -> tuple[list[ApprovalAuditEntry], str | None]: ...
  async def apply_retention(self) -> int: ...
  async def export_for_legal_hold(self, *, case_id: str, filter: dict[str, Any], destination: str) -> str: ...


class JSONLAuditWriter:
  def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
    selected_root = user_data_dir() / "audit" / "approvals" if root is None else root
    self.root = Path(os.path.abspath(os.fspath(selected_root)))
    self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    self._require_private_directory(self.root)

  @staticmethod
  def _require_private_directory(path: Path) -> os.stat_result:
    directory_stat = os.lstat(path)
    return JSONLAuditWriter._require_private_directory_stat(
      directory_stat,
      path=path,
    )

  @staticmethod
  def _require_private_directory_stat(
    directory_stat: os.stat_result,
    *,
    path: Path,
  ) -> os.stat_result:
    if (
      not stat.S_ISDIR(directory_stat.st_mode)
      or directory_stat.st_uid != os.geteuid()
      or stat.S_IMODE(directory_stat.st_mode) & 0o022
    ):
      raise RuntimeError(
        f"approval audit directory has unsafe identity: {path}"
      )
    return directory_stat

  async def write(self, entry: ApprovalAuditEntry) -> None:
    await asyncio.to_thread(self._write_sync, entry)

  def _write_sync(self, entry: ApprovalAuditEntry) -> None:
    payload = entry.to_json_dict()
    encoded = (
      json.dumps(
        payload,
        sort_keys=True,
        default=str,
      )
      + "\n"
    ).encode("utf-8")
    if len(encoded) > APPROVAL_AUDIT_MAX_RECORD_BYTES:
      raise RuntimeError(
        "approval audit record exceeds its byte limit"
      )
    parent_stat = self._require_private_directory(self.root)
    directory_fd = os.open(
      self.root,
      os.O_RDONLY
      | getattr(os, "O_DIRECTORY", 0)
      | getattr(os, "O_CLOEXEC", 0),
    )
    try:
      opened_parent_stat = self._require_private_directory_stat(
        os.fstat(directory_fd),
        path=self.root,
      )
      if (
        opened_parent_stat.st_dev != parent_stat.st_dev
        or opened_parent_stat.st_ino != parent_stat.st_ino
      ):
        raise RuntimeError(
          f"approval audit directory identity changed: {self.root}"
        )
      segment = 0
      while True:
        path = self._segment_path(entry.ts, segment)
        if self._append_to_segment(
          directory_fd=directory_fd,
          parent_stat=parent_stat,
          path=path,
          payload=payload,
          encoded=encoded,
          entry_id=entry.entry_id,
        ):
          return
        # The segment is at capacity: roll to the next segment for the same
        # day instead of refusing the append.
        segment += 1
    finally:
      os.close(directory_fd)

  def _append_to_segment(
    self,
    *,
    directory_fd: int,
    parent_stat: os.stat_result,
    path: Path,
    payload: dict[str, Any],
    encoded: bytes,
    entry_id: str,
  ) -> bool:
    """Append one record into a day segment.

    Returns True when the record was appended (or an identical record with
    the same entry_id already exists) and False when the segment has no
    capacity left, in which case the caller rolls to the next segment.
    """

    fd = os.open(
      path.name,
      os.O_RDWR
      | os.O_CREAT
      | os.O_APPEND
      | getattr(os, "O_CLOEXEC", 0)
      | getattr(os, "O_NOFOLLOW", 0),
      0o600,
      dir_fd=directory_fd,
    )
    try:
      file_stat = os.fstat(fd)
      if (
        not stat.S_ISREG(file_stat.st_mode)
        or file_stat.st_nlink != 1
        or file_stat.st_uid != os.geteuid()
      ):
        raise RuntimeError(
          f"approval audit file has unsafe identity: {path}"
        )
      if stat.S_IMODE(file_stat.st_mode) != 0o600:
        os.fchmod(fd, 0o600)
        file_stat = os.fstat(fd)

      def require_bound_state(file_fd: int) -> None:
        current_parent_stat = self._require_private_directory_stat(
          os.fstat(directory_fd),
          path=path.parent,
        )
        visible_parent_stat = self._require_private_directory(
          path.parent
        )
        current_file_stat = os.fstat(file_fd)
        visible_file_stat = os.stat(
          path.name,
          dir_fd=directory_fd,
          follow_symlinks=False,
        )
        if (
          current_parent_stat.st_dev != parent_stat.st_dev
          or current_parent_stat.st_ino != parent_stat.st_ino
          or visible_parent_stat.st_dev != parent_stat.st_dev
          or visible_parent_stat.st_ino != parent_stat.st_ino
          or not stat.S_ISREG(current_file_stat.st_mode)
          or current_file_stat.st_dev != file_stat.st_dev
          or current_file_stat.st_ino != file_stat.st_ino
          or current_file_stat.st_uid != os.geteuid()
          or current_file_stat.st_nlink != 1
          or stat.S_IMODE(current_file_stat.st_mode) != 0o600
          or not stat.S_ISREG(visible_file_stat.st_mode)
          or visible_file_stat.st_dev != file_stat.st_dev
          or visible_file_stat.st_ino != file_stat.st_ino
          or visible_file_stat.st_uid != os.geteuid()
          or visible_file_stat.st_nlink != 1
          or stat.S_IMODE(visible_file_stat.st_mode) != 0o600
        ):
          raise RuntimeError(
            f"approval audit file identity changed: {path}"
          )

      require_bound_state(fd)
      with os.fdopen(fd, "r+b") as handle:
        fd = -1
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
          original_size = os.fstat(handle.fileno()).st_size
          handle.seek(0)
          record_count = 0
          while True:
            line = handle.readline(
              APPROVAL_AUDIT_MAX_RECORD_BYTES + 1
            )
            if not line:
              break
            record_count += 1
            if len(line) > APPROVAL_AUDIT_MAX_RECORD_BYTES:
              raise RuntimeError(
                "approval audit record exceeds its byte limit"
              )
            if not line.endswith(b"\n"):
              raise RuntimeError(
                "approval audit file has an incomplete record"
              )
            if not line.strip():
              continue
            existing = json.loads(line)
            if not isinstance(existing, dict):
              raise RuntimeError(
                "approval audit file contains a non-object record"
              )
            if existing.get("entry_id") != entry_id:
              continue
            if existing != payload:
              raise RuntimeError(
                "approval audit entry_id was reused with different content"
              )
            os.fsync(handle.fileno())
            require_bound_state(handle.fileno())
            os.fsync(directory_fd)
            return True
          if (
            original_size + len(encoded) > APPROVAL_AUDIT_MAX_FILE_BYTES
            or record_count >= APPROVAL_AUDIT_MAX_RECORDS
          ):
            return False
          handle.seek(0, os.SEEK_END)
          try:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            require_bound_state(handle.fileno())
            os.fsync(directory_fd)
          except BaseException:
            try:
              handle.seek(original_size)
              handle.truncate()
              handle.flush()
              os.fsync(handle.fileno())
              os.fsync(directory_fd)
            except BaseException:
              pass
            raise
          return True
        finally:
          fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
      if fd != -1:
        os.close(fd)

  async def flush(self) -> None:
    return None

  async def query(
    self,
    *,
    approval_id: str | None = None,
    request_id: str | None = None,
    tool_call_id: str | None = None,
    approval_chain_id: str | None = None,
    event_type: str | None = None,
    user_id: str | None = None,
    tool_name: str | None = None,
    tenant_id: str | None = None,
    profile: str | None = None,
    date_range: tuple[datetime, datetime] | None = None,
    legal_hold_only: bool = False,
    limit: int = 100,
    cursor: str | None = None,
    order: Literal["asc", "desc"] = "desc",
  ) -> tuple[list[ApprovalAuditEntry], str | None]:
    filters = {
      "approval_id": approval_id,
      "request_id": request_id,
      "tool_call_id": tool_call_id,
      "approval_chain_id": approval_chain_id,
      "event_type": event_type,
      "user_id": user_id,
      "tool_name": tool_name,
      "tenant_id": tenant_id,
      "profile": profile,
      "date_range": [_dt_text(v) for v in date_range] if date_range else None,
      "legal_hold_only": legal_hold_only,
      "order": order,
    }
    cursor_state = _decode_cursor(cursor)
    filters_hash = _filters_hash(filters)
    if cursor_state is not None and cursor_state.get("filters_hash") != filters_hash:
      raise ValueError("audit query cursor does not match filters")

    entries = []
    for path in sorted(self.root.glob("*.jsonl")):
      entries.extend(self._read_file(path))
    entries.sort(key=lambda entry: (entry.ts, entry.entry_id), reverse=order == "desc")
    if cursor_state is not None:
      last_ts = datetime.fromisoformat(cursor_state["last_ts"])
      last_entry_id = str(cursor_state["last_entry_id"])
      entries = [
        entry for entry in entries
        if ((entry.ts, entry.entry_id) < (last_ts, last_entry_id) if order == "desc" else (entry.ts, entry.entry_id) > (last_ts, last_entry_id))
      ]

    filtered = [
      entry for entry in entries
      if _matches(
        entry,
        approval_id=approval_id,
        request_id=request_id,
        tool_call_id=tool_call_id,
        approval_chain_id=approval_chain_id,
        event_type=event_type,
        user_id=user_id,
        tool_name=tool_name,
        tenant_id=tenant_id,
        profile=profile,
        date_range=date_range,
        legal_hold_only=legal_hold_only,
      )
    ]
    page = filtered[: max(0, limit)]
    next_cursor = None
    if len(filtered) > len(page) and page:
      last = page[-1]
      next_cursor = _encode_cursor(
        {
          "order": order,
          "last_ts": last.ts.isoformat(),
          "last_entry_id": last.entry_id,
          "filters_hash": filters_hash,
        }
      )
    return page, next_cursor

  async def apply_retention(self) -> int:
    now = datetime.now(UTC)
    deleted = 0
    for path in sorted(self.root.glob("*.jsonl")):
      entries = self._read_file(path)
      if not entries:
        continue
      if any(entry.legal_hold for entry in entries):
        continue
      retention_days = max(
        APPROVAL_AUDIT_RETENTION_DAYS[entry.retention_class] for entry in entries
      )
      newest = max(entry.ts for entry in entries)
      if newest + timedelta(days=retention_days) < now:
        path.unlink(missing_ok=True)
        deleted += 1
    return deleted

  async def export_for_legal_hold(self, *, case_id: str, filter: dict[str, Any], destination: str) -> str:
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    archive_path = dest / f"{case_id}-approval-audit.tar.gz"
    manifest = {"case_id": case_id, "files": []}
    with tarfile.open(archive_path, "w:gz") as archive:
      for path in sorted(self.root.glob("*.jsonl")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["files"].append({"path": str(path), "sha256": digest})
        archive.add(path, arcname=path.name)
      manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
      info = tarfile.TarInfo("manifest.json")
      info.size = len(manifest_bytes)
      archive.addfile(info, fileobj=_BytesReader(manifest_bytes))
    os.chmod(archive_path, 0o600)
    return str(archive_path)

  def _segment_path(self, ts: datetime, index: int) -> Path:
    if ts.tzinfo is None:
      ts = ts.astimezone()
    day = ts.astimezone().date().isoformat()
    name = f"{day}.jsonl" if index == 0 else f"{day}.{index}.jsonl"
    return self.root / name

  def _read_file(self, path: Path) -> list[ApprovalAuditEntry]:
    entries: list[ApprovalAuditEntry] = []
    try:
      lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
      return entries
    for line in lines:
      if not line.strip():
        continue
      entries.append(ApprovalAuditEntry.from_json_dict(json.loads(line)))
    return entries


class _BytesReader:
  def __init__(self, data: bytes) -> None:
    self._data = data
    self._offset = 0

  def read(self, size: int = -1) -> bytes:
    if size is None or size < 0:
      size = len(self._data) - self._offset
    chunk = self._data[self._offset:self._offset + size]
    self._offset += len(chunk)
    return chunk


def _matches(entry: ApprovalAuditEntry, **filters: Any) -> bool:
  for key in ("approval_id", "request_id", "tool_call_id", "approval_chain_id", "event_type", "user_id", "tool_name", "tenant_id", "profile"):
    value = filters.get(key)
    if value is not None and getattr(entry, key) != value:
      return False
  if filters.get("legal_hold_only") and not entry.legal_hold:
    return False
  date_range = filters.get("date_range")
  if date_range is not None:
    start, end = date_range
    if entry.ts < start or entry.ts > end:
      return False
  return True


def _dt_text(value: datetime) -> str:
  return value.isoformat()


def _filters_hash(filters: dict[str, Any]) -> str:
  return hashlib.sha256(json.dumps(filters, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _encode_cursor(payload: dict[str, Any]) -> str:
  return base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True).encode("utf-8")).decode("ascii")


def _decode_cursor(value: str | None) -> dict[str, Any] | None:
  if not value:
    return None
  return json.loads(base64.urlsafe_b64decode(value.encode("ascii")).decode("utf-8"))
