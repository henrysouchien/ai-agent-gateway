from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

from .approval_audit import ApprovalAuditEntry


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
  def __init__(self, root: str | os.PathLike[str] = "data/audit/approvals") -> None:
    self.root = Path(root)
    self.root.mkdir(parents=True, exist_ok=True)

  async def write(self, entry: ApprovalAuditEntry) -> None:
    path = self._path_for_ts(entry.ts)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
      os.fchmod(fd, 0o600)
      with os.fdopen(fd, "a", encoding="utf-8") as handle:
        fd = -1
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(entry.to_json_dict(), sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
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
      retention_days = max(_retention_days(entry.retention_class) for entry in entries)
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

  def _path_for_ts(self, ts: datetime) -> Path:
    if ts.tzinfo is None:
      ts = ts.astimezone()
    return self.root / f"{ts.astimezone().date().isoformat()}.jsonl"

  def _read_file(self, path: Path) -> list[ApprovalAuditEntry]:
    entries: list[ApprovalAuditEntry] = []
    try:
      lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
      return entries
    for line in lines:
      if not line.strip():
        continue
      payload = json.loads(line)
      payload["ts"] = datetime.fromisoformat(payload["ts"])
      entries.append(ApprovalAuditEntry(**payload))
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


def _retention_days(retention_class: str) -> int:
  if retention_class == "dev":
    return 30
  if retention_class == "compliance":
    return 2555
  return 365


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
