"""Emergency durability and fail-closed controls for commercial usage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

from .usage_outbox import CommercialUsageOutbox


_SPOOL_MAGIC = "HANK-COMMERCIAL-USAGE-SPOOL-V1"
_FRAME_MAGIC = "BATCH"


class CommercialUsageCircuitOpen(RuntimeError):
  """New Hank-funded work is blocked while commercial durability is unsafe."""


class CommercialUsageSpoolError(RuntimeError):
  """Emergency spool framing, persistence, or replay failed."""


@dataclass(frozen=True)
class CommercialUsageCircuitSnapshot:
  tripped: bool
  reason: str | None
  tripped_at: str | None
  byok_allowed_until: str | None


class CommercialUsageCircuitBreaker:
  def __init__(
    self,
    state_path: str | Path,
    *,
    byok_grace_seconds: float = 900.0,
    now: Callable[[], datetime] | None = None,
  ) -> None:
    if not 0 <= byok_grace_seconds <= 86_400:
      raise ValueError("commercial usage BYOK incident grace must be between 0 and 86400 seconds")
    self.path = Path(state_path).expanduser()
    self.lock_path = Path(f"{self.path}.lock")
    self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    self._byok_grace_seconds = float(byok_grace_seconds)
    self._now = now or (lambda: datetime.now(timezone.utc))

  @property
  def snapshot(self) -> CommercialUsageCircuitSnapshot:
    with self._locked():
      return self._read_unlocked()

  def trip(
    self, reason: str, *, now: datetime | None = None, replace_reason: bool = False
  ) -> bool:
    normalized = str(reason or "commercial usage durability unavailable")[:4_096]
    with self._locked():
      existing = self._read_unlocked()
      if existing.tripped and not replace_reason:
        return False
      timestamp = (now or self._now()).astimezone(timezone.utc)
      has_valid_deadline = (
        existing.tripped
        and isinstance(existing.tripped_at, str)
        and isinstance(existing.byok_allowed_until, str)
      )
      tripped_at = existing.tripped_at if has_valid_deadline else self._time_text(timestamp)
      byok_until = (
        existing.byok_allowed_until
        if has_valid_deadline else self._time_text(
          timestamp + timedelta(seconds=self._byok_grace_seconds)
        )
      )
      self._write_unlocked(CommercialUsageCircuitSnapshot(
        True,
        normalized,
        tripped_at,
        byok_until,
      ))
      return not existing.tripped

  def reset(self) -> None:
    """Operator-controlled reset after durable recovery/reconciliation."""
    with self._locked():
      self._write_unlocked(CommercialUsageCircuitSnapshot(False, None, None, None))

  def assert_work_allowed(
    self,
    billing_mode: Literal["byok", "metered"],
    *,
    now: datetime | None = None,
  ) -> None:
    snapshot = self.snapshot
    if not snapshot.tripped:
      return
    if billing_mode == "byok" and snapshot.byok_allowed_until is not None:
      deadline = datetime.fromisoformat(snapshot.byok_allowed_until.replace("Z", "+00:00"))
      current = (now or self._now()).astimezone(timezone.utc)
      if current <= deadline:
        return
    raise CommercialUsageCircuitOpen(snapshot.reason or "commercial usage circuit is open")

  @staticmethod
  def _time_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

  def _read_unlocked(self) -> CommercialUsageCircuitSnapshot:
    if not self.path.exists():
      return CommercialUsageCircuitSnapshot(False, None, None, None)
    try:
      document = json.loads(self.path.read_text(encoding="utf-8"))
      snapshot = CommercialUsageCircuitSnapshot(
        tripped=document["tripped"],
        reason=document.get("reason"),
        tripped_at=document.get("tripped_at"),
        byok_allowed_until=document.get("byok_allowed_until"),
      )
      if document.get("version") != 1 or type(snapshot.tripped) is not bool:
        raise ValueError("invalid circuit state")
      if snapshot.tripped and (
        not isinstance(snapshot.reason, str)
        or not isinstance(snapshot.tripped_at, str)
        or not isinstance(snapshot.byok_allowed_until, str)
      ):
        raise ValueError("invalid tripped circuit state")
      return snapshot
    except Exception:
      return CommercialUsageCircuitSnapshot(
        True,
        "commercial usage circuit state is corrupt",
        None,
        None,
      )

  def _write_unlocked(self, snapshot: CommercialUsageCircuitSnapshot) -> None:
    body = json.dumps(
      {"version": 1, **snapshot.__dict__}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    temporary = self.path.with_name(f".{self.path.name}.{uuid4()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
      _write_all(descriptor, body)
      os.fsync(descriptor)
    finally:
      os.close(descriptor)
    os.replace(temporary, self.path)
    os.chmod(self.path, 0o600)
    _fsync_directory(self.path.parent)

  def _locked(self):
    breaker = self

    class _Lock:
      def __enter__(self):
        self.fd = os.open(breaker.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        os.chmod(breaker.lock_path, 0o600)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

      def __exit__(self, exc_type, exc, traceback):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)

    return _Lock()


def _write_all(fd: int, data: bytes) -> None:
  view = memoryview(data)
  while view:
    written = os.write(fd, view)
    if written <= 0:
      raise CommercialUsageSpoolError("emergency spool write made no progress")
    view = view[written:]


def _fsync_directory(path: Path) -> None:
  descriptor = os.open(path, os.O_RDONLY)
  try:
    os.fsync(descriptor)
  finally:
    os.close(descriptor)


class CommercialUsageEmergencySpool:
  """Process-safe checksum-framed append log with atomic replay cursor."""

  def __init__(
    self,
    path: str | Path,
    *,
    max_batch_bytes: int = 4_000_000,
    max_spool_bytes: int = 256_000_000,
  ) -> None:
    if max_batch_bytes <= 0 or max_spool_bytes <= max_batch_bytes:
      raise ValueError("emergency spool size limits are invalid")
    self.path = Path(path).expanduser()
    self.lock_path = Path(f"{self.path}.lock")
    self.cursor_path = Path(f"{self.path}.cursor")
    self._max_batch_bytes = int(max_batch_bytes)
    self._max_spool_bytes = int(max_spool_bytes)
    self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

  def append_batch(self, payloads: list[dict[str, Any]]) -> None:
    if not payloads:
      raise ValueError("emergency commercial usage batch cannot be empty")
    try:
      body = json.dumps(
        {"payloads": payloads}, sort_keys=True, separators=(",", ":"), ensure_ascii=False
      ).encode("utf-8")
    except (TypeError, ValueError) as exc:
      raise CommercialUsageSpoolError("emergency usage batch is not JSON serializable") from exc
    checksum = hashlib.sha256(body).hexdigest()
    if len(body) > self._max_batch_bytes:
      raise CommercialUsageSpoolError("emergency usage batch exceeds spool frame limit")
    frame = f"{_FRAME_MAGIC} {len(body)} {checksum}\n".encode("ascii") + body + b"\n"
    with self._locked():
      file_id, header_end, data = self._read_or_initialize()
      self._parse_frames(data, header_end, require_complete=True)
      if len(data) + len(frame) > self._max_spool_bytes:
        raise CommercialUsageSpoolError("emergency usage spool high-water reached")
      descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND)
      try:
        _write_all(descriptor, frame)
        os.fsync(descriptor)
      finally:
        os.close(descriptor)
      os.chmod(self.path, 0o600)
      _ = file_id

  def replay_into(self, outbox: CommercialUsageOutbox, *, limit: int | None = None) -> int:
    if limit is not None and limit <= 0:
      raise ValueError("emergency spool replay limit must be positive")
    replayed = 0
    with self._locked():
      if not self.path.exists():
        return 0
      file_id, header_end, data = self._read_or_initialize()
      cursor = self._read_cursor(file_id=file_id, header_end=header_end, file_size=len(data))
      frames = self._parse_frames(data, cursor, require_complete=True)
      for end_offset, payloads in frames:
        if limit is not None and replayed >= limit:
          break
        outbox.enqueue_batch(payloads)
        self._write_cursor(file_id=file_id, offset=end_offset)
        replayed += 1
    return replayed

  def pending_batches(self) -> int:
    with self._locked():
      if not self.path.exists():
        return 0
      file_id, header_end, data = self._read_or_initialize()
      cursor = self._read_cursor(file_id=file_id, header_end=header_end, file_size=len(data))
      return len(self._parse_frames(data, cursor, require_complete=True))

  def _locked(self):
    spool = self

    class _Lock:
      def __enter__(self):
        self.fd = os.open(spool.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        os.chmod(spool.lock_path, 0o600)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

      def __exit__(self, exc_type, exc, traceback):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)

    return _Lock()

  def _read_or_initialize(self) -> tuple[str, int, bytes]:
    if not self.path.exists():
      file_id = str(uuid4())
      header = f"{_SPOOL_MAGIC} {file_id}\n".encode("ascii")
      descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
      try:
        _write_all(descriptor, header)
        os.fsync(descriptor)
      finally:
        os.close(descriptor)
      _fsync_directory(self.path.parent)
    if self.path.stat().st_size > self._max_spool_bytes:
      raise CommercialUsageSpoolError("emergency usage spool exceeds configured limit")
    data = self.path.read_bytes()
    newline = data.find(b"\n")
    if newline < 0:
      raise CommercialUsageSpoolError("emergency spool header is incomplete")
    try:
      magic, file_id = data[:newline].decode("ascii").split(" ", 1)
    except (UnicodeDecodeError, ValueError) as exc:
      raise CommercialUsageSpoolError("emergency spool header is invalid") from exc
    if magic != _SPOOL_MAGIC or not file_id:
      raise CommercialUsageSpoolError("emergency spool identity is invalid")
    return file_id, newline + 1, data

  def _parse_frames(
    self,
    data: bytes, start: int, *, require_complete: bool
  ) -> list[tuple[int, list[dict[str, Any]]]]:
    frames = []
    offset = start
    while offset < len(data):
      newline = data.find(b"\n", offset)
      if newline < 0:
        if require_complete:
          raise CommercialUsageSpoolError("emergency spool frame header is incomplete")
        break
      try:
        magic, raw_length, expected_checksum = data[offset:newline].decode("ascii").split(" ")
        length = int(raw_length)
      except (UnicodeDecodeError, ValueError) as exc:
        raise CommercialUsageSpoolError("emergency spool frame header is invalid") from exc
      if length > self._max_batch_bytes:
        raise CommercialUsageSpoolError("emergency spool frame length exceeds configured limit")
      body_start = newline + 1
      body_end = body_start + length
      frame_end = body_end + 1
      if magic != _FRAME_MAGIC or length < 2 or frame_end > len(data) or data[body_end:frame_end] != b"\n":
        raise CommercialUsageSpoolError("emergency spool frame is incomplete")
      body = data[body_start:body_end]
      if not hmac_compare(hashlib.sha256(body).hexdigest(), expected_checksum):
        raise CommercialUsageSpoolError("emergency spool frame checksum mismatch")
      try:
        document = json.loads(body)
      except (ValueError, UnicodeError) as exc:
        raise CommercialUsageSpoolError("emergency spool frame JSON is invalid") from exc
      payloads = document.get("payloads") if isinstance(document, dict) else None
      if not isinstance(payloads, list) or not payloads or not all(isinstance(item, dict) for item in payloads):
        raise CommercialUsageSpoolError("emergency spool frame payload batch is invalid")
      frames.append((frame_end, payloads))
      offset = frame_end
    return frames

  def _read_cursor(self, *, file_id: str, header_end: int, file_size: int) -> int:
    if not self.cursor_path.exists():
      return header_end
    try:
      document = json.loads(self.cursor_path.read_text(encoding="utf-8"))
      offset = int(document["offset"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
      raise CommercialUsageSpoolError("emergency spool cursor is invalid") from exc
    if document.get("version") != 1 or document.get("file_id") != file_id:
      raise CommercialUsageSpoolError("emergency spool cursor identity is invalid")
    if offset < header_end or offset > file_size:
      raise CommercialUsageSpoolError("emergency spool cursor offset is invalid")
    return offset

  def _write_cursor(self, *, file_id: str, offset: int) -> None:
    document = json.dumps(
      {"version": 1, "file_id": file_id, "offset": offset},
      sort_keys=True,
      separators=(",", ":"),
    ).encode("utf-8")
    temporary = self.cursor_path.with_name(f".{self.cursor_path.name}.{uuid4()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
      _write_all(descriptor, document)
      os.fsync(descriptor)
    finally:
      os.close(descriptor)
    os.replace(temporary, self.cursor_path)
    _fsync_directory(self.cursor_path.parent)


def hmac_compare(left: str, right: str) -> bool:
  import hmac

  return hmac.compare_digest(left, right)


class ResilientCommercialUsageSink:
  """Primary outbox + emergency spool without retrying paid provider work."""

  def __init__(
    self,
    *,
    outbox: CommercialUsageOutbox,
    spool: CommercialUsageEmergencySpool,
    circuit_breaker: CommercialUsageCircuitBreaker,
    max_backlog: int,
    max_storage_bytes: int,
    alert: Callable[[str, dict[str, Any]], None] | None = None,
  ) -> None:
    if max_backlog <= 0 or max_storage_bytes <= 0:
      raise ValueError("commercial usage durability high-water limits must be positive")
    self._outbox = outbox
    self._spool = spool
    self._breaker = circuit_breaker
    self._max_backlog = int(max_backlog)
    self._max_storage_bytes = int(max_storage_bytes)
    self._alert = alert

  def __call__(self, payloads: list[dict[str, Any]]) -> Literal[
    "outbox", "emergency_spool", "lost"
  ]:
    try:
      self._outbox.enqueue_batch(payloads)
      return "outbox"
    except Exception as primary_error:
      try:
        self._breaker.trip(f"primary commercial usage outbox failed: {primary_error}")
      except Exception as breaker_error:
        self._notify("commercial_usage.circuit_state_failed", breaker_error, len(payloads))
      self._notify("commercial_usage.primary_outbox_failed", primary_error, len(payloads))
    try:
      self._spool.append_batch(payloads)
      self._notify("commercial_usage.emergency_spool_used", None, len(payloads))
      return "emergency_spool"
    except Exception as spool_error:
      try:
        self._breaker.trip(
          f"all commercial usage durability failed: {spool_error}", replace_reason=True
        )
      except Exception as breaker_error:
        self._notify("commercial_usage.circuit_state_failed", breaker_error, len(payloads))
      self._notify("commercial_usage.all_durability_failed", spool_error, len(payloads))
    # Never raise after a provider result: the client must not retry paid work.
    return "lost"

  def producer(
    self,
    *,
    claim: Any = None,
    lineage: Any = None,
    work_start: Any = None,
    enabled: bool = True,
    reconciliation_tracker: Any | None = None,
    on_reconciliation: Callable[[Any], Any] | None = None,
  ) -> Any:
    from .commercial_usage import CommercialUsageProducer

    def persist_and_observe(report: Any) -> Any:
      self.record_reconciliation(report)
      if (
        on_reconciliation is not None
        and on_reconciliation != self.record_reconciliation
      ):
        return on_reconciliation(report)
      return None

    return CommercialUsageProducer(
      enabled=enabled,
      claim=claim,
      lineage=lineage,
      sink=self,
      work_start=work_start,
      reconciliation_tracker=reconciliation_tracker,
      on_reconciliation=persist_and_observe,
    )

  def record_reconciliation(self, report: Any) -> None:
    """Persist parity evidence without turning paid work into a client retry."""
    try:
      self._outbox.record_reconciliation_report(report)
      return
    except Exception as exc:
      try:
        self._breaker.trip(
          f"commercial usage reconciliation evidence failed: {exc}"
        )
      except Exception as breaker_error:
        self._notify(
          "commercial_usage.reconciliation_circuit_state_failed",
          breaker_error,
          0,
        )
      self._notify("commercial_usage.reconciliation_evidence_failed", exc, 0)

  def assert_work_allowed(self, billing_mode: Literal["byok", "metered"]) -> None:
    try:
      health = self._outbox.health()
    except Exception as exc:
      reason = f"commercial usage outbox health failed: {exc}"
      self._trip_and_alert("commercial_usage.outbox_health_failed", reason, exc)
      self._breaker.assert_work_allowed(billing_mode)
      return
    if not bool(health.get("ok")):
      self._trip_and_alert(
        "commercial_usage.outbox_unhealthy",
        "commercial usage outbox health check is unhealthy",
      )
    if int(health["backlog_count"]) >= self._max_backlog:
      self._trip_and_alert(
        "commercial_usage.backlog_high_water",
        "commercial usage outbox backlog high-water reached",
      )
    if int(health.get("reconciliation_shipment_backlog_count", 0)) >= self._max_backlog:
      self._trip_and_alert(
        "commercial_usage.reconciliation_backlog_high_water",
        "commercial usage reconciliation shipment backlog high-water reached",
      )
    spool_bytes = self._spool.path.stat().st_size if self._spool.path.exists() else 0
    if int(health["storage_bytes"]) + spool_bytes >= self._max_storage_bytes:
      self._trip_and_alert(
        "commercial_usage.disk_high_water",
        "commercial usage outbox disk high-water reached",
      )
    try:
      if self._spool.pending_batches() > 0:
        self._trip_and_alert(
          "commercial_usage.emergency_replay_pending",
          "commercial usage emergency spool replay is pending",
        )
    except Exception as exc:
      self._trip_and_alert(
        "commercial_usage.emergency_spool_health_failed",
        f"commercial usage emergency spool health failed: {exc}",
        exc,
      )
    if self._breaker.snapshot.reason == "commercial usage circuit state is corrupt":
      self._notify("commercial_usage.circuit_state_corrupt", None, 0)
    self._breaker.assert_work_allowed(billing_mode)

  def replay_emergency(self, *, limit: int | None = None) -> int:
    return self._spool.replay_into(self._outbox, limit=limit)

  def _notify(self, code: str, error: Exception | None, batch_size: int) -> None:
    if self._alert is None:
      return
    try:
      self._alert(code, {
        "batch_size": batch_size,
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error)[:1_024] if error is not None else None,
      })
    except Exception:
      pass

  def _trip_and_alert(
    self, code: str, reason: str, error: Exception | None = None
  ) -> None:
    try:
      newly_tripped = self._breaker.trip(reason)
    except Exception as exc:
      self._notify(code, error or exc, 0)
      raise CommercialUsageCircuitOpen(reason) from exc
    if newly_tripped:
      self._notify(code, error, 0)


@dataclass(frozen=True)
class CommercialUsageDurability:
  """Canonical production bootstrap; enabled producers cannot bypass resilience."""

  outbox: CommercialUsageOutbox
  spool: CommercialUsageEmergencySpool
  circuit_breaker: CommercialUsageCircuitBreaker
  sink: ResilientCommercialUsageSink

  @classmethod
  def create(
    cls,
    *,
    outbox_path: str | Path,
    spool_path: str | Path,
    circuit_state_path: str | Path,
    max_backlog: int,
    max_storage_bytes: int,
    byok_grace_seconds: float = 900.0,
    alert: Callable[[str, dict[str, Any]], None] | None = None,
  ) -> "CommercialUsageDurability":
    outbox = CommercialUsageOutbox(outbox_path)
    spool = CommercialUsageEmergencySpool(spool_path)
    breaker = CommercialUsageCircuitBreaker(
      circuit_state_path, byok_grace_seconds=byok_grace_seconds
    )
    sink = ResilientCommercialUsageSink(
      outbox=outbox,
      spool=spool,
      circuit_breaker=breaker,
      max_backlog=max_backlog,
      max_storage_bytes=max_storage_bytes,
      alert=alert,
    )
    return cls(outbox=outbox, spool=spool, circuit_breaker=breaker, sink=sink)

  def producer(
    self,
    *,
    claim: Any = None,
    lineage: Any = None,
    work_start: Any = None,
    enabled: bool = True,
    reconciliation_tracker: Any | None = None,
    on_reconciliation: Callable[[Any], Any] | None = None,
  ) -> Any:
    return self.sink.producer(
      claim=claim,
      lineage=lineage,
      work_start=work_start,
      enabled=enabled,
      reconciliation_tracker=reconciliation_tracker,
      on_reconciliation=on_reconciliation,
    )
