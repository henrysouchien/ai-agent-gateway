"""Authenticated delivery of durable commercial usage outbox rows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import random
import re
from typing import Any, Callable, Literal, Protocol
from urllib.parse import urlparse
from uuid import uuid4

import httpx

from .usage_outbox import CommercialUsageOutbox, CommercialUsageOutboxRow
from .usage_outbox import CommercialUsageReconciliationShipmentRow


AcceptanceStatus = Literal[
  "accepted", "duplicate", "conflict", "rejected_retryable", "rejected_terminal"
]
_ACCEPTANCE_STATUSES = frozenset({
  "accepted", "duplicate", "conflict", "rejected_retryable", "rejected_terminal",
})
_KEY_ID = re.compile(r"^[a-zA-Z0-9._:-]{1,128}$")
_NONCE = re.compile(r"^[a-zA-Z0-9._:-]{1,128}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_STABLE_CODE = re.compile(r"^[a-z][a-z0-9._:-]{0,127}$")
MAX_SHIPPER_BATCH_SIZE = 1_000


class CommercialUsageDeliveryError(RuntimeError):
  """Base class for delivery failures that leave leased rows retryable."""


class CommercialUsageResponseError(CommercialUsageDeliveryError):
  """The ingest service response was ambiguous or invalid."""


class CommercialUsagePermanentDeliveryError(CommercialUsageDeliveryError):
  """A deterministic local payload/batch failure that retries cannot repair."""


@dataclass(frozen=True)
class UsageAcceptance:
  environment: str
  source_event_id: str
  status: AcceptanceStatus
  canonical_event_id: str | None
  reason_code: str | None


class CommercialUsageBatchSender(Protocol):
  async def send_batch(self, payloads: list[dict[str, Any]]) -> list[UsageAcceptance]: ...


@dataclass(frozen=True)
class ReconciliationAcceptance:
  environment: str
  source_product: str
  request_id: str
  session_id: str
  evidence_revision: int
  report_sha256: str
  status: Literal["accepted", "duplicate", "conflict", "rejected_retryable"]
  reason_code: str | None


class CommercialUsageReconciliationBatchSender(Protocol):
  async def send_batch(
    self, reports: list[dict[str, Any]]
  ) -> list[ReconciliationAcceptance]: ...


def _canonical_request_body(payloads: list[dict[str, Any]]) -> bytes:
  return json.dumps(
    {"events": payloads},
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
  ).encode("utf-8")


def _signature_message(
  *, method: str, path: str, timestamp: str, nonce: str, body: bytes
) -> bytes:
  body_sha256 = hashlib.sha256(body).hexdigest()
  return f"{method.upper()}\n{path}\n{timestamp}\n{nonce}\n{body_sha256}".encode("utf-8")


class CommercialUsageIngestClient:
  """HMAC-authenticated client for the canonical internal batch endpoint."""

  def __init__(
    self,
    *,
    base_url: str,
    key_id: str,
    secret: bytes,
    environment: str,
    timeout_seconds: float = 10.0,
    max_batch_size: int = 100,
    max_body_bytes: int = 1_000_000,
    allow_insecure_http: bool = False,
    http_client: httpx.AsyncClient | None = None,
    now: Callable[[], datetime] | None = None,
    nonce: Callable[[], str] | None = None,
  ) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in ({"https", "http"} if allow_insecure_http else {"https"}):
      raise ValueError("commercial usage ingest requires an HTTPS base URL")
    if (
      not parsed.netloc
      or parsed.username is not None
      or parsed.password is not None
      or parsed.query
      or parsed.fragment
      or not _KEY_ID.fullmatch(key_id)
      or len(secret) < 32
    ):
      raise ValueError("commercial usage service authentication configuration is invalid")
    if (
      not environment
      or timeout_seconds <= 0
      or max_batch_size <= 0
      or max_batch_size > MAX_SHIPPER_BATCH_SIZE
      or max_body_bytes <= 0
    ):
      raise ValueError("commercial usage ingest client limits are invalid")
    self._url = base_url.rstrip("/") + "/internal/commercial/usage-events:batch"
    self._path = urlparse(self._url).path
    self._key_id = key_id
    self._secret = bytes(secret)
    self._environment = environment
    self._timeout_seconds = float(timeout_seconds)
    self._max_batch_size = int(max_batch_size)
    self._max_body_bytes = int(max_body_bytes)
    self._http_client = http_client
    self._now = now or (lambda: datetime.now(timezone.utc))
    self._nonce = nonce or (lambda: str(uuid4()))

  async def send_batch(self, payloads: list[dict[str, Any]]) -> list[UsageAcceptance]:
    if not payloads or len(payloads) > self._max_batch_size:
      raise CommercialUsagePermanentDeliveryError("commercial usage ingest batch size is invalid")
    identities = []
    for payload in payloads:
      event_id = str(payload.get("source_event_id") or "").strip()
      environment = str(payload.get("environment") or "").strip()
      if not event_id or environment != self._environment:
        raise CommercialUsagePermanentDeliveryError(
          "commercial usage ingest payload identity/environment is invalid"
        )
      identities.append((environment, event_id))
    if len(set(identities)) != len(identities):
      raise CommercialUsagePermanentDeliveryError(
        "commercial usage ingest batch contains duplicate identities"
      )

    try:
      body = _canonical_request_body(payloads)
    except (TypeError, ValueError) as exc:
      raise CommercialUsagePermanentDeliveryError(
        "commercial usage ingest payload is not JSON serializable"
      ) from exc
    if len(body) > self._max_body_bytes:
      raise CommercialUsagePermanentDeliveryError(
        "commercial usage ingest body exceeds configured limit"
      )
    request_time = self._now()
    if request_time.tzinfo is None:
      raise ValueError("commercial usage ingest clock must be timezone-aware")
    timestamp = str(int(request_time.astimezone(timezone.utc).timestamp()))
    nonce = self._nonce()
    if not isinstance(nonce, str) or not _NONCE.fullmatch(nonce):
      raise ValueError("commercial usage ingest nonce is invalid")
    signature = hmac.new(
      self._secret,
      _signature_message(
        method="POST", path=self._path, timestamp=timestamp, nonce=nonce, body=body
      ),
      hashlib.sha256,
    ).hexdigest()
    headers = {
      "content-type": "application/json",
      "x-hank-service-key-id": self._key_id,
      "x-hank-request-timestamp": timestamp,
      "x-hank-request-nonce": nonce,
      "x-hank-request-signature": f"v1={signature}",
    }
    if self._http_client is not None:
      response = await self._http_client.post(
        self._url, content=body, headers=headers, timeout=self._timeout_seconds
      )
    else:
      async with httpx.AsyncClient() as client:
        response = await client.post(
          self._url, content=body, headers=headers, timeout=self._timeout_seconds
        )
    if response.status_code < 200 or response.status_code >= 300:
      if response.status_code in {400, 413, 422}:
        raise CommercialUsagePermanentDeliveryError(
          f"commercial usage ingest rejected the request with HTTP {response.status_code}"
        )
      raise CommercialUsageDeliveryError(
        f"commercial usage ingest returned HTTP {response.status_code}"
      )
    if len(response.content) > self._max_body_bytes:
      raise CommercialUsageResponseError("commercial usage ingest response exceeds configured limit")
    try:
      document = response.json()
    except (ValueError, UnicodeError) as exc:
      raise CommercialUsageResponseError("commercial usage ingest response is not JSON") from exc
    return self._validate_response(document, expected=set(identities))

  @staticmethod
  def _validate_response(
    document: Any, *, expected: set[tuple[str, str]]
  ) -> list[UsageAcceptance]:
    raw_results = document.get("results") if isinstance(document, dict) else None
    if not isinstance(raw_results, list):
      raise CommercialUsageResponseError("commercial usage ingest response has no results")
    results: list[UsageAcceptance] = []
    observed: set[tuple[str, str]] = set()
    for item in raw_results:
      if not isinstance(item, dict):
        raise CommercialUsageResponseError("commercial usage ingest result is not an object")
      environment = str(item.get("environment") or "").strip()
      event_id = str(item.get("source_event_id") or "").strip()
      status = str(item.get("status") or "").strip()
      identity = (environment, event_id)
      if identity not in expected or identity in observed or status not in _ACCEPTANCE_STATUSES:
        raise CommercialUsageResponseError("commercial usage ingest result identity/status is invalid")
      canonical_event_id = item.get("canonical_event_id")
      reason_code = item.get("reason_code")
      if canonical_event_id is not None and not isinstance(canonical_event_id, str):
        raise CommercialUsageResponseError("commercial usage canonical identity is invalid")
      if reason_code is not None and not isinstance(reason_code, str):
        raise CommercialUsageResponseError("commercial usage reason code is invalid")
      if status in {"accepted", "duplicate"} and not str(canonical_event_id or "").strip():
        raise CommercialUsageResponseError("accepted commercial usage result lacks canonical identity")
      observed.add(identity)
      results.append(UsageAcceptance(
        environment=environment,
        source_event_id=event_id,
        status=status,  # type: ignore[arg-type]
        canonical_event_id=canonical_event_id,
        reason_code=reason_code,
      ))
    if observed != expected:
      raise CommercialUsageResponseError("commercial usage ingest response is incomplete")
    return results


class CommercialUsageReconciliationIngestClient:
  """HMAC client for the separate reconciliation-evidence endpoint."""

  def __init__(
    self,
    *,
    base_url: str,
    key_id: str,
    secret: bytes,
    environment: str,
    timeout_seconds: float = 10.0,
    max_batch_size: int = 100,
    max_body_bytes: int = 1_000_000,
    allow_insecure_http: bool = False,
    http_client: httpx.AsyncClient | None = None,
    now: Callable[[], datetime] | None = None,
    nonce: Callable[[], str] | None = None,
  ) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in ({"https", "http"} if allow_insecure_http else {"https"}):
      raise ValueError("commercial reconciliation ingest requires an HTTPS base URL")
    if (
      not parsed.netloc
      or parsed.username is not None
      or parsed.password is not None
      or parsed.query
      or parsed.fragment
      or not _KEY_ID.fullmatch(key_id)
      or len(secret) < 32
      or not environment
      or timeout_seconds <= 0
      or not 0 < max_batch_size <= MAX_SHIPPER_BATCH_SIZE
      or max_body_bytes <= 0
    ):
      raise ValueError("commercial reconciliation service configuration is invalid")
    self._url = base_url.rstrip("/") + "/internal/commercial/usage-reconciliation:batch"
    self._path = urlparse(self._url).path
    self._key_id = key_id
    self._secret = bytes(secret)
    self._environment = environment
    self._timeout_seconds = float(timeout_seconds)
    self._max_batch_size = int(max_batch_size)
    self._max_body_bytes = int(max_body_bytes)
    self._http_client = http_client
    self._now = now or (lambda: datetime.now(timezone.utc))
    self._nonce = nonce or (lambda: str(uuid4()))

  async def send_batch(
    self, reports: list[dict[str, Any]]
  ) -> list[ReconciliationAcceptance]:
    if not reports or len(reports) > self._max_batch_size:
      raise CommercialUsagePermanentDeliveryError(
        "commercial reconciliation ingest batch size is invalid"
      )
    identities = []
    for report in reports:
      evidence = report.get("evidence") if isinstance(report, dict) else None
      report_sha256 = report.get("report_sha256") if isinstance(report, dict) else None
      if not isinstance(evidence, dict) or not isinstance(report_sha256, str):
        raise CommercialUsagePermanentDeliveryError(
          "commercial reconciliation envelope is invalid"
        )
      identity = (
        evidence.get("environment"), evidence.get("source_product"),
        evidence.get("request_id"), evidence.get("session_id"),
        evidence.get("evidence_revision"), report_sha256,
      )
      if (
        identity[0] != self._environment
        or any(not isinstance(value, str) or not value for value in identity[:4])
        or not isinstance(identity[4], int) or isinstance(identity[4], bool)
        or not _SHA256.fullmatch(identity[5])
      ):
        raise CommercialUsagePermanentDeliveryError(
          "commercial reconciliation identity/environment is invalid"
        )
      identities.append(identity)
    if len(set(identities)) != len(identities):
      raise CommercialUsagePermanentDeliveryError(
        "commercial reconciliation batch contains duplicate identities"
      )
    try:
      body = json.dumps(
        {"reports": reports}, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
      ).encode("utf-8")
    except (TypeError, ValueError) as exc:
      raise CommercialUsagePermanentDeliveryError(
        "commercial reconciliation evidence is not JSON serializable"
      ) from exc
    if len(body) > self._max_body_bytes:
      raise CommercialUsagePermanentDeliveryError(
        "commercial reconciliation body exceeds configured limit"
      )
    request_time = self._now()
    if request_time.tzinfo is None:
      raise ValueError("commercial reconciliation ingest clock must be timezone-aware")
    timestamp = str(int(request_time.astimezone(timezone.utc).timestamp()))
    nonce = self._nonce()
    if not isinstance(nonce, str) or not _NONCE.fullmatch(nonce):
      raise ValueError("commercial reconciliation ingest nonce is invalid")
    signature = hmac.new(
      self._secret,
      _signature_message(
        method="POST", path=self._path, timestamp=timestamp, nonce=nonce, body=body
      ),
      hashlib.sha256,
    ).hexdigest()
    headers = {
      "content-type": "application/json",
      "x-hank-service-key-id": self._key_id,
      "x-hank-request-timestamp": timestamp,
      "x-hank-request-nonce": nonce,
      "x-hank-request-signature": f"v1={signature}",
    }
    if self._http_client is not None:
      response = await self._http_client.post(
        self._url, content=body, headers=headers, timeout=self._timeout_seconds
      )
    else:
      async with httpx.AsyncClient() as client:
        response = await client.post(
          self._url, content=body, headers=headers, timeout=self._timeout_seconds
        )
    if not 200 <= response.status_code < 300:
      if response.status_code in {400, 413, 422}:
        raise CommercialUsagePermanentDeliveryError(
          f"commercial reconciliation ingest rejected HTTP {response.status_code}"
        )
      raise CommercialUsageDeliveryError(
        f"commercial reconciliation ingest returned HTTP {response.status_code}"
      )
    if len(response.content) > self._max_body_bytes:
      raise CommercialUsageResponseError(
        "commercial reconciliation response exceeds configured limit"
      )
    try:
      document = response.json()
    except (ValueError, UnicodeError) as exc:
      raise CommercialUsageResponseError(
        "commercial reconciliation response is not JSON"
      ) from exc
    raw_results = document.get("results") if isinstance(document, dict) else None
    if not isinstance(raw_results, list):
      raise CommercialUsageResponseError("commercial reconciliation response has no results")
    expected = set(identities)
    observed = set()
    results = []
    for item in raw_results:
      if not isinstance(item, dict):
        raise CommercialUsageResponseError("commercial reconciliation result is invalid")
      identity = (
        item.get("environment"), item.get("source_product"), item.get("request_id"),
        item.get("session_id"), item.get("evidence_revision"), item.get("report_sha256"),
      )
      status = item.get("status")
      reason = item.get("reason_code")
      if (
        item.get("schema_version") != 1
        or any(not isinstance(value, str) for value in identity[:4])
        or not isinstance(identity[4], int) or isinstance(identity[4], bool)
        or not isinstance(identity[5], str) or not _SHA256.fullmatch(identity[5])
        or identity not in expected or identity in observed
        or status not in {"accepted", "duplicate", "conflict", "rejected_retryable"}
        or (reason is not None and (
          not isinstance(reason, str) or not _STABLE_CODE.fullmatch(reason)
        ))
        or (status in {"conflict", "rejected_retryable"})
           != (reason is not None)
      ):
        raise CommercialUsageResponseError(
          "commercial reconciliation result identity/status is invalid"
        )
      observed.add(identity)
      results.append(ReconciliationAcceptance(
        environment=identity[0], source_product=identity[1],
        request_id=identity[2], session_id=identity[3], evidence_revision=identity[4],
        report_sha256=identity[5], status=status, reason_code=reason,
      ))
    if observed != expected:
      raise CommercialUsageResponseError("commercial reconciliation response is incomplete")
    return results


@dataclass(frozen=True)
class CommercialUsageShipperConfig:
  batch_size: int = 100
  lease_seconds: float = 30.0
  base_backoff_seconds: float = 1.0
  max_backoff_seconds: float = 300.0
  jitter_ratio: float = 0.2
  max_event_attempts: int = 12
  poll_interval_seconds: float = 1.0

  def __post_init__(self) -> None:
    if (
      self.batch_size <= 0
      or self.batch_size > MAX_SHIPPER_BATCH_SIZE
      or self.lease_seconds <= 0
      or self.base_backoff_seconds <= 0
      or self.max_backoff_seconds < self.base_backoff_seconds
      or not 0 <= self.jitter_ratio <= 1
      or self.max_event_attempts <= 0
      or self.poll_interval_seconds <= 0
    ):
      raise ValueError("commercial usage shipper configuration is invalid")


class CommercialUsageShipper:
  """Lease, deliver, and fence outbox rows without ambiguous acceptance."""

  def __init__(
    self,
    *,
    outbox: CommercialUsageOutbox,
    sender: CommercialUsageBatchSender,
    config: CommercialUsageShipperConfig | None = None,
    random_source: random.Random | None = None,
    metric: Callable[[str, int], None] | None = None,
  ) -> None:
    self._outbox = outbox
    self._sender = sender
    self._config = config or CommercialUsageShipperConfig()
    self._random = random_source or random.Random()
    self._metric = metric

  async def run_once(self, *, now: datetime | None = None) -> int:
    current = now or datetime.now(timezone.utc)
    health = self._outbox.health(now=current)
    self._emit_metric("commercial_usage.backlog", int(health["backlog_count"]))
    self._emit_metric(
      "commercial_usage.oldest_backlog_age_seconds",
      int(health["oldest_backlog_age_seconds"] or 0),
    )
    rows = self._outbox.lease_batch(
      limit=self._config.batch_size,
      lease_for=timedelta(seconds=self._config.lease_seconds),
      now=current,
    )
    if not rows:
      return 0
    await self._deliver_rows(rows, current=current)
    return len(rows)

  async def _deliver_rows(
    self, rows: list[CommercialUsageOutboxRow], *, current: datetime
  ) -> None:
    try:
      results = await self._sender.send_batch([row.payload for row in rows])
    except CommercialUsagePermanentDeliveryError as exc:
      if len(rows) > 1:
        midpoint = len(rows) // 2
        await self._deliver_rows(rows[:midpoint], current=current)
        await self._deliver_rows(rows[midpoint:], current=current)
      else:
        row = rows[0]
        changed = self._outbox.mark_dead(
          row.event_id,
          row.sending_lease_token or "",
          error=f"permanent_delivery_error:{type(exc).__name__}:{exc}",
        )
        self._emit_metric("commercial_usage.dead", int(changed))
        self._emit_metric("commercial_usage.rejected_terminal", int(changed))
      return
    except Exception as exc:
      error = f"delivery_error:{type(exc).__name__}:{exc}"
      for row in rows:
        self._retry(row, current=current, error=error, terminal_on_exhaustion=False)
      self._emit_metric("commercial_usage.delivery_error", len(rows))
      return

    by_id = {result.source_event_id: result for result in results}
    expected = {(row.payload["environment"], row.event_id) for row in rows}
    observed = {(result.environment, result.source_event_id) for result in results}
    valid_statuses = all(result.status in _ACCEPTANCE_STATUSES for result in results)
    if len(results) != len(rows) or len(by_id) != len(results) or observed != expected or not valid_statuses:
      for row in rows:
        self._retry(
          row,
          current=current,
          error="ambiguous_response:result identity mismatch",
          terminal_on_exhaustion=False,
        )
      self._emit_metric("commercial_usage.ambiguous_response", len(rows))
      return

    for row in rows:
      result = by_id[row.event_id]
      reason = (result.reason_code or result.status)[:512]
      if result.status in {"accepted", "duplicate"}:
        changed = self._outbox.mark_accepted(
          row.event_id,
          row.sending_lease_token or "",
          ingest_status=result.status,
          canonical_event_id=result.canonical_event_id or "",
          reason_code=result.reason_code,
          accepted_at=current,
        )
        self._emit_metric(
          "commercial_usage.accepted" if result.status == "accepted" else "commercial_usage.duplicate",
          int(changed),
        )
        if changed:
          self._emit_metric(
            "commercial_usage.ingest_lag_seconds",
            self._event_lag_seconds(row, current=current),
          )
      elif result.status in {"conflict", "rejected_terminal"}:
        changed = self._outbox.mark_dead(
          row.event_id,
          row.sending_lease_token or "",
          error=f"{result.status}:{reason}",
          ingest_status=result.status,
          canonical_event_id=result.canonical_event_id,
          reason_code=reason,
          decided_at=current,
        )
        self._emit_metric("commercial_usage.dead", int(changed))
        self._emit_metric(f"commercial_usage.{result.status}", int(changed))
      else:
        self._emit_metric("commercial_usage.rejected_retryable", 1)
        if "unknown_rate" in reason:
          self._emit_metric("commercial_usage.unknown_rate_version", 1)
        self._retry(
          row,
          current=current,
          error=f"rejected_retryable:{reason}",
          terminal_on_exhaustion=True,
        )

  @staticmethod
  def _event_lag_seconds(row: CommercialUsageOutboxRow, *, current: datetime) -> int:
    try:
      occurred = datetime.fromisoformat(
        str(row.payload.get("occurred_at") or "").replace("Z", "+00:00")
      )
      if occurred.tzinfo is None:
        return 0
      return max(0, int((current.astimezone(timezone.utc) - occurred).total_seconds()))
    except (TypeError, ValueError):
      return 0

  async def run_forever(self, stop: asyncio.Event) -> None:
    while not stop.is_set():
      await self.run_once()
      try:
        await asyncio.wait_for(stop.wait(), timeout=self._config.poll_interval_seconds)
      except TimeoutError:
        continue

  def _retry(
    self,
    row: CommercialUsageOutboxRow,
    *,
    current: datetime,
    error: str,
    terminal_on_exhaustion: bool,
  ) -> None:
    token = row.sending_lease_token or ""
    if terminal_on_exhaustion and row.attempt_count >= self._config.max_event_attempts:
      changed = self._outbox.mark_dead(row.event_id, token, error=f"attempts_exhausted:{error}")
      self._emit_metric("commercial_usage.dead", int(changed))
      return
    exponent = min(52, max(0, row.attempt_count - 1))
    raw_delay = min(
      self._config.max_backoff_seconds,
      self._config.base_backoff_seconds * (2 ** exponent),
    )
    jitter = raw_delay * self._config.jitter_ratio
    delay = max(0.0, raw_delay + self._random.uniform(-jitter, jitter))
    changed = self._outbox.mark_retryable(
      row.event_id,
      token,
      next_attempt_at=current + timedelta(seconds=delay),
      error=error,
    )
    self._emit_metric("commercial_usage.retryable", int(changed))

  def _emit_metric(self, name: str, value: int) -> None:
    if self._metric is not None:
      try:
        self._metric(name, value)
      except Exception:
        pass


class CommercialUsageReconciliationShipper:
  """Deliver durable reconciliation envelopes without bypassing revision order."""

  def __init__(
    self,
    *,
    outbox: CommercialUsageOutbox,
    sender: CommercialUsageReconciliationBatchSender,
    config: CommercialUsageShipperConfig | None = None,
    random_source: random.Random | None = None,
    metric: Callable[[str, int], None] | None = None,
  ) -> None:
    self._outbox = outbox
    self._sender = sender
    self._config = config or CommercialUsageShipperConfig()
    self._random = random_source or random.Random()
    self._metric = metric
    self._outbox.enable_reconciliation_shipping()

  async def run_once(self, *, now: datetime | None = None) -> int:
    current = now or datetime.now(timezone.utc)
    health = self._outbox.health(now=current)
    self._emit_metric(
      "commercial_usage_reconciliation.backlog",
      int(health["reconciliation_shipment_backlog_count"]),
    )
    rows = self._outbox.lease_reconciliation_batch(
      limit=self._config.batch_size,
      lease_for=timedelta(seconds=self._config.lease_seconds),
      now=current,
    )
    if not rows:
      return 0
    await self._deliver_rows(rows, current=current)
    return len(rows)

  async def _deliver_rows(
    self,
    rows: list[CommercialUsageReconciliationShipmentRow],
    *,
    current: datetime,
  ) -> None:
    try:
      try:
        envelopes = [row.envelope for row in rows]
      except Exception as exc:
        raise CommercialUsagePermanentDeliveryError(
          f"stored reconciliation envelope is invalid: {type(exc).__name__}:{exc}"
        ) from exc
      results = await self._sender.send_batch(envelopes)
    except CommercialUsagePermanentDeliveryError as exc:
      if len(rows) > 1:
        midpoint = len(rows) // 2
        await self._deliver_rows(rows[:midpoint], current=current)
        await self._deliver_rows(rows[midpoint:], current=current)
      else:
        changed = self._outbox.mark_reconciliation_dead(
          rows[0].report_id,
          rows[0].sending_lease_token or "",
          error=f"permanent_delivery_error:{type(exc).__name__}:{exc}",
        )
        self._emit_metric("commercial_usage_reconciliation.dead", int(changed))
      return
    except Exception as exc:
      for row in rows:
        self._retry(
          row,
          current=current,
          error=f"delivery_error:{type(exc).__name__}:{exc}",
          terminal_on_exhaustion=False,
        )
      self._emit_metric("commercial_usage_reconciliation.delivery_error", len(rows))
      return
    by_digest = {result.report_sha256: result for result in results}
    expected = {
      (
        row.environment, row.source_product, row.request_id, row.session_id,
        row.revision, row.report_sha256,
      )
      for row in rows
    }
    observed = {
      (
        result.environment, result.source_product, result.request_id,
        result.session_id, result.evidence_revision, result.report_sha256,
      )
      for result in results
    }
    valid_results = all(
      result.status in {"accepted", "duplicate", "conflict", "rejected_retryable"}
      and (result.reason_code is None or isinstance(result.reason_code, str))
      for result in results
    )
    if (
      len(results) != len(rows)
      or len(by_digest) != len(results)
      or observed != expected
      or not valid_results
    ):
      for row in rows:
        self._retry(
          row, current=current, error="ambiguous_response:identity mismatch",
          terminal_on_exhaustion=False,
        )
      self._emit_metric("commercial_usage_reconciliation.ambiguous_response", len(rows))
      return
    for row in rows:
      result = by_digest[row.report_sha256]
      reason = (result.reason_code or result.status)[:512]
      if result.status in {"accepted", "duplicate"}:
        changed = self._outbox.mark_reconciliation_accepted(
          row.report_id,
          row.sending_lease_token or "",
          ingest_status=result.status,
          reason_code=result.reason_code,
          accepted_at=current,
        )
        self._emit_metric(
          f"commercial_usage_reconciliation.{result.status}", int(changed)
        )
      elif result.status == "conflict":
        changed = self._outbox.mark_reconciliation_dead(
          row.report_id,
          row.sending_lease_token or "",
          error=f"conflict:{reason}",
          ingest_status="conflict",
          reason_code=reason,
          decided_at=current,
        )
        self._emit_metric("commercial_usage_reconciliation.conflict", int(changed))
      else:
        self._retry(
          row,
          current=current,
          error=f"rejected_retryable:{reason}",
          terminal_on_exhaustion=True,
        )

  async def run_forever(self, stop: asyncio.Event) -> None:
    while not stop.is_set():
      await self.run_once()
      try:
        await asyncio.wait_for(stop.wait(), timeout=self._config.poll_interval_seconds)
      except TimeoutError:
        continue

  def _retry(
    self,
    row: CommercialUsageReconciliationShipmentRow,
    *,
    current: datetime,
    error: str,
    terminal_on_exhaustion: bool,
  ) -> None:
    token = row.sending_lease_token or ""
    if terminal_on_exhaustion and row.attempt_count >= self._config.max_event_attempts:
      changed = self._outbox.mark_reconciliation_dead(
        row.report_id, token, error=f"attempts_exhausted:{error}"
      )
      self._emit_metric("commercial_usage_reconciliation.dead", int(changed))
      return
    exponent = min(52, max(0, row.attempt_count - 1))
    raw_delay = min(
      self._config.max_backoff_seconds,
      self._config.base_backoff_seconds * (2 ** exponent),
    )
    jitter = raw_delay * self._config.jitter_ratio
    delay = max(0.0, raw_delay + self._random.uniform(-jitter, jitter))
    changed = self._outbox.mark_reconciliation_retryable(
      row.report_id,
      token,
      next_attempt_at=current + timedelta(seconds=delay),
      error=error,
    )
    self._emit_metric("commercial_usage_reconciliation.retryable", int(changed))

  def _emit_metric(self, name: str, value: int) -> None:
    if self._metric is not None:
      try:
        self._metric(name, value)
      except Exception:
        pass
