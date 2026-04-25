from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol


DEFAULT_USAGE_DLQ_PATH = Path("~/.gateway/usage_dlq.jsonl").expanduser()


@dataclass(frozen=True)
class UsageEvent:
  user_id: str
  session_id: str
  request_id: str
  parent_turn_id: str | None
  timestamp: float
  model: str
  input_tokens: int
  output_tokens: int
  cache_read_tokens: int
  cache_creation_tokens: int
  cost_usd: float
  rate_table_version: str
  billing_mode: Literal["byok", "metered"]
  channel: str | None
  event_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(frozen=True)
class SessionUsageSummary:
  """Authoritative per-session billing total.

  `cost` is the billing total for the completed session. Stream-level
  `estimated_cost` values remain live per-turn estimates and can differ when
  background tasks complete before or after the main stream completes.
  """

  user_id: str
  session_id: str
  request_id: str
  input_tokens: int
  output_tokens: int
  cache_read_tokens: int
  cache_creation_tokens: int
  cost: float
  turns: int
  channel: str | None
  started_at: float
  ended_at: float
  drain_complete: bool = True
  in_flight_task_count: int = 0


@dataclass
class UsageTotal:
  user_id: str
  input_tokens: int
  output_tokens: int
  cache_read_tokens: int
  cache_creation_tokens: int
  cost_usd: float
  event_count: int
  since: float | None
  until: float | None


def normalize_identity(
  user_id: str | None,
  rate_table_version: str | None,
  billing_mode: str | None,
  channel: str | None,
) -> tuple[str, str, Literal["byok", "metered"], str | None]:
  """Apply the gateway's default usage identity semantics."""
  normalized_user_id = str(user_id or "_default")
  normalized_rate_table_version = str(rate_table_version or "unknown")
  normalized_billing_mode: Literal["byok", "metered"] = (
    "metered" if str(billing_mode or "").strip().lower() == "metered" else "byok"
  )
  normalized_channel = channel.strip() if isinstance(channel, str) and channel.strip() else None
  return normalized_user_id, normalized_rate_table_version, normalized_billing_mode, normalized_channel


class _UsageAggregator:
  """Accumulates UsageEvent objects within one asyncio event loop.

  This is an internal runner coordination primitive. It is protected by an
  asyncio.Lock and is not cross-loop or thread safe.
  """

  def __init__(
    self,
    *,
    user_id: str,
    session_id: str,
    request_id: str,
    channel: str | None,
    started_at: float | None = None,
  ) -> None:
    self._lock = asyncio.Lock()
    self._user_id = user_id
    self._session_id = session_id
    self._request_id = request_id
    self._channel = channel
    self._started_at = started_at if started_at is not None else time.time()
    self._input_tokens = 0
    self._output_tokens = 0
    self._cache_read_tokens = 0
    self._cache_creation_tokens = 0
    self._cost = 0.0
    self._turns = 0
    self._closed = False

  async def record(self, event: UsageEvent) -> bool:
    async with self._lock:
      if self._closed:
        return False
      self._input_tokens += int(event.input_tokens or 0)
      self._output_tokens += int(event.output_tokens or 0)
      self._cache_read_tokens += int(event.cache_read_tokens or 0)
      self._cache_creation_tokens += int(event.cache_creation_tokens or 0)
      self._cost += float(event.cost_usd or 0.0)
      self._turns += 1
      return True

  async def close(self) -> None:
    async with self._lock:
      self._closed = True

  async def set_turns(self, turns: int) -> None:
    async with self._lock:
      self._turns = max(0, int(turns or 0))

  async def snapshot(
    self,
    *,
    ended_at: float | None = None,
    drain_complete: bool = True,
    in_flight_task_count: int = 0,
  ) -> SessionUsageSummary:
    async with self._lock:
      return SessionUsageSummary(
        user_id=self._user_id,
        session_id=self._session_id,
        request_id=self._request_id,
        input_tokens=self._input_tokens,
        output_tokens=self._output_tokens,
        cache_read_tokens=self._cache_read_tokens,
        cache_creation_tokens=self._cache_creation_tokens,
        cost=self._cost,
        turns=self._turns,
        channel=self._channel,
        started_at=self._started_at,
        ended_at=ended_at if ended_at is not None else time.time(),
        drain_complete=drain_complete,
        in_flight_task_count=in_flight_task_count,
      )


class UsageLedger(Protocol):
  async def record(self, event: UsageEvent) -> None: ...

  async def get_total(
    self,
    user_id: str,
    *,
    since: float | None = None,
    until: float | None = None,
    billing_mode: Literal["byok", "metered"] | None = None,
    model: str | None = None,
  ) -> UsageTotal: ...


class SqliteUsageLedger:
  """Reference implementation backed by SQLite with WAL mode."""

  def __init__(self, db_path: str | Path):
    self._db_path = Path(db_path)
    self._db_path.parent.mkdir(parents=True, exist_ok=True)
    self._init_conn = self._connect()
    self._ensure_schema(self._init_conn)
    self._closed = False

  def _connect(self) -> sqlite3.Connection:
    conn = sqlite3.connect(
      self._db_path,
      timeout=5.0,
      isolation_level=None,
      check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn

  @staticmethod
  def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS usage_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        request_id TEXT NOT NULL,
        parent_turn_id TEXT,
        timestamp REAL NOT NULL,
        model TEXT NOT NULL,
        input_tokens INTEGER NOT NULL,
        output_tokens INTEGER NOT NULL,
        cache_read_tokens INTEGER NOT NULL,
        cache_creation_tokens INTEGER NOT NULL,
        cost_usd REAL NOT NULL,
        rate_table_version TEXT NOT NULL,
        billing_mode TEXT NOT NULL CHECK (billing_mode IN ('byok', 'metered')),
        channel TEXT
      )
      """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_user_time ON usage_events(user_id, timestamp DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_user_mode ON usage_events(user_id, billing_mode, timestamp DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_request ON usage_events(request_id)")

  def _assert_open(self) -> None:
    if self._closed:
      raise RuntimeError("Usage ledger is closed")

  def _record_sync(self, event: UsageEvent) -> None:
    self._assert_open()
    with self._connect() as conn:
      self._ensure_schema(conn)
      conn.execute(
        """
        INSERT INTO usage_events (
          user_id,
          session_id,
          request_id,
          parent_turn_id,
          timestamp,
          model,
          input_tokens,
          output_tokens,
          cache_read_tokens,
          cache_creation_tokens,
          cost_usd,
          rate_table_version,
          billing_mode,
          channel
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
          event.user_id,
          event.session_id,
          event.request_id,
          event.parent_turn_id,
          event.timestamp,
          event.model,
          event.input_tokens,
          event.output_tokens,
          event.cache_read_tokens,
          event.cache_creation_tokens,
          event.cost_usd,
          event.rate_table_version,
          event.billing_mode,
          event.channel,
        ),
      )

  async def record(self, event: UsageEvent) -> None:
    await asyncio.to_thread(self._record_sync, event)

  def _get_total_sync(
    self,
    user_id: str,
    *,
    since: float | None,
    until: float | None,
    billing_mode: Literal["byok", "metered"] | None,
    model: str | None,
  ) -> UsageTotal:
    self._assert_open()
    where = ["user_id = ?"]
    params: list[object] = [user_id]
    if since is not None:
      where.append("timestamp >= ?")
      params.append(since)
    if until is not None:
      where.append("timestamp <= ?")
      params.append(until)
    if billing_mode is not None:
      where.append("billing_mode = ?")
      params.append(billing_mode)
    if model is not None:
      where.append("model = ?")
      params.append(model)

    query = f"""
      SELECT
        COALESCE(SUM(input_tokens), 0) AS input_tokens,
        COALESCE(SUM(output_tokens), 0) AS output_tokens,
        COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
        COALESCE(SUM(cache_creation_tokens), 0) AS cache_creation_tokens,
        COALESCE(SUM(cost_usd), 0.0) AS cost_usd,
        COUNT(*) AS event_count
      FROM usage_events
      WHERE {" AND ".join(where)}
    """
    with self._connect() as conn:
      self._ensure_schema(conn)
      row = conn.execute(query, params).fetchone()
    return UsageTotal(
      user_id=user_id,
      input_tokens=int(row["input_tokens"] or 0),
      output_tokens=int(row["output_tokens"] or 0),
      cache_read_tokens=int(row["cache_read_tokens"] or 0),
      cache_creation_tokens=int(row["cache_creation_tokens"] or 0),
      cost_usd=float(row["cost_usd"] or 0.0),
      event_count=int(row["event_count"] or 0),
      since=since,
      until=until,
    )

  async def get_total(
    self,
    user_id: str,
    *,
    since: float | None = None,
    until: float | None = None,
    billing_mode: Literal["byok", "metered"] | None = None,
    model: str | None = None,
  ) -> UsageTotal:
    return await asyncio.to_thread(
      self._get_total_sync,
      user_id,
      since=since,
      until=until,
      billing_mode=billing_mode,
      model=model,
    )

  def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    self._init_conn.close()


def write_dlq(event: UsageEvent, spool_path: Path) -> None:
  """Append event as JSON line to spool file. Called when record() fails."""
  spool_path = Path(spool_path).expanduser()
  spool_path.parent.mkdir(parents=True, exist_ok=True)
  with spool_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")


async def replay_dlq(ledger: UsageLedger, spool_path: Path) -> dict:
  """Read spool, attempt to write each event to ledger, return stats."""
  spool_path = Path(spool_path).expanduser()
  stats = {"total": 0, "replayed": 0, "failed": 0, "invalid": 0}
  if not spool_path.exists():
    return stats

  raw_lines = spool_path.read_text(encoding="utf-8").splitlines()
  keep_lines: list[str] = []
  for line in raw_lines:
    if not line.strip():
      continue
    stats["total"] += 1
    try:
      payload = json.loads(line)
      event = UsageEvent(**payload)
    except Exception:
      stats["invalid"] += 1
      keep_lines.append(line)
      continue

    try:
      await ledger.record(event)
    except Exception:
      stats["failed"] += 1
      keep_lines.append(line)
      continue

    stats["replayed"] += 1

  if keep_lines:
    spool_path.write_text("".join(f"{line}\n" for line in keep_lines), encoding="utf-8")
  else:
    spool_path.unlink(missing_ok=True)
  return stats


__all__ = [
  "DEFAULT_USAGE_DLQ_PATH",
  "SqliteUsageLedger",
  "SessionUsageSummary",
  "UsageEvent",
  "UsageLedger",
  "UsageTotal",
  "normalize_identity",
  "replay_dlq",
  "write_dlq",
]
