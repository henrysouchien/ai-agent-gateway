from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias

from model_authority.bind import CapabilityBind

BillingMode: TypeAlias = Literal["byok", "metered"]
UsageState: TypeAlias = Literal[
  "succeeded", "failed_billable", "failed_unbilled", "canceled"
]
UsageValue: TypeAlias = (
  int
  | str
  | dict[str, int]
  | dict[str, str]
  | None
)

UsageTotals: TypeAlias = dict[str, UsageValue]


DEFAULT_USAGE_DLQ_PATH = Path("~/.gateway/usage_dlq.jsonl").expanduser()


def default_usage_dlq_path(user_id: str) -> Path:
  """Return the canonical owner-scoped usage DLQ spool path for a user."""
  clean_id = str(user_id or "").strip()
  if not clean_id:
    raise ValueError("user_id must not be blank")
  user_data = os.getenv("USER_DATA_DIR", "").strip()
  if user_data:
    base = Path(user_data).expanduser()
  else:
    base = Path("~/.gateway").expanduser()
  return base / "users" / clean_id / "usage_dlq.jsonl"

@dataclass(frozen=True)
class UsageEvent:
  """Secret-free usage observation with admitted and reported identity split.

  ``capability_bind`` is the authoritative admitted identity receipt and the
  billed identity.  A compatibility ``model`` or ``provider`` projection that
  differs from it is billed as the bind says, and the projected value is kept
  in ``projection_divergence``; neither is replaced with
  ``provider_reported_model``.
  """

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
  billing_mode: BillingMode
  channel: str | None
  provider: str
  capability_bind: dict[str, str]
  provider_reported_model: str | None
  reasoning_tokens_observed: int | None = None
  provider_reported_cost_usd: str | None = None
  separately_billed_tool_cost_usd: str = "0"
  provider_units: str | int | float | None = None
  provider_unit_deltas: dict[str, int] | None = None
  is_batch: bool = False
  product_id: str | None = None
  cost_known: bool = True
  event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
  projection_divergence: dict[str, str] | None = None
  selection_provenance: dict[str, Any] | None = None
  stage_id: str | None = None
  # Provider diagnostics for the call (providers/base.py StreamEvent); not billed,
  # not copied into the commercial usage payload.
  provider_response_id: str | None = None
  provider_request_id: str | None = None
  service_tier: str | None = None
  thinking_chars: int = 0

  def __post_init__(self) -> None:
    bind = CapabilityBind.from_json(self.capability_bind)
    divergence = dict(self.projection_divergence or {})
    if self.model != bind.upstream_model:
      divergence["model"] = self.model
    if self.provider != bind.provider:
      divergence["provider"] = self.provider
    object.__setattr__(self, "model", bind.upstream_model)
    object.__setattr__(self, "provider", bind.provider)
    object.__setattr__(self, "projection_divergence", divergence or None)
    object.__setattr__(self, "capability_bind", bind.to_json())
    if self.provider_reported_model is not None:
      if not isinstance(self.provider_reported_model, str):
        raise ValueError("provider_reported_model must be a string when present")
      reported = self.provider_reported_model.strip()
      if not reported:
        raise ValueError("provider_reported_model must be non-empty when present")
      object.__setattr__(self, "provider_reported_model", reported)


@dataclass(frozen=True)
class ProviderModelUsage:
  """Tokens and cost for one bucket; the latest event supplies its pricing version."""

  provider: str | None
  model: str | None
  input_tokens: int
  output_tokens: int
  cache_read_tokens: int
  cache_creation_tokens: int
  cost: float
  turns: int
  usage_event_count: int
  usage_event_ids: tuple[str, ...]
  capability_bind: dict[str, str] | None
  provider_reported_model: str | None
  cost_known: bool
  rate_table_version: str | None

@dataclass(frozen=True)
class SessionUsageSummary:
  """Authoritative per-session billing total.

  `cost` is the billing total for the completed session. Stream-level
  `estimated_cost` values remain live per-turn estimates and can differ when
  background tasks complete before or after the main stream completes.
  Wall time includes request drain/finalizers. TTFT measures the first
  non-empty assistant text from stream start, not provider or tool activity.
  ``provider_model_usage`` splits that total by provider, model and bind;
  ``model`` and ``provider`` remain the last observation.
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
  product_id: str | None = None
  model: str | None = None
  provider: str | None = None
  rate_table_version: str | None = None
  billing_mode: BillingMode | None = None
  context_surfaces: list[dict[str, Any]] = field(default_factory=list)
  usage_event_count: int = 0
  usage_event_ids: tuple[str, ...] = ()
  compaction_count: int = 0
  capability_bind: dict[str, str] | None = None
  provider_reported_model: str | None = None
  stream_started_at: float | None = None
  first_text_at: float | None = None
  projection_divergence: dict[str, str] | None = None
  cost_known: bool = True
  unit_kind: str = "none"
  unit_id: str | None = None
  stage_id: str | None = None
  provider_model_usage: tuple[ProviderModelUsage, ...] = ()

  @property
  def wall_time_s(self) -> float:
    return self.ended_at - self.started_at

  @property
  def ttft_s(self) -> float | None:
    if self.first_text_at is None or self.stream_started_at is None:
      return None
    return self.first_text_at - self.stream_started_at

  def __post_init__(self) -> None:
    if self.usage_event_count < 0:
      raise ValueError("usage_event_count cannot be negative")
    if self.capability_bind is not None:
      if self.usage_event_count == 0:
        raise ValueError(
          "usage summaries with a capability bind require provider observations"
        )
      # The bind is the billed identity; a differing projection is kept, not billed.
      bind = CapabilityBind.from_json(self.capability_bind)
      divergence = dict(self.projection_divergence or {})
      if self.model is not None and self.model != bind.upstream_model:
        divergence["model"] = self.model
      if self.provider is not None and self.provider != bind.provider:
        divergence["provider"] = self.provider
      object.__setattr__(self, "projection_divergence", divergence or None)
      object.__setattr__(self, "capability_bind", bind.to_json())
      object.__setattr__(self, "model", bind.upstream_model)
      object.__setattr__(self, "provider", bind.provider)
    elif (
      self.usage_event_count
      or self.model is not None
      or self.provider is not None
      or self.provider_reported_model is not None
    ):
      raise ValueError(
        "usage summaries with provider observations require a capability bind"
      )
    if self.provider_reported_model is not None:
      if not isinstance(self.provider_reported_model, str):
        raise ValueError("provider_reported_model must be a string when present")
      reported = self.provider_reported_model.strip()
      if not reported:
        raise ValueError("provider_reported_model must be non-empty when present")
      object.__setattr__(self, "provider_reported_model", reported)


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
) -> tuple[str, str, BillingMode, str | None]:
  """Validate and normalize the usage identity attached to billing records."""
  normalized_user_id = str(user_id or "").strip()
  if not normalized_user_id:
    raise ValueError("user_id is required for usage identity")

  normalized_rate_table_version = str(rate_table_version or "").strip()
  if not normalized_rate_table_version:
    raise ValueError("rate_table_version is required for usage identity")

  normalized_billing_mode_raw = str(billing_mode or "").strip().lower()
  if (
    normalized_billing_mode_raw != "byok"
    and normalized_billing_mode_raw != "metered"
  ):
    raise ValueError("billing_mode must be 'byok' or 'metered'")
  normalized_channel = channel.strip() if isinstance(channel, str) and channel.strip() else None
  return normalized_user_id, normalized_rate_table_version, normalized_billing_mode_raw, normalized_channel


def _capability_bind_key(capability_bind: dict[str, str] | None) -> str:
  if not capability_bind:
    return ""
  return json.dumps(capability_bind, sort_keys=True, default=str)


def _provider_model_key(
  provider: str | None,
  model: str | None,
  capability_bind: dict[str, str] | None,
) -> tuple[str | None, str | None, str]:
  return provider, model, _capability_bind_key(capability_bind)


def _merge_provider_model_usage(
  parts: list[SessionUsageSummary],
) -> tuple[ProviderModelUsage, ...]:
  """Merge hop buckets in insertion order. Later observation metadata wins."""
  merged: dict[tuple[str | None, str | None, str], ProviderModelUsage] = {}
  for part in parts:
    for entry in part.provider_model_usage:
      key = _provider_model_key(entry.provider, entry.model, entry.capability_bind)
      existing = merged.get(key)
      if existing is None:
        merged[key] = entry
        continue
      merged[key] = replace(
        existing,
        input_tokens=existing.input_tokens + entry.input_tokens,
        output_tokens=existing.output_tokens + entry.output_tokens,
        cache_read_tokens=existing.cache_read_tokens + entry.cache_read_tokens,
        cache_creation_tokens=existing.cache_creation_tokens + entry.cache_creation_tokens,
        cost=existing.cost + entry.cost,
        turns=existing.turns + entry.turns,
        usage_event_count=existing.usage_event_count + entry.usage_event_count,
        usage_event_ids=existing.usage_event_ids + entry.usage_event_ids,
        capability_bind=(
          dict(entry.capability_bind) if entry.capability_bind else None
        ),
        provider_reported_model=entry.provider_reported_model,
        rate_table_version=entry.rate_table_version,
        cost_known=existing.cost_known and entry.cost_known,
      )
  return tuple(merged.values())


class _UsageAggregator:
  """Accumulates UsageEvent objects within one asyncio event loop.

  This is an internal runner coordination primitive. It is protected by an
  asyncio.Lock and is not cross-loop or thread safe.

  A delegated child run keeps its own aggregator, built with `parent=` the
  delegating run's: its `snapshot()` is the child's own billing row, the unit
  (`declare_unit`) is owned by the root of the tree, and the root's
  `fleet_snapshot()` is the whole dispatch tree's total. `snapshot()` of any
  aggregator counts only the runs that recorded into it directly. Both also
  keep ``provider_model_usage``, one bucket per provider, model and capability
  bind, so a fallback hop does not erase earlier spend.
  """

  def __init__(
    self,
    *,
    user_id: str,
    session_id: str,
    request_id: str,
    channel: str | None,
    rate_table_version: str | None = None,
    billing_mode: BillingMode | None = None,
    started_at: float | None = None,
    parent: "_UsageAggregator | None" = None,
    stage_id: str | None = None,
  ) -> None:
    self._lock = asyncio.Lock()
    self._user_id = user_id
    self._session_id = session_id
    self._request_id = request_id
    self._channel = channel
    self._rate_table_version = rate_table_version
    self._billing_mode: BillingMode | None = billing_mode
    self._started_at = started_at if started_at is not None else time.time()
    self._input_tokens = 0
    self._output_tokens = 0
    self._cache_read_tokens = 0
    self._cache_creation_tokens = 0
    self._cost = 0.0
    self._turns = 0
    self._last_model: str | None = None
    self._last_provider: str | None = None
    self._last_capability_bind: dict[str, str] | None = None
    self._last_provider_reported_model: str | None = None
    self._event_ids: list[str] = []
    self._provider_model_buckets: dict[
      tuple[str | None, str | None, str], ProviderModelUsage
    ] = {}
    self._compaction_count = 0
    self._closed = False
    self._unit_kind = "none"
    self._unit_id = request_id
    self._cost_known = True
    self._parent = parent
    self._children: list[_UsageAggregator] = []
    self._stage_id = stage_id if isinstance(stage_id, str) and stage_id else None
    if parent is not None:
      parent._children.append(self)

  def bind_cost_stage_id(self, stage_id: str) -> None:
    """Key this run's cost row by an attempt id already minted at spawn.

    Root and captured-pipeline aggregators leave this unset. A resumed child
    is a new attempt, so it binds a new id and lands a separate row.
    """
    if not isinstance(stage_id, str) or not stage_id or stage_id != stage_id.strip():
      return
    self._stage_id = stage_id

  def _accumulate_provider_model(self, event: UsageEvent) -> None:
    bind = dict(event.capability_bind) if event.capability_bind else None
    key = _provider_model_key(event.provider, event.model, bind)
    known = bool(getattr(event, "cost_known", True))
    existing = self._provider_model_buckets.get(key)
    input_tokens = int(event.input_tokens or 0)
    output_tokens = int(event.output_tokens or 0)
    cache_read_tokens = int(event.cache_read_tokens or 0)
    cache_creation_tokens = int(event.cache_creation_tokens or 0)
    cost = float(event.cost_usd or 0.0)
    if existing is None:
      self._provider_model_buckets[key] = ProviderModelUsage(
        provider=event.provider,
        model=event.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cost=cost,
        turns=1,
        usage_event_count=1,
        usage_event_ids=(event.event_id,),
        capability_bind=bind,
        provider_reported_model=event.provider_reported_model,
        rate_table_version=event.rate_table_version,
        cost_known=known,
      )
      return
    self._provider_model_buckets[key] = replace(
      existing,
      input_tokens=existing.input_tokens + input_tokens,
      output_tokens=existing.output_tokens + output_tokens,
      cache_read_tokens=existing.cache_read_tokens + cache_read_tokens,
      cache_creation_tokens=existing.cache_creation_tokens + cache_creation_tokens,
      cost=existing.cost + cost,
      turns=existing.turns + 1,
      usage_event_count=existing.usage_event_count + 1,
      usage_event_ids=existing.usage_event_ids + (event.event_id,),
      capability_bind=bind,
      provider_reported_model=event.provider_reported_model,
      rate_table_version=event.rate_table_version,
      cost_known=existing.cost_known and known,
    )

  async def record(self, event: UsageEvent) -> bool:
    if self._stage_id is not None and not getattr(event, "stage_id", None):
      object.__setattr__(event, "stage_id", self._stage_id)
    async with self._lock:
      if self._closed:
        return False
      self._input_tokens += int(event.input_tokens or 0)
      self._output_tokens += int(event.output_tokens or 0)
      self._cache_read_tokens += int(event.cache_read_tokens or 0)
      self._cache_creation_tokens += int(event.cache_creation_tokens or 0)
      self._cost += float(event.cost_usd or 0.0)
      self._turns += 1
      self._last_model = event.model
      self._last_provider = event.provider
      self._last_capability_bind = dict(event.capability_bind)
      self._last_provider_reported_model = event.provider_reported_model
      self._event_ids.append(event.event_id)
      if not getattr(event, "cost_known", True):
        self._cost_known = False
      self._accumulate_provider_model(event)
      return True

  def record_compaction_nowait(self) -> bool:
    """Record one live compaction on the aggregator's owning event loop."""
    if self._closed:
      return False
    self._compaction_count += 1
    return True

  async def declare_unit(self, unit_kind: str, unit_id: str | None = None) -> tuple[str, str]:
    """Record the agent's declared outcome id (`A1` … `F2`) for this unit of work.

    `unit_kind` is carried as declared; an undeclared unit stays `none`.
    `unit_id` defaults to the request id, which every delegated child shares.
    A child's declaration is the tree's: the root records it.
    Returns the `(unit_kind, unit_id)` now recorded.
    """
    if self._parent is not None:
      return await self._parent.declare_unit(unit_kind, unit_id)
    async with self._lock:
      self._unit_kind = str(unit_kind)
      if unit_id is not None:
        self._unit_id = str(unit_id)
      return self._unit_kind, self._unit_id

  async def _unit(self) -> tuple[str, str]:
    if self._parent is not None:
      return await self._parent._unit()
    async with self._lock:
      return self._unit_kind, self._unit_id

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
    context_surfaces: list[dict[str, Any]] | None = None,
    stream_started_at: float | None = None,
    first_text_at: float | None = None,
  ) -> SessionUsageSummary:
    unit_kind, unit_id = await self._unit()
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
        stream_started_at=stream_started_at,
        first_text_at=first_text_at,
        drain_complete=drain_complete,
        in_flight_task_count=in_flight_task_count,
        compaction_count=self._compaction_count,
        model=self._last_model,
        provider=self._last_provider,
        capability_bind=self._last_capability_bind,
        provider_reported_model=self._last_provider_reported_model,
        rate_table_version=self._rate_table_version,
        billing_mode=self._billing_mode,
        context_surfaces=[
          dict(surface)
          for surface in (context_surfaces or [])
          if isinstance(surface, dict)
        ],
        usage_event_count=len(self._event_ids),
        usage_event_ids=tuple(self._event_ids),
        cost_known=self._cost_known,
        unit_kind=unit_kind,
        unit_id=unit_id,
        stage_id=self._stage_id,
        provider_model_usage=tuple(self._provider_model_buckets.values()),
      )

  async def fleet_snapshot(
    self,
    *,
    ended_at: float | None = None,
    drain_complete: bool = True,
    in_flight_task_count: int = 0,
    context_surfaces: list[dict[str, Any]] | None = None,
    stream_started_at: float | None = None,
    first_text_at: float | None = None,
  ) -> SessionUsageSummary:
    """This aggregator's own usage plus every descendant's, under this identity.

    The total the commercial reconciliation and the session recap read; the
    per-session billing rows read `snapshot()`.
    """
    kwargs: dict[str, Any] = {
      "ended_at": ended_at if ended_at is not None else time.time(),
      "drain_complete": drain_complete,
      "in_flight_task_count": in_flight_task_count,
      "context_surfaces": context_surfaces,
      "stream_started_at": stream_started_at,
      "first_text_at": first_text_at,
    }
    own = await self.snapshot(**kwargs)
    descendants = [
      await child.fleet_snapshot(**kwargs) for child in tuple(self._children)
    ]
    if not descendants:
      return own
    parts = [own, *descendants]
    observed = next((part for part in parts if part.usage_event_count), own)
    return replace(
      own,
      input_tokens=sum(part.input_tokens for part in parts),
      output_tokens=sum(part.output_tokens for part in parts),
      cache_read_tokens=sum(part.cache_read_tokens for part in parts),
      cache_creation_tokens=sum(part.cache_creation_tokens for part in parts),
      cost=sum(part.cost for part in parts),
      turns=sum(part.turns for part in parts),
      compaction_count=sum(part.compaction_count for part in parts),
      usage_event_count=sum(part.usage_event_count for part in parts),
      usage_event_ids=tuple(
        event_id for part in parts for event_id in part.usage_event_ids
      ),
      cost_known=all(part.cost_known for part in parts),
      model=observed.model,
      provider=observed.provider,
      capability_bind=observed.capability_bind,
      provider_reported_model=observed.provider_reported_model,
      projection_divergence=observed.projection_divergence,
      provider_model_usage=_merge_provider_model_usage(parts),
    )


class UsageLedger(Protocol):
  async def record(self, event: UsageEvent) -> None: ...

  async def get_total(
    self,
    user_id: str,
    *,
    since: float | None = None,
    until: float | None = None,
    billing_mode: BillingMode | None = None,
    model: str | None = None,
    provider: str | None = None,
  ) -> UsageTotal: ...


class SqliteUsageLedger:
  """Reference implementation backed by SQLite with WAL mode."""

  def __init__(self, db_path: str | Path, *, dlq_path: Path | str | None = None):
    self._db_path = Path(db_path)
    self._db_path.parent.mkdir(parents=True, exist_ok=True)
    self._dlq_path = Path(dlq_path).expanduser() if dlq_path is not None else None
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
        provider TEXT,
        capability_bind_json TEXT,
        provider_reported_model TEXT,
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
    columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(usage_events)").fetchall()}
    if "provider" not in columns:
      conn.execute("ALTER TABLE usage_events ADD COLUMN provider TEXT")
    for column in ("capability_bind_json", "provider_reported_model"):
      if column not in columns:
        conn.execute(f"ALTER TABLE usage_events ADD COLUMN {column} TEXT")
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
          provider,
          capability_bind_json,
          provider_reported_model,
          input_tokens,
          output_tokens,
          cache_read_tokens,
          cache_creation_tokens,
          cost_usd,
          rate_table_version,
          billing_mode,
          channel
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
          event.user_id,
          event.session_id,
          event.request_id,
          event.parent_turn_id,
          event.timestamp,
          event.model,
          event.provider,
          (
            json.dumps(
              event.capability_bind,
              sort_keys=True,
              separators=(",", ":"),
              ensure_ascii=False,
              allow_nan=False,
            )
            if event.capability_bind is not None else None
          ),
          event.provider_reported_model,
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
    billing_mode: BillingMode | None,
    model: str | None,
    provider: str | None,
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
    if provider is not None:
      where.append("provider = ?")
      params.append(provider)

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

  async def drain_dlq(self, user_id: str, *, spool_path: Path | str | None = None) -> dict[str, int]:
    """Drain spooled DLQ events for user_id into this ledger."""
    clean_id = str(user_id or "").strip()
    if not clean_id:
      return {"total": 0, "replayed": 0, "failed": 0, "invalid": 0}
    target_spool: Path | None
    if spool_path is not None:
      target_spool = Path(spool_path).expanduser()
    elif self._dlq_path is not None:
      target_spool = self._dlq_path
    else:
      target_spool = default_usage_dlq_path(clean_id)
    return await replay_dlq(self, target_spool, user_id=clean_id)

  async def get_total(
    self,
    user_id: str,
    *,
    since: float | None = None,
    until: float | None = None,
    billing_mode: BillingMode | None = None,
    model: str | None = None,
    provider: str | None = None,
    drain_dlq: bool = True,
  ) -> UsageTotal:
    if drain_dlq:
      await self.drain_dlq(user_id)
    return await asyncio.to_thread(
      self._get_total_sync,
      user_id,
      since=since,
      until=until,
      billing_mode=billing_mode,
      model=model,
      provider=provider,
    )
  def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    self._init_conn.close()


def write_dlq(event: UsageEvent, spool_path: Path | str | None = None) -> Path:
  """Append event as JSON line to spool file. Called when record() fails.

  When spool_path is omitted or equals DEFAULT_USAGE_DLQ_PATH, the event is
  partitioned into the owning user's DLQ spool (default_usage_dlq_path).
  """
  if spool_path is None or Path(spool_path).expanduser() == DEFAULT_USAGE_DLQ_PATH:
    resolved_path = default_usage_dlq_path(event.user_id)
  else:
    resolved_path = Path(spool_path).expanduser()
  resolved_path.parent.mkdir(parents=True, exist_ok=True)
  with resolved_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({
      "usage_event_schema_version": 2,
      "event": asdict(event),
    }, sort_keys=True) + "\n")
  return resolved_path


async def replay_dlq(
  ledger: UsageLedger,
  spool_path: Path | str | None = None,
  *,
  user_id: str | None = None,
) -> dict[str, int]:
  """Read spool, attempt to write each event to ledger, return stats.

  When user_id is provided, only events matching user_id are replayed onto
  ledger; events belonging to other owners remain in the spool and are not
  replayed into another user's totals.
  """
  target_user = str(user_id or "").strip() if user_id is not None else None
  if spool_path is None:
    if target_user is None:
      raise ValueError("user_id is required when spool_path is omitted")
    resolved_path = default_usage_dlq_path(target_user)
  else:
    resolved_path = Path(spool_path).expanduser()

  stats = {"total": 0, "replayed": 0, "failed": 0, "invalid": 0}
  if not resolved_path.exists():
    return stats

  raw_lines = resolved_path.read_text(encoding="utf-8").splitlines()
  keep_lines: list[str] = []
  for line in raw_lines:
    if not line.strip():
      continue
    try:
      payload = json.loads(line)
      if (
        not isinstance(payload, dict)
        or payload.get("usage_event_schema_version") != 2
        or not isinstance(payload.get("event"), dict)
      ):
        raise ValueError("usage DLQ envelope version is unsupported")
      event = UsageEvent(**payload["event"])
    except Exception:
      stats["total"] += 1
      stats["invalid"] += 1
      keep_lines.append(line)
      continue

    if target_user is not None and event.user_id != target_user:
      keep_lines.append(line)
      continue

    stats["total"] += 1
    try:
      await ledger.record(event)
    except Exception:
      stats["failed"] += 1
      keep_lines.append(line)
      continue

    stats["replayed"] += 1

  if keep_lines:
    resolved_path.write_text("".join(f"{line}\n" for line in keep_lines), encoding="utf-8")
  else:
    resolved_path.unlink(missing_ok=True)
  return stats

__all__ = [
  "DEFAULT_USAGE_DLQ_PATH",
  "default_usage_dlq_path",
  "SqliteUsageLedger",
  "SessionUsageSummary",
  "ProviderModelUsage",
  "UsageEvent",
  "UsageLedger",
  "UsageTotal",
  "normalize_identity",
  "replay_dlq",
  "write_dlq",
]
