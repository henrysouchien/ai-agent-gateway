import asyncio
from dataclasses import asdict
import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from agent_workflow_contracts import CapabilityBind

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.multi_user.billing import (  # noqa: E402
  SessionUsageSummary,
  SqliteUsageLedger,
  UsageEvent,
  _UsageAggregator,
  normalize_identity,
  replay_dlq,
  write_dlq,
)


def _run(coro):
  return asyncio.run(coro)


def _bind_receipt(*, provider: str, model: str) -> dict[str, str]:
  return CapabilityBind(
    schema_version="1.0",
    capability_id="session.driver",
    model_key=f"test.{provider}.{model}",
    provider=provider,
    upstream_model=model,
    adapter=f"test.{provider}",
    protocol_profile="test.reasoning",
    route="test.in_process",
    effort="none",
    credential_principal="service",
    credential_ref=f"test-service:{provider}",
    run_mode="interactive",
    registry_revision="test-v1",
    policy_revision="test-v1",
    selection_source="capability_default",
  ).receipt()


def _event(
  *,
  user_id: str = "alice",
  request_id: str = "req-1",
  timestamp: float = 100.0,
  model: str = "claude-sonnet-4-6",
  billing_mode: str = "metered",
  channel: str | None = "web",
  parent_turn_id: str | None = None,
  provider: str = "anthropic",
  provider_reported_model: str | None = None,
) -> UsageEvent:
  return UsageEvent(
    user_id=user_id,
    session_id="sess-1",
    request_id=request_id,
    parent_turn_id=parent_turn_id,
    timestamp=timestamp,
    model=model,
    input_tokens=100,
    output_tokens=50,
    cache_read_tokens=10,
    cache_creation_tokens=5,
    cost_usd=0.125,
    rate_table_version="2026-04-08",
    billing_mode=billing_mode,  # type: ignore[arg-type]
    channel=channel,
    provider=provider,
    capability_bind=_bind_receipt(provider=provider, model=model),
    provider_reported_model=provider_reported_model,
  )


def test_sqlite_usage_ledger_record_inserts_all_fields(tmp_path: Path) -> None:
  db_path = tmp_path / "usage.db"
  ledger = SqliteUsageLedger(db_path)

  _run(ledger.record(_event(parent_turn_id="tool-1")))

  with sqlite3.connect(db_path) as conn:
    row = conn.execute(
      """
      SELECT user_id, session_id, request_id, parent_turn_id, timestamp, model,
             provider,
             input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens,
             cost_usd, rate_table_version, billing_mode, channel
      FROM usage_events
      """
    ).fetchone()

  assert row == (
    "alice",
    "sess-1",
    "req-1",
    "tool-1",
    100.0,
    "claude-sonnet-4-6",
    "anthropic",
    100,
    50,
    10,
    5,
    0.125,
    "2026-04-08",
    "metered",
    "web",
  )


def test_sqlite_usage_ledger_get_total_filters_and_sums(tmp_path: Path) -> None:
  ledger = SqliteUsageLedger(tmp_path / "usage.db")
  events = [
    _event(user_id="alice", request_id="req-1", timestamp=100.0, billing_mode="metered", model="claude-sonnet-4-6"),
    UsageEvent(
      user_id="alice",
      session_id="sess-2",
      request_id="req-2",
      parent_turn_id=None,
      timestamp=200.0,
      model="claude-opus-4-6",
      input_tokens=40,
      output_tokens=20,
      cache_read_tokens=4,
      cache_creation_tokens=2,
      cost_usd=0.05,
      rate_table_version="2026-04-08",
      billing_mode="byok",
      channel="cli",
      provider="codex",
      capability_bind=_bind_receipt(provider="codex", model="claude-opus-4-6"),
      provider_reported_model=None,
    ),
    UsageEvent(
      user_id="bob",
      session_id="sess-3",
      request_id="req-3",
      parent_turn_id=None,
      timestamp=300.0,
      model="claude-sonnet-4-6",
      input_tokens=999,
      output_tokens=999,
      cache_read_tokens=0,
      cache_creation_tokens=0,
      cost_usd=9.99,
      rate_table_version="2026-04-08",
      billing_mode="metered",
      channel="web",
      provider="anthropic",
      capability_bind=_bind_receipt(
        provider="anthropic", model="claude-sonnet-4-6"
      ),
      provider_reported_model=None,
    ),
  ]
  for event in events:
    _run(ledger.record(event))

  total = _run(ledger.get_total("alice"))
  assert total.input_tokens == 140
  assert total.output_tokens == 70
  assert total.cache_read_tokens == 14
  assert total.cache_creation_tokens == 7
  assert total.cost_usd == pytest.approx(0.175)
  assert total.event_count == 2

  metered = _run(ledger.get_total("alice", billing_mode="metered"))
  assert metered.input_tokens == 100
  assert metered.output_tokens == 50
  assert metered.event_count == 1

  ranged = _run(ledger.get_total("alice", since=150.0, until=250.0))
  assert ranged.input_tokens == 40
  assert ranged.output_tokens == 20
  assert ranged.event_count == 1

  model_filtered = _run(ledger.get_total("alice", model="claude-opus-4-6"))
  assert model_filtered.input_tokens == 40
  assert model_filtered.output_tokens == 20
  assert model_filtered.event_count == 1

  provider_filtered = _run(ledger.get_total("alice", provider="codex"))
  assert provider_filtered.input_tokens == 40
  assert provider_filtered.output_tokens == 20
  assert provider_filtered.event_count == 1


def test_sqlite_usage_ledger_get_total_returns_zeros_when_empty(tmp_path: Path) -> None:
  ledger = SqliteUsageLedger(tmp_path / "usage.db")

  total = _run(ledger.get_total("nobody"))

  assert total.user_id == "nobody"
  assert total.input_tokens == 0
  assert total.output_tokens == 0
  assert total.cache_read_tokens == 0
  assert total.cache_creation_tokens == 0
  assert total.cost_usd == 0.0
  assert total.event_count == 0


def test_sqlite_usage_ledger_handles_concurrent_writes(tmp_path: Path) -> None:
  ledger = SqliteUsageLedger(tmp_path / "usage.db")

  def _write(index: int) -> None:
    _run(
      ledger.record(
        UsageEvent(
          user_id="alice",
          session_id=f"sess-{index}",
          request_id=f"req-{index}",
          parent_turn_id=None,
          timestamp=float(index),
          model="claude-sonnet-4-6",
          input_tokens=10,
          output_tokens=5,
          cache_read_tokens=1,
          cache_creation_tokens=0,
          cost_usd=0.01,
          rate_table_version="2026-04-08",
          billing_mode="metered",
          channel="web",
          provider="anthropic",
          capability_bind=_bind_receipt(
            provider="anthropic", model="claude-sonnet-4-6"
          ),
          provider_reported_model=None,
        )
      )
    )

  with ThreadPoolExecutor(max_workers=10) as executor:
    list(executor.map(_write, range(10)))

  total = _run(ledger.get_total("alice"))
  assert total.input_tokens == 100
  assert total.output_tokens == 50
  assert total.event_count == 10


def test_dlq_write_and_replay_round_trip(tmp_path: Path) -> None:
  ledger = SqliteUsageLedger(tmp_path / "usage.db")
  spool_path = tmp_path / "usage_dlq.jsonl"
  event = _event()

  write_dlq(event, spool_path)

  assert spool_path.exists()
  assert spool_path.read_text(encoding="utf-8").count("\n") == 1

  stats = _run(replay_dlq(ledger, spool_path))

  assert stats == {"total": 1, "replayed": 1, "failed": 0, "invalid": 0}
  assert not spool_path.exists()
  total = _run(ledger.get_total("alice"))
  assert total.event_count == 1


def test_dlq_spool_survives_as_file_on_disk(tmp_path: Path) -> None:
  spool_path = tmp_path / "nested" / "usage_dlq.jsonl"

  write_dlq(_event(), spool_path)

  assert spool_path.exists()
  line = spool_path.read_text(encoding="utf-8").strip()
  assert '"request_id": "req-1"' in line


def test_dlq_replay_retains_unversioned_historical_payload(tmp_path: Path) -> None:
  ledger = SqliteUsageLedger(tmp_path / "usage.db")
  spool_path = tmp_path / "usage_dlq.jsonl"
  spool_path.write_text(json.dumps(asdict(_event())) + "\n", encoding="utf-8")

  stats = _run(replay_dlq(ledger, spool_path))

  assert stats == {"total": 1, "replayed": 0, "failed": 0, "invalid": 1}
  assert spool_path.exists()
  assert _run(ledger.get_total("alice")).event_count == 0


def test_sqlite_usage_ledger_schema_migration_is_idempotent(tmp_path: Path) -> None:
  db_path = tmp_path / "usage.db"
  with sqlite3.connect(db_path) as conn:
    conn.execute(
      """
      CREATE TABLE usage_events (
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
    conn.execute(
      """
      INSERT INTO usage_events (
        user_id, session_id, request_id, parent_turn_id, timestamp, model,
        input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens,
        cost_usd, rate_table_version, billing_mode, channel
      ) VALUES ('legacy', 'sess-old', 'req-old', NULL, 1.0, 'claude-sonnet-4-6', 1, 1, 0, 0, 0.01, 'v1', 'byok', 'cli')
      """
    )
    conn.commit()

  ledger_one = SqliteUsageLedger(db_path)
  ledger_one.close()

  ledger_two = SqliteUsageLedger(db_path)
  _run(ledger_two.record(_event()))

  with sqlite3.connect(db_path) as conn:
    count = conn.execute("SELECT COUNT(*) FROM usage_events").fetchone()[0]
    columns = {row[1] for row in conn.execute("PRAGMA table_info(usage_events)").fetchall()}
    legacy_provider = conn.execute("SELECT provider FROM usage_events WHERE request_id = 'req-old'").fetchone()[0]
    new_provider = conn.execute("SELECT provider FROM usage_events WHERE request_id = 'req-1'").fetchone()[0]
    journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

  assert count == 2
  assert "provider" in columns
  assert {
    "capability_bind_json",
    "provider_reported_model",
  } <= columns
  assert legacy_provider is None
  assert new_provider == "anthropic"
  assert str(journal_mode).lower() == "wal"


def test_usage_event_rejects_unversioned_old_dlq_payload() -> None:
  payload = _event().__dict__.copy()
  payload.pop("event_id", None)
  payload.pop("provider", None)
  for field_name in (
    "capability_bind",
    "provider_reported_model",
  ):
    payload.pop(field_name, None)

  with pytest.raises(TypeError):
    UsageEvent(**payload)


def test_normalize_identity_requires_explicit_billing_identity() -> None:
  with pytest.raises(ValueError, match="user_id"):
    normalize_identity(None, "v1", "byok", "  ")
  with pytest.raises(ValueError, match="reserved"):
    normalize_identity("_default", "v1", "byok", "  ")
  with pytest.raises(ValueError, match="rate_table_version"):
    normalize_identity("alice", None, "byok", "  ")
  with pytest.raises(ValueError, match="billing_mode"):
    normalize_identity("alice", "v1", "other", "cli")
  assert normalize_identity("alice", "v1", " metered ", " web ") == ("alice", "v1", "metered", "web")


def test_usage_aggregator_accumulates_and_snapshots() -> None:
  async def case() -> None:
    aggregator = _UsageAggregator(user_id="alice", session_id="sess", request_id="req", channel="web", started_at=1.0)
    await aggregator.record(_event(
      request_id="req",
      timestamp=2.0,
      provider_reported_model="claude-sonnet-4-6-20260801",
    ))
    await aggregator.record(
      UsageEvent(
        user_id="alice",
        session_id="sub:sess",
        request_id="req",
        parent_turn_id="tool-1",
        timestamp=3.0,
        model="gpt-5.5",
        provider="codex",
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=1,
        cache_creation_tokens=2,
        cost_usd=0.025,
        rate_table_version="2026-04-08",
        billing_mode="metered",
        channel="web",
        capability_bind=_bind_receipt(provider="codex", model="gpt-5.5"),
        provider_reported_model=None,
      )
    )
    summary = await aggregator.snapshot(ended_at=4.0, drain_complete=False, in_flight_task_count=1)
    assert summary.input_tokens == 110
    assert summary.output_tokens == 55
    assert summary.cache_read_tokens == 11
    assert summary.cache_creation_tokens == 7
    assert summary.cost == pytest.approx(0.15)
    assert summary.turns == 2
    assert summary.channel == "web"
    assert summary.started_at == 1.0
    assert summary.ended_at == 4.0
    assert summary.drain_complete is False
    assert summary.in_flight_task_count == 1
    assert summary.compaction_count == 0
    assert summary.model == "gpt-5.5"
    assert summary.provider == "codex"
    assert summary.capability_bind == _bind_receipt(
      provider="codex", model="gpt-5.5"
    )
    assert summary.provider_reported_model is None

  _run(case())


def test_usage_aggregator_compaction_count_is_exact_and_closed_fail_closed() -> None:
  async def case() -> None:
    aggregator = _UsageAggregator(
      user_id="alice",
      session_id="sess",
      request_id="req",
      channel="web",
    )

    assert aggregator.record_compaction_nowait() is True
    assert aggregator.record_compaction_nowait() is True
    await aggregator.close()
    assert aggregator.record_compaction_nowait() is False

    summary = await aggregator.snapshot()
    assert summary.compaction_count == 2

  _run(case())


def test_usage_summary_identity_requires_a_provider_observation_and_receipt() -> None:
  fields = {
    "user_id": "alice",
    "session_id": "sess",
    "request_id": "req",
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_tokens": 0,
    "cache_creation_tokens": 0,
    "cost": 0.0,
    "turns": 0,
    "channel": "web",
    "started_at": 1.0,
    "ended_at": 2.0,
  }

  with pytest.raises(ValueError, match="require provider observations"):
    SessionUsageSummary(
      **fields,
      capability_bind=_bind_receipt(
        provider="anthropic", model="claude-sonnet-4-6"
      ),
    )
  with pytest.raises(ValueError, match="require a capability bind"):
    SessionUsageSummary(
      **fields,
      provider_reported_model="claude-sonnet-4-6-20260801",
    )


def test_usage_aggregator_concurrent_record_calls() -> None:
  async def case() -> None:
    aggregator = _UsageAggregator(user_id="alice", session_id="sess", request_id="req", channel=None)
    await asyncio.gather(*(aggregator.record(_event(request_id=f"req-{idx}", channel=None)) for idx in range(25)))
    summary = await aggregator.snapshot()
    assert summary.turns == 25
    assert summary.input_tokens == 2500
    assert summary.output_tokens == 1250

  _run(case())
