import asyncio
from dataclasses import asdict
import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from model_authority.bind import CapabilityBind

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.multi_user.billing import (  # noqa: E402
  ProviderModelUsage,
  SessionUsageSummary,
  SqliteUsageLedger,
  UsageEvent,
  _UsageAggregator,
  default_usage_dlq_path,
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
  ).to_json()


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
def test_dlq_event_for_owner_a_replays_only_onto_a_ledger(tmp_path: Path) -> None:
  """Invariant: A usage event that fails to post is drained to the owning user's ledger.
  One user's spool is not replayed into another user's totals.
  """
  spool_path = tmp_path / "shared_spool.jsonl"
  event_alice = _event(user_id="alice", request_id="req-alice")
  event_bob = _event(user_id="bob", request_id="req-bob")

  write_dlq(event_alice, spool_path)
  write_dlq(event_bob, spool_path)
  assert spool_path.exists()
  assert spool_path.read_text(encoding="utf-8").count("\n") == 2

  ledger_alice = SqliteUsageLedger(tmp_path / "alice.db", dlq_path=spool_path)
  ledger_bob = SqliteUsageLedger(tmp_path / "bob.db", dlq_path=spool_path)

  # 1. Querying alice's total triggers the production drain call site in get_total -> drain_dlq
  total_alice = _run(ledger_alice.get_total("alice"))
  assert total_alice.event_count == 1
  assert total_alice.input_tokens == 100
  assert total_alice.cost_usd == pytest.approx(0.125)

  # Bob's events were not replayed into alice's ledger
  assert _run(ledger_alice.get_total("bob", drain_dlq=False)).event_count == 0

  # Bob's event is still preserved in the spool
  assert spool_path.exists()
  spool_content = spool_path.read_text(encoding="utf-8")
  assert '"user_id": "bob"' in spool_content
  assert '"user_id": "alice"' not in spool_content

  # 2. Querying bob's total triggers the drain for bob
  total_bob = _run(ledger_bob.get_total("bob"))
  assert total_bob.event_count == 1
  assert total_bob.input_tokens == 100
  assert total_bob.cost_usd == pytest.approx(0.125)

  # Alice's events were not replayed into bob's ledger
  assert _run(ledger_bob.get_total("alice", drain_dlq=False)).event_count == 0

  # Spool is now completely drained and unlinked
  assert not spool_path.exists()


def test_default_usage_dlq_path_partitions_by_user_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  data_dir = tmp_path / "user_data"
  monkeypatch.setenv("USER_DATA_DIR", str(data_dir))

  path_alice = default_usage_dlq_path("alice")
  path_bob = default_usage_dlq_path("bob")

  assert path_alice != path_bob
  assert path_alice == data_dir / "users" / "alice" / "usage_dlq.jsonl"
  assert path_bob == data_dir / "users" / "bob" / "usage_dlq.jsonl"

  event_alice = _event(user_id="alice", request_id="req-alice-def")
  event_bob = _event(user_id="bob", request_id="req-bob-def")

  # write_dlq without explicit spool partitions by default
  written_alice = write_dlq(event_alice)
  written_bob = write_dlq(event_bob)

  assert written_alice == path_alice
  assert written_bob == path_bob
  assert path_alice.exists()
  assert path_bob.exists()

  ledger = SqliteUsageLedger(tmp_path / "central.db")
  # get_total("alice") drains alice's default spool
  tot_alice = _run(ledger.get_total("alice"))
  assert tot_alice.event_count == 1
  assert not path_alice.exists()
  # bob's default spool is untouched
  assert path_bob.exists()
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
  assert normalize_identity("_default", "v1", "byok", "  ")[0] == "_default"
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


def test_usage_aggregator_snapshot_carries_declared_unit() -> None:
  async def case() -> None:
    aggregator = _UsageAggregator(user_id="alice", session_id="sess", request_id="req", channel="web")

    undeclared = await aggregator.snapshot()
    assert (undeclared.unit_kind, undeclared.unit_id) == ("none", "req")

    await aggregator.declare_unit("A5")
    declared = await aggregator.snapshot()
    assert (declared.unit_kind, declared.unit_id) == ("A5", "req")

    await aggregator.declare_unit("A1", "unit-7")
    redeclared = await aggregator.snapshot()
    assert (redeclared.unit_kind, redeclared.unit_id) == ("A1", "unit-7")

  _run(case())


def test_child_aggregator_keeps_its_own_usage_and_shares_the_roots_unit() -> None:
  async def case() -> None:
    root = _UsageAggregator(user_id="alice", session_id="sess", request_id="req", channel="web", started_at=1.0)
    child = _UsageAggregator(
      user_id="alice", session_id="sub0:sess", request_id="req", channel="web",
      started_at=2.0, parent=root,
    )
    grandchild = _UsageAggregator(
      user_id="alice", session_id="sub0:sub0:sess", request_id="req", channel="web",
      parent=child,
    )
    await root.record(_event(request_id="req"))
    await child.record(_event(request_id="req", model="gpt-5.5", provider="codex"))
    await grandchild.record(_event(request_id="req", model="gpt-5.5", provider="codex"))

    # A declaration made anywhere in the tree is the root's, and every row reads it.
    assert await grandchild.declare_unit("A5") == ("A5", "req")
    for aggregator in (root, child, grandchild):
      summary = await aggregator.snapshot()
      assert (summary.unit_kind, summary.unit_id) == ("A5", "req")

    root_row = await root.snapshot()
    child_row = await child.snapshot()
    assert root_row.session_id == "sess"
    assert (root_row.input_tokens, root_row.turns, root_row.model) == (100, 1, "claude-sonnet-4-6")
    assert child_row.session_id == "sub0:sess"
    assert child_row.started_at == 2.0
    assert (child_row.input_tokens, child_row.turns, child_row.model) == (100, 1, "gpt-5.5")
    assert child_row.provider == "codex"

  _run(case())


def test_fleet_snapshot_sums_the_dispatch_tree_under_the_roots_identity() -> None:
  async def case() -> None:
    root = _UsageAggregator(user_id="alice", session_id="sess", request_id="req", channel="web")
    child = _UsageAggregator(
      user_id="alice", session_id="sub0:sess", request_id="req", channel="web", parent=root,
    )
    solo = _UsageAggregator(user_id="alice", session_id="solo", request_id="req", channel="web")
    root_event = _event(request_id="req")
    child_event = _event(request_id="req", model="gpt-5.5", provider="codex")
    await root.record(root_event)
    await child.record(child_event)
    root.record_compaction_nowait()
    child.record_compaction_nowait()
    await solo.record(_event(request_id="req"))

    fleet = await root.fleet_snapshot()
    assert fleet.session_id == "sess"
    assert (fleet.input_tokens, fleet.output_tokens, fleet.turns) == (200, 100, 2)
    assert fleet.cost == pytest.approx(0.25)
    assert fleet.compaction_count == 2
    assert fleet.usage_event_count == 2
    assert fleet.usage_event_ids == (root_event.event_id, child_event.event_id)
    assert (await root.snapshot()).input_tokens == 100

    # A run with no delegated child reports its own usage unchanged.
    solo_fleet = await solo.fleet_snapshot()
    assert (solo_fleet.session_id, solo_fleet.input_tokens, solo_fleet.turns) == ("solo", 100, 1)

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


def test_usage_event_bills_the_bind_and_keeps_a_differing_projection(tmp_path: Path) -> None:
  event = UsageEvent(
    **{
      **_event().__dict__,
      "model": "claude-opus-4-6",
      "provider": "codex",
      "capability_bind": _bind_receipt(provider="anthropic", model="claude-sonnet-4-6"),
      "projection_divergence": None,
    }
  )

  assert (event.model, event.provider) == ("claude-sonnet-4-6", "anthropic")
  assert event.projection_divergence == {"model": "claude-opus-4-6", "provider": "codex"}

  spool_path = tmp_path / "usage_dlq.jsonl"
  write_dlq(event, spool_path)
  ledger = SqliteUsageLedger(tmp_path / "usage.db")
  assert _run(replay_dlq(ledger, spool_path))["replayed"] == 1
  total = _run(ledger.get_total("alice", model="claude-sonnet-4-6"))
  assert total.event_count == 1


def test_usage_summary_bills_the_bind_and_keeps_a_differing_projection() -> None:
  summary = SessionUsageSummary(
    user_id="alice",
    session_id="sess",
    request_id="req",
    input_tokens=1,
    output_tokens=1,
    cache_read_tokens=0,
    cache_creation_tokens=0,
    cost=0.0,
    turns=1,
    channel="web",
    started_at=1.0,
    ended_at=2.0,
    usage_event_count=1,
    model="claude-opus-4-6",
    capability_bind=_bind_receipt(provider="anthropic", model="claude-sonnet-4-6"),
  )

  assert (summary.model, summary.provider) == ("claude-sonnet-4-6", "anthropic")
  assert summary.projection_divergence == {"model": "claude-opus-4-6"}


def test_usage_aggregator_concurrent_record_calls() -> None:
  async def case() -> None:
    aggregator = _UsageAggregator(user_id="alice", session_id="sess", request_id="req", channel=None)
    await asyncio.gather(*(aggregator.record(_event(request_id=f"req-{idx}", channel=None)) for idx in range(25)))
    summary = await aggregator.snapshot()
    assert summary.turns == 25
    assert summary.input_tokens == 2500
    assert summary.output_tokens == 1250

  _run(case())


def test_snapshot_keeps_per_provider_model_buckets_and_last_identity() -> None:
  async def case() -> None:
    aggregator = _UsageAggregator(
      user_id="alice", session_id="sess", request_id="req", channel="web", started_at=1.0,
    )
    first = _event(
      request_id="req",
      provider_reported_model="claude-sonnet-4-6-20260801",
    )
    second = UsageEvent(
      user_id="alice",
      session_id="sess-1",
      request_id="req",
      parent_turn_id=None,
      timestamp=3.0,
      model="claude-sonnet-4-6",
      provider="anthropic",
      input_tokens=20,
      output_tokens=10,
      cache_read_tokens=2,
      cache_creation_tokens=1,
      cost_usd=0.02,
      rate_table_version="2026-10-15",
      billing_mode="metered",
      channel="web",
      capability_bind=_bind_receipt(provider="anthropic", model="claude-sonnet-4-6"),
      provider_reported_model="claude-sonnet-4-6-later",
    )
    unknown = UsageEvent(
      user_id="alice",
      session_id="sess-1",
      request_id="req",
      parent_turn_id=None,
      timestamp=4.0,
      model="gpt-5.5",
      provider="codex",
      input_tokens=7,
      output_tokens=3,
      cache_read_tokens=1,
      cache_creation_tokens=0,
      cost_usd=0.01,
      rate_table_version="2026-10-16",
      billing_mode="metered",
      channel="web",
      capability_bind=_bind_receipt(provider="codex", model="gpt-5.5"),
      provider_reported_model="gpt-5.5-reported",
      cost_known=False,
    )
    await aggregator.record(first)
    await aggregator.record(second)
    await aggregator.record(unknown)
    summary = await aggregator.snapshot(ended_at=5.0)

    assert summary.input_tokens == 127
    assert summary.output_tokens == 63
    assert summary.cache_read_tokens == 13
    assert summary.cache_creation_tokens == 6
    assert summary.cost == pytest.approx(0.155)
    assert summary.turns == 3
    assert summary.model == "gpt-5.5"
    assert summary.provider == "codex"
    assert summary.cost_known is False
    assert summary.usage_event_ids == (first.event_id, second.event_id, unknown.event_id)

    assert len(summary.provider_model_usage) == 2
    sonnet, gpt = summary.provider_model_usage
    assert isinstance(sonnet, ProviderModelUsage)
    assert (sonnet.provider, sonnet.model) == ("anthropic", "claude-sonnet-4-6")
    assert (sonnet.input_tokens, sonnet.output_tokens) == (120, 60)
    assert (sonnet.cache_read_tokens, sonnet.cache_creation_tokens) == (12, 6)
    assert sonnet.cost == pytest.approx(0.145)
    assert sonnet.turns == 2
    assert sonnet.usage_event_count == 2
    assert sonnet.usage_event_ids == (first.event_id, second.event_id)
    assert sonnet.cost_known is True
    assert sonnet.capability_bind == _bind_receipt(
      provider="anthropic", model="claude-sonnet-4-6",
    )
    assert sonnet.provider_reported_model == "claude-sonnet-4-6-later"
    assert sonnet.rate_table_version == "2026-10-15"

    assert (gpt.provider, gpt.model) == ("codex", "gpt-5.5")
    assert (
      gpt.input_tokens, gpt.output_tokens, gpt.cache_read_tokens, gpt.cache_creation_tokens,
    ) == (7, 3, 1, 0)
    assert gpt.cost == pytest.approx(0.01)
    assert gpt.turns == 1
    assert gpt.usage_event_count == 1
    assert gpt.usage_event_ids == (unknown.event_id,)
    assert gpt.cost_known is False
    assert gpt.provider_reported_model == "gpt-5.5-reported"
    assert gpt.rate_table_version == "2026-10-16"

  _run(case())


def test_fleet_snapshot_merges_provider_model_buckets() -> None:
  async def case() -> None:
    root = _UsageAggregator(user_id="alice", session_id="sess", request_id="req", channel="web")
    child = _UsageAggregator(
      user_id="alice", session_id="sub0:sess", request_id="req", channel="web", parent=root,
    )
    root_event = _event(
      request_id="req",
      provider_reported_model="claude-sonnet-4-6-root",
    )
    child_same = UsageEvent(
      user_id="alice",
      session_id="sess-1",
      request_id="req",
      parent_turn_id=None,
      timestamp=3.0,
      model="claude-sonnet-4-6",
      provider="anthropic",
      input_tokens=4,
      output_tokens=2,
      cache_read_tokens=1,
      cache_creation_tokens=0,
      cost_usd=0.01,
      rate_table_version="2026-10-15",
      billing_mode="metered",
      channel="web",
      capability_bind=_bind_receipt(provider="anthropic", model="claude-sonnet-4-6"),
      provider_reported_model="claude-sonnet-4-6-child",
      cost_known=False,
    )
    child_other = _event(request_id="req", model="gpt-5.5", provider="codex")
    await root.record(root_event)
    await child.record(child_same)
    await child.record(child_other)

    fleet = await root.fleet_snapshot()
    assert (fleet.input_tokens, fleet.output_tokens, fleet.turns) == (204, 102, 3)
    assert fleet.cost == pytest.approx(0.26)
    assert fleet.model == "claude-sonnet-4-6"
    assert fleet.provider == "anthropic"
    assert fleet.cost_known is False
    assert fleet.usage_event_ids == (
      root_event.event_id, child_same.event_id, child_other.event_id,
    )

    assert len(fleet.provider_model_usage) == 2
    sonnet, gpt = fleet.provider_model_usage
    assert (sonnet.provider, sonnet.model) == ("anthropic", "claude-sonnet-4-6")
    assert (sonnet.input_tokens, sonnet.output_tokens) == (104, 52)
    assert (sonnet.cache_read_tokens, sonnet.cache_creation_tokens) == (11, 5)
    assert sonnet.cost == pytest.approx(0.135)
    assert sonnet.turns == 2
    assert sonnet.usage_event_count == 2
    assert sonnet.usage_event_ids == (root_event.event_id, child_same.event_id)
    assert sonnet.cost_known is False
    assert sonnet.provider_reported_model == "claude-sonnet-4-6-child"
    assert sonnet.rate_table_version == "2026-10-15"
    assert sonnet.capability_bind == _bind_receipt(
      provider="anthropic", model="claude-sonnet-4-6",
    )

    assert (gpt.provider, gpt.model, gpt.cost_known, gpt.turns) == ("codex", "gpt-5.5", True, 1)
    assert gpt.usage_event_ids == (child_other.event_id,)
    assert gpt.rate_table_version == "2026-04-08"
    assert gpt.input_tokens == 100
    assert gpt.cost == pytest.approx(0.125)

    root_only = await root.snapshot()
    assert len(root_only.provider_model_usage) == 1
    assert root_only.provider_model_usage[0].usage_event_ids == (root_event.event_id,)
    assert root_only.provider_model_usage[0].rate_table_version == "2026-04-08"
    assert root_only.cost_known is True

  _run(case())
