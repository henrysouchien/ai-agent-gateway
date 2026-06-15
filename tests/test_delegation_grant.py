import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.approval_policy import DelegationGrant
from agent_gateway.approval_store import SQLiteApprovalStore


BASE_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _run(awaitable: Any) -> Any:
  return asyncio.run(awaitable)


def _store(tmp_path: Path) -> SQLiteApprovalStore:
  return SQLiteApprovalStore(tmp_path / "approvals.sqlite3")


def _grant(**overrides: Any) -> DelegationGrant:
  values: dict[str, Any] = {
    "delegation_id": "delegation-1",
    "delegator_user_id": "alice",
    "delegator_run_id": "run-1",
    "delegator_session_id": "chat-session-1",
    "delegator_profile": "chat",
    "delegator_channel": "web",
    "bound_excel_session_id": "excel-session-1",
    "bound_relay_request_id": "relay-request-1",
    "bound_workbook": "Budget.xlsx",
    "tool_class_ceiling": frozenset({"read", "pure_transform", "state_write"}),
    "args_predicate": {"ticker": "MSFT", "account_id": "acct-1"},
    "window_seconds": 120,
    "exclude_external_write_bypass": True,
    "created_at": BASE_NOW,
    "expires_at": BASE_NOW + timedelta(seconds=120),
    "revoked_at": None,
    "consumed_at": None,
  }
  values.update(overrides)
  return DelegationGrant(**values)


def test_create_and_get_round_trips_all_fields(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    grant = _grant(
      delegation_id="delegation-roundtrip",
      delegator_run_id="run-roundtrip",
      delegator_session_id="chat-session-roundtrip",
      delegator_profile="analyst",
      delegator_channel="cli",
      bound_excel_session_id="excel-roundtrip",
      bound_relay_request_id="relay-roundtrip",
      bound_workbook="Forecast.xlsx",
      tool_class_ceiling=frozenset({"read", "artifact_write"}),
      args_predicate={"portfolio": "core", "tickers": ["MSFT", "AAPL"]},
      window_seconds=300,
      exclude_external_write_bypass=False,
      created_at=BASE_NOW + timedelta(minutes=1),
      expires_at=BASE_NOW + timedelta(minutes=6),
      revoked_at=BASE_NOW + timedelta(minutes=2),
      consumed_at=BASE_NOW + timedelta(minutes=3),
    )

    created = await store.create_delegation_grant(grant)
    stored = await store.get_delegation_grant("delegation-roundtrip")

    assert created == grant
    assert stored == grant
    assert stored is not None
    assert stored.tool_class_ceiling == frozenset({"read", "artifact_write"})
    assert stored.args_predicate == {"portfolio": "core", "tickers": ["MSFT", "AAPL"]}
    assert stored.bound_excel_session_id == "excel-roundtrip"
    assert stored.bound_relay_request_id == "relay-roundtrip"
    assert stored.bound_workbook == "Forecast.xlsx"

  _run(_case())


def test_claim_success_consumes_once(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    await store.create_delegation_grant(_grant(delegation_id="delegation-claim"))

    claim_time = BASE_NOW + timedelta(seconds=10)
    claimed = await store.claim_delegation_grant(
      delegation_id="delegation-claim",
      bound_relay_request_id="relay-request-1",
      bound_excel_session_id="excel-session-1",
      now=claim_time,
    )
    second = await store.claim_delegation_grant(
      delegation_id="delegation-claim",
      bound_relay_request_id="relay-request-1",
      bound_excel_session_id="excel-session-1",
      now=claim_time + timedelta(seconds=1),
    )

    assert claimed is not None
    assert claimed.consumed_at == claim_time
    assert second is None

  _run(_case())


def test_claim_wrong_relay_request_id_does_not_consume(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    await store.create_delegation_grant(_grant(delegation_id="delegation-wrong-request"))

    wrong = await store.claim_delegation_grant(
      delegation_id="delegation-wrong-request",
      bound_relay_request_id="wrong-relay-request",
      bound_excel_session_id="excel-session-1",
      now=BASE_NOW + timedelta(seconds=10),
    )
    stored_after_wrong = await store.get_delegation_grant("delegation-wrong-request")
    correct = await store.claim_delegation_grant(
      delegation_id="delegation-wrong-request",
      bound_relay_request_id="relay-request-1",
      bound_excel_session_id="excel-session-1",
      now=BASE_NOW + timedelta(seconds=11),
    )

    assert wrong is None
    assert stored_after_wrong is not None
    assert stored_after_wrong.consumed_at is None
    assert correct is not None
    assert correct.consumed_at == BASE_NOW + timedelta(seconds=11)

  _run(_case())


def test_claim_wrong_excel_session_id_returns_none(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    await store.create_delegation_grant(_grant(delegation_id="delegation-wrong-session"))

    claimed = await store.claim_delegation_grant(
      delegation_id="delegation-wrong-session",
      bound_relay_request_id="relay-request-1",
      bound_excel_session_id="wrong-excel-session",
      now=BASE_NOW + timedelta(seconds=10),
    )

    assert claimed is None

  _run(_case())


def test_claim_expired_grant_returns_none(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    await store.create_delegation_grant(
      _grant(
        delegation_id="delegation-expired",
        expires_at=BASE_NOW - timedelta(seconds=1),
      )
    )

    claimed = await store.claim_delegation_grant(
      delegation_id="delegation-expired",
      bound_relay_request_id="relay-request-1",
      bound_excel_session_id="excel-session-1",
      now=BASE_NOW,
    )

    assert claimed is None

  _run(_case())


def test_claim_revoked_grant_returns_none(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    await store.create_delegation_grant(
      _grant(
        delegation_id="delegation-revoked",
        revoked_at=BASE_NOW + timedelta(seconds=1),
      )
    )

    claimed = await store.claim_delegation_grant(
      delegation_id="delegation-revoked",
      bound_relay_request_id="relay-request-1",
      bound_excel_session_id="excel-session-1",
      now=BASE_NOW + timedelta(seconds=2),
    )

    assert claimed is None

  _run(_case())


def test_revoke_then_get_sets_revoked_at_and_missing_get_returns_none(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    await store.create_delegation_grant(_grant(delegation_id="delegation-revoke"))

    revoked_at = BASE_NOW + timedelta(seconds=30)
    await store.revoke_delegation_grant("delegation-revoke", revoked_at=revoked_at)

    stored = await store.get_delegation_grant("delegation-revoke")
    missing = await store.get_delegation_grant("missing-delegation")

    assert stored is not None
    assert stored.revoked_at == revoked_at
    assert missing is None

  _run(_case())
