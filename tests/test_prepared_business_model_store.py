from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from agent_gateway.approval_policy import ApprovalRequest, RunContext, build_approval_request
from agent_gateway import approval_store as approval_store_module
from agent_gateway.approval_store import (
  ApprovalMaintenanceResult,
  PreparedReconciliationCursor,
  PreparedReconciliationConflict,
  PreparedReconciliationResult,
  SQLiteApprovalStore,
)
from agent_gateway.prepared_business_model_store import (
  PreparedBusinessModelChange,
  PreparedBusinessModelError,
  PreparedBusinessModelLifecycle,
)


def _canonical(value: object) -> bytes:
  return json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
  ).encode()


def _record(
  *,
  locator: str = "request-1",
  intent_digest: str = "1" * 64,
  expires_at: str | None = "2099-01-01T00:00:00+00:00",
  approval_id: str = "approval-1",
  approval_chain_id: str = "chain-1",
) -> PreparedBusinessModelChange:
  payload = _canonical({"change_set_id": "2" * 64, "profile": "accept-only"})
  return PreparedBusinessModelChange(
    schema_version="prepared-business-model-change.v1",
    caller_kind="research_http",
    user_scope="tenant-1:user-7",
    idempotency_locator=locator,
    intent_digest=intent_digest,
    prepared_payload=payload,
    change_set_id="2" * 64,
    change_hash="3" * 64,
    base_vector_hash="4" * 64,
    prepared_payload_digest=hashlib.sha256(payload).hexdigest(),
    lifecycle=PreparedBusinessModelLifecycle.PENDING,
    created_at=datetime.now(UTC).isoformat(),
    expires_at=expires_at,
    approval_id=approval_id,
    approval_chain_id=approval_chain_id,
  )


def _request_and_record(
  *,
  locator: str,
  approval_expires_at: datetime | None = None,
) -> tuple[ApprovalRequest, PreparedBusinessModelChange]:
  request = build_approval_request(
    tool_call_id=f"business-model-http:user-7:{locator}",
    tool_name="accept_business_model_revision",
    tool_class="state_write",
    tool_args_redacted={"request_id": locator},
    args_hash="1" * 64,
    run_context=RunContext(
      user_id="tenant-1:user-7",
      request_id=locator,
      profile="research_http",
      channel="web",
    ),
    state="pending_user",
  )
  request = replace(
    request,
    expires_at=approval_expires_at or request.requested_at + timedelta(minutes=15),
  )
  return request, _record(
    locator=locator,
    approval_id=request.approval_id,
    approval_chain_id=request.approval_chain_id,
  )


@pytest.mark.asyncio
async def test_insert_or_verify_is_prepare_once_and_conflicting_intent_fails(tmp_path) -> None:
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  record = _record()

  first, created = await store.insert_or_verify_prepared_business_model_change(record)
  replay, replay_created = await store.insert_or_verify_prepared_business_model_change(record)

  assert created is True
  assert replay_created is False
  assert replay == first
  with pytest.raises(PreparedBusinessModelError, match="contradictory intent"):
    await store.insert_or_verify_prepared_business_model_change(
      _record(intent_digest="9" * 64)
    )
  with pytest.raises(PreparedBusinessModelError, match="contradictory intent"):
    await store.insert_or_verify_prepared_business_model_change(
      _record(approval_id="approval-other", approval_chain_id="chain-other")
    )


@pytest.mark.asyncio
async def test_approval_and_prepared_change_are_created_atomically(
  tmp_path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  request = build_approval_request(
    tool_call_id="business-model-http:user-7:request-1",
    tool_name="accept_business_model_revision",
    tool_class="state_write",
    tool_args_redacted={"request_id": "request-1"},
    args_hash="1" * 64,
    run_context=RunContext(
      user_id="tenant-1:user-7",
      request_id="request-1",
      profile="research_http",
      channel="web",
    ),
    state="pending",
  )
  request = replace(
    request,
    expires_at=request.requested_at + timedelta(minutes=15),
  )
  record = _record(
    approval_id=request.approval_id,
    approval_chain_id=request.approval_chain_id,
  )
  store = SQLiteApprovalStore(tmp_path / "atomic.sqlite3")

  approval, stored, created = (
    await store.create_or_get_with_prepared_business_model_change(request, record)
  )
  replay_approval, replay_record, replay_created = (
    await store.create_or_get_with_prepared_business_model_change(request, record)
  )

  assert created is True
  assert replay_created is False
  assert replay_approval == approval
  assert replay_record == stored
  assert stored.approval_id == approval.approval_id
  assert await store.get_by_tool_call_id(request.tool_call_id) == approval

  failing_store = SQLiteApprovalStore(tmp_path / "rollback.sqlite3")
  from agent_gateway import approval_store as approval_store_module

  def fail_prepared_insert(*_args, **_kwargs):
    raise RuntimeError("injected prepared insert failure")

  monkeypatch.setattr(
    approval_store_module._prepared_bm,
    "insert_or_verify",
    fail_prepared_insert,
  )
  with pytest.raises(RuntimeError, match="injected prepared insert failure"):
    await failing_store.create_or_get_with_prepared_business_model_change(
      request,
      record,
    )
  assert await failing_store.get_by_tool_call_id(request.tool_call_id) is None


@pytest.mark.asyncio
async def test_prepared_change_without_finite_expiry_is_rejected_atomically(tmp_path) -> None:
  request = build_approval_request(
    tool_call_id="business-model-http:user-7:no-expiry",
    tool_name="accept_business_model_revision",
    tool_class="state_write",
    tool_args_redacted={"request_id": "no-expiry"},
    args_hash="1" * 64,
    run_context=RunContext(
      user_id="tenant-1:user-7",
      request_id="no-expiry",
      profile="research_http",
      channel="web",
    ),
    state="pending",
  )
  record = _record(
    locator="no-expiry",
    expires_at=None,
    approval_id=request.approval_id,
    approval_chain_id=request.approval_chain_id,
  )
  store = SQLiteApprovalStore(tmp_path / "no-expiry.sqlite3")

  with pytest.raises(PreparedBusinessModelError, match="finite expiry"):
    await store.create_or_get_with_prepared_business_model_change(request, record)
  assert await store.get_by_tool_call_id(request.tool_call_id) is None


@pytest.mark.asyncio
async def test_prepared_lifecycle_authorize_consume_and_precommit_supersede(tmp_path) -> None:
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  await store.insert_or_verify_prepared_business_model_change(_record())
  authorized = await store.transition_prepared_business_model_change(
    caller_kind="research_http",
    user_scope="tenant-1:user-7",
    idempotency_locator="request-1",
    expected=PreparedBusinessModelLifecycle.PENDING,
    target=PreparedBusinessModelLifecycle.AUTHORIZED,
    approval_id="approval-1",
    approval_chain_id="chain-1",
  )
  assert authorized.lifecycle is PreparedBusinessModelLifecycle.AUTHORIZED

  receipt = _canonical({"status": "COMMITTED", "change_set_id": "2" * 64})
  consumed_at = datetime.now(UTC).isoformat()
  consumed = await store.transition_prepared_business_model_change(
    caller_kind="research_http",
    user_scope="tenant-1:user-7",
    idempotency_locator="request-1",
    expected=PreparedBusinessModelLifecycle.AUTHORIZED,
    target=PreparedBusinessModelLifecycle.CONSUMED,
    execution_receipt=receipt,
    checkpoint_id="5" * 64,
    consumed_at=consumed_at,
  )
  assert consumed.lifecycle is PreparedBusinessModelLifecycle.CONSUMED
  assert consumed.checkpoint_id == "5" * 64
  assert consumed.execution_receipt_digest == hashlib.sha256(receipt).hexdigest()
  replay = await store.transition_prepared_business_model_change(
    caller_kind="research_http",
    user_scope="tenant-1:user-7",
    idempotency_locator="request-1",
    expected=PreparedBusinessModelLifecycle.AUTHORIZED,
    target=PreparedBusinessModelLifecycle.CONSUMED,
    approval_id="approval-1",
    approval_chain_id="chain-1",
    execution_receipt=receipt,
    checkpoint_id="5" * 64,
    consumed_at=consumed_at,
  )
  assert replay == consumed
  with pytest.raises(PreparedBusinessModelError, match="evidence conflicts"):
    await store.transition_prepared_business_model_change(
      caller_kind="research_http",
      user_scope="tenant-1:user-7",
      idempotency_locator="request-1",
      expected=PreparedBusinessModelLifecycle.AUTHORIZED,
      target=PreparedBusinessModelLifecycle.CONSUMED,
      approval_id="approval-1",
      approval_chain_id="chain-1",
      execution_receipt=_canonical({"status": "contradictory"}),
      checkpoint_id="5" * 64,
      consumed_at=consumed_at,
    )

  await store.insert_or_verify_prepared_business_model_change(
    _record(
      locator="request-2",
      approval_id="approval-2",
      approval_chain_id="chain-2",
    )
  )
  await store.transition_prepared_business_model_change(
    caller_kind="research_http",
    user_scope="tenant-1:user-7",
    idempotency_locator="request-2",
    expected=PreparedBusinessModelLifecycle.PENDING,
    target=PreparedBusinessModelLifecycle.AUTHORIZED,
    approval_id="approval-2",
    approval_chain_id="chain-2",
  )
  superseded = await store.transition_prepared_business_model_change(
    caller_kind="research_http",
    user_scope="tenant-1:user-7",
    idempotency_locator="request-2",
    expected=PreparedBusinessModelLifecycle.AUTHORIZED,
    target=PreparedBusinessModelLifecycle.SUPERSEDED_PRECOMMIT,
    execution_receipt=_canonical({"status": "FAILED_PRECOMMIT"}),
    restoration_digest="6" * 64,
  )
  assert superseded.lifecycle is PreparedBusinessModelLifecycle.SUPERSEDED_PRECOMMIT
  assert superseded.restoration_digest == "6" * 64


@pytest.mark.asyncio
async def test_prepared_lifecycle_rejects_arbitrary_edges_and_invalid_terminal_rows(
  tmp_path,
) -> None:
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  await store.insert_or_verify_prepared_business_model_change(_record())
  with pytest.raises(PreparedBusinessModelError, match="invalid prepared lifecycle"):
    await store.transition_prepared_business_model_change(
      caller_kind="research_http",
      user_scope="tenant-1:user-7",
      idempotency_locator="request-1",
      expected=PreparedBusinessModelLifecycle.PENDING,
      target=PreparedBusinessModelLifecycle.CONSUMED,
      approval_id="approval-1",
      approval_chain_id="chain-1",
      execution_receipt=_canonical({"status": "COMMITTED"}),
      checkpoint_id="5" * 64,
      consumed_at=datetime.now(UTC).isoformat(),
    )
  with pytest.raises(PreparedBusinessModelError, match="CONSUMED requires"):
    replace(_record(), lifecycle=PreparedBusinessModelLifecycle.CONSUMED)
  with pytest.raises(PreparedBusinessModelError, match="must start PENDING"):
    await store.insert_or_verify_prepared_business_model_change(
      replace(
        _record(locator="terminal-insert"),
        lifecycle=PreparedBusinessModelLifecycle.AUTHORIZED,
        approval_id="approval-2",
        approval_chain_id="chain-2",
      )
    )


@pytest.mark.asyncio
async def test_pending_prepared_records_expire_without_changing_payload(tmp_path) -> None:
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  expired_at = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
  record = _record(expires_at=expired_at)
  await store.insert_or_verify_prepared_business_model_change(record)

  assert await store.expire_prepared_business_model_changes(
    now=datetime.now(UTC).isoformat()
  ) == 1
  loaded = await store.get_prepared_business_model_change(
    caller_kind=record.caller_kind,
    user_scope=record.user_scope,
    idempotency_locator=record.idempotency_locator,
  )
  assert loaded is not None
  assert loaded.lifecycle is PreparedBusinessModelLifecycle.EXPIRED
  assert loaded.prepared_payload == record.prepared_payload

  authorized = _record(locator="authorized", expires_at=expired_at)
  await store.insert_or_verify_prepared_business_model_change(authorized)
  await store.transition_prepared_business_model_change(
    caller_kind=authorized.caller_kind,
    user_scope=authorized.user_scope,
    idempotency_locator=authorized.idempotency_locator,
    expected=PreparedBusinessModelLifecycle.PENDING,
    target=PreparedBusinessModelLifecycle.AUTHORIZED,
    approval_id=authorized.approval_id,
    approval_chain_id=authorized.approval_chain_id,
  )
  assert await store.expire_prepared_business_model_changes(
    now=datetime.now(UTC).isoformat()
  ) == 0
  retained = await store.get_prepared_business_model_change(
    caller_kind=authorized.caller_kind,
    user_scope=authorized.user_scope,
    idempotency_locator=authorized.idempotency_locator,
  )
  assert retained is not None
  assert retained.lifecycle is PreparedBusinessModelLifecycle.AUTHORIZED


@pytest.mark.asyncio
async def test_targeted_reconciliation_gives_prepared_ttl_precedence_over_approval(
  tmp_path,
) -> None:
  store = SQLiteApprovalStore(tmp_path / "ttl-first.sqlite3")
  request, record = _request_and_record(
    locator="ttl-first",
    approval_expires_at=datetime(2026, 1, 1, tzinfo=UTC),
  )
  await store.create_or_get_with_prepared_business_model_change(request, record)
  with store._connection() as conn:
    conn.execute(
      "UPDATE prepared_business_model_change SET expires_at = ? WHERE idempotency_locator = ?",
      ("2026-01-01T00:00:00Z", record.idempotency_locator),
    )
  approved = await store.transition_state(
    request.approval_id,
    "approved",
    expected_state_version=request.state_version,
  )
  assert approved.state == "approved"

  result = await store.reconcile_prepared_business_model_change(
    caller_kind=record.caller_kind,
    user_scope=record.user_scope,
    idempotency_locator=record.idempotency_locator,
    now=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
  )

  assert result.transitioned is True
  assert result.conflict is None
  assert result.record is not None
  assert result.record.lifecycle is PreparedBusinessModelLifecycle.EXPIRED


@pytest.mark.asyncio
async def test_maintenance_expires_approval_and_reconciles_prepared_in_one_pass(
  tmp_path,
) -> None:
  store = SQLiteApprovalStore(tmp_path / "maintenance.sqlite3")
  now = datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC)
  request, record = _request_and_record(
    locator="maintenance",
    approval_expires_at=datetime(2026, 1, 1, tzinfo=UTC),
  )
  await store.create_or_get_with_prepared_business_model_change(request, record)

  result = await store.maintain_pending(now=now)

  assert result.approvals_expired == 1
  assert result.prepared.scanned == 1
  assert result.prepared.expired == 1
  assert (await store.get(request.approval_id)).state == "expired"
  reconciled = await store.get_prepared_business_model_change(
    caller_kind=record.caller_kind,
    user_scope=record.user_scope,
    idempotency_locator=record.idempotency_locator,
  )
  assert reconciled is not None
  assert reconciled.lifecycle is PreparedBusinessModelLifecycle.EXPIRED
  assert await store.expire_pending(now=now) == 0


@pytest.mark.asyncio
async def test_prepared_reconciliation_cursor_pages_and_wraps_unresolved_rows(
  tmp_path,
) -> None:
  store = SQLiteApprovalStore(tmp_path / "cursor.sqlite3")
  requests: list[ApprovalRequest] = []
  for locator in ("a-approved", "b-approved", "z-pending"):
    request, record = _request_and_record(locator=locator)
    await store.create_or_get_with_prepared_business_model_change(request, record)
    requests.append(request)
  for request in requests[:2]:
    await store.transition_state(
      request.approval_id,
      "approved",
      expected_state_version=request.state_version,
    )

  first = await store.maintain_pending(prepared_page_size=2)
  assert first.prepared.scanned == 2
  assert first.prepared.authorized == 2
  assert first.prepared.cursor is not None
  assert first.prepared.wrapped is False

  second = await store.maintain_pending(
    prepared_cursor=first.prepared.cursor,
    prepared_page_size=2,
  )
  assert second.prepared.scanned == 1
  assert second.prepared.authorized == 0
  assert second.prepared.cursor is not None
  assert second.prepared.cursor.idempotency_locator == "z-pending"

  wrapped = await store.maintain_pending(
    prepared_cursor=second.prepared.cursor,
    prepared_page_size=2,
  )
  assert wrapped.prepared.wrapped is True
  assert wrapped.prepared.scanned == 1
  assert wrapped.prepared.cursor is not None
  assert wrapped.prepared.cursor.idempotency_locator == "z-pending"


@pytest.mark.asyncio
async def test_prepared_reconciliation_reports_typed_nonterminal_conflicts(
  tmp_path,
) -> None:
  store = SQLiteApprovalStore(tmp_path / "conflicts.sqlite3")
  missing = _record(
    locator="a-missing",
    approval_id="missing-approval",
    approval_chain_id="missing-chain",
  )
  await store.insert_or_verify_prepared_business_model_change(missing)
  lineage_request, lineage = _request_and_record(locator="b-lineage")
  await store.create_or_get_with_prepared_business_model_change(
    lineage_request,
    lineage,
  )
  unknown_request, unknown = _request_and_record(locator="c-unknown")
  await store.create_or_get_with_prepared_business_model_change(
    unknown_request,
    unknown,
  )
  with store._connection() as conn:
    conn.execute(
      "UPDATE approval_requests SET approval_chain_id = ? WHERE approval_id = ?",
      ("conflicting-chain", lineage_request.approval_id),
    )
    conn.execute(
      "UPDATE approval_requests SET state = ? WHERE approval_id = ?",
      ("future_state", unknown_request.approval_id),
    )

  result = await store.maintain_pending(prepared_page_size=10)

  assert result.prepared.scanned == 3
  assert result.prepared.missing_approval == 1
  assert result.prepared.lineage_conflict == 1
  assert result.prepared.unknown_approval_state == 1
  assert result.prepared.conflict_count == 3
  targeted = await store.reconcile_prepared_business_model_change(
    caller_kind=missing.caller_kind,
    user_scope=missing.user_scope,
    idempotency_locator=missing.idempotency_locator,
  )
  assert targeted.conflict is PreparedReconciliationConflict.MISSING_APPROVAL


@pytest.mark.asyncio
async def test_two_store_reconcilers_authorize_once_without_duplicate_transition(
  tmp_path,
) -> None:
  path = tmp_path / "two-store.sqlite3"
  first_store = SQLiteApprovalStore(path)
  second_store = SQLiteApprovalStore(path)
  request, record = _request_and_record(locator="two-store")
  await first_store.create_or_get_with_prepared_business_model_change(request, record)
  await first_store.transition_state(
    request.approval_id,
    "approved",
    expected_state_version=request.state_version,
  )

  async def reconcile(store: SQLiteApprovalStore):
    return await store.reconcile_prepared_business_model_change(
      caller_kind=record.caller_kind,
      user_scope=record.user_scope,
      idempotency_locator=record.idempotency_locator,
    )

  first, second = await asyncio.gather(
    reconcile(first_store),
    reconcile(second_store),
  )

  assert first.record is not None and second.record is not None
  assert first.record.lifecycle is PreparedBusinessModelLifecycle.AUTHORIZED
  assert second.record.lifecycle is PreparedBusinessModelLifecycle.AUTHORIZED
  assert int(first.transitioned) + int(second.transitioned) == 1


@pytest.mark.asyncio
async def test_maintenance_loop_logs_bounded_summary_and_typed_conflicts(
  monkeypatch: pytest.MonkeyPatch,
  caplog: pytest.LogCaptureFixture,
) -> None:
  class StopLoop(RuntimeError):
    pass

  sleep_calls = 0

  async def stop_after_one_iteration(_seconds: float) -> None:
    nonlocal sleep_calls
    sleep_calls += 1
    if sleep_calls > 1:
      raise StopLoop

  cursor = PreparedReconciliationCursor(
    caller_kind="research_http",
    user_scope="sensitive-user-scope",
    idempotency_locator="sensitive-request-locator",
  )

  class Store:
    async def maintain_pending(self, **_kwargs) -> ApprovalMaintenanceResult:
      return ApprovalMaintenanceResult(
        approvals_expired=2,
        prepared=PreparedReconciliationResult(
          scanned=4,
          authorized=1,
          expired=1,
          missing_approval=1,
          cursor=cursor,
          wrapped=True,
        ),
      )

  monkeypatch.setattr(
    approval_store_module.asyncio,
    "sleep",
    stop_after_one_iteration,
  )
  with caplog.at_level(logging.INFO, logger="agent_gateway.approval_store"):
    with pytest.raises(StopLoop):
      await approval_store_module.expire_pending_loop(Store(), interval_seconds=0)

  summary = next(
    record for record in caplog.records
    if record.message == "approval maintenance completed"
  )
  warning = next(
    record for record in caplog.records
    if record.message == "prepared BusinessModel reconciliation conflicts"
  )
  assert summary.approvals_expired == 2
  assert summary.prepared_scanned == 4
  assert summary.prepared_cursor == cursor.log_token
  assert summary.prepared_cursor_wrapped is True
  assert warning.prepared_missing_approval == 1
  assert "sensitive-user-scope" not in summary.message
  assert "sensitive-request-locator" not in warning.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ("mode", "grant_reference", "cache_reference"),
  [
    ("HUMAN", None, None),
    ("AUTO_ALLOW", None, None),
    ("CACHE_HIT", None, "session-cache:tool:scope"),
    ("PERSISTENT_GRANT", "grant-1", None),
    ("OWNER_CONTROL_PLANE", None, None),
  ],
)
async def test_approval_store_persists_typed_authorization_mode(
  tmp_path,
  mode,
  grant_reference,
  cache_reference,
) -> None:
  store = SQLiteApprovalStore(tmp_path / f"{mode}.sqlite3")
  request = build_approval_request(
    tool_call_id=f"call-{mode}",
    tool_name="fms_persist_business_model",
    tool_class="state_write",
    tool_args_redacted={"ticker": "PCTY"},
    args_hash="args-hash",
    run_context=RunContext(user_id="alice", request_id="request-1"),
  )
  request = replace(
    request,
    authorization_mode=mode,
    grant_reference=grant_reference,
    cache_reference=cache_reference,
  )
  await store.create(request)
  loaded = await store.get(request.approval_id)
  assert loaded is not None
  assert loaded.authorization_mode == mode
  assert loaded.grant_reference == grant_reference
  assert loaded.cache_reference == cache_reference
