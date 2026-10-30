from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from agent_gateway.approval_remote import (
  ApprovalRemoteOperation,
  AutonomousPublicationReservation,
  BoundApprovalRemoteClient,
  PublishAbortResult,
  PublishCommitResult,
  PublishReserveResult,
  encode_operation_result,
  parse_remote_request,
)
from agent_gateway.approval_remote_store import (
  RemoteApprovalRequestStore,
  RuntimeExecutionAuditEmitter,
)

import pytest

from agent_gateway import approval_store as approval_store_module
from agent_gateway.approval_policy import ApprovalRequest, ApprovalVote
from agent_gateway.approval_store import SQLiteApprovalStore


def _request(*, deadline: datetime) -> ApprovalRequest:
  return ApprovalRequest(
    approval_id="approval-deadline",
    tool_call_id="tool-deadline",
    parent_approval_id=None,
    approval_chain_id="approval-deadline",
    request_id="run-deadline",
    session_id="session-deadline",
    run_id="run-deadline",
    user_id="owner",
    profile="analyst",
    channel="tui",
    tool_name="memory_write",
    tool_class="state_write",
    tool_args_redacted={},
    args_hash="deadline-hash",
    reason="deadline test",
    blast_radius_summary="state_write:memory_write",
    state="pending_user",
    requested_at=datetime(2034, 12, 31, tzinfo=UTC),
    expires_at=deadline,
    approval_constraint="standard",
  )


def _vote(*, decided_at: datetime) -> ApprovalVote:
  return ApprovalVote(
    vote_id="vote-deadline",
    approval_id="approval-deadline",
    decider_id="owner",
    decider_role="owner",
    decision="approved",
    decision_reason="reviewed",
    decided_at=decided_at,
  )


def _delivery() -> dict[str, str]:
  return {
    "task_id": "bg_deadline",
    "control_run_id": "run-deadline",
    "session_id": "session-deadline",
    "channel_id": "a" * 64,
    "tool_call_id": "tool-deadline",
    "nonce": "nonce-deadline",
  }


@pytest.mark.parametrize("trusted_offset_ns", [0, 1, 3_600 * 10**9])
def test_vote_at_or_after_recorded_expiry_is_delivered(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path,
  trusted_offset_ns: int,
) -> None:
  # Elapsed time never decides a non-money approval: a late authenticated vote
  # is recorded and its autonomous delivery is queued.
  deadline = datetime(2035, 1, 1, tzinfo=UTC)
  deadline_ns = approval_store_module._datetime_to_epoch_ns(deadline)
  monkeypatch.setattr(
    approval_store_module.time,
    "time_ns",
    lambda: deadline_ns + trusted_offset_ns,
  )
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  asyncio.run(store.create(_request(deadline=deadline)))

  resolved = asyncio.run(
    store.record_vote_with_autonomous_delivery(
      "approval-deadline",
      _vote(decided_at=deadline),
      delivery=_delivery(),
    )
  )

  assert resolved.state == "approved"
  with store._connection() as conn:
    vote_count = conn.execute(
      "SELECT COUNT(*) FROM approval_votes WHERE approval_id = ?",
      ("approval-deadline",),
    ).fetchone()[0]
  assert vote_count == 1
  delivery = asyncio.run(
    store.get_autonomous_approval_delivery(
      "approval-deadline",
      tool_call_id="tool-deadline",
      nonce="nonce-deadline",
    )
  )
  assert delivery is not None
  assert delivery["approved"] is True


def test_vote_refuses_server_clock_rollback_without_mutating_approval(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path,
) -> None:
  observed = datetime(2035, 1, 1, 12, tzinfo=UTC)
  deadline = datetime(2035, 1, 1, 13, tzinfo=UTC)
  observed_ns = approval_store_module._datetime_to_epoch_ns(observed)
  monkeypatch.setattr(
    approval_store_module.time,
    "time_ns",
    lambda: observed_ns,
  )
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  asyncio.run(store.create(_request(deadline=deadline)))
  monkeypatch.setattr(
    approval_store_module.time,
    "time_ns",
    lambda: observed_ns - 1,
  )

  with pytest.raises(RuntimeError, match="clock rollback"):
    asyncio.run(
      store.record_vote_with_autonomous_delivery(
        "approval-deadline",
        _vote(decided_at=observed),
        delivery=_delivery(),
      )
    )

  stored = asyncio.run(store.get("approval-deadline"))
  assert stored is not None
  assert stored.state == "pending_user"
  assert asyncio.run(
    store.get_autonomous_approval_delivery(
      "approval-deadline",
      tool_call_id="tool-deadline",
      nonce="nonce-deadline",
    )
  ) is None


class _PublicationTransport:
  def __init__(
    self, replies: list[tuple[ApprovalRemoteOperation, Any]],
  ) -> None:
    self.replies = iter(replies)
    self.calls: list[tuple[ApprovalRemoteOperation, dict[str, Any]]] = []

  def post(
    self, operation: ApprovalRemoteOperation, body: bytes,
  ) -> tuple[int, bytes]:
    decoded_operation, arguments = parse_remote_request(
      body, url_operation=operation,
    )
    self.calls.append((decoded_operation, arguments))
    expected_operation, result = next(self.replies)
    assert operation == expected_operation
    return 200, encode_operation_result(operation, result)


def _remote_store(client: BoundApprovalRemoteClient) -> RemoteApprovalRequestStore:
  return RemoteApprovalRequestStore(
    client,
    audit_emitter=RuntimeExecutionAuditEmitter(client),
  )


def test_remote_execution_audit_reuses_original_input_identity(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  from agent_gateway.approval_audit import build_audit_entry
  from agent_gateway.approval_remote import RemoteExecutionAuditEmitter

  raw_args = {"private": "original input"}
  redaction_calls: list[dict[str, Any]] = []

  def redact(tool_name: str, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    redaction_calls.append(dict(args))
    return {"private": "<original-redaction>"}

  request = _request(deadline=datetime(2035, 1, 1, tzinfo=UTC))
  original = build_audit_entry(
    request=request, raw_tool_args=raw_args,
    deployment_secret=b"original-secret", key_id="original-key",
    event_type="tool_executed_success", outcome="success",
    tool_input_redactor=redact,
  )
  request = replace(
    request,
    args_hash=original.args_hash,
    tool_args_redacted=original.tool_args_redacted,
  )
  entries: list[Any] = []

  async def transport(self: Any, **kwargs: Any) -> None:
    assert kwargs["raw_tool_args"] == {}
    entries.append(build_audit_entry(
      **kwargs, deployment_secret=b"edge-secret", key_id="edge-key",
      event_type="tool_executed_success", tool_input_redactor=redact,
    ))

  monkeypatch.setattr(RemoteExecutionAuditEmitter, "emit_execution_outcome", transport)
  client = BoundApprovalRemoteClient(
    sync_transport=_PublicationTransport([]), audit_emitter=None,
  )
  store = _remote_store(client)
  asyncio.run(store.audit_emitter.emit_execution_outcome(
    request, raw_args, "success",
  ))

  assert redaction_calls == [raw_args]
  assert entries[0].args_hash == original.args_hash
  assert entries[0].args_hash.startswith("hmac-sha256-v1:original-key:")
  assert entries[0].tool_args_redacted == {"private": "<original-redaction>"}
  assert raw_args == {"private": "original input"}


@pytest.mark.parametrize("body_fails", [False, True])
def test_remote_append_transaction_reserves_then_commits_or_aborts(
  body_fails: bool,
) -> None:
  reservation = AutonomousPublicationReservation(
    decision_id="approval-deadline", tool_call_id="tool-deadline",
    nonce="nonce-deadline", approved=True, placement_generation=7,
  )
  finish_operation: ApprovalRemoteOperation = (
    "publish.abort" if body_fails else "publish.commit"
  )
  transport = _PublicationTransport([
    ("publish.reserve", PublishReserveResult(reservation=reservation)),
    (
      finish_operation,
      PublishAbortResult(released=True) if body_fails
      else PublishCommitResult(state="published"),
    ),
  ])
  store = _remote_store(
    BoundApprovalRemoteClient(sync_transport=transport, audit_emitter=None),
  )

  async def publish() -> None:
    # Sync publication remains usable inside the caller's active event loop.
    with store.autonomous_approval_delivery_append_transaction(
      "approval-deadline", tool_call_id="tool-deadline",
      nonce="nonce-deadline", approved=True,
    ):
      assert [operation for operation, _ in transport.calls] == ["publish.reserve"]
      if body_fails:
        raise ValueError("append failed")

  if body_fails:
    with pytest.raises(ValueError, match="append failed"):
      asyncio.run(publish())
  else:
    asyncio.run(publish())

  assert [operation for operation, _ in transport.calls] == [
    "publish.reserve", finish_operation,
  ]
  assert transport.calls[1][1] == {"reservation": reservation}
  assert isinstance(store.audit_emitter, RuntimeExecutionAuditEmitter)


def test_remote_duplicate_transaction_reconciles_before_body() -> None:
  transport = _PublicationTransport([
    ("publish.reconcile_duplicate", {"published": True}),
  ])
  store = _remote_store(
    BoundApprovalRemoteClient(sync_transport=transport),
  )
  with store.autonomous_approval_delivery_duplicate_transaction(
    "approval-deadline", tool_call_id="tool-deadline",
    nonce="nonce-deadline", approved=True,
  ):
    assert transport.calls == [(
      "publish.reconcile_duplicate",
      {
        "approval_id": "approval-deadline", "tool_call_id": "tool-deadline",
        "nonce": "nonce-deadline", "approved": True,
      },
    )]
  assert len(transport.calls) == 1
  assert store.audit_emitter is not None


class _AsyncApprovalTransport:
  def __init__(self, transport: _PublicationTransport) -> None:
    self.transport = transport

  async def post(
    self, operation: ApprovalRemoteOperation, body: bytes,
  ) -> tuple[int, bytes]:
    return self.transport.post(operation, body)


@pytest.mark.parametrize("agentcore", [False, True])
def test_remote_get_uses_placement_transport_and_preserves_owner_scope(
  agentcore: bool,
) -> None:
  request = _request(deadline=datetime(2035, 1, 1, tzinfo=UTC))
  sync_transport = _PublicationTransport(
    [] if agentcore else [("request.get", request)],
  )
  tunnel_transport = _PublicationTransport(
    [("request.get", request)] if agentcore else [],
  )
  store = _remote_store(BoundApprovalRemoteClient(
    sync_transport=sync_transport,
    async_transport=_AsyncApprovalTransport(tunnel_transport) if agentcore else None,
  ))

  result = asyncio.run(store.get("approval-deadline", owner_user_id="owner"))

  assert result == request
  selected = tunnel_transport if agentcore else sync_transport
  unselected = sync_transport if agentcore else tunnel_transport
  assert selected.calls == [(
    "request.get", {"approval_id": "approval-deadline", "owner_user_id": "owner"},
  )]
  assert unselected.calls == []


def test_remote_pending_delivery_scan_preserves_uncapped_sequence_window() -> None:
  transport = _PublicationTransport([("outbox.list_pending", [])])
  store = _remote_store(
    BoundApprovalRemoteClient(sync_transport=transport),
  )

  assert asyncio.run(store.list_pending_autonomous_approval_deliveries(
    after_sequence=12, through_sequence=400,
  )) == []
  assert transport.calls == [(
    "outbox.list_pending", {"after_sequence": 12, "through_sequence": 400},
  )]
