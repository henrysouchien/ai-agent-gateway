from __future__ import annotations

import json

import pytest

from agent_gateway.approval_policy import RunContext, build_approval_request
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.raw_patch_authorization_store import (
  RawPatchAuthorizationError,
  claim,
  encode_reference,
  release_claim,
)


@pytest.mark.asyncio
async def test_reference_resolution_and_receipt_consumption_are_exact_and_idempotent(
  tmp_path,
) -> None:
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  request = build_approval_request(
    tool_call_id="tool-call-1",
    tool_name="apply_patch_ops",
    tool_class="state_write",
    tool_args_redacted={},
    args_hash="args-hash",
    run_context=RunContext(user_id="alice", request_id="request-1"),
  )
  prepared = json.dumps(
    {"schema_version": 1, "binding": {"change_set_id": "bound"}},
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
  ).encode()
  await store.create_raw_patch_authorization(
    request,
    prepared_payload=prepared,
  )

  resolved = await store.resolve_raw_patch_authorization_reference(
    encode_reference(request.tool_call_id)
  )
  assert resolved == request

  bound, bound_created = await store.bind_raw_patch_prepared_authorization(
    approval_id=request.approval_id,
    prepared_payload=prepared,
  )
  rebound, rebound_created = await store.bind_raw_patch_prepared_authorization(
    approval_id=request.approval_id,
    prepared_payload=prepared,
  )
  assert bound_created is False
  assert rebound_created is False
  assert rebound == bound
  with pytest.raises(RawPatchAuthorizationError, match="different prepared"):
    await store.bind_raw_patch_prepared_authorization(
      approval_id=request.approval_id,
      prepared_payload=b'{"different":true}',
    )


@pytest.mark.asyncio
async def test_raw_patch_approval_and_prepared_bytes_are_created_atomically(
  tmp_path,
) -> None:
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  request = build_approval_request(
    tool_call_id="tool-call-invalid",
    tool_name="apply_patch_ops",
    tool_class="state_write",
    tool_args_redacted={},
    args_hash="args-hash",
    run_context=RunContext(user_id="alice", request_id="request-invalid"),
  )

  with pytest.raises(RawPatchAuthorizationError):
    await store.create_raw_patch_authorization(
      request,
      prepared_payload=b"not-json",
    )

  assert await store.get(request.approval_id) is None
  assert (
    await store.resolve_raw_patch_authorization_reference(
      encode_reference(request.tool_call_id)
    )
    is None
  )


@pytest.mark.asyncio
async def test_claim_excludes_approval_state_transition_until_finalization(
  tmp_path,
) -> None:
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  request = build_approval_request(
    tool_call_id="tool-call-claimed",
    tool_name="apply_patch_ops",
    tool_class="state_write",
    tool_args_redacted={},
    args_hash="args-hash",
    run_context=RunContext(user_id="alice", request_id="request-claimed"),
  )
  await store.create(request)
  digest = "sha256:" + "a" * 64
  with store._connection() as conn:
    conn.execute("BEGIN IMMEDIATE")
    claim(
      conn,
      approval_id=request.approval_id,
      claim_token="claim-token",
      change_set_id="reviewed-change:v1:" + "b" * 64,
      change_hash=digest,
      prepared_payload_digest=digest,
      claimed_at="2026-01-01T00:00:00+00:00",
      lease_seconds=10.0,
    )
    with pytest.raises(RawPatchAuthorizationError, match="active execution claim"):
      claim(
        conn,
        approval_id=request.approval_id,
        claim_token="competing-token",
        change_set_id="reviewed-change:v1:" + "b" * 64,
        change_hash=digest,
        prepared_payload_digest=digest,
        claimed_at="2026-01-01T00:00:05+00:00",
      )
    reclaimed, reclaimed_created = claim(
      conn,
      approval_id=request.approval_id,
      claim_token="takeover-token",
      change_set_id="reviewed-change:v1:" + "b" * 64,
      change_hash=digest,
      prepared_payload_digest=digest,
      claimed_at="2026-01-01T00:00:11+00:00",
    )
    assert reclaimed_created is True
    assert reclaimed.claim_token == "takeover-token"
    conn.commit()

  with pytest.raises(RawPatchAuthorizationError, match="cannot transition"):
    await store.transition_state(request.approval_id, "denied")

  with store._connection() as conn:
    release_claim(
      conn,
      approval_id=request.approval_id,
      claim_token="takeover-token",
    )
    conn.commit()
  transitioned = await store.transition_state(request.approval_id, "denied")
  assert transitioned.state == "denied"
