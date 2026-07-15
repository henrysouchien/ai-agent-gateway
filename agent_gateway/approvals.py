from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from fastapi.encoders import jsonable_encoder

from .approval_policy import (
  ApprovalVote,
  PersistentGrant,
  approval_is_executable,
  new_approval_id,
  utc_now,
)
from .approval_store import PersistentGrantCancellationFenced
from .session import GatewaySession


log = logging.getLogger("agent_gateway.approvals")

TERMINAL_APPROVAL_STATES = frozenset({"auto_approved", "auto_denied", "approved", "denied", "expired"})


class ApprovalActionError(Exception):
  def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
    super().__init__(str(payload.get("error") or payload))
    self.status_code = status_code
    self.payload = payload


def _approval_request_to_dict(request_record: Any) -> dict[str, Any]:
  if is_dataclass(request_record):
    encoded = jsonable_encoder(asdict(request_record))
    if isinstance(encoded, dict):
      encoded.pop("notification_policy", None)
      if encoded.get("notification") is None:
        encoded.pop("notification", None)
    return encoded
  encoded = jsonable_encoder(request_record)
  if isinstance(encoded, dict):
    result = dict(encoded)
    result.pop("notification_policy", None)
    if result.get("notification") is None:
      result.pop("notification", None)
    return result
  return {"approval": encoded}


def _resolve_store_and_policy(
  *,
  target_session: GatewaySession,
  app_state: Any,
  authoritative_store: Any | None = None,
  authoritative_policy: Any | None = None,
) -> tuple[Any | None, Any | None]:
  if authoritative_store is not None or authoritative_policy is not None:
    return authoritative_store, authoritative_policy
  store = getattr(target_session, "approval_store", None) or getattr(app_state, "gateway_approval_store", None)
  policy = getattr(target_session, "approval_policy", None) or getattr(app_state, "gateway_approval_policy", None)
  return store, policy


def _approval_matches_authoritative_identity(
  request_record: Any,
  expected: Mapping[str, Any],
) -> bool:
  for field in (
    "approval_id",
    "tool_call_id",
    "user_id",
    "request_id",
    "run_id",
    "session_id",
  ):
    if str(getattr(request_record, field, "") or "") != str(expected.get(field) or ""):
      return False
  actual_channel = str(getattr(request_record, "channel", "") or "").strip().lower()
  expected_channel = str(expected.get("channel") or "").strip().lower()
  return actual_channel == expected_channel


def _approval_resolution_lock(target_session: GatewaySession, tool_call_id: str) -> Any:
  locks = getattr(target_session, "_approval_resolution_locks", None)
  if locks is None:
    locks = {}
    setattr(target_session, "_approval_resolution_locks", locks)
  lock = locks.get(tool_call_id)
  if lock is None:
    lock = asyncio.Lock()
    locks[tool_call_id] = lock
  return lock


def _cancellation_requested(pending_entry: dict[str, Any]) -> bool:
  return pending_entry.get("_approval_cancel_requested") is True


async def _revoke_persistent_grants_for_cancel(store: Any, approval_id: str) -> None:
  revoke = getattr(store, "revoke_persistent_grants_for_approval", None)
  if not callable(revoke):
    raise RuntimeError("approval store cannot revoke cancellation-raced grants")
  try:
    await revoke(approval_id)
  except asyncio.CancelledError:
    # Store audit hooks run after the SQLite commit. Retry once so both a
    # pre-commit cancellation and a post-commit audit cancellation converge on
    # the same durable no-active-grant state.
    await revoke(approval_id)


def _replace_queue_with_cancellation(approval_queue: Any, payload: dict[str, Any]) -> None:
  while True:
    try:
      approval_queue.get_nowait()
    except asyncio.QueueEmpty:
      break
  try:
    approval_queue.put_nowait(payload)
  except asyncio.QueueFull as exc:
    raise ApprovalActionError(409, {"error": "Approval cancellation queue unavailable"}) from exc


def _release_cancelled_approval(
  *,
  target_session: GatewaySession,
  pending_entry: dict[str, Any],
  tool_call_id: str,
  approval_id: str,
) -> None:
  pending_entry["status"] = "approval_received"
  pending_entry["_approval_cancelled"] = True
  approval_queue = target_session.approval_queues.get(tool_call_id)
  if approval_queue is not None:
    _replace_queue_with_cancellation(
      approval_queue,
      {
        "approved": False,
        "allow_tool_type": False,
        "approval_id": approval_id,
        "denied_by": None,
      },
    )


def _validate_pending_entry(*, pending_entry: dict[str, Any], tool_call_id: str, nonce: str, target_session: GatewaySession) -> Any:
  if pending_entry.get("status") == "approval_received":
    raise ApprovalActionError(409, {"error": "Approval already submitted"})

  if pending_entry.get("status") != "approval_pending":
    raise ApprovalActionError(409, {"error": "Invalid pending tool state for approval submission"})

  if pending_entry.get("nonce") != nonce:
    raise ApprovalActionError(409, {"error": "Nonce mismatch"})

  approval_queue = target_session.approval_queues.get(tool_call_id)
  if approval_queue is None:
    raise ApprovalActionError(404, {"error": "Missing approval queue for tool call"})

  return approval_queue


async def _record_vote_and_unblock(
  *,
  target_session: GatewaySession,
  pending_entry: dict[str, Any],
  tool_call_id: str,
  nonce: str,
  decider_id: str,
  decider_role: str | None,
  approved: bool,
  allow_tool_type: bool,
  reason: str | None,
  app_state: Any,
  denied_by: str | None = None,
  authoritative_store: Any | None = None,
  authoritative_policy: Any | None = None,
  authoritative_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  lock = _approval_resolution_lock(target_session, tool_call_id)
  async with lock:
    return await _record_vote_and_unblock_locked(
      target_session=target_session,
      pending_entry=pending_entry,
      tool_call_id=tool_call_id,
      nonce=nonce,
      decider_id=decider_id,
      decider_role=decider_role,
      approved=approved,
      allow_tool_type=allow_tool_type,
      reason=reason,
      app_state=app_state,
      denied_by=denied_by,
      authoritative_store=authoritative_store,
      authoritative_policy=authoritative_policy,
      authoritative_identity=authoritative_identity,
    )


async def _record_vote_and_unblock_locked(
  *,
  target_session: GatewaySession,
  pending_entry: dict[str, Any],
  tool_call_id: str,
  nonce: str,
  decider_id: str,
  decider_role: str | None,
  approved: bool,
  allow_tool_type: bool,
  reason: str | None,
  app_state: Any,
  denied_by: str | None = None,
  authoritative_store: Any | None = None,
  authoritative_policy: Any | None = None,
  authoritative_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  approval_queue = _validate_pending_entry(
    pending_entry=pending_entry,
    tool_call_id=tool_call_id,
    nonce=nonce,
    target_session=target_session,
  )
  approval_id = pending_entry.get("approval_id")
  store, policy = _resolve_store_and_policy(
    target_session=target_session,
    app_state=app_state,
    authoritative_store=authoritative_store,
    authoritative_policy=authoritative_policy,
  )
  effective_denied_by = denied_by if not approved else None

  if not approval_id:
    pending_entry["status"] = "approval_received"
    await approval_queue.put({"approved": approved, "allow_tool_type": allow_tool_type, "denied_by": effective_denied_by})
    return {"status": "ok"}

  if store is None or policy is None:
    raise ApprovalActionError(503, {"error": "Approval subsystem unavailable", "approval_id": str(approval_id)})

  request_record = await store.get(str(approval_id))
  if request_record is None:
    raise ApprovalActionError(404, {"error": "Approval request not found", "approval_id": str(approval_id)})
  if authoritative_identity is not None and not _approval_matches_authoritative_identity(
    request_record,
    authoritative_identity,
  ):
    raise ApprovalActionError(404, {"error": "Approval request not found", "approval_id": str(approval_id)})

  if request_record.state == "expired":
    raise ApprovalActionError(410, {"error": "Approval request expired", "approval_id": str(approval_id)})

  if request_record.state in TERMINAL_APPROVAL_STATES:
    raise ApprovalActionError(409, {"error": "Approval request already resolved", "approval_id": str(approval_id)})

  if request_record.state != "pending_user":
    raise ApprovalActionError(
      409,
      {
        "error": "Invalid approval request state for approval submission",
        "approval_id": str(approval_id),
        "state": request_record.state,
      },
    )

  if approved and request_record.approval_constraint == "legacy_unknown":
    raise ApprovalActionError(
      409,
      {
        "error": "Approval constraint is unknown; replan and reauthorize",
        "approval_id": str(approval_id),
      },
    )
  if approved and request_record.approval_constraint == "fresh_human_owner":
    if (
      decider_id != request_record.required_owner_user_id
      or decider_role != "owner"
    ):
      raise ApprovalActionError(
        403,
        {
          "error": "Exact promotion requires approval by its frozen owner",
          "approval_id": str(approval_id),
        },
      )
    allow_tool_type = False

  if approved and not policy.role_authorized_for_class(
    decider_role=decider_role,
    tool_class=request_record.tool_class,
  ):
    raise ApprovalActionError(
      403,
      {
        "error": "Role is not authorized to approve this tool class",
        "approval_id": str(approval_id),
        "tool_class": request_record.tool_class,
      },
    )

  vote = ApprovalVote(
    vote_id=new_approval_id(),
    approval_id=str(approval_id),
    decider_id=decider_id,
    decider_role=decider_role,
    decision="approved" if approved else "denied",
    decision_reason="Auto-denied by relay chat policy" if effective_denied_by == "relay_policy" and reason is None else reason,
    decided_at=utc_now(),
    external_callback_id=None,
  )
  try:
    request_record = await store.record_vote(str(approval_id), vote)
  except ValueError as exc:
    raise ApprovalActionError(
      409,
      {
        "error": "Approval constraint rejected this decision",
        "approval_id": str(approval_id),
      },
    ) from exc
  if _cancellation_requested(pending_entry):
    return {
      "status": "cancellation_pending",
      "approval": _approval_request_to_dict(request_record),
    }
  if request_record.state == "expired":
    raise ApprovalActionError(410, {"error": "Approval request expired", "approval_id": str(approval_id)})
  if request_record.state not in ("approved", "denied"):
    if request_record.state in TERMINAL_APPROVAL_STATES:
      raise ApprovalActionError(409, {"error": "Approval request already resolved", "approval_id": str(approval_id)})
    return {
      "status": "vote_recorded",
      "votes_received_count": request_record.votes_received_count,
      "required_decider_count": request_record.required_decider_count,
      "eligible_decider_count": request_record.eligible_decider_count,
    }

  await policy.on_resolve(request=request_record)
  if _cancellation_requested(pending_entry):
    return {
      "status": "cancellation_pending",
      "approval": _approval_request_to_dict(request_record),
    }
  if (
    request_record.state == "approved"
    and request_record.approval_constraint == "standard"
    and allow_tool_type
    and request_record.persistent_grant_scope
  ):
    try:
      await store.create_persistent_grant(
        PersistentGrant(
          grant_id=new_approval_id(),
          user_id=request_record.user_id,
          tool_name=request_record.tool_name,
          scope_hint=request_record.persistent_grant_scope,
          args_predicate=request_record.args_predicate,
          granted_at=utc_now(),
          expires_at=None,
          revoked_at=None,
          granted_via_approval_id=str(approval_id),
          policy_id=request_record.policy_id,
        )
      )
    except PersistentGrantCancellationFenced:
      if _cancellation_requested(pending_entry):
        return {
          "status": "cancellation_pending",
          "approval": _approval_request_to_dict(request_record),
        }
      raise ApprovalActionError(
        409,
        {
          "error": "Persistent approval grant is fenced for cancellation",
          "approval_id": str(approval_id),
        },
      ) from None

  if _cancellation_requested(pending_entry):
    await _revoke_persistent_grants_for_cancel(store, str(approval_id))
    return {
      "status": "cancellation_pending",
      "approval": _approval_request_to_dict(request_record),
    }
  pending_entry["status"] = "approval_received"
  executable = approval_is_executable(request_record)
  await approval_queue.put(
    {
      "approved": executable,
      "allow_tool_type": (
        request_record.approval_constraint == "standard"
        and allow_tool_type
      ),
      "approval_id": str(approval_id),
      "denied_by": effective_denied_by,
    }
  )
  log.info(
    "Tool approval: %s | approved=%s allow_tool_type=%s approval_id=%s",
    pending_entry.get("tool_name", "?"),
    executable,
    allow_tool_type and request_record.approval_constraint == "standard",
    approval_id,
  )
  return {"approval": _approval_request_to_dict(request_record)}


async def _force_deny_pending_and_unblock(
  *,
  target_session: GatewaySession,
  pending_entry: dict[str, Any],
  tool_call_id: str,
  nonce: str,
  decider_id: str,
  decider_role: str | None,
  reason: str,
  app_state: Any,
) -> dict[str, Any]:
  """Close approval state when its owning runtime is being terminated.

  Runtime teardown is not a quorum vote. It atomically denies a pending durable
  request, then releases the in-memory waiter so no approval can be orphaned.
  """
  approval_queue = _validate_pending_entry(
    pending_entry=pending_entry,
    tool_call_id=tool_call_id,
    nonce=nonce,
    target_session=target_session,
  )
  approval_id = pending_entry.get("approval_id")
  if not approval_id:
    pending_entry["status"] = "approval_received"
    await approval_queue.put({"approved": False, "allow_tool_type": False})
    return {"status": "ok"}

  store, policy = _resolve_store_and_policy(
    target_session=target_session,
    app_state=app_state,
  )
  if store is None or policy is None:
    raise ApprovalActionError(
      503,
      {"error": "Approval subsystem unavailable", "approval_id": str(approval_id)},
    )
  force_deny = getattr(store, "force_deny_pending", None)
  if not callable(force_deny):
    raise ApprovalActionError(
      503,
      {"error": "Approval store does not support runtime teardown", "approval_id": str(approval_id)},
    )
  try:
    request_record, transitioned = await force_deny(
      str(approval_id),
      decider_id=decider_id,
      decider_role=decider_role,
      decision_reason=reason,
    )
  except KeyError as exc:
    raise ApprovalActionError(
      404,
      {"error": "Approval request not found", "approval_id": str(approval_id)},
    ) from exc
  except RuntimeError as exc:
    raise ApprovalActionError(
      409,
      {
        "error": "Approval request cannot be terminated",
        "approval_id": str(approval_id),
      },
    ) from exc

  if request_record.state == "expired":
    raise ApprovalActionError(
      410,
      {"error": "Approval request expired", "approval_id": str(approval_id)},
    )
  if request_record.state not in {"denied", "auto_denied"}:
    raise ApprovalActionError(
      409,
      {
        "error": "Approval request resolved before runtime teardown",
        "approval_id": str(approval_id),
        "state": request_record.state,
      },
    )
  if transitioned:
    try:
      await policy.on_resolve(request=request_record)
    except Exception:
      log.warning(
        "Approval teardown policy callback failed for %s",
        approval_id,
        exc_info=True,
      )
  pending_entry["status"] = "approval_received"
  await approval_queue.put(
    {
      "approved": False,
      "allow_tool_type": False,
      "approval_id": str(approval_id),
      "denied_by": "runtime_teardown",
    }
  )
  return {"approval": _approval_request_to_dict(request_record)}


async def _cancel_pending_approval_and_unblock(
  *,
  target_session: GatewaySession,
  pending_entry: dict[str, Any],
  tool_call_id: str,
  nonce: str,
  decider_id: str,
  decider_role: str | None,
  reason: str,
  app_state: Any,
  authoritative_store: Any,
  authoritative_policy: Any,
  expected_owner_user_id: str,
  expected_request_id: str,
  expected_run_id: str,
  expected_session_id: str,
  expected_channel: str | None,
  system_override: bool = False,
  role_authorization_prechecked: bool = False,
  release_queue: bool = True,
) -> dict[str, Any]:
  """Fail closed at a run-cancellation boundary and release only denial."""

  pending_entry["_approval_cancel_requested"] = True
  approval_id = str(pending_entry.get("approval_id") or "").strip()
  if not approval_id:
    raise ApprovalActionError(409, {"error": "Approval cancellation identity unavailable"})
  if pending_entry.get("nonce") != nonce:
    raise ApprovalActionError(409, {"error": "Nonce mismatch"})
  _ = app_state
  store = authoritative_store
  policy = authoritative_policy
  if store is None or policy is None:
    raise ApprovalActionError(
      503,
      {"error": "Approval subsystem unavailable", "approval_id": approval_id},
    )
  fence_grants = getattr(store, "fence_persistent_grants_for_cancellation", None)
  if not callable(fence_grants):
    raise ApprovalActionError(
      503,
      {
        "error": "Approval cancellation grant quarantine unavailable",
        "approval_id": approval_id,
      },
    )
  try:
    _, fence_identity_matches = await fence_grants(
      approval_id,
      expected_tool_call_id=tool_call_id,
      expected_user_id=expected_owner_user_id,
      expected_request_id=expected_request_id,
      expected_run_id=expected_run_id,
      expected_session_id=expected_session_id,
      expected_channel=expected_channel,
    )
  except KeyError as exc:
    raise ApprovalActionError(
      404,
      {"error": "Approval request not found", "approval_id": approval_id},
    ) from exc
  if not fence_identity_matches:
    if release_queue:
      _release_cancelled_approval(
        target_session=target_session,
        pending_entry=pending_entry,
        tool_call_id=tool_call_id,
        approval_id=approval_id,
      )
    return {"identity_matches": False, "quarantined": True}

  lock = _approval_resolution_lock(target_session, tool_call_id)
  async with lock:
    terminalize = getattr(store, "terminalize_pending_for_cancellation", None)
    if not callable(terminalize):
      raise ApprovalActionError(
        503,
        {"error": "Approval cancellation terminalization unavailable", "approval_id": approval_id},
      )
    request_record = await store.get(approval_id)
    if request_record is None:
      raise ApprovalActionError(
        404,
        {"error": "Approval request not found", "approval_id": approval_id},
      )
    normalized_expected_channel = str(expected_channel or "").strip().lower()
    identity_matches = all((
      str(getattr(request_record, "tool_call_id", "") or "") == tool_call_id,
      str(getattr(request_record, "user_id", "") or "") == expected_owner_user_id,
      str(getattr(request_record, "request_id", "") or "") == expected_request_id,
      str(getattr(request_record, "run_id", "") or "") == expected_run_id,
      str(getattr(request_record, "session_id", "") or "") == expected_session_id,
      str(getattr(request_record, "channel", "") or "").strip().lower()
      == normalized_expected_channel,
    ))
    if not identity_matches:
      if release_queue:
        _release_cancelled_approval(
          target_session=target_session,
          pending_entry=pending_entry,
          tool_call_id=tool_call_id,
          approval_id=approval_id,
        )
      return {"identity_matches": False, "quarantined": True}
    if not system_override and not role_authorization_prechecked and not policy.role_authorized_for_class(
      decider_role=decider_role,
      tool_class=request_record.tool_class,
    ):
      raise ApprovalActionError(
        403,
        {
          "error": "Role is not authorized to cancel this approval",
          "approval_id": approval_id,
          "tool_class": request_record.tool_class,
        },
      )
    try:
      request_record, transitioned, identity_matches = await terminalize(
        approval_id,
        expected_tool_call_id=tool_call_id,
        expected_user_id=expected_owner_user_id,
        expected_request_id=expected_request_id,
        expected_run_id=expected_run_id,
        expected_session_id=expected_session_id,
        expected_channel=expected_channel,
        decider_id=decider_id,
        decider_role=decider_role,
        decision_reason=reason,
      )
    except asyncio.CancelledError:
      request_record = await store.get(approval_id)
      if request_record is None:
        raise ApprovalActionError(
          404,
          {"error": "Approval request not found", "approval_id": approval_id},
        ) from None
      identity_matches = all((
        str(getattr(request_record, "tool_call_id", "") or "") == tool_call_id,
        str(getattr(request_record, "user_id", "") or "") == expected_owner_user_id,
        str(getattr(request_record, "request_id", "") or "") == expected_request_id,
        str(getattr(request_record, "run_id", "") or "") == expected_run_id,
        str(getattr(request_record, "session_id", "") or "") == expected_session_id,
        str(getattr(request_record, "channel", "") or "").strip().lower()
        == normalized_expected_channel,
      ))
      if not identity_matches or request_record.state not in TERMINAL_APPROVAL_STATES:
        raise
      transitioned = True
    except KeyError as exc:
      raise ApprovalActionError(
        404,
        {"error": "Approval request not found", "approval_id": approval_id},
      ) from exc
    if identity_matches and request_record.state == "pending_user":
      raise ApprovalActionError(
        409,
        {"error": "Approval cancellation did not terminalize", "approval_id": approval_id},
      )
    if identity_matches and request_record.state not in TERMINAL_APPROVAL_STATES:
      raise ApprovalActionError(
        409,
        {
          "error": "Approval cancellation reached an invalid durable state",
          "approval_id": approval_id,
          "state": request_record.state,
        },
      )
    if transitioned and identity_matches:
      try:
        await policy.on_resolve(request=request_record)
      except asyncio.CancelledError:
        log.warning(
          "Approval cancellation resolve hook was cancelled for %s",
          approval_id,
          exc_info=True,
        )
      except Exception:
        log.warning(
          "Approval cancellation resolve hook failed for %s",
          approval_id,
          exc_info=True,
        )
    if identity_matches:
      try:
        await _revoke_persistent_grants_for_cancel(store, approval_id)
      except asyncio.CancelledError:
        raise ApprovalActionError(
          503,
          {
            "error": "Approval cancellation grant cleanup was interrupted",
            "approval_id": approval_id,
          },
        ) from None
      except Exception as exc:
        raise ApprovalActionError(
          503,
          {
            "error": "Approval cancellation grant cleanup failed",
            "approval_id": approval_id,
          },
        ) from exc

    if release_queue:
      _release_cancelled_approval(
        target_session=target_session,
        pending_entry=pending_entry,
        tool_call_id=tool_call_id,
        approval_id=approval_id,
      )
    result: dict[str, Any] = {
      "identity_matches": identity_matches,
      "quarantined": (
        not identity_matches
        or request_record.state not in {"denied", "auto_denied"}
      ),
    }
    if identity_matches:
      result["approval"] = _approval_request_to_dict(request_record)
    return result


__all__ = [
  "ApprovalActionError",
  "_approval_request_to_dict",
  "_force_deny_pending_and_unblock",
  "_cancel_pending_approval_and_unblock",
  "_release_cancelled_approval",
  "_record_vote_and_unblock",
]
