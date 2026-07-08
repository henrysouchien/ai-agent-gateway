from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any

from fastapi.encoders import jsonable_encoder

from .approval_policy import ApprovalVote, PersistentGrant, new_approval_id, utc_now
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


def _resolve_store_and_policy(*, target_session: GatewaySession, app_state: Any) -> tuple[Any | None, Any | None]:
  store = getattr(target_session, "approval_store", None) or getattr(app_state, "gateway_approval_store", None)
  policy = getattr(target_session, "approval_policy", None) or getattr(app_state, "gateway_approval_policy", None)
  return store, policy


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
) -> dict[str, Any]:
  approval_queue = _validate_pending_entry(
    pending_entry=pending_entry,
    tool_call_id=tool_call_id,
    nonce=nonce,
    target_session=target_session,
  )
  approval_id = pending_entry.get("approval_id")
  store, policy = _resolve_store_and_policy(target_session=target_session, app_state=app_state)
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
  request_record = await store.record_vote(str(approval_id), vote)
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
  if request_record.state == "approved" and allow_tool_type and request_record.persistent_grant_scope:
    await store.create_persistent_grant(
      PersistentGrant(
        grant_id=new_approval_id(),
        user_id=target_session.user_id,
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

  pending_entry["status"] = "approval_received"
  await approval_queue.put(
    {
      "approved": request_record.state == "approved",
      "allow_tool_type": allow_tool_type,
      "approval_id": str(approval_id),
      "denied_by": effective_denied_by,
    }
  )
  log.info(
    "Tool approval: %s | approved=%s allow_tool_type=%s approval_id=%s",
    pending_entry.get("tool_name", "?"),
    request_record.state == "approved",
    allow_tool_type,
    approval_id,
  )
  return {"approval": _approval_request_to_dict(request_record)}


__all__ = [
  "ApprovalActionError",
  "_approval_request_to_dict",
  "_record_vote_and_unblock",
]
