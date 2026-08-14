from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent_gateway.approvals import (
  TERMINAL_APPROVAL_STATES,
  ApprovalActionError,
  _approval_request_to_dict,
  _record_vote_and_unblock,
)
from agent_gateway.autonomous_runner import AutonomousRegistry, AutonomousTask
from agent_gateway.batch_approval_projection import (
  ApprovalProjection,
  approval_record_matches_projection,
)
from agent_gateway.session import AuthManager, GatewaySession

from .runs_helpers import (
  _record_owner_user_id,
  _session_matches_owner,
  _session_owner_user_id,
)
from .autonomous_approval_delivery import (
  autonomous_approval_authoritative_identity,
  autonomous_approval_delivery_context,
  autonomous_decision_unavailable,
  autonomous_run_accepts_approval_decisions,
  deliver_autonomous_approval_outbox,
)
from .autonomous_approval_drainer import (
  wake_autonomous_approval_delivery,
)


class ControlApprovalDecisionRequest(BaseModel):
  approved: bool
  allow_tool_type: bool = False
  reason: str | None = None


def _require_bearer_session(request: Request, auth: AuthManager) -> GatewaySession:
  token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
  return auth.verify_token(token)


def _require_control_session(session: GatewaySession) -> None:
  if session.kind != "control":
    raise HTTPException(status_code=401, detail="Control session required")


def _json_error(status_code: int, message: str) -> JSONResponse:
  return JSONResponse({"error": message}, status_code=status_code)


def _target_chat_session_for_user(
  auth: AuthManager,
  run_id: str,
  user_id: str,
  channel: str | None,
) -> GatewaySession | None:
  session = auth.session_store.get_session(run_id)
  if session is None or session.kind != "chat" or not _session_matches_owner(session, user_id):
    return None
  if not _channel_matches(session.channel, channel):
    return None
  return session


def _normalize_channel(value: str | None) -> str | None:
  normalized = value.strip().lower() if isinstance(value, str) and value.strip() else None
  return normalized


def _channel_matches(record_channel: str | None, authenticated_channel: str | None) -> bool:
  expected = _normalize_channel(record_channel)
  actual = _normalize_channel(authenticated_channel)
  return expected is None or actual == expected


def _autonomous_record_for_user(
  registry: AutonomousRegistry | None,
  run_id: str,
  user_id: str,
  channel: str | None,
) -> AutonomousTask | None:
  if registry is None:
    return None
  record = registry._find_by_control_run_id(run_id)
  if record is None or _record_owner_user_id(record) != user_id:
    return None
  if not _channel_matches(record.channel, channel):
    return None
  return record


def _autonomous_pending_event(record: AutonomousTask, approval_id: str) -> dict[str, Any] | None:
  for event in reversed(record.event_lines or []):
    if event.get("type") != "tool_approval_request":
      continue
    if str(event.get("approval_id") or "") == approval_id:
      return dict(event)
  return None


def _find_pending_approval(session: GatewaySession, approval_id: str) -> tuple[str, dict[str, Any]] | None:
  for tool_call_id, pending in session.pending_tools.items():
    if str(pending.get("approval_id") or "") == approval_id:
      return tool_call_id, pending
  return None


def _delegation_id_for_approval(request_record: Any) -> str | None:
  value = (
    request_record.get("delegation_id")
    if isinstance(request_record, dict)
    else getattr(request_record, "delegation_id", None)
  )
  return str(value) if value else None


def _delegator_user_id_for_grant(grant: Any) -> str | None:
  value = (
    grant.get("delegator_user_id")
    if isinstance(grant, dict)
    else getattr(grant, "delegator_user_id", None)
  )
  return str(value) if value else None


async def _delegation_grant_for_id(
  store: Any,
  delegation_id: str,
  delegation_grant_cache: dict[str, Any | None],
) -> Any | None:
  if delegation_id not in delegation_grant_cache:
    delegation_grant_cache[delegation_id] = await store.get_delegation_grant(delegation_id)
  return delegation_grant_cache[delegation_id]


async def _approval_visible_via_delegation(
  store: Any,
  request_record: Any,
  user_id: str,
  delegation_grant_cache: dict[str, Any | None],
) -> bool:
  delegation_id = _delegation_id_for_approval(request_record)
  if delegation_id is None:
    return False
  grant = await _delegation_grant_for_id(store, delegation_id, delegation_grant_cache)
  return grant is not None and _delegator_user_id_for_grant(grant) == user_id


async def _approval_records_for_session(
  store: Any,
  session: GatewaySession,
  *,
  delegated_to_user_id: str | None = None,
  delegation_grant_cache: dict[str, Any | None] | None = None,
) -> list[dict[str, Any]]:
  approvals: list[dict[str, Any]] = []
  seen: set[str] = set()
  grant_cache = delegation_grant_cache if delegation_grant_cache is not None else {}
  for pending in session.pending_tools.values():
    approval_id = pending.get("approval_id")
    if not approval_id:
      continue
    approval_id_text = str(approval_id)
    if approval_id_text in seen:
      continue
    seen.add(approval_id_text)
    request_record = await store.get(approval_id_text)
    if request_record is not None and request_record.state == "pending_user":
      if delegated_to_user_id is not None and not await _approval_visible_via_delegation(
        store,
        request_record,
        delegated_to_user_id,
        grant_cache,
      ):
        continue
      approvals.append(_approval_request_to_dict(request_record))
  return approvals


async def _delegated_pending_approval_for_user(
  auth: AuthManager,
  store: Any,
  run_id: str,
  approval_id: str,
  user_id: str,
  channel: str | None,
  delegation_grant_cache: dict[str, Any | None],
) -> tuple[GatewaySession, str, dict[str, Any]] | None:
  session = auth.session_store.get_session(run_id)
  if session is None or session.kind != "chat" or not _session_matches_owner(session, user_id):
    return None
  if _channel_matches(session.channel, channel):
    return None
  pending = _find_pending_approval(session, approval_id)
  if pending is None:
    return None
  request_record = await store.get(approval_id)
  if request_record is None:
    return None
  if not await _approval_visible_via_delegation(store, request_record, user_id, delegation_grant_cache):
    return None
  tool_call_id, pending_entry = pending
  return session, tool_call_id, pending_entry


async def _approval_records_for_autonomous_task(store: Any, record: AutonomousTask) -> list[dict[str, Any]]:
  if not autonomous_run_accepts_approval_decisions(record):
    return []
  approvals: list[dict[str, Any]] = []
  seen: set[str] = set()
  for event in record.event_lines or []:
    if event.get("type") != "tool_approval_request":
      continue
    approval_id = event.get("approval_id")
    if not approval_id:
      continue
    approval_id_text = str(approval_id)
    if approval_id_text in seen:
      continue
    seen.add(approval_id_text)
    request_record = await store.get(approval_id_text)
    if request_record is not None and request_record.state == "pending_user":
      tool_call_id = str(event.get("tool_call_id") or "")
      if not tool_call_id:
        continue
      expected_identity = autonomous_approval_authoritative_identity(
        record,
        approval_id=approval_id_text,
        tool_call_id=tool_call_id,
      )
      if any(
        getattr(request_record, field_name, None) != expected
        for field_name, expected in expected_identity.items()
      ):
        continue
      approval = _approval_request_to_dict(request_record)
      approval["session_id"] = record.session_id
      approval["run_id"] = record.control_run_id
      approvals.append(approval)
  return approvals


def _batch_projection_for_user(
  batch_task_registry: Any,
  *,
  run_id: str,
  approval_id: str,
  owner_user_id: str,
  channel: str | None,
) -> ApprovalProjection | None:
  projections = getattr(batch_task_registry, "approval_projections", None)
  if projections is None:
    return None
  return projections.find_projection(
    run_id=run_id,
    approval_id=approval_id,
    owner_user_id=owner_user_id,
    channel=channel,
  )


async def _validated_batch_projection(
  projection: ApprovalProjection | None,
) -> tuple[ApprovalProjection, Any] | None:
  if projection is None or projection.store is None or projection.policy is None:
    return None
  request_record = await projection.store.get(projection.approval_id)
  if request_record is None or not approval_record_matches_projection(
    request_record,
    projection,
  ):
    return None
  return projection, request_record


async def _approval_records_for_batches(
  batch_task_registry: Any,
  *,
  owner_user_id: str,
  channel: str | None,
) -> list[dict[str, Any]]:
  projections = getattr(batch_task_registry, "approval_projections", None)
  if projections is None:
    return []
  approvals: list[dict[str, Any]] = []
  for projection in projections.projections_for_owner(
    owner_user_id=owner_user_id,
    channel=channel,
  ):
    validated = await _validated_batch_projection(projection)
    if validated is None:
      continue
    _, request_record = validated
    if request_record.state != "pending_user":
      continue
    approval = _approval_request_to_dict(request_record)
    approval["session_id"] = projection.run_id
    approval["run_id"] = projection.run_id
    approval["batch_id"] = projection.batch_id
    approval["stage_run_seq"] = projection.stage_run_seq
    approvals.append(approval)
  return approvals


async def _visible_pending_approval_for_run(
  *,
  auth: AuthManager,
  store: Any | None,
  autonomous_registry: AutonomousRegistry | None,
  run_id: str,
  approval_id: str,
  authenticated: GatewaySession,
  batch_task_registry: Any = None,
) -> tuple[Any, Any] | None:
  owner_user_id = _session_owner_user_id(authenticated)
  target_session = _target_chat_session_for_user(auth, run_id, owner_user_id, authenticated.channel)
  if target_session is not None and store is not None:
    pending = _find_pending_approval(target_session, approval_id)
    if pending is not None:
      request_record = await store.get(approval_id)
      return (store, request_record) if request_record is not None else None

  if store is not None:
    delegated_target = await _delegated_pending_approval_for_user(
      auth,
      store,
      run_id,
      approval_id,
      owner_user_id,
      authenticated.channel,
      {},
    )
    if delegated_target is not None:
      request_record = await store.get(approval_id)
      return (store, request_record) if request_record is not None else None

  batch_projection = _batch_projection_for_user(
    batch_task_registry,
    run_id=run_id,
    approval_id=approval_id,
    owner_user_id=owner_user_id,
    channel=authenticated.channel,
  )
  validated = await _validated_batch_projection(batch_projection)
  if validated is not None:
    projection, request_record = validated
    return projection.store, request_record

  record = _autonomous_record_for_user(
    autonomous_registry,
    run_id,
    owner_user_id,
    authenticated.channel,
  )
  if record is None:
    return None
  if not autonomous_run_accepts_approval_decisions(record):
    return None
  if _autonomous_pending_event(record, approval_id) is None:
    return None
  if store is None:
    return None
  request_record = await store.get(approval_id)
  return (store, request_record) if request_record is not None else None


def _approval_retry_state_error(request_record: Any, approval_id: str) -> JSONResponse | None:
  state = getattr(request_record, "state", None)
  if state == "expired":
    return _json_error(410, "Approval request expired")
  if state in TERMINAL_APPROVAL_STATES:
    return _json_error(409, "Approval request already resolved")
  if state != "pending_user":
    return JSONResponse(
      {
        "error": "Invalid approval request state for notification retry",
        "approval_id": approval_id,
        "state": state,
      },
      status_code=409,
    )
  return None


def build_approvals_router(
  *,
  auth: AuthManager,
  autonomous_registry: AutonomousRegistry | None = None,
) -> APIRouter:
  router = APIRouter()

  @router.get("/approvals")
  async def list_approvals(request: Request) -> JSONResponse:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    owner_user_id = _session_owner_user_id(authenticated)
    auth.session_store.cleanup_expired()
    store = getattr(request.app.state, "gateway_approval_store", None)

    approvals: list[dict[str, Any]] = []
    delegation_grant_cache: dict[str, Any | None] = {}
    if store is not None:
      for session in auth.session_store.sessions.values():
        if session.kind != "chat" or not _session_matches_owner(session, owner_user_id):
          continue
        if _channel_matches(session.channel, authenticated.channel):
          approvals.extend(await _approval_records_for_session(store, session))
          continue
        approvals.extend(
          await _approval_records_for_session(
            store,
            session,
            delegated_to_user_id=owner_user_id,
            delegation_grant_cache=delegation_grant_cache,
          )
        )
    if store is not None and autonomous_registry is not None:
      for record in autonomous_registry._tasks.values():
        if _record_owner_user_id(record) != owner_user_id:
          continue
        if not _channel_matches(record.channel, authenticated.channel):
          continue
        approvals.extend(await _approval_records_for_autonomous_task(store, record))
    approvals.extend(
      await _approval_records_for_batches(
        getattr(request.app.state, "batch_task_registry", None),
        owner_user_id=owner_user_id,
        channel=authenticated.channel,
      )
    )
    if store is None and not approvals:
      return _json_error(503, "Approval subsystem unavailable")
    return JSONResponse({"approvals": approvals})

  @router.post("/runs/{run_id}/approvals/{approval_id}/notifications/retry")
  async def retry_approval_notification(request: Request, run_id: str, approval_id: str) -> JSONResponse:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    store = getattr(request.app.state, "gateway_approval_store", None)
    visible = await _visible_pending_approval_for_run(
      auth=auth,
      store=store,
      autonomous_registry=autonomous_registry,
      run_id=run_id,
      approval_id=approval_id,
      authenticated=authenticated,
      batch_task_registry=getattr(request.app.state, "batch_task_registry", None),
    )
    if visible is None:
      if store is None:
        return _json_error(503, "Approval subsystem unavailable")
      return _json_error(404, "Run approval not found")
    authoritative_store, request_record = visible

    state_error = _approval_retry_state_error(request_record, approval_id)
    if state_error is not None:
      return state_error

    retry_failed = getattr(authoritative_store, "retry_failed_approval_notifications", None)
    if retry_failed is None:
      return _json_error(503, "Approval notification retry unavailable")
    retry_result = await retry_failed(approval_id)
    requeued = int(retry_result.get("requeued") or 0)
    notification = retry_result.get("notification")
    notification_state = notification.get("state") if isinstance(notification, dict) else None
    queueable = requeued > 0 or notification_state == "pending"
    delivery_scheduled = False
    if queueable:
      schedule_delivery = getattr(authoritative_store, "schedule_approval_notification_delivery", None)
      if schedule_delivery is not None:
        delivery_scheduled = bool(schedule_delivery())
    return JSONResponse(
      {
        "status": "queued" if queueable else "not_retryable",
        "approval_id": approval_id,
        "requeued": requeued,
        "delivery_scheduled": delivery_scheduled,
        "notification": notification,
      }
    )

  @router.post("/runs/{run_id}/approvals/{approval_id}")
  async def resolve_approval(
    request: Request,
    run_id: str,
    approval_id: str,
    payload: ControlApprovalDecisionRequest,
  ) -> JSONResponse:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    owner_user_id = _session_owner_user_id(authenticated)
    target_session = _target_chat_session_for_user(auth, run_id, owner_user_id, authenticated.channel)
    batch_request_record = None
    batch_projection = None
    if target_session is None:
      projection = _batch_projection_for_user(
        getattr(request.app.state, "batch_task_registry", None),
        run_id=run_id,
        approval_id=approval_id,
        owner_user_id=owner_user_id,
        channel=authenticated.channel,
      )
      validated = await _validated_batch_projection(projection)
      if validated is not None:
        batch_projection, batch_request_record = validated
        target_session = batch_projection.session
    delegated_pending: tuple[str, dict[str, Any]] | None = None
    if target_session is None:
      store = getattr(request.app.state, "gateway_approval_store", None)
      if store is not None:
        delegated_target = await _delegated_pending_approval_for_user(
          auth,
          store,
          run_id,
          approval_id,
          owner_user_id,
          authenticated.channel,
          {},
        )
        if delegated_target is not None:
          target_session, tool_call_id, pending_entry = delegated_target
          delegated_pending = (tool_call_id, pending_entry)
      if target_session is None:
        record = _autonomous_record_for_user(
          autonomous_registry,
          run_id,
          owner_user_id,
          authenticated.channel,
        )
        if record is None:
          return _json_error(404, "Run approval not found")

        pending_event = _autonomous_pending_event(record, approval_id)
        if pending_event is None:
          return _json_error(404, "Run approval not found")
        tool_call_id = str(pending_event.get("tool_call_id") or "")
        nonce = str(pending_event.get("nonce") or "")
        if not tool_call_id or not nonce:
          return _json_error(409, "Approval request is missing routing metadata")
        if store is None:
          return _json_error(
            503,
            "Autonomous approval delivery outbox unavailable",
          )
        get_delivery = getattr(
          store,
          "get_autonomous_approval_delivery",
          None,
        )
        if not callable(get_delivery):
          return _json_error(
            503,
            "Autonomous approval delivery outbox unavailable",
          )
        existing_delivery = await get_delivery(
          approval_id,
          tool_call_id=tool_call_id,
          nonce=nonce,
        )
        if existing_delivery is not None:
          wake_autonomous_approval_delivery(request.app.state)
          request_record = await store.get(approval_id)
          if request_record is None:
            return _json_error(
              409,
              "Autonomous approval delivery has no durable request",
            )
          try:
            await deliver_autonomous_approval_outbox(
              registry=autonomous_registry,
              store=store,
              record=record,
              request_record=request_record,
              delivery=existing_delivery,
              approval_id=approval_id,
              tool_call_id=tool_call_id,
              nonce=nonce,
              approved=payload.approved,
              user_id=owner_user_id,
              channel=authenticated.channel,
            )
          except PermissionError:
            return _json_error(404, "Run approval not found")
          except OSError as exc:
            return _json_error(
              409,
              f"Autonomous approval inbox unavailable: {exc}",
            )
          except (RuntimeError, ValueError) as exc:
            return _json_error(409, str(exc))
          return JSONResponse({
            "approval": _approval_request_to_dict(request_record),
          })
        unavailable = autonomous_decision_unavailable(record)
        if unavailable is not None:
          return _json_error(409, unavailable)
        pending_entry = {
          "approval_id": approval_id,
          "nonce": nonce,
          "requested_at": pending_event.get("ts") or int(time.time()),
          "status": "approval_pending",
          "tool_name": pending_event.get("tool_name"),
          "tool_input": dict(pending_event.get("tool_input") or {}),
          "resolved_qualifier": pending_event.get("resolved_qualifier"),
          "reason": pending_event.get("reason"),
          "allow_persistent_approval": bool(pending_event.get("allow_persistent_approval", False)),
        }
        approval_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        shim_session = GatewaySession(
          session_id=record.session_id,
          api_key_hash="control-autonomous",
          created_at=int(record.started_at),
          expires_at=int(time.time()) + 600,
          user_id=owner_user_id,
          user_email=record.user_email,
          role=authenticated.role,
          kind="chat",
          channel=record.channel,
        )
        shim_session.owner_user_id = owner_user_id
        shim_session.raw_user_id = getattr(record, "raw_user_id", None) or record.user_id
        shim_session.approval_store = getattr(request.app.state, "gateway_approval_store", None)
        shim_session.approval_policy = getattr(request.app.state, "gateway_approval_policy", None)
        shim_session.pending_tools[tool_call_id] = pending_entry
        shim_session.approval_queues[tool_call_id] = approval_queue
        try:
          result = await _record_vote_and_unblock(
            target_session=shim_session,
            pending_entry=pending_entry,
            tool_call_id=tool_call_id,
            nonce=nonce,
            decider_id=_session_owner_user_id(authenticated),
            decider_role=getattr(authenticated, "role", None),
            approved=payload.approved,
            allow_tool_type=payload.allow_tool_type,
            reason=payload.reason,
            app_state=request.app.state,
            authoritative_store=store,
            authoritative_policy=getattr(
              request.app.state,
              "gateway_approval_policy",
              None,
            ),
            authoritative_identity=autonomous_approval_authoritative_identity(
              record,
              approval_id=approval_id,
              tool_call_id=tool_call_id,
            ),
            autonomous_delivery=autonomous_approval_delivery_context(
              record,
              tool_call_id=tool_call_id,
              nonce=nonce,
            ),
          )
        except ApprovalActionError as exc:
          return JSONResponse(exc.payload, status_code=exc.status_code)
        finally:
          wake_autonomous_approval_delivery(request.app.state)
        if "approval" in result:
          delivery = await get_delivery(
            approval_id,
            tool_call_id=tool_call_id,
            nonce=nonce,
          )
          request_record = await store.get(approval_id)
          if delivery is None or request_record is None:
            return _json_error(
              503,
              "Autonomous approval delivery outbox commit unavailable",
            )
          try:
            await deliver_autonomous_approval_outbox(
              registry=autonomous_registry,
              store=store,
              record=record,
              request_record=request_record,
              delivery=delivery,
              approval_id=approval_id,
              tool_call_id=tool_call_id,
              nonce=nonce,
              approved=payload.approved,
              user_id=owner_user_id,
              channel=authenticated.channel,
            )
          except PermissionError:
            return _json_error(404, "Run approval not found")
          except OSError as exc:
            return _json_error(409, f"Autonomous approval inbox unavailable: {exc}")
          except (RuntimeError, ValueError) as exc:
            return _json_error(409, str(exc))
        return JSONResponse(result)

    if delegated_pending is None:
      pending = _find_pending_approval(target_session, approval_id)
      if pending is None:
        return _json_error(404, "Run approval not found")
      tool_call_id, pending_entry = pending
    else:
      tool_call_id, pending_entry = delegated_pending

    if batch_request_record is not None:
      if batch_request_record.state == "expired":
        return _json_error(410, "Approval request expired")
      if batch_request_record.state in TERMINAL_APPROVAL_STATES:
        return _json_error(409, "Approval request already resolved")
      if batch_request_record.state != "pending_user":
        return _json_error(409, "Invalid approval request state for approval submission")

    try:
      result = await _record_vote_and_unblock(
        target_session=target_session,
        pending_entry=pending_entry,
        tool_call_id=tool_call_id,
        nonce=str(pending_entry.get("nonce") or ""),
        decider_id=_session_owner_user_id(authenticated),
        decider_role=getattr(authenticated, "role", None),
        approved=payload.approved,
        allow_tool_type=payload.allow_tool_type,
        reason=payload.reason,
        app_state=request.app.state,
        authoritative_store=(
          batch_projection.store if batch_projection is not None else None
        ),
        authoritative_policy=(
          batch_projection.policy if batch_projection is not None else None
        ),
        authoritative_identity=(
          {
            "approval_id": batch_projection.approval_id,
            "tool_call_id": batch_projection.tool_call_id,
            "user_id": batch_projection.owner_user_id,
            "request_id": batch_projection.run_id,
            "run_id": batch_projection.run_id,
            "session_id": batch_projection.session_id,
            "channel": batch_projection.channel,
          }
          if batch_projection is not None
          else None
        ),
      )
    except ApprovalActionError as exc:
      return JSONResponse(exc.payload, status_code=exc.status_code)
    return JSONResponse(result)

  return router


__all__ = [
  "ControlApprovalDecisionRequest",
  "build_approvals_router",
]
