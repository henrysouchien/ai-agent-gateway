from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Callable

from fastapi import APIRouter, Body, HTTPException, Query, Request, Response

from agent_gateway.approval_route import bind_session_approval_route
from agent_gateway.approvals import ApprovalActionError, _record_vote_and_unblock
from agent_gateway.autonomous_runner import AutonomousRegistry, AutonomousTask
from agent_gateway.control_run_lifecycle import is_control_run_active_state
from agent_gateway.control_skill_catalog import (
  ControlSkillCatalog,
  ControlSkillUnavailableError,
)
from agent_gateway.session import AuthManager
from agent_gateway.named_refusal import NamedRefusal


from .runs_helpers import (  # noqa: F401
  ChatRunResponse,
  AutonomousRunResponse,
  DispatchScope,
  ChatMessage,
  ChatDispatchRequest,
  AutonomousDispatchRequest,
  ControlRunDispatchRequest,
  AutonomousDispatchResponse,
  ChatDispatchResponse,
  RunDispatchResponse,
  RunResponse,
  RunsListResponse,
  RunLogsResponse,
  ChatContinuationRequest,
  AutonomousRunMessageRequest,
  AutonomousResumeRequest,
  RunMessageRequest,
  RunEnvelopeResponse,
  _require_bearer_session,
  _autonomous_state,
  _chat_run_from_session,
  _chat_session_has_run_activity,
  _deny_autonomous_pending_approvals_for_cancel,
  _autonomous_task_resumable,
  _autonomous_run_from_task,
  _chat_session_for_user,
  _message_delivery_target,
  _require_autonomous_registry,
  _autonomous_task_for_user,
  _record_owner_user_id,
  _render_log_line,
  _session_matches_owner,
  _session_owner_user_id,
  _normalize_channel,
  _run_channel_matches,
  _require_run_channel,
  _require_autonomous_channel,
)
from .runs_resume_helpers import (  # noqa: F401
  _build_autonomous_resume_context,
)
from .runs_chat_helpers import (
  _require_control_session,
  _require_chat_session_for_run,
  _run_state_event,
  _latest_user_message_content,
  _control_message_id,
  _has_parent_message_event,
  _publish_control_event,
  _cleanup_run_buffer,
  _cancel_control_chat_background_tasks,
  cleanup_control_chat_tasks,
  _dispatch_control_chat_turn,
)

def build_runs_router(
  *,
  auth: AuthManager,
  skill_catalog: ControlSkillCatalog,
  autonomous_registry: AutonomousRegistry | None = None,
  dispatch_scope_validator: Any | None = None,
  control_profile_loader: Callable[[str], Any] | None = None,
) -> APIRouter:
  if not isinstance(skill_catalog, ControlSkillCatalog):
    raise TypeError("skill_catalog must implement ControlSkillCatalog")
  router = APIRouter(prefix="/runs")

  def _require_autonomous_profile(profile_name: str) -> None:
    if control_profile_loader is None:
      return
    try:
      profile = control_profile_loader(profile_name)
    except Exception as exc:
      raise HTTPException(
        status_code=422,
        detail=f"Unknown autonomous profile: {profile_name}",
      ) from exc
    if not profile.supports_autonomous_execution:
      raise HTTPException(
        status_code=422,
        detail=f"Profile {profile_name!r} is interactive-only",
      )

  def _require_dispatchable_skill(skill_name: str) -> None:
    """Answer the launch question before a run id is charged.

    The catalog owns launchability: ``can_launch`` is its verdict for an
    ordinary control dispatch, and ``blocked_reason`` is the definition's own
    sentence. ``catalog: false`` withholds a skill from the advertised
    listing and never from this question, so every skill this catalog
    defines is judged here before the run is charged.

    Whether a name exists at all is not this catalog's authority — the
    runner resolves the skill it launches — so an unresolvable selector
    carries no verdict and is dispatched unchanged.
    """

    try:
      skill = skill_catalog.resolve_skill(skill_name)
    except ControlSkillUnavailableError:
      return
    if skill.can_launch:
      return
    raise NamedRefusal(
      "skill_autonomous_run_not_allowed",
      skill.blocked_reason
      or f"Skill '{skill.name}' cannot be launched by the control plane.",
      transport="invalid",
    )

  def _dispatch_scope_payload(scope: DispatchScope | None) -> dict[str, Any] | None:
    if scope is None:
      return None
    return scope.model_dump()

  async def _validated_dispatch_scope_payload(
    scope: DispatchScope | None,
    *,
    session: Any,
  ) -> dict[str, Any] | None:
    payload = _dispatch_scope_payload(scope)
    if payload is None or dispatch_scope_validator is None:
      return payload
    try:
      validation_result = dispatch_scope_validator(session, dict(payload))
      if inspect.isawaitable(validation_result):
        validation_result = await validation_result
    except HTTPException:
      raise
    except ValueError as exc:
      raise HTTPException(
        status_code=422,
        detail={
          "error": "dispatch_scope_validation_failed",
          "message": str(exc) or "Selected dispatch scope is not valid.",
        },
      ) from exc
    except Exception as exc:
      raise HTTPException(
        status_code=422,
        detail={
          "error": "dispatch_scope_validation_failed",
          "message": "Selected dispatch scope could not be validated.",
        },
      ) from exc
    if validation_result is None:
      validation_result = payload
    if not isinstance(validation_result, dict):
      raise HTTPException(
        status_code=422,
        detail={
          "error": "dispatch_scope_validation_failed",
          "message": "Dispatch scope validator returned an invalid payload.",
        },
      )
    try:
      return DispatchScope.model_validate(validation_result).model_dump()
    except ValueError as exc:
      raise HTTPException(
        status_code=422,
        detail={
          "error": "dispatch_scope_validation_failed",
          "message": "Dispatch scope validator returned a non-redacted scope.",
        },
      ) from exc

  @router.post("", response_model=RunDispatchResponse)
  async def dispatch_run(
    request: Request,
    payload: ControlRunDispatchRequest,
    http_response: Response,
  ) -> RunDispatchResponse:
    authenticated = _require_bearer_session(request, auth)
    owner_user_id = _session_owner_user_id(authenticated)
    if payload.kind == "chat":
      _require_control_session(authenticated)
      requested_channel = _normalize_channel(payload.channel)
      session_channel = _normalize_channel(authenticated.channel)
      if session_channel is not None and requested_channel is not None and session_channel != requested_channel:
        raise HTTPException(status_code=401, detail="Channel mismatch")
      channel = session_channel or requested_channel
      context: dict[str, Any] = dict(payload.context or {})
      if channel is not None:
        context["channel"] = channel
      stage_skill_route: dict[str, str] | None = None
      if payload.skill is not None:
        context["skill"] = payload.skill
        stage_skill_route = {
          "route_kind": "stage",
          "skill_name": payload.skill,
        }
        context["stage_skill_route"] = dict(stage_skill_route)
      if payload.ticker is not None:
        context["ticker"] = payload.ticker
      dispatch_scope = await _validated_dispatch_scope_payload(
        payload.dispatch_scope,
        session=authenticated,
      )
      if dispatch_scope is not None:
        context["dispatch_scope"] = dispatch_scope
        context.setdefault("portfolio_name", dispatch_scope["portfolio_name"])

      chat_session = auth.session_store.create_session(
        api_key_hash=authenticated.api_key_hash,
        user_id=owner_user_id,
        user_email=authenticated.user_email,
        risk_user_id=authenticated.risk_user_id,
        role=authenticated.role,
        kind="chat",
        auth_config=authenticated.auth_config,
        model_entitled_capabilities=authenticated.model_entitled_capabilities,
        model_entitled_keys=authenticated.model_entitled_keys,
        tenant_id=authenticated.tenant_id,
        session_credential_handle=authenticated.session_credential_handle,
        allow_service_for_interactive=(
          authenticated.allow_service_for_interactive
        ),
      )
      chat_session.owner_user_id = owner_user_id
      chat_session.raw_user_id = authenticated.user_id
      chat_session.user_slug = getattr(authenticated, "user_slug", None)
      chat_session.user_aliases = tuple(getattr(authenticated, "user_aliases", ()) or (owner_user_id,))
      chat_session.identity_status = getattr(authenticated, "identity_status", None)
      chat_session.channel = channel
      chat_session.is_public = channel == "public"
      bind_session_approval_route(
        chat_session,
        getattr(request.app.state, "gateway_approval_store", None),
        getattr(request.app.state, "gateway_approval_policy", None),
      )
      chat_session.max_budget_usd = payload.max_budget_usd
      chat_session.initial_message = payload.message
      chat_session.stage_skill_route = (
        dict(stage_skill_route) if stage_skill_route is not None else None
      )
      if dispatch_scope is not None:
        chat_session.dispatch_scope = dispatch_scope

      try:
        run = await _dispatch_control_chat_turn(
          request=request,
          session=chat_session,
          messages=[ChatMessage(role="user", content=payload.message)],
          request_id=None,
          context=context,
          model_key=payload.model_key,
          effort=payload.effort,
          catalog_revision=payload.catalog_revision,
          deadline_sec=payload.deadline_sec,
        )
        response = ChatDispatchResponse(
          run=run,
          chat_session_token=auth.issue_token(chat_session),
          chat_session_id=chat_session.session_id,
          chat_session_expires_at=chat_session.expires_at,
        )
      except BaseException:
        await auth.session_store.expire_session_async(chat_session.session_id)
        raise
      http_response.headers["Cache-Control"] = "private, no-store"
      return response

    _require_control_session(authenticated)
    requested_channel = _normalize_channel(payload.channel)
    session_channel = _normalize_channel(authenticated.channel)
    if session_channel is not None and requested_channel is not None and session_channel != requested_channel:
      raise HTTPException(status_code=401, detail="Channel mismatch")
    channel = session_channel or requested_channel
    if not payload.profile or not payload.mode:
      raise HTTPException(status_code=422, detail="profile and mode are required")
    _require_autonomous_profile(payload.profile)

    registry = _require_autonomous_registry(autonomous_registry)
    registry.set_user_event_bus(getattr(request.app.state, "user_event_bus", None))
    try:
      if payload.skill:
        _require_dispatchable_skill(payload.skill)
      start_payload = await registry.start(
        role=authenticated.role,
        profile=payload.profile,
        mode=payload.mode,
        task=payload.task,
        skill=payload.skill,
        pack=payload.pack,
        context=payload.context,
        ticker=payload.ticker,
        research_file_id=payload.research_file_id,
        max_budget_usd=payload.max_budget_usd,
        channel=channel,
        user_id=authenticated.user_id,
        user_email=authenticated.user_email,
        owner_user_id=owner_user_id,
        user_slug=getattr(authenticated, "user_slug", None),
        risk_user_id=authenticated.risk_user_id,
        user_aliases=list(getattr(authenticated, "user_aliases", ()) or (owner_user_id,)),
        identity_status=getattr(authenticated, "identity_status", None),
        dispatch_scope=await _validated_dispatch_scope_payload(payload.dispatch_scope, session=authenticated),
      )
    except ValueError as exc:
      raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NamedRefusal as exc:
      raise HTTPException(status_code=exc.http_status, detail=str(exc)) from exc
    except RuntimeError as exc:
      raise HTTPException(status_code=500, detail=str(exc)) from exc
    record = _autonomous_task_for_user(registry, str(start_payload["task_id"]), owner_user_id)
    run = _autonomous_run_from_task(record)
    return AutonomousDispatchResponse(
      run=run,
      task_id=record.task_id,
      run_id=record.control_run_id,
      log_path=str(record.log_path),
      started_at=int(record.started_at),
      cmd=list(record.cmd),
    )

  @router.post("/{control_run_id}/messages", response_model=RunEnvelopeResponse)
  async def continue_chat_run(
    request: Request,
    control_run_id: str,
    payload: RunMessageRequest = Body(...),
  ) -> RunEnvelopeResponse:
    authenticated = _require_bearer_session(request, auth)
    owner_user_id = _session_owner_user_id(authenticated)
    delivery_target = _message_delivery_target(
      auth,
      autonomous_registry,
      control_run_id,
      owner_user_id,
    )
    if isinstance(delivery_target, AutonomousTask):
      _require_control_session(authenticated)
      if not isinstance(payload, AutonomousRunMessageRequest):
        raise HTTPException(status_code=422, detail="Autonomous runs require message")
      if (
        not is_control_run_active_state(
          _autonomous_state(delivery_target.state)
        )
        and not _autonomous_task_resumable(delivery_target)
      ):
        raise HTTPException(
          status_code=409,
          detail="Run does not accept additional messages",
        )

      assert autonomous_registry is not None
      autonomous_registry.set_user_event_bus(getattr(request.app.state, "user_event_bus", None))
      try:
        delivery = await autonomous_registry.send_operator_message(
          control_run_id,
          user_id=owner_user_id,
          channel=authenticated.channel,
          message=payload.message,
          message_id=payload.message_id,
        )
      except PermissionError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
      except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
      except NamedRefusal as exc:
        raise HTTPException(status_code=exc.http_status, detail=str(exc)) from exc
      except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

      return RunEnvelopeResponse(
        run=_autonomous_run_from_task(delivery_target),
        message_id=str(delivery.get("message_id") or ""),
        delivery_status=delivery.get("delivery_status"),
      )

    target_session = delivery_target
    _require_chat_session_for_run(authenticated, target_session)
    if not isinstance(payload, ChatContinuationRequest):
      raise HTTPException(status_code=422, detail="Chat runs require messages")

    context = dict(payload.context or {})
    if target_session.channel is not None:
      context["channel"] = target_session.channel
    dispatch_scope = getattr(target_session, "dispatch_scope", None)
    if isinstance(dispatch_scope, dict):
      context["dispatch_scope"] = dict(dispatch_scope)
      portfolio_name = dispatch_scope.get("portfolio_name")
      if isinstance(portfolio_name, str) and portfolio_name.strip():
        context.setdefault("portfolio_name", portfolio_name)
    latest_user_message = _latest_user_message_content(list(payload.messages))
    message_id = _control_message_id(payload.request_id) if latest_user_message is not None else None
    if message_id is not None and _has_parent_message_event(target_session, message_id):
      return RunEnvelopeResponse(
        run=_chat_run_from_session(target_session),
        message_id=message_id,
        delivery_status="duplicate",
      )
    run = await _dispatch_control_chat_turn(
      request=request,
      session=target_session,
      messages=list(payload.messages),
      request_id=message_id or payload.request_id,
      context=context,
      model_key=payload.model_key,
      effort=payload.effort,
      catalog_revision=payload.catalog_revision,
      deadline_sec=payload.deadline_sec,
      record_parent_message=True,
    )
    return RunEnvelopeResponse(
      run=run,
      message_id=message_id,
      delivery_status="delivered" if message_id is not None else None,
    )

  @router.post("/{control_run_id}/resume", response_model=AutonomousDispatchResponse)
  async def resume_autonomous_run(
    request: Request,
    control_run_id: str,
    payload: AutonomousResumeRequest | None = Body(default=None),
  ) -> AutonomousDispatchResponse:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    owner_user_id = _session_owner_user_id(authenticated)

    registry = _require_autonomous_registry(autonomous_registry)
    registry.set_user_event_bus(getattr(request.app.state, "user_event_bus", None))
    record = _autonomous_task_for_user(registry, control_run_id, owner_user_id)
    _require_autonomous_channel(record, authenticated.channel)

    async with record.resume_lock, registry.run_mutation_lock:
      if not _autonomous_task_resumable(record):
        raise HTTPException(status_code=409, detail="Autonomous run is not resumable")

      for resumed_run_id in reversed(record.resumed_as):
        resumed_record = registry._find_by_control_run_id(resumed_run_id)
        resumed_state = _autonomous_run_from_task(resumed_record).state if resumed_record is not None else None
        if is_control_run_active_state(resumed_state):
          raise HTTPException(status_code=409, detail="Autonomous run already has an active resume")

      resume_payload = payload or AutonomousResumeRequest()
      resume_context = _build_autonomous_resume_context(record, resume_payload)
      _require_autonomous_profile(record.profile)
      try:
        start_payload = await registry.start(
          # Resume follows current authority in both promotion and revocation directions.
          role=authenticated.role,
          profile=record.profile,
          mode=record.mode,
          task=record.task,
          skill=record.skill,
          context=resume_context,
          ticker=record.ticker,
          research_file_id=record.research_file_id,
          max_budget_usd=getattr(record, "max_budget_usd", None),
          channel=record.channel,
          user_id=getattr(record, "raw_user_id", None) or authenticated.user_id,
          user_email=authenticated.user_email,
          owner_user_id=owner_user_id,
          user_slug=getattr(authenticated, "user_slug", None) or record.user_slug,
          risk_user_id=authenticated.risk_user_id or record.risk_user_id,
          user_aliases=list(getattr(authenticated, "user_aliases", ()) or record.user_aliases or (owner_user_id,)),
          identity_status=getattr(authenticated, "identity_status", None) or record.identity_status,
          dispatch_scope=record.dispatch_scope,
          resumed_from=record.control_run_id,
        )
      except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
      except NamedRefusal as exc:
        raise HTTPException(status_code=exc.http_status, detail=str(exc)) from exc
      except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

      resumed_record = _autonomous_task_for_user(registry, str(start_payload["task_id"]), owner_user_id)
      record.resumed_as.append(resumed_record.control_run_id)
      resume_event = {
        "type": "run_resumed",
        "run_id": record.control_run_id,
        "control_run_id": record.control_run_id,
        "resumed_run_id": resumed_record.control_run_id,
        "resumed_task_id": resumed_record.task_id,
        "request_id": resume_payload.request_id,
        "ts": int(time.time()),
      }
      resumed_event = {
        "type": "run_resumed_from",
        "run_id": resumed_record.control_run_id,
        "control_run_id": resumed_record.control_run_id,
        "resumed_from": record.control_run_id,
        "resumed_from_task_id": record.task_id,
        "request_id": resume_payload.request_id,
        "ts": int(time.time()),
      }
      await registry._record_and_publish_event(record, resume_event)
      await registry._record_and_publish_event(resumed_record, resumed_event)

      return AutonomousDispatchResponse(
        run=_autonomous_run_from_task(resumed_record),
        task_id=resumed_record.task_id,
        run_id=resumed_record.control_run_id,
        log_path=str(resumed_record.log_path),
        started_at=int(resumed_record.started_at),
        cmd=list(resumed_record.cmd),
        resumed_from=record.control_run_id,
      )

  @router.get("", response_model=RunsListResponse)
  async def list_runs(
    request: Request,
    state: str | None = Query(default=None),
    kind: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
  ) -> RunsListResponse:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    owner_user_id = _session_owner_user_id(authenticated)
    if kind is not None and kind not in {"chat", "autonomous"}:
      return RunsListResponse(runs=[])

    runs: list[RunResponse] = []
    if kind in {None, "chat"}:
      runs.extend(
        _chat_run_from_session(session)
        for session in auth.session_store.visible_sessions_snapshot()
        if (
          session.kind == "chat"
          and _session_matches_owner(session, owner_user_id)
          and _run_channel_matches(session.channel, authenticated.channel)
          and _chat_session_has_run_activity(session)
        )
      )
    if kind in {None, "autonomous"} and autonomous_registry is not None:
      runs.extend(
        _autonomous_run_from_task(record)
        for record in autonomous_registry._tasks.values()
        if _record_owner_user_id(record) == owner_user_id and _run_channel_matches(record.channel, authenticated.channel)
      )
    if state is not None:
      runs = [run for run in runs if run.state == state]
    runs.sort(key=lambda run: run.started_at, reverse=True)
    return RunsListResponse(runs=runs[:limit])

  @router.get("/{control_run_id}", response_model=RunResponse)
  async def get_run(request: Request, control_run_id: str) -> RunResponse:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    owner_user_id = _session_owner_user_id(authenticated)
    if control_run_id.startswith("bg_") or (
      autonomous_registry is not None
      and any(task.control_run_id == control_run_id for task in autonomous_registry._tasks.values())
    ):
      record = _autonomous_task_for_user(autonomous_registry, control_run_id, owner_user_id)
      _require_autonomous_channel(record, authenticated.channel)
      return _autonomous_run_from_task(record)
    session = _chat_session_for_user(auth, control_run_id, owner_user_id)
    _require_run_channel(session.channel, authenticated.channel)
    return _chat_run_from_session(session)

  @router.get("/{control_run_id}/logs", response_model=RunLogsResponse)
  async def get_run_logs(
    request: Request,
    control_run_id: str,
    tail: int = Query(default=200, ge=0, le=5000),
  ) -> RunLogsResponse:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    owner_user_id = _session_owner_user_id(authenticated)
    if control_run_id.startswith("bg_") or (
      autonomous_registry is not None
      and any(task.control_run_id == control_run_id for task in autonomous_registry._tasks.values())
    ):
      registry = _require_autonomous_registry(autonomous_registry)
      record = _autonomous_task_for_user(registry, control_run_id, owner_user_id)
      _require_autonomous_channel(record, authenticated.channel)
      logs = registry.logs(record.task_id, tail=tail)
      total_lines = int(logs.get("total_lines", 0) or 0)
      return RunLogsResponse(
        run_id=record.control_run_id,
        log_lines=list(logs.get("lines", [])),
        more_available=total_lines > tail,
      )
    session = _chat_session_for_user(auth, control_run_id, owner_user_id)
    _require_run_channel(session.channel, authenticated.channel)
    total_events = len(session.event_history)
    events = session.event_history.snapshot(tail=tail)
    return RunLogsResponse(
      run_id=session.session_id,
      log_lines=[_render_log_line(event) for event in events],
      more_available=total_events > tail,
    )

  @router.delete("/{control_run_id}", response_model=RunResponse)
  async def delete_run(request: Request, control_run_id: str) -> RunResponse:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    owner_user_id = _session_owner_user_id(authenticated)
    if not control_run_id.startswith("bg_"):
      session = auth.session_store.get_session(control_run_id)
      if session is not None:
        if session.kind != "chat" or not _session_matches_owner(session, owner_user_id):
          raise HTTPException(status_code=404, detail="Run not found")
        _require_run_channel(session.channel, authenticated.channel)
        pending_snapshot = [
          (tool_call_id, entry)
          for tool_call_id, entry in session.pending_tools.items()
          if entry.get("approval_id")
        ]
        for tool_call_id, pending_entry in pending_snapshot:
          try:
            await _record_vote_and_unblock(
              target_session=session,
              pending_entry=pending_entry,
              tool_call_id=tool_call_id,
              nonce=str(pending_entry.get("nonce") or ""),
              decider_id=owner_user_id,
              decider_role=getattr(authenticated, "role", None),
              approved=False,
              allow_tool_type=False,
              reason="run_cancelled",
              app_state=request.app.state,
            )
          except ApprovalActionError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.payload) from exc

        cancelled_event = _run_state_event(control_run_id, "cancelled")
        session.event_history.append(cancelled_event)
        await _publish_control_event(
          request.app.state,
          session,
          control_run_id,
          cancelled_event,
        )
        await _cancel_control_chat_background_tasks(
          session,
          settle_timeout=0.05 if pending_snapshot else 0.0,
        )
        run = _chat_run_from_session(session)
        await auth.session_store.expire_session_async(control_run_id)
        await _cleanup_run_buffer(
          request.app.state,
          session,
          control_run_id,
        )
        return run
    registry = _require_autonomous_registry(autonomous_registry)
    record = _autonomous_task_for_user(registry, control_run_id, owner_user_id)
    _require_autonomous_channel(record, authenticated.channel)
    approval_error: Exception | None = None
    try:
      await _deny_autonomous_pending_approvals_for_cancel(
        registry=registry,
        record=record,
        authenticated=authenticated,
        app_state=request.app.state,
      )
    except Exception as exc:
      approval_error = exc
    await asyncio.shield(registry.cancel(record.task_id))
    if isinstance(approval_error, ApprovalActionError):
      raise HTTPException(
        status_code=approval_error.status_code,
        detail=approval_error.payload,
      ) from approval_error
    if approval_error is not None:
      raise HTTPException(
        status_code=503,
        detail={
          "error": (
            "Autonomous run was cancelled but pending approval "
            "settlement failed"
          ),
        },
      ) from approval_error
    return _autonomous_run_from_task(record)

  return router


__all__ = [
  "ChatRunResponse",
  "ChatDispatchRequest",
  "ChatDispatchResponse",
  "ChatContinuationRequest",
  "AutonomousDispatchResponse",
  "AutonomousDispatchRequest",
  "AutonomousResumeRequest",
  "AutonomousRunResponse",
  "RunLogsResponse",
  "RunsListResponse",
  "build_runs_router",
  "cleanup_control_chat_tasks",
]
