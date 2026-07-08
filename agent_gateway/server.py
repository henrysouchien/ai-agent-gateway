from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from ._provider_utils import _get_allowed_models_for_provider_name
from .artifact_paths import (
  ArtifactPath as ArtifactPath,
  ArtifactPathError as ArtifactPathError,
  artifact_json_paths_for_request as artifact_json_paths_for_request,
  artifact_json_path_for_request as artifact_json_path_for_request,
  letter_docx_path_for_request as letter_docx_path_for_request,
  reject_unsafe_path as reject_unsafe_path,
  ticker_artifact_paths_for_request as ticker_artifact_paths_for_request,
)
from .auth import (
  ChannelMismatchError,
  CredentialsTimeoutError,
  CrossUserReuseError,
  MissingUserIdError,
)
from .autonomous_runner import AutonomousRegistry
from .control_plane import create_control_plane_router
from .control_plane.middleware import add_control_plane_version_header_middleware
from .control_plane.session import _resolve_control_identity
from .event_log import EventLog, UserEventBus
from .approvals import ApprovalActionError, _record_vote_and_unblock  # noqa: F401
from .approval_resolver import resolve_policy  # noqa: F401 - compatibility alias
from .approval_store import SQLiteApprovalStore, expire_pending_loop  # noqa: F401
from .package_info import package_health
from .providers import StreamEvent
from .session import AuthManager, GatewaySession, SessionStore, StreamSubscriber

from . import server_chat_helpers as _server_chat_helpers  # noqa: F401 - dynamic streaming deps alias
from . import server_chat_control_routes as _server_chat_control_routes
from . import server_artifact_routes as _server_artifact_routes  # noqa: F401 - dynamic artifact route deps alias
from . import server_streaming as _server_streaming
from . import server_tool_routes as _server_tool_routes
from .server_models import (  # noqa: F401
  SystemPrompt,
  ExecutionLocationResolver,
  BuildChatRuntime,
  RequestApproval,
  BuildRunner,
  _AGENT_API_CLAIM_AUDIENCE,
  _AGENT_API_CLAIM_CLOCK_SKEW_SECONDS,
  _AGENT_API_CLAIM_NONCE_HEX_LENGTH,
  _AGENT_API_CLAIM_HEADERS,
  _AGENT_API_CLAIM_MAX_TTL_SECONDS_DEFAULT,
  _ARTIFACT_DOCX_MEDIA_TYPE,
  _ARTIFACT_ORIGIN_VALUES,
  _ARTIFACT_ORIGIN_FILTER_VALUES,
  _ARTIFACT_VISIBILITY_VALUES,
  _ARTIFACT_VISIBILITY_FILTER_VALUES,
  _ARTIFACT_INDEX_RECENT_LIMIT,
  _DEFAULT_CHAT_PROFILE,
  _CHAT_PROFILE_ALIASES,
  _ACTIVE_TURN_GRACE_SECONDS,
  _STREAM_SUBSCRIBER_QUEUE_MAX,
  _STREAM_SUBSCRIBER_KEEPALIVE_SECONDS,
  _SIDECAR_SLUG_RE,
  _STREAM_SUBSCRIBER_DONE,
  ChatInitRequest,
  ModelCatalog,
  ChatInitResponse,
  ChatMessage,
  ChatRequest,
  ChatRecapRequest,
  ChatCancelRequest,
  _resolve_chat_profile_name,
  ChatTurnInputs,
  ChatTurnResult,
  ToolResultRequest,
  ToolApprovalRequest,
  ChatRuntime,
  _build_runner_with_started_at,
  _call_build_chat_runtime,
  RequestContext,
  GatewayServerConfig,
)
from .server_artifact_helpers import (  # noqa: F401
  _model_to_dict,
  _normalize_prefix,
  _route_path,
  _default_control_skills_dir,
  _default_autonomous_api_dir,
  _default_autonomous_log_dir,
  _resolve_compaction_trigger,
  _sanitize_for_json,
  _json_dumps,
  _claim_ttl_ceiling_seconds,
  _verify_signed_user_claim,
  _artifact_auth_dependency,
  _extract_agent_claim_headers,
  _verify_agent_claim_headers,
  _artifact_json_response,
  _artifact_payload_from_path,
  _decorate_artifact_payload,
  _artifact_effective_fields,
  _artifact_research_file_classification,
  _artifact_sidecar_classification,
  _artifact_origin_kind,
  _artifact_origin_kind_filter,
  _artifact_visibility,
  _artifact_visibility_filter,
  _artifact_origin_ref,
  _artifact_request_filters,
  _artifact_payload_matches_filters,
  _int_or_none,
  _artifact_research_file_id_token_present,
  _query_int_or_none,
  _query_str_or_none,
  _assert_artifact_path_still_safe,
  _file_cache_headers,
  _letter_filename,
  _normalize_request_user_id,
  _resolver_contract_payload,
  _error_payload,
)
from .server_chat_helpers import (  # noqa: F401
  _drain_result_queue,
  _now_iso,
  _sidecar_slug,
  _maybe_write_chat_log_meta,
  _write_transcript,
  _cleanup_old_transcripts,
  _compute_session_recap_payload,
  _compute_cumulative_session_recap_payload,
  _redact_tool_input_for_event,
  _cleanup_sessions_loop,
  _maybe_await,
  _cancel_active_turn_cleanup_handle,
  _clear_active_turn,
  _active_turn_is_running,
  _schedule_active_turn_clear,
  _cancel_active_turn_runner,
  _cleanup_active_turn_on_expiry,
  _event_for_wire,
  _resolve_schema_version,
  _stream_envelope,
  _legacy_request_approval,
  _make_request_approval,
  _chat_turn_state_from_events,
  _chat_run_state_event,
  _latest_chat_run_state,
  _dispatch_chat_turn,
  _init_approval_subsystem,
)


(
  _disconnect_stream_subscriber_for_backpressure,
  _pump_stream_subscriber,
  _register_stream_subscriber,
  _cleanup_stream_subscriber,
  _stream_subscriber_sse,
) = _server_streaming.bind_streaming_helpers(lambda: globals())


def create_gateway_app(config: GatewayServerConfig) -> FastAPI:
  """Create a FastAPI gateway application from explicit runtime configuration.

  The returned app exposes:

  - `POST {prefix}/chat/init`
  - `POST {prefix}/chat`
  - `POST {prefix}/chat/tool-result`
  - `POST {prefix}/chat/tool-approval`
  - `POST {prefix}/control/session`
  - `GET  {prefix}/control/health`
  - `GET  {prefix}/control/runs`
  - `GET  {prefix}/control/runs/{run_id}`
  - `GET  {prefix}/control/runs/{run_id}/logs`
  - `POST {prefix}/control/runs/{run_id}/resume`
  - `GET  {prefix}/control/schedules`
  - `GET  {prefix}/control/schedules/{name}`
  - `GET  {prefix}/control/schedules/{name}/logs`
  - `POST {prefix}/control/schedules`
  - `PUT  {prefix}/control/schedules/{name}/enabled`
  - `DELETE {prefix}/control/schedules/{name}`
  - `GET  {prefix}/control/profiles`
  - `GET  {prefix}/control/skills`
  - `GET  {prefix}/control/skills/{name}`
  - `GET  {prefix}/health`

  `create_gateway_app()` is the extensibility point for production deployments.
  Unlike `create_agent()`, it does not assume Anthropic, a particular tool
  policy, or a particular runtime builder. You provide the `build_chat_runtime`
  callback and the server handles the HTTP, session, SSE, and approval loop.

  Args:
    config: `GatewayServerConfig` describing auth, routing, callbacks, and the
      runtime factory.

  Returns:
    A FastAPI application ready to serve the gateway HTTP API.
  """
  if config.build_chat_runtime is None:
    raise ValueError("GatewayServerConfig.build_chat_runtime is required")
  if config.allowed_models is None:
    provider_name = str(getattr(config.default_provider, "name", "anthropic") or "anthropic")
    config.allowed_models = _get_allowed_models_for_provider_name(provider_name)

  auth = config.auth_manager or AuthManager(
    secret=config.jwt_secret,
    valid_keys=config.valid_api_keys,
    session_store=SessionStore(ttl=config.session_ttl),
  )
  from .control_plane.runs import cleanup_control_chat_tasks
  from .control_plane.batches import BatchTaskRegistry

  transcript_dir = Path(config.transcript_dir) if config.transcript_dir is not None else None
  if transcript_dir is not None:
    transcript_dir.mkdir(parents=True, exist_ok=True)

  async def _write_session_gc_recap_on_expiry(session: GatewaySession) -> None:
    active_turn = session.active_turn
    if active_turn is None:
      return
    try:
      await _cancel_active_turn_runner(active_turn)
      if not active_turn.event_log.entries:
        return
      recap_payload = _compute_session_recap_payload(session, active_turn, trigger="session_gc")
      _write_transcript(
        transcript_dir,
        session.session_id,
        recap_payload,
        user_id=session.user_id,
        channel=session.channel,
      )
    except Exception:
      log.warning("session_recap GC write failed for %s", session.session_id, exc_info=True)

  auth.session_store.add_on_expiry(_write_session_gc_recap_on_expiry)
  auth.session_store.add_on_expiry(_cleanup_active_turn_on_expiry)
  auth.session_store.add_on_expiry(cleanup_control_chat_tasks)
  log = logging.getLogger(config.log_name)
  route_prefix = _normalize_prefix(config.prefix)

  @asynccontextmanager
  async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(
      _cleanup_sessions_loop(
        auth.session_store,
        transcript_dir=transcript_dir,
        transcript_retention_days=config.transcript_retention_days,
      )
    )
    approval_expire_task = asyncio.create_task(expire_pending_loop(app.state.gateway_approval_store))
    startup_complete = False
    try:
      await _maybe_await(config.on_startup)
      agent_schedule_runner = getattr(app.state, "agent_run_schedule_runner", None)
      if agent_schedule_runner is not None:
        agent_schedule_runner.start()
      startup_complete = True
      yield
    finally:
      if startup_complete:
        await _maybe_await(config.on_shutdown)
      agent_schedule_runner = getattr(app.state, "agent_run_schedule_runner", None)
      if agent_schedule_runner is not None:
        await agent_schedule_runner.shutdown()
      subprocess_registry = getattr(app.state, "subprocess_registry", None)
      if subprocess_registry is not None:
        await subprocess_registry.shutdown()
      batch_task_registry = getattr(app.state, "batch_task_registry", None)
      if batch_task_registry is not None:
        await batch_task_registry.shutdown()
      user_event_bus = getattr(app.state, "user_event_bus", None)
      if user_event_bus is not None:
        await user_event_bus.shutdown()
      cleanup_task.cancel()
      approval_expire_task.cancel()
      await asyncio.gather(cleanup_task, approval_expire_task, return_exceptions=True)

  app = FastAPI(lifespan=lifespan)
  app.state.auth = auth
  app.state.gateway_config = config

  async def _build_chat_runtime_for_dispatch(
    *,
    session: GatewaySession,
    request: ChatRequest,
    channel: str | None,
    auth_manager: AuthManager | None,
  ) -> ChatRuntime:
    _ = auth_manager
    try:
      from .control_plane.valuation_ready_tools import make_valuation_ready_skill_tool_bundle

      session.gateway_local_skill_tools = [
        make_valuation_ready_skill_tool_bundle(app_state=app.state, session=session)
      ]
    except Exception:
      log.warning("Gateway-local skill tool injection failed", exc_info=True)
    return await _call_build_chat_runtime(
      config.build_chat_runtime,
      session=session,
      request=request,
      channel=channel,
      auth_manager=auth,
    )

  setattr(_build_chat_runtime_for_dispatch, "_gateway_auth_manager", auth)
  setattr(_build_chat_runtime_for_dispatch, "_gateway_auth_config", config.auth_config)
  setattr(_build_chat_runtime_for_dispatch, "_gateway_allowed_models", config.allowed_models)
  setattr(_build_chat_runtime_for_dispatch, "_gateway_channel_profile_allowlist", config.channel_profile_allowlist)
  setattr(_build_chat_runtime_for_dispatch, "_gateway_log", log)
  setattr(_build_chat_runtime_for_dispatch, "_gateway_resolver_timeout_seconds", config.resolver_timeout_seconds)
  app.state.gateway_build_chat_runtime = _build_chat_runtime_for_dispatch
  app.state.user_event_bus = UserEventBus()
  _init_approval_subsystem(app, config)
  app.state.subprocess_registry = AutonomousRegistry(
    api_dir=_default_autonomous_api_dir(),
    python_executable=os.getenv("AGENT_GATEWAY_AUTONOMOUS_PYTHON", "").strip() or sys.executable,
    log_dir=_default_autonomous_log_dir(),
    max_running=int(os.getenv("AGENT_GATEWAY_AUTONOMOUS_MAX_RUNNING", "2") or "2"),
    approval_db_path=getattr(app.state.gateway_approval_store, "path", None),
  )
  from .control_plane.schedules import AgentRunScheduleRunner, AgentRunScheduleStore

  app.state.agent_run_schedule_store = AgentRunScheduleStore()
  app.state.agent_run_schedule_runner = AgentRunScheduleRunner(
    store=app.state.agent_run_schedule_store,
    autonomous_registry=app.state.subprocess_registry,
    user_event_bus_factory=lambda: getattr(app.state, "user_event_bus", None),
  )
  app.state.batch_task_registry = BatchTaskRegistry()
  control_prefix = _route_path(route_prefix, "/control")
  add_control_plane_version_header_middleware(
    app,
    path_prefixes={control_prefix, "/control"},
  )

  app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.cors_origins),
    allow_credentials=False,
    allow_methods=list(config.cors_allow_methods),
    allow_headers=list(config.cors_allow_headers),
  )

  router = APIRouter(prefix=route_prefix)

  @router.post("/chat/init", response_model=ChatInitResponse)
  async def chat_init(payload: ChatInitRequest) -> ChatInitResponse:
    auth.validate_api_key(payload.api_key)
    try:
      schema_version = _resolve_schema_version(payload.schema_version)
    except HTTPException as exc:
      return JSONResponse({"error": "unsupported_schema_version", "message": str(exc.detail)}, status_code=exc.status_code)
    resolver = config.credentials_resolver
    try:
      resolved_user_id = _normalize_request_user_id(payload.user_id)
    except MissingUserIdError as exc:
      status, error_payload = _error_payload(exc)
      return JSONResponse(error_payload, status_code=status)
    resolved_user_email = (
      payload.user_email.strip() if isinstance(payload.user_email, str) else payload.user_email
    )
    if resolved_user_email == "":
      resolved_user_email = None

    resolved_auth_config: dict[str, Any] | None = None
    resolved_channel: str | None = None
    resolved_risk_user_id: int = 0
    resolved_role: str = "owner"
    if resolver is not None:
      try:
        result = await asyncio.wait_for(
          resolver(payload.api_key, payload),
          timeout=config.resolver_timeout_seconds,
        )
      except asyncio.TimeoutError:
        status, error_payload = _error_payload(
          CredentialsTimeoutError(
            f"Credential resolution for user '{resolved_user_id or 'unresolved'}' timed out after "
            f"{config.resolver_timeout_seconds:.1f}s. Check the resolver latency or raise resolver_timeout_seconds."
          ),
          user_id=resolved_user_id,
          timeout_seconds=config.resolver_timeout_seconds,
        )
        return JSONResponse(error_payload, status_code=status)
      except Exception as exc:
        status, error_payload = _error_payload(exc, user_id=resolved_user_id)
        return JSONResponse(error_payload, status_code=status)
      try:
        resolved_user_id = _normalize_request_user_id(result.user_id)
      except MissingUserIdError as exc:
        status, error_payload = _error_payload(exc)
        return JSONResponse(error_payload, status_code=status)
      if resolved_user_id is None:
        status, error_payload = _error_payload(
          MissingUserIdError("credential resolver must return user_id.")
        )
        return JSONResponse(error_payload, status_code=status)
      resolved_channel = result.channel
      resolved_user_email = result.user_email or resolved_user_email
      try:
        resolved_risk_user_id = int(result.risk_user_id) if result.risk_user_id is not None else 0
      except (TypeError, ValueError):
        resolved_risk_user_id = 0
      if resolved_risk_user_id <= 0:
        status, error_payload = _resolver_contract_payload(
          "credential resolver must return a positive risk_user_id.",
          user_id=resolved_user_id,
        )
        return JSONResponse(error_payload, status_code=status)
      if result.role not in {"owner", "invite"}:
        status, error_payload = _resolver_contract_payload(
          "credential resolver must return role='owner' or role='invite'.",
          user_id=resolved_user_id,
        )
        return JSONResponse(error_payload, status_code=status)
      resolved_role = result.role
      resolved_auth_config = result.auth_config.to_dict()
      claimed_init_channel: str | None = None
      if isinstance(payload.context, dict):
        raw_claim = payload.context.get("channel")
        if isinstance(raw_claim, str) and raw_claim.strip():
          claimed_init_channel = raw_claim.strip().lower()
      if (
        claimed_init_channel is not None
        and resolved_channel is not None
        and claimed_init_channel != resolved_channel
      ):
        status, error_payload = _error_payload(
          ChannelMismatchError(
            f"context.channel={claimed_init_channel!r} does not match key channel={resolved_channel!r}. "
            "The API key is bound to a specific channel; the request must claim the same channel."
          ),
          user_id=resolved_user_id,
        )
        return JSONResponse(error_payload, status_code=status)
    elif resolved_user_id is None:
      status, error_payload = _error_payload(
        MissingUserIdError("user_id is required in /chat/init when no credentials resolver is configured.")
      )
      return JSONResponse(error_payload, status_code=status)

    try:
      identity = _resolve_control_identity(
        user_id=resolved_user_id,
        risk_user_id=resolved_risk_user_id,
        user_email=resolved_user_email,
        role=resolved_role,
        channel=resolved_channel,
      )
    except (ValueError, SystemExit) as exc:
      status, error_payload = _resolver_contract_payload(str(exc), user_id=resolved_user_id)
      return JSONResponse(error_payload, status_code=status)

    session = auth.session_store.create_session(
      api_key_hash=AuthManager.hash_api_key(payload.api_key),
      user_id=resolved_user_id,
      user_email=identity.user_email,
      risk_user_id=identity.risk_user_id,
      role=resolved_role,  # type: ignore[arg-type]
      kind="chat",
      auth_config=resolved_auth_config,
      schema_version=schema_version,
      owner_user_id=str(identity.owner_user_id),
      raw_user_id=str(identity.raw_user_id),
      user_slug=identity.user_slug,
      user_aliases=tuple(str(alias) for alias in identity.aliases),
      identity_status=str(identity.identity_status),
    )
    session.channel = resolved_channel
    session.is_public = resolved_channel == "public"
    session.approval_store = app.state.gateway_approval_store
    session.approval_policy = app.state.gateway_approval_policy
    if config.on_session_created is not None:
      try:
        config.on_session_created(session, payload.api_key, payload)
      except Exception:
        auth.session_store.expire_session(session.session_id)
        raise
    token = auth.issue_token(session)
    log.info("Session created: %s", session.session_id)
    return ChatInitResponse(
      user_id=resolved_user_id,
      session_token=token,
      session_id=session.session_id,
      expires_at=session.expires_at,
      schema_version=session.schema_version,
      model_catalog=config.model_catalog,
    )

  @router.post("/chat")
  async def chat_stream(request: Request, body: ChatRequest = Body(...)) -> StreamingResponse:
    token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
    if isinstance(body.user_id, str):
      try:
        body.user_id = _normalize_request_user_id(body.user_id)
      except MissingUserIdError as exc:
        status, error_payload = _error_payload(exc)
        return JSONResponse(error_payload, status_code=status)
    if isinstance(body.request_id, str):
      body.request_id = body.request_id.strip() or None
    session, claims = auth.verify_token_with_payload(token)
    jwt_user_id = str(claims.get("user_id") or session.user_id).strip()
    if not jwt_user_id:
      status, error_payload = _error_payload(MissingUserIdError(), session_id=session.session_id)
      return JSONResponse(error_payload, status_code=status)
    strict_mode = config.credentials_resolver is not None
    if body.user_id is not None and body.user_id != jwt_user_id:
      status, error_payload = _error_payload(
        CrossUserReuseError(
          f"Session for user '{jwt_user_id}' cannot be reused for user '{body.user_id}'. "
          "Create a separate gateway session per end user."
        ),
        session_id=session.session_id,
        session_user=jwt_user_id,
        request_user=body.user_id,
      )
      return JSONResponse(error_payload, status_code=status)
    if strict_mode:
      if body.user_id is None:
        status, error_payload = _error_payload(MissingUserIdError(), session_id=session.session_id)
        return JSONResponse(error_payload, status_code=status)
    body.user_id = body.user_id or jwt_user_id
    body.request_id = body.request_id or str(uuid.uuid4())
    if session.kind != "chat":
      return JSONResponse(
        {"error": "invalid_session_kind", "message": "control sessions cannot dispatch chat turns"},
        status_code=400,
      )
    raw_channel = (body.context or {}).get("channel")
    claimed_channel = raw_channel.strip().lower() if isinstance(raw_channel, str) else None
    channel = session.channel
    if channel is None:
      channel = claimed_channel
    elif claimed_channel and claimed_channel != channel:
      log.warning("Client claimed channel=%s; session bound to %s; using session", claimed_channel, channel)
    sid = session.session_id

    headers = {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-store, must-revalidate",
      "X-Accel-Buffering": "no",
      "Connection": "keep-alive",
    }
    user_event_bus = app.state.user_event_bus

    async def _on_chat_event(event: StreamEvent) -> None:
      event_dict = dict(event)  # type: ignore[arg-type]
      event_dict.setdefault("run_id", sid)
      try:
        await user_event_bus.publish(
          user_id=session.user_id,
          control_run_id=sid,
          event=event_dict,
        )
      except Exception as exc:
        log.warning("UserEventBus publish failed for %s: %s", sid, exc)
      if config.on_event is not None:
        try:
          config.on_event(event_dict, sid)
        except Exception:
          pass

    event_log = EventLog(session_id=sid)
    inputs = ChatTurnInputs(
      messages=list(body.messages),
      request_id=body.request_id,
      context=dict(body.context or {}),
      metadata=dict(body.metadata or {}),
      model=body.model,
    )
    dispatch_task = asyncio.create_task(
      _dispatch_chat_turn(
        session,
        inputs,
        event_log=event_log,
        on_event=_on_chat_event,
        build_chat_runtime=app.state.gateway_build_chat_runtime,
        credentials_resolver=config.credentials_refresh_resolver,  # type: ignore[arg-type]
        transcript_dir=transcript_dir,
        publish_lifecycle_events=True,
      )
    )
    await asyncio.sleep(0)
    if dispatch_task.done():
      exc = dispatch_task.exception()
      if exc is not None:
        raise exc
    active_turn = session.active_turn
    subscriber: StreamSubscriber | None = None
    if active_turn is not None and active_turn.event_log is event_log:
      subscriber = _register_stream_subscriber(
        active_turn,
        after_seq=0,
        client_label="post",
      )

    async def finalize_dispatch_task() -> None:
      try:
        await dispatch_task
      except asyncio.CancelledError:
        pass
      except Exception as exc:
        log.warning("chat turn dispatch failed for %s: %s", sid, exc)
      finally:
        await user_event_bus.cleanup_run(session.user_id, sid)

    asyncio.create_task(finalize_dispatch_task())

    cleanup_complete = False
    cleanup_lock = asyncio.Lock()

    async def response_cleanup() -> None:
      nonlocal cleanup_complete
      async with cleanup_lock:
        if cleanup_complete:
          return
        active_turn = session.active_turn
        if active_turn is not None and active_turn.event_log is event_log and subscriber is not None:
          await _cleanup_stream_subscriber(active_turn, subscriber.subscriber_id)
        cleanup_complete = True

    async def event_generator():
      try:
        if active_turn is None or subscriber is None:
          return
        async for chunk in _stream_subscriber_sse(
          session=session,
          active_turn=active_turn,
          subscriber=subscriber,
          transcript_dir=transcript_dir,
          channel=channel,
          write_transcript=True,
          log=log,
        ):
          yield chunk
      finally:
        await response_cleanup()

    return StreamingResponse(event_generator(), headers=headers, background=BackgroundTask(response_cleanup))

  @router.get("/chat/subscribe")
  async def chat_subscribe(request: Request) -> StreamingResponse:
    return await _server_chat_control_routes.chat_subscribe_response(
      request,
      auth=auth,
      get_bearer_token=AuthManager.get_bearer_token,
      register_stream_subscriber=_register_stream_subscriber,
      cleanup_stream_subscriber=_cleanup_stream_subscriber,
      stream_subscriber_sse=_stream_subscriber_sse,
      transcript_dir=transcript_dir,
      log=log,
    )

  @router.post("/chat/cancel")
  async def chat_cancel(request: Request, body: ChatCancelRequest = Body(...)) -> JSONResponse:
    return await _server_chat_control_routes.chat_cancel_response(
      request,
      body,
      auth=auth,
      get_bearer_token=AuthManager.get_bearer_token,
      cancel_active_turn_runner=_cancel_active_turn_runner,
      clear_active_turn=_clear_active_turn,
    )

  @router.post("/chat/recap")
  async def chat_recap(request: Request, body: ChatRecapRequest = Body(...)) -> JSONResponse:
    return await _server_chat_control_routes.chat_recap_response(
      request,
      body,
      auth=auth,
      get_bearer_token=AuthManager.get_bearer_token,
      transcript_dir=transcript_dir,
      compute_session_recap_payload=_compute_session_recap_payload,
      compute_cumulative_session_recap_payload=_compute_cumulative_session_recap_payload,
      event_for_wire=_event_for_wire,
      write_transcript=_write_transcript,
    )

  @router.get("/artifacts/{ticker}/{skill}/latest")
  async def artifact_latest(request: Request, ticker: str, skill: str) -> JSONResponse:
    return _server_artifact_routes.artifact_latest_response(globals(), request, ticker, skill)

  @router.get("/artifacts/{ticker}/{skill}/{artifact_id}")
  async def artifact_by_id(
    request: Request,
    ticker: str,
    skill: str,
    artifact_id: str,
  ) -> JSONResponse:
    return _server_artifact_routes.artifact_by_id_response(globals(), request, ticker, skill, artifact_id)

  @router.get("/artifacts/{ticker}")
  async def artifact_index(request: Request, ticker: str) -> JSONResponse:
    return _server_artifact_routes.artifact_index_response(globals(), request, ticker)

  @router.get("/letters/{ticker}/{artifact_id}")
  async def letter_by_id(request: Request, ticker: str, artifact_id: str) -> FileResponse:
    return _server_artifact_routes.letter_by_id_response(globals(), request, ticker, artifact_id)

  @router.get("/artifacts/{artifact_path:path}")
  async def artifact_path_guard(request: Request, artifact_path: str) -> JSONResponse:
    return _server_artifact_routes.artifact_path_guard_response(globals(), request, artifact_path)

  @router.get("/letters/{letter_path:path}")
  async def letter_path_guard(request: Request, letter_path: str) -> JSONResponse:
    return _server_artifact_routes.letter_path_guard_response(globals(), request, letter_path)

  @router.post("/chat/tool-result")
  async def tool_result(request: Request, payload: ToolResultRequest) -> JSONResponse:
    return await _server_tool_routes.tool_result_response(
      request,
      payload,
      auth=auth,
      log=log,
      time_time=time.time,
    )

  @router.post("/chat/tool-approval")
  async def tool_approval(request: Request, payload: ToolApprovalRequest) -> JSONResponse:
    return await _server_tool_routes.tool_approval_response(
      request,
      payload,
      auth=auth,
      record_vote_and_unblock=_record_vote_and_unblock,
    )

  @router.get("/health")
  async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "package": package_health()})

  from .control_plane.dashboard_artifacts import build_dashboard_artifacts_router
  from .control_plane.html_artifacts import build_html_artifacts_router

  router.include_router(build_dashboard_artifacts_router(artifact_auth_dependency=_artifact_auth_dependency))
  router.include_router(build_html_artifacts_router(artifact_auth_dependency=_artifact_auth_dependency))
  app.include_router(router)
  app.include_router(
    create_control_plane_router(
      auth=auth,
      credentials_resolver=config.credentials_resolver,
      resolver_timeout_seconds=config.resolver_timeout_seconds,
      route_prefix=route_prefix,
      skills_dir=config.control_skills_dir or _default_control_skills_dir(),
      artifact_auth_dependency=_artifact_auth_dependency,
      autonomous_registry=app.state.subprocess_registry,
      agent_schedule_store=app.state.agent_run_schedule_store,
      agent_schedule_runner=app.state.agent_run_schedule_runner,
      approval_store=app.state.gateway_approval_store,
      approval_policy=app.state.gateway_approval_policy,
      dispatch_scope_validator=config.dispatch_scope_validator,
    ),
    prefix=control_prefix,
  )
  app.state.gateway_chat_init = chat_init
  app.state.gateway_chat_stream = chat_stream
  app.state.gateway_control_session = next(
    route.endpoint for route in app.routes if getattr(route, "path", None) == f"{control_prefix}/session"
  )
  app.state.gateway_control_health = next(
    route.endpoint for route in app.routes if getattr(route, "path", None) == f"{control_prefix}/health"
  )
  app.state.gateway_artifact_latest = artifact_latest
  app.state.gateway_artifact_by_id = artifact_by_id
  app.state.gateway_artifact_index = artifact_index
  app.state.gateway_letter_by_id = letter_by_id
  app.state.gateway_tool_result = tool_result
  app.state.gateway_tool_approval = tool_approval
  app.state.gateway_health = health
  return app


__all__ = [
  "ChatInitRequest",
  "ChatInitResponse",
  "ChatMessage",
  "ChatRequest",
  "ChatTurnInputs",
  "ChatTurnResult",
  "ChatRuntime",
  "GatewayServerConfig",
  "ModelCatalog",
  "RequestContext",
  "ToolApprovalRequest",
  "ToolResultRequest",
  "create_gateway_app",
]
