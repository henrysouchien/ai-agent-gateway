from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import functools
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, Body, FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from .artifact_paths import (
  ArtifactPath as ArtifactPath,
  ArtifactPathError as ArtifactPathError,
  artifact_json_paths_for_request as artifact_json_paths_for_request,
  artifact_json_path_for_request as artifact_json_path_for_request,
  letter_docx_path_for_request as letter_docx_path_for_request,
  reject_unsafe_path as reject_unsafe_path,
  ticker_artifact_paths_for_request as ticker_artifact_paths_for_request,
  user_workspace_root as user_workspace_root,
)
from .auth import (
  ChannelMismatchError,
  CredentialsTimeoutError,
  CrossUserReuseError,
  MissingUserIdError,
)
from .autonomous_runner import AutonomousRegistry
from .skills import SkillLoader
from .control_plane import create_control_plane_router
from .control_plane.middleware import add_control_plane_version_header_middleware
from .control_plane.session import _resolve_control_identity
from .commercial_work_start import (
  COMMERCIAL_CLAIM_HEADER,
  COMMERCIAL_WORK_AUTHORIZATION_HEADER,
  CommercialWorkStartError,
)
from .capability_binding import (
  CAPABILITY_IDS,
  CapabilityResolutionError,
  eligible_model_choices,
)
from .model_registry import GATEWAY_EXECUTED_CAPABILITY_IDS
from .event_log import EventLog, UserEventBus
from .dispatcher_factory import GatewayDispatcherDeps
from .approvals import ApprovalActionError, _record_vote_and_unblock  # noqa: F401
from .approval_store import expire_pending_loop  # noqa: F401
from .package_info import (
  CONTRACT_CHAT_ATTACHMENTS_V1,
  CONTRACT_INVESTMENT_SELECTED_CONTENT_V1,
  package_health,
)
from .investment_capability_claim import (
  investment_capability_signing_available,
)
from .providers import (
  ModelProvider,
  StreamEvent,
  installed_adapter_providers,
)
from .runner_introspection import exception_traceback_already_logged
from .session import (
  AuthManager,
  GatewaySession,
  SessionStore,
  StreamSubscriber,
  bind_session_credentials,
  session_owner_user_id,
)
from .ui_blocks_metrics import snapshot as ui_blocks_metrics_snapshot
from .ui_blocks_store import read_ui_blocks_payload as read_ui_blocks_payload

from . import server_chat_helpers as _server_chat_helpers  # noqa: F401 - dynamic streaming deps alias
from . import server_chat_control_routes as _server_chat_control_routes
from . import server_artifact_routes as _server_artifact_routes  # noqa: F401 - dynamic artifact route deps alias
from . import server_streaming as _server_streaming
from . import server_tool_routes as _server_tool_routes
from . import server_workflow_output_routes as _server_workflow_output_routes
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
  ChatInitResponse,
  ChatMessage,
  ChatRequest,
  ChatRecapRequest,
  ChatCancelRequest,
  _resolve_chat_profile_name,
  ChatTurnInputs,
  ChatTurnResult,
  PreparedChatTurn,
  ToolResultRequest,
  ToolApprovalRequest,
  ChatRuntime,
  ModelPreferenceResponse,
  ModelPreferenceUpdate,
  _build_runner_with_started_at,
  _call_build_chat_runtime,
  RequestContext,
  GatewayServerConfig,
  MaterializedCredential,
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
  bind_init_capability_selections,
  build_capability_choices,
  _capability_execution_resolver_for_session,
  prepare_session_driver_turn,
  _dispatch_chat_turn,
  _init_approval_subsystem,
)


def _generic_skill_resume_allowed_resolver(
  skills_dir: Path,
) -> Callable[[str], bool]:
  loader = SkillLoader(skills_dir)

  def resolve(skill: str) -> bool:
    try:
      profile = loader.load(skill)
    except (FileNotFoundError, ValueError):
      return False
    return bool(profile.resumable) and profile.mutation_mode != "model_writer"

  return resolve


(
  _disconnect_stream_subscriber_for_backpressure,
  _pump_stream_subscriber,
  _register_stream_subscriber,
  _cleanup_stream_subscriber,
  _stream_subscriber_sse,
) = _server_streaming.bind_streaming_helpers(lambda: globals())


def _find_route_endpoint_by_name(routes: list[Any], route_name: str) -> Any:
  """Resolve an endpoint across eager and FastAPI 0.139 lazy router trees."""
  pending = list(routes)
  visited: set[int] = set()
  while pending:
    route = pending.pop()
    route_id = id(route)
    if route_id in visited:
      continue
    visited.add(route_id)
    if getattr(route, "name", None) == route_name:
      endpoint = getattr(route, "endpoint", None)
      if endpoint is not None:
        return endpoint
    original_router = getattr(route, "original_router", None)
    nested_routes = getattr(original_router, "routes", None)
    if nested_routes is not None:
      pending.extend(nested_routes)
  raise LookupError(f"Gateway route endpoint not found: {route_name}")


def _public_request_validation_errors(
  exc: RequestValidationError,
) -> list[dict[str, Any]]:
  """Return useful validation diagnostics without reflecting request values."""

  public_fields = ("type", "loc", "msg", "url")
  return [
    {
      field: error[field]
      for field in public_fields
      if field in error
    }
    for error in exc.errors()
  ]


async def _drain_shielded_lifecycle_task(
  task: asyncio.Task[Any],
) -> None:
  """Wait for lifecycle work despite repeated cancellation of its owner."""

  while True:
    try:
      await asyncio.shield(task)
      return
    except asyncio.CancelledError:
      if task.done():
        task.result()
        return


def _has_investment_selected_content_reader(mcp_client: Any) -> bool:
  """Project the optional Investment reader from the loaded MCP catalog."""

  if not investment_capability_signing_available():
    return False

  is_mcp_tool = getattr(mcp_client, "is_mcp_tool", None)
  get_server_for_tool = getattr(mcp_client, "get_server_for_tool", None)
  if not callable(is_mcp_tool) or not callable(get_server_for_tool):
    return False
  try:
    return bool(is_mcp_tool("get_investment_artifact")) and (
      get_server_for_tool("get_investment_artifact") == "idea-workbench-mcp"
    )
  except Exception:
    return False


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
  if (
    config.session_execution_policy_resolver is not None
    and not callable(config.session_execution_policy_resolver)
  ):
    raise TypeError("session_execution_policy_resolver must be callable")
  if (
    config.selected_content_admitter is not None
    and not callable(config.selected_content_admitter)
  ):
    raise TypeError("selected_content_admitter must be callable")
  if not isinstance(config.allow_service_credentials_for_interactive, bool):
    raise ValueError(
      "allow_service_credentials_for_interactive must be a bool"
    )
  if config.model_registry is None or config.model_selection_policy is None:
    raise ValueError("model_registry and model_selection_policy are required")
  if not str(config.tenant_id or "").strip():
    raise ValueError("tenant_id is required with model-selection authority")
  config.model_selection_policy.admit_registry(config.model_registry)

  # Adapter support comes from the installed adapters' own declarations, never
  # a hand-maintained table.  `installed_adapter_providers()` maps each
  # declared adapter id to the class implementing it.
  installed_providers = installed_adapter_providers()

  capability_adapter_cache: dict[str, ModelProvider] = {}
  configured_default_provider = config.default_provider
  if configured_default_provider is not None:
    configured_family = str(
      getattr(configured_default_provider, "name", "") or ""
    ).strip().lower()
    if configured_family == "agent-sdk":
      configured_family = "anthropic"
    default_adapter = next(
      (
        adapter_id
        for adapter_id, provider_class in installed_providers.items()
        if provider_class.adapter_route_support().provider == configured_family
      ),
      None,
    )
    if default_adapter is not None:
      capability_adapter_cache[default_adapter] = configured_default_provider

  def _resolve_capability_adapter(adapter_id: str) -> ModelProvider:
    normalized_adapter = str(adapter_id or "").strip()
    provider = capability_adapter_cache.get(normalized_adapter)
    # Explicitly injected implementations — a configured `default_provider`
    # (already cached) or anything a deployment-supplied
    # `capability_adapter_resolver` returns — are vouched for by that trusted
    # configuration.  Only implementations the built-in factory constructs
    # from installed declarations are closure-checked against them below.
    deployment_vouched = provider is not None
    if provider is None:
      if config.capability_adapter_resolver is not None:
        provider = config.capability_adapter_resolver(normalized_adapter)
        deployment_vouched = True
      else:
        factory = installed_providers.get(normalized_adapter)
        if factory is None:
          raise ValueError(
            f"no installed capability adapter for {normalized_adapter!r}"
          )
        provider = factory()
    expected_providers = {
      entry.provider
      for entry in config.model_registry.models.values()
      if entry.adapter == normalized_adapter
    }
    resolved_family = str(getattr(provider, "name", "") or "").strip().lower()
    if resolved_family == "agent-sdk":
      resolved_family = "anthropic"
    if expected_providers != {resolved_family}:
      raise ValueError(
        f"capability adapter {normalized_adapter!r} returned provider "
        f"{resolved_family!r}; expected {sorted(expected_providers)}"
      )

    # Closure against the implementation's own protocol declaration.  On the
    # built-in path the installed adapter class must declare this adapter id,
    # and every gateway-executed registry entry bound to it must use a
    # declared protocol profile and route.  Deployment-vouched implementations
    # (test doubles and app-owned wrappers included) skip only this
    # declaration check; the packaged/deployment-selected INITIAL artifacts
    # are still closed against installed declarations at import admission.
    if not deployment_vouched:
      declared = getattr(provider, "adapter_route_support", None)
      declaration = declared() if callable(declared) else None
      if declaration is None:
        raise ValueError(
          f"capability adapter {normalized_adapter!r} declares no protocol support"
        )
      if declaration.adapter != normalized_adapter:
        raise ValueError(
          f"capability adapter {normalized_adapter!r} resolved an "
          f"implementation declaring {declaration.adapter!r}"
        )
      for entry in config.model_registry.models.values():
        if entry.adapter != normalized_adapter:
          continue
        if not (set(entry.capabilities) & GATEWAY_EXECUTED_CAPABILITY_IDS):
          continue
        if not declaration.supports(entry):
          raise ValueError(
            f"registry entry {entry.key!r} requires protocol profile "
            f"{entry.protocol_profile!r} on route {entry.route!r}, which "
            f"installed adapter {normalized_adapter!r} does not declare"
          )
    capability_adapter_cache[normalized_adapter] = provider
    return provider

  # Full startup closure over the configured registry (design § Configuration
  # and evolution step 4): every entry serving a capability this gateway
  # process executes must resolve to an installed, declared adapter now —
  # never `provider_unavailable` at first use.  Entries serving only
  # externally-executed capabilities (risk.*, investment.*) are admitted
  # registry facts for their own serving processes and are excluded here by
  # explicit designation, not silently skipped.
  for adapter_id in sorted({
    entry.adapter
    for entry in config.model_registry.models.values()
    if set(entry.capabilities) & GATEWAY_EXECUTED_CAPABILITY_IDS
  }):
    _resolve_capability_adapter(adapter_id)

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
    retention_task: asyncio.Task[Any] | None = None

    async def _retention_loop() -> None:
      sweeper = config.retention_sweeper
      if sweeper is None:
        return
      interval = max(60.0, float(getattr(sweeper, "interval_seconds", 6 * 3600)))
      while True:
        try:
          await asyncio.to_thread(sweeper.sweep)
        except asyncio.CancelledError:
          raise
        except Exception:
          log.warning("durable-artifact retention sweep refused or failed", exc_info=True)
        await asyncio.sleep(interval)

    if config.retention_sweeper is not None:
      retention_task = asyncio.create_task(_retention_loop())
    config_started = False
    schedule_start_attempted = False
    approval_delivery_start_attempted = False
    primary_failure: BaseException | None = None

    def record_failure(
      phase: str,
      failure: BaseException,
    ) -> None:
      nonlocal primary_failure
      if primary_failure is None:
        primary_failure = failure
        return
      try:
        primary_failure.add_note(
          f"Additional gateway teardown failure in {phase}: "
          f"{type(failure).__name__}: {failure}"
        )
      except AttributeError:
        pass
      log.error(
        "Additional gateway teardown failure in %s",
        phase,
        exc_info=(
          type(failure),
          failure,
          failure.__traceback__,
        ),
      )

    async def cleanup_phase(
      phase: str,
      cleanup: Any,
    ) -> None:
      try:
        await cleanup()
      except BaseException as exc:
        record_failure(phase, exc)

    async def teardown() -> None:
      if config_started:
        await cleanup_phase(
          "configured shutdown",
          lambda: _maybe_await(config.on_shutdown),
        )

      maintenance_tasks = [
        cleanup_task,
        approval_expire_task,
        *([retention_task] if retention_task is not None else []),
      ]
      for maintenance_task in maintenance_tasks:
        maintenance_task.cancel()

      agent_schedule_runner = getattr(
        app.state,
        "agent_run_schedule_runner",
        None,
      )
      if (
        schedule_start_attempted
        and agent_schedule_runner is not None
      ):
        await cleanup_phase(
          "agent schedule runner",
          agent_schedule_runner.shutdown,
        )
      approval_delivery_coordinator = getattr(
        app.state,
        "autonomous_approval_delivery_coordinator",
        None,
      )
      if (
        approval_delivery_start_attempted
        and approval_delivery_coordinator is not None
      ):
        await cleanup_phase(
          "autonomous approval delivery coordinator",
          approval_delivery_coordinator.shutdown,
        )
      subprocess_registry = getattr(
        app.state,
        "subprocess_registry",
        None,
      )
      if subprocess_registry is not None:
        await cleanup_phase(
          "autonomous subprocess registry",
          subprocess_registry.shutdown,
        )
      batch_task_registry = getattr(
        app.state,
        "batch_task_registry",
        None,
      )
      if batch_task_registry is not None:
        await cleanup_phase(
          "batch task registry",
          lambda: batch_task_registry.shutdown(
            app_state=app.state
          ),
        )
      maintenance_results = await asyncio.gather(
        *maintenance_tasks,
        return_exceptions=True,
      )
      for index, result in enumerate(maintenance_results):
        if (
          isinstance(result, BaseException)
          and not isinstance(result, asyncio.CancelledError)
        ):
          record_failure(
            f"maintenance task {index}",
            result,
          )
      user_event_bus = getattr(
        app.state,
        "user_event_bus",
        None,
      )
      if user_event_bus is not None:
        await cleanup_phase(
          "user event bus",
          user_event_bus.shutdown,
        )

    try:
      await _maybe_await(config.on_startup)
      config_started = True
      approval_delivery_coordinator = getattr(
        app.state,
        "autonomous_approval_delivery_coordinator",
        None,
      )
      if approval_delivery_coordinator is not None:
        approval_delivery_start_attempted = True
        approval_delivery_coordinator.start()
      agent_schedule_runner = getattr(app.state, "agent_run_schedule_runner", None)
      if agent_schedule_runner is not None:
        schedule_start_attempted = True
        agent_schedule_runner.start()
      yield
    except BaseException as exc:
      primary_failure = exc
    finally:
      teardown_task = asyncio.create_task(teardown())
      try:
        await asyncio.shield(teardown_task)
      except asyncio.CancelledError as exc:
        record_failure("lifespan teardown cancellation", exc)
        try:
          await _drain_shielded_lifecycle_task(
            teardown_task
          )
        except BaseException as teardown_exc:
          record_failure(
            "lifespan teardown task",
            teardown_exc,
          )
      except BaseException as exc:
        record_failure("lifespan teardown task", exc)
    if primary_failure is not None:
      raise primary_failure.with_traceback(
        primary_failure.__traceback__
      )

  app = FastAPI(lifespan=lifespan)

  @app.exception_handler(RequestValidationError)
  async def _sanitize_request_validation_error(
    _request: Request,
    exc: RequestValidationError,
  ):
    if any(
      "capability_selections" in error.get("loc", ())
      for error in exc.errors()
    ):
      return JSONResponse(
        {
          "error_code": "capability_selection_invalid",
          "message": (
            "capability_selections must map capability IDs to strict "
            "model/effort selections"
          ),
        },
        status_code=422,
      )
    sensitive_error_types = {
      "commercial_bearer_material_forbidden",
    }
    if any(error.get("type") in sensitive_error_types for error in exc.errors()):
      return JSONResponse(
        {
          "error": "commercial_bearer_material_in_body",
          "message": "Commercial bearer material is accepted only in dedicated headers.",
        },
        status_code=422,
      )
    return JSONResponse(
      status_code=422,
      content={
        "detail": jsonable_encoder(_public_request_validation_errors(exc)),
      },
    )

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
    dispatcher_deps = getattr(
      _build_chat_runtime_for_dispatch,
      "_gateway_dispatcher_deps",
      None,
    )
    if dispatcher_deps is not None:
      setattr(session, "_gateway_dispatcher_deps", dispatcher_deps)
    try:
      from .control_plane.valuation_ready_tools import make_valuation_ready_skill_tool_bundle

      session.gateway_local_skill_tools = [
        make_valuation_ready_skill_tool_bundle(app_state=app.state, session=session)
      ]
    except Exception:
      log.warning("Gateway-local skill tool injection failed", exc_info=True)
    runtime = await _call_build_chat_runtime(
      config.build_chat_runtime,
      session=session,
      request=request,
      channel=channel,
      auth_manager=auth,
      storage_root=app.state.autonomous_storage_root,
    )
    runtime.purpose = session.purpose
    return runtime

  setattr(_build_chat_runtime_for_dispatch, "_gateway_auth_manager", auth)
  setattr(
    _build_chat_runtime_for_dispatch,
    "_gateway_model_registry",
    config.model_registry,
  )
  setattr(
    _build_chat_runtime_for_dispatch,
    "_gateway_model_selection_policy",
    config.model_selection_policy,
  )
  setattr(
    _build_chat_runtime_for_dispatch,
    "_gateway_session_execution_policy_resolver",
    config.session_execution_policy_resolver,
  )
  setattr(
    _build_chat_runtime_for_dispatch,
    "_gateway_service_provider_handles",
    dict(config.service_provider_handles),
  )
  setattr(
    _build_chat_runtime_for_dispatch,
    "_gateway_service_auth_config_resolver",
    config.service_auth_config_resolver,
  )
  setattr(
    _build_chat_runtime_for_dispatch,
    "_gateway_capability_adapter_resolver",
    _resolve_capability_adapter,
  )
  setattr(
    _build_chat_runtime_for_dispatch,
    "_gateway_model_preference_store",
    config.model_preference_store,
  )
  setattr(
    _build_chat_runtime_for_dispatch,
    "_gateway_tenant_id",
    config.tenant_id,
  )
  setattr(_build_chat_runtime_for_dispatch, "_gateway_channel_profile_allowlist", config.channel_profile_allowlist)
  setattr(_build_chat_runtime_for_dispatch, "_gateway_log", log)
  setattr(
    _build_chat_runtime_for_dispatch,
    "_gateway_selected_content_admitter",
    config.selected_content_admitter,
  )
  setattr(_build_chat_runtime_for_dispatch, "_gateway_resolver_timeout_seconds", config.resolver_timeout_seconds)
  app.state.gateway_build_chat_runtime = _build_chat_runtime_for_dispatch
  app.state.user_event_bus = UserEventBus()
  _init_approval_subsystem(app, config)
  dispatcher_deps = GatewayDispatcherDeps(
    mcp_client=config.mcp_client,
    approval_store=app.state.gateway_approval_store,
    approval_policy=app.state.gateway_approval_policy,
    mcp_meta_inject_servers=(
      config.mcp_meta_inject_servers or frozenset()
    ),
  )
  app.state.gateway_dispatcher_deps = dispatcher_deps
  setattr(
    _build_chat_runtime_for_dispatch,
    "_gateway_dispatcher_deps",
    dispatcher_deps,
  )
  app.state.gateway_claim_signing_authority = (
    config.claim_signing_authority
  )
  autonomous_storage_root = _default_autonomous_log_dir()
  control_skills_root = (
    config.control_skills_dir or _default_control_skills_dir()
  )
  skill_resume_allowed_resolver = (
    config.autonomous_skill_resume_allowed_resolver
    if config.autonomous_skill_resume_allowed_resolver is not None
    else _generic_skill_resume_allowed_resolver(control_skills_root)
  )
  app.state.autonomous_storage_root = autonomous_storage_root
  app.state.subprocess_registry = AutonomousRegistry(
    api_dir=(
      config.autonomous_api_dir
      if config.autonomous_api_dir is not None
      else _default_autonomous_api_dir()
    ),
    tenant_id=config.tenant_id,
    python_executable=os.getenv("AGENT_GATEWAY_AUTONOMOUS_PYTHON", "").strip() or sys.executable,
    log_dir=autonomous_storage_root,
    max_running=int(os.getenv("AGENT_GATEWAY_AUTONOMOUS_MAX_RUNNING", "2") or "2"),
    approval_store=app.state.gateway_approval_store,
    service_provider_handles=config.service_provider_handles,
    autonomous_capability_binding_resolver=(
      config.autonomous_capability_binding_resolver
    ),
    skill_resume_allowed_resolver=skill_resume_allowed_resolver,
    claim_signing_authority=config.claim_signing_authority,
  )
  from .control_plane.autonomous_approval_drainer import (
    AutonomousApprovalDeliveryCoordinator,
  )

  app.state.autonomous_approval_delivery_coordinator = (
    AutonomousApprovalDeliveryCoordinator(
      store=app.state.gateway_approval_store,
      registry=app.state.subprocess_registry,
    )
  )
  from .control_plane.schedules import (
    AgentRunScheduleRunner,
    ScheduleStoreUnreadableError,
    agent_run_schedule_users_root,
    schedule_store_for,
  )

  app.state.agent_run_schedule_store_for = functools.cache(schedule_store_for)
  app.state.agent_run_schedule_runner = AgentRunScheduleRunner(
    store_for=app.state.agent_run_schedule_store_for,
    users_root=agent_run_schedule_users_root(),
    autonomous_registry=app.state.subprocess_registry,
    user_event_bus_factory=lambda: getattr(app.state, "user_event_bus", None),
  )

  @app.exception_handler(ScheduleStoreUnreadableError)
  async def _schedule_store_unreadable_handler(
    _request: Request,
    exc: ScheduleStoreUnreadableError,
  ) -> JSONResponse:
    return JSONResponse(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      content={
        "error": "schedule_store_unreadable",
        "message": str(exc),
      },
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
  async def chat_init(payload: ChatInitRequest, response: Response) -> ChatInitResponse:
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
    resolved_role: str = "invite"
    resolved_capabilities: frozenset[str] = frozenset()
    resolved_model_entitled_capabilities: frozenset[str] = frozenset()
    resolved_model_entitled_keys: frozenset[str] = frozenset()
    resolved_credential_principal = None
    resolved_allow_service_for_interactive = (
      config.allow_service_credentials_for_interactive
    )
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
      resolved_capabilities = result.capabilities
      resolved_model_entitled_capabilities = result.model_entitled_capabilities
      resolved_model_entitled_keys = result.model_entitled_keys
      resolved_auth_config = result.auth_config.to_dict()
      resolved_credential_principal = result.credential_principal
      resolved_allow_service_for_interactive = (
        result.allow_service_for_interactive
      )
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
      capabilities=resolved_capabilities,
      model_entitled_capabilities=resolved_model_entitled_capabilities,
      model_entitled_keys=resolved_model_entitled_keys,
      kind="chat",
      auth_config=resolved_auth_config,
      schema_version=schema_version,
      owner_user_id=str(identity.owner_user_id),
      raw_user_id=str(identity.raw_user_id),
      user_slug=identity.user_slug,
      user_aliases=tuple(str(alias) for alias in identity.aliases),
      identity_status=str(identity.identity_status),
      tenant_id=config.tenant_id,
      allow_service_for_interactive=resolved_allow_service_for_interactive,
    )
    session.channel = resolved_channel
    session.is_public = resolved_channel == "public"
    session.approval_store = app.state.gateway_approval_store
    session.approval_policy = app.state.gateway_approval_policy
    try:
      if config.on_session_created is not None:
        config.on_session_created(session, payload.api_key, payload)
      if (
        session.session_credential_handle is None
        and resolved_credential_principal is not None
        and config.tenant_id is not None
      ):
        bind_session_credentials(
          session,
          tenant_id=config.tenant_id,
          credential_principal=resolved_credential_principal,
          allow_service_for_interactive=(
            resolved_allow_service_for_interactive
          ),
        )
      bind_init_capability_selections(
        session=session,
        selections=payload.capability_selections,
        build_chat_runtime=app.state.gateway_build_chat_runtime,
      )
    except CapabilityResolutionError as exc:
      auth.session_store.expire_session(session.session_id)
      return JSONResponse(
        {
          **exc.receipt(),
          "message": str(exc),
        },
        status_code=422,
      )
    except Exception:
      auth.session_store.expire_session(session.session_id)
      raise
    token = auth.issue_token(session)
    log.info("Session created: %s", session.session_id)
    response.headers["Cache-Control"] = "private, no-store"
    return ChatInitResponse(
      user_id=resolved_user_id,
      session_token=token,
      session_id=session.session_id,
      expires_at=session.expires_at,
      schema_version=session.schema_version,
      capability_choices=build_capability_choices(
        session=session,
        build_chat_runtime=app.state.gateway_build_chat_runtime,
      ),
    )

  def _model_preference_session(request: Request) -> GatewaySession:
    token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
    session, _claims = auth.verify_token_with_payload(token)
    if session.kind != "chat":
      raise HTTPException(
        status_code=400,
        detail="model preferences require a chat session",
      )
    return session

  @router.put(
    "/model-preferences/{capability_id}",
    response_model=ModelPreferenceResponse,
  )
  async def put_model_preference(
    capability_id: str,
    request: Request,
    body: ModelPreferenceUpdate,
  ) -> ModelPreferenceResponse:
    session = _model_preference_session(request)
    store = config.model_preference_store
    if store is None:
      raise HTTPException(status_code=503, detail="model preference store is unavailable")
    normalized_capability = str(capability_id or "").strip()
    policy = config.model_selection_policy.capabilities.get(normalized_capability)
    if policy is None or not policy.allow_saved_preference:
      raise HTTPException(
        status_code=422,
        detail=f"saved preference is not allowed for {normalized_capability!r}",
      )
    resolver = _capability_execution_resolver_for_session(
      session=session,
      build_chat_runtime=app.state.gateway_build_chat_runtime,
    )
    choices = {
      choice.key: choice
      for choice in eligible_model_choices(
        normalized_capability,
        registry=resolver.registry,
        selection_policy=resolver.selection_policy,
        auth=resolver.auth_context,
      )
    }
    # The observed catalog revision is concurrency context, not authority: a
    # still-eligible key is accepted after a harmless refresh, while a key or
    # effort that no longer resolves under the current catalog names the
    # stale revision so the client refreshes choices (plan §6.C/§8).
    stale_catalog = (
      body.catalog_revision is not None
      and body.catalog_revision != resolver.registry.revision
    )

    def _stale_catalog_response() -> JSONResponse:
      return JSONResponse(
        {
          "error_code": "capability_catalog_stale",
          "capability_id": normalized_capability,
          "model_key": body.model_key,
          "catalog_revision": resolver.registry.revision,
          "eligible_model_keys": sorted(choices),
          "message": (
            "The preference was made against a stale catalog revision and no "
            "longer resolves; refresh eligible choices and select again."
          ),
        },
        status_code=422,
      )

    choice = choices.get(body.model_key)
    if choice is None:
      if stale_catalog:
        return _stale_catalog_response()
      return JSONResponse(
        {
          "error_code": "capability_model_unavailable",
          "capability_id": normalized_capability,
          "model_key": body.model_key,
          "eligible_model_keys": sorted(choices),
          "message": "The requested saved preference is not session-eligible.",
        },
        status_code=422,
      )
    effort = body.effort or choice.default_effort
    if effort not in choice.supported_efforts:
      if stale_catalog:
        return _stale_catalog_response()
      return JSONResponse(
        {
          "error_code": "capability_effort_unsupported",
          "capability_id": normalized_capability,
          "model_key": body.model_key,
          "supported_efforts": list(choice.supported_efforts),
          "message": "The requested saved effort is not supported.",
        },
        status_code=422,
      )
    stored = store.put(
      tenant_id=resolver.auth_context.tenant_id,
      actor_id=resolver.auth_context.actor_id,
      capability_id=normalized_capability,
      model_key=body.model_key,
      effort=effort,
    )
    return ModelPreferenceResponse(
      capability=normalized_capability,
      model_key=stored.model_key,
      effort=stored.effort,
    )

  @router.delete(
    "/model-preferences/{capability_id}",
    response_model=ModelPreferenceResponse,
  )
  async def delete_model_preference(
    capability_id: str,
    request: Request,
  ) -> ModelPreferenceResponse:
    session = _model_preference_session(request)
    store = config.model_preference_store
    if store is None:
      raise HTTPException(status_code=503, detail="model preference store is unavailable")
    normalized_capability = str(capability_id or "").strip()
    policy = config.model_selection_policy.capabilities.get(normalized_capability)
    if policy is None or not policy.allow_saved_preference:
      raise HTTPException(
        status_code=422,
        detail=f"saved preference is not allowed for {normalized_capability!r}",
      )
    resolver = _capability_execution_resolver_for_session(
      session=session,
      build_chat_runtime=app.state.gateway_build_chat_runtime,
    )
    store.delete(
      tenant_id=resolver.auth_context.tenant_id,
      actor_id=resolver.auth_context.actor_id,
      capability_id=normalized_capability,
    )
    return ModelPreferenceResponse(
      capability=normalized_capability,
      model_key=None,
      effort=None,
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
    if body.attachments and config.selected_content_admitter is None:
      return JSONResponse(
        {
          "error": "chat_attachments_unsupported",
          "message": "This gateway does not accept chat attachments.",
        },
        status_code=400,
      )
    if (
      body.investment_artifact_selection is not None
      and config.selected_content_admitter is None
    ):
      return JSONResponse(
        {
          "error": "investment_selected_content_unsupported",
          "message": "This gateway does not accept Investment selections.",
        },
        status_code=400,
      )
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
    commercial_work_start = None
    commercial_dispatch_owner = None
    prepared_turn: PreparedChatTurn | None = None
    inputs = ChatTurnInputs(
      messages=list(body.messages),
      request_id=body.request_id,
      context=dict(body.context or {}),
      metadata=dict(body.metadata or {}),
      model_key=body.model_key,
      effort=body.effort,
      catalog_revision=body.catalog_revision,
      ui_blocks_contract=body.ui_blocks_contract,
      attachments=tuple(body.attachments),
      investment_artifact_selection=body.investment_artifact_selection,
    )
    commercial_gate = config.commercial_work_start_gate
    if (
      commercial_gate is not None
      and commercial_gate.enabled
      and (
        _active_turn_is_running(session.active_turn)
        or session.stream_active
        or session._commercial_dispatch_owner is not None
      )
    ):
      return JSONResponse(
        {
          "error": "turn_already_running",
          "message": "A turn is already running; subscribe via /chat/subscribe",
        },
        status_code=409,
      )
    raw_purpose = (body.context or {}).get("purpose")
    normalized_purpose = (
      raw_purpose.strip().lower()
      if isinstance(raw_purpose, str) and raw_purpose.strip()
      else None
    )
    if normalized_purpose == "normalizer":
      return JSONResponse(
        {
          "error": "purpose_unavailable",
          "message": "The normalizer chat purpose is unavailable.",
        },
        status_code=400,
      )
    session.purpose = normalized_purpose
    try:
      pending_work_start = None
      if commercial_gate is None:
        if (
          request.headers.get(COMMERCIAL_CLAIM_HEADER) is not None
          or request.headers.get(COMMERCIAL_WORK_AUTHORIZATION_HEADER) is not None
        ):
          return JSONResponse(
            {
              "error": "commercial_work_start_disabled",
              "message": "Commercial work-start authorization is disabled.",
            },
            status_code=403,
          )
      else:
        if commercial_gate.enabled:
          commercial_dispatch_owner = object()
          session._commercial_dispatch_owner = commercial_dispatch_owner
      prepared_turn = prepare_session_driver_turn(
        session,
        inputs,
        build_chat_runtime=app.state.gateway_build_chat_runtime,
      )
      if commercial_gate is not None:
        pending_work_start = commercial_gate.verify_request(
          request.headers,
          session=session,
          request=prepared_turn.request,
          channel=prepared_turn.channel,
        )
        # Keep the bounded local attach synchronous while this opaque owner is
        # held. A detached worker cannot be cancelled safely after SQLite has
        # begun committing the one-time consumption record.
        commercial_work_start = commercial_gate.consume(pending_work_start)
      inputs.commercial_work_start = commercial_work_start
      inputs.commercial_dispatch_owner = commercial_dispatch_owner
    except CapabilityResolutionError as exc:
      if session._commercial_dispatch_owner is commercial_dispatch_owner:
        session._commercial_dispatch_owner = None
      return JSONResponse(
        {
          **exc.receipt(),
          "message": str(exc),
        },
        status_code=400,
      )
    except CommercialWorkStartError as exc:
      if session._commercial_dispatch_owner is commercial_dispatch_owner:
        session._commercial_dispatch_owner = None
      log.warning(
        "Commercial work start rejected | session=%s | code=%s",
        sid,
        exc.code,
      )
      return JSONResponse(
        {"error": exc.code, "message": str(exc)},
        status_code=exc.status_code,
      )
    except HTTPException as exc:
      if session._commercial_dispatch_owner is commercial_dispatch_owner:
        session._commercial_dispatch_owner = None
      return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    except BaseException:
      if session._commercial_dispatch_owner is commercial_dispatch_owner:
        session._commercial_dispatch_owner = None
      raise

    headers = {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-store, must-revalidate",
      "X-Accel-Buffering": "no",
      "Connection": "keep-alive",
    }
    user_event_bus = app.state.user_event_bus
    event_owner_user_id = session_owner_user_id(session)
    publish_event = getattr(user_event_bus, "publish", None)
    cleanup_event_run = getattr(user_event_bus, "cleanup_run", None)
    if not callable(publish_event) or not callable(cleanup_event_run):
      raise RuntimeError(
        "session event delivery bus does not implement its bound domain"
      )

    async def _on_chat_event(event: StreamEvent) -> None:
      event_dict = dict(event)  # type: ignore[arg-type]
      event_dict.setdefault("run_id", sid)
      try:
        await publish_event(
          user_id=event_owner_user_id,
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

    event_log = EventLog(session_id=sid, defer_terminal_close=bool(body.drain_trailing))
    try:
      dispatch_task = asyncio.create_task(
        _dispatch_chat_turn(
          session,
          inputs,
          event_log=event_log,
          on_event=_on_chat_event,
          build_chat_runtime=app.state.gateway_build_chat_runtime,
          transcript_dir=transcript_dir,
          publish_lifecycle_events=True,
          prepared_turn=prepared_turn,
          required_event_delivery=False,
        )
      )
    except BaseException:
      if session._commercial_dispatch_owner is commercial_dispatch_owner:
        session._commercial_dispatch_owner = None
      raise
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
        log.warning(
          "chat turn dispatch failed for %s: %s",
          sid,
          exc,
          exc_info=None if exception_traceback_already_logged(exc) else exc,
        )
      finally:
        await cleanup_event_run(event_owner_user_id, sid)

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

  @router.get("/workflow-outputs/{workflow_run_id}/{output_id}")
  async def workflow_output(
    request: Request,
    workflow_run_id: str,
    output_id: str,
    download: bool = False,
  ) -> Response:
    return await _server_workflow_output_routes.workflow_output_response(
      request,
      workflow_run_id,
      output_id,
      auth=auth,
      download=download,
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

  @router.get("/ui-blocks/{ui_blocks_id}")
  async def ui_blocks_by_id(request: Request, ui_blocks_id: str) -> JSONResponse:
    return _server_artifact_routes.ui_blocks_by_id_response(globals(), request, ui_blocks_id)

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
    approval_delivery_fatal_error = getattr(
      app.state.autonomous_approval_delivery_coordinator,
      "fatal_error",
      None,
    )
    additional_contracts: set[str] = set()
    if config.selected_content_admitter is not None:
      additional_contracts.add(CONTRACT_CHAT_ATTACHMENTS_V1)
      if _has_investment_selected_content_reader(config.mcp_client):
        additional_contracts.add(CONTRACT_INVESTMENT_SELECTED_CONTENT_V1)
    return JSONResponse(
      {
        "status": (
          "error"
          if approval_delivery_fatal_error is not None
          else "ok"
        ),
        "package": package_health(
          additional_contracts=frozenset(additional_contracts)
        ),
        "counters": ui_blocks_metrics_snapshot(),
      },
      status_code=(
        503
        if approval_delivery_fatal_error is not None
        else 200
      ),
    )

  from .control_plane.dashboard_artifacts import build_dashboard_artifacts_router
  from .control_plane.canvas_artifacts import build_canvas_artifacts_router
  from .control_plane.html_artifacts import build_html_artifacts_router

  router.include_router(build_dashboard_artifacts_router(artifact_auth_dependency=_artifact_auth_dependency))
  router.include_router(build_canvas_artifacts_router(artifact_auth_dependency=_artifact_auth_dependency))
  router.include_router(build_html_artifacts_router(artifact_auth_dependency=_artifact_auth_dependency))
  app.include_router(router)
  app.include_router(
    create_control_plane_router(
      auth=auth,
      credentials_resolver=config.credentials_resolver,
      resolver_timeout_seconds=config.resolver_timeout_seconds,
      tenant_id=config.tenant_id,
      allow_service_credentials_for_interactive=(
        config.allow_service_credentials_for_interactive
      ),
      route_prefix=route_prefix,
      skills_dir=control_skills_root,
      artifact_auth_dependency=_artifact_auth_dependency,
      autonomous_registry=app.state.subprocess_registry,
      agent_schedule_store_for=app.state.agent_run_schedule_store_for,
      agent_schedule_runner=app.state.agent_run_schedule_runner,
      approval_store=app.state.gateway_approval_store,
      approval_policy=app.state.gateway_approval_policy,
      dispatch_scope_validator=config.dispatch_scope_validator,
    ),
    prefix=control_prefix,
  )
  app.state.gateway_chat_init = chat_init
  app.state.gateway_chat_stream = chat_stream
  app.state.gateway_control_session = _find_route_endpoint_by_name(
    app.routes,
    "control_session",
  )
  app.state.gateway_control_health = _find_route_endpoint_by_name(
    app.routes,
    "control_health",
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
  "MaterializedCredential",
  "PreparedChatTurn",
  "RequestContext",
  "ToolApprovalRequest",
  "ToolResultRequest",
  "create_gateway_app",
]
