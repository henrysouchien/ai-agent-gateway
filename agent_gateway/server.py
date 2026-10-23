"""Generic FastAPI and SSE composition root for the agent gateway.

``create_gateway_app`` turns application-supplied auth, model policy, runtime
factories, and lifecycle hooks into the package's HTTP surface. Route modules
and ``control_plane/`` own endpoint implementations; ``AgentRunner`` and
``ToolDispatcher`` own model/tool execution. The embedding product remains the
authority for domain policy and resolved capabilities. See
``packages/agent-gateway/README.md``.
"""

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
from typing import Any, TYPE_CHECKING

from fastapi import APIRouter, Body, FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

if TYPE_CHECKING:
  from .mcp_client import McpClientManager

from .auth import (
  ChannelMismatchError,
  CredentialsTimeoutError,
  CrossUserReuseError,
  MissingUserIdError,
)
from .autonomous_runner import AutonomousRegistry
from .skills import SkillLoader
from .directory_control_skill_catalog import DirectoryControlSkillCatalog
from .skill_limits import (
  AutonomousSkillAdmissionPolicy,
  AutonomousSkillAdmissionPolicyResolver,
  SkillExecutionLimits,
)
from .control_plane import create_control_plane_router
from .control_plane.middleware import add_control_plane_version_header_middleware
from .control_plane.session import _resolve_control_identity
from hank_contracts.commercial_work import CommercialWorkStartError
from model_authority.binding import CapabilityResolutionError, eligible_model_choices
from model_authority.capabilities import GATEWAY_EXECUTED_CAPABILITY_IDS
from .event_log import EventLog, UserEventBus
from .approval_route import bind_session_approval_route
from .dispatcher_factory import GatewayDispatcherDeps
from .approvals import ApprovalActionError, _record_vote_and_unblock  # noqa: F401
from .approval_store import maintain_pending_loop
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
  installed_adapter_providers,
)
from .runner_introspection import exception_traceback_already_logged
from .session import (
  AuthManager,
  GatewaySession,
  SessionStore,
  SessionStream,
  bind_session_credentials,
  session_owner_user_id,
)
from .undo_token_lifetime import (
  UndoTokenLifetimeRegistry,
  cancel_parked_approvals,
  conversation_id as undo_conversation_id,
  guard_root_continue,
  owner_id as undo_owner_id,
  observe_runner_results,
  create_undo_token_router,
)
from .ui_blocks_metrics import snapshot as ui_blocks_metrics_snapshot
from .launch_nonce_store import (
  launch_nonce_store_path,
  launch_nonce_store_row_count,
)

from . import server_chat_helpers as _server_chat_helpers  # noqa: F401 - dynamic streaming deps alias
from . import server_chat_control_routes as _server_chat_control_routes
from . import server_streaming as _server_streaming
from . import server_tool_routes as _server_tool_routes
from . import server_workflow_output_routes as _server_workflow_output_routes
from .server_models import (  # noqa: F401
  SystemPrompt,
  BuildChatRuntime,
  RequestApproval,
  BuildRunner,
  _AGENT_API_CLAIM_AUDIENCE,
  _STREAM_SUBSCRIBER_QUEUE_MAX,
  _STREAM_SUBSCRIBER_KEEPALIVE_SECONDS,
  _STREAM_SUBSCRIBER_DONE,
  ChatInitRequest,
  ChatInitResponse,
  ChatMessage,
  ChatRequest,
  ChatRecapRequest,
  ChatCancelRequest,
  ChatTodoPatchRequest,
  ChatTurnInputs,
  ChatTurnResult,
  PreparedChatTurn,
  ToolResultRequest,
  ToolApprovalRequest,
  ChatRuntime,
  ModelPreferenceResponse,
  ModelPreferenceUpdate,
  ModelPresetResponse,
  ModelPresetUpdate,
  _call_build_chat_runtime,
  RequestContext,
  GatewayServerConfig,
  MaterializedCredential,
  SessionResumeRefusalReason,
)
from .server_artifact_helpers import (  # noqa: F401
  _normalize_prefix,
  _route_path,
  _default_control_skills_dir,
  _default_autonomous_api_dir,
  _default_autonomous_log_dir,
  _resolve_compaction_trigger,
  _json_dumps,
  _artifact_auth_dependency,
  _normalize_request_user_id,
  _resolver_contract_payload,
  _error_payload,
)
from .server_chat_helpers import (  # noqa: F401
  _now_iso,
  _write_transcript,
  _cleanup_old_transcripts,
  _compute_session_recap_payload,
  _compute_cumulative_session_recap_payload,
  _redact_tool_input_for_event,
  _cleanup_sessions_loop,
  _maybe_await,
  _clear_active_turn,
  _active_turn_is_running,
  _cancel_active_turn_runner,
  _cleanup_active_turn_on_expiry,
  _event_for_wire,
  _resolve_schema_version,
  _make_request_approval,
  bind_init_capability_selections,
  build_capability_choices,
  _capability_execution_resolver_for_session,
  prepare_session_driver_turn,
  _dispatch_chat_turn,
  _await_dispatched_turn,
  _init_approval_subsystem,
)


def _generic_skill_admission_policy_resolver(
  skills_dir: Path,
) -> AutonomousSkillAdmissionPolicyResolver:
  loader = SkillLoader(skills_dir)

  def resolve(skill_name: str) -> AutonomousSkillAdmissionPolicy:
    profile = loader.load(skill_name)
    return AutonomousSkillAdmissionPolicy(
      skill_resume_allowed=(
        bool(profile.resumable)
        and profile.mutation_mode != "model_writer"
      ),
      execution_limits=SkillExecutionLimits(max_tokens=profile.max_tokens),
    )

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


def _has_investment_selected_content_reader(
  mcp_client: McpClientManager | None,
) -> bool:
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


class _CapabilityAdapterFamilyMismatch(ValueError):
  """Installed resolver returned a provider family the registry does not name.

  Wiring, not artifact data: startup still fails.  Distinct so the closure's
  observe-and-allow catch does not swallow it.
  """


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
  from .policy_imports import configure_server_policy

  configure_server_policy(config.server_policy)
  build_chat_runtime = config.build_chat_runtime
  if build_chat_runtime is None:
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
  model_registry = config.model_registry
  model_selection_policy = config.model_selection_policy
  if model_registry is None or model_selection_policy is None:
    raise ValueError("model_registry and model_selection_policy are required")
  if not str(config.tenant_id or "").strip():
    raise ValueError("tenant_id is required with model-selection authority")
  model_selection_policy.admit_registry(model_registry)

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
        if (
          (support := provider_class.adapter_route_support()) is not None
          and support.provider == configured_family
        )
      ),
      None,
    )
    if default_adapter is not None:
      capability_adapter_cache[default_adapter] = configured_default_provider
  log = logging.getLogger(config.log_name)

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
      for entry in model_registry.models.values()
      if entry.adapter == normalized_adapter
    }
    resolved_family = str(getattr(provider, "name", "") or "").strip().lower()
    if resolved_family == "agent-sdk":
      resolved_family = "anthropic"
    if expected_providers != {resolved_family}:
      raise _CapabilityAdapterFamilyMismatch(
        f"capability adapter {normalized_adapter!r} returned provider "
        f"{resolved_family!r}; expected {sorted(expected_providers)}"
      )

    # On the built-in path the installed class must declare this adapter id.
    # A gateway-executed entry whose profile or route that declaration does
    # not cover is logged and left for selection to skip
    # (``provider_unavailable``); the adapter still resolves for the entries
    # it does declare.  Deployment-vouched implementations skip this check.
    if not deployment_vouched:
      declaration = provider.adapter_route_support()
      if declaration is None:
        raise ValueError(
          f"capability adapter {normalized_adapter!r} declares no protocol support"
        )
      if declaration.adapter != normalized_adapter:
        raise ValueError(
          f"capability adapter {normalized_adapter!r} resolved an "
          f"implementation declaring {declaration.adapter!r}"
        )
      for entry in model_registry.models.values():
        if entry.adapter != normalized_adapter:
          continue
        if not (set(entry.capabilities) & GATEWAY_EXECUTED_CAPABILITY_IDS):
          continue
        if not declaration.supports(entry):
          log.warning(
            "registry entry %r requires protocol profile %r on route %r, which "
            "installed adapter %r does not declare; selection skips it",
            entry.key,
            entry.protocol_profile,
            entry.route,
            normalized_adapter,
          )
    capability_adapter_cache[normalized_adapter] = provider
    return provider

  # Observe installed adapters against the configured registry.  An entry
  # whose adapter is not installed, or whose installed adapter does not
  # declare its profile and route, is logged and skipped at selection
  # (``provider_unavailable``) — the artifact deploys ahead of this process's
  # code, so that fact is not a startup failure.  A resolver that returns the
  # wrong provider family is wiring and still fails startup.  Entries serving
  # only externally-executed capabilities are excluded by designation; they
  # are facts for their own processes.
  for adapter_id in sorted({
    entry.adapter
    for entry in model_registry.models.values()
    if set(entry.capabilities) & GATEWAY_EXECUTED_CAPABILITY_IDS
  }):
    try:
      _resolve_capability_adapter(adapter_id)
    except _CapabilityAdapterFamilyMismatch:
      raise
    except (LookupError, ValueError) as exc:
      log.warning(
        "capability adapter %r is unavailable in this process (%s); "
        "its entries are skipped at selection",
        adapter_id,
        exc,
      )

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
  route_prefix = _normalize_prefix(config.prefix)

  @asynccontextmanager
  async def lifespan(app: FastAPI):
    # The product host installs its root hook on app.state after this app is built;
    # guard whatever it installed so a closed conversation cannot be cold-restored.
    app.state.root_continue_hook = guard_root_continue(
      app.state.root_continue_hook, app, config.tenant_id,
    )
    cleanup_task = asyncio.create_task(
      _cleanup_sessions_loop(
        auth.session_store,
        transcript_dir=transcript_dir,
        transcript_retention_days=config.transcript_retention_days,
      )
    )
    approval_maintenance_task = asyncio.create_task(maintain_pending_loop(app.state.gateway_approval_store))
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

    async def _drain_active_turns_on_shutdown() -> None:
      # Cancel and await every live turn's runner (and the child runners it
      # awaits) while the MCP client is still up, so each reaches
      # `_finalize_run` and writes its `on_session_summary` row.
      for session in list(auth.session_store.sessions.values()):
        active_turn = session.active_turn
        if active_turn is None or not _active_turn_is_running(active_turn):
          continue
        try:
          await _cancel_active_turn_runner(active_turn)
          _clear_active_turn(session, active_turn)
        except Exception:
          log.warning(
            "active turn shutdown drain failed for %s",
            session.session_id,
            exc_info=True,
          )

    async def teardown() -> None:
      await cleanup_phase(
        "active session turns",
        _drain_active_turns_on_shutdown,
      )
      if config_started:
        await cleanup_phase(
          "configured shutdown",
          lambda: _maybe_await(config.on_shutdown),
        )

      maintenance_tasks = [
        cleanup_task,
        approval_maintenance_task,
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
  dispatcher_deps: GatewayDispatcherDeps | None = None
  async def _build_chat_runtime_for_dispatch(
    session: GatewaySession,
    request: ChatRequest,
    channel: str | None,
    auth_manager: AuthManager | None,
    *,
    # The app-owned autonomous root below is authoritative.
    storage_root: Path | None = None,
  ) -> ChatRuntime:
    _ = auth_manager, storage_root
    if dispatcher_deps is not None:
      session._gateway_dispatcher_deps = dispatcher_deps
    try:
      from .control_plane.valuation_ready_tools import make_valuation_ready_skill_tool_bundle

      session.gateway_local_skill_tools = [
        make_valuation_ready_skill_tool_bundle(app_state=app.state, session=session)
      ]
    except Exception:
      log.warning("Gateway-local skill tool injection failed", exc_info=True)
    runtime = await _call_build_chat_runtime(
      build_chat_runtime,
      session=session,
      request=request,
      channel=channel,
      auth_manager=auth,
      storage_root=app.state.autonomous_storage_root,
    )
    runtime.purpose = session.purpose
    observe_runner_results(runtime, session, app.state.undo_token_lifetimes)
    return runtime

  setattr(_build_chat_runtime_for_dispatch, "_gateway_auth_manager", auth)
  setattr(
    _build_chat_runtime_for_dispatch,
    "_gateway_model_registry",
    model_registry,
  )
  setattr(
    _build_chat_runtime_for_dispatch,
    "_gateway_model_selection_policy",
    model_selection_policy,
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
  app.state.root_continue_hook = config.root_continue_hook
  app.state.workflow_continue_hook = config.workflow_continue_hook
  app.state.child_continue_hook = config.child_continue_hook
  _init_approval_subsystem(app, config)
  dispatcher_deps = GatewayDispatcherDeps(
    mcp_client=config.mcp_client,
    approval_store=app.state.gateway_approval_store,
    approval_policy=app.state.gateway_approval_policy,
    mcp_meta_inject_servers=(
      config.mcp_meta_inject_servers or frozenset()
    ),
    tool_registration_catalog=config.tool_registration_catalog,
    tool_policy_implementations=config.tool_policy_implementations,
    redaction_context_factory=config.redaction_context_factory,
  )
  app.state.gateway_dispatcher_deps = dispatcher_deps
  app.state.gateway_claim_signing_authority = (
    config.claim_signing_authority
  )
  app.state.gateway_skill_application = config.skill_application
  autonomous_storage_root = _default_autonomous_log_dir()
  if (
    config.control_skill_catalog is not None
    and config.control_skills_dir is not None
  ):
    raise ValueError(
      "control_skill_catalog and control_skills_dir are mutually exclusive"
    )
  control_skills_root: Path | None = None
  if config.control_skill_catalog is not None:
    control_skill_catalog = config.control_skill_catalog
  else:
    control_skills_root = (
      config.control_skills_dir or _default_control_skills_dir()
    )
    control_skill_catalog = DirectoryControlSkillCatalog(
      control_skills_root
    )
  autonomous_skill_admission_policy_resolver = (
    config.autonomous_skill_admission_policy_resolver
  )
  if autonomous_skill_admission_policy_resolver is None:
    if control_skills_root is None:
      control_skills_root = _default_control_skills_dir()
    autonomous_skill_admission_policy_resolver = (
      _generic_skill_admission_policy_resolver(control_skills_root)
    )

  def _issue_autonomous_session_token(session) -> str:
    auth.session_store.register_session(session)
    try:
      return auth.issue_token(session)
    except BaseException:
      auth.session_store.expire_session(session.session_id)
      raise

  async def _expire_autonomous_session(session_id: str) -> None:
    await auth.session_store.expire_session_async(session_id)

  app.state.autonomous_storage_root = autonomous_storage_root
  app.state.subprocess_registry = AutonomousRegistry(
    api_dir=(
      config.autonomous_api_dir
      if config.autonomous_api_dir is not None
      else _default_autonomous_api_dir()
    ),
    tenant_id=config.tenant_id,
    identity_resolver=config.identity_resolver,
    mcp_user_key_lookup=config.mcp_user_key_lookup,
    python_executable=os.getenv("AGENT_GATEWAY_AUTONOMOUS_PYTHON", "").strip() or sys.executable,
    log_dir=autonomous_storage_root,
    max_running=int(os.getenv("AGENT_GATEWAY_AUTONOMOUS_MAX_RUNNING", "2") or "2"),
    approval_store=app.state.gateway_approval_store,
    approval_policy=app.state.gateway_approval_policy,
    service_provider_handles=config.service_provider_handles,
    autonomous_capability_binding_resolver=(
      config.autonomous_capability_binding_resolver
    ),
    autonomous_selection_policy=config.autonomous_selection_policy,
    autonomous_skill_admission_policy_resolver=(
      autonomous_skill_admission_policy_resolver
    ),
    claim_signing_authority=config.claim_signing_authority,
    autonomous_session_token_issuer=_issue_autonomous_session_token,
    autonomous_session_expirer=_expire_autonomous_session,
  )
  app.state.undo_token_lifetimes = UndoTokenLifetimeRegistry(
    app.state.subprocess_registry._log_dir / "workbook-undo.sqlite3"
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
  app.state.batch_backend = config.batch_backend
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
  async def chat_init(
    payload: ChatInitRequest,
    response: Response,
  ) -> ChatInitResponse | JSONResponse:
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
    resolved_unknown_model_entitled_capabilities: frozenset[str] = frozenset()
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
      resolved_unknown_model_entitled_capabilities = (
        result.unknown_model_entitled_capabilities
      )
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
        identity_resolver=config.identity_resolver,
      )
    except (ValueError, SystemExit) as exc:
      status, error_payload = _resolver_contract_payload(str(exc), user_id=resolved_user_id)
      return JSONResponse(error_payload, status_code=status)

    api_key_hash = AuthManager.hash_api_key(payload.api_key)
    resume_log_path: Path | None = None
    fork_source_log_path: Path | None = None
    presented_log_session_id = (
      payload.resume_session_id
      if payload.resume_session_id is not None
      else payload.fork_from_session_id
    )
    if presented_log_session_id is not None:
      # Resolve, check and bind with no await in between: two inits presenting
      # one log cannot both pass the live-session check.
      resume_resolver = config.session_resume_resolver
      resume_outcome: Path | SessionResumeRefusalReason = (
        resume_resolver(
          resume_session_id=presented_log_session_id,
          user_id=resolved_user_id,
          channel=resolved_channel,
          api_key_hash=api_key_hash,
        )
        if resume_resolver is not None
        else "not_found"
      )
      if isinstance(resume_outcome, Path) and payload.resume_session_id is not None:
        prior_session = auth.session_store.live_session_bound_to_log(
          resume_outcome
        )
        if prior_session is not None:
          # The resolver proved this log is the caller's (same user and api
          # key), so the session holding it is the caller's own: a turn still
          # answering keeps it; an idle one is retired as the new one binds,
          # so two live sessions never hold one log.
          prior_turn = prior_session.active_turn
          if prior_turn is not None and prior_turn.in_progress:
            resume_outcome = "in_use"
          else:
            auth.session_store.expire_session(prior_session.session_id)
      if not isinstance(resume_outcome, Path):
        return JSONResponse(
          {"error": "session_resume_refused", "reason": resume_outcome},
          status_code=409,
        )
      if payload.resume_session_id is not None:
        resume_log_path = resume_outcome
      else:
        # A fork only reads the caller's log as its starting history, so the
        # session holding that log keeps it, idle or answering.
        fork_source_log_path = resume_outcome

    session = auth.session_store.create_session(
      api_key_hash=api_key_hash,
      user_id=resolved_user_id,
      user_email=identity.user_email,
      risk_user_id=identity.risk_user_id,
      role=resolved_role,
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
    session.resume_log_path = resume_log_path
    session.fork_source_log_path = fork_source_log_path
    if app.state.undo_token_lifetimes.is_closed(
      owner=undo_owner_id(session), conversation=undo_conversation_id(session),
    ):
      auth.session_store.expire_session(session.session_id)
      return JSONResponse(
        {"error": "session_resume_refused", "reason": "not_found"},
        status_code=409,
      )
    session.is_public = resolved_channel == "public"
    bind_session_approval_route(
      session,
      app.state.gateway_approval_store,
      app.state.gateway_approval_policy,
    )
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
        model_preset=payload.model_preset,
        build_chat_runtime=app.state.gateway_build_chat_runtime,
      )
    except CapabilityResolutionError as exc:
      auth.session_store.expire_session(session.session_id)
      return JSONResponse(
        {
          **exc.to_error(),
          "message": str(exc),
        },
        status_code=422,
      )
    except Exception:
      auth.session_store.expire_session(session.session_id)
      raise
    token = auth.issue_token(session)
    log.info(
      "Session created: %s%s",
      session.session_id,
      (
        f" (resumed from {payload.resume_session_id})"
        if resume_log_path is not None
        else f" (forked from {payload.fork_from_session_id})"
        if fork_source_log_path is not None
        else ""
      ),
    )
    response.headers["Cache-Control"] = "private, no-store"
    return ChatInitResponse(
      user_id=resolved_user_id,
      session_token=token,
      unknown_model_entitled_capabilities=sorted(
        resolved_unknown_model_entitled_capabilities
      ),
      session_id=session.session_id,
      expires_at=session.expires_at,
      schema_version=session.schema_version,
      capability_choices=build_capability_choices(
        session=session,
        build_chat_runtime=app.state.gateway_build_chat_runtime,
      ),
      resumed_from_session_id=(
        payload.resume_session_id if resume_log_path is not None else None
      ),
      forked_from_session_id=(
        payload.fork_from_session_id if fork_source_log_path is not None else None
      ),
    )

  def _model_preference_session(request: Request) -> GatewaySession:
    token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
    session, _claims = auth.verify_token_with_payload(token)
    return session

  @router.put(
    "/model-preferences/{capability_id}",
    response_model=ModelPreferenceResponse,
  )
  async def put_model_preference(
    capability_id: str,
    request: Request,
    body: ModelPreferenceUpdate,
  ) -> ModelPreferenceResponse | JSONResponse:
    session = _model_preference_session(request)
    store = config.model_preference_store
    if store is None:
      raise HTTPException(status_code=503, detail="model preference store is unavailable")
    normalized_capability = str(capability_id or "").strip()
    policy = model_selection_policy.capabilities.get(normalized_capability)
    if policy is None:
      raise HTTPException(
        status_code=422,
        detail=f"no model-selection policy exists for {normalized_capability!r}",
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

  @router.put("/model-presets/self", response_model=ModelPresetResponse)
  async def put_model_preset(
    request: Request,
    body: ModelPresetUpdate,
  ) -> ModelPresetResponse | JSONResponse:
    session = _model_preference_session(request)
    store = config.model_preference_store
    if store is None:
      raise HTTPException(status_code=503, detail="model preference store is unavailable")
    known_preset_names = store.known_preset_names
    if body.name not in known_preset_names:
      return JSONResponse(
        {
          "error_code": "model_preset_unknown",
          "name": body.name,
          "policy_revision": model_selection_policy.revision,
          "known_preset_names": sorted(known_preset_names),
          "message": "The requested preset is not published by model-selection policy.",
        },
        status_code=422,
      )
    resolver = _capability_execution_resolver_for_session(
      session=session,
      build_chat_runtime=app.state.gateway_build_chat_runtime,
    )
    stored = store.put_user_preset(
      tenant_id=resolver.auth_context.tenant_id,
      actor_id=resolver.auth_context.actor_id,
      name=body.name,
      policy_revision=model_selection_policy.revision,
    )
    return ModelPresetResponse(
      name=stored.name,
      policy_revision=stored.policy_revision,
      updated_at=stored.updated_at,
      known_preset_names=sorted(known_preset_names),
    )

  @router.get("/model-presets/self", response_model=ModelPresetResponse)
  async def get_model_preset(request: Request) -> ModelPresetResponse:
    session = _model_preference_session(request)
    store = config.model_preference_store
    if store is None:
      raise HTTPException(status_code=503, detail="model preference store is unavailable")
    resolver = _capability_execution_resolver_for_session(
      session=session,
      build_chat_runtime=app.state.gateway_build_chat_runtime,
    )
    stored = store.get_user_preset(
      tenant_id=resolver.auth_context.tenant_id,
      actor_id=resolver.auth_context.actor_id,
    )
    return ModelPresetResponse(
      name=stored.name if stored is not None else None,
      policy_revision=stored.policy_revision if stored is not None else None,
      updated_at=stored.updated_at if stored is not None else None,
      known_preset_names=sorted(store.known_preset_names),
    )

  @router.delete("/model-presets/self", response_model=ModelPresetResponse)
  async def delete_model_preset(request: Request) -> ModelPresetResponse:
    session = _model_preference_session(request)
    store = config.model_preference_store
    if store is None:
      raise HTTPException(status_code=503, detail="model preference store is unavailable")
    resolver = _capability_execution_resolver_for_session(
      session=session,
      build_chat_runtime=app.state.gateway_build_chat_runtime,
    )
    store.delete_user_preset(
      tenant_id=resolver.auth_context.tenant_id,
      actor_id=resolver.auth_context.actor_id,
    )
    return ModelPresetResponse(
      name=None, policy_revision=None, updated_at=None,
      known_preset_names=sorted(store.known_preset_names),
    )

  @router.post("/chat", response_model=None)
  async def chat_stream(
    request: Request,
    body: ChatRequest = Body(...),
  ) -> StreamingResponse | JSONResponse:
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
    if app.state.undo_token_lifetimes.is_closed(
      owner=undo_owner_id(session), conversation=undo_conversation_id(session),
    ):
      return JSONResponse({"error": "conversation_closed"}, status_code=409)
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
    # A control bearer is the owner's control-plane credential and owns no chat
    # transcript or runner; a chat run is started by POST /control/runs
    # kind=chat, which mints the chat session. This is the only path on which a
    # control session can reach turn dispatch (the control-run paths dispatch
    # chat-kind sessions only), checked before any commercial authorization is
    # consumed.
    if session.kind != "chat":
      return JSONResponse(
        {
          "error": "invalid_session_kind",
          "message": (
            "control sessions own no chat transcript; start a chat run with "
            "POST /control/runs kind=chat, which mints the chat session"
          ),
        },
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
    # A session admits one active turn (its transcript and runner state take
    # one writer); under commercial gating this is checked before consume so a
    # one-time work authorization is never spent on a turn that cannot attach.
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
    session.purpose = (
      raw_purpose.strip().lower()
      if isinstance(raw_purpose, str) and raw_purpose.strip()
      else None
    )
    try:
      verified_commercial_claim = None
      if commercial_gate is not None:
        verified_commercial_claim = commercial_gate.verify_claim(
          request.headers,
          session=session,
        )
        if commercial_gate.enabled:
          commercial_dispatch_owner = object()
          session._commercial_dispatch_owner = commercial_dispatch_owner
      prepared_turn = prepare_session_driver_turn(
        session,
        inputs,
        build_chat_runtime=app.state.gateway_build_chat_runtime,
        commercial_account_id=(
          verified_commercial_claim.claim.commercial_account_id
          if verified_commercial_claim is not None
          else None
        ),
      )
      if commercial_gate is not None:
        pending_work_start = commercial_gate.authorize_request(
          verified_commercial_claim,
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
          **exc.to_error(),
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
    user_event_bus: UserEventBus = app.state.user_event_bus
    event_owner_user_id = session_owner_user_id(session)
    if TYPE_CHECKING:
      publish_event = user_event_bus.publish
      cleanup_event_run = user_event_bus.cleanup_run
    else:
      publish_event = getattr(user_event_bus, "publish", None)
      cleanup_event_run = getattr(user_event_bus, "cleanup_run", None)
    if not callable(publish_event) or not callable(cleanup_event_run):
      raise RuntimeError(
        "session event delivery bus does not implement its bound domain"
      )

    async def _on_chat_event(event: dict[str, Any]) -> None:
      event_dict = dict(event)
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

    # A chat turn keeps its log open past the terminal until the turn ends, so
    # the answer turn's citation_validation record still lands after the client
    # is released (RunnerWithCitationValidation); the dispatch closes the log.
    event_log = EventLog(session_id=sid, defer_terminal_close=True)
    active_turn = SessionStream(event_log=event_log, runner_task=None)
    try:
      dispatch_task = asyncio.create_task(
        _dispatch_chat_turn(
          session,
          inputs,
          turn=active_turn,
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
    await _await_dispatched_turn(active_turn, dispatch_task)
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
        await _cleanup_stream_subscriber(active_turn, subscriber.subscriber_id)
        cleanup_complete = True

    async def event_generator():
      try:
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

  async def _append_todo_snapshot_event(session: Any, event: dict) -> None:
    """Persist the accepted checklist change to the session's durable log."""
    log = getattr(session, "agent_session_log", None)
    if log is not None:
      await log.append(event)

  def _emit_todo_snapshot_event(session: Any, event: dict) -> None:
    """Push the same event onto the session's live event log so clients see it."""
    active = getattr(session, "active_turn", None)
    if active is not None and getattr(active, "event_log", None) is not None:
      active.event_log.append(event)

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

  @router.get("/chat/todo")
  async def chat_todo_get(request: Request) -> JSONResponse:
    return await _server_chat_control_routes.chat_todo_get_response(
      request,
      auth=auth,
      get_bearer_token=AuthManager.get_bearer_token,
    )

  @router.patch("/chat/todo")
  async def chat_todo_patch(
    request: Request,
    body: ChatTodoPatchRequest = Body(...),
  ) -> JSONResponse:
    return await _server_chat_control_routes.chat_todo_patch_response(
      request,
      body,
      auth=auth,
      get_bearer_token=AuthManager.get_bearer_token,
      append_todo_snapshot=_append_todo_snapshot_event,
      emit_to_session=_emit_todo_snapshot_event,
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


  @router.post("/chat/tool-result")
  async def tool_result(request: Request, payload: ToolResultRequest) -> JSONResponse:
    return await _server_tool_routes.tool_result_response(
      request,
      payload,
      auth=auth,
      log=log,
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
        "status": "ok",
        "package": package_health(
          additional_contracts=frozenset(additional_contracts)
        ),
        "counters": ui_blocks_metrics_snapshot(),
        # Informational: the launch-nonce ledger's admission row count is a
        # measurement and never changes ``status`` or the HTTP code.
        "autonomous_admission_ledger": {
          "rows": launch_nonce_store_row_count(
            launch_nonce_store_path(
              app.state.subprocess_registry._log_dir.resolve()
            )
          ),
        },
        # Informational: a failing approval-delivery coordinator is product
        # state reported for its readers; it never changes ``status`` or the
        # HTTP code, which assert only that this process is serving.
        "approval_delivery": {
          "fatal_error": approval_delivery_fatal_error,
        },
      },
    )

  async def _end_undo_conversation(session: GatewaySession, conversation: str) -> None:
    owner = undo_owner_id(session)
    registry = app.state.undo_token_lifetimes
    registry.begin_close(owner=owner, conversation=conversation)
    session_ids = registry.session_ids(owner=owner, conversation=conversation)
    # Auth expiry is deliberately absent: only this explicit end retires holders.
    for candidate in auth.session_store.visible_sessions_snapshot():
      if undo_owner_id(candidate) == owner and undo_conversation_id(candidate) == conversation:
        session_ids.add(candidate.session_id)
        if candidate.active_turn is not None:
          await _cancel_active_turn_runner(candidate.active_turn)
        await auth.session_store.expire_session_async(candidate.session_id)
    # After turns are cancelled nothing can park a new approval; deny the parked ones.
    await cancel_parked_approvals(
      app.state.gateway_approval_store, app.state.gateway_approval_policy,
      tenant_storage_id=session_owner_user_id(session), session_ids=session_ids,
    )
    registry.close_conversation(owner=owner, conversation=conversation)

  router.include_router(create_undo_token_router(
    auth=auth, registry=app.state.undo_token_lifetimes,
    end_conversation=_end_undo_conversation,
  ))

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
      control_skill_catalog=control_skill_catalog,
      control_profile_names_provider=config.control_profile_names_provider,
      control_profile_loader=config.control_profile_loader,
      identity_resolver=config.identity_resolver,
      autonomous_registry=app.state.subprocess_registry,
      agent_schedule_store_for=app.state.agent_run_schedule_store_for,
      agent_schedule_runner=app.state.agent_run_schedule_runner,
      operator_schedule_backend=config.operator_schedule_backend,
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
