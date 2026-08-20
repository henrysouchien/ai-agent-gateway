from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional

from fastapi import FastAPI, HTTPException

from .agent_session_log import _atomic_write_sidecar
from .approval_audit import ApprovalAuditEmitter
from .approval_notifications import (
  build_env_approval_notification_destination_resolver,
  build_env_telegram_approval_notification_sender,
)
from .approval_resolver import resolve_policy
from .approval_store import SQLiteApprovalStore, resolve_approval_db_path
from .audit_resolver import resolve_audit_writer
from .capability_binding import (
  CAPABILITY_IDS,
  AuthContext,
  CapabilityId,
  CapabilityResolutionError,
  CredentialHandle,
  ModelSelectionIntent,
  eligible_model_choices,
  resolve_capability_model,
  saved_preference_ineligibility,
)
from .capability_execution import CapabilityExecutionResolver
from .control_run_lifecycle import coerce_control_run_state
from .model_registry import (
  GATEWAY_EXECUTED_CAPABILITY_IDS,
  ProductModelRegistry,
  ProductModelSelectionPolicy,
)
from .event_adapter import adapt_event
from .event_log import EventLog, log_has_terminal
from .events import DEFAULT_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS
from .product_config import gateway_product_id
from .providers import StreamEvent
from .runner_introspection import exception_traceback_already_logged
from .session import (
  GatewaySession,
  SessionStore,
  SessionStream,
  StreamSubscriber,
  bind_session_capability_selections,
)
from .session_recap import emit_recap_then_terminal
from .selected_content import SelectedContentAdmission
from . import server_chat_stream_core as _chat_stream_core
from . import server_chat_transcripts as _chat_transcripts
from .tool_dispatcher import ApprovalDecision, ApprovalRequest
from .tool_redaction import get_audit_hmac_secret, redact_tool_input as default_redact_tool_input
from .ui_blocks_run import (
  UiBlocksRunContext,
  UiBlocksRunRegistry,
  activate_ui_blocks_run,
  reset_ui_blocks_run,
)
from .server_artifact_helpers import _json_dumps, _model_to_dict
from .server_models import (
  BuildChatRuntime,
  CapabilityChoice,
  CapabilityChoiceNotice,
  CapabilityChoiceResponse,
  CapabilityChoiceSelection,
  CapabilitySelection,
  ChatRequest,
  ChatRuntime,
  ChatTurnInputs,
  ChatTurnResult,
  GatewayServerConfig,
  MaterializedCredential,
  PreparedChatTurn,
  RequestApproval,
  SessionExecutionPolicy,
  _ACTIVE_TURN_GRACE_SECONDS,
  _STREAM_SUBSCRIBER_DONE,
  _STREAM_SUBSCRIBER_KEEPALIVE_SECONDS,
  _STREAM_SUBSCRIBER_QUEUE_MAX,
  _build_runner_with_started_at,
  _call_build_chat_runtime,
  _resolve_chat_profile_name,
  log,
)

def _drain_result_queue(queue: Optional[asyncio.Queue]) -> None:
  if queue is None:
    return
  while True:
    try:
      queue.get_nowait()
    except asyncio.QueueEmpty:
      break


def _now_iso() -> str:
  return _chat_transcripts._now_iso()


def _sidecar_slug(value: str | None) -> str | None:
  return _chat_transcripts._sidecar_slug(value)


def _maybe_write_chat_log_meta(
  transcript_dir: Path,
  session_id: str,
  *,
  user_id: str | None,
  channel: str | None,
) -> None:
  return _chat_transcripts._maybe_write_chat_log_meta(
    transcript_dir,
    session_id,
    user_id=user_id,
    channel=channel,
    atomic_write_sidecar=_atomic_write_sidecar,
    now_iso=_now_iso,
    product_id_resolver=gateway_product_id,
    sidecar_slug=_sidecar_slug,
  )


def _write_transcript(
  transcript_dir: Optional[Path],
  session_id: str,
  entry: Dict[str, Any],
  *,
  user_id: str | None = None,
  channel: str | None = None,
) -> None:
  return _chat_transcripts._write_transcript(
    transcript_dir,
    session_id,
    entry,
    user_id=user_id,
    channel=channel,
    write_chat_log_meta=_maybe_write_chat_log_meta,
    warn=log.warning,
  )


def _cleanup_old_transcripts(
  transcript_dir: Optional[Path],
  retention_days: int,
  *,
  now: float | None = None,
) -> int:
  return _chat_transcripts._cleanup_old_transcripts(transcript_dir, retention_days, now=now)


def _compute_session_recap_payload(
  session: GatewaySession,
  active_turn: SessionStream,
  *,
  trigger: str,
) -> Dict[str, Any]:
  return _chat_transcripts._compute_session_recap_payload(session, active_turn, trigger=trigger)


def _read_session_transcript_events(
  transcript_dir: Optional[Path],
  session_id: str,
) -> list[Dict[str, Any]]:
  return _chat_transcripts._read_session_transcript_events(transcript_dir, session_id)


def _compute_cumulative_session_recap_payload(
  session: GatewaySession,
  active_turn: SessionStream | None,
  transcript_dir: Optional[Path],
  *,
  trigger: str,
) -> Dict[str, Any]:
  return _chat_transcripts._compute_cumulative_session_recap_payload(
    session,
    active_turn,
    transcript_dir,
    trigger=trigger,
    event_for_wire=_event_for_wire,
  )


def _redact_tool_input_for_event(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
  try:
    return default_redact_tool_input(
      tool_name,
      tool_input,
      deployment_secret=get_audit_hmac_secret(),
    )
  except Exception:
    from .secret_boundary import sanitization_failure_tool_input

    return sanitization_failure_tool_input()


async def _cleanup_sessions_loop(
  session_store: SessionStore,
  *,
  transcript_dir: Optional[Path] = None,
  transcript_retention_days: int = 7,
) -> None:
  while True:
    await asyncio.sleep(300)
    await session_store.cleanup_expired_async()
    try:
      _cleanup_old_transcripts(transcript_dir, transcript_retention_days)
    except Exception:
      log.warning("Transcript retention cleanup failed", exc_info=True)


async def _maybe_await(callback: Optional[Callable[..., Any]]) -> None:
  if callback is None:
    return
  result = callback()
  if inspect.isawaitable(result):
    await result


def _cancel_active_turn_cleanup_handle(active_turn: SessionStream) -> None:
  cleanup_handle = active_turn.cleanup_handle
  if cleanup_handle is None:
    return
  cleanup_handle.cancel()
  active_turn.cleanup_handle = None


def _clear_active_turn(session: GatewaySession, active_turn: SessionStream) -> None:
  if session.active_turn is not active_turn:
    return
  _cancel_active_turn_cleanup_handle(active_turn)
  for subscriber in active_turn.subscribers.values():
    pump_task = subscriber.pump_task
    if pump_task is not None and not pump_task.done():
      pump_task.cancel()
  active_turn.subscribers.clear()
  session.active_turn = None
  session.stream_active = False


def _active_turn_is_running(active_turn: SessionStream | None) -> bool:
  return active_turn is not None and active_turn.is_running


def _schedule_active_turn_clear(session: GatewaySession, active_turn: SessionStream) -> None:
  if session.active_turn is not active_turn:
    return
  _cancel_active_turn_cleanup_handle(active_turn)
  loop = asyncio.get_running_loop()
  active_turn.cleanup_handle = loop.call_later(_ACTIVE_TURN_GRACE_SECONDS, _clear_active_turn, session, active_turn)


async def _cancel_active_turn_runner(active_turn: SessionStream) -> None:
  _cancel_active_turn_cleanup_handle(active_turn)
  task = active_turn.runner_task
  if task is not None and not task.done():
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _cleanup_active_turn_on_expiry(session: GatewaySession) -> None:
  active_turn = session.active_turn
  if active_turn is None:
    return
  await _cancel_active_turn_runner(active_turn)
  _clear_active_turn(session, active_turn)


def _event_for_wire(entry: Any, event_log: EventLog) -> Dict[str, Any]:
  return _chat_stream_core.event_for_wire(entry, event_log, product_id_resolver=gateway_product_id)


def _resolve_schema_version(schema_version: int | None) -> int:
  return _chat_stream_core.resolve_schema_version(
    schema_version,
    default_schema_version=DEFAULT_SCHEMA_VERSION,
    supported_schema_versions=SUPPORTED_SCHEMA_VERSIONS,
    http_exception_cls=HTTPException,
  )


def _stream_envelope(*, entry: Any, session_id: str, schema_version: int, event: Dict[str, Any]) -> Dict[str, Any]:
  return _chat_stream_core.stream_envelope(
    entry=entry,
    session_id=session_id,
    schema_version=schema_version,
    event=event,
  )


def _disconnect_stream_subscriber_for_backpressure(
  subscriber: StreamSubscriber,
  *,
  done_marker: object = _STREAM_SUBSCRIBER_DONE,
) -> None:
  return _chat_stream_core.disconnect_stream_subscriber_for_backpressure(
    subscriber,
    done_marker=done_marker,
  )


async def _pump_stream_subscriber(
  active_turn: SessionStream,
  subscriber: StreamSubscriber,
  after_seq: int,
  *,
  done_marker: object = _STREAM_SUBSCRIBER_DONE,
  disconnect_stream_subscriber_for_backpressure: Callable[[StreamSubscriber], None] = _disconnect_stream_subscriber_for_backpressure,
) -> None:
  await _chat_stream_core.pump_stream_subscriber(
    active_turn,
    subscriber,
    after_seq,
    done_marker=done_marker,
    disconnect_stream_subscriber_for_backpressure=disconnect_stream_subscriber_for_backpressure,
  )


def _register_stream_subscriber(
  active_turn: SessionStream,
  *,
  after_seq: int,
  client_label: str | None,
  queue_max: int = _STREAM_SUBSCRIBER_QUEUE_MAX,
  pump_stream_subscriber: Callable[[SessionStream, StreamSubscriber, int], asyncio.Task | Any] | None = None,
) -> StreamSubscriber:
  return _chat_stream_core.register_stream_subscriber(
    active_turn,
    after_seq=after_seq,
    client_label=client_label,
    queue_max=queue_max,
    queue_factory=asyncio.Queue,
    task_factory=asyncio.create_task,
    uuid_hex_factory=lambda: uuid.uuid4().hex,
    time_fn=time.time,
    default_pump_stream_subscriber=_pump_stream_subscriber,
    pump_stream_subscriber=pump_stream_subscriber,
  )


async def _cleanup_stream_subscriber(active_turn: SessionStream, subscriber_id: str) -> None:
  await _chat_stream_core.cleanup_stream_subscriber(
    active_turn,
    subscriber_id,
    gather_fn=asyncio.gather,
  )


async def _stream_subscriber_sse(
  *,
  session: GatewaySession,
  active_turn: SessionStream,
  subscriber: StreamSubscriber,
  transcript_dir: Path | None,
  channel: str | None,
  write_transcript: bool,
  log: logging.Logger,
  keepalive_seconds: float = _STREAM_SUBSCRIBER_KEEPALIVE_SECONDS,
  done_marker: object = _STREAM_SUBSCRIBER_DONE,
  cleanup_stream_subscriber: Callable[[SessionStream, str], Any] = _cleanup_stream_subscriber,
) -> AsyncIterator[bytes]:
  inner_stream = _chat_stream_core.stream_subscriber_sse(
    session=session,
    active_turn=active_turn,
    subscriber=subscriber,
    transcript_dir=transcript_dir,
    channel=channel,
    write_transcript=write_transcript,
    log=log,
    keepalive_seconds=keepalive_seconds,
    done_marker=done_marker,
    cleanup_stream_subscriber=cleanup_stream_subscriber,
    wait_for_fn=asyncio.wait_for,
    timeout_error=asyncio.TimeoutError,
    event_for_wire_fn=_event_for_wire,
    write_transcript_fn=_write_transcript,
    adapt_event_fn=adapt_event,
    stream_envelope_fn=_stream_envelope,
    json_dumps_fn=_json_dumps,
    default_schema_version=DEFAULT_SCHEMA_VERSION,
  )
  try:
    async for chunk in inner_stream:
      yield chunk
  finally:
    aclose = getattr(inner_stream, "aclose", None)
    if aclose is not None:
      await aclose()


def _legacy_request_approval(session: GatewaySession, event_log: EventLog) -> RequestApproval:
  async def request_approval(payload: ApprovalRequest) -> Optional[ApprovalDecision]:
    approval_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    session.pending_tools[payload.tool_call_id] = {
      "nonce": payload.nonce,
      "requested_at": int(time.time()),
      "status": "approval_pending",
      "tool_name": payload.tool_name,
      "resolved_qualifier": payload.resolved_qualifier,
    }
    session.approval_queues[payload.tool_call_id] = approval_queue

    event_log.append(
      {
        "type": "tool_approval_request",
        "tool_call_id": payload.tool_call_id,
        "nonce": payload.nonce,
        "tool_name": payload.tool_name,
        "tool_input": _redact_tool_input_for_event(payload.tool_name, payload.tool_input),
        "resolved_qualifier": payload.resolved_qualifier,
        "reason": payload.reason,
        "allow_persistent_approval": payload.allow_persistent_approval,
      }
    )

    try:
      approval = await approval_queue.get()
    finally:
      session.pending_tools.pop(payload.tool_call_id, None)
      session.approval_queues.pop(payload.tool_call_id, None)

    return ApprovalDecision(
      approved=bool(approval.get("approved")),
      allow_tool_type=bool(approval.get("allow_tool_type")),
    )

  return request_approval


def _make_request_approval(
  session: GatewaySession,
  event_log: EventLog,
  *,
  store: Any | None = None,
  policy: Any | None = None,
) -> RequestApproval:
  resolved_store = store or getattr(session, "approval_store", None)
  resolved_policy = policy or getattr(session, "approval_policy", None)
  if resolved_store is None or resolved_policy is None:
    return _legacy_request_approval(session, event_log)
  # Dispatcher-owned lifecycle is used when ToolDispatcher receives the same
  # store/policy bundle. Keep this callback as the legacy pending-tools
  # transport for older callers that have not been constructor-injected yet.
  return _legacy_request_approval(session, event_log)


def _chat_turn_state_from_events(events: list[dict[str, Any]]) -> str:
  for event in events:
    event_type = event.get("type")
    if event_type == "error":
      return "failed"
    if event_type in {
      "operator_pause",
      "budget_exceeded",
      "max_turns_reached",
    }:
      continue
    if event_type == "stream_complete":
      disposition = event.get("terminal_disposition")
      if disposition == "completed":
        return "completed"
      if disposition == "interrupted":
        # Mirrors the autonomous classification in
        # AutonomousRunnerState._terminal_event_outcome: an interrupted terminal
        # that names the budget ceiling is a finished, non-resumable stop, not a
        # resumable interrupt. An `error` terminal earlier in the log still wins
        # above, so a channel failure keeps reporting `failed`.
        if event.get("reason") == "budget_exceeded":
          return "budget_limited"
        return "interrupted"
      return "failed"
  return "completed" if events else "starting"


def _chat_run_state_event(session_id: str, state: str) -> dict[str, Any]:
  return {
    "type": "run_state_changed",
    "run_id": session_id,
    "control_run_id": session_id,
    "state": state,
    "ts": int(time.time()),
  }


def _latest_chat_run_state(session: GatewaySession, session_id: str) -> str | None:
  for event in reversed(session.event_history.snapshot()):
    if event.get("type") != "run_state_changed":
      continue
    event_run_id = event.get("run_id") or event.get("control_run_id")
    if event_run_id is not None and event_run_id != session_id:
      continue
    state = coerce_control_run_state(event.get("state"))
    if state is not None:
      return state
  return None


_SAVED_PREFERENCE_NOT_APPLIED_DETAILS: dict[str, str] = {
  "model_unknown": "its model key is no longer in the catalog",
  "model_deprecated": "its model is deprecated",
  "model_disabled": "its model is disabled",
  "model_revoked": "its model is revoked",
  "model_hidden": "its model is no longer offered for selection",
  "model_not_allowed": "its model is not allowed for this capability",
  "effort_unsupported": "its stored effort is not supported by its model",
  "credential_ineligible": (
    "no eligible credential or entitlement covers its model"
  ),
}


def build_capability_choices(
  *,
  session: GatewaySession,
  build_chat_runtime: BuildChatRuntime,
) -> dict[str, CapabilityChoiceResponse]:
  """Return only authenticated, session-executable stable-key choices."""

  resolver = _capability_execution_resolver_for_session(
    session=session,
    build_chat_runtime=build_chat_runtime,
  )
  registry = resolver.registry
  policy_artifact = resolver.selection_policy
  responses: dict[str, CapabilityChoiceResponse] = {}
  preference_store = getattr(
    build_chat_runtime,
    "_gateway_model_preference_store",
    None,
  )
  for capability_id, policy in sorted(policy_artifact.capabilities.items()):
    if capability_id != "session.driver" and not policy.allow_authenticated_run_override:
      continue
    eligible = eligible_model_choices(
      capability_id,
      registry=registry,
      selection_policy=policy_artifact,
      auth=resolver.auth_context,
    )
    choices = [
      CapabilityChoice(
        model_key=choice.key,
        label=choice.label,
        supported_efforts=list(choice.supported_efforts),
        default_effort=choice.default_effort,
        lifecycle=choice.lifecycle,
      )
      for choice in eligible
    ]
    notices: list[CapabilityChoiceNotice] = []
    selected: CapabilityChoiceSelection | None = None
    if policy.default.kind == "inherit_parent":
      notices.append(CapabilityChoiceNotice(
        code="inherits_parent",
        message=f"{capability_id} inherits the exact parent binding by default.",
      ))
    else:
      saved_preference = None
      if capability_id == "session.driver" and preference_store is not None:
        saved_preference = preference_store.get(
          tenant_id=resolver.auth_context.tenant_id,
          actor_id=resolver.auth_context.actor_id,
          capability_id=capability_id,
        )
      try:
        bind = resolve_capability_model(
          capability_id,
          registry=registry,
          selection_policy=policy_artifact,
          auth=resolver.auth_context,
          saved_preference=saved_preference,
          trusted_channel=resolver.trusted_channel,
        )
      except CapabilityResolutionError as exc:
        notices.append(CapabilityChoiceNotice(
          code=exc.code,
          message=str(exc),
          model_key=exc.model_key,
        ))
      else:
        entry = registry.require(bind.model_key)
        selected = CapabilityChoiceSelection(
          model_key=bind.model_key,
          label=entry.label,
          effort=bind.effort,
          reason=bind.selection_source,
        )
        if (
          saved_preference is not None
          and bind.selection_source != "saved_preference"
        ):
          reason = saved_preference_ineligibility(
            saved_preference,
            capability_id=capability_id,
            registry=registry,
            policy=policy,
            auth=resolver.auth_context,
            trusted_channel=resolver.trusted_channel,
          )
          detail = _SAVED_PREFERENCE_NOT_APPLIED_DETAILS.get(
            reason or "",
            "it is currently ineligible",
          )
          notices.append(CapabilityChoiceNotice(
            code="saved_preference_not_applied",
            message=(
              f"The saved preference was not applied because {detail}; "
              "it remains saved until replaced or cleared."
            ),
            model_key=saved_preference.model_key,
            reason=reason,
          ))
    responses[capability_id] = CapabilityChoiceResponse(
      capability=capability_id,
      catalog_revision=registry.revision,
      policy_revision=policy_artifact.revision,
      selected=selected,
      notices=notices,
      choices=choices,
    )
  return responses


def _session_driver_auth_context(
  *,
  session: GatewaySession,
  tenant_id: str,
  service_provider_handles: dict[str, CredentialHandle],
) -> AuthContext:
  session_handle = session.session_credential_handle
  user_handles: dict[str, CredentialHandle] = {}
  service_handles = dict(service_provider_handles)
  if session_handle is not None:
    if session_handle.principal == "user":
      user_handles[session_handle.provider] = session_handle
    else:
      service_handles[session_handle.provider] = session_handle

  actor_id = str(session.owner_user_id or session.user_id or "").strip()
  return AuthContext(
    run_mode="interactive",
    actor_id=actor_id,
    tenant_id=tenant_id,
    user_provider_handles=user_handles,
    service_provider_handles=service_handles,
    entitled_capabilities=session.model_entitled_capabilities,
    entitled_model_keys=session.model_entitled_keys,
    allow_service_for_interactive=session.allow_service_for_interactive,
  )


def _materialize_credential(
  *,
  session: GatewaySession,
  handle: CredentialHandle,
  service_auth_config_resolver: Callable[
    [CredentialHandle],
    MaterializedCredential,
  ] | None,
) -> MaterializedCredential:
  if handle is session.session_credential_handle:
    if session.auth_config is None:
      raise RuntimeError(
        "selected session credential handle has no session credential material"
      )
    return MaterializedCredential(
      handle=handle,
      auth_config=session.auth_config,
    )

  if handle.principal != "service":
    raise RuntimeError("non-session user credential handles are not supported")
  if service_auth_config_resolver is None:
    raise RuntimeError(
      f"service credential materializer is missing for provider {handle.provider}"
    )
  materialized = service_auth_config_resolver(handle)
  if not isinstance(materialized, MaterializedCredential):
    raise RuntimeError(
      "service credential materializer must return MaterializedCredential"
    )
  if materialized.handle is not handle:
    raise RuntimeError(
      "service credential materializer returned a different credential handle"
    )
  return materialized


def _capability_execution_resolver_for_session(
  *,
  session: GatewaySession,
  build_chat_runtime: BuildChatRuntime,
  run_overrides: dict[CapabilityId, ModelSelectionIntent] | None = None,
) -> CapabilityExecutionResolver:
  registry: ProductModelRegistry | None = getattr(
    build_chat_runtime,
    "_gateway_model_registry",
    None,
  )
  selection_policy: ProductModelSelectionPolicy | None = getattr(
    build_chat_runtime,
    "_gateway_model_selection_policy",
    None,
  )
  if registry is None or selection_policy is None:
    raise CapabilityResolutionError(
      "capability_policy_missing",
      "deployment capability policy is not configured",
      capability_id="session.driver",
    )

  configured_tenant = str(
    getattr(build_chat_runtime, "_gateway_tenant_id", "") or ""
  ).strip()
  session_tenant = str(session.tenant_id or "").strip()
  tenant_id = session_tenant or configured_tenant
  if not tenant_id:
    raise RuntimeError("tenant_id is required for capability-bound chat")
  if (
    configured_tenant
    and session_tenant
    and configured_tenant != session_tenant
  ):
    raise RuntimeError("session tenant does not match gateway tenant")

  service_handles = dict(
    getattr(
      build_chat_runtime,
      "_gateway_service_provider_handles",
      {},
    )
    or {}
  )
  auth = _session_driver_auth_context(
    session=session,
    tenant_id=tenant_id,
    service_provider_handles=service_handles,
  )
  adapter_resolver = getattr(
    build_chat_runtime,
    "_gateway_capability_adapter_resolver",
    None,
  )
  if adapter_resolver is None:
    raise RuntimeError("capability adapter resolver is not configured")
  return CapabilityExecutionResolver(
    registry=registry,
    selection_policy=selection_policy,
    auth_context=auth,
    credential_materializer=lambda handle: _materialize_credential(
      session=session,
      handle=handle,
      service_auth_config_resolver=getattr(
        build_chat_runtime,
        "_gateway_service_auth_config_resolver",
        None,
      ),
    ),
    adapter_resolver=adapter_resolver,
    trusted_channel=session.channel,
    authenticated_run_overrides=(
      dict(session.capability_run_overrides)
      if run_overrides is None
      else run_overrides
    ),
    executable_capability_ids=GATEWAY_EXECUTED_CAPABILITY_IDS,
  )


def bind_init_capability_selections(
  *,
  session: GatewaySession,
  selections: dict[str, CapabilitySelection],
  build_chat_runtime: BuildChatRuntime,
) -> None:
  """Validate, materialize, and freeze init-time role selections."""

  selection_policy: ProductModelSelectionPolicy | None = getattr(
    build_chat_runtime,
    "_gateway_model_selection_policy",
    None,
  )
  if selection_policy is None:
    raise CapabilityResolutionError(
      "capability_policy_missing",
      "deployment capability policy is not configured",
      capability_id="session.driver",
    )

  normalized: dict[CapabilityId, CapabilitySelection] = {}
  for raw_capability_id, selection in selections.items():
    capability_id = str(raw_capability_id or "").strip()
    if capability_id not in CAPABILITY_IDS:
      raise CapabilityResolutionError(
        "unknown_capability",
        f"unknown capability_id: {capability_id!r}",
        capability_id=capability_id,
      )
    policy = selection_policy.capabilities.get(capability_id)
    if (
      policy is None
      or capability_id in {"session.driver", "node.fork"}
      or not policy.allow_authenticated_run_override
    ):
      raise CapabilityResolutionError(
        "capability_model_not_allowed",
        f"run selection is not allowed for {capability_id}",
        capability_id=capability_id,
      )
    normalized[capability_id] = selection  # type: ignore[index]

  run_overrides: dict[CapabilityId, ModelSelectionIntent] = {}
  for capability_id, selection in normalized.items():
    run_overrides[capability_id] = ModelSelectionIntent(
      model_key=selection.model_key,
      effort=selection.effort,
      source="explicit_user",
    )

  resolver = _capability_execution_resolver_for_session(
    session=session,
    build_chat_runtime=build_chat_runtime,
    run_overrides=run_overrides,
  )
  for capability_id in sorted(normalized):
    resolver.resolve(capability_id)
  bind_session_capability_selections(
    session,
    run_overrides=run_overrides,
  )


def prepare_session_driver_turn(
  session: GatewaySession,
  inputs: ChatTurnInputs,
  *,
  build_chat_runtime: BuildChatRuntime,
) -> PreparedChatTurn:
  """Build and bind one trusted chat request before any billable side effect."""

  request_id = (
    inputs.request_id.strip()
    if isinstance(inputs.request_id, str)
    else inputs.request_id
  )
  request_id = request_id or str(uuid.uuid4())
  context = dict(inputs.context or {})
  context["profile"] = _resolve_chat_profile_name(context)
  request = ChatRequest(
    messages=list(inputs.messages),
    user_id=session.user_id,
    request_id=request_id,
    context=context,
    metadata=dict(inputs.metadata or {}),
    model_key=inputs.model_key,
    effort=inputs.effort,
    catalog_revision=inputs.catalog_revision,
    ui_blocks_contract=inputs.ui_blocks_contract,
    attachments=tuple(inputs.attachments),
    investment_artifact_selection=inputs.investment_artifact_selection,
  )

  raw_channel = context.get("channel")
  claimed_channel = (
    raw_channel.strip().lower()
    if isinstance(raw_channel, str)
    else None
  )
  channel = session.channel or claimed_channel
  channel_profile_allowlist = getattr(
    build_chat_runtime,
    "_gateway_channel_profile_allowlist",
    None,
  )
  if channel_profile_allowlist is not None and session.channel is not None:
    allowed_profiles = channel_profile_allowlist.get(session.channel)
    if (
      allowed_profiles is not None
      and context["profile"] not in set(allowed_profiles)
    ):
      raise HTTPException(
        status_code=403,
        detail=(
          f"Profile '{context['profile']}' is not permitted on "
          f"channel '{session.channel}'"
        ),
      )

  capability_execution_resolver = _capability_execution_resolver_for_session(
    session=session,
    build_chat_runtime=build_chat_runtime,
  )
  execution_policy_resolver = getattr(
    build_chat_runtime,
    "_gateway_session_execution_policy_resolver",
    None,
  )
  try:
    execution_policy = (
      execution_policy_resolver(context)
      if callable(execution_policy_resolver)
      else None
    )
  except Exception as exc:
    raise HTTPException(
      status_code=400,
      detail=f"Invalid session execution policy: {exc}",
    ) from exc
  if execution_policy is not None and not isinstance(
    execution_policy, SessionExecutionPolicy
  ):
    raise RuntimeError(
      "session_execution_policy_resolver must return "
      "SessionExecutionPolicy or None"
    )
  explicit_intent = (
    ModelSelectionIntent(
      model_key=request.model_key,
      effort=request.effort,
      source="explicit_user",
      catalog_revision=request.catalog_revision,
    )
    if request.model_key is not None
    else None
  )
  preference_store = getattr(
    build_chat_runtime,
    "_gateway_model_preference_store",
    None,
  )
  saved_preference = (
    preference_store.get(
      tenant_id=capability_execution_resolver.auth_context.tenant_id,
      actor_id=capability_execution_resolver.auth_context.actor_id,
      capability_id="session.driver",
    )
    if preference_store is not None and explicit_intent is None
    else None
  )
  capability_execution = capability_execution_resolver.resolve(
    "session.driver",
    explicit_intent=explicit_intent,
    saved_preference=saved_preference,
  )
  request._bind_session_execution_policy(execution_policy)
  request._bind_session_driver(
    capability_execution=capability_execution,
    capability_execution_resolver=capability_execution_resolver,
  )
  return PreparedChatTurn(
    session_id=session.session_id,
    request=request,
    channel=channel,
  )


async def _dispatch_chat_turn(
  session: GatewaySession,
  inputs: ChatTurnInputs,
  *,
  event_log: EventLog,
  on_event: Callable[[StreamEvent], Awaitable[None]],
  build_chat_runtime: BuildChatRuntime,
  transcript_dir: Path | None,
  publish_lifecycle_events: bool = False,
  prepared_turn: PreparedChatTurn | None = None,
  required_event_delivery: bool = False,
) -> ChatTurnResult:
  """Run one chat turn outside the ASGI response lifecycle."""
  try:
    return await _dispatch_chat_turn_body(
      session,
      inputs,
      event_log=event_log,
      on_event=on_event,
      build_chat_runtime=build_chat_runtime,
      transcript_dir=transcript_dir,
      publish_lifecycle_events=publish_lifecycle_events,
      prepared_turn=prepared_turn,
      required_event_delivery=required_event_delivery,
    )
  finally:
    if (
      inputs.commercial_dispatch_owner is not None
      and session._commercial_dispatch_owner is inputs.commercial_dispatch_owner
    ):
      session._commercial_dispatch_owner = None
    if getattr(event_log, "defer_terminal_close", False):
      event_log.close()


async def _dispatch_chat_turn_body(
  session: GatewaySession,
  inputs: ChatTurnInputs,
  *,
  event_log: EventLog,
  on_event: Callable[[StreamEvent], Awaitable[None]],
  build_chat_runtime: BuildChatRuntime,
  transcript_dir: Path | None,
  publish_lifecycle_events: bool = False,
  prepared_turn: PreparedChatTurn | None = None,
  required_event_delivery: bool = False,
) -> ChatTurnResult:
  if (
    inputs.commercial_dispatch_owner is not None
    and session._commercial_dispatch_owner is not inputs.commercial_dispatch_owner
  ):
    raise HTTPException(status_code=409, detail="Commercial dispatch ownership was lost")
  if session.kind != "chat":
    raise HTTPException(status_code=400, detail="control sessions cannot dispatch chat turns")
  if _active_turn_is_running(session.active_turn):
    raise HTTPException(status_code=409, detail="A turn is already running; subscribe via /chat/subscribe")
  if session.active_turn is not None:
    _clear_active_turn(session, session.active_turn)
  if session.stream_active:
    raise HTTPException(status_code=409, detail="A turn is already running; subscribe via /chat/subscribe")

  prepared = prepared_turn or prepare_session_driver_turn(
    session,
    inputs,
    build_chat_runtime=build_chat_runtime,
  )
  if prepared.session_id != session.session_id:
    raise RuntimeError("prepared chat turn belongs to a different session")
  request = prepared.request
  request_id = request.request_id or str(uuid.uuid4())
  context = dict(request.context or {})
  request_metadata = dict(request.metadata or {})
  request._bind_commercial_work_start(inputs.commercial_work_start)
  ui_blocks_run = UiBlocksRunContext(
    capability=request.ui_blocks_contract,
    turn_key=uuid.uuid4().hex,
    session_id=session.session_id,
    registry=UiBlocksRunRegistry(),
  )
  request._bind_ui_blocks_run(ui_blocks_run)

  channel = prepared.channel
  log = getattr(build_chat_runtime, "_gateway_log", logging.getLogger("agent_gateway.server"))
  auth_manager = getattr(build_chat_runtime, "_gateway_auth_manager", None)

  if inputs.commercial_dispatch_owner is not None:
    if session._commercial_dispatch_owner is not inputs.commercial_dispatch_owner:
      raise HTTPException(
        status_code=409, detail="Commercial dispatch ownership was lost"
      )
    session._commercial_dispatch_owner = None
  session.stream_active = True
  sid = session.session_id
  active_turn = SessionStream(
    event_log=event_log,
    runner_task=asyncio.current_task(),
  )
  session.active_turn = active_turn

  previous_on_event = getattr(event_log, "_on_event", None)
  previous_session_id = getattr(event_log, "_session_id", "")
  fanout_stop = object()
  fanout_queue: asyncio.Queue[Any] = asyncio.Queue()
  fanout_failure: asyncio.Future[BaseException] = (
    asyncio.get_running_loop().create_future()
  )

  async def _fanout_worker() -> None:
    while True:
      event = await fanout_queue.get()
      if event is fanout_stop:
        return
      try:
        await on_event(dict(event))
      except asyncio.CancelledError as exc:
        if required_event_delivery:
          if not fanout_failure.done():
            fanout_failure.set_result(exc)
          return
        raise
      except Exception as exc:
        if required_event_delivery:
          if not fanout_failure.done():
            fanout_failure.set_result(exc)
          return
        log.warning("chat turn event fan-out failed for %s: %s", sid, exc)

  def _record_event(event: Dict[str, Any], event_session_id: str) -> None:
    event_copy = dict(event)
    session.event_history.append(event_copy)
    fanout_queue.put_nowait(event_copy)
    if previous_on_event is not None:
      try:
        previous_on_event(event, event_session_id)
      except Exception:
        pass

  async def _stop_fanout_worker() -> None:
    fanout_queue.put_nowait(fanout_stop)
    await asyncio.gather(fanout_task, return_exceptions=True)

  setattr(event_log, "_on_event", _record_event)
  setattr(event_log, "_session_id", sid)
  fanout_task = asyncio.create_task(_fanout_worker())
  if publish_lifecycle_events:
    _record_event(_chat_run_state_event(sid, "running"), sid)

  runtime: ChatRuntime | None = None
  runner: Any | None = None
  runner_task: asyncio.Task[Any] | None = None

  async def _safe_fire_disconnect() -> None:
    if runtime is None:
      return
    try:
      await runtime.on_disconnect()
    except Exception as exc:
      log.warning("on_disconnect failed for %s: %s", sid, exc)

  started_at = float(session.created_at)

  def _emit_terminal(event: Dict[str, Any]) -> None:
    emit_recap_then_terminal(
      event_log,
      event,
      session_id=sid,
      started_at=started_at,
    )

  try:
    messages = [_model_to_dict(message) for message in inputs.messages]
    last_user = next((message["content"] for message in reversed(messages) if message.get("role") == "user"), "")
    if not session.initial_message and last_user:
      session.initial_message = str(last_user)
    log.info("Chat request | session=%s | msgs=%d | user=%s", sid, len(messages), last_user[:200])
    event = {"type": "chat_request", "messages": messages, "context": context}
    if request_metadata:
      event["metadata"] = request_metadata
    pid = gateway_product_id()
    if pid is not None:
      event["product_id"] = pid
    _write_transcript(
      transcript_dir=transcript_dir,
      session_id=sid,
      entry=event,
      user_id=session.user_id,
      channel=channel,
    )

    ui_blocks_token = activate_ui_blocks_run(ui_blocks_run)
    try:
      capability_execution = request.capability_execution
      if capability_execution is None:
        raise RuntimeError(
          "prepared chat turn has no session.driver capability execution"
        )
      capability_bind = capability_execution.bind
      _record_event(
        {
          "type": "capability_bound",
          **capability_bind.receipt(),
        },
        sid,
      )
      runtime = await _call_build_chat_runtime(
        build_chat_runtime,
        session=session,
        request=request,
        channel=channel,
        auth_manager=auth_manager,
      )
      if runtime.capability_execution is not capability_execution:
        raise RuntimeError(
          "chat runtime did not preserve the exact session.driver execution"
        )
      selected_content_admitter = getattr(
        build_chat_runtime,
        "_gateway_selected_content_admitter",
        None,
      )
      selected_content_admission = SelectedContentAdmission()
      if selected_content_admitter is not None:
        selected_content_admission = selected_content_admitter(session, request)
        if inspect.isawaitable(selected_content_admission):
          selected_content_admission = await selected_content_admission
        if not isinstance(selected_content_admission, SelectedContentAdmission):
          raise TypeError(
            "selected_content_admitter must return SelectedContentAdmission"
          )
        if selected_content_admission.model_context:
          if isinstance(runtime.system_prompt, str):
            runtime.system_prompt = (
              f"{runtime.system_prompt}\n\n{selected_content_admission.model_context}"
            )
          else:
            runtime.system_prompt = [
              *runtime.system_prompt,
              (selected_content_admission.model_context, False),
            ]
      runner = _build_runner_with_started_at(runtime.build_runner, event_log, sid, started_at)
      bind_selected_content = getattr(runner, "bind_selected_content", None)
      if selected_content_admission.bindings and not callable(bind_selected_content):
        raise RuntimeError(
          "chat runner cannot commit selected content"
        )
      if callable(bind_selected_content):
        bind_selected_content(selected_content_admission.bindings)
      set_purpose = getattr(runner, "set_purpose", None)
      if callable(set_purpose):
        set_purpose(runtime.purpose)
      if getattr(runner, "capability_execution", None) is not capability_execution:
        raise RuntimeError(
          "chat runner did not preserve the exact session.driver execution"
        )
      clear_credential_refresher = getattr(
        runner,
        "set_credential_refresher",
        None,
      )
      if callable(clear_credential_refresher):
        clear_credential_refresher(None)
    finally:
      reset_ui_blocks_run(ui_blocks_token)
    setattr(event_log, "_gateway_execution_location", runtime.execution_location)
    if runtime.disconnect_handler is None:
      runner_on_disconnect = getattr(runner, "on_disconnect", None)
      if callable(runner_on_disconnect):
        runtime.disconnect_handler = runner_on_disconnect

    async def run_agent() -> None:
      try:
        await runner.run(
          messages=messages,
          system_prompt=runtime.system_prompt,
          max_turns=runtime.max_turns,
        )
        if not log_has_terminal(event_log):
          log.warning(
            "chat runner completed without a terminal event for %s; emitting stream-closed error",
            sid,
          )
      except asyncio.CancelledError:
        raise
      except Exception as exc:
        error_text = str(exc)
        log.error(
          "chat runner failed for %s: %s",
          sid,
          error_text,
          exc_info=None if exception_traceback_already_logged(exc) else exc,
        )
        _emit_terminal({"type": "error", "error": error_text})
      finally:
        if not log_has_terminal(event_log):
          _emit_terminal({"type": "error", "error": "stream closed"})

    runner_task = asyncio.create_task(run_agent())
    # shield: a client-disconnect cancel of the enclosing dispatch_task must NOT
    # auto-propagate into runner_task here. The `except asyncio.CancelledError` block
    # below fires the cooperative disconnect (sets the tool abort_event and yields)
    # BEFORE cancelling runner_task, so an in-flight tool call gets the cooperative
    # abort handshake. Plain `await runner_task` cancels runner_task first, pre-empting
    # that handshake — the PR4 (47b91a31) regression this restores.
    if required_event_delivery:
      done, _pending = await asyncio.wait(
        (runner_task, fanout_failure),
        return_when=asyncio.FIRST_COMPLETED,
      )
      if fanout_failure in done:
        delivery_error = fanout_failure.result()
        await _safe_fire_disconnect()
        runner_task.cancel()
        await asyncio.gather(runner_task, return_exceptions=True)
        raise RuntimeError(
          "required chat turn event delivery failed"
        ) from delivery_error
      await runner_task
    else:
      await asyncio.shield(runner_task)
    session.stream_active = False
    if publish_lifecycle_events:
      events = [dict(entry.event) for entry in event_log.entries]
      if _latest_chat_run_state(session, sid) != "cancelled":
        _record_event(_chat_run_state_event(sid, _chat_turn_state_from_events(events)), sid)
  except asyncio.CancelledError:
    await _safe_fire_disconnect()
    if runner_task is not None:
      runner_task.cancel()
    await asyncio.gather(
      *(task for task in (runner_task,) if task is not None),
      return_exceptions=True,
    )
    if publish_lifecycle_events and _latest_chat_run_state(session, sid) != "cancelled":
      _record_event(_chat_run_state_event(sid, "cancelled"), sid)
    _emit_terminal({"type": "error", "error": "stream closed"})
    raise
  except Exception as exc:
    session.stream_active = False
    if publish_lifecycle_events:
      _emit_terminal({"type": "error", "error": str(exc)})
      _record_event(_chat_run_state_event(sid, "failed"), sid)
    _emit_terminal({"type": "error", "error": "stream failed"})
    raise
  finally:
    if not log_has_terminal(event_log):
      _emit_terminal({"type": "error", "error": "stream closed"})
    event_log.close()
    await _stop_fanout_worker()
    setattr(event_log, "_on_event", previous_on_event)
    setattr(event_log, "_session_id", previous_session_id)
    session.pending_tools.clear()
    session.approval_queues.clear()
    _drain_result_queue(session.result_queue)
    session.stream_active = False
    if session.active_turn is active_turn:
      _schedule_active_turn_clear(session, active_turn)

  if required_event_delivery and fanout_failure.done():
    raise RuntimeError(
      "required chat turn event delivery failed"
    ) from fanout_failure.result()

  events = [dict(entry.event) for entry in event_log.entries]
  return ChatTurnResult(
    session_id=sid,
    request_id=request_id,
    state=_chat_turn_state_from_events(events),
    events=events,
  )


def _init_approval_subsystem(app: FastAPI, config: GatewayServerConfig) -> None:
  audit_writer = resolve_audit_writer()
  audit_emitter = ApprovalAuditEmitter(
    writer=audit_writer,
    deployment_secret=config.audit_hmac_secret_resolver(),
    key_id=config.audit_hmac_key_id_resolver(),
    tool_input_redactor=config.tool_input_redactor,
  )
  store = SQLiteApprovalStore(
    path=resolve_approval_db_path(),
    audit_emitter=audit_emitter,
    notification_destination_resolver=build_env_approval_notification_destination_resolver(),
    notification_sender=build_env_telegram_approval_notification_sender(),
  )
  policy = resolve_policy(store=store)
  app.state.gateway_approval_audit_writer = audit_writer
  app.state.gateway_approval_audit_emitter = audit_emitter
  app.state.gateway_approval_store = store
  app.state.gateway_approval_policy = policy
