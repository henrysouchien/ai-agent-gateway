from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional

from fastapi import FastAPI, HTTPException

from ._provider_utils import _get_allowed_models_for_provider_name
from .agent_session_log import _atomic_write_sidecar
from .approval_audit import ApprovalAuditEmitter
from .approval_notifications import (
  build_env_approval_notification_destination_resolver,
  build_env_telegram_approval_notification_sender,
)
from .approval_resolver import resolve_policy
from .approval_store import SQLiteApprovalStore
from .audit_resolver import resolve_audit_writer
from .auth import CredentialRefreshRequest, CredentialsResolver, ProviderCredentialFailure
from .event_adapter import adapt_event
from .event_log import EventLog, log_has_terminal
from .events import DEFAULT_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS
from .product_config import gateway_product_id
from .providers import StreamEvent
from .session import GatewaySession, SessionStore, SessionStream, StreamSubscriber
from .session_recap import emit_recap_then_terminal
from . import server_chat_stream_core as _chat_stream_core
from . import server_chat_transcripts as _chat_transcripts
from .tool_dispatcher import ApprovalDecision, ApprovalRequest
from .tool_redaction import get_audit_hmac_secret, redact_tool_input as default_redact_tool_input
from .server_artifact_helpers import _json_dumps, _model_to_dict
from .server_models import (
  BuildChatRuntime,
  ChatRequest,
  ChatRuntime,
  ChatTurnInputs,
  ChatTurnResult,
  GatewayServerConfig,
  RequestApproval,
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
    return dict(tool_input)


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
      denied_by=None if approval.get("approved") else approval.get("denied_by"),
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
  for event in reversed(events):
    event_type = event.get("type")
    if event_type == "error":
      return "failed"
    if event_type == "stream_complete":
      return "completed"
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
    state = event.get("state")
    if state in {"starting", "running", "approval_pending", "completed", "failed", "cancelled"}:
      return str(state)
  return None


async def _dispatch_chat_turn(
  session: GatewaySession,
  inputs: ChatTurnInputs,
  *,
  event_log: EventLog,
  on_event: Callable[[StreamEvent], Awaitable[None]],
  build_chat_runtime: BuildChatRuntime,
  credentials_resolver: CredentialsResolver | None,
  transcript_dir: Path | None,
  publish_lifecycle_events: bool = False,
) -> ChatTurnResult:
  """Run one chat turn outside the ASGI response lifecycle."""
  try:
    return await _dispatch_chat_turn_body(
      session,
      inputs,
      event_log=event_log,
      on_event=on_event,
      build_chat_runtime=build_chat_runtime,
      credentials_resolver=credentials_resolver,
      transcript_dir=transcript_dir,
      publish_lifecycle_events=publish_lifecycle_events,
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
  credentials_resolver: CredentialsResolver | None,
  transcript_dir: Path | None,
  publish_lifecycle_events: bool = False,
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

  request_id = inputs.request_id.strip() if isinstance(inputs.request_id, str) else inputs.request_id
  request_id = request_id or str(uuid.uuid4())
  context = dict(inputs.context or {})
  context["profile"] = _resolve_chat_profile_name(context)
  request_metadata = dict(inputs.metadata or {})
  request = ChatRequest(
    messages=list(inputs.messages),
    user_id=session.user_id,
    request_id=request_id,
    context=context,
    metadata=request_metadata,
    model=inputs.model,
  )
  request._bind_commercial_work_start(inputs.commercial_work_start)

  raw_channel = context.get("channel")
  claimed_channel = raw_channel.strip().lower() if isinstance(raw_channel, str) else None
  channel = session.channel or claimed_channel
  log = getattr(build_chat_runtime, "_gateway_log", logging.getLogger("agent_gateway.server"))
  resolver_timeout_seconds = float(getattr(build_chat_runtime, "_gateway_resolver_timeout_seconds", 5.0))
  config_auth_config = getattr(build_chat_runtime, "_gateway_auth_config", {})
  allowed_models = getattr(build_chat_runtime, "_gateway_allowed_models", None)
  channel_profile_allowlist = getattr(build_chat_runtime, "_gateway_channel_profile_allowlist", None)
  auth_manager = getattr(build_chat_runtime, "_gateway_auth_manager", None)
  if channel_profile_allowlist is not None and session.channel is not None:
    allowed_profiles = channel_profile_allowlist.get(session.channel)
    if allowed_profiles is not None and context["profile"] not in set(allowed_profiles):
      raise HTTPException(
        status_code=403,
        detail=f"Profile '{context['profile']}' is not permitted on channel '{session.channel}'",
      )

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

  async def _fanout_worker() -> None:
    while True:
      event = await fanout_queue.get()
      if event is fanout_stop:
        return
      try:
        await on_event(dict(event))
      except Exception as exc:
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

  async def _credential_refresher(failure: ProviderCredentialFailure) -> Dict[str, Any] | None:
    resolver = credentials_resolver
    if resolver is None:
      return None
    session_auth_config = session.auth_config or config_auth_config
    request_billing_mode = (
      session_auth_config.get("billing_mode")
      or getattr(runner, "_billing_mode", None)
      or config_auth_config.get("billing_mode")
    )
    request_rate_table_version = (
      session_auth_config.get("rate_table_version")
      or getattr(runner, "_rate_table_version", None)
      or config_auth_config.get("rate_table_version")
    )
    refresh_request = CredentialRefreshRequest(
      user_id=session.user_id,
      user_email=session.user_email,
      session_id=session.session_id,
      api_key_hash=session.api_key_hash,
      channel=channel,
      provider=failure.provider,
      billing_mode=str(request_billing_mode or "") or None,
      rate_table_version=str(request_rate_table_version or "") or None,
      model=str(session_auth_config.get("model", "") or "") or None,
      auth_mode=str(session_auth_config.get("auth_mode", "") or "") or None,
      request_id=request_id,
      failure=failure,
    )
    try:
      auth_config = await asyncio.wait_for(
        resolver(refresh_request),  # type: ignore[arg-type]
        timeout=resolver_timeout_seconds,
      )
    except Exception as exc:
      log.warning(
        "Credential refresh failed | session=%s user=%s provider=%s kind=%s detail=%s",
        session.session_id,
        session.user_id,
        failure.provider,
        failure.kind,
        exc,
      )
      return None
    if auth_config is None:
      log.info(
        "Credential refresh unavailable | session=%s user=%s provider=%s kind=%s",
        session.session_id,
        session.user_id,
        failure.provider,
        failure.kind,
      )
      return None
    if auth_config.provider != failure.provider:
      log.warning(
        "Credential refresh returned provider=%s for provider=%s; ignoring",
        auth_config.provider,
        failure.provider,
      )
      return None
    refreshed_auth_config = auth_config.to_dict()
    if request_billing_mode:
      refreshed_auth_config["billing_mode"] = request_billing_mode
    if request_rate_table_version:
      refreshed_auth_config["rate_table_version"] = request_rate_table_version
    session.auth_config = refreshed_auth_config
    return dict(refreshed_auth_config)

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

    runtime = await _call_build_chat_runtime(
      build_chat_runtime,
      session=session,
      request=request,
      channel=channel,
      auth_manager=auth_manager,
    )
    session_auth_config = session.auth_config or config_auth_config
    resolved_model = runtime.model_override or request.model or str(session_auth_config.get("model", "")).strip() or None
    if resolved_model:
      resolved_provider_name = str(getattr(runtime, "resolved_provider_name", "") or "").strip().lower()
      if resolved_provider_name:
        provider_allowed_models = _get_allowed_models_for_provider_name(resolved_provider_name)
        if provider_allowed_models and resolved_model not in provider_allowed_models:
          raise HTTPException(status_code=400, detail=f"Invalid model: {resolved_model}")
      elif allowed_models and resolved_model not in allowed_models:
        raise HTTPException(status_code=400, detail=f"Invalid model: {resolved_model}")
    runner = _build_runner_with_started_at(runtime.build_runner, event_log, sid, started_at)
    setattr(event_log, "_gateway_execution_location", runtime.execution_location)
    set_credential_refresher = getattr(runner, "set_credential_refresher", None)
    if callable(set_credential_refresher) and credentials_resolver is not None:
      set_credential_refresher(_credential_refresher)
    if runtime.disconnect_handler is None:
      runner_on_disconnect = getattr(runner, "on_disconnect", None)
      if callable(runner_on_disconnect):
        runtime.disconnect_handler = runner_on_disconnect

    async def run_agent() -> None:
      try:
        await runner.run(
          messages=messages,
          system_prompt=runtime.system_prompt,
          model_override=runtime.model_override,
          max_turns=runtime.max_turns,
        )
      except asyncio.CancelledError:
        raise
      except Exception as exc:
        _emit_terminal({"type": "error", "error": str(exc)})
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
    audit_emitter=audit_emitter,
    notification_destination_resolver=build_env_approval_notification_destination_resolver(),
    notification_sender=build_env_telegram_approval_notification_sender(),
  )
  policy = resolve_policy(store=store)
  app.state.gateway_approval_audit_writer = audit_writer
  app.state.gateway_approval_audit_emitter = audit_emitter
  app.state.gateway_approval_store = store
  app.state.gateway_approval_policy = policy
