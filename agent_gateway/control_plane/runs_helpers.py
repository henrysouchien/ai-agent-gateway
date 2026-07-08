from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

from agent_gateway.approvals import ApprovalActionError, _record_vote_and_unblock
from agent_gateway.autonomous_runner import AutonomousRegistry, AutonomousTask
from agent_gateway.fixture_gate import (
  FIXTURE_APPROVAL_HTML_ARTIFACT_SKILL_NAME,
  FIXTURE_TERMINAL_FAILURE_SKILL_NAME,
  fixture_provider_available,
  is_fixture_profile_name,
  is_fixture_skill_name,
)
from agent_gateway.session import AuthManager, GatewaySession
from agent_gateway.control_run_lifecycle import (
  CONTROL_ACTIVE_RUN_STATES,
  CONTROL_RUN_STATES,
  CONTROL_TERMINAL_RUN_STATES,
  canonical_control_run_state,
  coerce_control_run_state,
  is_autonomous_run_internal_messageable_state,
  is_autonomous_run_internal_resumable_state,
  is_control_run_active_state,
)
from .runs_models import (
  AutonomousDispatchRequest,
  AutonomousDispatchResponse,
  AutonomousResumeRequest,
  AutonomousRunMessageRequest,
  AutonomousRunResponse,
  AutonomousRunState,
  ChatContinuationRequest,
  ChatDispatchRequest,
  ChatDispatchResponse,
  ChatMessage,
  ChatRunResponse,
  ChatRunState,
  ControlRunDispatchRequest,
  DispatchScope,
  PendingApprovalResponse,
  RunDispatchResponse,
  RunEnvelopeResponse,
  RunLogsResponse,
  RunMessageRequest,
  RunResponse,
  RunsListResponse,
  VerdictSummaryResponse,
)

__all__ = [
  "AutonomousDispatchRequest",
  "AutonomousDispatchResponse",
  "AutonomousResumeRequest",
  "AutonomousRunMessageRequest",
  "AutonomousRunResponse",
  "AutonomousRunState",
  "ChatContinuationRequest",
  "ChatDispatchRequest",
  "ChatDispatchResponse",
  "ChatMessage",
  "ChatRunResponse",
  "ChatRunState",
  "ControlRunDispatchRequest",
  "DispatchScope",
  "PendingApprovalResponse",
  "RunDispatchResponse",
  "RunEnvelopeResponse",
  "RunLogsResponse",
  "RunMessageRequest",
  "RunResponse",
  "RunsListResponse",
  "VerdictSummaryResponse",
]

_SKILL_LOADER_MODULE_NAMES = frozenset({"agent", "agent.skills", "agent.skills.loader"})

_CHAT_RUN_STATES = set(CONTROL_RUN_STATES)
_TERMINAL_RUN_STATES = set(CONTROL_TERMINAL_RUN_STATES)
_CONTROL_CHAT_TASK_PREFIX = "control_chat_turn:"
_AUTONOMOUS_RESUME_EVENT_TAIL = 40
_AUTONOMOUS_RESUME_EVENT_BLOCK_MAX_CHARS = 1200
_AUTONOMOUS_RESUME_LOG_TAIL = 80
_AUTONOMOUS_RESUME_LOG_BLOCK_MAX_CHARS = 800
_AUTONOMOUS_RESUME_OPERATOR_MESSAGE_TAIL = 40
_AUTONOMOUS_RESUME_TOOL_RESULT_TAIL = 16
_AUTONOMOUS_RESUME_TOOL_RESULT_BLOCK_MAX_CHARS = 8500
_AUTONOMOUS_RESUME_TOOL_RESULT_DICT_HEAD_ITEMS = 8
_AUTONOMOUS_RESUME_TOOL_RESULT_DICT_TAIL_ITEMS = 6
_AUTONOMOUS_RESUME_TOOL_RESULT_STRING_MAX_CHARS = 240
_AUTONOMOUS_RESUME_TOOL_RESULT_LIST_MAX_ITEMS = 20
_AUTONOMOUS_RESUME_TOOL_RESULT_VALUE_MAX_CHARS = 280
_AUTONOMOUS_RESUME_CONTEXT_MAX_CHARS = 12000
_AUTONOMOUS_RESUME_OPERATOR_BLOCK_MAX_CHARS = 1500
_AUTONOMOUS_RESUME_ORIGINAL_CONTEXT_BLOCK_MAX_CHARS = 1200
_QA_FIXTURE_BRIDGE_HEADER = "x-agent-control-qa-bridge"
_QA_FIXTURE_APPROVAL_ARTIFACT_VALUE = "fixture-approval-artifact"
_QA_FIXTURE_TERMINAL_FAILURE_VALUE = "fixture-terminal-failure"


def _iso_from_unix(timestamp: int | float | None) -> str:
  try:
    value = float(timestamp if timestamp is not None else 0)
  except (TypeError, ValueError):
    value = 0
  return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _require_bearer_session(request: Request, auth: AuthManager) -> GatewaySession:
  token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
  return auth.verify_token(token)


def _session_owner_user_id(session: GatewaySession) -> str:
  owner_user_id = getattr(session, "owner_user_id", None)
  if isinstance(owner_user_id, str) and owner_user_id.strip():
    return owner_user_id.strip()
  risk_user_id = int(getattr(session, "risk_user_id", 0) or 0)
  if risk_user_id > 0:
    return str(risk_user_id)
  return str(session.user_id)


def _record_owner_user_id(record: AutonomousTask) -> str:
  owner_user_id = getattr(record, "owner_user_id", None)
  if isinstance(owner_user_id, str) and owner_user_id.strip():
    return owner_user_id.strip()
  return str(record.user_id)


def _positive_risk_user_id(value: Any) -> int | None:
  try:
    risk_user_id = int(value or 0)
  except (TypeError, ValueError):
    return None
  return risk_user_id if risk_user_id > 0 else None


def _identity_aliases(raw_aliases: Any, owner_user_id: str) -> list[str]:
  aliases = [owner_user_id]
  if isinstance(raw_aliases, (list, tuple)):
    for alias in raw_aliases:
      normalized = str(alias).strip()
      if normalized and normalized not in aliases:
        aliases.append(normalized)
  return aliases


def _session_matches_owner(session: GatewaySession, owner_user_id: str) -> bool:
  return _session_owner_user_id(session) == owner_user_id


def _state_from_session(session: GatewaySession, events: list[dict[str, Any]]) -> ChatRunState:
  if any(pending.get("status") == "approval_pending" for pending in session.pending_tools.values()):
    return "approval_pending"
  if session.stream_active or (session.active_turn is not None and session.active_turn.is_running):
    return "running"
  raw_terminal_state: ChatRunState | None = None
  for event in reversed(events):
    event_type = event.get("type")
    if event_type == "run_state_changed":
      state = coerce_control_run_state(event.get("state"))
      if state is not None:
        return state
    if raw_terminal_state is None:
      if event_type == "error":
        raw_terminal_state = "failed"
      elif event_type == "stream_complete":
        raw_terminal_state = "completed"
  return raw_terminal_state or "starting"


def _ended_at_from_events(events: list[dict[str, Any]]) -> str | None:
  for event in reversed(events):
    if event.get("type") != "run_state_changed":
      continue
    state = coerce_control_run_state(event.get("state"))
    if state not in CONTROL_TERMINAL_RUN_STATES:
      continue
    return _iso_from_unix(event.get("ts"))
  return None


def _skill_run_ids(events: list[dict[str, Any]]) -> list[str]:
  seen: set[str] = set()
  ordered: list[str] = []
  for event in events:
    raw = event.get("skill_run_id")
    if not isinstance(raw, str) or raw in seen:
      continue
    seen.add(raw)
    ordered.append(raw)
  return ordered


def _current_verdict(events: list[dict[str, Any]]) -> VerdictSummaryResponse | None:
  for event in reversed(events):
    if event.get("type") != "skill_result_captured":
      continue
    skill_run_id = event.get("skill_run_id")
    if not isinstance(skill_run_id, str):
      continue
    verdict = _verdict_summary_from_skill_result(event)
    if verdict is None:
      continue
    return VerdictSummaryResponse(
      verdict_token=verdict["verdict_token"],
      confidence=str(verdict["confidence"]) if verdict.get("confidence") is not None else None,
      one_line_summary=str(verdict.get("one_line_summary") or ""),
      skill_run_id=skill_run_id,
    )
  return None


def _coerce_cost_usd(value: Any) -> float | None:
  if value is None:
    return None
  try:
    cost = float(value)
  except (TypeError, ValueError):
    return None
  return cost if cost >= 0 and cost < float("inf") else None


def _usage_cost_usd(usage: Any) -> float | None:
  if not isinstance(usage, dict):
    return None
  for key in ("cost_usd", "total_cost_usd", "estimated_cost_usd", "estimated_cost"):
    cost = _coerce_cost_usd(usage.get(key))
    if cost is not None:
      return cost
  return None


def _events_cost_usd(events: list[dict[str, Any]]) -> float | None:
  stream_total = 0.0
  has_stream_cost = False
  partial_skill_total = 0.0
  has_partial_skill_cost = False
  partial_turn_total = 0.0
  has_partial_turn_cost = False

  for event in events:
    event_type = event.get("type")
    if event_type == "stream_complete":
      cost = _usage_cost_usd(event.get("usage"))
      if cost is not None:
        stream_total += cost
        has_stream_cost = True
        partial_skill_total = 0.0
        has_partial_skill_cost = False
        partial_turn_total = 0.0
        has_partial_turn_cost = False
    elif event_type == "skill_result_captured":
      cost = _coerce_cost_usd(event.get("cost_usd"))
      if cost is not None:
        partial_skill_total += cost
        has_partial_skill_cost = True
    elif event_type == "turn_complete":
      cost = _usage_cost_usd(event.get("usage"))
      if cost is not None:
        partial_turn_total += cost
        has_partial_turn_cost = True

  if has_stream_cost:
    partial_total = partial_skill_total if has_partial_skill_cost else partial_turn_total
    return round(stream_total + partial_total, 6)
  if has_partial_skill_cost:
    return round(partial_skill_total, 6)
  if has_partial_turn_cost:
    return round(partial_turn_total, 6)
  return None


def _verdict_summary_from_skill_result(event: dict[str, Any]) -> dict[str, Any] | None:
  verdict_echo = event.get("verdict_echo")
  if not isinstance(verdict_echo, dict):
    verdict_echo = None
    fms_results = event.get("fms_results")
    if isinstance(fms_results, list):
      for result in reversed(fms_results):
        if isinstance(result, dict) and isinstance(result.get("verdict_echo"), dict):
          verdict_echo = result["verdict_echo"]
          break
  if not isinstance(verdict_echo, dict):
    return None
  token = verdict_echo.get("verdict_token") or verdict_echo.get("verdict")
  if not token:
    return None
  return {
    "verdict_token": str(token),
    "confidence": verdict_echo.get("confidence"),
    "one_line_summary": (
      verdict_echo.get("one_line_summary")
      or verdict_echo.get("summary")
      or verdict_echo.get("message")
      or str(token)
    ),
  }


def _autonomous_state(state: str) -> AutonomousRunState:
  return canonical_control_run_state(state)


def _pending_approval(session: GatewaySession) -> PendingApprovalResponse | None:
  for tool_call_id, pending in sorted(session.pending_tools.items()):
    if pending.get("status") != "approval_pending":
      continue
    return PendingApprovalResponse(
      pending_id=str(tool_call_id),
      tool_name=str(pending.get("tool_name") or ""),
      tool_input=dict(pending.get("tool_input") or {}),
      resolved_qualifier=(
        str(pending["resolved_qualifier"]) if pending.get("resolved_qualifier") is not None else None
      ),
      reason=str(pending["reason"]) if pending.get("reason") is not None else None,
      allow_persistent_approval=bool(pending.get("allow_persistent_approval", False)),
      requested_at=_iso_from_unix(pending.get("requested_at")),
    )
  return None


def _chat_run_from_session(session: GatewaySession) -> ChatRunResponse:
  events = session.event_history.snapshot()
  state = _state_from_session(session, events)
  owner_user_id = _session_owner_user_id(session)
  return ChatRunResponse(
    kind="chat",
    run_id=session.session_id,
    session_id=session.session_id,
    agent="hank",
    channel=session.channel or "web",
    user_id=owner_user_id,
    owner_user_id=owner_user_id,
    raw_user_id=getattr(session, "raw_user_id", None) or session.user_id,
    user_slug=getattr(session, "user_slug", None),
    risk_user_id=_positive_risk_user_id(getattr(session, "risk_user_id", None)),
    user_email=session.user_email,
    user_aliases=_identity_aliases(getattr(session, "user_aliases", None), owner_user_id),
    identity_status=getattr(session, "identity_status", None),
    state=state,
    started_at=_iso_from_unix(session.created_at),
    ended_at=_ended_at_from_events(events) if state in _TERMINAL_RUN_STATES else None,
    cost_usd=_events_cost_usd(events),
    initial_message=session.initial_message,
    skill_run_ids=_skill_run_ids(events),
    current_verdict=_current_verdict(events),
    pending_approval=_pending_approval(session),
    dispatch_scope=session.dispatch_scope,
  )


def _chat_session_has_run_activity(session: GatewaySession) -> bool:
  if session.initial_message.strip():
    return True
  if len(session.event_history) > 0:
    return True
  if session.stream_active or session.active_turn is not None:
    return True
  if session.pending_tools:
    return True
  return any(str(task_key).startswith(_CONTROL_CHAT_TASK_PREFIX) for task_key in session.control_chat_tasks)


def _autonomous_events(record: AutonomousTask) -> list[dict[str, Any]]:
  return list(record.event_lines or [])


def _autonomous_has_pending_approval(events: list[dict[str, Any]]) -> bool:
  requested: set[str] = set()
  decided: set[str] = set()
  for event in events:
    tool_call_id = event.get("tool_call_id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
      continue
    if event.get("type") == "tool_approval_request":
      requested.add(tool_call_id)
    elif event.get("type") == "tool_approval_decided":
      decided.add(tool_call_id)
  return bool(requested - decided)


def _autonomous_pending_approval_events(record: AutonomousTask) -> list[dict[str, Any]]:
  events: list[dict[str, Any]] = []
  decided_approval_ids: set[str] = set()
  decided_tool_call_ids: set[str] = set()
  for event in record.event_lines or []:
    if event.get("type") not in {"approval_decision_sent", "tool_approval_decided"}:
      continue
    approval_id = str(event.get("approval_id") or "")
    tool_call_id = str(event.get("tool_call_id") or "")
    if approval_id:
      decided_approval_ids.add(approval_id)
    if tool_call_id:
      decided_tool_call_ids.add(tool_call_id)

  seen: set[str] = set()
  for event in record.event_lines or []:
    if event.get("type") != "tool_approval_request":
      continue
    approval_id = str(event.get("approval_id") or "")
    tool_call_id = str(event.get("tool_call_id") or "")
    if not approval_id or approval_id in seen:
      continue
    if approval_id in decided_approval_ids or tool_call_id in decided_tool_call_ids:
      continue
    seen.add(approval_id)
    events.append(dict(event))
  return events


def _autonomous_pending_entry_from_event(event: dict[str, Any]) -> dict[str, Any]:
  return {
    "approval_id": str(event.get("approval_id") or ""),
    "nonce": str(event.get("nonce") or ""),
    "requested_at": event.get("ts") or int(time.time()),
    "status": "approval_pending",
    "tool_name": event.get("tool_name"),
    "tool_input": dict(event.get("tool_input") or {}),
    "resolved_qualifier": event.get("resolved_qualifier"),
    "reason": event.get("reason"),
    "allow_persistent_approval": bool(event.get("allow_persistent_approval", False)),
  }


async def _deny_autonomous_pending_approvals_for_cancel(
  *,
  registry: AutonomousRegistry,
  record: AutonomousTask,
  authenticated: GatewaySession,
  app_state: Any,
) -> None:
  pending_events = _autonomous_pending_approval_events(record)
  if not pending_events:
    return
  owner_user_id = _session_owner_user_id(authenticated)

  store = getattr(app_state, "gateway_approval_store", None)
  policy = getattr(app_state, "gateway_approval_policy", None)
  if store is None or policy is None:
    raise ApprovalActionError(503, {"error": "Approval subsystem unavailable"})

  for event in pending_events:
    approval_id = str(event.get("approval_id") or "")
    tool_call_id = str(event.get("tool_call_id") or "")
    nonce = str(event.get("nonce") or "")
    if not approval_id or not tool_call_id or not nonce:
      continue
    request_record = await store.get(approval_id)
    if request_record is None or request_record.state != "pending_user":
      continue

    pending_entry = _autonomous_pending_entry_from_event(event)
    approval_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    shim_session = GatewaySession(
      session_id=record.control_run_id,
      api_key_hash="control-autonomous",
      created_at=int(record.started_at),
      expires_at=int(time.time()) + 600,
      user_id=_record_owner_user_id(record),
      user_email=record.user_email,
      role=authenticated.role,
      kind="chat",
      channel=record.channel,
    )
    shim_session.approval_store = store
    shim_session.approval_policy = policy
    shim_session.pending_tools[tool_call_id] = pending_entry
    shim_session.approval_queues[tool_call_id] = approval_queue

    await _record_vote_and_unblock(
      target_session=shim_session,
      pending_entry=pending_entry,
      tool_call_id=tool_call_id,
      nonce=nonce,
      decider_id=owner_user_id,
      decider_role=getattr(authenticated, "role", None),
      approved=False,
      allow_tool_type=False,
      reason="run_cancelled",
      app_state=app_state,
    )
    try:
      await registry.send_approval_decision(
        record.control_run_id,
        user_id=owner_user_id,
        channel=authenticated.channel,
        approval_id=approval_id,
        tool_call_id=tool_call_id,
        nonce=nonce,
        approved=False,
        allow_tool_type=False,
        reason="run_cancelled",
      )
    except (OSError, PermissionError, RuntimeError, ValueError):
      pass


def _loader_api() -> Any:
  try:
    from agent.skills import loader as skill_loader
  except ModuleNotFoundError as exc:
    if exc.name not in _SKILL_LOADER_MODULE_NAMES:
      raise
    from api.agent.skills import loader as skill_loader  # type: ignore
  return skill_loader


def _skill_is_resumable(skills_dir: Path | None, skill: str | None) -> bool:
  if skills_dir is None or not skill:
    return False
  try:
    metadata = _loader_api().load_skill_metadata(skill, Path(skills_dir), include_catalog_false=True)
  except (FileNotFoundError, ValueError):
    return False
  if metadata is None:
    return False
  if not bool(getattr(metadata, "resumable", False)):
    return False
  return str(getattr(metadata, "mutation_mode", "") or "").strip() != "model_writer"


def _autonomous_task_resumable(record: AutonomousTask, skills_dir: Path | None) -> bool:
  if record.mode != "skill" or not record.skill:
    return False
  if not is_autonomous_run_internal_resumable_state(record.state):
    return False
  return _skill_is_resumable(skills_dir, record.skill)


def _autonomous_task_messageable(
  record: AutonomousTask,
  *,
  state: AutonomousRunState,
  events: list[dict[str, Any]],
) -> bool:
  if not is_control_run_active_state(state):
    return False
  if not is_autonomous_run_internal_messageable_state(record.state):
    return False
  if record.operator_inbox_path is None:
    return False
  if record.proc is not None and record.proc.returncode is not None:
    return False
  return not any(event.get("type") == "stream_complete" for event in events)


def _autonomous_run_from_task(record: AutonomousTask, *, skills_dir: Path | None = None) -> AutonomousRunResponse:
  events = _autonomous_events(record)
  resumed_as = list(record.resumed_as)
  state = _autonomous_state(record.state)
  if state in CONTROL_ACTIVE_RUN_STATES and _autonomous_has_pending_approval(events):
    state = "approval_pending"
  owner_user_id = _record_owner_user_id(record)
  return AutonomousRunResponse(
    kind="autonomous",
    run_id=record.control_run_id,
    task_id=record.task_id,
    agent="hank",
    profile=record.profile,
    mode=record.mode,
    skill=record.skill,
    task=record.task,
    ticker=record.ticker,
    channel=record.channel or "web",
    user_id=owner_user_id,
    owner_user_id=owner_user_id,
    raw_user_id=record.raw_user_id or record.user_id,
    user_slug=record.user_slug,
    risk_user_id=_positive_risk_user_id(record.risk_user_id),
    user_email=record.user_email,
    user_aliases=_identity_aliases(record.user_aliases, owner_user_id),
    identity_status=record.identity_status,
    state=state,
    messageable=_autonomous_task_messageable(record, state=state, events=events),
    started_at=_iso_from_unix(record.started_at),
    ended_at=_iso_from_unix(record.completed_at) if record.completed_at is not None else None,
    cost_usd=_events_cost_usd(events),
    skill_run_ids=_skill_run_ids(events),
    current_verdict=_current_verdict(events),
    resumable=_autonomous_task_resumable(record, skills_dir),
    resumed_from=record.resumed_from,
    resumed_as=resumed_as,
    latest_resume_run_id=resumed_as[-1] if resumed_as else None,
    dispatch_scope=record.dispatch_scope,
    schedule_id=record.schedule_id,
    schedule_name=record.schedule_name,
  )


def _chat_session_for_user(auth: AuthManager, run_id: str, user_id: str) -> GatewaySession:
  session = auth.session_store.get_session(run_id)
  if (
    session is None
    or session.kind != "chat"
    or not _session_matches_owner(session, user_id)
    or not _chat_session_has_run_activity(session)
  ):
    raise HTTPException(status_code=404, detail="Run not found")
  return session


def _require_autonomous_registry(registry: AutonomousRegistry | None) -> AutonomousRegistry:
  if registry is None:
    raise HTTPException(status_code=503, detail="Autonomous registry unavailable")
  return registry


def _autonomous_task_for_user(
  registry: AutonomousRegistry | None,
  control_run_id: str,
  user_id: str,
) -> AutonomousTask:
  resolved_registry = _require_autonomous_registry(registry)
  record = resolved_registry._tasks.get(control_run_id)
  if record is None:
    record = next(
      (task for task in resolved_registry._tasks.values() if task.control_run_id == control_run_id),
      None,
    )
  if record is None or _record_owner_user_id(record) != user_id:
    raise HTTPException(status_code=404, detail="Run not found")
  return record


def _render_log_line(event: dict[str, Any]) -> str:
  return json.dumps(event, sort_keys=True, default=str)


def _session_has_cancel_event(session: GatewaySession) -> bool:
  return any(
    event.get("type") == "run_state_changed" and event.get("state") == "cancelled"
    for event in session.event_history.snapshot()
  )


def _normalize_channel(channel: str | None) -> str | None:
  if not isinstance(channel, str):
    return None
  normalized = channel.strip().lower()
  return normalized or None


def _run_channel_matches(run_channel: str | None, authenticated_channel: str | None) -> bool:
  normalized_run_channel = _normalize_channel(run_channel)
  if normalized_run_channel is None:
    return True
  return _normalize_channel(authenticated_channel) == normalized_run_channel


def _require_run_channel(run_channel: str | None, authenticated_channel: str | None) -> None:
  if not _run_channel_matches(run_channel, authenticated_channel):
    raise HTTPException(status_code=404, detail="Run not found")


def _require_autonomous_channel(record: AutonomousTask, channel: str | None) -> None:
  _require_run_channel(record.channel, channel)


def _payload_field_was_set(payload: Any, field_name: str) -> bool:
  fields_set = getattr(payload, "model_fields_set", None)
  if fields_set is None:
    fields_set = getattr(payload, "__fields_set__", set())
  return field_name in fields_set


def _require_web_safe_autonomous_dispatch(
  payload: AutonomousDispatchRequest,
  *,
  channel: str | None,
  qa_fixture_bridge: str | None = None,
) -> None:
  if _normalize_channel(channel) != "web":
    return

  profile = str(payload.profile or "").strip()
  skill = str(payload.skill or "").strip()
  if (
    qa_fixture_bridge == _QA_FIXTURE_APPROVAL_ARTIFACT_VALUE
    and fixture_provider_available()
    and is_fixture_profile_name(profile)
    and skill == FIXTURE_APPROVAL_HTML_ARTIFACT_SKILL_NAME
    and bool(payload.dev_mode)
  ):
    return
  if (
    qa_fixture_bridge == _QA_FIXTURE_TERMINAL_FAILURE_VALUE
    and fixture_provider_available()
    and is_fixture_profile_name(profile)
    and skill == FIXTURE_TERMINAL_FAILURE_SKILL_NAME
    and bool(payload.dev_mode)
  ):
    return

  if (
    _payload_field_was_set(payload, "dev_mode")
    or bool(payload.dev_mode)
    or is_fixture_profile_name(profile)
    or is_fixture_skill_name(skill)
  ):
    raise HTTPException(
      status_code=403,
      detail={
        "error": "web_control_dev_dispatch_forbidden",
        "message": "Web Agent Control cannot launch fixture or dev-mode autonomous runs.",
      },
    )


def _qa_fixture_bridge_requested(request: Request) -> str | None:
  marker = request.headers.get(_QA_FIXTURE_BRIDGE_HEADER, "")
  normalized = marker.strip().lower()
  if normalized in {_QA_FIXTURE_APPROVAL_ARTIFACT_VALUE, _QA_FIXTURE_TERMINAL_FAILURE_VALUE}:
    return normalized
  return None
