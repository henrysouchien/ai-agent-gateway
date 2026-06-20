from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from fastapi import Body, HTTPException, Request
from pydantic import BaseModel, Field

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

ChatRunState = Literal[
  "starting",
  "queued",
  "waiting",
  "running",
  "approval_pending",
  "completed",
  "failed",
  "cancelled",
  "budget_limited",
  "blocked",
  "remediating",
]
AutonomousRunState = Literal[
  "starting",
  "queued",
  "waiting",
  "running",
  "approval_pending",
  "completed",
  "failed",
  "cancelled",
  "budget_limited",
  "blocked",
  "remediating",
  "interrupted",
]
_CHAT_RUN_STATES = {
  "starting",
  "queued",
  "waiting",
  "running",
  "approval_pending",
  "remediating",
  "completed",
  "failed",
  "cancelled",
  "budget_limited",
  "blocked",
}
_TERMINAL_RUN_STATES = {"completed", "failed", "cancelled", "budget_limited", "blocked"}
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


class VerdictSummaryResponse(BaseModel):
  verdict_token: str
  confidence: str | None
  one_line_summary: str
  skill_run_id: str


class PendingApprovalResponse(BaseModel):
  pending_id: str
  tool_name: str
  tool_input: dict[str, Any]
  resolved_qualifier: str | None
  reason: str | None
  allow_persistent_approval: bool
  requested_at: str


class ChatRunResponse(BaseModel):
  kind: Literal["chat"]
  run_id: str
  session_id: str
  agent: Literal["hank"]
  channel: str
  user_id: str
  state: ChatRunState
  started_at: str
  ended_at: str | None
  cost_usd: float | None
  initial_message: str
  skill_run_ids: list[str]
  current_verdict: VerdictSummaryResponse | None
  pending_approval: PendingApprovalResponse | None


class AutonomousRunResponse(BaseModel):
  kind: Literal["autonomous"]
  run_id: str
  task_id: str
  agent: Literal["hank"]
  profile: str
  mode: str
  skill: str | None
  task: str | None
  ticker: str | None
  channel: str
  user_id: str
  state: AutonomousRunState
  messageable: bool = False
  started_at: str
  ended_at: str | None
  cost_usd: float | None
  skill_run_ids: list[str]
  current_verdict: VerdictSummaryResponse | None
  resumable: bool = False
  resumed_from: str | None = None
  resumed_as: list[str] = Field(default_factory=list)
  latest_resume_run_id: str | None = None


class ChatMessage(BaseModel):
  role: str
  content: str


class ChatDispatchRequest(BaseModel):
  kind: Literal["chat"]
  message: str = Field(..., min_length=1)
  channel: str = Field(..., min_length=1)
  skill: str | None = None
  ticker: str | None = None
  dev_mode: bool = False
  deadline_sec: int | None = Field(default=None, ge=1)


class AutonomousDispatchRequest(BaseModel):
  kind: Literal["autonomous"]
  profile: str | None = None
  mode: str | None = None
  skill: str | None = None
  task: str | None = None
  ticker: str | None = None
  context: str | None = None
  channel: str | None = None
  dev_mode: bool = False


ControlRunDispatchRequest = Annotated[
  Union[ChatDispatchRequest, AutonomousDispatchRequest],
  Body(discriminator="kind"),
]


class AutonomousDispatchResponse(BaseModel):
  run: AutonomousRunResponse
  task_id: str
  run_id: str
  log_path: str
  started_at: int
  cmd: list[str]
  resumed_from: str | None = None


class ChatDispatchResponse(BaseModel):
  run: ChatRunResponse
  chat_session_token: str
  chat_session_id: str
  chat_session_expires_at: int


RunDispatchResponse = Union[ChatDispatchResponse, AutonomousDispatchResponse]
RunResponse = Union[ChatRunResponse, AutonomousRunResponse]


class RunsListResponse(BaseModel):
  runs: list[RunResponse]


class RunLogsResponse(BaseModel):
  run_id: str
  log_lines: list[str]
  more_available: bool


class ChatContinuationRequest(BaseModel):
  messages: list[ChatMessage]
  request_id: str | None = None
  context: dict[str, Any] = Field(default_factory=dict)
  model: str | None = None
  deadline_sec: int | None = Field(default=None, ge=1)


class AutonomousRunMessageRequest(BaseModel):
  message: str = Field(..., min_length=1)
  message_id: str | None = None


class AutonomousResumeRequest(BaseModel):
  context: str | None = None
  message: str | None = None
  request_id: str | None = None


RunMessageRequest = Union[ChatContinuationRequest, AutonomousRunMessageRequest]


class RunEnvelopeResponse(BaseModel):
  run: RunResponse
  message_id: str | None = None
  delivery_status: Literal["delivered", "duplicate"] | None = None


def _iso_from_unix(timestamp: int | float | None) -> str:
  try:
    value = float(timestamp if timestamp is not None else 0)
  except (TypeError, ValueError):
    value = 0
  return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _require_bearer_session(request: Request, auth: AuthManager) -> GatewaySession:
  token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
  return auth.verify_token(token)


def _state_from_session(session: GatewaySession, events: list[dict[str, Any]]) -> ChatRunState:
  if any(pending.get("status") == "approval_pending" for pending in session.pending_tools.values()):
    return "approval_pending"
  if session.stream_active or (session.active_turn is not None and session.active_turn.is_running):
    return "running"
  raw_terminal_state: ChatRunState | None = None
  for event in reversed(events):
    event_type = event.get("type")
    if event_type == "run_state_changed":
      state = event.get("state")
      if state in _CHAT_RUN_STATES:
        return state  # type: ignore[return-value]
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
    state = event.get("state")
    if state not in _TERMINAL_RUN_STATES:
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
  if state == "killed":
    return "cancelled"
  if state in {"budget_limited", "budget_exceeded"}:
    return "budget_limited"
  if state == "blocked":
    return "blocked"
  if state in {"finished", "completed"}:
    return "completed"
  if state == "failed":
    return "failed"
  if state in {"queued", "waiting", "running", "approval_pending", "remediating", "interrupted"}:
    return state
  if state == "starting":
    return "starting"
  return "running"


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
  return ChatRunResponse(
    kind="chat",
    run_id=session.session_id,
    session_id=session.session_id,
    agent="hank",
    channel=session.channel or "web",
    user_id=session.user_id,
    state=state,
    started_at=_iso_from_unix(session.created_at),
    ended_at=_ended_at_from_events(events) if state in _TERMINAL_RUN_STATES else None,
    cost_usd=_events_cost_usd(events),
    initial_message=session.initial_message,
    skill_run_ids=_skill_run_ids(events),
    current_verdict=_current_verdict(events),
    pending_approval=_pending_approval(session),
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
      user_id=record.user_id,
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
      decider_id=authenticated.user_id,
      decider_role=getattr(authenticated, "role", None),
      approved=False,
      allow_tool_type=False,
      reason="run_cancelled",
      app_state=app_state,
    )
    try:
      await registry.send_approval_decision(
        record.control_run_id,
        user_id=authenticated.user_id,
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
  except ModuleNotFoundError:
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
  if _autonomous_state(record.state) not in {"failed", "cancelled", "interrupted"}:
    return False
  return _skill_is_resumable(skills_dir, record.skill)


def _autonomous_task_messageable(
  record: AutonomousTask,
  *,
  state: AutonomousRunState,
  events: list[dict[str, Any]],
) -> bool:
  if state not in {"running", "waiting", "approval_pending"}:
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
  if state == "running" and _autonomous_has_pending_approval(events):
    state = "approval_pending"
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
    user_id=record.user_id,
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
  )


def _chat_session_for_user(auth: AuthManager, run_id: str, user_id: str) -> GatewaySession:
  session = auth.session_store.get_session(run_id)
  if (
    session is None
    or session.kind != "chat"
    or session.user_id != user_id
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
  if record is None or record.user_id != user_id:
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


def _payload_field_was_set(payload: BaseModel, field_name: str) -> bool:
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
