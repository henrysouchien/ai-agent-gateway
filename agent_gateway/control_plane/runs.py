from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from fastapi import APIRouter, Body, HTTPException, Query, Request
from pydantic import BaseModel, Field

from agent_gateway.approvals import ApprovalActionError, _record_vote_and_unblock
from agent_gateway.autonomous_runner import AutonomousRegistry, AutonomousTask
from agent_gateway.event_log import EventLog
from agent_gateway.fixture_gate import (
  FIXTURE_APPROVAL_HTML_ARTIFACT_SKILL_NAME,
  FIXTURE_TERMINAL_FAILURE_SKILL_NAME,
  fixture_provider_available,
  is_fixture_profile_name,
  is_fixture_skill_name,
)
from agent_gateway.session import AuthManager, GatewaySession


ChatRunState = Literal["starting", "queued", "waiting", "running", "approval_pending", "completed", "failed", "cancelled"]
AutonomousRunState = Literal[
  "starting",
  "queued",
  "waiting",
  "running",
  "approval_pending",
  "completed",
  "failed",
  "cancelled",
  "interrupted",
]
_CONTROL_CHAT_TASK_PREFIX = "control_chat_turn:"
_AUTONOMOUS_RESUME_EVENT_TAIL = 40
_AUTONOMOUS_RESUME_LOG_TAIL = 80
_AUTONOMOUS_RESUME_OPERATOR_MESSAGE_TAIL = 40
_AUTONOMOUS_RESUME_CONTEXT_MAX_CHARS = 12000
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
      if state in {"starting", "running", "approval_pending", "completed", "failed", "cancelled"}:
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
    if state not in {"completed", "failed", "cancelled"}:
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
  if state in {"finished", "completed"}:
    return "completed"
  if state == "failed":
    return "failed"
  if state in {"queued", "waiting", "running", "approval_pending", "interrupted"}:
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
    ended_at=_ended_at_from_events(events) if state in {"completed", "failed", "cancelled"} else None,
    cost_usd=None,
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
  return bool(getattr(metadata, "resumable", False)) if metadata is not None else False


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
    cost_usd=None,
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


def _clip_resume_text(text: str, *, limit: int) -> str:
  if len(text) <= limit:
    return text
  return text[: max(0, limit - 120)].rstrip() + "\n[truncated for autonomous resume context]"


def _tail_text_lines(path: Path, line_count: int) -> list[str]:
  if line_count <= 0:
    return []
  recent = deque(maxlen=line_count)
  try:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
      for line in handle:
        recent.append(line.rstrip("\n"))
  except FileNotFoundError:
    return []
  return list(recent)


def _operator_message_tail(record: AutonomousTask) -> list[dict[str, Any]]:
  path = record.operator_inbox_path
  if path is None or not path.exists():
    return []
  messages: list[dict[str, Any]] = []
  for line in _tail_text_lines(path, _AUTONOMOUS_RESUME_OPERATOR_MESSAGE_TAIL):
    try:
      payload = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(payload, dict):
      messages.append(payload)
  return messages


def _file_tail_lines(path: Path, line_count: int) -> list[str]:
  return _tail_text_lines(path, line_count)


def _build_autonomous_resume_context(record: AutonomousTask, payload: AutonomousResumeRequest) -> str:
  parts: list[str] = [
    (
      f"Resume autonomous control run {record.control_run_id}. "
      f"The prior top-level process ended with state={_autonomous_state(record.state)} "
      f"and exit_code={record.exit_code}."
    ),
    (
      "Continue from the latest safe point. Do not repeat durable writes that already succeeded; "
      "inspect current state before writing again, and summarize what was resumed."
    ),
  ]
  operator_note = payload.message or payload.context
  if isinstance(operator_note, str) and operator_note.strip():
    parts.append(f"Operator resume instruction:\n{operator_note.strip()}")

  operator_messages = _operator_message_tail(record)
  if operator_messages:
    parts.append(
      "Recent operator messages:\n"
      + json.dumps(operator_messages, sort_keys=True, default=str, indent=2)
    )

  events = _autonomous_events(record)[-_AUTONOMOUS_RESUME_EVENT_TAIL:]
  if events:
    parts.append("Recent control events:\n" + json.dumps(events, sort_keys=True, default=str, indent=2))

  if record.log_path.exists():
    log_lines = _file_tail_lines(record.log_path, _AUTONOMOUS_RESUME_LOG_TAIL)
    if log_lines:
      parts.append("Recent log tail:\n" + "\n".join(log_lines))

  if record.context:
    parts.append(
      "Original context:\n"
      + _clip_resume_text(record.context, limit=max(1000, _AUTONOMOUS_RESUME_CONTEXT_MAX_CHARS // 3))
    )
  return _clip_resume_text("\n\n".join(parts), limit=_AUTONOMOUS_RESUME_CONTEXT_MAX_CHARS)


def _require_control_session(session: GatewaySession) -> None:
  if session.kind != "control":
    raise HTTPException(status_code=401, detail="Control session required")


def _require_chat_session_for_run(authenticated: GatewaySession, target: GatewaySession) -> None:
  if authenticated.kind == "chat" and authenticated.session_id == target.session_id:
    return
  if authenticated.kind == "control" and authenticated.user_id == target.user_id:
    _require_run_channel(target.channel, authenticated.channel)
    return
  raise HTTPException(status_code=401, detail="Chat session token or matching control session required for this run")


def _transcript_dir_from_app_state(app_state: Any) -> Path | None:
  config = getattr(app_state, "gateway_config", None)
  raw = getattr(config, "transcript_dir", None)
  if raw is None:
    return None
  return Path(raw)


def _run_state_event(control_run_id: str, state: str) -> dict[str, Any]:
  return {
    "type": "run_state_changed",
    "run_id": control_run_id,
    "control_run_id": control_run_id,
    "state": state,
    "ts": int(time.time()),
  }


def _event_for_run(event: dict[str, Any], control_run_id: str) -> dict[str, Any]:
  event_copy = dict(event)
  event_copy.setdefault("run_id", control_run_id)
  event_copy.setdefault("control_run_id", control_run_id)
  return event_copy


def _latest_user_message_content(messages: list[ChatMessage]) -> str | None:
  for message in reversed(messages):
    if message.role.strip().lower() != "user":
      continue
    content = message.content.strip()
    if content:
      return content
  return None


def _control_message_id(request_id: str | None) -> str:
  normalized = request_id.strip() if isinstance(request_id, str) else ""
  return normalized or str(uuid.uuid4())


def _has_parent_message_event(session: GatewaySession, message_id: str) -> bool:
  return any(
    event.get("type") == "parent_message_sent"
    and (event.get("message_id") == message_id or event.get("request_id") == message_id)
    for event in session.event_history.snapshot()
  )


async def _maybe_call_on_event(callback: Any, event: dict[str, Any], control_run_id: str) -> None:
  if not callable(callback):
    return
  result = callback(event, control_run_id)
  if inspect.isawaitable(result):
    await result


async def _publish_control_event(app_state: Any, user_id: str, control_run_id: str, event: dict[str, Any]) -> None:
  event_copy = _event_for_run(event, control_run_id)
  user_event_bus = getattr(app_state, "user_event_bus", None)
  if user_event_bus is not None:
    await user_event_bus.publish(
      user_id=user_id,
      control_run_id=control_run_id,
      event=event_copy,
    )
  config = getattr(app_state, "gateway_config", None)
  await _maybe_call_on_event(getattr(config, "on_event", None), event_copy, control_run_id)


async def _cleanup_run_buffer(app_state: Any, user_id: str, control_run_id: str) -> None:
  user_event_bus = getattr(app_state, "user_event_bus", None)
  if user_event_bus is not None:
    await user_event_bus.cleanup_run(user_id, control_run_id)


async def _record_chat_parent_message_event(
  *,
  app_state: Any,
  session: GatewaySession,
  messages: list[ChatMessage],
  request_id: str | None,
) -> None:
  message = _latest_user_message_content(messages)
  if message is None:
    return

  control_run_id = session.session_id
  message_id = _control_message_id(request_id)
  if _has_parent_message_event(session, message_id):
    return

  sent_at = time.time()
  event = {
    "type": "parent_message_sent",
    "run_id": control_run_id,
    "control_run_id": control_run_id,
    "session_id": control_run_id,
    "message_id": message_id,
    "request_id": message_id,
    "message": message,
    "channel": session.channel or "web",
    "sender": {
      "session_id": control_run_id,
      "user_id": session.user_id,
    },
    "sent_at": sent_at,
    "ts": sent_at,
  }
  session.event_history.append(event)
  await _publish_control_event(app_state, session.user_id, control_run_id, event)


async def _cancel_control_chat_background_tasks(
  session: GatewaySession,
  *,
  settle_timeout: float = 0.0,
) -> None:
  task_items: list[tuple[str, asyncio.Future[Any]]] = []
  for task_key, task in list(session.control_chat_tasks.items()):
    if not str(task_key).startswith(_CONTROL_CHAT_TASK_PREFIX):
      continue
    task_items.append((task_key, task))

  if not task_items:
    return

  tasks = [task for _task_key, task in task_items]
  if settle_timeout > 0:
    await asyncio.wait(tasks, timeout=settle_timeout)

  for task_key, task in task_items:
    session.control_chat_tasks.pop(task_key, None)
    if not task.done():
      task.cancel()

  await asyncio.gather(*tasks, return_exceptions=True)


async def cleanup_control_chat_tasks(session: GatewaySession) -> None:
  await _cancel_control_chat_background_tasks(session)


async def _finalize_control_chat_task(
  *,
  task: asyncio.Task[Any],
  session: GatewaySession,
  app_state: Any,
  task_key: str,
) -> None:
  state = "failed"
  try:
    result = await task
    state = str(getattr(result, "state", "") or _state_from_session(session, session.event_history.snapshot()))
  except asyncio.CancelledError:
    state = "cancelled"
  except Exception:
    state = "failed"
  finally:
    session.control_chat_tasks.pop(task_key, None)
    if _session_has_cancel_event(session):
      await _cleanup_run_buffer(app_state, session.user_id, session.session_id)
      return
    if state not in {"approval_pending", "running", "starting"}:
      await _cleanup_run_buffer(app_state, session.user_id, session.session_id)


async def _dispatch_control_chat_turn(
  *,
  request: Request,
  session: GatewaySession,
  messages: list[ChatMessage],
  request_id: str | None,
  context: dict[str, Any],
  model: str | None,
  deadline_sec: int | None,
  record_parent_message: bool = False,
) -> ChatRunResponse:
  from agent_gateway.server import ChatMessage as ServerChatMessage
  from agent_gateway.server import ChatTurnInputs, _dispatch_chat_turn

  app_state = request.app.state
  control_run_id = session.session_id
  pending_seen = asyncio.Event()
  server_messages = [
    message if isinstance(message, ServerChatMessage) else ServerChatMessage(role=message.role, content=message.content)
    for message in messages
  ]
  if record_parent_message:
    if session.stream_active or (session.active_turn is not None and session.active_turn.is_running):
      raise HTTPException(status_code=409, detail="A turn is already running; subscribe via /chat/subscribe")
    await _record_chat_parent_message_event(
      app_state=app_state,
      session=session,
      messages=messages,
      request_id=request_id,
    )

  async def _on_event(event: dict[str, Any]) -> None:
    event_with_run = _event_for_run(event, control_run_id)
    await _publish_control_event(app_state, session.user_id, control_run_id, event_with_run)
    if event_with_run.get("type") == "tool_approval_request":
      await _publish_control_event(
        app_state,
        session.user_id,
        control_run_id,
        _run_state_event(control_run_id, "approval_pending"),
      )
      pending_seen.set()

  event_log = EventLog(session_id=control_run_id)
  dispatch_task = asyncio.create_task(
    _dispatch_chat_turn(
      session,
      ChatTurnInputs(
        messages=server_messages,
        request_id=request_id,
        context=dict(context),
        metadata=None,
        model=model,
      ),
      event_log=event_log,
      on_event=_on_event,
      build_chat_runtime=app_state.gateway_build_chat_runtime,
      credentials_resolver=getattr(getattr(app_state, "gateway_config", None), "credentials_refresh_resolver", None),
      transcript_dir=_transcript_dir_from_app_state(app_state),
      publish_lifecycle_events=True,
    )
  )
  task_key = f"control_chat_turn:{control_run_id}:{id(dispatch_task)}"
  session.control_chat_tasks[task_key] = dispatch_task

  pending_task = asyncio.create_task(pending_seen.wait())
  timeout = float(deadline_sec) if deadline_sec is not None else None
  done, _pending = await asyncio.wait(
    {dispatch_task, pending_task},
    timeout=timeout,
    return_when=asyncio.FIRST_COMPLETED,
  )

  if dispatch_task in done:
    session.control_chat_tasks.pop(task_key, None)
    pending_task.cancel()
    await asyncio.gather(pending_task, return_exceptions=True)
    result = await dispatch_task
    state = str(getattr(result, "state", "") or _state_from_session(session, session.event_history.snapshot()))
    if state not in {"approval_pending", "running", "starting"}:
      await _cleanup_run_buffer(app_state, session.user_id, control_run_id)
    return _chat_run_from_session(session)

  if pending_task in done:
    await asyncio.gather(pending_task, return_exceptions=True)
  else:
    pending_task.cancel()
    await asyncio.gather(pending_task, return_exceptions=True)

  asyncio.create_task(
    _finalize_control_chat_task(
      task=dispatch_task,
      session=session,
      app_state=app_state,
      task_key=task_key,
    )
  )
  return _chat_run_from_session(session)


def build_runs_router(
  *,
  auth: AuthManager,
  autonomous_registry: AutonomousRegistry | None = None,
  skills_dir: Path | None = None,
) -> APIRouter:
  router = APIRouter(prefix="/runs")
  skills_root = Path(skills_dir) if skills_dir is not None else None

  @router.post("", response_model=RunDispatchResponse)
  async def dispatch_run(request: Request, payload: ControlRunDispatchRequest) -> RunDispatchResponse:
    authenticated = _require_bearer_session(request, auth)
    if payload.kind == "chat":
      _require_control_session(authenticated)
      requested_channel = _normalize_channel(payload.channel)
      session_channel = _normalize_channel(authenticated.channel)
      if session_channel is not None and requested_channel is not None and session_channel != requested_channel:
        raise HTTPException(status_code=401, detail="Channel mismatch")
      channel = session_channel or requested_channel
      chat_session = auth.session_store.create_session(
        api_key_hash=authenticated.api_key_hash,
        user_id=authenticated.user_id,
        user_email=authenticated.user_email,
        risk_user_id=authenticated.risk_user_id,
        role=authenticated.role,
        kind="chat",
        auth_config=authenticated.auth_config,
      )
      chat_session.channel = channel
      chat_session.is_public = channel == "public"
      chat_session.approval_store = getattr(request.app.state, "gateway_approval_store", None)
      chat_session.approval_policy = getattr(request.app.state, "gateway_approval_policy", None)
      chat_session.initial_message = payload.message

      context: dict[str, Any] = {"channel": channel} if channel is not None else {}
      if payload.skill is not None:
        context["skill"] = payload.skill
      if payload.ticker is not None:
        context["ticker"] = payload.ticker
      if payload.dev_mode:
        context["dev_mode"] = True

      run = await _dispatch_control_chat_turn(
        request=request,
        session=chat_session,
        messages=[ChatMessage(role="user", content=payload.message)],
        request_id=None,
        context=context,
        model=None,
        deadline_sec=payload.deadline_sec,
      )
      token = auth.issue_token(chat_session)
      return ChatDispatchResponse(
        run=run,
        chat_session_token=token,
        chat_session_id=chat_session.session_id,
        chat_session_expires_at=chat_session.expires_at,
      )

    _require_control_session(authenticated)
    requested_channel = _normalize_channel(payload.channel)
    session_channel = _normalize_channel(authenticated.channel)
    if session_channel is not None and requested_channel is not None and session_channel != requested_channel:
      raise HTTPException(status_code=401, detail="Channel mismatch")
    channel = session_channel or requested_channel
    _require_web_safe_autonomous_dispatch(
      payload,
      channel=channel,
      qa_fixture_bridge=_qa_fixture_bridge_requested(request),
    )

    if not payload.profile or not payload.mode:
      raise HTTPException(status_code=422, detail="profile and mode are required")

    registry = _require_autonomous_registry(autonomous_registry)
    registry.set_user_event_bus(getattr(request.app.state, "user_event_bus", None))
    try:
      start_payload = await registry.start(
        profile=payload.profile,
        mode=payload.mode,
        task=payload.task,
        skill=payload.skill,
        context=payload.context,
        ticker=payload.ticker,
        channel=channel,
        dev_mode=payload.dev_mode,
        user_id=authenticated.user_id,
        user_email=authenticated.user_email,
      )
    except ValueError as exc:
      raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
      detail = str(exc)
      status_code = 429 if "concurrency limit" in detail.lower() else 409
      raise HTTPException(status_code=status_code, detail=detail) from exc
    record = _autonomous_task_for_user(registry, str(start_payload["task_id"]), authenticated.user_id)
    run = _autonomous_run_from_task(record, skills_dir=skills_root)
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
    target_session = auth.session_store.get_session(control_run_id)
    if target_session is None:
      if autonomous_registry is not None:
        try:
          record = _autonomous_task_for_user(autonomous_registry, control_run_id, authenticated.user_id)
        except HTTPException as exc:
          if exc.status_code != 404:
            raise
        else:
          _require_control_session(authenticated)
          if not isinstance(payload, AutonomousRunMessageRequest):
            raise HTTPException(status_code=422, detail="Autonomous runs require message")

          autonomous_registry.set_user_event_bus(getattr(request.app.state, "user_event_bus", None))
          try:
            delivery = await autonomous_registry.send_operator_message(
              control_run_id,
              user_id=authenticated.user_id,
              channel=authenticated.channel,
              message=payload.message,
              message_id=payload.message_id,
            )
          except PermissionError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
          except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
          except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

          return RunEnvelopeResponse(
            run=_autonomous_run_from_task(record, skills_dir=skills_root),
            message_id=str(delivery.get("message_id") or ""),
            delivery_status=delivery.get("delivery_status"),  # type: ignore[arg-type]
          )
      raise HTTPException(status_code=404, detail="Run not found")
    if target_session.kind != "chat":
      raise HTTPException(status_code=409, detail="Run does not accept additional messages")
    if not _chat_session_has_run_activity(target_session):
      raise HTTPException(status_code=404, detail="Run not found")
    _require_chat_session_for_run(authenticated, target_session)
    if not isinstance(payload, ChatContinuationRequest):
      raise HTTPException(status_code=422, detail="Chat runs require messages")

    context = dict(payload.context or {})
    if target_session.channel is not None:
      context.setdefault("channel", target_session.channel)
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
      model=payload.model,
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

    registry = _require_autonomous_registry(autonomous_registry)
    registry.set_user_event_bus(getattr(request.app.state, "user_event_bus", None))
    record = _autonomous_task_for_user(registry, control_run_id, authenticated.user_id)
    _require_autonomous_channel(record, authenticated.channel)

    async with record.resume_lock:
      if not _autonomous_task_resumable(record, skills_root):
        raise HTTPException(status_code=409, detail="Autonomous run is not resumable")

      for resumed_run_id in reversed(record.resumed_as):
        resumed_record = registry._find_by_control_run_id(resumed_run_id)
        resumed_state = _autonomous_run_from_task(resumed_record, skills_dir=skills_root).state if resumed_record is not None else None
        if resumed_state in {"starting", "running", "approval_pending"}:
          raise HTTPException(status_code=409, detail="Autonomous run already has an active resume")

      resume_payload = payload or AutonomousResumeRequest()
      resume_context = _build_autonomous_resume_context(record, resume_payload)
      try:
        start_payload = await registry.start(
          profile=record.profile,
          mode=record.mode,
          task=record.task,
          skill=record.skill,
          context=resume_context,
          ticker=record.ticker,
          channel=record.channel,
          dev_mode=record.dev_mode,
          user_id=authenticated.user_id,
          user_email=authenticated.user_email,
          resumed_from=record.control_run_id,
        )
      except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
      except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

      resumed_record = _autonomous_task_for_user(registry, str(start_payload["task_id"]), authenticated.user_id)
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
        run=_autonomous_run_from_task(resumed_record, skills_dir=skills_root),
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
    auth.session_store.cleanup_expired()
    if kind is not None and kind not in {"chat", "autonomous"}:
      return RunsListResponse(runs=[])

    runs: list[RunResponse] = []
    if kind in {None, "chat"}:
      runs.extend(
        _chat_run_from_session(session)
        for session in auth.session_store.sessions.values()
        if (
          session.kind == "chat"
          and session.user_id == authenticated.user_id
          and _run_channel_matches(session.channel, authenticated.channel)
          and _chat_session_has_run_activity(session)
        )
      )
    if kind in {None, "autonomous"} and autonomous_registry is not None:
      runs.extend(
        _autonomous_run_from_task(record, skills_dir=skills_root)
        for record in autonomous_registry._tasks.values()
        if record.user_id == authenticated.user_id and _run_channel_matches(record.channel, authenticated.channel)
      )
    if state is not None:
      runs = [run for run in runs if run.state == state]
    runs.sort(key=lambda run: run.started_at, reverse=True)
    return RunsListResponse(runs=runs[:limit])

  @router.get("/{control_run_id}", response_model=RunResponse)
  async def get_run(request: Request, control_run_id: str) -> RunResponse:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    if control_run_id.startswith("bg_") or (
      autonomous_registry is not None
      and any(task.control_run_id == control_run_id for task in autonomous_registry._tasks.values())
    ):
      record = _autonomous_task_for_user(autonomous_registry, control_run_id, authenticated.user_id)
      _require_autonomous_channel(record, authenticated.channel)
      return _autonomous_run_from_task(record, skills_dir=skills_root)
    session = _chat_session_for_user(auth, control_run_id, authenticated.user_id)
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
    if control_run_id.startswith("bg_") or (
      autonomous_registry is not None
      and any(task.control_run_id == control_run_id for task in autonomous_registry._tasks.values())
    ):
      registry = _require_autonomous_registry(autonomous_registry)
      record = _autonomous_task_for_user(registry, control_run_id, authenticated.user_id)
      _require_autonomous_channel(record, authenticated.channel)
      logs = registry.logs(record.task_id, tail=tail)
      total_lines = int(logs.get("total_lines", 0) or 0)
      return RunLogsResponse(
        run_id=record.control_run_id,
        log_lines=list(logs.get("lines", [])),
        more_available=total_lines > tail,
      )
    session = _chat_session_for_user(auth, control_run_id, authenticated.user_id)
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
    if not control_run_id.startswith("bg_"):
      session = auth.session_store.get_session(control_run_id)
      if session is not None:
        if session.kind != "chat" or session.user_id != authenticated.user_id:
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
              decider_id=authenticated.user_id,
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
        await _publish_control_event(request.app.state, session.user_id, control_run_id, cancelled_event)
        await _cancel_control_chat_background_tasks(
          session,
          settle_timeout=0.05 if pending_snapshot else 0.0,
        )
        run = _chat_run_from_session(session)
        await auth.session_store.expire_session_async(control_run_id)
        await _cleanup_run_buffer(request.app.state, session.user_id, control_run_id)
        return run
    registry = _require_autonomous_registry(autonomous_registry)
    record = _autonomous_task_for_user(registry, control_run_id, authenticated.user_id)
    _require_autonomous_channel(record, authenticated.channel)
    try:
      await _deny_autonomous_pending_approvals_for_cancel(
        registry=registry,
        record=record,
        authenticated=authenticated,
        app_state=request.app.state,
      )
    except ApprovalActionError as exc:
      raise HTTPException(status_code=exc.status_code, detail=exc.payload) from exc
    await registry.cancel(record.task_id)
    return _autonomous_run_from_task(record, skills_dir=skills_root)

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
