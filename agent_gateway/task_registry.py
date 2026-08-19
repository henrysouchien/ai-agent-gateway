from __future__ import annotations

import asyncio
import html
import json
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Callable, Literal, Protocol

from agent_workflow_contracts import (
  AdmittedTask,
  AgentCompletionEnvelope,
  ParentResultPolicy,
  TaskResult,
  TaskResultRef,
)

from .agent_session_log_records import (
  EVENT_SCHEMA_VERSION,
  is_current_event_schema_version,
)
from .events import AgentCompletionEvent, event_from_dict
from .skill_lifecycle import (
  SKILL_RESULT_CORE_FIELDS,
  SkillLifecycleArtifactIdentity,
  TopLevelSkillLifecycleMetadata,
)

log = logging.getLogger("agent_gateway.task_registry")


class TaskState(Enum):
  PENDING = "pending"
  RUNNING = "running"
  COMPLETED = "completed"
  FAILED = "failed"
  KILLED = "killed"
  INTERRUPTED = "interrupted"


TerminationIntent = Literal["cancelled", "killed"]
CompletionPersistenceState = Literal[
  "not_started",
  "in_flight",
  "committed",
  "uncertain",
]
NotificationDeliveryState = Literal[
  "not_queued",
  "queued",
  "delivered",
  "payload_omitted",
  "queue_omitted",
]
_NOTIFICATION_RETRIEVAL_REQUIRED_STATES = frozenset({
  "payload_omitted",
  "queue_omitted",
})
TASK_NOTIFICATION_INLINE_PAYLOAD_MAX_BYTES = 32_768
TASK_NOTIFICATION_OVERFLOW_SAMPLE_TASK_IDS = 5


def _notification_payload_preflight_reason(payload: Any) -> str | None:
  """Reject unbounded or unserializable payloads before JSON allocation."""
  remaining = TASK_NOTIFICATION_INLINE_PAYLOAD_MAX_BYTES
  active_container_ids: set[int] = set()
  stack: list[tuple[Any, bool]] = [(payload, False)]

  while stack:
    value, exiting = stack.pop()
    if exiting:
      active_container_ids.discard(id(value))
      continue

    if value is None:
      remaining -= 4
    elif isinstance(value, bool):
      remaining -= 5
    elif isinstance(value, str):
      if len(value) > remaining:
        return "payload_too_large"
      try:
        remaining -= len(value.encode("utf-8")) + 2
      except UnicodeEncodeError:
        return "non_utf8_payload"
    elif isinstance(value, int):
      digits = max(1, int(value.bit_length() * 0.302) + 2)
      remaining -= digits + (1 if value < 0 else 0)
    elif isinstance(value, float):
      remaining -= 32
    elif isinstance(value, dict):
      container_id = id(value)
      if container_id in active_container_ids:
        return "non_json_payload"
      if len(value) > remaining:
        return "payload_too_large"
      active_container_ids.add(container_id)
      remaining -= 2 + (2 * len(value))
      stack.append((value, True))
      for key, item in reversed(list(value.items())):
        stack.append((item, False))
        stack.append((key, False))
    elif isinstance(value, (list, tuple)):
      container_id = id(value)
      if container_id in active_container_ids:
        return "non_json_payload"
      if len(value) > remaining:
        return "payload_too_large"
      active_container_ids.add(container_id)
      remaining -= 2 + len(value)
      stack.append((value, True))
      stack.extend((item, False) for item in reversed(value))
    else:
      return "non_json_payload"

    if remaining < 0:
      return "payload_too_large"

  return None


class ResumeSuccessorConflictError(RuntimeError):
  """Raised when a deterministic resume ID belongs to another lineage."""

  def __init__(
    self,
    *,
    task_id: str,
    original_task_id: str,
    existing: "TaskEntry",
  ) -> None:
    self.task_id = task_id
    self.original_task_id = original_task_id
    self.existing_original_task_id = existing.original_task_id
    self.existing_task_type = existing.task_type
    super().__init__(
      (
        f"Resume successor {task_id} is already bound to "
        f"original_task_id={existing.original_task_id!r}, "
        f"task_type={existing.task_type!r}; requested "
        f"original_task_id={original_task_id!r}"
      )
    )


class NotificationRetrievalCapacityError(RuntimeError):
  """Raised when unretrieved omitted results exhaust bounded retention."""


class RequiredSkillResultMarkerValidationError(RuntimeError):
  """A durable required-skill marker does not match its task lifecycle."""


class TaskDurableEventConflictError(RuntimeError):
  """One durable task identity has conflicting registration or settlement."""


class TaskResultSettlementError(ValueError):
  """The outer task completion conflicts with its embedded TaskResult."""


def task_state_for_result(result: TaskResult) -> TaskState:
  status = result.execution.status
  if status in {"succeeded", "skipped"}:
    return TaskState.COMPLETED
  if status == "cancelled":
    return TaskState.KILLED
  if status == "interrupted":
    return TaskState.INTERRUPTED
  return TaskState.FAILED


def validate_task_result_settlement(
  result: TaskResult,
  *,
  final_state: TaskState | str,
  error: dict[str, Any] | None,
) -> TaskState:
  """Validate the redundant outer settlement against canonical execution."""

  try:
    actual = (
      final_state
      if isinstance(final_state, TaskState)
      else TaskState(final_state)
    )
  except (TypeError, ValueError) as exc:
    raise TaskResultSettlementError(
      "task completion final_state is not canonical"
    ) from exc
  expected = task_state_for_result(result)
  if actual is not expected:
    raise TaskResultSettlementError(
      "task completion final_state conflicts with embedded TaskResult "
      f"execution: expected {expected.value}, got {actual.value}"
    )
  if error is not None and result.execution.status != "failed":
    raise TaskResultSettlementError(
      "task completion error conflicts with non-failed TaskResult execution"
    )
  return actual


def validate_task_result_admission(
  result: TaskResult,
  admitted_task: AdmittedTask,
) -> None:
  """Require exact logical, physical, and authority provenance identity."""

  provenance = result.provenance
  if (
    result.logical_task != admitted_task.logical_task
    or result.attempt != admitted_task.attempt
    or provenance.admitted_task_digest
    != admitted_task.admitted_task_digest
    or provenance.model_bind_digest != admitted_task.model_bind_digest
    or provenance.capability_binding_digest
    != admitted_task.capability_binding_digest
    or provenance.tool_grant_digest != admitted_task.tool_grant_digest
  ):
    raise TaskResultSettlementError(
      "TaskResult does not settle the exact admitted task authority"
    )


def _durable_event_payload(event: dict[str, Any]) -> dict[str, Any]:
  return {
    key: value
    for key, value in event.items()
    if key != "_durable_seq"
  }


def _require_exact_duplicate_events(
  events: list[dict[str, Any]],
  *,
  task_id: str,
  event_type: str,
) -> dict[str, Any] | None:
  if not events:
    return None
  first = _durable_event_payload(events[0])
  if any(
    _durable_event_payload(candidate) != first
    for candidate in events[1:]
  ):
    raise TaskDurableEventConflictError(
      f"Task {task_id} has conflicting durable {event_type} events"
    )
  return events[0]


def _required_skill_lifecycle_identity(
  *,
  lifecycle: dict[str, Any],
  task_id: str,
) -> TopLevelSkillLifecycleMetadata:
  lifecycle_fields = {
    "schema_version",
    "skill_run_id",
    "skill",
    "scope",
    "ticker",
    "portfolio_id",
  }
  if set(lifecycle) != lifecycle_fields:
    raise RequiredSkillResultMarkerValidationError(
      f"Task {task_id} required skill lifecycle has an invalid v2 schema"
    )
  if (
    type(lifecycle.get("schema_version")) is not int
    or lifecycle["schema_version"] != 2
  ):
    raise RequiredSkillResultMarkerValidationError(
      f"Task {task_id} required skill lifecycle has an invalid schema_version"
    )
  skill_run_id = lifecycle.get("skill_run_id")
  skill = lifecycle.get("skill")
  scope = lifecycle.get("scope")
  ticker = lifecycle.get("ticker")
  portfolio_id = lifecycle.get("portfolio_id")
  if (
    not isinstance(skill_run_id, str)
    or not skill_run_id
    or skill_run_id != skill_run_id.strip()
    or len(skill_run_id) > 128
    or not isinstance(skill, str)
    or not skill
    or skill != skill.strip()
    or len(skill) > 256
    or type(scope) is not str
    or (
      ticker is not None
      and (
        not isinstance(ticker, str)
        or not ticker
        or ticker != ticker.strip()
        or len(ticker) > 64
      )
    )
    or (
      portfolio_id is not None
      and (
        not isinstance(portfolio_id, str)
        or not portfolio_id
        or portfolio_id != portfolio_id.strip()
        or len(portfolio_id) > 256
      )
    )
  ):
    raise RequiredSkillResultMarkerValidationError(
      f"Task {task_id} required skill lifecycle identity is invalid"
    )
  try:
    artifact_identity = SkillLifecycleArtifactIdentity(
      scope=scope,
      ticker=ticker,
      portfolio_id=portfolio_id,
    )
    validated = TopLevelSkillLifecycleMetadata(
      skill_run_id=skill_run_id,
      skill=skill,
      **artifact_identity.identity_fields(),
    )
  except ValueError as exc:
    raise RequiredSkillResultMarkerValidationError(
      f"Task {task_id} required skill lifecycle identity is invalid: "
      f"{exc}"
    ) from exc
  return validated


def validate_required_skill_result_marker(
  *,
  lifecycle: dict[str, Any],
  marker: dict[str, Any],
  task_id: str,
  registration_seq: Any,
  completion_seq: Any,
  marker_seq: Any,
) -> dict[str, Any]:
  """Validate one marker against the exact v2 lifecycle and durable bounds."""

  lifecycle_identity = _required_skill_lifecycle_identity(
    lifecycle=lifecycle,
    task_id=task_id,
  )
  if marker.get("type") != "skill_result_captured":
    raise RequiredSkillResultMarkerValidationError(
      f"Task {task_id} required skill result has an invalid event type"
    )
  expected_identity = {
    "task_id": task_id,
    **lifecycle_identity.identity_fields(),
  }
  for identity_field, expected in expected_identity.items():
    if (
      identity_field not in marker
      or marker[identity_field] != expected
      or type(marker[identity_field]) is not type(expected)
    ):
      raise RequiredSkillResultMarkerValidationError(
        f"Task {task_id} required skill result has a mismatched "
        f"{identity_field}"
      )
  missing_core_fields = (
    SKILL_RESULT_CORE_FIELDS - set(marker)
  )
  if missing_core_fields:
    raise RequiredSkillResultMarkerValidationError(
      f"Task {task_id} required skill result is missing core fields: "
      + ", ".join(sorted(missing_core_fields))
    )
  core_event = {
    field_name: marker[field_name]
    for field_name in SKILL_RESULT_CORE_FIELDS
  }
  try:
    canonical_core = lifecycle_identity.normalize_result_event(core_event)
  except RuntimeError as exc:
    raise RequiredSkillResultMarkerValidationError(
      f"Task {task_id} required skill result core is invalid: {exc}"
    ) from exc
  if not all(
    type(seq) is int and seq > 0
    for seq in (registration_seq, completion_seq, marker_seq)
  ):
    raise RequiredSkillResultMarkerValidationError(
      f"Task {task_id} required skill result has invalid durable bounds"
    )
  if not registration_seq < completion_seq < marker_seq:
    raise RequiredSkillResultMarkerValidationError(
      f"Task {task_id} required skill result falls outside task bounds"
    )
  return canonical_core


def resolve_required_skill_result_marker(
  *,
  lifecycle: dict[str, Any],
  task_id: str,
  registration_seq: Any,
  completion_seq: Any,
  marker_candidates: Iterable[tuple[Any, dict[str, Any]]],
) -> dict[str, Any] | None:
  """Select and validate the one lifecycle-global durable result marker."""

  lifecycle_identity = _required_skill_lifecycle_identity(
    lifecycle=lifecycle,
    task_id=task_id,
  )
  relevant_markers = [
    (marker_seq, marker)
    for marker_seq, marker in marker_candidates
    if (
      marker.get("type") == "skill_result_captured"
      and (
        marker.get("task_id") == task_id
        or marker.get("skill_run_id") == lifecycle_identity.skill_run_id
      )
    )
  ]
  if not relevant_markers:
    return None
  if len(relevant_markers) != 1:
    raise RequiredSkillResultMarkerValidationError(
      f"Task {task_id} has duplicate or conflicting required skill results"
    )
  marker_seq, marker = relevant_markers[0]
  canonical_core = validate_required_skill_result_marker(
    lifecycle=lifecycle,
    marker=marker,
    task_id=task_id,
    registration_seq=registration_seq,
    completion_seq=completion_seq,
    marker_seq=marker_seq,
  )
  canonical_marker = dict(marker)
  canonical_marker.update(canonical_core)
  return canonical_marker


@dataclass
class TaskProgress:
  tool_use_count: int = 0
  input_tokens: int = 0
  output_tokens: int = 0
  turn_count: int = 0
  last_tool_name: str | None = None
  last_activity_at: float = 0.0
  recent_tools: list[str] = field(default_factory=list)


@dataclass
class ParentMessage:
  message_id: str
  text: str
  sent_at: float
  task_id: str | None = None
  sent_seq: int | None = None


_WORKFLOW_TASK_CORRELATION_FIELDS = (
  "workflow_run_id",
  "plan_id",
  "phase_number",
  "revision",
  "node_id",
  "attempt_number",
  "item_key",
  "reserved_budget_usd",
)


def _task_registration_message_correlation(
  task_id: str,
  registration: dict[str, Any],
  event: dict[str, Any],
) -> bool:
  """Return whether one lifecycle fact matches the exact registration."""

  metadata = registration.get("metadata")
  capability_bind = registration.get("capability_bind")
  owner_runner_id = registration.get("owner_runner_id")
  owner_role = registration.get("owner_role")
  sub_agent_id = registration.get("sub_agent_id")
  call_index = registration.get("call_index")
  registered_task_type = registration.get("task_type")
  if (
    not isinstance(metadata, dict)
    or type(owner_runner_id) is not str
    or not owner_runner_id
    or type(owner_role) is not str
    or not owner_role
    or type(sub_agent_id) is not str
    or not sub_agent_id
    or type(call_index) is not int
    or call_index < 0
    or type(registered_task_type) is not str
    or not registered_task_type
    or not isinstance(capability_bind, dict)
    or not capability_bind
    or type(metadata.get("task_type")) is not str
    or not metadata["task_type"]
  ):
    return False
  for field_name, expected in (
    ("owner_runner_id", owner_runner_id),
    ("owner_role", owner_role),
    ("sub_agent_id", sub_agent_id),
    ("parent_turn_id", registration.get("parent_turn_id")),
    ("call_index", call_index),
    ("capability_bind", capability_bind),
  ):
    if metadata.get(field_name) != expected:
      return False
  expected_event = {
    "task_id": task_id,
    "owner_runner_id": owner_runner_id,
    "owner_role": owner_role,
    "sub_agent_id": sub_agent_id,
    "parent_turn_id": registration.get("parent_turn_id"),
    "call_index": call_index,
    "task_type": metadata["task_type"],
    "capability_bind": capability_bind,
  }
  if registration.get("original_task_id") is not None:
    expected_event["original_task_id"] = registration["original_task_id"]
  for field_name in _WORKFLOW_TASK_CORRELATION_FIELDS:
    if field_name in registration:
      expected_event[field_name] = registration[field_name]
  return all(
    event.get(field_name) == expected
    and type(event.get(field_name)) is type(expected)
    for field_name, expected in expected_event.items()
  )


def _rehydrate_parent_messages(
  task_id: str,
  *,
  registrations: list[dict[str, Any]],
  completions: list[dict[str, Any]],
  events: list[dict[str, Any]],
) -> dict[str, ParentMessage]:
  """Validate the durable acceptance set for one reconstructed task."""

  if not events:
    return {}
  if len(registrations) != 1 or len(completions) > 1:
    raise ValueError(
      f"Task {task_id} has ambiguous durable lifecycle bounds"
    )
  registration = registrations[0]
  completion = completions[0] if completions else None
  registration_seq = registration.get("_durable_seq")
  completion_seq = (
    completion.get("_durable_seq")
    if completion is not None
    else None
  )
  if (
    type(registration_seq) is not int
    or registration_seq <= 0
    or (
      completion is not None
      and (
        type(completion_seq) is not int
        or completion_seq <= registration_seq
        or not _task_registration_message_correlation(
          task_id,
          registration,
          completion,
        )
      )
    )
  ):
    raise ValueError(
      f"Task {task_id} has invalid durable lifecycle bounds"
    )

  accepted: dict[str, ParentMessage] = {}
  for event in events:
    message_id = event.get("message_id")
    message = event.get("message")
    durable_seq = event.get("_durable_seq")
    sent_at = event.get("sent_at")
    if (
      event.get("type") != "parent_message_sent"
      or event.get("task_id") != task_id
      or type(message_id) is not str
      or not message_id
      or message_id != message_id.strip()
      or len(message_id) > 512
      or len(message_id.encode("utf-8")) > 512
      or type(message) is not str
      or not message
      or message != message.strip()
      or len(message) > 12_000
      or len(message.encode("utf-8")) > 32 * 1024
      or type(durable_seq) is not int
      or durable_seq <= registration_seq
      or (
        completion_seq is not None
        and durable_seq >= completion_seq
      )
      or isinstance(sent_at, bool)
      or not isinstance(sent_at, (int, float))
      or not _task_registration_message_correlation(
        task_id,
        registration,
        event,
      )
    ):
      raise ValueError(
        f"Task {task_id} has an invalid durable parent-message acceptance"
      )
    if message_id in accepted:
      raise ValueError(
        f"Task {task_id} has duplicate durable parent-message identity "
        f"{message_id!r}"
      )
    accepted[message_id] = ParentMessage(
      message_id=message_id,
      text=message,
      sent_at=float(sent_at),
      task_id=task_id,
      sent_seq=durable_seq,
    )
  return accepted


def format_parent_messages_for_model(parent_messages: list[ParentMessage]) -> str:
  # Neutral framing on purpose: a user-turn message that self-asserts elevated
  # authority ("authenticated... controlling parent session... follow as user
  # intent") pattern-matches prompt injection and newer models refuse to honor
  # it (ACUI-3). The user-turn channel of an autonomous run is already the
  # operator channel; label provenance, don't claim trust.
  lines = ["Operator update for this task:"]
  lines.extend(
    f"- id={message.message_id}: {message.text}"
    for message in parent_messages
  )
  return "\n".join(lines)


@dataclass
class TaskEntry:
  task_id: str
  task_type: str
  agent_name: str | None = None
  state: TaskState = TaskState.PENDING
  asyncio_task: asyncio.Task[Any] | None = None
  started_at: float = field(default_factory=time.time)
  completed_at: float | None = None
  result: dict[str, Any] | None = None
  error: dict[str, Any] | None = None
  progress: TaskProgress = field(default_factory=TaskProgress)
  metadata: dict[str, Any] = field(default_factory=dict)
  capability_bind_receipt: dict[str, str] | None = None
  admitted_task: AdmittedTask | None = None
  task_result: TaskResult | None = None
  parent_result_policy: ParentResultPolicy | None = None
  completion_envelope: AgentCompletionEnvelope | None = None
  message_inbox: asyncio.Queue[ParentMessage] = field(default_factory=asyncio.Queue)
  accepted_parent_messages: dict[str, ParentMessage] = field(default_factory=dict)
  delivered_messages: set[str] = field(default_factory=set)
  original_task_id: str | None = None
  reconstructed_from_log: bool = False
  termination_intent: TerminationIntent | None = None
  registration_persistence_state: CompletionPersistenceState = "not_started"
  registration_persistence_error: str | None = None
  initialization_task: asyncio.Task[Any] | None = field(
    default=None,
    repr=False,
    compare=False,
  )
  start_hook_invoked: bool = False
  pending_final_state: TaskState | None = None
  pending_result: dict[str, Any] | None = None
  pending_error: dict[str, Any] | None = None
  completion_finalizer_detached: bool = False
  completion_callback: Callable[["TaskEntry"], Any] | None = field(
    default=None,
    repr=False,
    compare=False,
  )
  completion_callback_invoked: bool = False
  required_skill_result_event_factory: Callable[
    [
      "TaskEntry",
      dict[str, Any] | None,
      dict[str, Any] | None,
    ],
    dict[str, Any],
  ] | None = field(
    default=None,
    repr=False,
    compare=False,
  )
  required_skill_result_projector: Callable[
    [dict[str, Any]],
    Any,
  ] | None = field(
    default=None,
    repr=False,
    compare=False,
  )
  required_skill_result_task: asyncio.Task[bool] | None = field(
    default=None,
    repr=False,
    compare=False,
  )
  required_skill_result_settled: bool = False
  required_skill_result_projected: bool = False
  completion_persistence_state: CompletionPersistenceState = "not_started"
  completion_persistence_error: str | None = None
  notification_delivery_state: NotificationDeliveryState = "not_queued"
  notification_generation: int = 0
  #: The parent exercised the delivered result handle's read grant at least
  #: once (CUR-E2E-08). In-memory only. After a registry rebuild the
  #: delivery state itself resets to "not_queued", so the natural-finish
  #: reminder does not re-fire in a resumed process at all — the protection
  #: is per-live-run and fail-open, never at-least-once across restarts.
  result_content_read: bool = False
  finalization_lock: asyncio.Lock = field(
    default_factory=asyncio.Lock,
    repr=False,
    compare=False,
  )

  @property
  def completed(self) -> bool:
    return self.state in _TERMINAL_STATES


@dataclass
class TaskNotification:
  task_id: str
  agent_name: str | None
  event: str
  summary: str
  timestamp: float
  payload: dict[str, Any]
  notification_generation: int | None = None
  omission_reason_override: str | None = field(
    default=None,
    repr=False,
    compare=False,
  )
  omission_guidance_override: str | None = field(
    default=None,
    repr=False,
    compare=False,
  )
  _payload_json: str | None = field(init=False, repr=False, compare=False)
  _payload_xml_text: str | None = field(
    init=False,
    repr=False,
    compare=False,
  )
  _payload_omission_reason: str | None = field(
    init=False,
    repr=False,
    compare=False,
  )
  _queue_token: int | None = field(
    default=None,
    init=False,
    repr=False,
    compare=False,
  )

  def __post_init__(self) -> None:
    self._payload_xml_text = None
    if self.omission_reason_override is not None:
      self._payload_json = None
      self._payload_omission_reason = self.omission_reason_override
      return
    preflight_reason = _notification_payload_preflight_reason(self.payload)
    if preflight_reason is not None:
      self._payload_json = None
      self._payload_omission_reason = preflight_reason
      return
    try:
      payload_json = json.dumps(
        self.payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
      )
    except (TypeError, ValueError, RecursionError):
      self._payload_json = None
      self._payload_omission_reason = "non_json_payload"
    else:
      self._payload_json = payload_json
      self._payload_omission_reason = None
    if self._payload_json is not None:
      try:
        payload_size = len(self._payload_json.encode("utf-8"))
      except UnicodeEncodeError:
        self._payload_json = None
        self._payload_omission_reason = "non_utf8_payload"
      else:
        if payload_size > TASK_NOTIFICATION_INLINE_PAYLOAD_MAX_BYTES:
          self._payload_json = None
          self._payload_omission_reason = "payload_too_large"
        else:
          payload_xml_text = html.escape(self._payload_json)
          if (
            len(payload_xml_text.encode("utf-8"))
            > TASK_NOTIFICATION_INLINE_PAYLOAD_MAX_BYTES
          ):
            self._payload_json = None
            self._payload_omission_reason = "payload_too_large"
          else:
            self._payload_xml_text = payload_xml_text

  def omission_marker(
    self,
    reason: str,
    *,
    task_id: str | None = None,
    summary: str | None = None,
    guidance: str | None = None,
  ) -> TaskNotification:
    """Return a bounded notification that directs explicit result retrieval."""
    return TaskNotification(
      task_id=task_id or self.task_id,
      agent_name=self.agent_name,
      event=self.event,
      summary=(self.summary if summary is None else summary)[:2000],
      timestamp=self.timestamp,
      payload={},
      notification_generation=self.notification_generation,
      omission_reason_override=reason,
      omission_guidance_override=guidance,
    )

  def inline_payload(self) -> tuple[str | None, str | None]:
    """Return the canonical completion-time payload snapshot, when bounded."""
    return self._payload_json, self._payload_omission_reason

  def format_xml(self) -> str:
    """Render as <task-notification> XML block."""
    safe_summary = html.escape(self.summary[:2000]) if self.summary else ""
    safe_task_id = html.escape(self.task_id, quote=True)
    parts = [f'<task-notification task_id="{safe_task_id}">']
    parts.append(f"  <status>{self.event}</status>")
    if self.agent_name:
      parts.append(f"  <agent>{html.escape(self.agent_name)}</agent>")
    result_kind = self.payload.get("kind")
    if result_kind in {"narrative", "report", "unstructured"}:
      parts.append(f"  <result-kind>{result_kind}</result-kind>")
      if result_kind == "unstructured":
        reason = self.payload.get("reason")
        if isinstance(reason, str) and reason:
          parts.append(f"  <reason>{html.escape(reason)}</reason>")
    payload_json, omission_reason = self.inline_payload()
    if payload_json is not None:
      parts.append(
        "  <result encoding=\"json\">"
        f"{self._payload_xml_text or ''}"
        "</result>"
      )
    else:
      omission_guidance = (
        self.omission_guidance_override
        or (
          "Retrieve this explicitly with get_background_result using task_id="
          f"{self.task_id}."
        )
      )
      parts.append(
        "  <result-omitted "
        f'reason="{html.escape(omission_reason or "unavailable", quote=True)}">'
        f"{html.escape(omission_guidance)}"
        "</result-omitted>"
      )
    if safe_summary:
      parts.append(f"  <summary>{safe_summary}</summary>")
    parts.append("</task-notification>")
    return "\n".join(parts)


@dataclass
class CoordinatorConfig:
  enabled: bool = False
  preamble: str | None = None
  worker_excluded_tools: set[str] | None = None
  auto_notify: bool = True
  max_workers: int = 3


COORDINATOR_DEFAULT_PREAMBLE = """You are operating in coordinator mode. Delegate tasks to workers:
- Use run_agent(background=true) to spawn workers
- Use send_message to guide running workers
- Synthesize worker results — never delegate understanding
- Let automatic completion notifications drive follow-up; do not poll running workers
- Use get_background_result only for a historical or explicitly omitted result,
  or one explicit blocking wait when automatic notifications are disabled
Worker model and effort are selected by authenticated server policy before launch."""


class TaskLifecycleListener(Protocol):
  def on_transition(self, entry: TaskEntry, old_state: TaskState, new_state: TaskState) -> None:
    ...


_TERMINAL_STATES = {TaskState.COMPLETED, TaskState.FAILED, TaskState.KILLED, TaskState.INTERRUPTED}


class NotificationQueue:
  def __init__(self, max_pending: int = 20):
    if max_pending <= 0:
      raise ValueError("max_pending must be positive")
    self._queue: list[TaskNotification] = []
    self._max_pending = max_pending
    self._overflow_marker: TaskNotification | None = None
    self._deferred_overflow_marker: TaskNotification | None = None
    self._deferred_overflow_count = 0
    self._deferred_overflow_sample_task_ids: list[str] = []
    self._next_queue_token = 0
    self._available = asyncio.Event()

  def _enqueue(self, notification: TaskNotification) -> None:
    if notification._queue_token is None:
      notification._queue_token = self._next_queue_token
      self._next_queue_token += 1
    self._queue.append(notification)

  @staticmethod
  def _overflow_summary(
    count: int,
    sample_task_ids: list[str],
  ) -> str:
    samples = ", ".join(sample_task_ids)
    return (
      f"{count} completed task result(s) omitted because notification "
      f"capacity was exhausted. Sample task IDs: {samples}."
    )

  @staticmethod
  def _overflow_guidance() -> str:
    return (
      "Retrieve omitted results explicitly with get_background_result using "
      "the task IDs returned by run_agent."
    )

  def _new_overflow_marker(
    self,
    notification: TaskNotification,
    *,
    summary: str,
  ) -> TaskNotification:
    return TaskNotification(
      task_id="multiple",
      agent_name=None,
      event="results_omitted",
      summary=summary,
      timestamp=notification.timestamp,
      payload={},
      notification_generation=notification.notification_generation,
      omission_reason_override="queue_capacity",
      omission_guidance_override=self._overflow_guidance(),
    )

  def _record_overflow(self, notification: TaskNotification) -> None:
    if self._overflow_marker is None:
      self._overflow_marker = self._new_overflow_marker(
        notification,
        summary=self._overflow_summary(1, [notification.task_id]),
      )
      self._enqueue(self._overflow_marker)
      return

    self._deferred_overflow_count += 1
    if (
      len(self._deferred_overflow_sample_task_ids)
      < TASK_NOTIFICATION_OVERFLOW_SAMPLE_TASK_IDS
    ):
      self._deferred_overflow_sample_task_ids.append(notification.task_id)
    if self._deferred_overflow_marker is None:
      self._deferred_overflow_marker = self._new_overflow_marker(
        notification,
        summary="",
      )
    self._deferred_overflow_marker.summary = self._overflow_summary(
      self._deferred_overflow_count,
      self._deferred_overflow_sample_task_ids,
    )

  def push(self, notification: TaskNotification) -> bool:
    bounded_notification = self._bounded_notification(notification)
    return self._push_bounded(
      notification=notification,
      bounded_notification=bounded_notification,
    )

  @staticmethod
  def _bounded_notification(
    notification: TaskNotification,
  ) -> TaskNotification:
    payload_json, omission_reason = notification.inline_payload()
    return (
      notification
      if payload_json is not None
      else notification.omission_marker(
        omission_reason or "unavailable",
      )
    )

  def _push_bounded(
    self,
    *,
    notification: TaskNotification,
    bounded_notification: TaskNotification,
  ) -> bool:
    if self._deferred_overflow_marker is not None:
      self._record_overflow(notification)
      self._available.set()
      return False
    primary_limit = (
      self._max_pending
      if self._overflow_marker is not None
      else self._max_pending - 1
    )
    if len(self._queue) < primary_limit:
      self._enqueue(bounded_notification)
      self._available.set()
      return True

    self._record_overflow(notification)
    self._available.set()
    return False

  def push_or_replace_pending(
    self,
    notification: TaskNotification,
  ) -> bool:
    """Push progress, replacing the same task's pending progress in place.

    Only ``plan_progress`` is replaceable. Approval and terminal events
    retain ordinary FIFO/overflow behavior.
    """
    if notification.event != "plan_progress":
      return self.push(notification)

    bounded_notification = self._bounded_notification(notification)
    for index, pending in enumerate(self._queue):
      if (
        pending.task_id == notification.task_id
        and pending.event == "plan_progress"
      ):
        bounded_notification._queue_token = pending._queue_token
        self._queue[index] = bounded_notification
        self._available.set()
        return True
    return self._push_bounded(
      notification=notification,
      bounded_notification=bounded_notification,
    )

  def _rotate_overflow_marker(
    self,
    removed: list[TaskNotification],
  ) -> None:
    """Promote the deferred overflow marker once the live one leaves."""
    if self._overflow_marker is None:
      return
    if not any(
      notification is self._overflow_marker
      for notification in removed
    ):
      return
    self._overflow_marker = self._deferred_overflow_marker
    self._deferred_overflow_marker = None
    self._deferred_overflow_count = 0
    self._deferred_overflow_sample_task_ids = []
    if self._overflow_marker is not None:
      self._enqueue(self._overflow_marker)

  def drain(self, max_count: int = 5) -> list[TaskNotification]:
    """Remove and return up to max_count notifications from the front."""
    result = self._queue[:max_count]
    self._queue = self._queue[max_count:]
    self._rotate_overflow_marker(result)
    if not self._queue:
      self._available.clear()
    return result

  def drain_delivered(
    self,
    delivered: Sequence[TaskNotification],
  ) -> list[TaskNotification]:
    """Remove exactly the delivered notification objects from the queue.

    Keyed by **object identity**, never by ``_queue_token``:
    ``push_or_replace_pending`` reuses a pending entry's token for a
    different, never-rendered payload, so a token-keyed drain would
    silently discard a replacement the model has not seen. Overflow-marker
    rotation matches ``drain``.
    """
    if not delivered:
      return []
    removed: list[TaskNotification] = []
    remaining: list[TaskNotification] = []
    for notification in self._queue:
      if any(notification is candidate for candidate in delivered):
        removed.append(notification)
      else:
        remaining.append(notification)
    self._queue = remaining
    self._rotate_overflow_marker(removed)
    if not self._queue:
      self._available.clear()
    return removed

  def peek(self, max_count: int | None = None) -> list[TaskNotification]:
    """Non-destructive peek at front of queue."""
    if max_count is None:
      return list(self._queue)
    return list(self._queue[:max_count])

  @property
  def pending_count(self) -> int:
    return len(self._queue)

  async def wait_until_available(self) -> None:
    """Wait until at least one notification is queued."""
    await self._available.wait()


def make_progress_tracker(entry: TaskEntry) -> Callable[[dict[str, Any], str], None]:
  """Return an on_event callback that updates entry.progress in-place."""

  def _track(event: dict[str, Any], session_id: str) -> None:
    _ = session_id
    event_type = event.get("type")
    if event_type == "tool_call_start":
      entry.progress.tool_use_count += 1
      name = str(event.get("tool_name", ""))
      entry.progress.last_tool_name = name
      entry.progress.last_activity_at = time.time()
      entry.progress.recent_tools.append(name)
      if len(entry.progress.recent_tools) > 10:
        entry.progress.recent_tools.pop(0)
    elif event_type == "turn_complete":
      entry.progress.turn_count += 1
      usage = event.get("usage", {})
      if not isinstance(usage, dict):
        usage = {}
      entry.progress.input_tokens += int(usage.get("input_tokens", 0) or 0)
      entry.progress.output_tokens += int(usage.get("output_tokens", 0) or 0)
      entry.progress.last_activity_at = time.time()

  return _track


class TaskRegistry:
  def __init__(self, *, max_inflight: int = 10, max_retained: int = 50, id_prefix: str = "bg") -> None:
    self._tasks: dict[str, TaskEntry] = {}
    self._seq = 0
    self._max_inflight = max_inflight
    self._max_retained = max_retained
    self._id_prefix = id_prefix
    self._listeners: list[TaskLifecycleListener] = []
    self._durable_skip_warned: set[str] = set()

  def register(
    self,
    task_type: str,
    agent_name: str | None = None,
    *,
    task_id: str | None = None,
    original_task_id: str | None = None,
    **metadata_kwargs: Any,
  ) -> TaskEntry:
    if not self.notification_retrieval_capacity_available():
      raise NotificationRetrievalCapacityError(
        "Unretrieved omitted background results exhausted bounded retention"
      )
    if task_id is None:
      task_id = f"{self._id_prefix}_{self._seq}"
      self._seq += 1
    elif task_id in self._tasks:
      raise ValueError(f"Task already registered: {task_id}")
    entry = TaskEntry(
      task_id=task_id,
      task_type=task_type,
      agent_name=agent_name,
      original_task_id=original_task_id,
      metadata=dict(metadata_kwargs),
    )
    self._tasks[task_id] = entry
    self._auto_evict_completed()
    return entry

  def claim_resume_successor(
    self,
    task_type: str,
    *,
    task_id: str,
    original_task_id: str,
    agent_name: str | None = None,
    **metadata_kwargs: Any,
  ) -> tuple[TaskEntry, bool]:
    """Atomically claim one deterministic resume successor in this registry.

    The method is synchronous by design: concurrent asyncio callers cannot
    interleave between the identity check and insertion.
    """

    if not task_id:
      raise ValueError("resume successor task_id must be non-empty")
    if not original_task_id:
      raise ValueError(
        "resume successor original_task_id must be non-empty"
      )
    existing = self._tasks.get(task_id)
    if existing is not None:
      if existing.original_task_id == original_task_id:
        return existing, False
      raise ResumeSuccessorConflictError(
        task_id=task_id,
        original_task_id=original_task_id,
        existing=existing,
      )
    return (
      self.register(
        task_type,
        agent_name=agent_name,
        task_id=task_id,
        original_task_id=original_task_id,
        **metadata_kwargs,
      ),
      True,
    )

  def admit(
    self,
    task_type: str,
    *,
    agent_name: str | None = None,
    task_id: str | None = None,
    original_task_id: str | None = None,
    reject_over_capacity: Callable[[int, int], dict[str, Any] | None],
    reject_retrieval_backpressure: Callable[[int, int], dict[str, Any]],
    **metadata_kwargs: Any,
  ) -> tuple[TaskEntry | None, dict[str, Any] | None]:
    """Decide background capacity and reserve the slot in one critical section.

    Synchronous by contract (A-M8 / T3-I03): no ``await`` may be introduced
    between the capacity verdict and the reservation, or N concurrent
    callers can all read the same stale ``admission_count`` and then all
    register afterwards. ``_max_inflight`` is the single ceiling (T3-I04);
    the RUNNING transition only logs an invariant violation.

    Returns ``(entry, None)`` when the slot is taken, or
    ``(None, rejection)`` when it is refused. A rejection means no slot was
    reserved and therefore no durable ``task_registered`` may be appended.
    """

    admission_count = self.admission_count
    reserved = self._tasks.get(task_id) if task_id else None
    if reserved is not None and reserved.state == TaskState.PENDING:
      admission_count -= 1
    limit_error = reject_over_capacity(admission_count, self._max_inflight)
    if limit_error is not None:
      return None, limit_error
    if not self.notification_retrieval_capacity_available(
      admission_count=admission_count,
    ):
      return None, reject_retrieval_backpressure(
        self.pending_notification_retrieval_count,
        self.notification_retrieval_retention_limit,
      )
    if original_task_id is not None:
      if not task_id:
        raise ValueError(
          "resume admission requires a deterministic successor task_id"
        )
      try:
        entry, _created = self.claim_resume_successor(
          task_type,
          agent_name=agent_name,
          task_id=task_id,
          original_task_id=original_task_id,
          **metadata_kwargs,
        )
      except ResumeSuccessorConflictError as exc:
        return None, {
          "code": "resume_successor_conflict",
          "message": str(exc),
        }
      return entry, None
    if reserved is not None and reserved.state == TaskState.PENDING:
      return reserved, None
    return (
      self.register(
        task_type,
        agent_name=agent_name,
        task_id=task_id,
        **metadata_kwargs,
      ),
      None,
    )

  def discard_unstarted(
    self,
    task_id: str,
    *,
    expected: TaskEntry,
  ) -> bool:
    """Remove one definitively unregistered task without disturbing replacements."""

    current = self._tasks.get(task_id)
    if (
      current is not expected
      or current.state != TaskState.PENDING
      or current.asyncio_task is not None
    ):
      return False
    self._tasks.pop(task_id, None)
    return True

  def transition(
    self,
    task_id: str,
    new_state: TaskState,
    *,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
  ) -> TaskEntry:
    entry = self._tasks.get(task_id)
    if entry is None:
      raise KeyError(f"Unknown task: {task_id}")
    if entry.state in _TERMINAL_STATES:
      return entry
    if (
      new_state == TaskState.RUNNING
      and entry.state != TaskState.RUNNING
      and self.inflight_count >= self._max_inflight
    ):
      # T3-I04 / D-A8-5: capacity is owned by ``admit`` alone. Reaching this
      # branch means an admitted task escaped the single ceiling, which is a
      # code defect, not a runtime condition to enforce against. Record the
      # invariant violation (this log line is the CUR-E2E-03 recurrence
      # detector) and proceed — refusing here strands the task with a durable
      # ``task_registered`` and no terminal.
      log.error(
        "task registry invariant violation: RUNNING transition exceeded the "
        "inflight ceiling (task_id=%s current_state=%s inflight_count=%d "
        "max_inflight=%d); admission is owned by TaskRegistry.admit",
        task_id,
        entry.state.value,
        self.inflight_count,
        self._max_inflight,
      )

    old_state = entry.state
    entry.state = new_state
    if result is not None:
      entry.result = dict(result)
    if error is not None:
      entry.error = dict(error)
    if new_state in _TERMINAL_STATES:
      entry.completed_at = time.time()
      entry.notification_generation += 1
    for listener in self._listeners:
      listener.on_transition(entry, old_state, new_state)
    return entry

  def finalize_interrupted(
    self,
    task_id: str,
    new_state: TaskState,
    *,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
  ) -> TaskEntry:
    """Finalize one replayed interrupted task after resume is abandoned."""

    entry = self._tasks.get(task_id)
    if entry is None:
      raise KeyError(f"Unknown task: {task_id}")
    if entry.state != TaskState.INTERRUPTED:
      return entry
    if new_state not in {
      TaskState.COMPLETED,
      TaskState.FAILED,
      TaskState.KILLED,
    }:
      raise ValueError(
        "interrupted tasks may only finalize to completed, failed, or killed"
      )

    old_state = entry.state
    entry.state = new_state
    entry.result = dict(result) if result is not None else None
    entry.error = dict(error) if error is not None else None
    entry.completed_at = time.time()
    entry.notification_generation += 1
    for listener in self._listeners:
      listener.on_transition(entry, old_state, new_state)
    return entry

  def adopt_interrupted(self, entry: TaskEntry) -> TaskEntry:
    """Adopt one lazily reconstructed interrupted task into this registry."""

    existing = self._tasks.get(entry.task_id)
    if existing is not None:
      return existing
    if (
      entry.state != TaskState.INTERRUPTED
      or not entry.reconstructed_from_log
    ):
      raise ValueError(
        "only reconstructed interrupted tasks may be adopted"
      )
    self._tasks[entry.task_id] = entry
    suffix = self._numeric_suffix(entry.task_id)
    if suffix is not None:
      self._seq = max(self._seq, suffix + 1)
    return entry

  def get(self, task_id: str) -> TaskEntry | None:
    return self._tasks.get(task_id)

  def list_tasks(self, *, state: TaskState | None = None) -> list[TaskEntry]:
    tasks = self._tasks.values()
    if state is not None:
      tasks = (entry for entry in tasks if entry.state == state)
    return sorted(tasks, key=lambda entry: entry.started_at)

  def kill(
    self,
    task_id: str,
    *,
    termination_intent: TerminationIntent = "killed",
  ) -> bool:
    entry = self._tasks.get(task_id)
    if entry is None or entry.state in _TERMINAL_STATES:
      return False
    if termination_intent not in {"cancelled", "killed"}:
      raise ValueError(f"Unsupported termination intent: {termination_intent}")

    prior_intent = entry.termination_intent
    if prior_intent == "cancelled" or prior_intent == termination_intent:
      return False
    entry.termination_intent = termination_intent
    if entry.asyncio_task is not None:
      entry.asyncio_task.cancel()
    return True

  def evict_completed(self, max_age_seconds: float = 300) -> int:
    now = time.time()
    evicted = 0
    candidates = [
      entry
      for entry in self.list_tasks()
      if (
        entry.state in _TERMINAL_STATES
        and entry.notification_delivery_state
        not in _NOTIFICATION_RETRIEVAL_REQUIRED_STATES
        and entry.completed_at is not None
        and now - entry.completed_at >= max_age_seconds
      )
    ]
    for entry in candidates:
      if self._tasks.pop(entry.task_id, None) is not None:
        evicted += 1
    return evicted

  def add_listener(self, listener: TaskLifecycleListener) -> None:
    self._listeners.append(listener)

  def _warn_skipped_durable_task(self, task_id: str, reason: str) -> None:
    """Log a durable-task rebuild skip once per (task, reason), not per pass.

    Rebuild and durable lookup run repeatedly; an unreadable record is a
    standing condition, not a new event each pass (same rationale as the
    autonomous-manifest warn-once skip).
    """

    key = f"{task_id}\0{reason}"
    if key in self._durable_skip_warned:
      return
    self._durable_skip_warned.add(key)
    log.warning(
      "Skipping durable task %s during registry rebuild: %s",
      task_id,
      reason,
    )

  def load_from_events(self, events: list[dict[str, Any]]) -> None:
    """Rebuild registry from durable task events without firing listeners.

    Tasks that were killed in a prior process rebuild as ``interrupted`` if no
    durable completion event exists; the log cannot distinguish those from
    tasks that were running when the process crashed.

    Only ``task_registered`` records at the current ``EVENT_SCHEMA_VERSION``
    that carry a ``capability_bind`` receipt are rebuilt. Older records are
    drained, not migrated (model-selection-authority plan section 6): they are
    loudly skipped here instead of being reconstructed as bind-less entries
    that would fail only at resume. Their numeric task-ID suffixes still
    advance the registry sequence so new registrations never collide with a
    skipped durable identity.
    """
    grouped: dict[str, dict[str, Any]] = {}
    global_skill_results: list[dict[str, Any]] = []
    max_suffix: int | None = None
    for event in events:
      event_type = str(event.get("type") or "")
      if event_type == "skill_result_captured":
        global_skill_results.append(dict(event))
      task_id = str(event.get("task_id") or "")
      if not task_id:
        continue
      bucket = grouped.setdefault(
        task_id,
        {
          "registrations": [],
          "completions": [],
          "agent_completions": [],
          "messages": [],
        },
      )
      if event_type == "task_registered":
        bucket["registrations"].append(dict(event))
      elif event_type == "task_completed":
        bucket["completions"].append(dict(event))
      elif event_type == "agent_completion":
        bucket["agent_completions"].append(dict(event))
      elif event_type == "parent_message_sent":
        bucket["messages"].append(dict(event))

    for task_id, bucket in grouped.items():
      registrations = list(bucket.get("registrations", []))
      if not registrations:
        continue
      registered = _require_exact_duplicate_events(
        registrations,
        task_id=task_id,
        event_type="task_registered",
      )
      if registered is None:  # pragma: no cover - guarded above
        continue
      suffix = self._numeric_suffix(task_id)
      if suffix is not None:
        max_suffix = suffix if max_suffix is None else max(max_suffix, suffix)
      raw_event_schema_version = registered.get("event_schema_version")
      if not is_current_event_schema_version(raw_event_schema_version):
        self._warn_skipped_durable_task(
          task_id,
          "task_registered record has unsupported event_schema_version "
          f"{raw_event_schema_version!r} (expected {EVENT_SCHEMA_VERSION}); "
          "pre-cutover durable tasks are drained, not rebuilt",
        )
        continue
      if not isinstance(registered.get("capability_bind"), dict):
        self._warn_skipped_durable_task(
          task_id,
          "task_registered record carries no capability_bind receipt; "
          "refusing to rebuild a task that could not reauthorize its "
          "exact binding",
        )
        continue
      completions = list(bucket.get("completions", []))
      completed = _require_exact_duplicate_events(
        completions,
        task_id=task_id,
        event_type="task_completed",
      )
      metadata = dict(registered.get("metadata") or {})
      for key in (
        "owner_runner_id",
        "owner_role",
        "sub_agent_id",
        "parent_turn_id",
        "call_index",
        "task_type",
        "capability_bind",
        "parent_session_id",
        "original_task_id",
      ):
        if key in registered:
          metadata[key] = registered.get(key)
      metadata["parent_messages"] = list(bucket.get("messages") or [])
      accepted_parent_messages = _rehydrate_parent_messages(
        task_id,
        registrations=registrations,
        completions=completions,
        events=list(bucket.get("messages") or []),
      )

      state = TaskState.INTERRUPTED
      result = None
      task_result: TaskResult | None = None
      completion_envelope: AgentCompletionEnvelope | None = None
      error = None
      completed_at = registered.get("started_at")
      if isinstance(completed, dict):
        raw_final_state = completed.get("final_state")
        try:
          state = TaskState(str(raw_final_state))
        except ValueError:
          state = TaskState.FAILED
        result = completed.get("result")
        error = completed.get("error")
        if isinstance(result, dict) and result.get("schema_version") == "2.0":
          try:
            task_result = TaskResult.model_validate(result)
          except (TypeError, ValueError) as exc:
            state = TaskState.FAILED
            error = {
              "code": "invalid_task_result",
              "message": f"Persisted canonical TaskResult is invalid: {exc}",
            }
          else:
            if task_result.attempt.physical_task_id != task_id:
              task_result = None
              state = TaskState.FAILED
              error = {
                "code": "invalid_task_result",
                "message": (
                  "Persisted TaskResult physical task identity does not "
                  f"match {task_id}"
                ),
              }
            else:
              try:
                state = validate_task_result_settlement(
                  task_result,
                  final_state=raw_final_state,
                  error=(error if isinstance(error, dict) else None),
                )
              except TaskResultSettlementError as exc:
                task_result = None
                state = TaskState.FAILED
                error = {
                  "code": "task_result_settlement_mismatch",
                  "message": str(exc),
                }
        if error is not None and not isinstance(error, dict):
          raw_error = error
          raw_error_preview = repr(raw_error)[:128]
          state = TaskState.FAILED
          error = {
            "code": "invalid_child_error",
            "message": (
              "Persisted child error must be an object or null; "
              f"got {type(raw_error).__name__}: {raw_error_preview}"
            ),
          }
        elif (
          str(registered.get("task_type") or "background_agent")
          in {"background_agent", "workflow_node"}
          and task_result is None
          and not (
            isinstance(error, dict)
            and error.get("code") in {
              "invalid_task_result",
              "task_result_settlement_mismatch",
            }
          )
        ):
          state = TaskState.FAILED
          error = {
            "code": "invalid_task_result",
            "message": "Agent task completion requires canonical TaskResult",
          }
        completed_at = completed.get("completed_at", completed_at)

      agent_completions = list(bucket.get("agent_completions", []))
      if agent_completions:
        try:
          typed_completions = [
            event_from_dict(candidate)
            for candidate in agent_completions
          ]
        except (KeyError, TypeError, ValueError) as exc:
          state = TaskState.FAILED
          error = {
            "code": "invalid_agent_completion",
            "message": f"Persisted AgentCompletionEvent is invalid: {exc}",
          }
        else:
          if any(
            not isinstance(candidate, AgentCompletionEvent)
            for candidate in typed_completions
          ):
            state = TaskState.FAILED
            error = {
              "code": "invalid_agent_completion",
              "message": "Persisted completion has the wrong typed event",
            }
          else:
            first_completion = typed_completions[0]
            if any(
              candidate.event_id != first_completion.event_id
              or candidate.fingerprint != first_completion.fingerprint
              or candidate.envelope != first_completion.envelope
              for candidate in typed_completions[1:]
            ):
              state = TaskState.FAILED
              error = {
                "code": "agent_completion_conflict",
                "message": (
                  "Durable agent completion identity was reused with "
                  "conflicting canonical payload"
                ),
              }
            else:
              completion_envelope = first_completion.envelope
          if (
            completion_envelope is not None
            and task_result is not None
            and completion_envelope.task_result_ref
            != TaskResultRef.from_result(task_result)
          ):
            state = TaskState.FAILED
            error = {
              "code": "agent_completion_result_mismatch",
              "message": (
                "Durable agent completion does not reference the exact "
                "canonical TaskResult"
              ),
            }
      termination_intent = None
      if state == TaskState.KILLED and task_result is not None:
        terminal_reason = str(
          task_result.execution.terminal_reason or ""
        ).partition(":")[0]
        if terminal_reason in {"cancelled", "killed"}:
          termination_intent = terminal_reason

      required_skill_result_settled = False
      lifecycle = metadata.get("required_skill_lifecycle")
      if (
        "required_skill_lifecycle" in metadata
        and isinstance(completed, dict)
      ):
        try:
          if not isinstance(lifecycle, dict):
            raise RequiredSkillResultMarkerValidationError(
              f"Task {task_id} required skill lifecycle is not an object"
            )
          marker = resolve_required_skill_result_marker(
            lifecycle=lifecycle,
            task_id=task_id,
            registration_seq=registered.get("_durable_seq"),
            completion_seq=completed.get("_durable_seq"),
            marker_candidates=(
              (
                skill_result.get("_durable_seq"),
                skill_result,
              )
              for skill_result in global_skill_results
            ),
          )
        except RequiredSkillResultMarkerValidationError as exc:
          metadata["_required_skill_result_validation_error"] = str(
            exc
          )
        else:
          required_skill_result_settled = marker is not None

      admitted_task: AdmittedTask | None = None
      raw_admitted_task = metadata.get("admitted_task")
      if raw_admitted_task is not None:
        try:
          admitted_task = AdmittedTask.model_validate(raw_admitted_task)
        except (TypeError, ValueError) as exc:
          state = TaskState.FAILED
          error = {
            "code": "invalid_admitted_task",
            "message": f"Persisted AdmittedTask is invalid: {exc}",
          }
      if admitted_task is not None and task_result is not None:
        try:
          validate_task_result_admission(task_result, admitted_task)
        except TaskResultSettlementError as exc:
          task_result = None
          state = TaskState.FAILED
          error = {
            "code": "task_result_admission_mismatch",
            "message": str(exc),
          }
      parent_result_policy: ParentResultPolicy | None = None
      raw_parent_policy = metadata.get("parent_result_policy")
      if raw_parent_policy is not None:
        try:
          parent_result_policy = ParentResultPolicy.model_validate(
            raw_parent_policy
          )
        except (TypeError, ValueError) as exc:
          state = TaskState.FAILED
          error = {
            "code": "invalid_parent_result_policy",
            "message": f"Persisted ParentResultPolicy is invalid: {exc}",
          }

      entry = TaskEntry(
        task_id=task_id,
        task_type=str(registered.get("task_type") or "background_agent"),
        agent_name=registered.get("agent_name"),
        state=state,
        started_at=float(registered.get("started_at") or time.time()),
        completed_at=float(completed_at or time.time()),
        result=dict(result) if isinstance(result, dict) else result,
        error=dict(error) if isinstance(error, dict) else error,
        metadata=metadata,
        capability_bind_receipt=dict(registered["capability_bind"]),
        admitted_task=admitted_task,
        task_result=task_result,
        parent_result_policy=parent_result_policy,
        completion_envelope=completion_envelope,
        accepted_parent_messages=accepted_parent_messages,
        delivered_messages=set(accepted_parent_messages),
        original_task_id=registered.get("original_task_id"),
        reconstructed_from_log=True,
        termination_intent=termination_intent,
        registration_persistence_state="committed",
        completion_persistence_state=(
          "committed"
          if isinstance(completed, dict)
          else "not_started"
        ),
        required_skill_result_settled=(
          required_skill_result_settled
        ),
      )
      self._tasks[task_id] = entry

    if max_suffix is not None:
      self._seq = max(self._seq, max_suffix + 1)
    self._auto_evict_completed()

  @property
  def inflight_count(self) -> int:
    return sum(1 for entry in self._tasks.values() if entry.state == TaskState.RUNNING)

  @property
  def admission_count(self) -> int:
    """Count tasks that have reserved or consumed one background slot."""

    return sum(
      1
      for entry in self._tasks.values()
      if entry.state in {TaskState.PENDING, TaskState.RUNNING}
    )

  @property
  def pending_notification_retrieval_count(self) -> int:
    return sum(
      1
      for entry in self._tasks.values()
      if (
        entry.notification_delivery_state
        in _NOTIFICATION_RETRIEVAL_REQUIRED_STATES
      )
    )

  @property
  def notification_retrieval_retention_limit(self) -> int:
    return (
      max(1, int(self._max_retained))
      + max(0, int(self._max_inflight))
    )

  def notification_retrieval_capacity_available(
    self,
    *,
    admission_count: int | None = None,
  ) -> bool:
    admitted = (
      self.admission_count
      if admission_count is None
      else max(0, int(admission_count))
    )
    return (
      self.pending_notification_retrieval_count + admitted
      < self.notification_retrieval_retention_limit
    )

  def mark_result_content_read(
    self,
    task_id: str,
    *,
    content_id: str,
  ) -> bool:
    """Record that the parent read the delivered result-handle content.

    Best-effort by design (CUR-E2E-08): the reader tool authorizes purely
    from the durable log and must keep working when the registry holds no
    live entry, so a miss here is not an error. The guard requires the
    entry's own completion envelope to have delivered exactly this
    ``content_id`` through a handle-shaped materialization.
    """
    entry = self._tasks.get(task_id)
    if entry is None or entry.completion_envelope is None:
      return False
    materialization = entry.completion_envelope.parent_materialization
    if getattr(materialization, "kind", None) not in {
      "result_handle",
      "authored_summary_with_result_handle",
    }:
      return False
    source = getattr(materialization, "source", None)
    if getattr(source, "content_id", None) != content_id:
      return False
    entry.result_content_read = True
    return True

  def mark_notification_payload_retrieved(
    self,
    task_id: str,
    *,
    notification_generation: int,
  ) -> bool:
    entry = self._tasks.get(task_id)
    if (
      entry is None
      or entry.notification_generation != notification_generation
      or entry.notification_delivery_state
      not in _NOTIFICATION_RETRIEVAL_REQUIRED_STATES
    ):
      return False
    entry.notification_delivery_state = "delivered"
    self._auto_evict_completed()
    return True

  def _auto_evict_completed(self) -> None:
    limit = max(0, self._max_retained)
    if len(self._tasks) <= limit:
      return
    completed = [
      entry
      for entry in self.list_tasks()
      if (
        entry.state in _TERMINAL_STATES
        and entry.notification_delivery_state
        not in _NOTIFICATION_RETRIEVAL_REQUIRED_STATES
      )
    ]
    while len(self._tasks) > limit and completed:
      oldest = completed.pop(0)
      self._tasks.pop(oldest.task_id, None)

  @staticmethod
  def _numeric_suffix(task_id: str) -> int | None:
    if re.search(r"_r\d+$", str(task_id)):
      return None
    try:
      return int(str(task_id).rsplit("_", 1)[-1])
    except (TypeError, ValueError):
      return None
