from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any, Awaitable, Callable, Dict, Iterable, Mapping

from agent_workflow_contracts import (
  AgentCompletionEnvelope,
  AgentOperationRef,
  AuthoredSummaryWithResultHandle,
  ChildEvidenceProjection,
  ContentHandle,
  ContentReadGrant,
  OrdinaryDelegationTaskRef,
  ParentResultPolicy,
  ProjectionInline,
  ResultHandle,
  ResultRequirement,
  SettlementProjection,
  TaskResult,
  TaskResultRef,
  TerminalNarrativeInlineExact,
  canonical_json_bytes,
  sha256_digest,
)

from .skill_lifecycle import (
  SkillLifecycleArtifactIdentity,
  TopLevelSkillLifecycleMetadata,
)
from .task_registry import NotificationQueue, TaskNotification, TaskState
from .workflow_evidence_provenance import build_child_evidence_projection

_BACKGROUND_RESULT_ACK_RESULT_KEY = "_background_result_ack"
_BACKGROUND_ERROR_CODE_MAX_CHARS = 128
_BACKGROUND_ERROR_MESSAGE_MAX_CHARS = 2_000
REQUIRED_SKILL_LIFECYCLE_METADATA_KEY = "required_skill_lifecycle"
_REQUIRED_SKILL_LIFECYCLE_SCHEMA_VERSION = 2
_REQUIRED_SKILL_LIFECYCLE_ID_MAX_CHARS = 128
_REQUIRED_SKILL_LIFECYCLE_SKILL_MAX_CHARS = 256
_REQUIRED_SKILL_LIFECYCLE_TICKER_MAX_CHARS = 64
_REQUIRED_SKILL_LIFECYCLE_PORTFOLIO_ID_MAX_CHARS = 256
WORKFLOW_TASK_METADATA_KEY = "workflow_task"
PARENT_RESULT_POLICY_METADATA_KEY = "parent_result_policy"
DEFAULT_PARENT_RESULT_MAX_INLINE_BYTES = 20_000
_WORKFLOW_TASK_ID_MAX_CHARS = 256
_WORKFLOW_TASK_ITEM_KEY_MAX_CHARS = 512


def _canonical_workflow_task_text(
  value: Any,
  *,
  field_name: str,
  max_chars: int,
) -> str:
  if (
    type(value) is not str
    or not value
    or value != value.strip()
  ):
    raise ValueError(
      f"{field_name} must be non-empty canonical text"
    )
  if len(value) > max_chars:
    raise ValueError(f"{field_name} is too long")
  if any(ord(character) < 32 or ord(character) == 127 for character in value):
    raise ValueError(f"{field_name} contains control characters")
  return value


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowTaskMetadata:
  """Trusted workflow identity attached to one native child attempt.

  Instances are constructed by server-side workflow code and deliberately do
  not accept an open metadata mapping.  The payload form is exact and
  secret-free so the same correlation can be placed on durable child
  registration and completion events.
  """

  workflow_run_id: str
  plan_id: str
  phase_number: int
  revision: int
  node_id: str
  attempt_number: int
  attempt_id: str
  operation: AgentOperationRef
  admitted_plan_digest: str
  admitted_task_digest: str
  model_bind_digest: str
  capability_binding_digest: str
  tool_grant_digest: str
  result_requirement: ResultRequirement
  item_key: str | None = None

  def __post_init__(self) -> None:
    for field_name in (
      "workflow_run_id",
      "plan_id",
      "node_id",
      "attempt_id",
    ):
      object.__setattr__(
        self,
        field_name,
        _canonical_workflow_task_text(
          getattr(self, field_name),
          field_name=field_name,
          max_chars=_WORKFLOW_TASK_ID_MAX_CHARS,
        ),
      )
    if self.item_key is not None:
      object.__setattr__(
        self,
        "item_key",
        _canonical_workflow_task_text(
          self.item_key,
          field_name="item_key",
          max_chars=_WORKFLOW_TASK_ITEM_KEY_MAX_CHARS,
        ),
      )
    for field_name in ("phase_number", "revision", "attempt_number"):
      value = getattr(self, field_name)
      if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    if not isinstance(self.operation, AgentOperationRef):
      raise TypeError("operation must be AgentOperationRef")
    if not isinstance(self.result_requirement, ResultRequirement):
      raise TypeError("result_requirement must be ResultRequirement")
    for field_name in (
      "admitted_plan_digest",
      "admitted_task_digest",
      "model_bind_digest",
      "capability_binding_digest",
      "tool_grant_digest",
    ):
      value = getattr(self, field_name)
      if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
      ):
        raise ValueError(f"{field_name} must be a canonical sha256 digest")
  def payload(self) -> dict[str, Any]:
    return {
      "workflow_run_id": self.workflow_run_id,
      "plan_id": self.plan_id,
      "phase_number": self.phase_number,
      "revision": self.revision,
      "node_id": self.node_id,
      "attempt_number": self.attempt_number,
      "attempt_id": self.attempt_id,
      "operation": self.operation.model_dump(mode="json"),
      "admitted_plan_digest": self.admitted_plan_digest,
      "admitted_task_digest": self.admitted_task_digest,
      "model_bind_digest": self.model_bind_digest,
      "capability_binding_digest": self.capability_binding_digest,
      "tool_grant_digest": self.tool_grant_digest,
      "result_requirement": self.result_requirement.model_dump(mode="json"),
      "item_key": self.item_key,
    }

  @classmethod
  def from_payload(cls, value: Mapping[str, Any]) -> "WorkflowTaskMetadata":
    if not isinstance(value, Mapping):
      raise ValueError("workflow task metadata must be an object")
    expected_fields = {
      "workflow_run_id",
      "plan_id",
      "phase_number",
      "revision",
      "node_id",
      "attempt_number",
      "attempt_id",
      "operation",
      "admitted_plan_digest",
      "admitted_task_digest",
      "model_bind_digest",
      "capability_binding_digest",
      "tool_grant_digest",
      "result_requirement",
      "item_key",
    }
    if set(value) != expected_fields:
      raise ValueError("workflow task metadata must use the exact schema")
    fields = {field_name: value[field_name] for field_name in expected_fields}
    fields["operation"] = AgentOperationRef.model_validate(fields["operation"])
    fields["result_requirement"] = ResultRequirement.model_validate(
      fields["result_requirement"]
    )
    return cls(**fields)


def normalize_workflow_task_metadata(
  value: WorkflowTaskMetadata | None,
) -> WorkflowTaskMetadata | None:
  """Accept only the frozen server-owned workflow correlation type."""

  if value is None:
    return None
  if type(value) is not WorkflowTaskMetadata:
    raise TypeError(
      "workflow_task_metadata must be exact WorkflowTaskMetadata"
    )
  return value


def workflow_owns_terminal_notification(entry: Any) -> bool:
  """Return whether workflow_run owns this task's terminal delivery boundary.

  Native workflow children still publish their durable task and workflow
  events.  Their parent-facing delivery is coalesced by workflow_run(observe),
  so emitting the generic background notification as well would race durable
  workflow settlement and repeatedly inject the wrong retrieval guidance.
  """

  metadata = getattr(entry, "metadata", None)
  return (
    isinstance(metadata, dict)
    and metadata.get(WORKFLOW_TASK_METADATA_KEY) is not None
  )


def workflow_obstruction_blocks_settlement(row: Any) -> bool:
  """Map one workflow-obstruction row to the §5.3 hold decision (T2-I04).

  States map per design §5.3: ``authoring``/``running``/``cancel_requested``
  obstruct clean session settlement unconditionally; ``awaiting_action``
  obstructs until its boundary notification was delivered and acked;
  ``terminal`` (or any unknown row) never obstructs.  Rows are duck-typed —
  the provider lives outside this package.
  """

  state = getattr(row, "state", None)
  if state in {"authoring", "running", "cancel_requested"}:
    return True
  if state == "awaiting_action":
    return not bool(getattr(row, "boundary_notification_acked", False))
  return False


class ParentResultMaterializationError(ValueError):
  """A canonical task result cannot satisfy its admitted parent policy."""


def ordinary_parent_result_policy(
  task_result: TaskResult,
) -> ParentResultPolicy:
  """Return the server-owned default for an ordinary direct delegation.

  Narrative delivery is terminal-message-first.  The bound applies only to
  the parent context view; the canonical result remains lossless.  Oversized
  values therefore become an explicit readable handle, never a clipped prefix.
  """

  if not isinstance(task_result, TaskResult):
    raise TypeError("ordinary parent-result policy requires TaskResult")
  preferred = (
    "terminal_narrative_inline_exact"
    if task_result.values.terminal_narrative is not None
    else "projection_inline"
  )
  return ParentResultPolicy(
    preferred=preferred,
    max_inline_bytes=DEFAULT_PARENT_RESULT_MAX_INLINE_BYTES,
    on_overflow="result_handle",
  )


def agent_completion_message_id(task_result: TaskResult) -> str:
  """Derive the stable delivery ID from one immutable result identity."""

  if not isinstance(task_result, TaskResult):
    raise TypeError("completion message identity requires TaskResult")
  digest = sha256_digest({
    "kind": "agent_completion",
    "task_result_ref": TaskResultRef.from_result(task_result).model_dump(
      mode="json"
    ),
  })
  return f"agent-completion:{digest.removeprefix('sha256:')}"


def _primary_result_content(task_result: TaskResult) -> ContentHandle:
  values = task_result.values
  if values.terminal_narrative is not None:
    return values.terminal_narrative
  if values.projection is not None:
    return values.projection.content
  if values.artifacts:
    raise ParentResultMaterializationError(
      "artifact-only result content has no registered direct-parent reader"
    )
  raise ParentResultMaterializationError(
    "task result has no canonical value for parent delivery"
  )


def _direct_parent_read_grant(
  source: ContentHandle,
  *,
  read_grant_factory: Callable[[ContentHandle], ContentReadGrant],
) -> ContentReadGrant:
  grant = read_grant_factory(source)
  if not isinstance(grant, ContentReadGrant):
    raise ParentResultMaterializationError(
      "read_grant_factory must return ContentReadGrant"
    )
  if grant.content_id != source.content_id or grant.scope != "direct_parent":
    raise ParentResultMaterializationError(
      "parent read grant must address the exact result content"
    )
  return grant


def build_agent_completion_envelope(
  task_result: TaskResult,
  *,
  policy: ParentResultPolicy,
  terminal_narrative_reader: Callable[[TaskResult], str],
  read_grant_factory: Callable[[ContentHandle], ContentReadGrant],
  authored_summary: str | None = None,
  message_id: str | None = None,
) -> AgentCompletionEnvelope:
  """Compile one lossless ordinary result into its direct-parent view.

  This function deliberately has no transcript fallback.  The terminal reader
  resolves the canonical narrative handle (and is expected to verify its
  digest); overflow follows the admitted policy before any bytes are read.
  """

  if not isinstance(task_result, TaskResult):
    raise TypeError("completion materialization requires TaskResult")
  if not isinstance(policy, ParentResultPolicy):
    raise TypeError("completion materialization requires ParentResultPolicy")
  if not isinstance(task_result.logical_task, OrdinaryDelegationTaskRef):
    raise ParentResultMaterializationError(
      "workflow-node results are delivered only through workflow aggregation"
    )

  values = task_result.values
  source = _primary_result_content(task_result)

  def result_handle() -> ResultHandle:
    return ResultHandle(
      source=source,
      read_grant=_direct_parent_read_grant(
        source,
        read_grant_factory=read_grant_factory,
      ),
    )

  def summary_with_handle() -> AuthoredSummaryWithResultHandle:
    summary = authored_summary.strip() if isinstance(authored_summary, str) else ""
    if not summary:
      raise ParentResultMaterializationError(
        "authored-summary delivery requires an operation-authored summary"
      )
    return AuthoredSummaryWithResultHandle(
      summary=summary,
      source=source,
      read_grant=_direct_parent_read_grant(
        source,
        read_grant_factory=read_grant_factory,
      ),
    )

  def overflow_materialization() -> AuthoredSummaryWithResultHandle | ResultHandle:
    if policy.on_overflow == "authored_summary_with_result_handle":
      return summary_with_handle()
    if policy.on_overflow == "result_handle":
      return result_handle()
    raise ParentResultMaterializationError(
      "canonical result exceeds the admitted parent inline limit"
    )

  preferred = policy.preferred
  if preferred == "terminal_narrative_inline_exact":
    narrative = values.terminal_narrative
    if narrative is None:
      raise ParentResultMaterializationError(
        "terminal narrative policy requires a canonical narrative"
      )
    source = narrative
    if narrative.content_bytes > policy.max_inline_bytes:
      materialization = overflow_materialization()
    else:
      exact_text = terminal_narrative_reader(task_result)
      if not isinstance(exact_text, str):
        raise ParentResultMaterializationError(
          "terminal narrative reader must return exact text"
        )
      materialization = TerminalNarrativeInlineExact(
        source=narrative,
        content=exact_text,
      )
  elif preferred == "projection_inline":
    projection = values.projection
    if projection is None or projection.inline_view is None:
      raise ParentResultMaterializationError(
        "projection-inline policy requires an exact inline projection"
      )
    source = projection.content
    if projection.content.content_bytes > policy.max_inline_bytes:
      materialization = overflow_materialization()
    else:
      materialization = ProjectionInline(
        source=projection.content,
        contract=projection.contract,
        value=projection.inline_view,
      )
  elif preferred == "authored_summary_with_result_handle":
    materialization = summary_with_handle()
  else:
    materialization = result_handle()

  child_evidence_payload = build_child_evidence_projection(task_result)
  return AgentCompletionEnvelope(
    message_id=message_id or agent_completion_message_id(task_result),
    task_result_ref=TaskResultRef.from_result(task_result),
    settlement_projection=SettlementProjection(
      execution_status=task_result.execution.status,
      outcome_disposition=(
        task_result.outcome.disposition
        if task_result.outcome is not None
        else None
      ),
    ),
    parent_materialization=materialization,
    child_evidence=(
      ChildEvidenceProjection.model_validate(child_evidence_payload)
      if child_evidence_payload is not None
      else None
    ),
  )


#: Kept for inline materializations, whose full prose already renders in the
#: notification payload and self-identifies.
_INLINE_COMPLETION_SUMMARY = (
  "Agent completed; consume the typed parent materialization."
)

#: The dispatch-objective echo is bounded well under TaskNotification's
#: 2000-char render truncation, leaving room for the instruction text.
_DISPATCH_OBJECTIVE_ECHO_MAX_CHARS = 360


def dispatch_objective_echo(entry: Any) -> str | None:
  """A bounded echo of the parent's own dispatch objective, or None.

  Fail-open by design (CUR-E2E-08): this runs inside the registry
  transition's notification listener, so it must never raise and must do no
  I/O. The objective is the parent's own words for the task — the identity a
  hex task_id cannot carry.
  """
  try:
    objective = getattr(getattr(entry, "admitted_task", None), "objective", None)
    if not isinstance(objective, str):
      metadata = getattr(entry, "metadata", None)
      admitted = metadata.get("admitted_task") if isinstance(metadata, dict) else None
      objective = admitted.get("objective") if isinstance(admitted, dict) else None
    if not isinstance(objective, str):
      return None
    echo = " ".join(objective.split())
    if not echo:
      return None
    if len(echo) > _DISPATCH_OBJECTIVE_ECHO_MAX_CHARS:
      echo = echo[: _DISPATCH_OBJECTIVE_ECHO_MAX_CHARS - 1] + "…"
    return echo
  except Exception:  # noqa: BLE001 - listener context: never raise
    return None


def _completion_notification_summary(
  entry: Any,
  envelope: AgentCompletionEnvelope,
) -> str:
  """A self-describing summary for handle-shaped deliveries (CUR-E2E-08).

  A bare ``result_handle`` notification used to carry only a constant
  string, a hex task_id, and an agent name identical across siblings — no
  token of what the child did. A parent juggling several children (VRT
  run 4: a delivery landing mid-recovery on another child's failure) never
  mapped the hex id back to its own dispatch, never read the handle, and
  closed claiming the settled task was still outstanding. The summary now
  carries the parent's own dispatch objective and the exact next action.
  Inline materializations keep the constant summary: their full prose is
  already in the payload.
  """
  materialization = getattr(envelope, "parent_materialization", None)
  if getattr(materialization, "kind", None) != "result_handle":
    return _INLINE_COMPLETION_SUMMARY
  source = getattr(materialization, "source", None)
  content_bytes = getattr(source, "content_bytes", None)
  size = (
    f" ({content_bytes} bytes)"
    if isinstance(content_bytes, int)
    else ""
  )
  echo = dispatch_objective_echo(entry)
  objective_clause = (
    f" Dispatched objective: {echo}"
    if echo is not None
    else ""
  )
  return (
    f"Agent completed. The full result{size} was delivered as a content "
    "handle — read it with get_agent_result_content (content_id and "
    "read_grant_id are in this notification's payload) before relying on "
    "or reporting this task's outcome. This task is settled, not running."
    f"{objective_clause}"
  )


def agent_completion_notification(
  entry: Any,
  envelope: AgentCompletionEnvelope,
  *,
  timestamp: float,
) -> TaskNotification:
  """Build one exact model-visible completion notification.

  Notification omission is not a valid overflow policy for ordinary agent
  completion.  The envelope must already have selected a handle before this
  transport boundary.
  """

  if workflow_owns_terminal_notification(entry):
    raise ParentResultMaterializationError(
      "workflow nodes cannot publish direct-parent completion notifications"
    )
  if not isinstance(envelope, AgentCompletionEnvelope):
    raise TypeError("completion notification requires AgentCompletionEnvelope")
  payload = envelope.model_dump(mode="json")
  # Validate canonical encodability before TaskNotification applies its own
  # bounded XML representation.
  canonical_json_bytes(payload)
  notification = TaskNotification(
    task_id=str(entry.task_id),
    agent_name=getattr(entry, "agent_name", None),
    event="completed",
    summary=_completion_notification_summary(entry, envelope),
    timestamp=timestamp,
    payload=payload,
    notification_generation=getattr(
      entry,
      "notification_generation",
      None,
    ),
  )
  _payload_json, omission_reason = notification.inline_payload()
  if omission_reason is not None:
    raise ParentResultMaterializationError(
      "agent completion envelope exceeded notification transport after "
      f"materialization ({omission_reason})"
    )
  return notification


@dataclass(frozen=True)
class BackgroundResultRequest:
  task_id: str
  wait: bool
  timeout: float
  cursor: str | None


@dataclass(frozen=True)
class PlanProgressSnapshot:
  plan_id: str
  phase: str
  nodes_total: int
  nodes_complete: int
  items_total: int
  items_complete: int
  current_node: str | None
  status: str

  def __post_init__(self) -> None:
    for field_name in ("plan_id", "phase", "status"):
      value = getattr(self, field_name)
      if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    if self.current_node is not None and (
      not isinstance(self.current_node, str)
      or not self.current_node.strip()
    ):
      raise ValueError("current_node must be null or non-empty text")
    for complete_name, total_name in (
      ("nodes_complete", "nodes_total"),
      ("items_complete", "items_total"),
    ):
      complete = getattr(self, complete_name)
      total = getattr(self, total_name)
      if (
        isinstance(complete, bool)
        or isinstance(total, bool)
        or not isinstance(complete, int)
        or not isinstance(total, int)
        or complete < 0
        or total < 0
        or complete > total
      ):
        raise ValueError(
          f"{complete_name} and {total_name} must satisfy "
          f"0 <= {complete_name} <= {total_name}"
        )

  def payload(self) -> dict[str, Any]:
    return {
      "plan_id": self.plan_id,
      "phase": self.phase,
      "nodes_total": self.nodes_total,
      "nodes_complete": self.nodes_complete,
      "items_total": self.items_total,
      "items_complete": self.items_complete,
      "current_node": self.current_node,
      "status": self.status,
    }


class PlanNotificationProducer:
  """Emit bounded plan notifications through the existing task queue."""

  def __init__(
    self,
    *,
    queue: NotificationQueue,
    entry: Any,
    item_cadence: int = 5,
    seconds_cadence: float = 15.0,
    monotonic: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], float] = time.time,
  ) -> None:
    if (
      isinstance(item_cadence, bool)
      or not isinstance(item_cadence, int)
      or item_cadence <= 0
    ):
      raise ValueError("item_cadence must be a positive integer")
    if (
      isinstance(seconds_cadence, bool)
      or not isinstance(seconds_cadence, (int, float))
      or not math.isfinite(float(seconds_cadence))
      or seconds_cadence <= 0
    ):
      raise ValueError("seconds_cadence must be positive and finite")
    if getattr(entry, "task_type", None) != "plan_run":
      raise ValueError("plan notifications require task_type='plan_run'")
    self._queue = queue
    self._entry = entry
    self._item_cadence = item_cadence
    self._seconds_cadence = float(seconds_cadence)
    self._monotonic = monotonic
    self._wall_clock = wall_clock
    self._last_progress_at = monotonic()
    self._last_items_complete = 0

  def _notification(
    self,
    *,
    event: str,
    snapshot: PlanProgressSnapshot,
  ) -> TaskNotification:
    self._entry.notification_generation += 1
    return TaskNotification(
      task_id=self._entry.task_id,
      agent_name=self._entry.agent_name,
      event=event,
      summary=(
        f"Plan {snapshot.plan_id}: {snapshot.status}; "
        f"{snapshot.nodes_complete}/{snapshot.nodes_total} nodes, "
        f"{snapshot.items_complete}/{snapshot.items_total} items"
      ),
      timestamp=self._wall_clock(),
      payload=snapshot.payload(),
      notification_generation=self._entry.notification_generation,
    )

  def emit_transition(self, snapshot: PlanProgressSnapshot) -> bool:
    return self._emit_progress(snapshot)

  def emit_item_completion(
    self,
    snapshot: PlanProgressSnapshot,
  ) -> bool:
    now = self._monotonic()
    item_delta = snapshot.items_complete - self._last_items_complete
    if (
      item_delta < self._item_cadence
      and now - self._last_progress_at < self._seconds_cadence
    ):
      return False
    return self._emit_progress(snapshot, emitted_at=now)

  def flush(self, snapshot: PlanProgressSnapshot) -> bool:
    return self._emit_progress(snapshot)

  def emit_approval_pending(
    self,
    snapshot: PlanProgressSnapshot,
  ) -> bool:
    return self._queue.push(
      self._notification(
        event="plan_approval_pending",
        snapshot=snapshot,
      )
    )

  def _emit_progress(
    self,
    snapshot: PlanProgressSnapshot,
    *,
    emitted_at: float | None = None,
  ) -> bool:
    self._last_progress_at = (
      self._monotonic()
      if emitted_at is None
      else emitted_at
    )
    self._last_items_complete = snapshot.items_complete
    return self._queue.push_or_replace_pending(
      self._notification(event="plan_progress", snapshot=snapshot)
    )


def background_timeout_value(raw_timeout: Any) -> float:
  timeout = 60.0 if raw_timeout is None else float(raw_timeout)
  return max(0.0, min(timeout, 120.0))


def ensure_sub_agent_semaphore(
  current_semaphore: Any,
  max_concurrent_sub_agents: int | None,
  *,
  semaphore_factory: Callable[[int], Any],
) -> Any:
  if current_semaphore is None and max_concurrent_sub_agents is not None:
    return semaphore_factory(max_concurrent_sub_agents)
  return current_semaphore


def parse_background_result_request(tool_input: Dict[str, Any]) -> tuple[BackgroundResultRequest | None, Dict[str, Any] | None]:
  raw_task_id = tool_input.get("task_id")
  if not isinstance(raw_task_id, str) or not raw_task_id.strip():
    return None, {"code": "invalid_input", "message": "task_id is required"}
  task_id = raw_task_id.strip()

  wait = tool_input.get("wait", False)
  if not isinstance(wait, bool):
    return None, {"code": "invalid_input", "message": "wait must be a boolean"}

  raw_timeout = tool_input.get("timeout")
  if raw_timeout is not None:
    if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float)):
      return None, {"code": "invalid_input", "message": "timeout must be a number"}
    if not math.isfinite(float(raw_timeout)):
      return None, {"code": "invalid_input", "message": "timeout must be finite"}
  timeout = background_timeout_value(raw_timeout)
  raw_cursor = tool_input.get("cursor")
  if raw_cursor is not None and (
    not isinstance(raw_cursor, str)
    or not raw_cursor.strip()
  ):
    return None, {
      "code": "invalid_input",
      "message": "cursor must be a non-empty string",
    }
  cursor = raw_cursor.strip() if isinstance(raw_cursor, str) else None
  if task_id == "*" and cursor is not None:
    return None, {
      "code": "invalid_input",
      "message": "cursor requires one exact task_id",
    }
  return BackgroundResultRequest(
    task_id=task_id,
    wait=wait,
    timeout=timeout,
    cursor=cursor,
  ), None


def background_elapsed_seconds(bg_task: Any, *, now: float) -> int:
  end_t = bg_task.completed_at if bg_task.completed_at is not None else now
  return max(0, int(end_t - bg_task.started_at))


def background_asyncio_tasks(running_entries: Iterable[Any]) -> list[Any]:
  return [
    bg_task.asyncio_task
    for bg_task in running_entries
    if bg_task.asyncio_task is not None
  ]


def background_wait_tasks(entries: Iterable[Any]) -> list[Any]:
  return [
    bg_task.asyncio_task
    for bg_task in entries
    if bg_task.asyncio_task is not None and not bg_task.completed
  ]


async def wait_for_background_tasks(
  entries: Iterable[Any],
  *,
  wait: bool,
  timeout: float,
  wait_fn: Callable[..., Awaitable[Any]],
) -> None:
  if not wait:
    return
  pending = background_wait_tasks(entries)
  if pending:
    await wait_fn(pending, timeout=timeout)


async def background_result_task(
  task_id: str,
  *,
  registry_lookup: Callable[[str], Any],
  log_lookup: Callable[[str], Awaitable[Any]],
) -> tuple[Any | None, Dict[str, Any] | None]:
  bg_task = registry_lookup(task_id)
  if bg_task is None:
    bg_task = await log_lookup(task_id)
    if bg_task is None:
      return None, {"code": "not_found", "message": f"Unknown background task: {task_id}"}
  return bg_task, None


def background_task_ids(running_entries: Iterable[Any]) -> list[str]:
  return [bg_task.task_id for bg_task in running_entries]


def background_task_ids_for_asyncio_tasks(
  running_entries: Iterable[Any],
  asyncio_tasks: Iterable[Any],
) -> list[str]:
  selected_tasks = set(asyncio_tasks)
  return [
    bg_task.task_id
    for bg_task in running_entries
    if bg_task.asyncio_task in selected_tasks
  ]


def kill_background_tasks(
  running_entries: Iterable[Any],
  *,
  kill_task: Callable[[str], None],
) -> None:
  for task_id in background_task_ids(running_entries):
    kill_task(task_id)


def kill_background_tasks_for_asyncio_tasks(
  running_entries: Iterable[Any],
  asyncio_tasks: Iterable[Any],
  *,
  kill_task: Callable[[str], None],
) -> None:
  for task_id in background_task_ids_for_asyncio_tasks(running_entries, asyncio_tasks):
    kill_task(task_id)


def _observe_completed_background_tasks(tasks: Iterable[Any]) -> None:
  for task in tasks:
    exception = getattr(task, "exception", None)
    if not callable(exception):
      continue
    try:
      exception()
    except BaseException:
      pass


async def drain_cancelled_background_tasks(
  running_entries: Iterable[Any],
  pending: Iterable[Any],
  *,
  kill_task: Callable[[str], None],
  wait_fn: Callable[..., Awaitable[Any]],
  timeout: float,
) -> set[Any]:
  selected = list(pending)
  kill_background_tasks(running_entries, kill_task=kill_task)
  if not selected:
    return set()
  done, still_pending = await wait_fn(selected, timeout=timeout)
  _observe_completed_background_tasks(done)
  return set(still_pending)


async def drain_still_pending_background_tasks(
  running_entries: Iterable[Any],
  pending: Iterable[Any],
  *,
  kill_task: Callable[[str], None],
  wait_fn: Callable[..., Awaitable[Any]],
  wait_timeout: float,
  drain_timeout: float,
) -> set[Any]:
  selected = list(pending)
  if not selected:
    return set()
  done, still_pending = await wait_fn(selected, timeout=wait_timeout)
  _observe_completed_background_tasks(done)
  if not still_pending:
    return set()
  kill_background_tasks_for_asyncio_tasks(
    running_entries,
    still_pending,
    kill_task=kill_task,
  )
  drained, remaining = await wait_fn(still_pending, timeout=drain_timeout)
  _observe_completed_background_tasks(drained)
  return set(remaining)


def background_task_limit_error(
  *,
  admission_count: int,
  max_background_tasks: int,
) -> Dict[str, Any] | None:
  if admission_count < max_background_tasks:
    return None
  return {
    "code": "max_background_tasks",
    "message": (
      f"Background task limit reached ({max_background_tasks}). "
      "Wait for an existing background task to finish before launching another."
    ),
  }


def call_before_background_task_start_hook(
  on_before_start: Callable[[], None] | None,
  *,
  log_session_id: str | Callable[[], str],
  logger: Any,
) -> None:
  if on_before_start is None:
    return
  try:
    on_before_start()
  except Exception as exc:
    session_id = log_session_id() if callable(log_session_id) else log_session_id
    logger.warning("[%s] on_before_start hook failed (non-fatal): %s", session_id, exc)


def entry_aware_background_handler(
  handler: Callable[..., Awaitable[Any]],
  entry: Any,
) -> Callable[..., Awaitable[Any]]:
  async def _entry_aware_handler(tool_input: Dict[str, Any], **kwargs: Any) -> Any:
    kwargs["task_entry"] = entry
    return await handler(tool_input, **kwargs)

  return _entry_aware_handler


async def resume_chain_depth(
  task_id: str,
  *,
  task_lookup: Callable[[str], Awaitable[Any]],
) -> int:
  depth = 0
  seen: set[str] = set()
  current_id: str | None = task_id
  while current_id:
    if current_id in seen:
      break
    seen.add(current_id)
    entry = await task_lookup(current_id)
    parent_id = entry.original_task_id if entry is not None else None
    if not parent_id:
      break
    depth += 1
    current_id = parent_id
  return depth


async def resume_root_task_id(
  task_id: str,
  *,
  task_lookup: Callable[[str], Awaitable[Any]],
) -> str:
  seen: set[str] = set()
  current_id = task_id
  while current_id not in seen:
    seen.add(current_id)
    entry = await task_lookup(current_id)
    parent_id = entry.original_task_id if entry is not None else None
    if not parent_id:
      return current_id
    current_id = parent_id
  return task_id


async def resumed_task_ids(
  task_id: str,
  *,
  task_entries: Callable[[], Iterable[Any]],
  resume_root: Callable[[str], Awaitable[str]],
) -> list[str]:
  root_id = await resume_root(task_id)
  resumed: list[str] = []
  for entry in task_entries():
    if entry.task_id == root_id or entry.original_task_id is None:
      continue
    if await resume_root(entry.task_id) == root_id:
      resumed.append(entry.task_id)
  return resumed


async def resume_task_id_override(
  original_task_id: str | None,
  *,
  max_resume_chain_depth: int,
  resume_chain_depth: Callable[[str], Awaitable[int]],
  resume_root: Callable[[str], Awaitable[str]],
) -> tuple[str | None, Dict[str, Any] | None]:
  if not original_task_id:
    return None, None

  depth = await resume_chain_depth(original_task_id)
  if depth >= max_resume_chain_depth:
    return None, {
      "code": "max_resume_chain_depth",
      "message": f"Resume chain depth limit reached ({max_resume_chain_depth}) for {original_task_id}",
    }

  root_id = await resume_root(original_task_id)
  return f"{root_id}_r{depth + 1}", None


def resume_root_task_id_from_registry(
  task_id: str,
  *,
  task_lookup: Callable[[str], Any],
) -> str:
  seen: set[str] = set()
  current_id = task_id
  while current_id not in seen:
    seen.add(current_id)
    entry = task_lookup(current_id)
    parent_id = entry.original_task_id if entry is not None else None
    if not parent_id:
      return current_id
    current_id = parent_id
  return task_id


def resumed_task_ids_from_registry(
  task_id: str,
  *,
  task_entries: Iterable[Any],
  task_lookup: Callable[[str], Any],
) -> list[str]:
  root_id = resume_root_task_id_from_registry(task_id, task_lookup=task_lookup)
  resumed: list[str] = []
  for entry in task_entries:
    if entry.task_id == root_id or entry.original_task_id is None:
      continue
    if resume_root_task_id_from_registry(entry.task_id, task_lookup=task_lookup) == root_id:
      resumed.append(entry.task_id)
  return resumed


def background_task_payload(
  bg_task: Any,
  *,
  elapsed_seconds: int,
  resumed_task_ids: list[str] | None = None,
  now: float,
) -> Dict[str, Any]:
  payload: Dict[str, Any] = {
    "task_id": bg_task.task_id,
    "status": "running",
  }
  if bg_task.agent_name:
    payload["agent"] = bg_task.agent_name

  if getattr(bg_task, "state", None) == TaskState.INTERRUPTED:
    metadata = getattr(bg_task, "metadata", {}) if isinstance(getattr(bg_task, "metadata", None), dict) else {}
    resumed_as = list(resumed_task_ids or [])
    interrupted_error = (
      bounded_background_error(bg_task.error)
      if isinstance(getattr(bg_task, "error", None), dict)
      else None
    )
    interrupted_message = (
      str(interrupted_error.get("message") or "")
      if interrupted_error is not None
      else ""
    ) or "Background task was interrupted by a gateway restart before completion."
    payload.update(
      {
        "status": "interrupted",
        "completed": True,
        "elapsed_seconds": elapsed_seconds,
        "started_at": bg_task.started_at,
        "owner_runner_id": metadata.get("owner_runner_id"),
        "owner_role": metadata.get("owner_role"),
        "sub_agent_id": metadata.get("sub_agent_id"),
        "parent_turn_id": metadata.get("parent_turn_id"),
        "call_index": metadata.get("call_index"),
        "task_type": metadata.get("task_type", getattr(bg_task, "task_type", None)),
        "capability_bind": (
          dict(getattr(bg_task, "capability_bind_receipt", None))
          if isinstance(getattr(bg_task, "capability_bind_receipt", None), dict)
          else metadata.get("capability_bind")
        ),
        "original_task_id": getattr(bg_task, "original_task_id", None),
        "resumable": bool(metadata.get("resumable", False)),
        "resumed_as": resumed_as,
        "latest_resume_task_id": resumed_as[-1] if resumed_as else None,
        "message": interrupted_message,
      }
    )
    if interrupted_error is not None:
      payload["error"] = interrupted_error
    return payload

  if getattr(bg_task, "state", None) == TaskState.KILLED:
    payload["status"] = "killed"
    payload["elapsed_seconds"] = elapsed_seconds
    if isinstance(bg_task.result, dict):
      for key, value in bg_task.result.items():
        if key not in {"task_id", "status", "agent"}:
          payload[key] = value
    return payload

  if bg_task.completed:
    payload["elapsed_seconds"] = elapsed_seconds
    if bg_task.error is not None:
      payload["status"] = "error"
      payload["error"] = bounded_background_error(bg_task.error)
      return payload
    payload["status"] = (
      "error"
      if getattr(bg_task, "state", None) == TaskState.FAILED
      else "completed"
    )
    if isinstance(bg_task.result, dict):
      for key, value in bg_task.result.items():
        if key not in {"task_id", "status", "agent"}:
          payload[key] = value
    elif bg_task.result is not None:
      payload["result"] = bg_task.result
    return payload

  payload["elapsed_seconds"] = elapsed_seconds
  progress = getattr(bg_task, "progress", None)
  if progress is not None and progress.tool_use_count > 0:
    payload["progress"] = {
      "tools_used": progress.tool_use_count,
      "turns": progress.turn_count,
      "last_tool": progress.last_tool_name,
      "idle_seconds": int(now - progress.last_activity_at) if progress.last_activity_at else None,
      "output_tokens": progress.output_tokens,
    }
  return payload


def bounded_background_error(
  error: Mapping[str, Any],
) -> dict[str, Any]:
  """Project arbitrary handler failures into one bounded model-facing shape."""

  raw_code = error.get("code")
  code = (
    str(raw_code).strip()
    if raw_code is not None
    else ""
  ) or "background_error"
  code = code[:_BACKGROUND_ERROR_CODE_MAX_CHARS]
  raw_message = (
    error.get("message")
    or error.get("error")
    or code
  )
  message = str(raw_message)
  bounded = {
    "code": code,
    "message": message[:_BACKGROUND_ERROR_MESSAGE_MAX_CHARS],
  }
  if isinstance(error.get("recoverable"), bool):
    bounded["recoverable"] = error["recoverable"]
  if len(message) > _BACKGROUND_ERROR_MESSAGE_MAX_CHARS:
    bounded.update({
      "message_truncated": True,
      "original_message_chars": len(message),
      "message_sha256": hashlib.sha256(
        message.encode("utf-8", errors="replace")
      ).hexdigest(),
    })
  return bounded


def background_task_reminder_text(
  running_tasks: Iterable[Any],
  *,
  elapsed_seconds: Callable[[Any], int],
) -> str:
  entries: list[str] = []
  for bg_task in running_tasks:
    parts = [f"running, {elapsed_seconds(bg_task)}s"]
    if bg_task.progress.tool_use_count > 0:
      parts.append(f"{bg_task.progress.tool_use_count} tools")
      if bg_task.progress.last_tool_name:
        parts.append(f"last: {bg_task.progress.last_tool_name}")
    status = ", ".join(parts)
    label = bg_task.task_id
    if bg_task.agent_name:
      label += f" ({bg_task.agent_name}, {status})"
    else:
      label += f" ({status})"
    entries.append(label)
  if not entries:
    return ""
  return "[Background tasks active: " + ", ".join(entries) + "]"


def task_correlation_payload(
  entry: Any,
  *,
  runner_id: str | None,
  role: str,
) -> Dict[str, Any]:
  metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
  payload = {
    "task_id": entry.task_id,
    "owner_runner_id": metadata.get("owner_runner_id", runner_id),
    "owner_role": metadata.get("owner_role", role),
    "sub_agent_id": metadata.get("sub_agent_id"),
    "parent_turn_id": metadata.get("parent_turn_id"),
    "call_index": metadata.get("call_index"),
    "task_type": metadata.get("task_type", "background"),
    "capability_bind": (
      dict(entry.capability_bind_receipt)
      if isinstance(entry.capability_bind_receipt, dict)
      else metadata.get("capability_bind")
    ),
  }
  if entry.original_task_id is not None:
    payload["original_task_id"] = entry.original_task_id
  cost_observation_threshold_usd = parse_cost_observation_threshold_usd(
    metadata.get("cost_observation_threshold_usd")
  )
  if cost_observation_threshold_usd is not None:
    payload["cost_observation_threshold_usd"] = (
      cost_observation_threshold_usd
    )
  raw_workflow_task = metadata.get(WORKFLOW_TASK_METADATA_KEY)
  if raw_workflow_task is not None:
    payload.update(
      WorkflowTaskMetadata.from_payload(raw_workflow_task).payload()
    )
  return payload


def task_completed_event_payload(
  entry: Any,
  final_state: TaskState,
  *,
  correlation_payload: Dict[str, Any],
  completed_at: float,
) -> Dict[str, Any]:
  payload = dict(correlation_payload)
  payload.update(
    {
      "type": "task_completed",
      "final_state": final_state.value,
      "completed_at": completed_at,
      "result": entry.result,
      "error": entry.error,
    }
  )
  return payload


def require_capability_bind_receipt(
  capability_bind_receipt: Any,
) -> dict[str, str]:
  """Refuse to build a durable registration without a capability bind.

  Durable task records reauthorize the exact binding at resume; a bind-less
  registration can only fail late. Refuse loudly at the write boundary
  instead (model-selection-authority review 2026-08-14, B7).
  """

  if (
    not isinstance(capability_bind_receipt, dict)
    or not capability_bind_receipt
  ):
    raise ValueError(
      "durable task registration requires a capability_bind receipt"
    )
  return dict(capability_bind_receipt)


def task_registered_event_payload(
  entry: Any,
  *,
  correlation_payload: Dict[str, Any],
  agent_name: str | None,
  parent_session_id: str,
) -> Dict[str, Any]:
  require_capability_bind_receipt(correlation_payload.get("capability_bind"))
  metadata = dict(entry.metadata)
  raw_workflow_task = metadata.get(WORKFLOW_TASK_METADATA_KEY)
  if raw_workflow_task is not None:
    metadata[WORKFLOW_TASK_METADATA_KEY] = (
      WorkflowTaskMetadata.from_payload(raw_workflow_task).payload()
    )
  return {
    "type": "task_registered",
    **correlation_payload,
    "task_type": entry.task_type,
    "agent_name": agent_name,
    "parent_session_id": parent_session_id,
    "metadata": metadata,
    "started_at": entry.started_at,
  }


def background_task_started_result(entry: Any, *, agent_name: str | None) -> Dict[str, Any]:
  result: Dict[str, Any] = {
    "task_id": entry.task_id,
    "status": "running",
  }
  if agent_name:
    result["agent"] = agent_name
  return result


def parse_cost_observation_threshold_usd(raw_threshold: Any) -> float | None:
  if raw_threshold is None or isinstance(raw_threshold, bool):
    return None
  try:
    parsed_threshold = float(raw_threshold)
  except (TypeError, ValueError):
    return None
  if math.isfinite(parsed_threshold) and parsed_threshold > 0:
    return parsed_threshold
  return None


def background_task_call_index(task_id: str) -> int:
  base_for_call_index = task_id.split("_r", 1)[0]
  try:
    return int(base_for_call_index.rsplit("_", 1)[-1])
  except ValueError:
    return 0


def background_task_registration_metadata(
  *,
  owner_runner_id: str | None,
  owner_role: str,
  sub_agent_id: str,
  parent_turn_id: str | None,
  call_index: int,
  capability_bind_receipt: dict[str, str],
  cost_observation_threshold_usd: float | None,
  original_task_id: str | None,
  tool_input: Dict[str, Any],
  required_skill_lifecycle: Mapping[str, Any] | None = None,
  workflow_task_metadata: WorkflowTaskMetadata | None = None,
) -> Dict[str, Any]:
  correlation: Dict[str, Any] = {
    "owner_runner_id": owner_runner_id,
    "owner_role": owner_role,
    "sub_agent_id": sub_agent_id,
    "parent_turn_id": parent_turn_id,
    "call_index": call_index,
    "task_type": "background",
  }
  correlation["capability_bind"] = require_capability_bind_receipt(
    capability_bind_receipt
  )
  if cost_observation_threshold_usd is not None:
    correlation["cost_observation_threshold_usd"] = (
      cost_observation_threshold_usd
    )
  if original_task_id is not None:
    correlation["original_task_id"] = original_task_id
  normalized_skill_lifecycle = normalize_required_skill_lifecycle(
    required_skill_lifecycle
  )
  if normalized_skill_lifecycle is not None:
    correlation[REQUIRED_SKILL_LIFECYCLE_METADATA_KEY] = (
      normalized_skill_lifecycle
    )
  normalized_workflow_task = normalize_workflow_task_metadata(
    workflow_task_metadata
  )
  if normalized_workflow_task is not None:
    correlation[WORKFLOW_TASK_METADATA_KEY] = (
      normalized_workflow_task.payload()
    )
  if "resumable" in tool_input:
    correlation["resumable"] = bool(tool_input.get("resumable"))
  return correlation


def prepare_background_task_registration(
  entry: Any,
  *,
  tool_input: Dict[str, Any],
  capability_bind_receipt: dict[str, str],
  owner_runner_id: str | None,
  owner_role: str,
  sub_agent_id_for_call_index: Callable[[int], str],
  parent_turn_id: str | None,
  original_task_id: str | None,
  required_skill_lifecycle: Mapping[str, Any] | None = None,
  workflow_task_metadata: WorkflowTaskMetadata | None = None,
) -> int:
  entry.capability_bind_receipt = require_capability_bind_receipt(
    capability_bind_receipt
  )
  cost_observation_threshold_usd = parse_cost_observation_threshold_usd(
    tool_input.get("cost_observation_threshold_usd")
  )
  call_index = background_task_call_index(entry.task_id)
  sub_agent_id = sub_agent_id_for_call_index(call_index)
  entry.metadata.update(
    background_task_registration_metadata(
      owner_runner_id=owner_runner_id,
      owner_role=owner_role,
      sub_agent_id=sub_agent_id,
      parent_turn_id=parent_turn_id,
      call_index=call_index,
      capability_bind_receipt=entry.capability_bind_receipt,
      cost_observation_threshold_usd=cost_observation_threshold_usd,
      original_task_id=original_task_id,
      tool_input=tool_input,
      required_skill_lifecycle=required_skill_lifecycle,
      workflow_task_metadata=workflow_task_metadata,
    )
  )
  return call_index


def normalize_required_skill_lifecycle(
  value: Mapping[str, Any] | None,
) -> Dict[str, Any] | None:
  """Validate the server-owned descriptor used to recover skill settlement."""

  if value is None:
    return None
  if not isinstance(value, Mapping):
    raise ValueError("required skill lifecycle metadata must be an object")
  lifecycle_fields = {
    "schema_version",
    "skill_run_id",
    "skill",
    "scope",
    "ticker",
    "portfolio_id",
  }
  if set(value) != lifecycle_fields:
    raise ValueError(
      "required skill lifecycle metadata must use the exact v2 schema"
    )
  schema_version = value.get("schema_version")
  if (
    type(schema_version) is not int
    or schema_version != _REQUIRED_SKILL_LIFECYCLE_SCHEMA_VERSION
  ):
    raise ValueError(
      "required skill lifecycle schema_version must be "
      f"{_REQUIRED_SKILL_LIFECYCLE_SCHEMA_VERSION}"
    )

  skill_run_id = value.get("skill_run_id")
  if (
    not isinstance(skill_run_id, str)
    or not skill_run_id
    or skill_run_id != skill_run_id.strip()
  ):
    raise ValueError(
      "required skill lifecycle skill_run_id must be a non-empty "
      "string without surrounding whitespace"
    )
  if len(skill_run_id) > _REQUIRED_SKILL_LIFECYCLE_ID_MAX_CHARS:
    raise ValueError("required skill lifecycle skill_run_id is too long")

  skill = value.get("skill")
  if (
    not isinstance(skill, str)
    or not skill
    or skill != skill.strip()
  ):
    raise ValueError(
      "required skill lifecycle skill must be a non-empty string "
      "without surrounding whitespace"
    )
  if len(skill) > _REQUIRED_SKILL_LIFECYCLE_SKILL_MAX_CHARS:
    raise ValueError("required skill lifecycle skill is too long")

  raw_ticker = value.get("ticker")
  if raw_ticker is None:
    ticker = None
  elif (
    isinstance(raw_ticker, str)
    and raw_ticker
    and raw_ticker == raw_ticker.strip()
  ):
    ticker = raw_ticker
    if len(ticker) > _REQUIRED_SKILL_LIFECYCLE_TICKER_MAX_CHARS:
      raise ValueError("required skill lifecycle ticker is too long")
  else:
    raise ValueError(
      "required skill lifecycle ticker must be null or a non-empty "
      "string without surrounding whitespace"
    )

  scope = value.get("scope")
  portfolio_id = value.get("portfolio_id")
  if type(scope) is not str:
    raise ValueError(
      "required skill lifecycle scope must be exactly 'ticker' or "
      "'portfolio'"
    )
  if (
    portfolio_id is not None
    and (
      not isinstance(portfolio_id, str)
      or not portfolio_id
      or portfolio_id != portfolio_id.strip()
    )
  ):
    raise ValueError(
      "required skill lifecycle portfolio_id must be null or a "
      "non-empty string without surrounding whitespace"
    )
  if (
    isinstance(portfolio_id, str)
    and len(portfolio_id)
    > _REQUIRED_SKILL_LIFECYCLE_PORTFOLIO_ID_MAX_CHARS
  ):
    raise ValueError("required skill lifecycle portfolio_id is too long")
  try:
    artifact_identity = SkillLifecycleArtifactIdentity(
      scope=scope,
      ticker=ticker,
      portfolio_id=portfolio_id,
    )
    lifecycle = TopLevelSkillLifecycleMetadata(
      skill_run_id=skill_run_id,
      skill=skill,
      **artifact_identity.identity_fields(),
    )
  except ValueError as exc:
    raise ValueError(
      f"required skill lifecycle identity is invalid: {exc}"
    ) from exc
  return {
    "schema_version": _REQUIRED_SKILL_LIFECYCLE_SCHEMA_VERSION,
    **lifecycle.identity_fields(),
  }
