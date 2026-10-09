from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING

from agent_workflow_contracts import (
  AgentOperationRef,
  AttemptRef,
  LogicalTaskRef,
  OrdinaryDelegationTaskRef,
  OutcomeRequirement,
  ResultRequirement,
  TaskResult,
  TaskResultProvenance,
  sha256_digest,
)

from .capability_execution import BoundCapabilityExecution
from model_authority.bind import CapabilityBind
from .event_log import EventLog
from .fork_ledger import ForkLedger
from .fork_request_handoff import ForkRequestHandoff
from .fork_task_registry import ForkTaskRegistry
from .runner_cleanup import cleanup_failure_notes
from .runner_introspection import derive_sub_agent_id
from .runner_session_lifecycle import _runner_attr
from .runner_budget import CostAccumulator
from .runner_sub_agents import (
  _ChildActivity,
  _await_child_under_activity_guard,
  _close_sub_runner,
  _runtime_exception_detail,
)
from .sub_agent_narrative_result import (
  final_child_visible_text,
  task_result_from_execution,
)
from .task_registry import TaskEntry
from .tool_dispatcher_helpers import LocalToolHandler
from .transcript import (
  build_synthetic_tool_results,
  detect_orphan_tool_uses,
)

if TYPE_CHECKING:
  from .context_builder import Message
  from .runner import AgentRunner
  from .runner_tool_execution import AgentRunnerDispatcher


log = logging.getLogger("agent_gateway.runner")
FORK_PLACEHOLDER = "Fork started — processing continues in the parent."
FORK_MARKER_EVENT_FIELD = "fork"
_LEARNING_FORK_OPERATION = AgentOperationRef(
  namespace="agent-operation",
  name="learning-fork",
  version="1.0",
  digest=sha256_digest({
    "namespace": "agent-operation",
    "name": "learning-fork",
    "version": "1.0",
    "execution_class": "fork",
  }),
)


class ForkMemoryWriteScope(Protocol):
  def __call__(
    self,
    handler: LocalToolHandler,
    *,
    fork_id: str,
    user_id: str,
  ) -> LocalToolHandler: ...


@dataclass(slots=True)
class LearningForkConfig:
  """Application-owned learning inputs and their lazy shared runtime."""

  directive: str
  scope_memory_write: ForkMemoryWriteScope
  ledger_path: Callable[[str], Path]
  _ledger: ForkLedger | None = field(default=None, init=False, repr=False)
  _registry: ForkTaskRegistry | None = field(default=None, init=False, repr=False)


@dataclass(slots=True)
class LearningForkWorkItem:
  """Runtime-only owner plus the immutable handoff retained by the registry."""

  parent: AgentRunner
  handoff: ForkRequestHandoff
  config: LearningForkConfig
  ledger: ForkLedger
  session_id: str
  owner: str
  user_id: str
  fork_events: Sequence[Any] = ()

  @property
  def _fork_retained_payload(self) -> Mapping[str, Any]:
    return self.handoff._fork_retained_payload

  def on_admission_settled(self, fork_id: str) -> None:
    self.ledger.write_receipt(
      fork_id=fork_id,
      session_id=self.session_id,
      owner=self.owner,
      receipt_text=_learning_receipt_text(self.fork_events),
    )


def _bind_learning_fork_execution(
  parent_execution: BoundCapabilityExecution,
) -> BoundCapabilityExecution:
  parent_execution.validate()
  parent = parent_execution.bind
  return BoundCapabilityExecution(
    bind=CapabilityBind.model_validate({
      **parent.to_json(),
      "capability_id": "node.fork",
      "selection_source": "parent_binding",
    }),
    registry=parent_execution.registry,
    adapter=parent_execution.adapter,
    auth_config=parent_execution.auth_config,
    resolver=parent_execution.resolver,
  )


def cross_check_learning_memory_writes(
  payload: Mapping[str, Any],
  event_entries: Sequence[Any],
) -> dict[str, Any]:
  """Record verified-not-trusted memory-write mismatches as report caveats."""

  checked = copy.deepcopy(dict(payload))
  claimed = {
    str(item.get("path") or "").strip()
    for item in checked.get("memory_writes", ())
    if isinstance(item, Mapping) and str(item.get("path") or "").strip()
  }
  observed: set[str] = set()
  for raw in event_entries:
    event = raw.event if hasattr(raw, "event") else raw
    if not isinstance(event, Mapping):
      continue
    result = event.get("result")
    if (
      event.get("type") == "tool_call_complete"
      and event.get("tool_name") == "memory_write"
      and not bool(event.get("is_error"))
      and event.get("error") is None
      and isinstance(result, Mapping)
      and str(result.get("file") or "").strip()
    ):
      observed.add(str(result["file"]).strip())
  if claimed == observed:
    return checked

  missing_count = len(claimed - observed)
  unclaimed_count = len(observed - claimed)
  caveat = (
    "memory_writes event-log mismatch: "
    f"{missing_count} claimed without evidence; "
    f"{unclaimed_count} observed but unclaimed"
  )
  caveats = [
    str(value)
    for value in checked.get("caveats", ())
    if isinstance(value, str)
  ]
  if caveat not in caveats:
    if len(caveats) < 20:
      caveats.append(caveat)
    elif caveats:
      caveats[-1] = caveat
  checked["caveats"] = caveats
  return checked


def _tool_use_ids(message: Mapping[str, Any]) -> list[str]:
  content = message.get("content")
  if not isinstance(content, list):
    return []
  return [
    str(block["id"])
    for block in content
    if (
      isinstance(block, Mapping)
      and block.get("type") == "tool_use"
      and isinstance(block.get("id"), str)
      and block["id"]
    )
  ]


def _tool_result_ids(message: Mapping[str, Any]) -> set[str]:
  content = message.get("content")
  if message.get("role") != "user" or not isinstance(content, list):
    return set()
  return {
    str(block["tool_use_id"])
    for block in content
    if (
      isinstance(block, Mapping)
      and block.get("type") == "tool_result"
      and isinstance(block.get("tool_use_id"), str)
    )
  }


def _sanitize_earlier_orphans(
  messages: Sequence[Message],
  *,
  preserve_last: bool,
  marker_message_index: int,
) -> tuple[list[dict[str, Any]], int]:
  source = copy.deepcopy(list(messages))
  sanitized: list[dict[str, Any]] = []
  inserted_before_marker = 0
  for index, message in enumerate(source):
    sanitized.append(message)
    if preserve_last and index == len(source) - 1:
      continue
    tool_use_ids = _tool_use_ids(message)
    if not tool_use_ids:
      continue
    next_results = (
      _tool_result_ids(source[index + 1])
      if index + 1 < len(source)
      else set()
    )
    if all(tool_use_id in next_results for tool_use_id in tool_use_ids):
      continue
    orphan_ids = detect_orphan_tool_uses([message])
    missing = [
      tool_use_id
      for tool_use_id in orphan_ids
      if tool_use_id not in next_results
    ]
    if missing:
      synthetic = build_synthetic_tool_results(missing)
      if (
        index + 1 < len(source)
        and source[index + 1].get("role") == "user"
      ):
        next_content = source[index + 1].get("content")
        if isinstance(next_content, list):
          source[index + 1]["content"] = synthetic + next_content
        elif isinstance(next_content, str):
          source[index + 1]["content"] = [
            *synthetic,
            {"type": "text", "text": next_content},
          ]
        else:
          source[index + 1]["content"] = synthetic
      else:
        sanitized.append({
          "role": "user",
          "content": synthetic,
        })
        if index < marker_message_index:
          inserted_before_marker += 1
  return sanitized, marker_message_index + inserted_before_marker


def fork_boilerplate(
  directive: str,
) -> str:
  normalized = str(directive or "").strip()
  if not normalized:
    raise ValueError("fork directive must be non-empty")
  return (
    "You are a side-quest fork of the parent conversation. Preserve the "
    "inherited context and stay concise. Finish with a normal assistant "
    "message containing the complete result. The runtime owns result typing "
    "and persistence.\n\n"
    f"Fork directive:\n{normalized}"
  )


def build_fork_messages(
  handoff: ForkRequestHandoff,
  directive: str,
) -> list[dict[str, Any]]:
  return _build_fork_context(handoff, directive)[0]


def _build_fork_context(
  handoff: ForkRequestHandoff,
  directive: str,
) -> tuple[list[dict[str, Any]], tuple[int, int]]:
  preserve_last = handoff.boundary_kind == "mid_turn"
  messages, marker_message_index = _sanitize_earlier_orphans(
    handoff.messages,
    preserve_last=preserve_last,
    marker_message_index=handoff.message_marker_position[0],
  )
  marker_position = (
    marker_message_index,
    handoff.message_marker_position[1],
  )
  boilerplate = fork_boilerplate(directive)
  if handoff.boundary_kind == "post_turn":
    messages.append({"role": "user", "content": boilerplate})
    return messages, marker_position

  if not messages or messages[-1].get("role") != "assistant":
    raise ValueError(
      "mid-turn fork handoff must end with an assistant tool-use message"
    )
  dangling_ids = detect_orphan_tool_uses(messages)
  if not dangling_ids:
    raise ValueError("mid-turn fork handoff has no dangling tool calls")
  content = [
    {
      "type": "tool_result",
      "tool_use_id": tool_use_id,
      "content": FORK_PLACEHOLDER,
    }
    for tool_use_id in dangling_ids
  ]
  content.append({"type": "text", "text": boilerplate})
  messages.append({"role": "user", "content": content})
  return messages, marker_position


def fork_suffix_messages(
  messages: Sequence[Message],
  marker_position: tuple[int, int],
) -> list[dict[str, Any]]:
  message_index, block_index = marker_position
  if message_index >= len(messages):
    raise ValueError("fork marker message index is outside child history")
  suffix: list[dict[str, Any]] = []
  boundary = copy.deepcopy(dict(messages[message_index]))
  content = boundary.get("content")
  if isinstance(content, list):
    boundary["content"] = content[block_index + 1:]
    if boundary["content"]:
      suffix.append(boundary)
  suffix.extend(copy.deepcopy(list(messages[message_index + 1:])))
  return suffix


def _fork_event_log(
  parent: Any,
  *,
  sub_session_id: str,
  activity: _ChildActivity,
) -> EventLog:
  parent_log = parent._log
  original_on_event = getattr(parent_log, "_on_event", None)

  def prepare_event(event: dict[str, Any]) -> dict[str, Any]:
    prepared = copy.deepcopy(event)
    prepared["sub_agent_id"] = sub_session_id
    prepared[FORK_MARKER_EVENT_FIELD] = True
    return prepared

  def on_event(event: dict[str, Any], session_id: str) -> None:
    activity.observe(event)
    if original_on_event is not None:
      original_on_event(event, session_id)

  return EventLog(
    prepare_event=prepare_event,
    on_event=on_event,
    on_event_error=getattr(parent_log, "_on_event_error", "ignore"),
    session_id=sub_session_id,
  )


async def spawn_fork_agent(
  parent: Any,
  directive: str,
  *,
  handoff: ForkRequestHandoff,
  capability_execution: BoundCapabilityExecution,
  logical_task: LogicalTaskRef,
  attempt: AttemptRef,
  result_requirement: ResultRequirement,
  result_provenance: TaskResultProvenance,
  dispatcher: AgentRunnerDispatcher,
  local_handler_scopes: (
    Mapping[str, Callable[[LocalToolHandler], LocalToolHandler]] | None
  ) = None,
  call_index: int = 0,
  parent_turn_id: str | None = None,
  task_entry: TaskEntry | None = None,
  event_log_observer: Callable[[EventLog], None] | None = None,
  child_cost_observer: Callable[[float], None] | None = None,
  independent_accounting: bool = False,
) -> tuple[TaskResult | None, dict[str, Any] | None]:
  if not isinstance(handoff, ForkRequestHandoff):
    raise TypeError("spawn_fork_agent requires a ForkRequestHandoff")
  if not isinstance(capability_execution, BoundCapabilityExecution):
    raise TypeError("spawn_fork_agent requires BoundCapabilityExecution")
  capability_execution.validate()
  if not isinstance(attempt, AttemptRef):
    raise TypeError("spawn_fork_agent requires an exact AttemptRef")
  if not isinstance(result_requirement, ResultRequirement):
    raise TypeError("spawn_fork_agent requires a ResultRequirement")
  if not isinstance(result_provenance, TaskResultProvenance):
    raise TypeError("spawn_fork_agent requires exact admitted provenance")
  if task_entry is not None and task_entry.admitted_task is not None:
    admitted = task_entry.admitted_task
    if (
      admitted.logical_task != logical_task
      or admitted.attempt != attempt
      or admitted.result_requirement != result_requirement
      or admitted.admitted_task_digest
      != result_provenance.admitted_task_digest
      or admitted.model_bind_digest != result_provenance.model_bind_digest
      or admitted.capability_binding_digest
      != result_provenance.capability_binding_digest
      or admitted.tool_grant_digest != result_provenance.tool_grant_digest
    ):
      raise ValueError("spawn_fork_agent identity differs from admitted task")
  sub_session_id = _runner_attr(
    parent,
    "_derive_sub_agent_id",
    derive_sub_agent_id,
  )(
    parent._full_session_id,
    call_index,
  )
  activity = _ChildActivity()
  sub_log = _fork_event_log(
    parent,
    sub_session_id=sub_session_id,
    activity=activity,
  )
  if event_log_observer is not None:
    event_log_observer(sub_log)
  child_dispatcher = dispatcher.fork_child(
    event_log=sub_log,
    session_id=sub_session_id,
    capability_bind=capability_execution.bind,
    spawning_turn=handoff.spawning_turn,
    local_handler_scopes=local_handler_scopes,
  )
  wire_tools = copy.deepcopy(list(handoff.wire_tools))

  def child_get_tool_definitions() -> list[dict[str, Any]]:
    return copy.deepcopy(wire_tools)

  child_accumulator = CostAccumulator()
  sub_runner = type(parent)(
    event_log=sub_log,
    dispatcher=child_dispatcher,
    session_id=sub_session_id,
    capability_execution=capability_execution,
    max_tokens_override=handoff.max_tokens,
    stream_stall_timeout=parent._stream_stall_timeout,
    mcp_client=parent._mcp_client,
    mcp_activation_fold=parent._mcp_activation_fold,
    excluded_tools=set(),
    get_tool_definitions=child_get_tool_definitions,
    on_tool_result=parent._on_tool_result,
    on_usage=parent._on_usage,
    on_session_summary=None,
    on_late_usage_event=parent._on_late_usage_event,
    on_tool_timing=parent._on_tool_timing,
    user_id=parent._usage_user_id,
    request_id=parent._request_id,
    parent_turn_id=parent_turn_id,
    billing_mode=handoff.billing_mode,
    rate_table_version=parent._rate_table_version,
    channel=parent._channel,
    usage_ledger_dlq_path=parent._usage_ledger_dlq_path,
    on_metric=parent._on_metric,
    sub_agent_config=None,
    compaction_trigger=parent._compaction_trigger,
    compaction_instructions=None,
    tool_call_timeout=parent._tool_call_timeout,
    _cost_accumulator=child_accumulator,
    _parent_aggregator=(
      None if independent_accounting else parent._aggregator
    ),
    max_concurrent_sub_agents=parent._max_concurrent_sub_agents,
    result_requirement=result_requirement,
    agent_session_log=parent._agent_session_log,
    message_inbox=task_entry.message_inbox if task_entry else None,
    max_resume_chain_depth=parent._max_resume_chain_depth,
    emit_session_recap=False,
    code_execution_spill_dir_provider=parent._spill_dir_provider,
    commercial_usage_producer=getattr(
      parent,
      "_commercial_usage_producer",
      None,
    ),
    workspace_dir=parent._workspace_dir,
    batch_id=getattr(parent, "_batch_id", None),
    context_surfaces=(
      parent._context_surfaces_provider
      or parent._context_surfaces_static
    ),
  )
  sub_runner._fork_mode = True
  sub_runner._soft_budget_kind = "fork"
  sub_runner._fork_suffix_tokens = 0
  sub_runner._tenant_id = handoff.tenant_id
  child_messages, child_marker_position = _build_fork_context(handoff, directive)
  sub_runner._fork_marker_position = child_marker_position

  stalled = False
  runtime_error_detail: str | None = None
  cancelled_error: asyncio.CancelledError | None = None
  cancellation_signal: str | None = None
  cleanup_warnings: list[str] = []
  run_coro = sub_runner.run(
    messages=child_messages[-1:],
    system_prompt=list(handoff.rendered_system_blocks),
    resume_initial_messages=child_messages,
  )
  try:
    silence = await _await_child_under_activity_guard(run_coro, activity)
    if silence is not None:
      stalled = True
      detail = f"Forked sub-agent stalled: no activity for {silence:.0f}s"
      log.warning("[%s] %s", sub_session_id, detail)
      sub_log.append({
        "type": "error",
        "error": detail,
        "error_sub_code": "stalled",
      })
  except asyncio.CancelledError as exc:
    cancelled_error = exc
    cancellation_signal = (
      task_entry.termination_intent
      if task_entry is not None and task_entry.termination_intent
      else "cancelled"
    )
    cleanup_warnings.extend(cleanup_failure_notes(exc))
    sub_log.append({"type": "error", "error": "Forked sub-agent cancelled"})
  except Exception as exc:
    runtime_error_detail = _runtime_exception_detail(exc)
    cleanup_warnings.extend(cleanup_failure_notes(exc))
    sub_log.append({"type": "error", "error": runtime_error_detail})
  finally:
    (
      cancelled_error,
      cancellation_signal,
      runtime_error_detail,
      cleanup_warnings,
    ) = await _close_sub_runner(
      sub_runner,
      sub_log,
      stalled=stalled,
      cancelled_error=cancelled_error,
      cancellation_signal=cancellation_signal,
      runtime_exception_detail=runtime_error_detail,
      task_entry=task_entry,
      cleanup_warnings=cleanup_warnings,
    )
    # Measurements, never stops: the fork's log is closed by its terminal
    # event, so they ride the structured run log beside chat_done.
    log.info(
      "fork measurements",
      extra={
        "data": {
          "event": "fork_measurements",
          "sub_agent_id": sub_session_id,
          "turns": _count_events(sub_log.entries, "turn_complete"),
          "suffix_tokens": sub_runner._fork_suffix_tokens,
        }
      },
    )
    if child_cost_observer is not None:
      child_cost_observer(child_accumulator.total)

  signals = []
  if cancellation_signal in {"cancelled", "killed"}:
    signals.append(cancellation_signal)
  sub_runner_id = getattr(sub_runner, "_runner_id", None)
  if not isinstance(sub_runner_id, str) or not sub_runner_id:
    raise RuntimeError("fork completion requires its exact durable runner_id")
  narrative_text = await final_child_visible_text(
    parent._agent_session_log,
    sub_session_id=sub_session_id,
    workspace_dir=parent._workspace_dir,
    runner_id=sub_runner_id,
  )
  result = task_result_from_execution(
    sub_log.entries,
    logical_task=logical_task,
    attempt=attempt,
    requirement=result_requirement,
    provenance=result_provenance,
    final_narrative=narrative_text.final_narrative,
    stalled=stalled,
    runtime_error_detail=runtime_error_detail,
    external_terminal_signals=signals,
    # B-3: the authority frozen at admission, never the ambient catalog.
    admitted_task=(
      task_entry.admitted_task if task_entry is not None else None
    ),
  )
  if cancelled_error is not None:
    if task_entry is not None:
      task_entry.task_result = result
      task_entry.result = result.model_dump(mode="json")
    raise cancelled_error
  return result, None


def _count_events(event_entries: Sequence[Any], event_type: str) -> int:
  count = 0
  for raw in event_entries:
    event = raw.event if hasattr(raw, "event") else raw
    if isinstance(event, Mapping) and event.get("type") == event_type:
      count += 1
  return count


def _learning_receipt_text(event_entries: Sequence[Any]) -> str:
  """Summarize deterministic tool effects, never the model's prose shape."""

  observed = 0
  for raw in event_entries:
    event = raw.event if hasattr(raw, "event") else raw
    if not isinstance(event, Mapping):
      continue
    if (
      event.get("type") == "tool_call_complete"
      and event.get("tool_name") == "memory_write"
      and not bool(event.get("is_error"))
      and event.get("error") is None
    ):
      observed += 1
  note_part = f"{observed} memory note" + ("" if observed == 1 else "s")
  return f"Self-learning fork completed: {note_part} written"


def learning_fork_result_provenance(
  *,
  physical_task_id: str,
  logical_task: OrdinaryDelegationTaskRef,
  attempt: AttemptRef,
  objective: str,
  result_requirement: ResultRequirement,
  execution: BoundCapabilityExecution,
  wire_tools: Sequence[Mapping[str, Any]],
) -> TaskResultProvenance:
  """Derive fork provenance from the canonical full-payload admission digest.

  ``admitted_task_digest`` has exactly one definition — the seal computed by
  :func:`agent_gateway.sub_agent.seal_admitted_task_payload` over the complete
  admission payload (component digests included). The learning fork's
  admission artifacts are its logical task, attempt, objective, operation ref,
  result requirement, inherited model bind, and its tool grant — the parent's
  exact wire tool surface; the digest covers all of them rather than a bespoke
  field subset.
  """
  from .sub_agent import seal_admitted_task_payload

  model_bind_digest = sha256_digest(execution.bind)
  capability_binding_digest = sha256_digest({
    "kind": "dynamic_parent_bind",
    "bind": execution.bind.model_dump(mode="json"),
  })
  tool_grant = {
    "kind": "parent_wire_surface",
    "tools": [str(definition.get("name") or "") for definition in wire_tools],
  }
  tool_grant_digest = sha256_digest(tool_grant)
  admission_payload = seal_admitted_task_payload({
    "schema_version": "1.0",
    "admitted_task_id": f"admitted:{physical_task_id}",
    "logical_task": logical_task.model_dump(mode="json"),
    "attempt": attempt.model_dump(mode="json"),
    "objective": objective,
    "operation": _LEARNING_FORK_OPERATION.model_dump(mode="json"),
    "result_requirement": result_requirement.model_dump(mode="json"),
    "model_bind": execution.bind.model_dump(mode="json"),
    "tool_grant": tool_grant,
    "model_bind_digest": model_bind_digest,
    "capability_binding_digest": capability_binding_digest,
    "tool_grant_digest": tool_grant_digest,
  })
  return TaskResultProvenance(
    admitted_task_digest=admission_payload["admitted_task_digest"],
    model_bind_digest=model_bind_digest,
    capability_binding_digest=capability_binding_digest,
    tool_grant_digest=tool_grant_digest,
  )


async def spawn_learning_fork(
  fork_id: str,
  raw_work_item: Any,
) -> Decimal:
  """Run one post-turn learning fork through the F1 fork machinery."""

  if not isinstance(raw_work_item, LearningForkWorkItem):
    raise TypeError("learning fork requires LearningForkWorkItem")
  work_item = raw_work_item
  parent = work_item.parent
  handoff = work_item.handoff
  execution = _bind_learning_fork_execution(
    parent._capability_execution
  )
  call_index = int(fork_id.rsplit("-", 1)[-1][:8], 16)
  physical_task_id = _runner_attr(
    parent,
    "_derive_sub_agent_id",
    derive_sub_agent_id,
  )(
    parent._full_session_id,
    call_index,
  )
  attempt = AttemptRef(
    attempt_number=1,
    attempt_id=f"{fork_id}:attempt:1",
    physical_task_id=physical_task_id,
  )
  logical_task = OrdinaryDelegationTaskRef(
    delegation_id=fork_id,
    operation=_LEARNING_FORK_OPERATION,
  )
  result_requirement = ResultRequirement(
    mode="narrative",
    projection=None,
    terminal_narrative="required",
    outcome=OutcomeRequirement(required=False, source="none"),
  )
  event_log_box: list[EventLog] = []
  result_provenance = learning_fork_result_provenance(
    physical_task_id=physical_task_id,
    logical_task=logical_task,
    attempt=attempt,
    objective=work_item.config.directive,
    result_requirement=result_requirement,
    execution=execution,
    wire_tools=handoff.wire_tools,
  )
  child_cost: list[float] = []
  result, error = await spawn_fork_agent(
    parent,
    work_item.config.directive,
    handoff=handoff,
    capability_execution=execution,
    logical_task=logical_task,
    attempt=attempt,
    result_requirement=result_requirement,
    result_provenance=result_provenance,
    dispatcher=parent._dispatcher,
    local_handler_scopes={
      "memory_write": lambda stock: work_item.config.scope_memory_write(
        stock,
        fork_id=fork_id,
        user_id=work_item.user_id,
      ),
    },
    call_index=call_index,
    parent_turn_id=str(getattr(parent, "_request_id", "") or ""),
    event_log_observer=event_log_box.append,
    child_cost_observer=child_cost.append,
    independent_accounting=True,
  )
  if error is not None or result is None or result.execution.status != "succeeded":
    raise RuntimeError("learning fork did not complete successfully")
  work_item.fork_events = event_log_box[0].entries if event_log_box else ()
  return Decimal(str(child_cost[-1] if child_cost else 0))


__all__ = [
  "FORK_PLACEHOLDER",
  "LearningForkConfig",
  "LearningForkWorkItem",
  "build_fork_messages",
  "cross_check_learning_memory_writes",
  "fork_suffix_messages",
  "learning_fork_result_provenance",
  "spawn_fork_agent",
  "spawn_learning_fork",
]
