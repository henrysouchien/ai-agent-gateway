from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, TYPE_CHECKING, cast

from agent_workflow_contracts import (
  AgentOperationSnapshot,
  AdmittedTask,
  AttemptRef,
  LogicalTaskRef,
  OperationUnavailable,
  OrdinaryDelegationTaskRef,
  OutcomeRequirement,
  PlatformToolCatalog,
  ResolvedAuthority,
  ResultRequirement,
  TaskResult,
  SemanticCapabilityRequirement,
  TaskResultProvenance,
  terminal_task_result,
)

from .capability_execution import (
  BoundCapabilityExecution,
  CapabilityExecutionResolver,
  materialize_frozen_bind,
)
from .admission import (
  AdmissionOwners,
  AdmittedContext,
  OperationRegistration,
  Refusal,
  RootTurnAdmission,
  SpawnRequest,
  admit,
  capture_job_context,
  context_for_child,
  current_parent_context,
  current_root_turn,
)
from .approval_route import ApprovalRoute, child_approval_route
from .capability_resolution import (
  OperationDeclaration,
  admitted_tool_routes,
  inherited_operation_declaration,
  resolve_operation_authority,
  snapshot_platform_catalog,
)
from .execution_identity import (
  DispatchIdentity,
  dispatch_identity,
  execution_identity_from_session,
)
from .execution_snapshot import (
  build_agent_execution_snapshot,
  provider_tool_definitions_for_grant,
  render_result_instructions,
)
from .event_log import EventLog
from .fork_ledger import ForkLedger
from .fork_request_handoff import ForkRequestHandoff
from .fork_task_registry import ForkTaskRegistry
from .job_caller import JobScope, JobScopedHandler
from .runner_cleanup import cleanup_failure_notes
from .operation_snapshot import WorkspaceScope
from .runner_introspection import derive_sub_agent_id
from .runner_session_lifecycle import _runner_attr
from .runner_budget import CostAccumulator
from .runner_sub_agents import (
  _ChildActivity,
  _await_child_under_activity_guard,
  _bind_child_registry,
  _close_sub_runner,
  _runtime_exception_detail,
)
from .session import GatewaySession, child_gateway_session
from .sub_agent_narrative_result import (
  final_child_visible_text,
  task_result_from_execution,
)
from .task_registry import RegistryOwner, TaskEntry, TaskRegistry
from .sub_agent_result_contract import _usage_observation
from .sub_agent_result_evidence import collect_sub_agent_result_evidence
from .tool_dispatcher import _SpawningTurn
from .tool_dispatcher_helpers import LocalToolHandler
from .tool_result_spill import job_spill_sink
from .tool_dispatcher_approval_lifecycle import ApprovalPark
from .transcript import (
  build_synthetic_tool_results,
  detect_orphan_tool_uses,
)

if TYPE_CHECKING:
  from .context_builder import Message
  from .operation_catalog import AgentOperationCatalog
  from .runner import AgentRunner
  from .runner_tool_execution import AgentRunnerDispatcher
  from .agent_session_log import AgentSessionLog


log = logging.getLogger("agent_gateway.runner")
FORK_PLACEHOLDER = "Fork started — processing continues in the parent."
FORK_MARKER_EVENT_FIELD = "fork"


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
  operation_snapshot_builder: Callable[
    [str, tuple[SemanticCapabilityRequirement, ...], WorkspaceScope],
    AgentOperationSnapshot,
  ]
  child_session_log_builder: Callable[[GatewaySession], AgentSessionLog]
  # Read-only: a fork id locates that fork's own journal, whoever owns it.
  fork_journal_reader: Callable[[str], AgentSessionLog | None]
  _ledger: ForkLedger | None = field(default=None, init=False, repr=False)
  _registry: ForkTaskRegistry | None = field(default=None, init=False, repr=False)
  # Post-turn forks outlive a closed foreground log; the next turn drains
  # only its session's measurements through the ordinary event writer.
  _pending_events: dict[str, list[dict[str, Any]]] = field(
    default_factory=dict, init=False, repr=False,
  )


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


@dataclass(frozen=True, slots=True)
class _ForkChildClosure:
  """How one fork child's run ended, after its runner closed."""

  stalled: bool
  approval_park: ApprovalPark | None
  cancelled_error: asyncio.CancelledError | None
  cancellation_signal: str | None
  runtime_error_detail: str | None


@dataclass(frozen=True, slots=True)
class _ForkChild:
  """One constructed fork child runner, its event log and its run."""

  runner: Any
  log: EventLog
  session_id: str
  run: Callable[[], Awaitable[_ForkChildClosure]]


@dataclass(frozen=True, slots=True)
class _ForkRunner:
  """One constructed fork runner and the event log its activity rides."""

  runner: Any
  log: EventLog
  session_id: str
  activity: _ChildActivity
  accumulator: CostAccumulator


def _construct_fork_runner(
  parent: Any,
  *,
  spawning_turn: _SpawningTurn,
  max_tokens: int,
  billing_mode: str,
  tenant_id: str | None,
  capability_execution: BoundCapabilityExecution,
  attempt: AttemptRef,
  result_requirement: ResultRequirement,
  dispatcher: AgentRunnerDispatcher,
  sub_session: Any | None,
  identity: DispatchIdentity | None,
  approval_route: ApprovalRoute | None,
  local_handler_scopes: (
    Mapping[str, Callable[[LocalToolHandler], LocalToolHandler]] | None
  ),
  call_index: int,
  parent_turn_id: str | None,
  task_entry: TaskEntry,
  tool_definitions: Sequence[Mapping[str, Any]],
  agent_session_log: AgentSessionLog | None,
  admitted_self_task: TaskEntry | None,
  independent_accounting: bool,
) -> _ForkRunner:
  """Construct a fork's runner from the parent runner's process wiring."""

  sub_session_id = (
    sub_session.session_id
    if sub_session is not None
    else _runner_attr(parent, "_derive_sub_agent_id", derive_sub_agent_id)(
      parent._full_session_id, call_index,
    )
  )
  child_context = (
    current_parent_context(sub_session) if sub_session is not None else None
  )
  child_spill_sink = (
    job_spill_sink(parent._spill_dir_provider, attempt.physical_task_id)
    if parent._spill_dir_provider is not None else None
  )
  child_runner_ref: list[Any] = []
  handler_scopes = dict(local_handler_scopes or {})
  if child_context is not None:
    child_scope = JobScope(
      caller=lambda: child_context,
      spill_sink=lambda: child_spill_sink,
      runner=lambda: child_runner_ref[0],
      session=sub_session,
    )

    def job_handler_scope(
      scope: Callable[[LocalToolHandler], LocalToolHandler] | None,
    ) -> Callable[[LocalToolHandler], LocalToolHandler]:
      def rebind(stock: LocalToolHandler) -> LocalToolHandler:
        rebound = cast(JobScopedHandler, stock).for_job(child_scope)
        return scope(rebound) if scope is not None else rebound
      return rebind

    for name, handler in dispatcher.local_tool_handlers().items():
      if isinstance(handler, JobScopedHandler):
        handler_scopes[name] = job_handler_scope(handler_scopes.get(name))
  activity = _ChildActivity()
  sub_log = _fork_event_log(
    parent,
    sub_session_id=sub_session_id,
    activity=activity,
  )
  child_dispatcher = dispatcher.fork_child(
    event_log=sub_log,
    session_id=sub_session_id,
    capability_bind=capability_execution.bind,
    spawning_turn=spawning_turn,
    identity=identity,
    session=sub_session,
    approval_route=approval_route,
    local_handler_scopes=handler_scopes,
  )
  child_tool_definitions = copy.deepcopy([dict(item) for item in tool_definitions])

  def child_get_tool_definitions() -> list[dict[str, Any]]:
    return copy.deepcopy(child_tool_definitions)

  child_accumulator = CostAccumulator()
  sub_runner = type(parent)(
    event_log=sub_log,
    dispatcher=child_dispatcher,
    session_id=sub_session_id,
    gateway_session=sub_session,
    context_research_file_id=(
      child_context.research_binding.research_file_id
      if child_context is not None else None
    ),
    capability_execution=capability_execution,
    max_tokens_override=max_tokens,
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
    billing_mode=billing_mode,
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
    containment=parent._containment,
    result_requirement=result_requirement,
    agent_session_log=agent_session_log,
    admitted_self_task=admitted_self_task,
    admitted_self_prior_evidence=None,
    message_inbox=task_entry.message_inbox if task_entry else None,
    max_resume_chain_depth=parent._max_resume_chain_depth,
    emit_session_recap=False,
    code_execution_spill_dir_provider=child_spill_sink,
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
  _bind_child_registry(
    sub_runner,
    parent_registry=parent._task_registry,
    attempt=attempt,
    sub_session=sub_session,
    task_entry=task_entry,
    spill_sink=child_spill_sink,
    bind_job_runner=child_runner_ref.append,
  )
  sub_runner._fork_mode = True
  sub_runner._soft_budget_kind = "fork"
  sub_runner._fork_suffix_tokens = 0
  sub_runner._tenant_id = tenant_id
  return _ForkRunner(
    runner=sub_runner, log=sub_log, session_id=sub_session_id,
    activity=activity, accumulator=child_accumulator,
  )


def _publish_fork_measurements(
  parent: Any,
  sub_log: EventLog,
  sub_runner: Any,
  *,
  sub_session_id: str,
  independent_accounting: bool,
) -> None:
  measurements = {
    "event": "fork_measurements",
    "sub_agent_id": sub_session_id,
    "turns": _count_events(sub_log.entries, "turn_complete"),
    "suffix_tokens": sub_runner._fork_suffix_tokens,
  }
  log.info("fork measurements", extra={"data": measurements})
  event = {
    "type": "fork_measurements",
    **{key: value for key, value in measurements.items() if key != "event"},
  }
  parent_session = getattr(parent, "_gateway_session", None)
  learning_config = getattr(parent_session, "learning_fork_config", None)
  if independent_accounting and parent_session is not None and learning_config is not None:
    learning_config._pending_events.setdefault(
      str(parent_session.session_id), [],
    ).append(event)
  else:
    parent._log.append(event)


def _prepare_fork_child(
  parent: Any,
  directive: str,
  *,
  handoff: ForkRequestHandoff,
  capability_execution: BoundCapabilityExecution,
  attempt: AttemptRef,
  result_requirement: ResultRequirement,
  dispatcher: AgentRunnerDispatcher,
  sub_session: Any | None,
  identity: DispatchIdentity | None,
  approval_route: ApprovalRoute | None,
  local_handler_scopes: (
    Mapping[str, Callable[[LocalToolHandler], LocalToolHandler]] | None
  ),
  call_index: int,
  parent_turn_id: str | None,
  task_entry: TaskEntry,
  tool_definitions: Sequence[Mapping[str, Any]],
  agent_session_log: AgentSessionLog | None,
  admitted_self_task: TaskEntry | None,
  child_cost_observer: Callable[[float], None] | None,
  independent_accounting: bool,
) -> _ForkChild:
  """Build the child runner synchronously; its run is awaited by the caller."""

  constructed = _construct_fork_runner(
    parent,
    spawning_turn=handoff.spawning_turn,
    max_tokens=handoff.max_tokens,
    billing_mode=handoff.billing_mode,
    tenant_id=handoff.tenant_id,
    capability_execution=capability_execution,
    attempt=attempt,
    result_requirement=result_requirement,
    dispatcher=dispatcher,
    sub_session=sub_session,
    identity=identity,
    approval_route=approval_route,
    local_handler_scopes=local_handler_scopes,
    call_index=call_index,
    parent_turn_id=parent_turn_id,
    task_entry=task_entry,
    tool_definitions=tool_definitions,
    agent_session_log=agent_session_log,
    admitted_self_task=admitted_self_task,
    independent_accounting=independent_accounting,
  )
  sub_runner = constructed.runner
  sub_log = constructed.log
  sub_session_id = constructed.session_id
  activity = constructed.activity
  child_accumulator = constructed.accumulator
  child_messages, child_marker_position = _build_fork_context(handoff, directive)
  sub_runner._fork_marker_position = child_marker_position

  async def run() -> _ForkChildClosure:
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
      silence = await _await_child_under_activity_guard(
        run_coro, activity,
        silence_timeout=parent._containment.child_silence_timeout_seconds,
      )
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
    except ApprovalPark:
      raise
    except Exception as exc:
      runtime_error_detail = _runtime_exception_detail(exc)
      cleanup_warnings.extend(cleanup_failure_notes(exc))
      sub_log.append({"type": "error", "error": runtime_error_detail})
    finally:
      approval_park = sub_runner._approval_park
      (
        cancelled_error,
        cancellation_signal,
        runtime_error_detail,
        cleanup_warnings,
      ) = await _close_sub_runner(
        sub_runner,
        sub_log,
        stalled=stalled,
        approval_park=approval_park,
        cancelled_error=cancelled_error,
        cancellation_signal=cancellation_signal,
        runtime_exception_detail=runtime_error_detail,
        task_entry=task_entry,
        cleanup_warnings=cleanup_warnings,
        sub_session=sub_session,
      )
      _publish_fork_measurements(
        parent,
        sub_log,
        sub_runner,
        sub_session_id=sub_session_id,
        independent_accounting=independent_accounting,
      )
      if child_cost_observer is not None:
        child_cost_observer(child_accumulator.total)
    return _ForkChildClosure(
      stalled=stalled,
      approval_park=approval_park,
      cancelled_error=cancelled_error,
      cancellation_signal=cancellation_signal,
      runtime_error_detail=runtime_error_detail,
    )

  return _ForkChild(
    runner=sub_runner, log=sub_log, session_id=sub_session_id, run=run,
  )


async def learning_fork_host_runner(
  parent: Any,
  journal: AgentSessionLog,
  *,
  fork_id: str,
) -> Any | None:
  """Rebuild a learning fork's runner on its own journal for a Continue.

  ``spawn_learning_fork`` built this runner when it admitted the fork; once
  that process is gone, a decision on the fork's parked call continues the
  fork from here. The fork's ``AdmittedTask``, as ``journal`` registered it,
  supplies the frozen bind, admitted context, grant and provider tool
  definitions; nothing is re-selected. ``parent`` is the owning session's
  restored runner and supplies process wiring only, as it did at spawn. The
  host never runs and records no root turn; the approved successor runs
  beneath it on the journal's own file. ``None`` when ``journal`` holds no
  registration of ``fork_id``.
  """

  entries, _ = await journal.query_current_strict(
    event_types={"task_registered"}, order="asc",
  )
  registry = TaskRegistry()
  registry.load_from_events([
    {**entry.event, "_durable_seq": entry.seq}
    for entry in entries if entry.event.get("task_id") == fork_id
  ])
  registered = registry.get(fork_id)
  if registered is None or not isinstance(registered.admitted_task, AdmittedTask):
    return None
  admitted = registered.admitted_task
  snapshot = admitted.execution_snapshot
  # spawn_learning_fork admits every fork with both.
  assert snapshot is not None and snapshot.admitted_context is not None
  assert admitted.model_bind is not None
  wire = snapshot.admitted_context
  parent_session = cast(GatewaySession, parent._gateway_session)
  config = cast(LearningForkConfig, parent_session.learning_fork_config)
  # The fork continues on its exact frozen credential generation, as an
  # ordinary child's resume does; CapabilityResolutionError when it cannot.
  execution = materialize_frozen_bind(
    cast(CapabilityExecutionResolver, parent.capability_execution.resolver),
    admitted.model_bind,
    user_scope=parent_session.risk_user_id,
    base_config=parent_session.auth_config or {},
  )
  approval_route = child_approval_route(
    current_parent_context(parent_session).approval_route,
  )
  child_context = context_for_child(
    AdmittedContext(wire, admitted, approval_route), job_id=fork_id,
  )
  child_session = child_gateway_session(
    parent_session,
    session_id=fork_id,
    tenant=wire.tenant,
    auth_config=execution.auth_config,
    user_id=parent_session.user_id,
    user_email=parent_session.user_email,
  )
  # The fork's session resumes onto the journal it wrote, never a fresh one.
  child_session.resume_log_path = journal.path
  capture_job_context(child_session, child_context)
  execution_identity = execution_identity_from_session(
    parent_session, bind=admitted.model_bind,
  )
  return _construct_fork_runner(
    parent,
    # An identity-bearing fork reads its scope from that identity, never the
    # spawning turn (ToolDispatcher._executing_context); the spawning root
    # turn is not recorded on the fork's journal and is not read here.
    spawning_turn=_SpawningTurn(context=child_context, root_turn=None),
    max_tokens=snapshot.max_tokens,
    billing_mode=str(getattr(parent, "_billing_mode", "") or ""),
    tenant_id=(
      execution_identity.tenant_id if execution_identity is not None else None
    ),
    capability_execution=execution,
    attempt=admitted.attempt,
    result_requirement=admitted.result_requirement,
    dispatcher=parent._dispatcher,
    sub_session=child_session,
    identity=dispatch_identity(
      session=parent_session,
      execution=execution_identity,
      context=child_context,
      attempt_lineage=None,
    ),
    approval_route=approval_route,
    local_handler_scopes={
      "memory_write": lambda stock: config.scope_memory_write(
        stock, fork_id=fork_id, user_id=parent_session.user_id,
      ),
    },
    call_index=0,
    parent_turn_id=str(getattr(parent, "_request_id", "") or ""),
    task_entry=registered,
    tool_definitions=[
      definition.definition
      for definition in snapshot.provider_tool_definitions or ()
    ],
    agent_session_log=config.child_session_log_builder(child_session),
    admitted_self_task=None,
    independent_accounting=True,
  ).runner


def spawn_fork_agent(
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
  sub_session: Any | None,
  identity: DispatchIdentity | None,
  approval_route: ApprovalRoute | None,
  bind_job_runner: Callable[[Any], None],
  local_handler_scopes: (
    Mapping[str, Callable[[LocalToolHandler], LocalToolHandler]] | None
  ) = None,
  call_index: int = 0,
  parent_turn_id: str | None = None,
  task_entry: TaskEntry,
  child_cost_observer: Callable[[float], None] | None = None,
) -> Coroutine[Any, Any, tuple[TaskResult | None, dict[str, Any] | None]]:
  """Run one declared fork child on the parent's journal and settle its result."""

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
  admitted = cast(AdmittedTask, task_entry.admitted_task)
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
  child = _prepare_fork_child(
    parent,
    directive,
    handoff=handoff,
    capability_execution=capability_execution,
    attempt=attempt,
    result_requirement=result_requirement,
    dispatcher=dispatcher,
    sub_session=sub_session,
    identity=identity,
    approval_route=approval_route,
    local_handler_scopes=local_handler_scopes,
    call_index=call_index,
    parent_turn_id=parent_turn_id,
    task_entry=task_entry,
    tool_definitions=handoff.wire_tools,
    agent_session_log=parent._agent_session_log,
    admitted_self_task=None,
    child_cost_observer=child_cost_observer,
    independent_accounting=False,
  )
  bind_job_runner(child.runner)
  sub_runner = child.runner
  sub_log = child.log
  sub_session_id = child.session_id

  async def drive() -> tuple[TaskResult | None, dict[str, Any] | None]:
    closure = await child.run()
    approval_park = closure.approval_park
    if (
      approval_park is not None
      and sub_runner._terminal_closure_event is not None
      and sub_runner._terminal_closure_event.get("terminal_disposition") == "awaiting_approval"
      and closure.cancelled_error is None
      and closure.runtime_error_detail is None
      and not closure.stalled
    ):
      evidence = collect_sub_agent_result_evidence(sub_log.entries, durable=False)
      return terminal_task_result(
        admitted,
        status="awaiting_approval",
        reason=(
          f"approval:{approval_park.approval_id} "
          f"call:{approval_park.original_tool_call_id}"
        ),
        tools_used=evidence.tools_used,
        usage=_usage_observation(evidence.usage),
      ), None

    signals = []
    if closure.cancellation_signal in {"cancelled", "killed"}:
      signals.append(closure.cancellation_signal)
    sub_runner_id = getattr(sub_runner, "_runner_id", None)
    if not isinstance(sub_runner_id, str) or not sub_runner_id:
      raise RuntimeError("fork completion requires its exact durable runner_id")
    narrative_text = await final_child_visible_text(
      parent._agent_session_log,
      sub_session_id=sub_session_id,
      role="sub_agent",
      sub_agent_id=sub_session_id,
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
      stalled=closure.stalled,
      runtime_error_detail=closure.runtime_error_detail,
      external_terminal_signals=signals,
      # B-3: the authority frozen at admission, never the ambient catalog.
      admitted_task=admitted,
    )
    if closure.cancelled_error is not None:
      task_entry.task_result = result
      task_entry.result = result.model_dump(mode="json")
      raise closure.cancelled_error
    return result, None

  return drive()


def _count_events(event_entries: Sequence[Any], event_type: str) -> int:
  count = 0
  for raw in event_entries:
    event = raw.event if hasattr(raw, "event") else raw
    if isinstance(event, Mapping) and event.get("type") == event_type:
      count += 1
  return count


def _learning_receipt_text(event_entries: Sequence[Any]) -> str:
  """The owner's next-turn note: deterministic tool effects and turns run."""

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
  turns = _count_events(event_entries, "turn_complete")
  turn_part = f"{turns} turn" + ("" if turns == 1 else "s")
  return f"Self-learning fork completed: {note_part} written in {turn_part}"


def _inherited_catalog(
  parent: Any,
  route_ceiling: frozenset[str],
) -> PlatformToolCatalog:
  return snapshot_platform_catalog(
    tool_ids=sorted(route_ceiling),
    local_tool_handlers=parent._dispatcher.local_tool_handlers(),
    mcp_client=parent._mcp_client,
  )


def spawn_learning_fork(
  fork_id: str,
  raw_work_item: Any,
) -> Awaitable[TaskEntry]:
  """Admit one learning child job of the parent chat and return its run.

  The registry calls this at submission, before ``create_task``: admission,
  bind materialization and registration read the parent's authority as the
  turn left it. A root parent offers its turn's wire tools minus the turn's
  withheld set and the session catalog's callable operations; a child parent
  offers its own canonical admitted task. ``admit`` issues the fork's grant.
  The child runs as its own top-level writer session on its own journal; its
  admitted self task settles the returned entry before the writer lease is
  released.
  """
  from .sub_agent import ordinary_admitted_task_factory

  if not isinstance(raw_work_item, LearningForkWorkItem):
    raise TypeError("learning fork requires LearningForkWorkItem")
  work_item = raw_work_item
  parent = work_item.parent
  handoff = work_item.handoff
  config = work_item.config
  parent_session = cast(GatewaySession, parent._gateway_session)
  parent_context = current_parent_context(parent_session)
  registry_owner = cast(RegistryOwner, parent._task_registry.owner)
  resolver = cast(
    CapabilityExecutionResolver, parent.capability_execution.resolver,
  )
  grant_id = f"tool-grant:{fork_id}"
  fork_bind = resolver.select_bind(
    "node.fork",
    parent_bind=handoff.capability_bind,
    selection_snapshot=parent_context.selection_snapshot,
  )
  # Read before the child session exists: that session holds no credential.
  identity = execution_identity_from_session(parent_session, bind=fork_bind)
  if registry_owner.parent is None:
    root_turn = cast(RootTurnAdmission, current_root_turn(parent_session))
    # The ceiling is the parent's effective surface: the tools its last request
    # advertised, less the turn's withheld set.  The wire list is not that: it
    # also carries anchor tools the session stopped advertising and definitions
    # kept only so a history reference stays valid.
    route_ceiling = frozenset(
      parent._last_request_advertised_tool_names
    ) - frozenset(cast(Sequence[str], root_turn.tool_grant["withheld_tools"]))
    catalog = _inherited_catalog(parent, route_ceiling)
    declaration = inherited_operation_declaration(
      catalog,
      operation_name="learning-fork",
      grant_id=grant_id,
      tool_ceiling=route_ceiling,
    )
    required_capabilities = declaration.required_capabilities
    workspace_scope = cast(WorkspaceScope, declaration.workspace_scope)
    root_operations = cast(
      "AgentOperationCatalog", parent_session.learning_fork_operation_catalog,
    )
    children = tuple(
      operation_ref.name
      for operation_ref, _description
      in root_operations.list_callable_operations_with_descriptions()
    )
  else:
    parent_entry = cast(
      TaskEntry, registry_owner.parent.get(registry_owner.job_id),
    )
    parent_task = cast(AdmittedTask, parent_entry.admitted_task)
    route_ceiling = frozenset(
      entry.tool_id for entry in parent_task.tool_grant.tools
    )
    catalog = _inherited_catalog(parent, route_ceiling)
    required_capabilities = parent_task.operation.required_capabilities
    workspace_scope = parent_task.operation.workspace_scope
    children = cast(tuple[str, ...], parent_context.children)
  operation = config.operation_snapshot_builder(
    config.directive, required_capabilities, workspace_scope,
  )

  def resolve_tool_authority(
    granted: frozenset[str],
  ) -> ResolvedAuthority | OperationUnavailable:
    return resolve_operation_authority(
      OperationDeclaration(
        operation_name=operation.operation.name,
        grant_id=grant_id,
        workspace_scope=workspace_scope,
        required_capabilities=required_capabilities,
        tool_ceiling=granted,
      ),
      catalog=catalog,
      identity=identity,
    )

  admission = admit(
    SpawnRequest(operation=operation.operation.name, task=config.directive),
    parent_context,
    owners=AdmissionOwners(
      operation=OperationRegistration(
        name=operation.operation.name,
        execution_class=operation.execution_class,
        route_ceiling=route_ceiling,
        operation_routes=frozenset(),
        children=children,
      ),
      capability_execution_resolver=resolver,
      session=parent_session,
      route_registered=frozenset(
        entry.tool_id for entry in catalog.tools
      ).__contains__,
      resolve_tool_authority=resolve_tool_authority,
      placement="in_process",
      containment=parent._containment,
      frozen_bind=fork_bind,
    ),
  )
  if isinstance(admission, (Refusal, OperationUnavailable)):
    raise RuntimeError(f"learning fork was not admitted: {admission.detail}")
  authority = admission.tool_authority
  execution = resolver.materialize_bind(admission.wire.capability_bind)
  result_requirement = ResultRequirement(
    mode="narrative",
    projection=None,
    terminal_narrative="required",
    outcome=OutcomeRequirement(required=False, source="none"),
  )
  # The wire's `defer_loading` flags describe the parent's request prefix; the
  # fork's grant is a different tool set with its own first request.
  provider_tool_definitions = provider_tool_definitions_for_grant(
    [
      {key: value for key, value in definition.items() if key != "defer_loading"}
      for definition in handoff.wire_tools
    ],
    grant=authority.grant,
  )
  execution_snapshot = build_agent_execution_snapshot(
    operation=operation,
    result_instructions=render_result_instructions(result_requirement),
    max_tokens=handoff.max_tokens,
    cost_observation_threshold_usd=None,
    provider_tool_definitions=provider_tool_definitions,
    max_resume_chain_depth=parent._max_resume_chain_depth,
    admitted_context=admission.wire,
  )
  admitted = ordinary_admitted_task_factory(
    operation=operation,
    execution_snapshot=execution_snapshot,
    capability_bindings=authority.bindings,
    tool_grant=authority.grant,
    tool_routes=admitted_tool_routes(authority),
    model_bind=execution.bind,
    result_requirement=result_requirement,
    objective=config.directive,
    parent_session=parent_session,
    logical_task_override=OrdinaryDelegationTaskRef(
      delegation_id=fork_id, operation=operation.operation,
    ),
  )(SimpleNamespace(task_id=fork_id))
  # The parent's registry entry is the child job's reservation handle.
  entry = parent._task_registry.register(
    "agent", agent_name=operation.operation.name, task_id=fork_id,
  )
  entry.admitted_task = admitted
  child_context = context_for_child(admission, job_id=fork_id)
  child_session = child_gateway_session(
    parent_session,
    session_id=fork_id,
    tenant=admission.wire.tenant,
    auth_config=execution.auth_config,
    user_id=parent_session.user_id,
    user_email=parent_session.user_email,
  )
  capture_job_context(child_session, child_context)
  child = _prepare_fork_child(
    parent,
    config.directive,
    handoff=handoff,
    capability_execution=execution,
    attempt=admitted.attempt,
    result_requirement=result_requirement,
    dispatcher=parent._dispatcher,
    sub_session=child_session,
    identity=dispatch_identity(
      session=parent_session,
      execution=authority.identity,
      context=child_context,
      attempt_lineage=None,
    ),
    approval_route=admission.approval_route,
    local_handler_scopes={
      "memory_write": lambda stock: config.scope_memory_write(
        stock,
        fork_id=fork_id,
        user_id=work_item.user_id,
      ),
    },
    call_index=0,
    parent_turn_id=str(getattr(parent, "_request_id", "") or ""),
    task_entry=entry,
    tool_definitions=[
      definition.definition for definition in provider_tool_definitions
    ],
    agent_session_log=config.child_session_log_builder(child_session),
    admitted_self_task=entry,
    child_cost_observer=None,
    independent_accounting=True,
  )

  async def run() -> TaskEntry:
    closure = await child.run()
    work_item.fork_events = child.log.entries
    if closure.cancelled_error is not None:
      raise closure.cancelled_error
    return cast(TaskEntry, child.runner._admitted_self_task)

  return run()


__all__ = [
  "FORK_PLACEHOLDER",
  "LearningForkConfig",
  "LearningForkWorkItem",
  "build_fork_messages",
  "cross_check_learning_memory_writes",
  "fork_suffix_messages",
  "learning_fork_host_runner",
  "spawn_fork_agent",
  "spawn_learning_fork",
]
