from __future__ import annotations

import asyncio
import copy
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

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
from .capability_binding import CapabilityBind
from .event_log import EventLog
from .fork_ledger import ForkLedger
from .fork_request_handoff import ForkRequestHandoff
from .fork_scope_receipt import (
  ForkScopeReceipt,
  ForkToolDecision,
  fork_scope_receipt_dict,
  parse_fork_scope_receipt,
)
from .fork_task_registry import learn_fork_budget_usd
from .runner_cleanup import cleanup_failure_notes
from .runner_introspection import derive_sub_agent_id
from .runner_session_lifecycle import _runner_attr
from .runner_state import ChildCostAccumulator
from .runner_sub_agents import (
  _authoritative_child_tool_getter,
  _close_sub_runner,
  _runtime_exception_detail,
)
from .sub_agent_narrative_result import (
  final_child_visible_text,
  task_result_from_execution,
)
from .task_registry import TaskEntry
from .transcript import (
  build_synthetic_tool_results,
  detect_orphan_tool_uses,
)


log = logging.getLogger("agent_gateway.runner")
FORK_PLACEHOLDER = "Fork started — processing continues in the parent."
FORK_SUFFIX_WRAP_UP_REMINDER = (
  "Fork suffix limit reached. Stop expanding the investigation and finish "
  "now with a complete normal final assistant message."
)
DEFAULT_FORK_BUDGET_USD = 5.0
DEFAULT_FORK_MAX_TURNS = 20
DEFAULT_FORK_SUFFIX_MAX_TOKENS = 20_000
FORK_SCOPE_RECEIPT_EVENT_TYPE = "fork_scope_receipt"
FORK_MARKER_EVENT_FIELD = "fork"
FORK_ORCHESTRATION_TOOLS = frozenset({
  "run_agent",
  "get_background_result",
  "resume_background_agent",
  "send_message",
})
LEARNING_FORK_ALLOWED_TOOLS = frozenset({
  "memory_recall",
  "memory_list",
  "memory_read",
  "memory_write",
  "list_skills",
})
LEARNING_FORK_MAX_TURNS = 10
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


@dataclass(slots=True)
class LearningForkWorkItem:
  """Runtime-only owner plus the immutable handoff retained by the registry."""

  parent: Any
  handoff: ForkRequestHandoff
  ledger: ForkLedger
  session_id: str
  owner: str
  user_id: str
  receipt_text: str | None = None

  @property
  def _fork_retained_payload(self) -> ForkRequestHandoff:
    return self.handoff

  def on_admission_settled(self, fork_id: str) -> None:
    if not self.receipt_text:
      raise RuntimeError("learning fork has no receipt to publish")
    self.ledger.write_receipt(
      fork_id=fork_id,
      session_id=self.session_id,
      owner=self.owner,
      receipt_text=self.receipt_text,
    )


def _bind_learning_fork_execution(
  parent_execution: BoundCapabilityExecution,
) -> BoundCapabilityExecution:
  parent_execution.validate()
  parent = parent_execution.bind
  return BoundCapabilityExecution(
    bind=CapabilityBind.model_validate({
      **parent.receipt(),
      "capability_id": "node.fork",
      "selection_source": "parent_binding",
    }),
    registry=parent_execution.registry,
    adapter=parent_execution.adapter,
    auth_config=parent_execution.auth_config,
  )


def _positive_env_int(name: str, default: int) -> int:
  raw = os.getenv(name)
  if raw is None:
    return default
  try:
    value = int(raw)
  except (TypeError, ValueError) as exc:
    raise ValueError(f"{name} must be a positive integer") from exc
  if value <= 0:
    raise ValueError(f"{name} must be a positive integer")
  return value


def fork_max_turns() -> int:
  return _positive_env_int("HANK_FORK_MAX_TURNS", DEFAULT_FORK_MAX_TURNS)


def fork_suffix_max_tokens() -> int:
  return _positive_env_int(
    "HANK_FORK_SUFFIX_MAX_TOKENS",
    DEFAULT_FORK_SUFFIX_MAX_TOKENS,
  )


def fork_budget_default() -> float:
  raw = os.getenv("HANK_FORK_BUDGET_USD")
  if raw is None:
    return DEFAULT_FORK_BUDGET_USD
  try:
    value = float(raw)
  except (TypeError, ValueError) as exc:
    raise ValueError("HANK_FORK_BUDGET_USD must be finite and positive") from exc
  if value <= 0 or value == float("inf") or value != value:
    raise ValueError("HANK_FORK_BUDGET_USD must be finite and positive")
  return value


def build_side_quest_tool_decisions(
  wire_tools: Sequence[Mapping[str, Any]],
  *,
  is_classified: Callable[[str], bool],
) -> tuple[ForkToolDecision, ...]:
  decisions: list[ForkToolDecision] = []
  seen: set[str] = set()
  for definition in wire_tools:
    name = str(definition.get("name") or "").strip()
    if not name or name in seen:
      raise ValueError("fork wire tools must have unique non-empty names")
    seen.add(name)
    if name in FORK_ORCHESTRATION_TOOLS:
      decision = "deny"
      reason = "orchestration surface"
    elif not is_classified(name):
      decision = "deny"
      reason = "unclassified tool"
    else:
      decision = "allow"
      reason = "parent surface"
    decisions.append(ForkToolDecision(name, decision, reason))
  return tuple(sorted(decisions, key=lambda item: item.tool))


def build_learning_fork_tool_decisions(
  wire_tools: Sequence[Mapping[str, Any]],
) -> tuple[ForkToolDecision, ...]:
  """Classify the exact wire surface against the closed five-tool allowlist."""

  decisions: list[ForkToolDecision] = []
  seen: set[str] = set()
  for definition in wire_tools:
    name = str(definition.get("name") or "").strip()
    if not name or name in seen:
      raise ValueError("fork wire tools must have unique non-empty names")
    seen.add(name)
    allowed = name in LEARNING_FORK_ALLOWED_TOOLS
    decisions.append(ForkToolDecision(
      name,
      "allow" if allowed else "deny",
      "learning fork closed allowlist" if allowed else "not in learning allowlist",
    ))
  return tuple(sorted(decisions, key=lambda item: item.tool))


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


class ForkPolicyDispatcher:
  """Dispatch-layer fork policy while preserving the parent's wire catalog."""

  def __init__(
    self,
    dispatcher: Any,
    *,
    wire_tools: Sequence[Mapping[str, Any]],
    receipt: ForkScopeReceipt,
  ) -> None:
    self._dispatcher = dispatcher
    self._wire_tools = copy.deepcopy(list(wire_tools))
    self._decisions = {
      decision.tool: decision
      for decision in receipt.tool_decisions
    }
    if set(self._decisions) != {
      str(definition.get("name") or "").strip()
      for definition in self._wire_tools
    }:
      raise ValueError(
        "fork dispatch policy must classify the exact wire tool set"
      )

  def __getattr__(self, name: str) -> Any:
    return getattr(self._dispatcher, name)

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return copy.deepcopy(self._wire_tools)

  async def dispatch(
    self,
    tool_call_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    **kwargs: Any,
  ) -> tuple[Any | None, dict[str, Any] | None]:
    decision = self._decisions.get(str(tool_name or "").strip())
    if decision is None or decision.decision != "allow":
      normalized = str(tool_name or "").strip() or "<unknown>"
      return None, {
        "code": "fork_policy_denied",
        "message": (
          f"fork policy: {normalized} is not available in a forked child"
        ),
        "data": {
          "tool": normalized,
          "fork_kind": "side_quest",
          "reason": (
            decision.reason if decision is not None else "unclassified tool"
          ),
        },
      }
    return await self._dispatcher.dispatch(
      tool_call_id,
      tool_name,
      tool_input,
      **kwargs,
    )


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
  messages: Sequence[Mapping[str, Any]],
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
    "inherited context, stay concise, and do not call run_agent or any "
    "background/resume management tool. Finish with a normal assistant "
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
  messages: Sequence[Mapping[str, Any]],
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
) -> EventLog:
  parent_log = parent._log
  original_on_event = getattr(parent_log, "_on_event", None)

  def prepare_event(event: dict[str, Any]) -> dict[str, Any]:
    prepared = copy.deepcopy(event)
    prepared["sub_agent_id"] = sub_session_id
    prepared[FORK_MARKER_EVENT_FIELD] = True
    return prepared

  def on_event(event: dict[str, Any], session_id: str) -> None:
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
  dispatcher: Any,
  scope_receipt: Mapping[str, Any],
  max_turns: int,
  max_budget_usd: float,
  suffix_ceiling: int,
  timeout: float | None = None,
  client_timeout: float = 90,
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
  if capability_execution.bind.capability_id != "node.fork":
    raise ValueError("spawn_fork_agent requires a node.fork capability bind")
  if not isinstance(attempt, AttemptRef):
    raise TypeError("spawn_fork_agent requires an exact AttemptRef")
  if not isinstance(result_requirement, ResultRequirement):
    raise TypeError("spawn_fork_agent requires a ResultRequirement")
  if result_requirement.mode != "narrative":
    raise ValueError("fork agents return a normal terminal message")
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
  receipt = parse_fork_scope_receipt(scope_receipt)
  if (
    handoff.capability_bind != parent._capability_execution.bind
    or receipt.capability_bind != capability_execution.bind
    or receipt.max_turns != max_turns
    or receipt.suffix_ceiling != suffix_ceiling
    or receipt.resolved_budget_usd != max_budget_usd
  ):
    raise ValueError("fork scope receipt does not match spawn configuration")
  sub_session_id = _runner_attr(
    parent,
    "_derive_sub_agent_id",
    derive_sub_agent_id,
  )(
    parent._full_session_id,
    call_index,
  )
  sub_log = _fork_event_log(parent, sub_session_id=sub_session_id)
  if event_log_observer is not None:
    event_log_observer(sub_log)
  child_dispatcher = copy.copy(dispatcher)
  if hasattr(child_dispatcher, "_event_log"):
    child_dispatcher._event_log = sub_log
  if hasattr(child_dispatcher, "_session_id"):
    child_dispatcher._session_id = sub_session_id
  policy_dispatcher = ForkPolicyDispatcher(
    child_dispatcher,
    wire_tools=handoff.wire_tools,
    receipt=receipt,
  )
  child_get_tool_definitions = _authoritative_child_tool_getter(
    policy_dispatcher,
    operation="spawn_fork_agent",
  )
  sub_log.append({
    "type": FORK_SCOPE_RECEIPT_EVENT_TYPE,
    "receipt": receipt.to_dict(),
  })
  child_accumulator = ChildCostAccumulator(
    None if independent_accounting else parent._cost_accumulator,
    max_budget_usd,
  )
  sub_runner = type(parent)(
    event_log=sub_log,
    dispatcher=policy_dispatcher,
    session_id=sub_session_id,
    capability_execution=capability_execution,
    client_timeout=client_timeout,
    max_tokens_override=handoff.max_tokens,
    per_turn_timeout=parent._per_turn_timeout,
    stream_stall_timeout=parent._stream_stall_timeout,
    mcp_client=parent._mcp_client,
    loaded_mcp_servers=parent._loaded_mcp_servers,
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
    on_max_turns=parent._on_max_turns,
    max_budget_usd=max_budget_usd,
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
  sub_runner._fork_suffix_max_tokens = suffix_ceiling
  sub_runner._fork_scope_receipt = receipt.to_dict()
  sub_runner._tenant_id = handoff.tenant_id
  child_messages, child_marker_position = _build_fork_context(handoff, directive)
  sub_runner._fork_marker_position = child_marker_position

  timed_out = False
  runtime_error_detail: str | None = None
  cancelled_error: asyncio.CancelledError | None = None
  cancellation_signal: str | None = None
  cleanup_warnings: list[str] = []
  run_coro = sub_runner.run(
    messages=child_messages[-1:],
    system_prompt=list(handoff.rendered_system_blocks),
    max_turns=max_turns,
    resume_initial_messages=child_messages,
  )
  try:
    if timeout is not None and timeout > 0:
      await asyncio.wait_for(run_coro, timeout=timeout)
    else:
      await run_coro
  except asyncio.TimeoutError as exc:
    timed_out = True
    cleanup_warnings.extend(cleanup_failure_notes(exc))
    sub_log.append({
      "type": "error",
      "error": f"Forked sub-agent timed out after {timeout}s",
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
      timed_out=timed_out,
      cancelled_error=cancelled_error,
      cancellation_signal=cancellation_signal,
      runtime_exception_detail=runtime_error_detail,
      task_entry=task_entry,
      cleanup_warnings=cleanup_warnings,
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
    timed_out=timed_out,
    timeout=timeout,
    runtime_error_detail=runtime_error_detail,
    external_terminal_signals=signals,
  )
  if cancelled_error is not None:
    if task_entry is not None:
      task_entry.task_result = result
      task_entry.result = result.model_dump(mode="json")
    raise cancelled_error
  return result, None


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
  from agent.shared.prompts.learn_directive import LEARN_DIRECTIVE
  from agent.shared.tool_handlers.fork_memory_write import (
    scope_fork_memory_write_handler,
  )
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
  parent_dispatcher = parent._dispatcher
  dispatcher = copy.copy(parent_dispatcher)
  local_handlers = dict(getattr(parent_dispatcher, "_local", {}))
  base_memory_write = local_handlers.get("memory_write")
  if not callable(base_memory_write):
    raise RuntimeError("learning fork requires the stock memory_write handler")
  local_handlers["memory_write"] = scope_fork_memory_write_handler(
    base_memory_write,
    fork_id=fork_id,
    user_id=work_item.user_id,
  )
  dispatcher._local = local_handlers

  budget = float(learn_fork_budget_usd())
  suffix_ceiling = fork_suffix_max_tokens()
  scope_receipt = fork_scope_receipt_dict(
    tool_decisions=build_learning_fork_tool_decisions(handoff.wire_tools),
    capability_bind=execution.bind,
    tenant_id=handoff.tenant_id,
    billing_mode=handoff.billing_mode,
    resolved_budget_usd=budget,
    max_turns=LEARNING_FORK_MAX_TURNS,
    suffix_ceiling=suffix_ceiling,
  )
  result_provenance = TaskResultProvenance(
    admitted_task_digest=sha256_digest({
      "logical_task": logical_task.model_dump(mode="json"),
      "attempt": attempt.model_dump(mode="json"),
      "objective": LEARN_DIRECTIVE,
      "result_requirement": result_requirement.model_dump(mode="json"),
    }),
    model_bind_digest=sha256_digest(execution.bind),
    capability_binding_digest=sha256_digest({
      "kind": "dynamic_parent_bind",
      "bind": execution.bind.model_dump(mode="json"),
    }),
    tool_grant_digest=sha256_digest(scope_receipt),
  )
  child_cost: list[float] = []
  result, error = await spawn_fork_agent(
    parent,
    LEARN_DIRECTIVE,
    handoff=handoff,
    capability_execution=execution,
    logical_task=logical_task,
    attempt=attempt,
    result_requirement=result_requirement,
    result_provenance=result_provenance,
    dispatcher=dispatcher,
    scope_receipt=scope_receipt,
    max_turns=LEARNING_FORK_MAX_TURNS,
    max_budget_usd=budget,
    suffix_ceiling=suffix_ceiling,
    call_index=call_index,
    parent_turn_id=str(getattr(parent, "_request_id", "") or ""),
    event_log_observer=event_log_box.append,
    child_cost_observer=child_cost.append,
    independent_accounting=True,
  )
  if error is not None or result is None or result.execution.status != "succeeded":
    raise RuntimeError("learning fork did not complete successfully")
  work_item.receipt_text = _learning_receipt_text(
    event_log_box[0].entries if event_log_box else (),
  )
  return Decimal(str(child_cost[-1] if child_cost else 0))


__all__ = [
  "DEFAULT_FORK_BUDGET_USD",
  "DEFAULT_FORK_MAX_TURNS",
  "DEFAULT_FORK_SUFFIX_MAX_TOKENS",
  "FORK_PLACEHOLDER",
  "FORK_SUFFIX_WRAP_UP_REMINDER",
  "ForkPolicyDispatcher",
  "LEARNING_FORK_ALLOWED_TOOLS",
  "LEARNING_FORK_MAX_TURNS",
  "LearningForkWorkItem",
  "build_fork_messages",
  "build_learning_fork_tool_decisions",
  "build_side_quest_tool_decisions",
  "cross_check_learning_memory_writes",
  "fork_budget_default",
  "fork_max_turns",
  "fork_suffix_max_tokens",
  "fork_suffix_messages",
  "spawn_fork_agent",
  "spawn_learning_fork",
]
