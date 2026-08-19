from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Literal, Mapping, Sequence, Set, Tuple

from .runner_budget import (
  BudgetCostProgress as BudgetCostProgress,
  BudgetExceededState as BudgetExceededState,
  ChildCostAccumulator as ChildCostAccumulator,
  CostAccumulator as CostAccumulator,
  ProviderRequestBudgetAdmission as ProviderRequestBudgetAdmission,
  ProviderRequestBudgetError as ProviderRequestBudgetError,
  admit_provider_request_budget as admit_provider_request_budget,
  budget_cost_progress as budget_cost_progress,
  budget_exceeded_state as budget_exceeded_state,
  budget_reason_suffix as budget_reason_suffix,
)
from .thinking import parse_effort


def normalized_run_config(
  auth_config: Dict[str, Any],
  *,
  upstream_model: str,
  effort: str,
) -> Dict[str, Any]:
  config = dict(auth_config)
  selection_fields = {
    "effort",
    "model",
    "model_key",
    "thinking",
    "thinking_enabled_requested",
  } & set(config)
  if selection_fields:
    raise ValueError(
      "credential auth_config must not contain model selection"
    )
  normalized_model = str(upstream_model or "").strip()
  if not normalized_model:
    raise ValueError("native execution requires a bound upstream model")
  requested_effort = parse_effort(
    effort,
    field_name="capability_execution.effort",
  )
  if requested_effort is None:
    raise ValueError(
      "native execution requires an explicitly bound effort"
    )
  config.update({
    "auth_mode": str(config.get("auth_mode", "api")).strip().lower(),
    "api_key": str(config.get("api_key", "")),
    "auth_token": str(config.get("auth_token", "")),
    "model": normalized_model,
    "effort": requested_effort.value,
    "thinking_enabled_requested": requested_effort.value != "none",
    "max_tokens": int(config.get("max_tokens", 16000)),
  })
  return config


@dataclass(frozen=True)
class MaxTokensSelection:
  value: int
  requested: int
  model_max_output: int
  clamped: bool


def select_run_max_tokens(
  config: Dict[str, Any],
  *,
  max_tokens_override: int | None,
  model_info: Any,
) -> MaxTokensSelection:
  requested = int(max_tokens_override if max_tokens_override is not None else config["max_tokens"])
  model_max_output = int(getattr(model_info, "max_output_tokens", 0) or 0)
  if model_max_output > 0 and requested > model_max_output:
    return MaxTokensSelection(
      value=model_max_output,
      requested=requested,
      model_max_output=model_max_output,
      clamped=True,
    )
  return MaxTokensSelection(
    value=requested,
    requested=requested,
    model_max_output=model_max_output,
    clamped=False,
  )


def assistant_turn_message(
  content_blocks: List[Any],
  *,
  provider: str,
  model: str,
  stop_reason: str | None,
) -> Dict[str, Any]:
  return {
    "role": "assistant",
    "content": list(content_blocks),
    "provider": provider,
    "model": model,
    "stop_reason": stop_reason,
  }


def user_turn_message(content: Any) -> Dict[str, Any]:
  return {"role": "user", "content": content}


def background_tasks_completed_user_message() -> Dict[str, Any]:
  return user_turn_message(
    "[System: New background-task notifications are included above. Use those "
    "outcomes directly. Each notification includes its complete payload or "
    "explicitly marks an omission. Do not call get_background_result for tasks "
    "from this run unless a notification explicitly says its payload was omitted.]"
  )


def tool_execution_order_indices(
  tool_names: Sequence[str],
) -> tuple[int, ...]:
  """Return the runtime execution order for one assistant tool-use turn."""
  return tuple(range(len(tool_names)))


@dataclass(frozen=True)
class TurnReminderState:
  text: str
  peeked_notification_count: int
  # The notification objects this request's reminder renders — the
  # delivered set acked at that request's response boundary (A-M7).
  delivered: Tuple[Any, ...] = ()


def turn_reminder_state(
  background_reminder: str,
  notification_reminder: str,
  *,
  pending_notification_count: int,
  max_notifications_per_turn: int,
  delivered: Sequence[Any] = (),
) -> TurnReminderState:
  return TurnReminderState(
    text="\n\n".join(filter(None, [background_reminder, notification_reminder])),
    peeked_notification_count=(
      min(pending_notification_count, max_notifications_per_turn)
      if notification_reminder
      else 0
    ),
    delivered=tuple(delivered),
  )


def completed_assistant_content_blocks(content_blocks: List[Any]) -> List[Any]:
  return [
    block
    for block in content_blocks
    if not (isinstance(block, dict) and block.get("type") == "tool_use")
  ]


@dataclass(frozen=True)
class MaxTokensContinuationDecision:
  attempts: int
  should_continue: bool


@dataclass(frozen=True)
class NoToolUseTurnOutcome:
  action: Literal["continue", "break"]
  messages: List[Dict[str, Any]]
  max_tokens_continuations: int
  reason: str
  max_tokens_decision: MaxTokensContinuationDecision | None = None
  runtime_guard: Tuple[str, str] | None = None


def max_tokens_continuation_decision(
  *,
  current_attempts: int,
  max_attempts: int,
  unbounded: bool = False,
) -> MaxTokensContinuationDecision:
  attempts = current_attempts + 1
  return MaxTokensContinuationDecision(
    attempts=attempts,
    should_continue=unbounded or attempts <= max_attempts,
  )


def no_tool_use_turn_outcome(
  *,
  content_blocks: List[Any],
  provider: str,
  model: str,
  stop_reason: str | None,
  pending_notification_count: int,
  max_tokens_continuations: int,
  max_tokens_max_attempts: int,
  max_tokens_nudge: str,
  unbounded_max_tokens_continuations: bool = False,
) -> NoToolUseTurnOutcome:
  if stop_reason in {"pause_turn", "compaction"}:
    return NoToolUseTurnOutcome(
      action="continue",
      messages=[
        assistant_turn_message(
          content_blocks,
          provider=provider,
          model=model,
          stop_reason=stop_reason,
        )
      ],
      max_tokens_continuations=max_tokens_continuations,
      reason=str(stop_reason),
    )

  if stop_reason == "max_tokens":
    continuation_decision = max_tokens_continuation_decision(
      current_attempts=max_tokens_continuations,
      max_attempts=max_tokens_max_attempts,
      unbounded=unbounded_max_tokens_continuations,
    )
    if continuation_decision.should_continue:
      messages: List[Dict[str, Any]] = []
      assistant_content = completed_assistant_content_blocks(content_blocks)
      if assistant_content:
        messages.append(
          assistant_turn_message(
            assistant_content,
            provider=provider,
            model=model,
            stop_reason=stop_reason,
          )
        )
      messages.append(user_turn_message(max_tokens_nudge))
      return NoToolUseTurnOutcome(
        action="continue",
        messages=messages,
        max_tokens_continuations=continuation_decision.attempts,
        reason="max_tokens_continue",
        max_tokens_decision=continuation_decision,
        runtime_guard=("max_tokens_truncation", max_tokens_nudge),
      )
    return NoToolUseTurnOutcome(
      action="break",
      messages=[],
      max_tokens_continuations=continuation_decision.attempts,
      reason="max_tokens_terminal",
      max_tokens_decision=continuation_decision,
    )

  if pending_notification_count > 0:
    return NoToolUseTurnOutcome(
      action="continue",
      messages=[
        assistant_turn_message(
          content_blocks,
          provider=provider,
          model=model,
          stop_reason=stop_reason,
        ),
        background_tasks_completed_user_message(),
      ],
      max_tokens_continuations=max_tokens_continuations,
      reason="pending_notifications",
    )

  return NoToolUseTurnOutcome(
    action="break",
    messages=[],
    max_tokens_continuations=max_tokens_continuations,
    reason="no_tool_use_terminal",
  )


def model_visible_extra_blocks(extra_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  return [block for block in extra_blocks if not block.get("_event_only")]


def should_execute_run_agent_batch(
  tool_name: str,
  effective_excluded_tools: Set[str],
) -> bool:
  return tool_name == "run_agent" and "run_agent" not in effective_excluded_tools


@dataclass(frozen=True)
class RunAgentBatch:
  batch: List[Tuple[int, str, str, Dict[str, Any]]]
  call_indices: List[int]
  next_index: int
  next_call_index: int


@dataclass(frozen=True)
class RunAgentBatchResult:
  tool_results_content: List[Dict[str, Any]]
  deferred_extras: List[Dict[str, Any]]
  tools_used: List[str]


@dataclass(frozen=True)
class RunAgentBatchCallResult:
  batch_result: RunAgentBatchResult
  next_index: int
  next_call_index: int


@dataclass(frozen=True)
class SingleToolCallStepResult:
  tool_result: RunAgentBatchResult
  next_index: int


@dataclass(frozen=True)
class ToolUseLoopResult:
  tool_results_content: List[Dict[str, Any]]
  tools_used: List[str]


def append_tool_execution_result(
  tool_results_content: List[Dict[str, Any]],
  deferred_extras: List[Dict[str, Any]],
  tools_used: List[str],
  result: RunAgentBatchResult,
) -> None:
  tool_results_content.extend(result.tool_results_content)
  deferred_extras.extend(result.deferred_extras)
  tools_used.extend(result.tools_used)


def append_deferred_tool_extras(
  tool_results_content: List[Dict[str, Any]],
  deferred_extras: List[Dict[str, Any]],
) -> None:
  tool_results_content.extend(deferred_extras)


def collect_run_agent_batch(
  tool_uses: List[Tuple[str, str, Dict[str, Any]]],
  *,
  start_index: int,
  start_call_index: int,
  run_agent_available: Callable[[], bool],
) -> RunAgentBatch:
  batch: List[Tuple[int, str, str, Dict[str, Any]]] = []
  call_indices: List[int] = []
  index = start_index
  call_index = start_call_index
  while index < len(tool_uses):
    tool_id, tool_name, tool_input = tool_uses[index]
    if tool_name != "run_agent" or not run_agent_available():
      break
    batch.append((index, tool_id, tool_name, tool_input))
    call_indices.append(call_index)
    call_index += 1
    index += 1
  return RunAgentBatch(
    batch=batch,
    call_indices=call_indices,
    next_index=index,
    next_call_index=call_index,
  )


async def execute_run_agent_with_optional_semaphore(
  tool_input: Dict[str, Any],
  *,
  sub_agent_semaphore: Any,
  execute: Callable[[], Awaitable[Any]],
) -> Any:
  if sub_agent_semaphore is not None and not bool(tool_input.get("background")):
    async with sub_agent_semaphore:
      return await execute()
  return await execute()


async def execute_run_agent_batch_item(
  tool_call_id: str,
  tool_call_name: str,
  tool_call_input: Dict[str, Any],
  call_index: int,
  *,
  base_kwargs: Dict[str, Any],
  sub_agent_semaphore: Any,
  execute_single_tool: Callable[..., Awaitable[Any]],
) -> Any:
  return await execute_run_agent_with_optional_semaphore(
    tool_call_input,
    sub_agent_semaphore=sub_agent_semaphore,
    execute=lambda: execute_single_tool(
      tool_call_id,
      tool_call_name,
      tool_call_input,
      base_kwargs,
      call_index=call_index,
    ),
  )


async def gather_run_agent_batch_results(
  batch: List[Tuple[int, str, str, Dict[str, Any]]],
  call_indices: List[int],
  *,
  execute: Callable[[str, str, Dict[str, Any], int], Awaitable[Any]],
  gather_fn: Callable[..., Awaitable[Any]],
) -> Any:
  return await gather_fn(
    *[
      execute(
        batch_tool_id,
        batch_tool_name,
        batch_tool_input,
        call_index,
      )
      for (_, batch_tool_id, batch_tool_name, batch_tool_input), call_index in zip(batch, call_indices)
    ],
    return_exceptions=True,
  )


def process_run_agent_batch_results(
  batch: List[Tuple[int, str, str, Dict[str, Any]]],
  results: List[Any],
  *,
  batch_error: Callable[[BaseException], Any],
  make_error_result: Callable[[str, str, str], Dict[str, Any]],
  model_visible_extra_blocks: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
  on_exception: Callable[[BaseException], None],
) -> RunAgentBatchResult:
  tool_results_content: List[Dict[str, Any]] = []
  deferred_extras: List[Dict[str, Any]] = []
  tools_used: List[str] = []
  for j, result_or_exc in enumerate(results):
    _, batch_tool_id, batch_tool_name, _ = batch[j]
    if isinstance(result_or_exc, BaseException):
      error = batch_error(result_or_exc)
      on_exception(result_or_exc)
      tool_results_content.append(
        make_error_result(batch_tool_id, error.code, error.message)
      )
      tools_used.append(batch_tool_name)
    else:
      result_entry, used_name, extra_blocks = result_or_exc
      tool_results_content.append(result_entry)
      deferred_extras.extend(model_visible_extra_blocks(extra_blocks))
      tools_used.append(used_name)
  return RunAgentBatchResult(
    tool_results_content=tool_results_content,
    deferred_extras=deferred_extras,
    tools_used=tools_used,
  )


def process_single_tool_result(
  result_entry: Dict[str, Any],
  used_name: str,
  extra_blocks: List[Dict[str, Any]],
  *,
  model_visible_extra_blocks: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
) -> RunAgentBatchResult:
  return RunAgentBatchResult(
    tool_results_content=[result_entry],
    deferred_extras=model_visible_extra_blocks(extra_blocks),
    tools_used=[used_name],
  )


async def execute_single_tool_call(
  tool_call_id: str,
  tool_call_name: str,
  tool_call_input: Dict[str, Any],
  *,
  base_kwargs: Dict[str, Any],
  execute_single_tool: Callable[..., Awaitable[Any]],
  model_visible_extra_blocks: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
) -> RunAgentBatchResult:
  result_entry, used_name, extra_blocks = await execute_single_tool(
    tool_call_id,
    tool_call_name,
    tool_call_input,
    base_kwargs,
  )
  return process_single_tool_result(
    result_entry,
    used_name,
    extra_blocks,
    model_visible_extra_blocks=model_visible_extra_blocks,
  )


async def execute_single_tool_call_step(
  tool_call_id: str,
  tool_call_name: str,
  tool_call_input: Dict[str, Any],
  *,
  current_index: int,
  base_kwargs: Dict[str, Any],
  execute_single_tool: Callable[..., Awaitable[Any]],
  model_visible_extra_blocks: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
) -> SingleToolCallStepResult:
  tool_result = await execute_single_tool_call(
    tool_call_id,
    tool_call_name,
    tool_call_input,
    base_kwargs=base_kwargs,
    execute_single_tool=execute_single_tool,
    model_visible_extra_blocks=model_visible_extra_blocks,
  )
  return SingleToolCallStepResult(
    tool_result=tool_result,
    next_index=current_index + 1,
  )


async def execute_run_agent_batch(
  batch: List[Tuple[int, str, str, Dict[str, Any]]],
  call_indices: List[int],
  *,
  execute: Callable[[str, str, Dict[str, Any], int], Awaitable[Any]],
  gather_fn: Callable[..., Awaitable[Any]],
  batch_error: Callable[[BaseException], Any],
  make_error_result: Callable[[str, str, str], Dict[str, Any]],
  model_visible_extra_blocks: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
  on_exception: Callable[[BaseException], None],
) -> RunAgentBatchResult:
  results = await gather_run_agent_batch_results(
    batch,
    call_indices,
    execute=execute,
    gather_fn=gather_fn,
  )
  return process_run_agent_batch_results(
    batch,
    results,
    batch_error=batch_error,
    make_error_result=make_error_result,
    model_visible_extra_blocks=model_visible_extra_blocks,
    on_exception=on_exception,
  )


async def execute_run_agent_batch_call(
  tool_uses: List[Tuple[str, str, Dict[str, Any]]],
  *,
  start_index: int,
  start_call_index: int,
  run_agent_available: Callable[[], bool],
  execute: Callable[[str, str, Dict[str, Any], int], Awaitable[Any]],
  gather_fn: Callable[..., Awaitable[Any]],
  batch_error: Callable[[BaseException], Any],
  make_error_result: Callable[[str, str, str], Dict[str, Any]],
  model_visible_extra_blocks: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
  on_exception: Callable[[BaseException], None],
) -> RunAgentBatchCallResult:
  batch_state = collect_run_agent_batch(
    tool_uses,
    start_index=start_index,
    start_call_index=start_call_index,
    run_agent_available=run_agent_available,
  )
  batch_result = await execute_run_agent_batch(
    batch_state.batch,
    batch_state.call_indices,
    execute=execute,
    gather_fn=gather_fn,
    batch_error=batch_error,
    make_error_result=make_error_result,
    model_visible_extra_blocks=model_visible_extra_blocks,
    on_exception=on_exception,
  )
  return RunAgentBatchCallResult(
    batch_result=batch_result,
    next_index=batch_state.next_index,
    next_call_index=batch_state.next_call_index,
  )


async def execute_tool_use_loop(
  tool_uses: List[Tuple[str, str, Dict[str, Any]]],
  *,
  base_kwargs: Dict[str, Any],
  sub_agent_semaphore: Any,
  effective_excluded_tools: Callable[[], Set[str]],
  execute_single_tool: Callable[..., Awaitable[Any]],
  gather_fn: Callable[..., Awaitable[Any]],
  batch_error: Callable[[BaseException], Any],
  make_error_result: Callable[[str, str, str], Dict[str, Any]],
  model_visible_extra_blocks: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
  on_exception: Callable[[BaseException], None],
  should_stop_after_tool_result: Callable[[], bool] = lambda: False,
) -> ToolUseLoopResult:
  ordered_tool_uses = [
    tool_uses[index]
    for index in tool_execution_order_indices(
      [tool_name for _, tool_name, _ in tool_uses],
    )
  ]

  tool_results_content: List[Dict[str, Any]] = []
  deferred_extras: List[Dict[str, Any]] = []
  tools_used: List[str] = []
  i = 0
  run_agent_seq = 0
  while i < len(ordered_tool_uses):
    tool_id, tool_name, tool_input = ordered_tool_uses[i]
    excluded_tools = effective_excluded_tools()
    if should_execute_run_agent_batch(tool_name, excluded_tools):
      batch_call = await execute_run_agent_batch_call(
        ordered_tool_uses,
        start_index=i,
        start_call_index=run_agent_seq,
        run_agent_available=lambda: "run_agent" not in effective_excluded_tools(),
        execute=lambda batch_tool_id, batch_tool_name, batch_tool_input, call_index: execute_run_agent_batch_item(
          batch_tool_id,
          batch_tool_name,
          batch_tool_input,
          call_index,
          base_kwargs=base_kwargs,
          sub_agent_semaphore=sub_agent_semaphore,
          execute_single_tool=execute_single_tool,
        ),
        gather_fn=gather_fn,
        batch_error=batch_error,
        make_error_result=make_error_result,
        model_visible_extra_blocks=model_visible_extra_blocks,
        on_exception=on_exception,
      )
      i = batch_call.next_index
      run_agent_seq = batch_call.next_call_index
      append_tool_execution_result(
        tool_results_content,
        deferred_extras,
        tools_used,
        batch_call.batch_result,
      )
    else:
      single_call = await execute_single_tool_call_step(
        tool_id,
        tool_name,
        tool_input,
        current_index=i,
        base_kwargs=base_kwargs,
        execute_single_tool=execute_single_tool,
        model_visible_extra_blocks=model_visible_extra_blocks,
      )
      append_tool_execution_result(
        tool_results_content,
        deferred_extras,
        tools_used,
        single_call.tool_result,
      )
      i = single_call.next_index
    if should_stop_after_tool_result():
      for skipped_tool_id, skipped_tool_name, _ in ordered_tool_uses[i:]:
        tool_results_content.append(
          make_error_result(
            skipped_tool_id,
            "tool_skipped_after_terminal_result",
            (
              f"Tool '{skipped_tool_name}' was skipped because an earlier "
              "tool result ended this turn"
            ),
          )
        )
      break

  append_deferred_tool_extras(tool_results_content, deferred_extras)
  return ToolUseLoopResult(
    tool_results_content=tool_results_content,
    tools_used=tools_used,
  )


@dataclass(frozen=True)
class SubAgentBatchError:
  code: str
  message: str


def sub_agent_batch_error(exc: BaseException) -> SubAgentBatchError:
  if isinstance(exc, asyncio.CancelledError):
    return SubAgentBatchError(code="cancelled", message="Sub-agent was cancelled")
  return SubAgentBatchError(code="sub_agent_error", message=str(exc) or "Sub-agent failed")


@dataclass(frozen=True)
class SessionDrainState:
  drain_complete: bool
  in_flight_task_count: int
  unsettled_workflow_obstruction_count: int = 0


def session_drain_state(
  running_entries: List[Any],
  *,
  shutdown_failed: bool,
  unsettled_workflow_obstructions: int = 0,
) -> SessionDrainState:
  """Derive drain completion from in-flight tasks and workflow obligations.

  An unsettled workflow obligation — a live drive or an unacked
  awaiting_action boundary — keeps the drain incomplete (§5.3, T2-I04) so
  a session can never report a clean settlement over a silently orphaned
  workflow run.
  """

  in_flight_task_count = sum(
    1
    for task in running_entries
    if task.asyncio_task is not None and not task.asyncio_task.done()
  )
  return SessionDrainState(
    drain_complete=(
      not shutdown_failed
      and in_flight_task_count == 0
      and unsettled_workflow_obstructions == 0
    ),
    in_flight_task_count=in_flight_task_count,
    unsettled_workflow_obstruction_count=unsettled_workflow_obstructions,
  )


def usage_cache_status(usage_totals: Dict[str, int]) -> str:
  if usage_totals["cache_read_input_tokens"] > 0:
    return f"hit ({usage_totals['cache_read_input_tokens']} tokens cached)"
  if usage_totals["cache_creation_input_tokens"] > 0:
    return f"write ({usage_totals['cache_creation_input_tokens']} tokens written)"
  return "miss"


@dataclass(frozen=True)
class RegistryScope:
  scope_type: str
  scope_id: str
  store_dir: str


@dataclass
class ToolResultContext:
  """Context passed to `on_tool_result` hooks.

  Hooks can inspect the original tool input, the normalized tool result, the
  emitted result entry, and timing metadata before the runner forwards the tool
  result back into the model conversation.
  """

  tool_name: str
  tool_input: Dict[str, Any]
  result: Any | None
  error: Dict[str, Any] | None
  duration_ms: int
  tool_call_id: str
  session_id: str
  server: str | None
  result_entry: Dict[str, Any] | None
  provider_id: str | None = None
  skill_run_id: str | None = None
  workspace_dir: str | None = None
  registry_scope: RegistryScope | None = None
  batch_id: int | str | None = None
  # Runtime-built, never model-visible: the bounded projection of what a
  # delegated child actually read, stripped from the tool result before the
  # model or the durable log sees it and delivered here instead so the parent
  # runtime can seed its citation registry (D-B4-2).
  child_evidence: Mapping[str, Any] | None = None
  # The dispatch boundary's own verdict for this call (B-1). Hooks that mint
  # citable evidence consume it instead of re-deriving success from per-tool
  # status conventions: a failed retrieval must never become a source.
  dispatch: Mapping[str, Any] | None = None
  boundary_sanitizer: Callable[[Any, str], Any] | None = field(
    default=None,
    repr=False,
  )


@dataclass
class SubAgentConfig:
  """Default settings applied to spawned sub-agents."""

  excluded_tools: Set[str]
  system_prompt: str | None = None
  max_turns: int = 15
  model: str | None = None


@dataclass
class StreamTurnResult:
  full_text: str = ""
  tool_uses: List[Tuple[str, str, Dict[str, Any]]] = field(default_factory=list)
  stop_reason: str | None = None
  first_token_t: float | None = None
  content_blocks: List[Dict[str, Any]] = field(default_factory=list)
  advertised_tool_names: frozenset[str] | None = None


@dataclass(frozen=True)
class StreamTurnFailure:
  error: Exception
  formatted_error: str
  is_context_length: bool = False


@dataclass(frozen=True)
class StreamTurnLogSummary:
  elapsed_s: float
  ttft_s: float | None
  text_chars: int
  tool_names: List[str]
  text_preview: str


def stream_turn_log_summary(
  turn: StreamTurnResult,
  *,
  turn_started_at: float,
  now: float,
) -> StreamTurnLogSummary:
  return StreamTurnLogSummary(
    elapsed_s=now - turn_started_at,
    ttft_s=(turn.first_token_t - turn_started_at) if turn.first_token_t else None,
    text_chars=len(turn.full_text),
    tool_names=[tool[1] for tool in turn.tool_uses],
    text_preview=turn.full_text[:150].replace("\n", " ") if turn.full_text else "",
  )


@dataclass
class BackgroundTask:
  task_id: str
  agent_name: str | None
  asyncio_task: asyncio.Task[Any] | None
  started_at: float
  result: Dict[str, Any] | None = None
  error: Dict[str, Any] | None = None
  completed: bool = False
  completed_at: float | None = None
