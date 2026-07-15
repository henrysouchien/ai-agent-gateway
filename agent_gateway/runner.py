from __future__ import annotations

import asyncio
import fcntl  # noqa: F401 - compatibility alias for runner session lifecycle monkeypatches
import inspect
import json  # noqa: F401 - compatibility alias for runner tool execution monkeypatches
import logging
import socket  # noqa: F401 - compatibility alias for runner session lifecycle monkeypatches
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from ._provider_utils import _get_default_model_for_provider  # noqa: F401 - compatibility alias
from .agent_session_log import AgentSessionLog
from .auth import ProviderCredentialFailure
from .context_builder import SessionContextBuilder
from .event_log import EventLog
from .mcp_client import McpClientManager
from .multi_user.billing import (
  DEFAULT_USAGE_DLQ_PATH,
  SessionUsageSummary,
  UsageEvent,
  _UsageAggregator,
  normalize_identity,
)
from .product_config import gateway_product_id  # noqa: F401 - compatibility alias
from .providers import ModelInfo, ModelProvider, ThinkingLevel  # noqa: F401 - compatibility aliases
from .runner_introspection import (
  derive_sub_agent_id as _derive_sub_agent_id,  # noqa: F401 - compatibility alias
  detect_keyword_param as _detect_keyword_param,
  detect_user_id_param as _detect_user_id_param,
  format_exc as _format_exc,  # noqa: F401 - compatibility alias
)
from .runner_limits import (
  COMPACTION_TRIGGER_PCT as COMPACTION_TRIGGER_PCT,
  CONTEXT_WARNING_PCT as CONTEXT_WARNING_PCT,
  MODEL_CONTEXT_LIMIT as MODEL_CONTEXT_LIMIT,
  effective_compaction_trigger as _effective_compaction_trigger,  # noqa: F401 - compatibility alias
  estimate_tokens as _estimate_tokens,  # noqa: F401 - compatibility alias
  model_context_window as _model_context_window,  # noqa: F401 - compatibility alias
  system_prompt_estimate_text as _system_prompt_estimate_text,  # noqa: F401 - compatibility alias
  token_breakdown_snapshot as _token_breakdown_snapshot,  # noqa: F401 - compatibility alias
  token_estimate_snapshot as _token_estimate_snapshot,  # noqa: F401 - compatibility alias
)
from .runner_prompt_rules import (
  last_user_message as _last_user_message,
  message_content_text as _message_content_text,  # noqa: F401 - compatibility alias
  messages_require_tool_only_turns as _messages_require_tool_only_turns,  # noqa: F401 - compatibility alias
  prepend_system_prompt_preamble as _prepend_system_prompt_preamble,  # noqa: F401 - compatibility alias
  system_prompt_requires_tool_only_turns as _system_prompt_requires_tool_only_turns,  # noqa: F401 - compatibility alias
  system_prompt_text as _system_prompt_text,  # noqa: F401 - compatibility alias
)
from .runner_run_loop_defaults import (
  MAX_NOTIFICATIONS_PER_TURN as _MAX_NOTIFICATIONS_PER_TURN,  # noqa: F401 - compatibility alias
  MAX_TOKENS_CONTINUATIONS as _MAX_TOKENS_CONTINUATIONS,  # noqa: F401 - compatibility alias
  MAX_TOKENS_NUDGE as _MAX_TOKENS_NUDGE,  # noqa: F401 - compatibility alias
)
from .runner_background_lifecycle import RunnerBackgroundLifecycleMixin
from .runner_hooks_lifecycle import RunnerHooksLifecycleMixin
from .runner_auth import (
  call_credential_refresher as _call_credential_refresher,  # noqa: F401 - compatibility alias
  merge_refreshed_auth_config as _merge_refreshed_auth_config,  # noqa: F401 - compatibility alias
)
from .runner_background_tasks import (
  background_asyncio_tasks as _background_asyncio_tasks,  # noqa: F401 - compatibility alias
  background_task_call_index as _background_task_call_index,  # noqa: F401 - compatibility alias
  background_elapsed_seconds as _background_elapsed_seconds,  # noqa: F401 - compatibility alias
  background_task_model as _background_task_model,  # noqa: F401 - compatibility alias
  background_task_payload as _background_task_payload,  # noqa: F401 - compatibility alias
  background_task_provider_name as _background_task_provider_name,  # noqa: F401 - compatibility alias
  background_task_ids as _background_task_ids,  # noqa: F401 - compatibility alias
  background_task_ids_for_asyncio_tasks as _background_task_ids_for_asyncio_tasks,  # noqa: F401 - compatibility alias
  background_task_limit_error as _background_task_limit_error,  # noqa: F401 - compatibility alias
  background_task_registration_metadata as _background_task_registration_metadata,  # noqa: F401 - compatibility alias
  background_task_reminder_text as _background_task_reminder_text,  # noqa: F401 - compatibility alias
  background_result_task as _background_result_task,  # noqa: F401 - compatibility alias
  background_result_tasks as _background_result_tasks,  # noqa: F401 - compatibility alias
  background_task_started_result as _background_task_started_result,  # noqa: F401 - compatibility alias
  background_timeout_value as _background_timeout_value,  # noqa: F401 - compatibility alias
  background_wait_tasks as _background_wait_tasks,  # noqa: F401 - compatibility alias
  call_before_background_task_start_hook as _call_before_background_task_start_hook,  # noqa: F401 - compatibility alias
  drain_cancelled_background_tasks as _drain_cancelled_background_tasks,  # noqa: F401 - compatibility alias
  drain_still_pending_background_tasks as _drain_still_pending_background_tasks,  # noqa: F401 - compatibility alias
  entry_aware_background_handler as _entry_aware_background_handler,  # noqa: F401 - compatibility alias
  ensure_sub_agent_semaphore as _ensure_sub_agent_semaphore,  # noqa: F401 - compatibility alias
  kill_background_tasks as _kill_background_tasks,  # noqa: F401 - compatibility alias
  kill_background_tasks_for_asyncio_tasks as _kill_background_tasks_for_asyncio_tasks,
  parse_background_result_request as _parse_background_result_request,  # noqa: F401 - compatibility alias
  parse_child_budget_usd as _parse_child_budget_usd,  # noqa: F401 - compatibility alias
  prepare_background_task_registration as _prepare_background_task_registration,  # noqa: F401 - compatibility alias
  resume_chain_depth as _resume_chain_depth,  # noqa: F401 - compatibility alias
  resume_root_task_id as _resume_root_task_id,  # noqa: F401 - compatibility alias
  resume_root_task_id_from_registry as _resume_root_task_id_from_registry,  # noqa: F401 - compatibility alias
  resume_task_id_override as _resume_task_id_override,  # noqa: F401 - compatibility alias
  resumed_task_ids as _resumed_task_ids,  # noqa: F401 - compatibility alias
  resumed_task_ids_from_registry as _resumed_task_ids_from_registry,  # noqa: F401 - compatibility alias
  sub_agent_result_from_log_entries as _sub_agent_result_from_log_entries,  # noqa: F401 - compatibility alias
  task_completed_event_payload as _task_completed_event_payload,  # noqa: F401 - compatibility alias
  task_correlation_payload as _task_correlation_payload,  # noqa: F401 - compatibility alias
  task_registered_event_payload as _task_registered_event_payload,  # noqa: F401 - compatibility alias
  wait_for_background_tasks as _wait_for_background_tasks,  # noqa: F401 - compatibility alias
)
from .runner_callbacks import (
  call_before_stream_complete_hook as _call_before_stream_complete_hook,  # noqa: F401 - compatibility alias
  call_metric_hook as _call_metric_hook,  # noqa: F401 - compatibility alias
  call_tool_timing_hook as _call_tool_timing_hook,  # noqa: F401 - compatibility alias
  call_tool_result_hook as _call_tool_result_hook,  # noqa: F401 - compatibility alias
)
from .runner_notifications import (
  build_notification_reminder as _build_notification_reminder,  # noqa: F401 - compatibility alias
  consume_notifications as _consume_notifications,  # noqa: F401 - compatibility alias
  inject_system_prompt_reminder as _inject_system_prompt_reminder,  # noqa: F401 - compatibility alias
)
from .runner_session_lifecycle import RunnerSessionLifecycleMixin
from .runner_sub_agents import RunnerSubAgentMixin
from .runner_stream_turn import RunnerStreamTurnMixin
from .runner_run_loop import RunnerRunLoopMixin
from .runner_tool_execution import RunnerToolExecutionMixin
from .runner_session_events import (
  build_assistant_message_event as _build_assistant_message_event,  # noqa: F401 - compatibility alias
  build_attach_event as _build_attach_event,  # noqa: F401 - compatibility alias
  build_budget_exceeded_event as _build_budget_exceeded_event,  # noqa: F401 - compatibility alias
  build_budget_exceeded_text_event as _build_budget_exceeded_text_event,  # noqa: F401 - compatibility alias
  build_chat_done_log_data as _build_chat_done_log_data,  # noqa: F401 - compatibility alias
  build_context_warning_log_data as _build_context_warning_log_data,  # noqa: F401 - compatibility alias
  build_detach_event as _build_detach_event,  # noqa: F401 - compatibility alias
  build_error_event as _build_error_event,  # noqa: F401 - compatibility alias
  build_interrupted_event as _build_interrupted_event,  # noqa: F401 - compatibility alias
  build_max_turns_reached_event as _build_max_turns_reached_event,  # noqa: F401 - compatibility alias
  build_max_turns_text_event as _build_max_turns_text_event,  # noqa: F401 - compatibility alias
  build_orphan_tool_call_interrupted_events as _build_orphan_tool_call_interrupted_events,  # noqa: F401 - compatibility alias
  build_operator_pause_event as _build_operator_pause_event,  # noqa: F401 - compatibility alias
  build_run_error_event as _build_run_error_event,  # noqa: F401 - compatibility alias
  build_runtime_guard_event as _build_runtime_guard_event,  # noqa: F401 - compatibility alias
  build_stream_complete_event as _build_stream_complete_event,  # noqa: F401 - compatibility alias
  build_stream_retry_event as _build_stream_retry_event,  # noqa: F401 - compatibility alias
  build_stub_response_events as _build_stub_response_events,
  build_tool_call_complete_event as _build_tool_call_complete_event,  # noqa: F401 - compatibility alias
  build_tool_call_start_event as _build_tool_call_start_event,  # noqa: F401 - compatibility alias
  build_turn_complete_log_data as _build_turn_complete_log_data,  # noqa: F401 - compatibility alias
  build_token_estimate_log_data as _build_token_estimate_log_data,  # noqa: F401 - compatibility alias
  build_turn_complete_event as _build_turn_complete_event,  # noqa: F401 - compatibility alias
  build_user_message_event as _build_user_message_event,  # noqa: F401 - compatibility alias
  durable_event_payload as _durable_event_payload,  # noqa: F401 - compatibility alias
  release_write_lease as _release_write_lease,  # noqa: F401 - compatibility alias
  run_detach_reason as _run_detach_reason,  # noqa: F401 - compatibility alias
  run_interrupted_reason as _run_interrupted_reason,  # noqa: F401 - compatibility alias
  shutdown_interrupted_reason as _shutdown_interrupted_reason,  # noqa: F401 - compatibility alias
  write_lease_metadata as _write_lease_metadata,  # noqa: F401 - compatibility alias
)
from .runner_skill_gate import (
  default_tool_definitions as _default_tool_definitions,
  effective_excluded_tools as _effective_excluded_tools,
  filter_excluded_tool_definitions as _filter_excluded_tool_definitions,
  is_report_door_clear_event as _is_report_door_clear_event,
  normalize_skill_deny as _normalize_skill_deny,
  normalize_skill_report_doors as _normalize_skill_report_doors,
)
from .runner_state import (
  BackgroundTask,
  ChildCostAccumulator,  # noqa: F401 - compatibility alias
  CostAccumulator,
  StreamTurnResult,  # noqa: F401 - compatibility alias
  SubAgentConfig,
  ToolResultContext,
  assistant_turn_message as _assistant_turn_message,  # noqa: F401 - compatibility alias
  background_tasks_completed_user_message as _background_tasks_completed_user_message,  # noqa: F401 - compatibility alias
  budget_cost_progress as _budget_cost_progress,  # noqa: F401 - compatibility alias
  budget_exceeded_state as _budget_exceeded_state,  # noqa: F401 - compatibility alias
  budget_reason_suffix as _budget_reason_suffix,  # noqa: F401 - compatibility alias
  execute_tool_use_loop as _execute_tool_use_loop,  # noqa: F401 - compatibility alias
  model_visible_extra_blocks as _model_visible_extra_blocks,  # noqa: F401 - compatibility alias
  no_tool_use_turn_outcome as _no_tool_use_turn_outcome,  # noqa: F401 - compatibility alias
  normalized_run_config as _normalized_run_config,  # noqa: F401 - compatibility alias
  select_run_max_tokens as _select_run_max_tokens,  # noqa: F401 - compatibility alias
  session_drain_state as _session_drain_state,  # noqa: F401 - compatibility alias
  stream_turn_log_summary as _stream_turn_log_summary,  # noqa: F401 - compatibility alias
  sub_agent_batch_error as _sub_agent_batch_error,  # noqa: F401 - compatibility alias
  turn_reminder_state as _turn_reminder_state,  # noqa: F401 - compatibility alias
  usage_cache_status as _usage_cache_status,  # noqa: F401 - compatibility alias
  user_turn_message as _user_turn_message,  # noqa: F401 - compatibility alias
)
from .runner_streaming import (
  STREAM_STALL_TIMEOUT as STREAM_STALL_TIMEOUT,
  STREAM_THINKING_STALL_TIMEOUT as STREAM_THINKING_STALL_TIMEOUT,
  classify_guard_outcome,  # noqa: F401 - compatibility alias
  effective_stream_stall_timeout,  # noqa: F401 - compatibility alias
  observed_thinking_in_messages,  # noqa: F401 - compatibility alias
  thinking_level,  # noqa: F401 - compatibility alias
)
from .runner_tool_audit import (
  get_tool_risk_value as _get_tool_risk_value,  # noqa: F401 - compatibility alias
  redact_tool_input_for_event as _redact_tool_input_for_event,  # noqa: F401 - compatibility alias
)
from .runner_usage import (
  apply_message_start_usage as _apply_message_start_usage,  # noqa: F401 - compatibility alias
  apply_usage_update as _apply_usage_update,  # noqa: F401 - compatibility alias
  build_usage_event as _build_usage_event,  # noqa: F401 - compatibility alias
  call_late_usage_event_hook as _call_late_usage_event_hook,  # noqa: F401 - compatibility alias
  call_session_summary_hook as _call_session_summary_hook,  # noqa: F401 - compatibility alias
  call_usage_event_hook as _call_usage_event_hook,  # noqa: F401 - compatibility alias
  empty_usage_totals as _empty_usage_totals,  # noqa: F401 - compatibility alias
  estimate_usage_cost as _estimate_usage_cost,  # noqa: F401 - compatibility alias
  turn_usage_payload as _turn_usage_payload,  # noqa: F401 - compatibility alias
  usage_delta as _usage_delta,  # noqa: F401 - compatibility alias
  usage_delta_state as _usage_delta_state,  # noqa: F401 - compatibility alias
  usage_has_tokens as _usage_has_tokens,  # noqa: F401 - compatibility alias
)
from .session_recap import emit_recap_then_terminal
from .task_registry import (
  COORDINATOR_DEFAULT_PREAMBLE,  # noqa: F401 - compatibility alias
  CoordinatorConfig,
  NotificationQueue,
  ParentMessage,
  TaskEntry,
  TaskNotification,
  TaskRegistry,
  TaskState,
  format_parent_messages_for_model,  # noqa: F401 - compatibility alias
  make_progress_tracker,  # noqa: F401 - compatibility alias
)
from .tool_dispatcher import ToolDispatcher
from .tool_display import resolve_display  # noqa: F401 - compatibility alias
from .tool_result_compaction import (
  MODEL_TOOL_RESULT_MAX_CHARS as MODEL_TOOL_RESULT_MAX_CHARS,
  MODEL_TOOL_RESULT_MAX_CHARS_ENV as MODEL_TOOL_RESULT_MAX_CHARS_ENV,
  MODEL_TOOL_RESULT_MIN_CHARS as MODEL_TOOL_RESULT_MIN_CHARS,
  SPILL_TRUNCATED_TOOL_RESULTS_ENV as SPILL_TRUNCATED_TOOL_RESULTS_ENV,
  annotate_result,
  compact_model_tool_result_entry as _compact_model_tool_result_entry,
  is_error_tool_result_entry,
  make_error_result,
  model_tool_result_max_chars as _model_tool_result_max_chars,  # noqa: F401 - compatibility alias
  scalar_preview_fields as _scalar_preview_fields,  # noqa: F401 - compatibility alias
  spill_truncated_tool_results_enabled as _spill_truncated_tool_results_enabled,  # noqa: F401 - compatibility alias
  truncate_model_tool_result_content as _truncate_model_tool_result_content,  # noqa: F401 - compatibility alias
  write_tool_result_spill,
)
from .tool_result_semantics import (
  classify_semantic_tool_error,  # noqa: F401 - compatibility alias
  is_semantic_tool_error as _is_soft_error,
)
from .tool_result_spill import SpillSink, normalize_spill_sink


kill_background_tasks = _kill_background_tasks
kill_background_tasks_for_asyncio_tasks = _kill_background_tasks_for_asyncio_tasks

log = logging.getLogger("agent_gateway.runner")
STREAM_GUARD_POLL_INTERVAL = 2.0
# Liveness is guarded by event-gap stall detection (retryable), NOT wall clock:
# thinking-turn duration is unpredictable, so per_turn_timeout should be None on
# thinking surfaces. If a caller does set it, it must comfortably EXCEED the
# stall allowance above or the terminal per-turn abort preempts the retryable
# stall guard on a slow first token (both timers start near turn start;
# per-turn always leads). See ACUI-25.
STREAM_RETRY_MAX = 3
STREAM_RETRY_DELAY = 2.0
STREAM_RETRY_BACKOFF = 2.0
# Backstop cap for inline run_agent dispatch (ACUI-1). Must comfortably exceed
# sub_agent.DEFAULT_SUB_AGENT_TIMEOUT_SECONDS (1800s) so the inner spawn
# timeout fires first and returns a clean tool error; this only triggers if
# that inner await itself never resolves.
_RUN_AGENT_DISPATCH_TIMEOUT_SECONDS = 2100.0
_ACTIVE_SKILL_DENY_RESULT_KEY = "_active_skill_deny"
_ACTIVE_SKILL_REPORT_DOORS_RESULT_KEY = "_active_skill_report_doors"
# Report doors are the only auto-clear doors in scope and normally return noop;
# staged is included defensively for preview-mode doors that stage artifacts.
_REPORT_DOOR_CLEAR_SUCCESS_STATUSES = frozenset({"noop", "staged"})
FinalAnswerGuard = Callable[
  [List[Dict[str, Any]], str, List[str], List[Dict[str, Any]], int],
  str | None,
]


OnToolResult = Callable[[ToolResultContext], Awaitable[List[Dict[str, Any]] | None]]
OnUsage = Callable[[UsageEvent], Awaitable[None] | None]
OnSessionSummary = Callable[[SessionUsageSummary], Awaitable[None] | None]
OnBeforeStreamComplete = Callable[..., Awaitable[None] | None]
OnToolTiming = Callable[..., None]
OnMaxTurns = Callable[[List[Dict[str, Any]], int], Awaitable[str | None]]
BackgroundTaskHandler = Callable[..., Awaitable[Tuple[Optional[Any], Optional[Dict[str, Any]]]]]
BackgroundTaskCallback = Callable[[BackgroundTask | TaskEntry], Awaitable[None] | None]
OnMetric = Callable[[str, int], None]
OnCredentialRefresh = Callable[[ProviderCredentialFailure], Awaitable[Dict[str, Any] | None] | Dict[str, Any] | None]
ShutdownSignalProvider = Callable[[], Dict[str, Any] | None]


class AgentRunner(
  RunnerSessionLifecycleMixin,
  RunnerBackgroundLifecycleMixin,
  RunnerHooksLifecycleMixin,
  RunnerSubAgentMixin,
  RunnerToolExecutionMixin,
  RunnerStreamTurnMixin,
  RunnerRunLoopMixin,
):
  """Run the model/tool loop for one gateway conversation.

  `AgentRunner` is responsible for streaming provider output, executing tool
  calls through `ToolDispatcher`, collecting usage, enforcing budgets and time
  limits, and appending client-visible events to `EventLog`.

  Constructor highlights:

  - `provider` supplies model-specific request/stream behavior.
  - `dispatcher` executes local or MCP tools.
  - `auth_config` carries provider credentials, model, and token limits.
  - `get_tool_definitions` supplies the current tool schema visible to the
    model.
  - `on_tool_result`, `on_usage`, and `on_tool_timing` are observability hooks.
  - `max_budget_usd` and `max_turns` stop the loop before it runs away.
  """

  def __init__(
    self,
    event_log: EventLog,
    dispatcher: ToolDispatcher,
    session_id: str,
    *,
    provider: ModelProvider,
    auth_config: Dict[str, Any] | None = None,
    allow_stub_response: bool = True,
    client_timeout: float | None = None,
    max_tokens_override: int | None = None,
    per_turn_timeout: float | None = None,
    stream_stall_timeout: float | None = None,
    mcp_client: McpClientManager | None = None,
    loaded_mcp_servers: Set[str] | None = None,
    excluded_tools: Set[str] | None = None,
    get_tool_definitions: Callable[[], List[Dict[str, Any]]] | None = None,
    on_tool_result: OnToolResult | None = None,
    on_usage: OnUsage | None = None,
    on_session_summary: OnSessionSummary | None = None,
    on_late_usage_event: OnUsage | None = None,
    on_before_stream_complete: OnBeforeStreamComplete | None = None,
    on_tool_timing: OnToolTiming | None = None,
    user_id: str | None = None,
    request_id: str | None = None,
    parent_turn_id: str | None = None,
    billing_mode: str | None = None,
    rate_table_version: str | None = None,
    channel: str | None = None,
    usage_ledger_dlq_path: Path | str | None = None,
    on_metric: OnMetric | None = None,
    on_credential_failure: OnCredentialRefresh | None = None,
    sub_agent_config: SubAgentConfig | None = None,
    compaction_trigger: int | None = None,
    compaction_instructions: str | None = None,
    tool_call_timeout: float | None = 120.0,
    on_max_turns: OnMaxTurns | None = None,
    final_answer_guard: FinalAnswerGuard | None = None,
    max_budget_usd: float | None = None,
    _cost_accumulator: CostAccumulator | None = None,
    _parent_aggregator: _UsageAggregator | None = None,
    max_concurrent_sub_agents: int | None = None,
    agent_session_log: AgentSessionLog | None = None,
    context_builder: SessionContextBuilder | None = None,
    task_registry: TaskRegistry | None = None,
    message_inbox: asyncio.Queue[ParentMessage] | None = None,
    coordinator: CoordinatorConfig | None = None,
    max_resume_chain_depth: int = 3,
    operator_pause_event: asyncio.Event | None = None,
    shutdown_signal_provider: ShutdownSignalProvider | None = None,
    started_at: float | None = None,
    emit_session_recap: bool = True,
    code_execution_spill_dir_provider: Callable[[], str] | SpillSink | None = None,
    skill_run_id: str | None = None,
    workspace_dir: str | Path | None = None,
    batch_id: int | str | None = None,
    context_surfaces: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]] | None = None,
    commercial_usage_producer: Any | None = None,
  ) -> None:
    if max_budget_usd is not None and max_budget_usd <= 0:
      raise ValueError("max_budget_usd must be positive when provided")
    if max_concurrent_sub_agents is not None and max_concurrent_sub_agents <= 0:
      raise ValueError("max_concurrent_sub_agents must be positive when provided")
    if max_resume_chain_depth <= 0:
      raise ValueError("max_resume_chain_depth must be positive")

    self._log = event_log
    self._dispatcher = dispatcher
    self._spill_dir_provider = normalize_spill_sink(code_execution_spill_dir_provider)
    self._provider = provider
    self._full_session_id = session_id or "no-session"
    self._sid = self._full_session_id[:12]
    self._session_started_at = float(started_at if started_at is not None else time.time())
    self._emit_session_recap = bool(emit_session_recap)
    self._skill_run_id = str(skill_run_id).strip() if skill_run_id else None
    self._workspace_dir = str(workspace_dir) if workspace_dir is not None else None
    self._batch_id = str(batch_id).strip() if batch_id is not None and str(batch_id).strip() else None
    self._auth_config = dict(auth_config or {})
    self._allow_stub_response = bool(allow_stub_response)
    self._effort_resolution = None
    self._client_timeout = client_timeout
    self._max_tokens_override = max_tokens_override
    self._per_turn_timeout = per_turn_timeout
    self._stream_stall_timeout = stream_stall_timeout
    self._mcp_client = mcp_client
    self._loaded_mcp_servers = loaded_mcp_servers if loaded_mcp_servers is not None else set()
    self._excluded_tools = set(excluded_tools or set())
    self._active_skill_deny: set[str] = set()
    self._active_skill_report_doors: dict[str, str] = {}
    self._get_tool_definitions = get_tool_definitions
    self._on_tool_result = on_tool_result
    self._on_usage = on_usage
    self._commercial_usage_producer = commercial_usage_producer
    self._on_session_summary = on_session_summary
    self._on_late_usage_event = on_late_usage_event
    self._on_before_stream_complete = on_before_stream_complete
    self._on_tool_timing = on_tool_timing
    self._on_tool_timing_accepts_user_id = _detect_user_id_param(on_tool_timing)
    self._on_tool_timing_accepts_context_surfaces = _detect_keyword_param(on_tool_timing, "context_surfaces")
    self._on_tool_timing_accepts_tool_call_id = _detect_keyword_param(on_tool_timing, "tool_call_id")
    self._on_tool_timing_accepts_request_id = _detect_keyword_param(on_tool_timing, "request_id")
    self._context_surfaces_provider = context_surfaces if callable(context_surfaces) else None
    self._context_surfaces_static = self._normalize_context_surfaces(None if callable(context_surfaces) else context_surfaces)
    self._request_id = str(request_id or uuid.uuid4())
    self._parent_turn_id = parent_turn_id
    self._usage_user_id, self._rate_table_version, self._billing_mode, self._channel = normalize_identity(
      user_id,
      rate_table_version,
      billing_mode,
      channel,
    )
    self._parent_aggregator = _parent_aggregator
    self._aggregator = _parent_aggregator or _UsageAggregator(
      user_id=self._usage_user_id,
      session_id=self._full_session_id,
      request_id=self._request_id,
      channel=self._channel,
      rate_table_version=self._rate_table_version,
      billing_mode=self._billing_mode,
    )
    self._summary_emitted = False
    self._usage_ledger_dlq_path = (
      Path(usage_ledger_dlq_path).expanduser() if usage_ledger_dlq_path is not None else DEFAULT_USAGE_DLQ_PATH
    )
    self._on_metric = on_metric
    self._on_credential_failure = on_credential_failure
    self._sub_agent_config = sub_agent_config
    self._compaction_trigger = compaction_trigger
    self._compaction_instructions = compaction_instructions
    self._tool_call_timeout = tool_call_timeout
    self._on_max_turns = on_max_turns
    self._final_answer_guard = final_answer_guard
    self._max_budget_usd = max_budget_usd if max_budget_usd is not None else (
      _cost_accumulator.budget if _cost_accumulator is not None else None
    )
    self._cost_accumulator = _cost_accumulator
    if self._cost_accumulator is None and self._max_budget_usd is not None:
      self._cost_accumulator = CostAccumulator(self._max_budget_usd)
    self._last_reported_cost = 0.0
    self._coordinator = coordinator
    self._max_resume_chain_depth = max_resume_chain_depth
    self._max_concurrent_sub_agents = max_concurrent_sub_agents
    self._max_background_tasks = max_concurrent_sub_agents or 3
    self._sub_agent_semaphore: asyncio.Semaphore | None = None
    self._active_client: Any | None = None
    self._disconnected = False
    self._tool_abort_event = asyncio.Event()
    try:
      dispatch_params = inspect.signature(self._dispatcher.dispatch).parameters
      self._dispatcher_accepts_abort_event = "abort_event" in dispatch_params or any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in dispatch_params.values()
      )
      self._dispatcher_accepts_skill_run_context = (
        "skill_run_id" in dispatch_params
        or "workspace_dir" in dispatch_params
        or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in dispatch_params.values())
      )
      self._dispatcher_accepts_readable_resource_snapshot = (
        "capture_readable_resource_snapshot" in dispatch_params
        or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in dispatch_params.values())
      )
    except (TypeError, ValueError):
      self._dispatcher_accepts_abort_event = False
      self._dispatcher_accepts_skill_run_context = False
      self._dispatcher_accepts_readable_resource_snapshot = False
    task_registry_auto_created = task_registry is None
    self._task_registry = task_registry or TaskRegistry(
      max_inflight=self._max_background_tasks,
      id_prefix="bg",
    )
    self._message_inbox = message_inbox
    self._operator_pause_event = operator_pause_event
    self._shutdown_signal_provider = shutdown_signal_provider
    self._notification_queue = NotificationQueue()
    if self._coordinator is not None and self._coordinator.enabled:
      self._max_background_tasks = self._coordinator.max_workers
      if task_registry_auto_created:
        self._task_registry._max_inflight = self._max_background_tasks
    notification_queue = self._notification_queue

    class _NotificationListener:
      def on_transition(self_listener, entry: TaskEntry, old_state: TaskState, new_state: TaskState) -> None:
        _ = self_listener, old_state
        if new_state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.KILLED):
          event_name = new_state.value
          summary = ""
          if entry.result and isinstance(entry.result, dict):
            summary = str(entry.result.get("response", ""))
          elif entry.error and isinstance(entry.error, dict):
            summary = str(entry.error.get("message", ""))
          notification_queue.push(
            TaskNotification(
              task_id=entry.task_id,
              agent_name=entry.agent_name,
              event=event_name,
              summary=summary,
              timestamp=time.time(),
              payload=entry.result or entry.error or {},
            )
          )

    notifications_enabled = not (
      self._coordinator is not None
      and self._coordinator.enabled
      and not self._coordinator.auto_notify
    )
    if notifications_enabled:
      self._task_registry.add_listener(_NotificationListener())
    self._agent_session_log = agent_session_log
    self._context_builder = context_builder
    self._gateway_session_id = self._full_session_id
    self._role = "sub_agent" if self._full_session_id.startswith("sub") and ":" in self._full_session_id else "writer"
    self._sub_agent_id = self._full_session_id if self._role == "sub_agent" else None
    self._client_kind = self._channel or ("cron" if self._agent_session_log is not None else "cli")
    self._runner_id: str | None = None
    self._write_lease_file: Any | None = None
    self._durable_attach_emitted = False
    self._last_durable_seq = 0
    self._last_assistant_message_seq: int | None = None
    self._task_registry_rebuild_lock = asyncio.Lock()
    self._task_registry_rebuilt = False

  @property
  def _background_tasks(self) -> Dict[str, TaskEntry]:
    return self._task_registry._tasks

  @property
  def effort_introspection(self) -> dict[str, Any] | None:
    """Requested/effective effort for the latest turn, without vendor fragments."""
    resolution = self._effort_resolution
    if resolution is None:
      return None
    return {
      "requested": resolution.requested.value,
      "effective": resolution.effective.value,
      "thinking_enabled_effective": resolution.thinking_enabled_effective,
    }

  @staticmethod
  def _normalize_context_surfaces(surfaces: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [
      dict(surface)
      for surface in (surfaces or [])
      if isinstance(surface, dict)
    ]

  def _context_surface_records(self) -> list[dict[str, Any]]:
    if self._context_surfaces_provider is None:
      return self._normalize_context_surfaces(self._context_surfaces_static)
    try:
      return self._normalize_context_surfaces(self._context_surfaces_provider())
    except Exception as exc:
      log.warning("[%s] context surface provider failed (non-fatal): %s", self._sid, exc)
      return self._normalize_context_surfaces(self._context_surfaces_static)

  def _append(self, event: Dict[str, Any]) -> None:
    if event.get("type") in {"stream_complete", "error"}:
      emit_recap_then_terminal(
        self._log,
        event,
        session_id=self._full_session_id,
        started_at=self._session_started_at,
        emit_recap=self._emit_session_recap,
      )
      return
    self._log.append(event)

  def request_operator_pause(self) -> None:
    if self._operator_pause_event is None:
      self._operator_pause_event = asyncio.Event()
    self._operator_pause_event.set()

  def _operator_pause_requested(self) -> bool:
    return bool(self._operator_pause_event is not None and self._operator_pause_event.is_set())

  def set_credential_refresher(self, callback: OnCredentialRefresh | None) -> None:
    self._on_credential_failure = callback

  def _extract_last_user_message(self, request_messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return _last_user_message(request_messages)


  @staticmethod
  def _annotate_result(result: Any, tool_name: str = "") -> Any:
    return annotate_result(result, tool_name=tool_name)

  @staticmethod
  def _make_error_result(
    tool_use_id: str,
    code: str,
    message: str,
    sub_code: str = "",
    data: Dict[str, Any] | None = None,
  ) -> Dict[str, Any]:
    return make_error_result(tool_use_id, code, message, sub_code, data)

  def _compact_model_tool_result_entry(
    self,
    result_entry: Dict[str, Any],
    *,
    tool_name: str,
  ) -> tuple[Dict[str, Any], Dict[str, Any]]:
    return _compact_model_tool_result_entry(
      result_entry,
      tool_name=tool_name,
      spill_sink=self._spill_dir_provider,
      log_session_id=self._sid,
      logger=log,
      uuid_factory=uuid.uuid4,
    )

  @staticmethod
  def _is_error_tool_result_entry(result_entry: Dict[str, Any], content: str) -> bool:
    return is_error_tool_result_entry(result_entry, content)

  @staticmethod
  def _write_tool_result_spill(
    *,
    work_dir: str,
    tool_name: str,
    tool_use_id: Any,
    content: str,
  ) -> tuple[str, str]:
    return write_tool_result_spill(
      work_dir=work_dir,
      tool_name=tool_name,
      tool_use_id=tool_use_id,
      content=content,
      uuid_factory=uuid.uuid4,
    )

  @staticmethod
  def _is_soft_error(result: Any) -> bool:
    return _is_soft_error(result)

  def _default_tool_definitions(self) -> List[Dict[str, Any]]:
    return _default_tool_definitions(self._get_tool_definitions, self._mcp_client)

  def _effective_excluded_tools(self) -> set[str]:
    return _effective_excluded_tools(self._excluded_tools, self._active_skill_deny)

  def _filter_excluded_tool_definitions(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _filter_excluded_tool_definitions(tools, self._effective_excluded_tools())

  def _rebuild_filtered_tool_definitions(self, base_kwargs: Dict[str, Any]) -> None:
    base_kwargs["tools"] = self._filter_excluded_tool_definitions(self._default_tool_definitions())

  def _activate_skill_report_doors(self, value: Any) -> None:
    normalized = _normalize_skill_report_doors(value)
    if normalized is not None:
      self._active_skill_report_doors = normalized

  def _activate_skill_deny(self, tool_names: Any, base_kwargs: Dict[str, Any]) -> None:
    denied = _normalize_skill_deny(tool_names)
    if denied is None:
      return
    if denied == self._active_skill_deny:
      return
    self._active_skill_deny = denied
    self._rebuild_filtered_tool_definitions(base_kwargs)

  def _clear_active_skill_if_report_door_completed(self, event: Dict[str, Any], base_kwargs: Dict[str, Any]) -> bool:
    tool_name = str(event.get("tool_name") or "").strip()
    expected_skill = self._active_skill_report_doors.get(tool_name)
    if not _is_report_door_clear_event(
      event,
      expected_skill=expected_skill,
      success_statuses=_REPORT_DOOR_CLEAR_SUCCESS_STATUSES,
    ):
      return False
    try:
      from .skill_context import clear_current_skill, current_skill

      active_skill = current_skill()
      if active_skill != expected_skill:
        return False
      clear_current_skill()
    except Exception:
      return False
    self._active_skill_deny.clear()
    self._active_skill_report_doors.clear()
    self._rebuild_filtered_tool_definitions(base_kwargs)
    return True

  def _refresh_tools(self, base_kwargs: Dict[str, Any], new_servers: List[str]) -> None:
    self._loaded_mcp_servers.update(new_servers)
    self._rebuild_filtered_tool_definitions(base_kwargs)

  async def _emit_stub_response(self, messages: List[Dict[str, Any]]) -> None:
    for event in _build_stub_response_events(messages, provider_name=self._provider.name):
      self._append(event)
      if event.get("type") == "text_delta":
        await asyncio.sleep(0.05)

  def _set_client(self, client: Any) -> None:
    """Track the active httpx client for cleanup on cancellation."""
    self._active_client = client

  async def _close_client(self, client: Any, timeout: float = 2.0) -> None:
    """Close a client and clear tracking if it's the active one."""
    if self._active_client is client:
      self._active_client = None
    await self._provider.close_client(client, timeout=timeout)

  async def force_close(self, timeout: float = 2.0) -> None:
    """Force-close the active client, if any. Safe to call multiple times."""
    client = self._active_client
    if client is not None:
      self._active_client = None
      await self._provider.close_client(client, timeout=timeout)

  async def on_disconnect(self) -> None:
    self._disconnected = True
    self._tool_abort_event.set()
    await asyncio.sleep(0)
    try:
      await self.force_close()
    except Exception as exc:
      log.warning("[%s] force_close on disconnect failed (non-fatal): %s", self._sid, exc)
