from __future__ import annotations

import asyncio
from copy import deepcopy
import fcntl  # noqa: F401 - compatibility alias for runner session lifecycle monkeypatches
import inspect
import json  # noqa: F401 - compatibility alias for runner tool execution monkeypatches
import logging
import socket  # noqa: F401 - compatibility alias for runner session lifecycle monkeypatches
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from agent_workflow_contracts import ResultRequirement

from .agent_session_log import AgentSessionLog
from .auth import ProviderCredentialFailure
from .capability_execution import BoundCapabilityExecution
from .context_builder import SessionContextBuilder
from .context_capture import ContextCapture
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
from .policy_controls import normalizer_excluded_tools
from .runner_introspection import (
  derive_sub_agent_id as _derive_sub_agent_id,  # noqa: F401 - compatibility alias
  detect_keyword_param as _detect_keyword_param,
  detect_user_id_param as _detect_user_id_param,
)
from .runner_limits import (
  CONTEXT_PRESSURE_REMINDER_PCT as CONTEXT_PRESSURE_REMINDER_PCT,
  conservative_request_input_token_bound_for_request as _conservative_request_input_token_bound_for_request,  # noqa: F401 - compatibility alias
  effective_compaction_trigger as _effective_compaction_trigger,  # noqa: F401 - compatibility alias
  model_context_window as _model_context_window,  # noqa: F401 - compatibility alias
  token_estimate_snapshot as _token_estimate_snapshot,  # noqa: F401 - compatibility alias
)
from .runner_prompt_rules import last_user_message as _last_user_message
from .runner_run_loop_defaults import (
  MAX_NOTIFICATIONS_PER_TURN as _MAX_NOTIFICATIONS_PER_TURN,  # noqa: F401 - compatibility alias
  MAX_TOKENS_CONTINUATIONS as _MAX_TOKENS_CONTINUATIONS,  # noqa: F401 - compatibility alias
  MAX_TOKENS_NUDGE as _MAX_TOKENS_NUDGE,  # noqa: F401 - compatibility alias
)
from .runner_background_lifecycle import RunnerBackgroundLifecycleMixin
from .runner_hooks_lifecycle import RunnerHooksLifecycleMixin
from .runner_auth import (
  merge_refreshed_auth_config as _merge_refreshed_auth_config,  # noqa: F401 - compatibility alias
)
from .runner_background_tasks import (
  background_asyncio_tasks as _background_asyncio_tasks,  # noqa: F401 - compatibility alias
  bounded_background_error as _bounded_background_error,
  background_task_started_result as _background_task_started_result,  # noqa: F401 - compatibility alias
  agent_completion_notification as _agent_completion_notification,
  drain_cancelled_background_tasks as _drain_cancelled_background_tasks,  # noqa: F401 - compatibility alias
  drain_still_pending_background_tasks as _drain_still_pending_background_tasks,  # noqa: F401 - compatibility alias
  entry_aware_background_handler as _entry_aware_background_handler,  # noqa: F401 - compatibility alias
  kill_background_tasks as _kill_background_tasks,  # noqa: F401 - compatibility alias
  kill_background_tasks_for_asyncio_tasks as _kill_background_tasks_for_asyncio_tasks,
  prepare_background_task_registration as _prepare_background_task_registration,  # noqa: F401 - compatibility alias
  task_registered_event_payload as _task_registered_event_payload,  # noqa: F401 - compatibility alias
  workflow_owns_terminal_notification as _workflow_owns_terminal_notification,
)
from .runner_callbacks import (
  call_metric_hook as _call_metric_hook,  # noqa: F401 - compatibility alias
)
from .runner_notifications import (
  build_notification_reminder as _build_notification_reminder,  # noqa: F401 - compatibility alias
  consume_notifications as _consume_notifications,  # noqa: F401 - compatibility alias
)
from .runner_session_lifecycle import RunnerSessionLifecycleMixin
from .skill_lifecycle import (
  TopLevelSkillAdmission,
  TopLevelSkillLifecycleMetadata,
  TopLevelSkillResultPolicy,
  TopLevelServerTerminalCause,
)
from .runner_sub_agents import RunnerSubAgentMixin
from .sub_agent_skill_state import result_response_text
from .runner_stream_turn import RunnerStreamTurnMixin
from .runner_run_loop import RunnerRunLoopMixin
from .runner_tool_execution import RunnerToolExecutionMixin
from .workflow_output_attachment import WorkflowOutputAttachment
from .runner_session_events import (
  build_attach_event as _build_attach_event,  # noqa: F401 - compatibility alias
  build_context_warning_log_data as _build_context_warning_log_data,  # noqa: F401 - compatibility alias
  build_orphan_tool_call_interrupted_events as _build_orphan_tool_call_interrupted_events,  # noqa: F401 - compatibility alias
  build_stub_response_events as _build_stub_response_events,
  build_tool_call_complete_event as _build_tool_call_complete_event,  # noqa: F401 - compatibility alias
  build_tool_call_start_event as _build_tool_call_start_event,  # noqa: F401 - compatibility alias
  durable_event_payload as _durable_event_payload,  # noqa: F401 - compatibility alias
  release_write_lease as _release_write_lease,  # noqa: F401 - compatibility alias
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
  budget_cost_progress as _budget_cost_progress,  # noqa: F401 - compatibility alias
  budget_exceeded_state as _budget_exceeded_state,  # noqa: F401 - compatibility alias
  execute_tool_use_loop as _execute_tool_use_loop,  # noqa: F401 - compatibility alias
  normalized_run_config as _normalized_run_config,  # noqa: F401 - compatibility alias
  turn_reminder_state as _turn_reminder_state,  # noqa: F401 - compatibility alias
)
from .runner_streaming import (
  STREAM_STALL_TIMEOUT as STREAM_STALL_TIMEOUT,
  STREAM_THINKING_STALL_TIMEOUT as STREAM_THINKING_STALL_TIMEOUT,
  classify_guard_outcome,  # noqa: F401 - compatibility alias
  effective_stream_stall_timeout,  # noqa: F401 - compatibility alias
  thinking_level,  # noqa: F401 - compatibility alias
)
from .runner_tool_audit import (
  get_tool_risk_value as _get_tool_risk_value,  # noqa: F401 - compatibility alias
  redact_tool_input_for_event as _redact_tool_input_for_event,  # noqa: F401 - compatibility alias
)
from .secret_boundary import SecretBoundary, sanitize_tool_event
from .runner_usage import (
  build_usage_event as _build_usage_event,  # noqa: F401 - compatibility alias
  record_compaction as _record_compaction,  # noqa: F401 - compatibility alias
  usage_delta as _usage_delta,  # noqa: F401 - compatibility alias
  usage_has_tokens as _usage_has_tokens,  # noqa: F401 - compatibility alias
)
from .session_recap import emit_recap_then_terminal
from .session import GatewaySession
from .task_registry import (
  CoordinatorConfig,
  NotificationQueue,
  ParentMessage,
  TaskEntry,
  TaskNotification,
  TaskRegistry,
  TaskState,
)
from .tool_dispatcher import ToolDispatcher
from .tool_display import resolve_display  # noqa: F401 - compatibility alias
from .tool_result_compaction import (
  MODEL_TOOL_RESULT_MAX_CHARS_ENV as MODEL_TOOL_RESULT_MAX_CHARS_ENV,
  SPILL_TRUNCATED_TOOL_RESULTS_ENV as SPILL_TRUNCATED_TOOL_RESULTS_ENV,
  annotate_result,
  compact_model_tool_result_entry as _compact_model_tool_result_entry,
  is_error_tool_result_entry,
  make_error_result,
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
_ACTIVE_SKILL_ALLOW_RESULT_KEY = "_active_skill_allow"
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

  - `capability_execution` freezes the complete capability bind, adapter, and
    credential material before construction.
  - `dispatcher` executes local or MCP tools.
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
    capability_execution: BoundCapabilityExecution,
    allow_stub_response: bool = False,
    client_timeout: float | None = None,
    max_tokens_override: int | None = None,
    per_turn_timeout: float | None = None,
    stream_stall_timeout: float | None = None,
    mcp_client: McpClientManager | None = None,
    loaded_mcp_servers: Set[str] | None = None,
    excluded_tools: Set[str] | None = None,
    purpose: str | None = None,
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
    max_concurrent_sub_agents: int | None = 4,
    result_requirement: ResultRequirement | None = None,
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
    top_level_skill_admission: TopLevelSkillAdmission | None = None,
    top_level_skill_lifecycle: TopLevelSkillLifecycleMetadata | None = None,
    top_level_skill_result_policy: TopLevelSkillResultPolicy | None = None,
    workspace_dir: str | Path | None = None,
    batch_id: int | str | None = None,
    context_surfaces: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]] | None = None,
    context_capture: ContextCapture | None = None,
    commercial_usage_producer: Any | None = None,
    gateway_session: GatewaySession | None = None,
  ) -> None:
    if (
      gateway_session is not None
      and type(gateway_session) is not GatewaySession
    ):
      raise TypeError(
        "AgentRunner gateway_session must be exact GatewaySession"
      )
    if (
      gateway_session is not None
      and (
        session_id != gateway_session.session_id
        or (
          user_id is not None
          and user_id != gateway_session.user_id
        )
        or (
          channel is not None
          and gateway_session.channel is not None
          and channel != gateway_session.channel
        )
      )
    ):
      raise ValueError(
        "AgentRunner identity does not match GatewaySession"
      )
    if not isinstance(capability_execution, BoundCapabilityExecution):
      raise TypeError(
        "AgentRunner requires a BoundCapabilityExecution"
      )
    capability_execution.validate()
    if max_budget_usd is not None and max_budget_usd <= 0:
      raise ValueError("max_budget_usd must be positive when provided")
    if max_concurrent_sub_agents is not None and max_concurrent_sub_agents <= 0:
      raise ValueError("max_concurrent_sub_agents must be positive when provided")
    if max_resume_chain_depth <= 0:
      raise ValueError("max_resume_chain_depth must be positive")
    if (
      result_requirement is not None
      and not isinstance(result_requirement, ResultRequirement)
    ):
      raise TypeError("result_requirement must be a ResultRequirement")
    if result_requirement is not None and result_requirement.mode != "narrative":
      raise ValueError("agent execution accepts terminal-message results only")
    if (
      top_level_skill_lifecycle is not None
      and not isinstance(
        top_level_skill_lifecycle,
        TopLevelSkillLifecycleMetadata,
      )
    ):
      raise TypeError(
        "top_level_skill_lifecycle must be "
        "TopLevelSkillLifecycleMetadata"
      )
    if (
      top_level_skill_result_policy is not None
      and not isinstance(
        top_level_skill_result_policy,
        TopLevelSkillResultPolicy,
      )
    ):
      raise TypeError(
        "top_level_skill_result_policy must be "
        "TopLevelSkillResultPolicy"
      )
    if (
      (top_level_skill_lifecycle is None)
      != (top_level_skill_result_policy is None)
    ):
      raise ValueError(
        "top_level_skill_lifecycle and "
        "top_level_skill_result_policy must be provided together"
      )
    if (
      top_level_skill_admission is not None
      and not isinstance(
        top_level_skill_admission,
        TopLevelSkillAdmission,
      )
    ):
      raise TypeError(
        "top_level_skill_admission must be TopLevelSkillAdmission"
      )
    if (
      top_level_skill_admission is not None
      and top_level_skill_lifecycle is None
    ):
      raise ValueError(
        "top_level_skill_admission requires a top-level lifecycle"
      )
    if (
      top_level_skill_lifecycle is not None
      and agent_session_log is not None
      and top_level_skill_admission is None
    ):
      raise ValueError(
        "Durable top-level skills require a pre-acquired admission"
      )
    if (
      top_level_skill_admission is not None
      and agent_session_log is None
    ):
      raise ValueError(
        "top_level_skill_admission requires an agent session log"
      )
    if (
      top_level_skill_admission is not None
      and top_level_skill_admission.state != "held"
    ):
      raise RuntimeError(
        "AgentRunner requires a held top-level skill admission"
      )
    if skill_run_id is not None and (
      type(skill_run_id) is not str
      or not skill_run_id
      or skill_run_id != skill_run_id.strip()
    ):
      raise ValueError(
        "skill_run_id must be a non-empty string without "
        "surrounding whitespace"
      )
    explicit_skill_run_id = skill_run_id
    if (
      top_level_skill_lifecycle is not None
      and explicit_skill_run_id is not None
      and explicit_skill_run_id
      != top_level_skill_lifecycle.skill_run_id
    ):
      raise ValueError(
        "skill_run_id must match top_level_skill_lifecycle"
      )

    self._log = event_log
    self._dispatcher = dispatcher
    self._gateway_session = gateway_session
    self._spill_dir_provider = normalize_spill_sink(code_execution_spill_dir_provider)
    self._capability_execution = capability_execution
    self._secret_boundary = SecretBoundary.from_capability_execution(
      capability_execution
    )
    bind_dispatcher_boundary = getattr(
      self._dispatcher,
      "bind_secret_boundary",
      None,
    )
    if callable(bind_dispatcher_boundary):
      bind_dispatcher_boundary(self._secret_boundary)
    self._provider = capability_execution.provider
    self._full_session_id = session_id or "no-session"
    self._sid = self._full_session_id[:12]
    self._session_started_at = float(started_at if started_at is not None else time.time())
    self._emit_session_recap = bool(emit_session_recap)
    self._top_level_skill_lifecycle = top_level_skill_lifecycle
    self._top_level_skill_result_policy = top_level_skill_result_policy
    self._top_level_skill_admission = top_level_skill_admission
    self._skill_run_id = (
      top_level_skill_lifecycle.skill_run_id
      if top_level_skill_lifecycle is not None
      else explicit_skill_run_id
    )
    self._workspace_dir = str(workspace_dir) if workspace_dir is not None else None
    self._batch_id = str(batch_id).strip() if batch_id is not None and str(batch_id).strip() else None
    self._auth_config = dict(capability_execution.auth_config)
    self._allow_stub_response = bool(allow_stub_response)
    self._effort_resolution = None
    self._client_timeout = client_timeout
    self._max_tokens_override = max_tokens_override
    self._per_turn_timeout = per_turn_timeout
    self._stream_stall_timeout = stream_stall_timeout
    self._mcp_client = mcp_client
    self._loaded_mcp_servers = loaded_mcp_servers if loaded_mcp_servers is not None else set()
    self._excluded_tools = set(excluded_tools or set())
    self.set_purpose(purpose)
    self._active_skill_allow: set[str] = set()
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
    self._context_capture = context_capture
    self._last_context_manifest_digest: str | None = None
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
    self._context_pressure_next_reminder_pct = (
      CONTEXT_PRESSURE_REMINDER_PCT
    )
    self._portable_compaction_failed_est_at: int | None = None
    self._portable_compaction_last_turn: int | None = None
    self._portable_compaction_floor_warned = False
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
    self._max_background_tasks = max_concurrent_sub_agents or 4
    self._result_requirement = result_requirement
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
      self._dispatcher_accepts_advertised_tool_names = (
        "advertised_tool_names" in dispatch_params
        or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in dispatch_params.values())
      )
    except (TypeError, ValueError):
      self._dispatcher_accepts_abort_event = False
      self._dispatcher_accepts_skill_run_context = False
      self._dispatcher_accepts_readable_resource_snapshot = False
      self._dispatcher_accepts_advertised_tool_names = False
    task_registry_auto_created = task_registry is None
    self._task_registry = task_registry or TaskRegistry(
      max_inflight=self._max_background_tasks,
      id_prefix="bg",
    )
    self._message_inbox = message_inbox
    self._operator_pause_event = operator_pause_event or asyncio.Event()
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
        if _workflow_owns_terminal_notification(entry):
          return
        if new_state in (
          TaskState.COMPLETED,
          TaskState.FAILED,
          TaskState.KILLED,
          TaskState.INTERRUPTED,
        ):
          if entry.completion_envelope is not None:
            notification = _agent_completion_notification(
              entry,
              entry.completion_envelope,
              timestamp=time.time(),
            )
            if notification_queue.push(notification):
              entry.notification_delivery_state = "queued"
            else:
              entry.notification_delivery_state = "queue_omitted"
            return
          event_name = new_state.value
          payload = entry.result or entry.error or {}
          if new_state == TaskState.FAILED and isinstance(
            entry.error,
            dict,
          ):
            payload = _bounded_background_error(entry.error)
          elif new_state == TaskState.INTERRUPTED:
            payload = self._background_task_payload(entry)
          summary = ""
          if entry.result and isinstance(entry.result, dict):
            summary = result_response_text(entry.result)
          elif entry.error and isinstance(entry.error, dict):
            summary = str(payload.get("message", ""))
          notification = TaskNotification(
            task_id=entry.task_id,
            agent_name=entry.agent_name,
            event=event_name,
            summary=summary,
            timestamp=time.time(),
            payload=payload,
            notification_generation=entry.notification_generation,
          )
          _, omission_reason = notification.inline_payload()
          if notification_queue.push(notification):
            entry.notification_delivery_state = (
              "queued"
              if omission_reason is None
              else "payload_omitted"
            )
          else:
            entry.notification_delivery_state = "queue_omitted"

    notifications_enabled = not (
      self._coordinator is not None
      and self._coordinator.enabled
      and not self._coordinator.auto_notify
    )
    self._background_notifications_enabled = notifications_enabled
    if notifications_enabled:
      self._task_registry.add_listener(_NotificationListener())
    self._agent_session_log = agent_session_log
    self._context_builder = context_builder
    self._gateway_session_id = (
      gateway_session.session_id
      if gateway_session is not None
      else self._full_session_id
    )
    self._role = "sub_agent" if self._full_session_id.startswith("sub") and ":" in self._full_session_id else "writer"
    self._sub_agent_id = self._full_session_id if self._role == "sub_agent" else None
    self._client_kind = self._channel or ("cron" if self._agent_session_log is not None else "cli")
    self._runner_id: str | None = None
    self._write_lease_file: Any | None = None
    self._pending_background_completion_appends: set[
      asyncio.Task[Any]
    ] = set()
    self._pending_background_initializations: set[
      asyncio.Task[Any]
    ] = set()
    self._pending_background_result_acks: dict[
      str,
      tuple[str, int],
    ] = {}
    self._deferred_write_lease_release = False
    self._writer_lease_poisoned = False
    self._write_lease_settlement_waiters: set[
      asyncio.Future[Any]
    ] = set()
    self._durable_attach_emitted = False
    self._last_durable_seq = 0
    self._last_assistant_message_seq: int | None = None
    self._pending_workflow_output_attachments: dict[
      str,
      WorkflowOutputAttachment,
    ] = {}
    # Runtime-owned verified workflow evidence projections keyed by
    # workflow_run_id; the final-answer guard consumes these as provenance.
    self._workflow_evidence_provenance: dict[str, dict[str, Any]] = {}
    self._task_registry_rebuild_lock = asyncio.Lock()
    self._task_registry_rebuilt = False
    self._top_level_skill_started_event: dict[str, Any] | None = None
    self._top_level_skill_started_task: asyncio.Task[Any] | None = None
    self._top_level_skill_started_committed = False
    self._top_level_skill_result_event: dict[str, Any] | None = None
    self._top_level_skill_completion_effect_plan: Any | None = None
    self._top_level_skill_result_task: asyncio.Task[Any] | None = None
    self._top_level_skill_result_failure: BaseException | None = None
    self._top_level_skill_result_failure_code: str | None = None
    self._top_level_skill_result_committed = False
    self._top_level_skill_started_projected = False
    self._top_level_skill_result_projected = False
    self._prepared_top_level_terminal_event: (
      dict[str, Any] | None
    ) = None
    self._deferred_top_level_terminal_event: (
      dict[str, Any] | None
    ) = None
    self._deferred_top_level_terminal_flushed = False
    self._top_level_skill_terminal_committed = False
    self._deferred_top_level_interrupted_event: (
      dict[str, Any] | None
    ) = None
    self._deferred_top_level_interrupted_flushed = False
    self._top_level_skill_detach_committed = False
    self._top_level_skill_settlement_complete = asyncio.Event()
    self._top_level_skill_settlement_error: (
      BaseException | None
    ) = None
    if top_level_skill_admission is not None:
      self._write_lease_file = top_level_skill_admission.transfer(
        log_path=agent_session_log.path,
        write_lease_path=agent_session_log.write_lease_path,
      )

  @property
  def _background_tasks(self) -> Dict[str, TaskEntry]:
    return self._task_registry._tasks

  @property
  def capability_execution(self) -> BoundCapabilityExecution:
    return self._capability_execution

  @property
  def committed_top_level_skill_result_event(
    self,
  ) -> dict[str, Any] | None:
    """Return a defensive copy of the committed top-level skill receipt."""

    if not self._top_level_skill_result_committed:
      return None
    event = self._top_level_skill_result_event
    if event is None:
      raise RuntimeError(
        "Committed top-level skill result is missing its event"
      )
    return deepcopy(event)

  @property
  def top_level_skill_enrolled(self) -> bool:
    return self._top_level_skill_lifecycle is not None

  def classify_server_cancellation_cause(
    self,
  ) -> TopLevelServerTerminalCause:
    """Classify an external cancellation without exception-text inference."""

    provider = self._shutdown_signal_provider
    if provider is not None:
      try:
        if provider():
          return "shutdown"
      except Exception:
        pass
    return "caller_cancellation"

  def set_server_terminal_cause(
    self,
    cause: TopLevelServerTerminalCause,
  ) -> bool:
    """Record a typed cause before the completion fence becomes irrevocable."""

    policy = self._top_level_skill_result_policy
    if policy is None or self._top_level_skill_result_committed:
      return False
    return policy.set_server_terminal_cause(cause)

  async def wait_for_top_level_skill_settlement(self) -> bool:
    """Wait until receipt, terminal, detach, and lease handoff all settle."""

    if self._top_level_skill_lifecycle is None:
      return False
    await self._top_level_skill_settlement_complete.wait()
    error = self._top_level_skill_settlement_error
    if error is not None:
      raise error
    if self._write_lease_file is not None:
      raise RuntimeError(
        "Top-level skill settlement completed before lease handoff"
      )
    if not self._top_level_skill_result_committed:
      raise RuntimeError(
        "Top-level skill settlement completed without a result receipt"
      )
    if self._deferred_top_level_terminal_event is None:
      raise RuntimeError(
        "Top-level skill settlement completed without a terminal event"
      )
    if not self._deferred_top_level_terminal_flushed:
      raise RuntimeError(
        "Top-level skill settlement completed before terminal projection"
      )
    if (
      self._durable_attach_emitted
      and not self._top_level_skill_detach_committed
    ):
      raise RuntimeError(
        "Top-level skill settlement completed before durable detach"
      )
    return True

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

  async def _call_on_tool_result(
    self,
    ctx: ToolResultContext,
  ) -> List[Dict[str, Any]]:
    extra_blocks = await RunnerHooksLifecycleMixin._call_on_tool_result(
      self,
      ctx,
    )
    policy = self._top_level_skill_result_policy
    observer = (
      policy.terminal_tool_result_observer
      if policy is not None
      else None
    )
    if observer is None:
      return extra_blocks
    disposition = observer(ctx)
    if inspect.isawaitable(disposition):
      disposition = await disposition
    if disposition is not None:
      if disposition not in {"success", "failure"}:
        raise RuntimeError(
          "Top-level terminal tool observer must return exactly "
          "'success', 'failure', or None"
        )
      self._stop_after_tool_results_reason = (
        "terminal_tool_result"
        if disposition == "success"
        else "terminal_tool_failure"
      )
      self._stop_after_tool_results_tool_name = getattr(
        ctx,
        "tool_name",
        None,
      )
      result = getattr(ctx, "result", None)
      self._stop_after_tool_results_status = (
        result.get("status")
        if isinstance(result, dict)
        else None
      )
      self._stop_after_tool_results_skill_run_id = getattr(
        ctx,
        "skill_run_id",
        None,
      )
    return extra_blocks

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
      log.warning(
        "[%s] context surface provider failed (non-fatal) | exception_type=%s",
        self._sid,
        type(exc).__name__,
      )
      return self._normalize_context_surfaces(self._context_surfaces_static)

  def _append(self, event: Dict[str, Any]) -> Any | None:
    event = sanitize_tool_event(
      event,
      sink="event_log",
      boundary=getattr(self, "_secret_boundary", None),
    )
    if event.get("type") == "compaction":
      _record_compaction(self._aggregator)
      self._context_pressure_next_reminder_pct = (
        CONTEXT_PRESSURE_REMINDER_PCT
      )
    if event.get("type") in {"stream_complete", "error"}:
      return emit_recap_then_terminal(
        self._log,
        event,
        session_id=self._full_session_id,
        started_at=self._session_started_at,
        emit_recap=self._emit_session_recap,
      )
    return self._log.append(event)

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

  def set_purpose(self, purpose: str | None) -> None:
    self._purpose = (
      purpose.strip().lower()
      if isinstance(purpose, str) and purpose.strip()
      else None
    )
  def _effective_excluded_tools(self) -> set[str]:
    excluded = _effective_excluded_tools(
      self._excluded_tools,
      self._active_skill_deny,
      self._active_skill_allow,
    )
    if self._purpose == "normalizer":
      excluded.update(normalizer_excluded_tools())
    return excluded

  def _filter_excluded_tool_definitions(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _filter_excluded_tool_definitions(tools, self._effective_excluded_tools())

  def _rebuild_filtered_tool_definitions(self, base_kwargs: Dict[str, Any]) -> None:
    base_kwargs["tools"] = self._filter_excluded_tool_definitions(self._default_tool_definitions())

  def _activate_skill_report_doors(self, value: Any) -> None:
    normalized = _normalize_skill_report_doors(value)
    if normalized is not None:
      self._active_skill_report_doors = normalized

  def _activate_skill_allow(self, tool_names: Any, base_kwargs: Dict[str, Any]) -> None:
    allowed = _normalize_skill_deny(tool_names)
    if allowed is None:
      return
    if allowed == self._active_skill_allow:
      return
    self._active_skill_allow = allowed
    self._rebuild_filtered_tool_definitions(base_kwargs)

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
    self._active_skill_allow.clear()
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
