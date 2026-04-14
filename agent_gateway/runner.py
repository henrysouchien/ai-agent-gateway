from __future__ import annotations

import asyncio
import fcntl
import inspect
import json
import logging
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Union

from .agent_session_log import AgentSessionLog
from .context_builder import SessionContextBuilder
from .event_log import EventLog
from .mcp_client import McpClientManager
from .multi_user.billing import DEFAULT_USAGE_DLQ_PATH, UsageEvent, write_dlq
from .providers import ModelInfo, ModelProvider, ThinkingLevel
from .task_registry import (
  COORDINATOR_DEFAULT_PREAMBLE,
  CoordinatorConfig,
  NotificationQueue,
  TaskEntry,
  TaskNotification,
  TaskRegistry,
  TaskState,
  make_progress_tracker,
)
from .tool_dispatcher import ToolDispatcher


log = logging.getLogger("agent_gateway.runner")
MODEL_CONTEXT_LIMIT = 200_000
CONTEXT_WARNING_PCT = 80
STREAM_STALL_TIMEOUT = 60  # max seconds between stream events before watchdog cancels
STREAM_RETRY_MAX = 3
STREAM_RETRY_DELAY = 2.0
STREAM_RETRY_BACKOFF = 2.0
_MAX_NOTIFICATIONS_PER_TURN = 5


def _estimate_tokens(text: str) -> int:
  """Rough token estimate: ~4 chars per token for English text + JSON overhead."""
  return max(1, len(text) // 4)


def _format_exc(exc: Exception) -> str:
  parts = [f"{type(exc).__name__}: {repr(exc)}"]
  seen = {id(exc)}
  cause = exc.__cause__
  while cause is not None and id(cause) not in seen:
    parts.append(f"caused by {type(cause).__name__}: {repr(cause)}")
    seen.add(id(cause))
    cause = cause.__cause__
  return " | ".join(parts)


def _get_tool_risk_value(tool_name: str) -> str:
  try:
    from api.agent.shared.tool_risk import get_tool_risk
  except Exception:
    return "side_effecting"
  try:
    return get_tool_risk(tool_name).value
  except Exception:
    return "side_effecting"


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


class CostAccumulator:
  """Running cost tracker shared across parent and sub-agent runners."""

  def __init__(self, budget: float) -> None:
    self.budget = budget
    self._total = 0.0

  def add(self, cost: float) -> None:
    self._total += cost

  @property
  def total(self) -> float:
    return self._total

  @property
  def exceeded(self) -> bool:
    return self._total >= self.budget


OnToolResult = Callable[[ToolResultContext], Awaitable[List[Dict[str, Any]] | None]]
OnUsage = Callable[[UsageEvent], Awaitable[None] | None]
OnToolTiming = Callable[[str, str, str | None, int, bool, int], None]
OnMaxTurns = Callable[[List[Dict[str, Any]], int], Awaitable[str | None]]
BackgroundTaskHandler = Callable[..., Awaitable[Tuple[Optional[Any], Optional[Dict[str, Any]]]]]
BackgroundTaskCallback = Callable[[BackgroundTask | TaskEntry], Awaitable[None] | None]
OnMetric = Callable[[str, int], None]


class AgentRunner:
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
    on_tool_timing: OnToolTiming | None = None,
    user_id: str | None = None,
    request_id: str | None = None,
    parent_turn_id: str | None = None,
    billing_mode: str = "byok",
    rate_table_version: str = "unknown",
    channel: str | None = None,
    usage_ledger_dlq_path: Path | str | None = None,
    on_metric: OnMetric | None = None,
    sub_agent_config: SubAgentConfig | None = None,
    compaction_trigger: int | None = None,
    compaction_instructions: str | None = None,
    tool_call_timeout: float | None = 120.0,
    on_max_turns: OnMaxTurns | None = None,
    max_budget_usd: float | None = None,
    _cost_accumulator: CostAccumulator | None = None,
    max_concurrent_sub_agents: int | None = None,
    agent_session_log: AgentSessionLog | None = None,
    context_builder: SessionContextBuilder | None = None,
    task_registry: TaskRegistry | None = None,
    message_inbox: asyncio.Queue[str] | None = None,
    coordinator: CoordinatorConfig | None = None,
  ) -> None:
    if max_budget_usd is not None and max_budget_usd <= 0:
      raise ValueError("max_budget_usd must be positive when provided")
    if max_concurrent_sub_agents is not None and max_concurrent_sub_agents <= 0:
      raise ValueError("max_concurrent_sub_agents must be positive when provided")

    self._log = event_log
    self._dispatcher = dispatcher
    self._provider = provider
    self._full_session_id = session_id or "no-session"
    self._sid = self._full_session_id[:12]
    self._auth_config = dict(auth_config or {})
    self._client_timeout = client_timeout
    self._max_tokens_override = max_tokens_override
    self._per_turn_timeout = per_turn_timeout
    self._stream_stall_timeout = stream_stall_timeout
    self._mcp_client = mcp_client
    self._loaded_mcp_servers = loaded_mcp_servers if loaded_mcp_servers is not None else set()
    self._excluded_tools = set(excluded_tools or set())
    self._get_tool_definitions = get_tool_definitions
    self._on_tool_result = on_tool_result
    self._on_usage = on_usage
    self._on_tool_timing = on_tool_timing
    self._usage_user_id = str(user_id or "_default")
    self._request_id = str(request_id or uuid.uuid4())
    self._parent_turn_id = parent_turn_id
    self._billing_mode = "metered" if str(billing_mode).strip().lower() == "metered" else "byok"
    self._rate_table_version = str(rate_table_version or "unknown")
    self._channel = channel.strip() if isinstance(channel, str) and channel.strip() else None
    self._usage_ledger_dlq_path = (
      Path(usage_ledger_dlq_path).expanduser() if usage_ledger_dlq_path is not None else DEFAULT_USAGE_DLQ_PATH
    )
    self._on_metric = on_metric
    self._sub_agent_config = sub_agent_config
    self._compaction_trigger = compaction_trigger
    self._compaction_instructions = compaction_instructions
    self._tool_call_timeout = tool_call_timeout
    self._on_max_turns = on_max_turns
    self._max_budget_usd = max_budget_usd if max_budget_usd is not None else (
      _cost_accumulator.budget if _cost_accumulator is not None else None
    )
    self._cost_accumulator = _cost_accumulator
    if self._cost_accumulator is None and self._max_budget_usd is not None:
      self._cost_accumulator = CostAccumulator(self._max_budget_usd)
    self._last_reported_cost = 0.0
    self._coordinator = coordinator
    self._max_concurrent_sub_agents = max_concurrent_sub_agents
    self._max_background_tasks = max_concurrent_sub_agents or 3
    self._sub_agent_semaphore: asyncio.Semaphore | None = None
    self._active_client: Any | None = None
    task_registry_auto_created = task_registry is None
    self._task_registry = task_registry or TaskRegistry(
      max_inflight=self._max_background_tasks,
      id_prefix="bg",
    )
    self._message_inbox = message_inbox
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

  @property
  def _background_tasks(self) -> Dict[str, TaskEntry]:
    return self._task_registry._tasks

  def _append(self, event: Dict[str, Any]) -> None:
    self._log.append(event)

  def _extract_last_user_message(self, request_messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for msg in reversed(request_messages):
      if msg.get("role") == "user":
        return dict(msg)
    return None

  async def _append_durable_event(self, event: Dict[str, Any]) -> Any | None:
    if self._agent_session_log is None or self._runner_id is None:
      return None
    payload = dict(event)
    payload.setdefault("runner_id", self._runner_id)
    payload.setdefault("role", self._role)
    if self._sub_agent_id is not None:
      payload.setdefault("sub_agent_id", self._sub_agent_id)
    entry = await self._agent_session_log.append(payload)
    self._last_durable_seq = entry.seq
    return entry

  async def _emit_attach_event(self) -> None:
    entry = await self._append_durable_event(
      {
        "type": "attach",
        "gateway_session_id": self._gateway_session_id,
        "started_at": time.time(),
        "client_kind": self._client_kind,
        "hostname": socket.gethostname(),
      }
    )
    self._durable_attach_emitted = entry is not None

  async def _append_user_message_event(self, message: Dict[str, Any]) -> None:
    await self._append_durable_event(
      {
        "type": "user_message",
        "content": message.get("content"),
        "client_kind": self._client_kind,
        "received_at": time.time(),
      }
    )

  async def _append_assistant_message_event(
    self,
    *,
    content_blocks: List[Dict[str, Any]],
    stop_reason: str | None,
    model: str,
    usage: Dict[str, int],
  ) -> None:
    entry = await self._append_durable_event(
      {
        "type": "assistant_message",
        "content_blocks": list(content_blocks),
        "stop_reason": stop_reason,
        "model": model,
        "usage": dict(usage),
      }
    )
    if entry is not None:
      self._last_assistant_message_seq = entry.seq

  async def _emit_interrupted_event(
    self,
    reason: str,
    *,
    runner_id: str | None = None,
    role: str | None = None,
    last_completed_seq: int | None = None,
    recovered_by_runner_id: str | None = None,
    recovered_at: float | None = None,
    extra_fields: Dict[str, Any] | None = None,
  ) -> None:
    payload: Dict[str, Any] = {
      "type": "interrupted",
      "reason": reason,
      "runner_id": runner_id or self._runner_id,
      "role": role or self._role,
      "last_completed_seq": self._last_durable_seq if last_completed_seq is None else last_completed_seq,
    }
    if recovered_by_runner_id is not None:
      payload["recovered_by_runner_id"] = recovered_by_runner_id
    if recovered_at is not None:
      payload["recovered_at"] = recovered_at
    if extra_fields:
      payload.update(extra_fields)
    await self._append_durable_event(payload)

  async def _emit_detach_event(self, reason: str) -> None:
    if not self._durable_attach_emitted:
      return
    await self._append_durable_event(
      {
        "type": "detach",
        "reason": reason,
        "ended_at": time.time(),
      }
    )

  async def _acquire_writer_lease_and_recover(self) -> None:
    if self._agent_session_log is None or self._role != "writer":
      return

    lease_file = self._agent_session_log.write_lease_path.open("a+b")
    try:
      fcntl.flock(lease_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
      lease_file.close()
      raise RuntimeError(f"Writer lease already held for {self._agent_session_log.path}") from exc
    self._write_lease_file = lease_file

    last_known_safe_seq = 0
    safe_entries, _ = await self._agent_session_log.query(
      event_types={"detach", "interrupted"},
      role="writer",
      order="desc",
      limit=1,
    )
    if safe_entries:
      last_known_safe_seq = safe_entries[0].seq
      self._last_durable_seq = last_known_safe_seq

    prior_writer_runner_id: str | None = None
    writer_lifecycle, _ = await self._agent_session_log.query(
      event_types={"attach", "detach", "interrupted"},
      role="writer",
      order="desc",
      limit=1,
    )
    if writer_lifecycle and writer_lifecycle[0].event.get("type") == "attach":
      prior_writer_runner_id = str(writer_lifecycle[0].event.get("runner_id") or "")

    orphan_entries, _ = await self._agent_session_log.query(
      event_types={"tool_call_start", "tool_call_complete", "tool_call_interrupted"},
      after_seq=last_known_safe_seq + 1,
      order="asc",
    )
    starts: Dict[str, Dict[str, Any]] = {}
    resolved_tool_ids: Set[str] = set()
    for entry in orphan_entries:
      event = entry.event
      tool_call_id = str(event.get("tool_call_id") or "")
      if not tool_call_id:
        continue
      event_type = str(event.get("type") or "")
      if event_type == "tool_call_start":
        starts.setdefault(tool_call_id, event)
      elif event_type in {"tool_call_complete", "tool_call_interrupted"}:
        resolved_tool_ids.add(tool_call_id)

    discovered_at = time.time()
    for tool_call_id, start_event in starts.items():
      if tool_call_id in resolved_tool_ids:
        continue
      synthetic_event: Dict[str, Any] = {
        "type": "tool_call_interrupted",
        "tool_call_id": tool_call_id,
        "tool_name": start_event.get("tool_name"),
        "tool_input": start_event.get("tool_input"),
        "original_started_at": start_event.get("started_at"),
        "discovered_at": discovered_at,
        "tool_risk": _get_tool_risk_value(str(start_event.get("tool_name") or "")),
        "runner_id": start_event.get("runner_id"),
        "role": start_event.get("role", "writer"),
      }
      if start_event.get("sub_agent_id") is not None:
        synthetic_event["sub_agent_id"] = start_event.get("sub_agent_id")
      await self._append_durable_event(synthetic_event)

    if prior_writer_runner_id:
      await self._emit_interrupted_event(
        "recovered_on_attach",
        runner_id=prior_writer_runner_id,
        role="writer",
        last_completed_seq=last_known_safe_seq,
        recovered_by_runner_id=self._runner_id,
        recovered_at=discovered_at,
      )

  def _write_lease_metadata(self) -> None:
    if self._agent_session_log is None or self._role != "writer" or self._runner_id is None:
      return
    payload = {
      "runner_id": self._runner_id,
      "gateway_session_id": self._gateway_session_id,
      "started_at": time.time(),
      "hostname": socket.gethostname(),
    }
    self._agent_session_log.write_lease_meta_path.write_text(
      json.dumps(payload, sort_keys=True),
      encoding="utf-8",
    )

  def _release_write_lease(self) -> None:
    if self._write_lease_file is None:
      return
    try:
      self._write_lease_file.close()
    finally:
      self._write_lease_file = None

  @staticmethod
  def _annotate_result(result: Any, tool_name: str = "") -> Any:
    """Add _runner_warning to generic results with detectable anomalies."""
    if not isinstance(result, dict):
      return result

    warnings: List[str] = []
    interceptor_warnings = result.pop("_interceptor_warnings", None)
    if isinstance(interceptor_warnings, list):
      for w in interceptor_warnings:
        warnings.append(f"Policy warning: {w}")

    low_match = result.get("low_match_warning")
    if low_match:
      warnings.append(f"Low match rate detected: {low_match}")

    if tool_name == "run_agent":
      sub_warning = result.get("warning")
      if sub_warning:
        warnings.append(f"Sub-agent warning: {sub_warning}")

    if not warnings:
      return result

    enriched = dict(result)
    enriched["_runner_warning"] = " | ".join(warnings)
    if low_match:
      enriched["_runner_warning_detail"] = str(low_match)
    return enriched

  @staticmethod
  def _make_error_result(
    tool_use_id: str,
    code: str,
    message: str,
    sub_code: str = "",
  ) -> Dict[str, Any]:
    error_dict = {"code": code, "message": message}
    if sub_code:
      error_dict["sub_code"] = sub_code
    return {
      "type": "tool_result",
      "tool_use_id": tool_use_id,
      "content": json.dumps({"error": error_dict}),
      "is_error": True,
    }

  @staticmethod
  def _is_soft_error(result: Any) -> bool:
    if not isinstance(result, dict):
      return False
    if result.get("success") is False:
      return True
    if result.get("status") == "error":
      return True
    return False

  def _default_tool_definitions(self) -> List[Dict[str, Any]]:
    if self._get_tool_definitions is not None:
      return list(self._get_tool_definitions())
    if self._mcp_client is not None:
      return self._mcp_client.get_tool_definitions()
    return []

  def _refresh_tools(self, base_kwargs: Dict[str, Any], new_servers: List[str]) -> None:
    self._loaded_mcp_servers.update(new_servers)
    new_tools = self._default_tool_definitions()
    if self._excluded_tools:
      new_tools = [tool for tool in new_tools if tool["name"] not in self._excluded_tools]
    base_kwargs["tools"] = new_tools

  async def _emit_stub_response(self, messages: List[Dict[str, Any]]) -> None:
    last_user = next((msg for msg in reversed(messages) if msg.get("role") == "user"), {})
    prompt = last_user.get("content") or "your request"
    response = f"Stub response (no {self._provider.name.title()} credential configured). You asked: {prompt}"
    for token in response.split():
      self._append({"type": "text_delta", "text": token + " "})
      await asyncio.sleep(0.05)
    self._append({"type": "stream_complete", "usage": {}})

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

  async def spawn_sub_agent(
    self,
    task: str,
    *,
    provider: ModelProvider | None = None,
    auth_config: Dict[str, Any] | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    dispatcher: ToolDispatcher,
    sub_session: Any | None = None,
    excluded_tools: Set[str] | None = None,
    max_turns: int | None,
    timeout: float | None,
    client_timeout: float = 90,
    per_turn_timeout: float | None = None,
    max_tokens: int = 32000,
    call_index: int = 0,
    parent_turn_id: str | None = None,
    task_entry: TaskEntry | None = None,
    on_sub_event: Optional[Callable[[Dict[str, Any], str], None]] = None,
  ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Run a focused sub-agent task and return its summarized result.

    This method is used by the built-in `run_agent` tool. The sub-agent shares
    the same budget accounting as the parent runner, but it gets a fresh
    `EventLog`, its own turn budget, and its own dispatcher. By default the
    sub-agent inherits the parent's provider, but callers may override it with
    an explicit `provider` + `auth_config` pair.
    """
    if self._sub_agent_config is not None:
      if model is None:
        model = self._sub_agent_config.model
      if system_prompt is None:
        system_prompt = self._sub_agent_config.system_prompt
      if excluded_tools is None:
        excluded_tools = set(self._sub_agent_config.excluded_tools)

    effective_provider = provider or self._provider
    if provider is not None:
      if auth_config is None:
        return None, {"code": "invalid_input", "message": "auth_config required when overriding provider"}
      effective_auth = dict(auth_config)
    else:
      effective_auth = getattr(sub_session, "auth_config", None) or self._auth_config

    sub_session_id = str(getattr(sub_session, "session_id", "") or f"sub{call_index}:{self._sid}")
    original_on_event = getattr(self._log, "_on_event", None)
    progress_cb = make_progress_tracker(task_entry) if task_entry else None

    def _composed_on_event(event: Dict[str, Any], session_id: str) -> None:
      if progress_cb is not None:
        try:
          progress_cb(event, session_id)
        except Exception:
          pass
      if original_on_event is not None:
        try:
          original_on_event(event, session_id)
        except Exception:
          pass
      if on_sub_event is not None:
        try:
          on_sub_event(event, session_id)
        except Exception:
          pass

    sub_log = EventLog(
      on_event=_composed_on_event,
      session_id=sub_session_id,
    )
    sub_runner = AgentRunner(
      event_log=sub_log,
      dispatcher=dispatcher,
      session_id=sub_session_id,
      provider=effective_provider,
      auth_config=effective_auth,
      client_timeout=client_timeout,
      max_tokens_override=max_tokens,
      per_turn_timeout=per_turn_timeout if per_turn_timeout is not None else self._per_turn_timeout,
      stream_stall_timeout=self._stream_stall_timeout,
      mcp_client=self._mcp_client,
      loaded_mcp_servers=self._loaded_mcp_servers,
      excluded_tools=excluded_tools or set(),
      get_tool_definitions=self._get_tool_definitions,
      on_tool_result=self._on_tool_result,
      on_usage=self._on_usage,
      on_tool_timing=self._on_tool_timing,
      user_id=getattr(sub_session, "user_id", None) or self._usage_user_id,
      request_id=self._request_id,
      parent_turn_id=parent_turn_id,
      billing_mode=self._billing_mode,
      rate_table_version=self._rate_table_version,
      channel=self._channel,
      usage_ledger_dlq_path=self._usage_ledger_dlq_path,
      on_metric=self._on_metric,
      sub_agent_config=self._sub_agent_config,
      compaction_trigger=self._compaction_trigger,
      compaction_instructions=None,
      tool_call_timeout=self._tool_call_timeout,
      on_max_turns=self._on_max_turns,
      max_budget_usd=self._max_budget_usd,
      _cost_accumulator=self._cost_accumulator,
      max_concurrent_sub_agents=self._max_concurrent_sub_agents,
      agent_session_log=self._agent_session_log,
      message_inbox=task_entry.message_inbox if task_entry else None,
    )

    timed_out = False
    coro = sub_runner.run(
      messages=[{"role": "user", "content": task}],
      system_prompt=system_prompt,
      model_override=model,
      max_turns=max_turns,
    )
    try:
      if timeout is not None and timeout > 0:
        await asyncio.wait_for(coro, timeout=timeout)
      else:
        await coro
    except asyncio.TimeoutError:
      timed_out = True
      sub_log.append({"type": "error", "error": f"Sub-agent timed out after {timeout}s"})
    except asyncio.CancelledError:
      log.warning("[%s] Sub-agent cancelled (parent disconnect or shutdown)", sub_session_id)
      sub_log.append({"type": "error", "error": "Sub-agent cancelled"})
      raise
    finally:
      await sub_runner.force_close(timeout=2.0)

    text_parts: List[str] = []
    tool_calls_made: List[str] = []
    usage: Dict[str, Any] = {}
    error_msg: str | None = None
    budget_exceeded = False
    max_turns_hit = False
    for entry in sub_log.entries:
      event = entry.event
      event_type = event.get("type")
      if event_type == "stream_retry":
        text_parts.clear()
        tool_calls_made.clear()
      elif event_type == "text_delta":
        text_parts.append(str(event.get("text", "")))
      elif event_type == "tool_call_start":
        tool_calls_made.append(str(event.get("tool_name", "")))
      elif event_type == "stream_complete":
        event_usage = event.get("usage")
        if isinstance(event_usage, dict):
          usage = event_usage
      elif event_type == "budget_exceeded":
        budget_exceeded = True
      elif event_type == "max_turns_reached":
        max_turns_hit = True
      elif event_type == "error":
        error_msg = str(event.get("error", "Sub-agent error"))

    result: Dict[str, Any] = {
      "response": "".join(text_parts).strip(),
      "tools_used": tool_calls_made,
      "usage": usage,
    }
    warnings: List[str] = []
    if timed_out:
      warnings.append(f"Sub-agent timed out after {timeout}s — partial results returned")
    elif error_msg:
      warnings.append(f"Sub-agent error: {error_msg}")
    if budget_exceeded:
      warnings.append("Sub-agent stopped: budget limit reached")
    if max_turns_hit:
      warnings.append("Sub-agent stopped: max turns reached — partial results")
    if warnings:
      result["warning"] = "; ".join(warnings)
    return result, None

  async def _call_on_tool_result(self, ctx: ToolResultContext) -> List[Dict[str, Any]]:
    if self._on_tool_result is None:
      return []
    try:
      extra_blocks = await self._on_tool_result(ctx)
    except Exception as exc:
      log.warning("[%s] on_tool_result hook failed (non-fatal): %s", self._sid, exc)
      return []
    if not extra_blocks:
      return []
    if isinstance(extra_blocks, list):
      return [block for block in extra_blocks if isinstance(block, dict)]
    return []

  def _ensure_sub_agent_semaphore(self) -> asyncio.Semaphore | None:
    if self._sub_agent_semaphore is None and self._max_concurrent_sub_agents is not None:
      self._sub_agent_semaphore = asyncio.Semaphore(self._max_concurrent_sub_agents)
    return self._sub_agent_semaphore

  @staticmethod
  def _background_timeout_value(raw_timeout: Any) -> float:
    timeout = 60.0 if raw_timeout is None else float(raw_timeout)
    return max(0.0, min(timeout, 120.0))

  def _background_elapsed_seconds(self, bg_task: BackgroundTask | TaskEntry) -> int:
    end_t = bg_task.completed_at if bg_task.completed_at is not None else time.time()
    return max(0, int(end_t - bg_task.started_at))

  def _background_task_payload(self, bg_task: BackgroundTask | TaskEntry) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
      "task_id": bg_task.task_id,
      "status": "running",
    }
    if bg_task.agent_name:
      payload["agent"] = bg_task.agent_name

    elapsed = self._background_elapsed_seconds(bg_task)
    if getattr(bg_task, "state", None) == TaskState.KILLED:
      payload["status"] = "killed"
      payload["elapsed_seconds"] = elapsed
      return payload
    if bg_task.completed:
      payload["elapsed_seconds"] = elapsed
      if bg_task.error is not None:
        payload["status"] = "error"
        payload["error"] = bg_task.error
        return payload
      payload["status"] = "completed"
      if isinstance(bg_task.result, dict):
        for key, value in bg_task.result.items():
          if key not in {"task_id", "status", "agent"}:
            payload[key] = value
      elif bg_task.result is not None:
        payload["result"] = bg_task.result
      return payload

    payload["elapsed_seconds"] = elapsed
    progress = getattr(bg_task, "progress", None)
    if progress is not None and progress.tool_use_count > 0:
      payload["progress"] = {
        "tools_used": progress.tool_use_count,
        "turns": progress.turn_count,
        "last_tool": progress.last_tool_name,
        "idle_seconds": int(time.time() - progress.last_activity_at) if progress.last_activity_at else None,
        "output_tokens": progress.output_tokens,
      }
    return payload

  def _background_task_reminder_text(self) -> str:
    running_tasks = self._task_registry.list_tasks(state=TaskState.RUNNING)
    if not running_tasks:
      return ""
    entries: List[str] = []
    for bg_task in running_tasks:
      parts = [f"running, {self._background_elapsed_seconds(bg_task)}s"]
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
    return "[Background tasks active: " + ", ".join(entries) + "]"

  def _build_notification_reminder(self) -> str:
    """Peek notifications into system prompt text. Non-destructive."""
    if self._notification_queue.pending_count == 0:
      return ""
    notifications = self._notification_queue.peek(max_count=_MAX_NOTIFICATIONS_PER_TURN)
    parts = [notification.format_xml() for notification in notifications]
    remaining = self._notification_queue.pending_count - len(notifications)
    if remaining > 0:
      parts.append(f"[{remaining} more task notification(s) pending]")
    return "\n".join(parts)

  def _consume_notifications(self, max_count: int) -> int:
    """Drain up to max_count notifications. Returns count consumed."""
    return len(self._notification_queue.drain(max_count=max_count))

  @staticmethod
  def _inject_system_prompt_reminder(
    system_prompt: Optional[Union[str, List[Tuple[str, bool]]]],
    reminder: str,
  ) -> Optional[Union[str, List[Tuple[str, bool]]]]:
    if not reminder:
      return system_prompt
    if isinstance(system_prompt, list):
      return [*system_prompt, (reminder, False)]
    base = system_prompt or ""
    if base:
      return f"{base}\n\n{reminder}"
    return reminder

  async def _run_background_agent(
    self,
    bg_task: TaskEntry,
    handler: BackgroundTaskHandler,
    tool_input: Dict[str, Any],
    call_index: int,
    on_complete: BackgroundTaskCallback | None = None,
  ) -> None:
    try:
      semaphore = self._ensure_sub_agent_semaphore()
      if semaphore is not None:
        async with semaphore:
          result, error = await handler(tool_input, call_index=call_index)
      else:
        result, error = await handler(tool_input, call_index=call_index)
      bg_task.result = result if isinstance(result, dict) else ({"result": result} if result is not None else None)
      bg_task.error = error
      if bg_task.error is not None:
        self._task_registry.transition(bg_task.task_id, TaskState.FAILED, error=bg_task.error)
      else:
        self._task_registry.transition(bg_task.task_id, TaskState.COMPLETED, result=bg_task.result)
    except asyncio.CancelledError:
      bg_task.error = {"code": "cancelled", "message": "Background task was cancelled"}
      self._task_registry.transition(bg_task.task_id, TaskState.FAILED, error=bg_task.error)
    except Exception as exc:
      bg_task.error = {"code": "background_error", "message": str(exc)}
      self._task_registry.transition(bg_task.task_id, TaskState.FAILED, error=bg_task.error)
    finally:
      if on_complete is not None:
        try:
          maybe_awaitable = on_complete(bg_task)
          if asyncio.iscoroutine(maybe_awaitable):
            await maybe_awaitable
        except Exception:
          pass

  async def _register_background_task(
    self,
    *,
    tool_input: Dict[str, Any],
    handler: BackgroundTaskHandler,
    agent_name: str | None = None,
    on_complete: BackgroundTaskCallback | None = None,
    on_before_start: Callable[[], None] | None = None,
  ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if self._task_registry.inflight_count >= self._max_background_tasks:
      return None, {
        "code": "max_background_tasks",
        "message": (
          f"Background task limit reached ({self._max_background_tasks}). "
          "Wait for an existing background task to finish before launching another."
        ),
      }

    if on_before_start is not None:
      try:
        on_before_start()
      except Exception as exc:
        log.warning("[%s] on_before_start hook failed (non-fatal): %s", self._sid, exc)

    entry = self._task_registry.register("background_agent", agent_name=agent_name)
    raw_provider_name = tool_input.get("provider_name", tool_input.get("provider"))
    if isinstance(raw_provider_name, str) and raw_provider_name.strip():
      entry.provider_name = raw_provider_name.strip()
    else:
      entry.provider_name = getattr(self._provider, "name", None)
    raw_model = tool_input.get("model")
    if isinstance(raw_model, str) and raw_model.strip():
      entry.model = raw_model.strip()
    else:
      auth_model = self._auth_config.get("model")
      if isinstance(auth_model, str) and auth_model.strip():
        entry.model = auth_model.strip()

    try:
      call_index = int(entry.task_id.rsplit("_", 1)[-1])
    except ValueError:
      call_index = 0

    async def _entry_aware_handler(ti: Dict[str, Any], **kwargs: Any) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
      kwargs["task_entry"] = entry
      return await handler(ti, **kwargs)

    entry.asyncio_task = asyncio.create_task(
      self._run_background_agent(
        entry,
        _entry_aware_handler,
        dict(tool_input),
        call_index,
        on_complete=on_complete,
      ),
      name=entry.task_id,
    )
    self._task_registry.transition(entry.task_id, TaskState.RUNNING)

    result: Dict[str, Any] = {
      "task_id": entry.task_id,
      "status": "running",
    }
    if agent_name:
      result["agent"] = agent_name
    return result, None

  async def get_background_result(
    self,
    tool_input: Dict[str, Any],
    **_: Any,
  ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    raw_task_id = tool_input.get("task_id")
    if not isinstance(raw_task_id, str) or not raw_task_id.strip():
      return None, {"code": "invalid_input", "message": "task_id is required"}
    task_id = raw_task_id.strip()

    wait = tool_input.get("wait", False)
    if not isinstance(wait, bool):
      return None, {"code": "invalid_input", "message": "wait must be a boolean"}

    raw_timeout = tool_input.get("timeout")
    if raw_timeout is not None and not isinstance(raw_timeout, (int, float)):
      return None, {"code": "invalid_input", "message": "timeout must be a number"}
    timeout = self._background_timeout_value(raw_timeout)

    if task_id == "*":
      selected = self._task_registry.list_tasks()
      if wait:
        pending = [
          bg_task.asyncio_task
          for bg_task in selected
          if bg_task.asyncio_task is not None and not bg_task.completed
        ]
        if pending:
          await asyncio.wait(pending, timeout=timeout)
      return {"tasks": [self._background_task_payload(bg_task) for bg_task in selected]}, None

    bg_task = self._task_registry.get(task_id)
    if bg_task is None:
      return None, {"code": "not_found", "message": f"Unknown background task: {task_id}"}

    if wait and bg_task.asyncio_task is not None and not bg_task.completed:
      await asyncio.wait([bg_task.asyncio_task], timeout=timeout)
    return self._background_task_payload(bg_task), None

  async def _shutdown_background_tasks(self, was_cancelled: bool) -> None:
    running_entries = self._task_registry.list_tasks(state=TaskState.RUNNING)
    pending = [
      bg_task.asyncio_task
      for bg_task in running_entries
      if bg_task.asyncio_task is not None
    ]
    if not pending:
      return

    try:
      if was_cancelled:
        for bg_task in running_entries:
          self._task_registry.kill(bg_task.task_id)
        await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=5.0)
        return

      _done, still_pending = await asyncio.wait(pending, timeout=30.0)
      if not still_pending:
        return
      for bg_task in running_entries:
        if bg_task.asyncio_task in still_pending:
          self._task_registry.kill(bg_task.task_id)
      await asyncio.wait_for(asyncio.gather(*still_pending, return_exceptions=True), timeout=5.0)
    except asyncio.TimeoutError:
      pass

  def _call_on_tool_timing(
    self,
    *,
    tool_name: str,
    server: str | None,
    duration_ms: int,
    is_error: bool,
    result_bytes: int,
  ) -> None:
    if self._on_tool_timing is None:
      return
    try:
      self._on_tool_timing(
        self._full_session_id,
        tool_name,
        server,
        duration_ms,
        is_error,
        result_bytes,
      )
    except Exception as exc:
      log.warning("[%s] on_tool_timing hook failed (non-fatal): %s", self._sid, exc)

  def _call_metric(self, name: str, value: int = 1) -> None:
    if self._on_metric is None:
      return
    try:
      self._on_metric(name, value)
    except Exception as exc:
      log.warning("[%s] metric hook failed (non-fatal): %s", self._sid, exc)

  @staticmethod
  def _usage_has_tokens(usage_totals: Dict[str, int]) -> bool:
    return any(
      int(usage_totals.get(key, 0) or 0) > 0
      for key in (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
      )
    )

  @staticmethod
  def _usage_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    return {
      "input_tokens": max(0, int(after.get("input_tokens", 0) or 0) - int(before.get("input_tokens", 0) or 0)),
      "output_tokens": max(0, int(after.get("output_tokens", 0) or 0) - int(before.get("output_tokens", 0) or 0)),
      "cache_read_input_tokens": max(
        0,
        int(after.get("cache_read_input_tokens", 0) or 0) - int(before.get("cache_read_input_tokens", 0) or 0),
      ),
      "cache_creation_input_tokens": max(
        0,
        int(after.get("cache_creation_input_tokens", 0) or 0)
        - int(before.get("cache_creation_input_tokens", 0) or 0),
      ),
    }

  def _build_usage_event(self, *, model: str, usage_totals: Dict[str, int]) -> UsageEvent:
    cost = self._estimate_usage_cost(model, usage_totals)
    return UsageEvent(
      user_id=self._usage_user_id,
      session_id=self._full_session_id,
      request_id=self._request_id,
      parent_turn_id=self._parent_turn_id,
      timestamp=time.time(),
      model=model,
      input_tokens=int(usage_totals["input_tokens"]),
      output_tokens=int(usage_totals["output_tokens"]),
      cache_read_tokens=int(usage_totals["cache_read_input_tokens"]),
      cache_creation_tokens=int(usage_totals["cache_creation_input_tokens"]),
      cost_usd=float(cost.total),
      rate_table_version=self._rate_table_version,
      billing_mode=self._billing_mode,
      channel=self._channel,
    )

  async def _call_on_usage(self, usage_event: UsageEvent) -> None:
    if self._on_usage is None:
      return
    try:
      result = self._on_usage(usage_event)
      if inspect.isawaitable(result):
        await result
    except Exception as exc:
      log.error("[%s] on_usage hook failed (non-fatal): %s", self._sid, exc)
      self._call_metric("gateway.usage_event_dropped", 1)
      if self._usage_ledger_dlq_path is not None:
        try:
          write_dlq(usage_event, self._usage_ledger_dlq_path)
        except Exception as dlq_exc:
          log.error("[%s] usage DLQ write failed (non-fatal): %s", self._sid, dlq_exc)

  def _estimate_usage_cost(self, model: str, usage_totals: Dict[str, int]):
    uncached_input = max(
      0,
      usage_totals["input_tokens"]
      - usage_totals["cache_read_input_tokens"]
      - usage_totals["cache_creation_input_tokens"],
    )
    return self._provider.estimate_cost(
      model,
      uncached_input,
      usage_totals["output_tokens"],
      cache_read_tokens=usage_totals["cache_read_input_tokens"],
      cache_creation_tokens=usage_totals["cache_creation_input_tokens"],
    )

  @staticmethod
  def _thinking_level(enabled: bool) -> ThinkingLevel:
    return ThinkingLevel.HIGH if enabled else ThinkingLevel.NONE

  @staticmethod
  def _classify_guard_outcome(
    guard_reason: tuple[str, str] | None,
    attempt: int,
    max_attempts: int,
  ) -> tuple[str, str, str]:
    """Classify guard outcome into (action, guard_error, guard_kind)."""
    if not guard_reason:
      return ("not_guard", "", "")
    guard_kind, guard_message = guard_reason
    guard_error = f"Stream watchdog: {guard_message}"
    if guard_kind == "stall" and attempt < max_attempts:
      return ("retry", guard_error, guard_kind)
    return ("abort", guard_error, guard_kind)

  async def _stream_turn(
    self,
    *,
    client: Any,
    config: Dict[str, Any],
    model_info: ModelInfo,
    system_prompt: Optional[Union[str, List[Tuple[str, bool]]]],
    current_messages: List[Dict[str, Any]],
    base_kwargs: Dict[str, Any],
    max_tokens: int,
    turn_count: int,
    turn_t0: float,
    turn_t0_mono: float,
    system_chars: int,
    tools_chars: int,
    usage_totals: Dict[str, int],
  ) -> Tuple[Any, StreamTurnResult] | None:
    last_event_at = time.monotonic()
    guard_reason: tuple[str, str] | None = None
    _effective_stall_timeout = self._stream_stall_timeout or STREAM_STALL_TIMEOUT

    def _make_params() -> Dict[str, Any]:
      normalized_messages = self._provider.normalize_messages(current_messages, model_info)
      return self._provider.build_request_params(
        model=config["model"],
        messages=normalized_messages,
        system_prompt=system_prompt,
        tools=base_kwargs.get("tools") or [],
        max_tokens=max_tokens,
        thinking_level=self._thinking_level(bool(config.get("thinking", True))),
        auth_mode=config["auth_mode"],
        compaction_trigger=self._compaction_trigger,
        compaction_instructions=self._compaction_instructions,
      )

    async def _consume_stream(params: Dict[str, Any], result: StreamTurnResult) -> None:
      nonlocal last_event_at
      first_turn = turn_count == 1
      log.debug("[%s] Turn %d stream open", self._sid, turn_count)

      async for event in self._provider.stream(client, params):
        last_event_at = time.monotonic()
        event_type = event.type

        if event_type == "message_start":
          usage_totals["input_tokens"] += event.input_tokens
          usage_totals["cache_creation_input_tokens"] += event.cache_creation_tokens
          usage_totals["cache_read_input_tokens"] += event.cache_read_tokens
          if first_turn:
            log.info(
              "[%s] Cache | read=%d create=%d uncached=%d",
              self._sid,
              event.cache_read_tokens,
              event.cache_creation_tokens,
              event.input_tokens,
            )
            if event.input_tokens > 0:
              msgs_chars = len(json.dumps(current_messages, default=str))
              total_chars = system_chars + tools_chars + msgs_chars
              if total_chars > 0:
                pct_system = round(system_chars / total_chars * 100)
                pct_tools = round(tools_chars / total_chars * 100)
                pct_messages = round(msgs_chars / total_chars * 100)
                tok_system = round(event.input_tokens * system_chars / total_chars)
                tok_tools = round(event.input_tokens * tools_chars / total_chars)
                tok_messages = event.input_tokens - tok_system - tok_tools
                log.info(
                  "[%s] Token breakdown | system=%d (%d%%) tools=%d (%d%%) messages=%d (%d%%) | total=%d",
                  self._sid,
                  tok_system,
                  pct_system,
                  tok_tools,
                  pct_tools,
                  tok_messages,
                  pct_messages,
                  event.input_tokens,
                  extra={
                    "data": {
                      "event": "token_breakdown",
                      "session_id": self._sid,
                      "turn": turn_count,
                      "input_tokens": event.input_tokens,
                      "est_system_tokens": tok_system,
                      "est_tools_tokens": tok_tools,
                      "est_messages_tokens": tok_messages,
                      "pct_system": pct_system,
                      "pct_tools": pct_tools,
                      "pct_messages": pct_messages,
                    }
                  },
                )
          continue

        if event_type == "text_delta":
          if result.first_token_t is None:
            result.first_token_t = time.time()
          text = str(event.text or "")
          self._append({"type": "text_delta", "text": text})
          result.full_text += text
          continue

        if event_type == "text_end":
          if isinstance(event.raw_block, dict):
            result.content_blocks.append(event.raw_block)
          continue

        if event_type == "thinking_delta":
          thinking_text = str(event.thinking_text or "")
          self._append({"type": "thinking_delta", "text": thinking_text})
          continue

        if event_type == "thinking_end":
          if isinstance(event.raw_block, dict):
            result.content_blocks.append(event.raw_block)
          log.info("[%s] Thinking block complete | %d chars", self._sid, len(str(event.thinking_text or "")))
          continue

        if event_type == "tool_use_start":
          continue

        if event_type == "tool_use_end":
          if isinstance(event.raw_block, dict):
            result.content_blocks.append(event.raw_block)
          result.tool_uses.append((event.tool_id, event.tool_name or "tool", dict(event.tool_input or {})))
          continue

        if event_type == "compaction":
          if isinstance(event.raw_block, dict):
            result.content_blocks.append(event.raw_block)
          content = event.raw_block.get("content") if isinstance(event.raw_block, dict) else event.text
          chars = len(content) if isinstance(content, str) else 0
          self._append({"type": "compaction", "chars": chars})
          log.info("[%s] Compaction block | %d chars", self._sid, chars)
          continue

        if event_type == "usage_update":
          usage_totals["output_tokens"] += event.output_tokens
          continue

        if event_type == "message_end":
          result.stop_reason = event.stop_reason or None

      log.debug("[%s] Turn %d stream end", self._sid, turn_count)

    async def _stream_guard(task: asyncio.Task, turn_start_mono: float) -> None:
      nonlocal guard_reason
      while not task.done():
        await asyncio.sleep(2.0)
        if task.done():
          return
        now = time.monotonic()
        stall = now - last_event_at
        if stall > _effective_stall_timeout:
          guard_reason = ("stall", f"no stream events for {stall:.0f}s")
          log.error("[%s] Turn %d watchdog (%s): %s", self._sid, turn_count, guard_reason[0], guard_reason[1])
          task.cancel()
          return
        if self._per_turn_timeout is not None and (now - turn_start_mono) > self._per_turn_timeout:
          guard_reason = ("timeout", f"turn timeout after {now - turn_start_mono:.0f}s")
          log.error("[%s] Turn %d watchdog (%s): %s", self._sid, turn_count, guard_reason[0], guard_reason[1])
          task.cancel()
          return

    stream_error: Exception | None = None
    for attempt in range(1 + STREAM_RETRY_MAX):
      if attempt > 0:
        await self._close_client(client, timeout=2.0)
        client = self._provider.create_client(config, timeout=self._client_timeout)
        self._set_client(client)
        log.warning(
          "[%s] Stream retry %d/%d on turn %d after %s",
          self._sid,
          attempt,
          STREAM_RETRY_MAX,
          turn_count,
          _format_exc(stream_error) if stream_error is not None else "unknown error",
        )
        delay = STREAM_RETRY_DELAY * (STREAM_RETRY_BACKOFF ** (attempt - 1))
        await asyncio.sleep(delay)

      last_event_at = time.monotonic()
      params = _make_params()
      result = StreamTurnResult()
      tokens_snapshot = dict(usage_totals)
      guard_reason = None
      stream_task = asyncio.create_task(_consume_stream(params, result))
      guard_task = asyncio.create_task(_stream_guard(stream_task, turn_t0_mono))
      try:
        done, pending = await asyncio.wait({stream_task, guard_task}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
          task.cancel()
        if pending:
          _, stuck = await asyncio.wait(pending, timeout=5.0)
          if stuck:
            log.warning("[%s] Turn %d: cancelled task stuck, force-closing client", self._sid, turn_count)
            await self._close_client(client, timeout=2.0)
            await asyncio.wait(stuck, timeout=2.0)
        if stream_task in done and not stream_task.cancelled():
          exc = stream_task.exception()
          if exc is not None:
            raise exc
      except asyncio.CancelledError:
        guard_task.cancel()
        stream_task.cancel()
        await self.force_close()
        raise
      except Exception as exc:
        stream_error = exc
        partial_usage = self._usage_delta(tokens_snapshot, usage_totals)
        usage_totals.clear()
        usage_totals.update(tokens_snapshot)

        action, guard_error, guard_kind = self._classify_guard_outcome(
          guard_reason,
          attempt,
          STREAM_RETRY_MAX,
        )
        if action != "not_guard":
          guard_message = guard_reason[1] if guard_reason else ""
          if action == "retry":
            stream_error = RuntimeError(guard_error)
            log.warning(
              "[%s] Stream watchdog stall on turn %d after %.1fs (attempt %d/%d), retrying: %s",
              self._sid,
              turn_count,
              time.time() - turn_t0,
              attempt + 1,
              1 + STREAM_RETRY_MAX,
              guard_message,
            )
            self._append({"type": "stream_retry", "attempt": attempt, "error": guard_error})
            continue
          log.error(
            "[%s] Stream watchdog on turn %d after %.1fs (%s): %s | %s",
            self._sid,
            turn_count,
            time.time() - turn_t0,
            guard_kind,
            guard_message,
            _format_exc(exc),
          )
          if self._usage_has_tokens(partial_usage):
            await self._call_on_usage(self._build_usage_event(model=config["model"], usage_totals=partial_usage))
          self._append({"type": "error", "error": guard_error})
          await self._close_client(client, timeout=5.0)
          return None

        formatted_exc = _format_exc(exc)
        if not self._provider.is_retryable_error(exc):
          log.error(
            "[%s] Stream error on turn %d after %.1fs (non-retryable): %s",
            self._sid,
            turn_count,
            time.time() - turn_t0,
            formatted_exc,
          )
          if self._usage_has_tokens(partial_usage):
            await self._call_on_usage(self._build_usage_event(model=config["model"], usage_totals=partial_usage))
          self._append({"type": "error", "error": formatted_exc})
          await self._close_client(client, timeout=5.0)
          return None

        log.warning(
          "[%s] Transient stream error on turn %d after %.1fs (attempt %d/%d): %s",
          self._sid,
          turn_count,
          time.time() - turn_t0,
          attempt + 1,
          1 + STREAM_RETRY_MAX,
          formatted_exc,
        )
        if attempt < STREAM_RETRY_MAX:
          self._append({"type": "stream_retry", "attempt": attempt, "error": formatted_exc})
          continue
      else:
        action, guard_error, _ = self._classify_guard_outcome(
          guard_reason,
          attempt,
          STREAM_RETRY_MAX,
        )
        if action != "not_guard":
          guard_message = guard_reason[1] if guard_reason else ""
          if action == "retry":
            stream_error = RuntimeError(guard_error)
            usage_totals.clear()
            usage_totals.update(tokens_snapshot)
            log.warning(
              "[%s] Stream watchdog stall on turn %d after %.1fs (attempt %d/%d), retrying: %s",
              self._sid,
              turn_count,
              time.time() - turn_t0,
              attempt + 1,
              1 + STREAM_RETRY_MAX,
              guard_message,
            )
            self._append({"type": "stream_retry", "attempt": attempt, "error": guard_error})
            continue
          self._append({"type": "error", "error": guard_error})
          await self._close_client(client, timeout=5.0)
          return None
        return client, result

    if stream_error is not None:
      formatted_exc = _format_exc(stream_error)
      log.error(
        "[%s] Stream failed on turn %d after %d retries: %s",
        self._sid,
        turn_count,
        STREAM_RETRY_MAX,
        formatted_exc,
      )
      self._append({"type": "error", "error": formatted_exc})
      await self._close_client(client, timeout=5.0)
    return None

  async def _execute_single_tool(
    self,
    tool_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
    base_kwargs: Dict[str, Any],
    call_index: int = 0,
  ) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
    tool_input_preview = json.dumps(tool_input, default=str)[:200]
    log.info(
      "[%s] Tool call: %s | input=%s",
      self._sid,
      tool_name,
      tool_input_preview,
      extra={
        "data": {
          "event": "tool_call",
          "session_id": self._sid,
          "tool": tool_name,
          "input_preview": tool_input_preview,
        }
      },
    )
    tool_t0 = time.time()
    server = self._mcp_client.get_server_for_tool(tool_name) if self._mcp_client is not None else None
    tool_start_event = {
      "type": "tool_call_start",
      "tool_call_id": tool_id,
      "tool_name": tool_name,
      "tool_input": tool_input,
      "execution_location": "backend",
      "call_index": call_index,
      "server": server,
      "started_at": tool_t0,
      "parent_assistant_message_seq": self._last_assistant_message_seq,
    }
    await self._append_durable_event(tool_start_event)
    self._append(tool_start_event)
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    cancelled_exc: Optional[asyncio.CancelledError] = None
    result_bytes = 0
    duration_ms = 0

    try:
      if tool_name in self._excluded_tools:
        error = {
          "code": "tool_excluded",
          "message": f"Tool '{tool_name}' is not available in this context",
        }
      else:
        dispatch_coro = self._dispatcher.dispatch(
          tool_id,
          tool_name,
          tool_input,
          call_index=call_index,
        )
        needs_approval = False
        requires_approval_fn = getattr(self._dispatcher, "requires_approval", None)
        if requires_approval_fn is not None:
          try:
            needs_approval = requires_approval_fn(tool_name, tool_input)
          except Exception:
            pass
        skip_timeout = tool_name in ("run_agent", "get_background_result") or needs_approval
        if self._tool_call_timeout is not None and not skip_timeout:
          try:
            result, error = await asyncio.wait_for(dispatch_coro, timeout=self._tool_call_timeout)
          except asyncio.TimeoutError:
            elapsed = time.time() - tool_t0
            log.error(
              "[%s] Tool %s timed out after %.1fs (limit %.0fs)",
              self._sid,
              tool_name,
              elapsed,
              self._tool_call_timeout,
            )
            error = {
              "code": "tool_timeout",
              "sub_code": "timeout",
              "message": f"Tool '{tool_name}' timed out after {self._tool_call_timeout:.0f}s. The tool call was cancelled. You may retry or skip this tool.",
            }
        else:
          result, error = await dispatch_coro

      tool_elapsed = time.time() - tool_t0
      if error:
        log.warning(
          "[%s] Tool %s error (%.1fs): %s",
          self._sid,
          tool_name,
          tool_elapsed,
          error,
          extra={
            "data": {
              "event": "tool_done",
              "session_id": self._sid,
              "tool": tool_name,
              "elapsed_s": round(tool_elapsed, 1),
              "server": server,
              "error": True,
              "error_detail": str(error)[:200],
              "error_sub_code": error.get("sub_code", "") if isinstance(error, dict) else "",
            }
          },
        )
      else:
        result_json = json.dumps(result, default=str) if result is not None else ""
        result_bytes = len(result_json)
        result_preview = result_json[:150] if result_json else "null"
        log.info(
          "[%s] Tool %s done (%.1fs) | result=%s",
          self._sid,
          tool_name,
          tool_elapsed,
          result_preview,
          extra={
            "data": {
              "event": "tool_done",
              "session_id": self._sid,
              "tool": tool_name,
              "elapsed_s": round(tool_elapsed, 1),
              "server": server,
              "result_bytes": result_bytes,
              "error": False,
            }
          },
        )
    except asyncio.CancelledError as exc:
      cancelled_exc = exc
      error = {"code": "cancelled", "message": "Task was cancelled"}
    except Exception as exc:
      log.error("[%s] Tool %s unhandled error: %s", self._sid, tool_name, exc)
      error = {"code": "internal_error", "message": str(exc)}
    finally:
      duration_ms = int((time.time() - tool_t0) * 1000)
      tool_complete_event = {
        "type": "tool_call_complete",
        "tool_call_id": tool_id,
        "tool_name": tool_name,
        "result": result,
        "error": error,
        "duration_ms": duration_ms,
        "server": server,
      }
      await self._append_durable_event(tool_complete_event)
      self._append(tool_complete_event)
      self._call_on_tool_timing(
        tool_name=tool_name,
        server=server,
        duration_ms=duration_ms,
        is_error=error is not None,
        result_bytes=result_bytes,
      )

    if cancelled_exc is not None:
      raise cancelled_exc

    if error is None and isinstance(result, dict):
      new_servers_raw = result.pop("_load_servers", None)
      if isinstance(new_servers_raw, list):
        new_servers = [str(server_name) for server_name in new_servers_raw if server_name]
        if new_servers:
          self._refresh_tools(base_kwargs, new_servers)
          log.info(
            "[%s] Loaded MCP servers: %s | total tools now: %d",
            self._sid,
            new_servers,
            len(base_kwargs.get("tools") or []),
          )

    model_result = result
    if error is None:
      model_result = self._annotate_result(result, tool_name=tool_name)

    if error is not None:
      result_entry = self._make_error_result(
        tool_id,
        str(error.get("code", "tool_error")),
        str(error.get("message", "Tool failed")),
        sub_code=str(error.get("sub_code", "")),
      )
    else:
      result_entry = {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": json.dumps(model_result, default=str),
      }
      if self._is_soft_error(model_result):
        result_entry["is_error"] = True

    extra_blocks = await self._call_on_tool_result(
      ToolResultContext(
        tool_name=tool_name,
        tool_input=dict(tool_input),
        result=result,
        error=error,
        duration_ms=duration_ms,
        tool_call_id=tool_id,
        session_id=self._full_session_id,
        server=server,
        result_entry=result_entry,
      )
    )
    return result_entry, tool_name, extra_blocks

  async def run(
    self,
    messages: List[Dict[str, Any]],
    system_prompt: Optional[Union[str, List[Tuple[str, bool]]]] = None,
    model_override: Optional[str] = None,
    max_turns: Optional[int] = None,
  ) -> None:
    """Execute the full chat loop and stream events into `EventLog`.

    Args:
      messages: Conversation history for the current request.
      system_prompt: Optional prompt string or cached prompt blocks.
      model_override: Optional per-request model override.
      max_turns: Optional maximum number of model/tool turns.

    Behavior:
      - emits `text_delta`, `thinking_delta`, and tool lifecycle events
      - retries transient stream failures
      - appends `stream_complete` on success
      - appends `error` on terminal failure
      - appends `max_turns_reached` or `budget_exceeded` when limits stop the
        loop early
    """
    was_cancelled = False
    run_error: BaseException | None = None
    try:
      if self._agent_session_log is None:
        self._runner_id = None
        self._last_assistant_message_seq = None
        self._durable_attach_emitted = False
      if self._agent_session_log is not None:
        self._runner_id = f"runner_{uuid.uuid4().hex}"
        self._last_durable_seq = 0
        self._last_assistant_message_seq = None
        self._durable_attach_emitted = False
        if self._role == "writer":
          await self._acquire_writer_lease_and_recover()
        await self._emit_attach_event()
        if self._role == "writer":
          self._write_lease_metadata()
        prior_messages = await self._context_builder.build() if self._context_builder is not None else []
        new_user_input = self._extract_last_user_message(messages)
        if new_user_input is not None:
          await self._append_user_message_event(new_user_input)
        messages = prior_messages + ([new_user_input] if new_user_input is not None else [])

      if self._coordinator is not None and self._coordinator.enabled:
        preamble = self._coordinator.preamble or COORDINATOR_DEFAULT_PREAMBLE
        if isinstance(system_prompt, list):
          system_prompt = [(preamble, False)] + list(system_prompt)
        elif system_prompt:
          system_prompt = f"{preamble}\n\n{system_prompt}"
        else:
          system_prompt = preamble

      config = dict(self._auth_config)
      config.update({
        "auth_mode": str(config.get("auth_mode", "api")).strip().lower(),
        "api_key": str(config.get("api_key", "")),
        "auth_token": str(config.get("auth_token", "")),
        "model": str(config.get("model", "claude-sonnet-4-6")),
        "max_tokens": int(config.get("max_tokens", 16000)),
        "thinking": bool(config.get("thinking", True)),
      })
      if model_override:
        config["model"] = model_override

      if not self._provider.has_active_credential(config):
        await self._emit_stub_response(messages)
        return

      try:
        client = self._provider.create_client(config, timeout=self._client_timeout)
        self._set_client(client)
      except Exception:
        await self._emit_stub_response(messages)
        return

      try:
        model_info = self._provider.get_model_info(config["model"])
      except Exception as exc:
        self._append({"type": "error", "error": str(exc)})
        await self._close_client(client, timeout=5.0)
        return

      cached_tools = self._default_tool_definitions()
      if self._excluded_tools:
        cached_tools = [tool for tool in cached_tools if tool["name"] not in self._excluded_tools]

      max_tokens = self._max_tokens_override if self._max_tokens_override is not None else config["max_tokens"]

      base_kwargs: Dict[str, Any] = {
        "tools": cached_tools,
      }
      if config["thinking"] and max_tokens >= 2048 and model_info.supports_thinking:
        log.info("[%s] Thinking enabled | max_tokens=%d", self._sid, max_tokens)
      elif not config["thinking"]:
        log.info("[%s] Thinking disabled | thinking=false", self._sid)
      elif max_tokens < 2048:
        log.info("[%s] Thinking disabled | max_tokens=%d too low (need >=2048)", self._sid, max_tokens)
      else:
        log.info("[%s] Thinking disabled | model=%s not supported", self._sid, config["model"])

      if self._compaction_trigger is not None:
        log.info("[%s] Compaction enabled | trigger=%d tokens", self._sid, self._compaction_trigger)

      log.info("[%s] Chat start | model=%s max_tokens=%d messages=%d", self._sid, config["model"], max_tokens, len(messages))

      chat_t0 = time.time()
      if isinstance(system_prompt, list):
        system_text = "\n\n".join(text for text, _should_cache in system_prompt if text)
      else:
        system_text = system_prompt or ""
      messages_text = json.dumps(messages, default=str)
      tools_text = json.dumps(cached_tools, default=str) if cached_tools else ""
      system_chars = len(system_text)
      est_system = _estimate_tokens(system_text)
      est_messages = _estimate_tokens(messages_text)
      est_tools = _estimate_tokens(tools_text) if tools_text else 0
      est_total = est_system + est_messages + est_tools
      if est_total > MODEL_CONTEXT_LIMIT * CONTEXT_WARNING_PCT / 100:
        log.warning(
          "[%s] Context usage high | est=%d tokens (%.0f%% of %dk limit)",
          self._sid,
          est_total,
          est_total / MODEL_CONTEXT_LIMIT * 100,
          MODEL_CONTEXT_LIMIT // 1000,
          extra={
            "data": {
              "event": "context_warning",
              "session_id": self._sid,
              "est_tokens": est_total,
              "limit": MODEL_CONTEXT_LIMIT,
              "pct": round(est_total / MODEL_CONTEXT_LIMIT * 100, 1),
            }
          },
        )
      log.info(
        "[%s] Pre-request estimate | system=%d msgs=%d tools=%d total=%d tokens (est)",
        self._sid,
        est_system,
        est_messages,
        est_tools,
        est_total,
        extra={
          "data": {
            "event": "token_estimate",
            "session_id": self._sid,
            "est_system_tokens": est_system,
            "est_messages_tokens": est_messages,
            "est_tools_tokens": est_tools,
            "est_total_tokens": est_total,
            "message_count": len(messages),
            "tool_count": len(cached_tools),
          }
        },
      )
      tools_chars = len(tools_text)
      usage_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
      }
      self._last_reported_cost = 0.0
      self._ensure_sub_agent_semaphore()
      turn_count = 0
      tools_used: List[str] = []
      current_messages = list(messages)

      while True:
        turn_count += 1
        if max_turns is not None and turn_count > max_turns:
          log.warning("[%s] Max turns (%d) reached, stopping", self._sid, max_turns)
          self._append({"type": "max_turns_reached", "turn_count": turn_count, "max_turns": max_turns})
          await self._emit_interrupted_event("max_turns_reached")
          summary_text = None
          if self._on_max_turns is not None:
            try:
              summary_text = await self._on_max_turns(current_messages, turn_count)
            except Exception as exc:
              log.warning("[%s] on_max_turns callback failed: %s", self._sid, exc)
          if summary_text:
            self._append({"type": "text_delta", "text": f"\n\n[Max turns reached]\n{summary_text}"})
          else:
            self._append({"type": "text_delta", "text": "\n\n[Sub-agent reached maximum turn limit]"})
          break
        turn_t0 = time.time()
        turn_t0_mono = time.monotonic()

        if turn_count > 1:
          turn_messages_text = json.dumps(current_messages, default=str)
          est_messages_turn = _estimate_tokens(turn_messages_text)
          current_tools = base_kwargs.get("tools") or []
          est_tools_turn = _estimate_tokens(json.dumps(current_tools, default=str)) if current_tools else 0
          est_turn = est_system + est_messages_turn + est_tools_turn
          if est_turn > MODEL_CONTEXT_LIMIT * CONTEXT_WARNING_PCT / 100:
            log.warning(
              "[%s] Context usage high | est=%d tokens (%.0f%% of %dk limit)",
              self._sid,
              est_turn,
              est_turn / MODEL_CONTEXT_LIMIT * 100,
              MODEL_CONTEXT_LIMIT // 1000,
              extra={
                "data": {
                  "event": "context_warning",
                  "session_id": self._sid,
                  "turn": turn_count,
                  "est_tokens": est_turn,
                  "limit": MODEL_CONTEXT_LIMIT,
                  "pct": round(est_turn / MODEL_CONTEXT_LIMIT * 100, 1),
                }
              },
            )
          log.info(
            "[%s] Turn %d pre-request | est=%d tokens",
            self._sid,
            turn_count,
            est_turn,
            extra={
              "data": {
                "event": "token_estimate",
                "session_id": self._sid,
                "turn": turn_count,
                "est_system_tokens": est_system,
                "est_messages_tokens": est_messages_turn,
                "est_tools_tokens": est_tools_turn,
                "est_total_tokens": est_turn,
                "message_count": len(current_messages),
                "tool_count": len(current_tools),
              }
            },
          )

        bg_reminder = self._background_task_reminder_text()
        notif_reminder = self._build_notification_reminder()
        peeked_notification_count = min(
          self._notification_queue.pending_count,
          _MAX_NOTIFICATIONS_PER_TURN,
        ) if notif_reminder else 0
        combined_reminder = "\n\n".join(filter(None, [bg_reminder, notif_reminder]))
        if combined_reminder:
          turn_system_prompt = self._inject_system_prompt_reminder(system_prompt, combined_reminder)
        else:
          turn_system_prompt = system_prompt
        if self._message_inbox is not None:
          parent_messages: list[str] = []
          while not self._message_inbox.empty():
            try:
              parent_messages.append(self._message_inbox.get_nowait())
            except asyncio.QueueEmpty:
              break
          if parent_messages:
            combined = "\n".join(f"[Message from parent agent]: {message}" for message in parent_messages)
            current_messages.append({"role": "user", "content": combined})
        turn_usage_before = dict(usage_totals)
        turn_result = await self._stream_turn(
          client=client,
          config=config,
          model_info=model_info,
          system_prompt=turn_system_prompt,
          current_messages=current_messages,
          base_kwargs=base_kwargs,
          max_tokens=max_tokens,
          turn_count=turn_count,
          turn_t0=turn_t0,
          turn_t0_mono=turn_t0_mono,
          system_chars=system_chars,
          tools_chars=tools_chars,
          usage_totals=usage_totals,
        )
        if turn_result is None:
          return
        client, turn = turn_result
        turn_usage = self._usage_delta(turn_usage_before, usage_totals)
        if self._usage_has_tokens(turn_usage):
          await self._call_on_usage(self._build_usage_event(model=config["model"], usage_totals=turn_usage))
        await self._append_assistant_message_event(
          content_blocks=turn.content_blocks,
          stop_reason=turn.stop_reason,
          model=config["model"],
          usage=turn_usage,
        )
        self._append(
          {
            "type": "turn_complete",
            "turn": turn_count,
            "usage": dict(turn_usage),
          }
        )

        turn_elapsed = time.time() - turn_t0
        ttft = (turn.first_token_t - turn_t0) if turn.first_token_t else None
        text_len = len(turn.full_text)
        tool_names = [tool[1] for tool in turn.tool_uses]
        text_preview = turn.full_text[:150].replace("\n", " ") if turn.full_text else ""

        log.info(
          "[%s] Turn %d complete | %.1fs | TTFT=%.2fs | text=%d chars | tools=%s | stop=%s | response=%s",
          self._sid,
          turn_count,
          turn_elapsed,
          ttft if ttft is not None else -1,
          text_len,
          tool_names or "none",
          turn.stop_reason,
          text_preview or "(none)",
          extra={
            "data": {
              "event": "turn_complete",
              "session_id": self._sid,
              "turn": turn_count,
              "elapsed_s": round(turn_elapsed, 1),
              "ttft_s": round(ttft, 2) if ttft is not None else None,
              "text_chars": text_len,
              "tools": tool_names,
              "stop_reason": turn.stop_reason,
            }
          },
        )

        if self._cost_accumulator is not None:
          running_cost = self._estimate_usage_cost(config["model"], usage_totals)
          incremental_cost = max(0.0, running_cost.total - self._last_reported_cost)
          if incremental_cost:
            self._cost_accumulator.add(incremental_cost)
          self._last_reported_cost = running_cost.total
          if self._cost_accumulator.exceeded:
            log.warning(
              "[%s] Budget exceeded: $%.4f >= $%.4f — stopping",
              self._sid,
              self._cost_accumulator.total,
              self._cost_accumulator.budget,
            )
            self._append(
              {
                "type": "budget_exceeded",
                "total_cost": round(self._cost_accumulator.total, 4),
                "budget": self._cost_accumulator.budget,
              }
            )
            self._append(
              {
                "type": "text_delta",
                "text": (
                  "\n\n"
                  f"[Budget limit reached: ${self._cost_accumulator.total:.4f} >= "
                  f"${self._cost_accumulator.budget:.4f}]"
                ),
              }
            )
            await self._emit_interrupted_event("budget_exceeded")
            break

        if not turn.tool_uses:
          if turn.stop_reason == "pause_turn":
            log.info("[%s] Pause turn — continuing", self._sid)
            assistant_content = list(turn.content_blocks)
            current_messages.append(
              {
                "role": "assistant",
                "content": assistant_content,
                "provider": self._provider.name,
                "model": config["model"],
                "stop_reason": turn.stop_reason,
              }
            )
            continue
          if turn.stop_reason == "compaction":
            log.info("[%s] Compaction pause — continuing", self._sid)
            assistant_content = list(turn.content_blocks)
            current_messages.append(
              {
                "role": "assistant",
                "content": assistant_content,
                "provider": self._provider.name,
                "model": config["model"],
                "stop_reason": turn.stop_reason,
              }
            )
            continue
          if self._notification_queue.pending_count > 0:
            assistant_content = list(turn.content_blocks)
            current_messages.append(
              {
                "role": "assistant",
                "content": assistant_content,
                "provider": self._provider.name,
                "model": config["model"],
                "stop_reason": turn.stop_reason,
              }
            )
            current_messages.append(
              {
                "role": "user",
                "content": "[System: Background tasks have completed. Check results with get_background_result.]",
              }
            )
            continue
          break

        assistant_content = list(turn.content_blocks)
        current_messages.append(
          {
            "role": "assistant",
            "content": assistant_content,
            "provider": self._provider.name,
            "model": config["model"],
            "stop_reason": turn.stop_reason,
          }
        )

        tool_results_content: List[Dict[str, Any]] = []
        i = 0
        run_agent_seq = 0
        while i < len(turn.tool_uses):
          tool_id, tool_name, tool_input = turn.tool_uses[i]
          if tool_name == "run_agent" and "run_agent" not in self._excluded_tools:
            batch: List[Tuple[int, str, str, Dict[str, Any]]] = []
            call_indices: List[int] = []
            while i < len(turn.tool_uses):
              batch_tool_id, batch_tool_name, batch_tool_input = turn.tool_uses[i]
              if batch_tool_name != "run_agent" or "run_agent" in self._excluded_tools:
                break
              batch.append((i, batch_tool_id, batch_tool_name, batch_tool_input))
              call_indices.append(run_agent_seq)
              run_agent_seq += 1
              i += 1

            async def _throttled(
              tool_call_id: str,
              tool_call_name: str,
              tool_call_input: Dict[str, Any],
              current_call_index: int,
            ) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
              if self._sub_agent_semaphore is not None:
                if not bool(tool_call_input.get("background")):
                  async with self._sub_agent_semaphore:
                    return await self._execute_single_tool(
                      tool_call_id,
                      tool_call_name,
                      tool_call_input,
                      base_kwargs,
                      call_index=current_call_index,
                    )
              if self._sub_agent_semaphore is not None and bool(tool_call_input.get("background")):
                return await self._execute_single_tool(
                  tool_call_id,
                  tool_call_name,
                  tool_call_input,
                  base_kwargs,
                  call_index=current_call_index,
                )
              if self._sub_agent_semaphore is not None:
                async with self._sub_agent_semaphore:
                  return await self._execute_single_tool(
                    tool_call_id,
                    tool_call_name,
                    tool_call_input,
                    base_kwargs,
                    call_index=current_call_index,
                  )
              return await self._execute_single_tool(
                tool_call_id,
                tool_call_name,
                tool_call_input,
                base_kwargs,
                call_index=current_call_index,
              )

            results = await asyncio.gather(
              *[
                _throttled(
                  batch_tool_id,
                  batch_tool_name,
                  batch_tool_input,
                  call_index,
                )
                for (_, batch_tool_id, batch_tool_name, batch_tool_input), call_index in zip(batch, call_indices)
              ],
              return_exceptions=True,
            )

            for j, result_or_exc in enumerate(results):
              _, batch_tool_id, batch_tool_name, _ = batch[j]
              if isinstance(result_or_exc, BaseException):
                if isinstance(result_or_exc, asyncio.CancelledError):
                  code = "cancelled"
                  message = "Sub-agent was cancelled"
                else:
                  code = "sub_agent_error"
                  message = str(result_or_exc) or "Sub-agent failed"
                log.warning("[%s] run_agent gather exception: %s", self._sid, result_or_exc)
                tool_results_content.append(self._make_error_result(batch_tool_id, code, message))
                tools_used.append(batch_tool_name)
              else:
                result_entry, used_name, extra_blocks = result_or_exc
                tool_results_content.append(result_entry)
                tool_results_content.extend(extra_blocks)
                tools_used.append(used_name)
          else:
            result_entry, used_name, extra_blocks = await self._execute_single_tool(
              tool_id,
              tool_name,
              tool_input,
              base_kwargs,
            )
            tool_results_content.append(result_entry)
            tool_results_content.extend(extra_blocks)
            tools_used.append(used_name)
            i += 1

        current_messages.append({"role": "user", "content": tool_results_content})
        if peeked_notification_count > 0:
          self._consume_notifications(max_count=peeked_notification_count)

        if turn.stop_reason == "end_turn":
          if self._notification_queue.pending_count > 0:
            current_messages.append(
              {
                "role": "user",
                "content": "[System: Background tasks have completed. Check results with get_background_result.]",
              }
            )
            continue
          break

      total_elapsed = time.time() - chat_t0
      cache_status = "miss"
      if usage_totals["cache_read_input_tokens"] > 0:
        cache_status = f"hit ({usage_totals['cache_read_input_tokens']} tokens cached)"
      elif usage_totals["cache_creation_input_tokens"] > 0:
        cache_status = f"write ({usage_totals['cache_creation_input_tokens']} tokens written)"

      cost = self._estimate_usage_cost(config["model"], usage_totals)

      log.info(
        "[%s] Chat done | %.1fs total | %d turns | tools=%s | tokens in=%d out=%d | cache=%s | cost=$%.4f",
        self._sid,
        total_elapsed,
        turn_count,
        tools_used or "none",
        usage_totals["input_tokens"],
        usage_totals["output_tokens"],
        cache_status,
        cost.total,
        extra={
          "data": {
            "event": "chat_done",
            "session_id": self._sid,
            "elapsed_s": round(total_elapsed, 1),
            "turns": turn_count,
            "tools": tools_used,
            "tokens_in": usage_totals["input_tokens"],
            "tokens_out": usage_totals["output_tokens"],
            "cache_read": usage_totals["cache_read_input_tokens"],
            "cache_write": usage_totals["cache_creation_input_tokens"],
            "cost": round(cost.total, 4),
          }
        },
      )

      self._append(
        {
          "type": "stream_complete",
          "usage": {
            "input_tokens": usage_totals["input_tokens"],
            "output_tokens": usage_totals["output_tokens"],
            "cache_creation_input_tokens": usage_totals["cache_creation_input_tokens"],
            "cache_read_input_tokens": usage_totals["cache_read_input_tokens"],
            "estimated_cost": round(cost.total, 4),
          },
        }
      )

      await self._close_client(client, timeout=5.0)
    except asyncio.CancelledError as exc:
      was_cancelled = True
      run_error = exc
    except BaseException as exc:
      run_error = exc
    finally:
      finalizer_error: BaseException | None = None
      try:
        await self._shutdown_background_tasks(was_cancelled)
        if self._agent_session_log is not None and self._durable_attach_emitted:
          if run_error is not None:
            reason = "graceful_shutdown"
            if isinstance(run_error, asyncio.CancelledError) and self._role == "sub_agent":
              reason = "sub_agent_cancelled"
            await self._emit_interrupted_event(reason)
          detach_reason = "completed"
          if isinstance(run_error, asyncio.CancelledError):
            detach_reason = "cancelled"
          elif run_error is not None:
            detach_reason = "error"
          await self._emit_detach_event(detach_reason)
      except BaseException as exc:
        finalizer_error = exc
      finally:
        try:
          self._release_write_lease()
        finally:
          await self.force_close()
      if run_error is not None:
        if finalizer_error is not None:
          raise finalizer_error from run_error
        raise run_error
      if finalizer_error is not None:
        raise finalizer_error
