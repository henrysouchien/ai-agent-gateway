from __future__ import annotations

import asyncio
import html
import time
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Callable, Protocol


class TaskState(Enum):
  PENDING = "pending"
  RUNNING = "running"
  COMPLETED = "completed"
  FAILED = "failed"
  KILLED = "killed"
  INTERRUPTED = "interrupted"


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
  provider_name: str | None = None
  model: str | None = None
  message_inbox: asyncio.Queue[ParentMessage] = field(default_factory=asyncio.Queue)
  delivered_messages: set[str] = field(default_factory=set)
  original_task_id: str | None = None
  reconstructed_from_log: bool = False

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

  def format_xml(self) -> str:
    """Render as <task-notification> XML block."""
    safe_summary = html.escape(self.summary[:2000]) if self.summary else ""
    parts = [f'<task-notification task_id="{self.task_id}">']
    parts.append(f"  <status>{self.event}</status>")
    if self.agent_name:
      parts.append(f"  <agent>{html.escape(self.agent_name)}</agent>")
    if safe_summary:
      parts.append(f"  <summary>{safe_summary}</summary>")
    parts.append("</task-notification>")
    return "\n".join(parts)


@dataclass
class ResolvedProvider:
  provider: Any
  auth_config: dict[str, Any]
  allowed_models: set[str] | None = None
  default_model: str | None = None


ProviderResolver = Callable[[str], ResolvedProvider]


@dataclass
class CoordinatorConfig:
  enabled: bool = False
  preamble: str | None = None
  worker_excluded_tools: set[str] | None = None
  auto_notify: bool = True
  max_workers: int = 3
  provider_resolver: ProviderResolver | None = None
  default_worker_provider: str | None = None
  default_worker_model: str | None = None


COORDINATOR_DEFAULT_PREAMBLE = """You are operating in coordinator mode. Delegate tasks to workers:
- Use run_agent(background=true) to spawn workers
- Use get_background_result(task_id="*") to check status
- Use send_message to guide running workers
- Synthesize worker results — never delegate understanding
Workers will notify you when they complete.
You may specify a provider for each worker (e.g. provider="openai") to route to different models."""


class TaskLifecycleListener(Protocol):
  def on_transition(self, entry: TaskEntry, old_state: TaskState, new_state: TaskState) -> None:
    ...


_TERMINAL_STATES = {TaskState.COMPLETED, TaskState.FAILED, TaskState.KILLED, TaskState.INTERRUPTED}


class NotificationQueue:
  def __init__(self, max_pending: int = 20):
    self._queue: list[TaskNotification] = []
    self._max_pending = max_pending

  def push(self, notification: TaskNotification) -> None:
    if len(self._queue) < self._max_pending:
      self._queue.append(notification)

  def drain(self, max_count: int = 5) -> list[TaskNotification]:
    """Remove and return up to max_count notifications from the front."""
    result = self._queue[:max_count]
    self._queue = self._queue[max_count:]
    return result

  def peek(self, max_count: int | None = None) -> list[TaskNotification]:
    """Non-destructive peek at front of queue."""
    if max_count is None:
      return list(self._queue)
    return list(self._queue[:max_count])

  @property
  def pending_count(self) -> int:
    return len(self._queue)


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

  def register(
    self,
    task_type: str,
    agent_name: str | None = None,
    *,
    task_id: str | None = None,
    original_task_id: str | None = None,
    **metadata_kwargs: Any,
  ) -> TaskEntry:
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
      raise RuntimeError(f"Task inflight limit reached ({self._max_inflight})")

    old_state = entry.state
    entry.state = new_state
    if result is not None:
      entry.result = dict(result)
    if error is not None:
      entry.error = dict(error)
    if new_state in _TERMINAL_STATES:
      entry.completed_at = time.time()
    for listener in self._listeners:
      listener.on_transition(entry, old_state, new_state)
    return entry

  def get(self, task_id: str) -> TaskEntry | None:
    return self._tasks.get(task_id)

  def list_tasks(self, *, state: TaskState | None = None) -> list[TaskEntry]:
    tasks = self._tasks.values()
    if state is not None:
      tasks = (entry for entry in tasks if entry.state == state)
    return sorted(tasks, key=lambda entry: entry.started_at)

  def kill(self, task_id: str) -> bool:
    entry = self._tasks.get(task_id)
    if entry is None or entry.state in _TERMINAL_STATES:
      return False
    self.transition(task_id, TaskState.KILLED)
    if entry.asyncio_task is not None:
      entry.asyncio_task.cancel()
    return True

  def evict_completed(self, max_age_seconds: float = 300) -> int:
    now = time.time()
    evicted = 0
    candidates = [
      entry
      for entry in self.list_tasks()
      if entry.state in _TERMINAL_STATES and entry.completed_at is not None and now - entry.completed_at >= max_age_seconds
    ]
    for entry in candidates:
      if self._tasks.pop(entry.task_id, None) is not None:
        evicted += 1
    return evicted

  def add_listener(self, listener: TaskLifecycleListener) -> None:
    self._listeners.append(listener)

  def load_from_events(self, events: list[dict[str, Any]]) -> None:
    """Rebuild registry from durable task events without firing listeners.

    Tasks that were killed in a prior process rebuild as ``interrupted`` if no
    durable completion event exists; v1 cannot distinguish those from tasks
    that were running when the process crashed.
    """
    grouped: dict[str, dict[str, Any]] = {}
    max_suffix: int | None = None
    for event in events:
      task_id = str(event.get("task_id") or "")
      if not task_id:
        continue
      bucket = grouped.setdefault(task_id, {"messages": []})
      event_type = str(event.get("type") or "")
      if event_type == "task_registered":
        bucket["registered"] = dict(event)
      elif event_type == "task_completed":
        bucket["completed"] = dict(event)
      elif event_type == "parent_message_sent":
        bucket["messages"].append(dict(event))

    for task_id, bucket in grouped.items():
      registered = bucket.get("registered")
      if not isinstance(registered, dict):
        continue
      suffix = self._numeric_suffix(task_id)
      if suffix is not None:
        max_suffix = suffix if max_suffix is None else max(max_suffix, suffix)
      completed = bucket.get("completed")
      metadata = dict(registered.get("metadata") or {})
      for key in (
        "owner_runner_id",
        "owner_role",
        "sub_agent_id",
        "parent_turn_id",
        "call_index",
        "task_type",
        "provider_name",
        "model",
        "parent_session_id",
        "original_task_id",
      ):
        if key in registered:
          metadata[key] = registered.get(key)
      metadata["parent_messages"] = list(bucket.get("messages") or [])

      state = TaskState.INTERRUPTED
      result = None
      error = None
      completed_at = registered.get("started_at")
      if isinstance(completed, dict):
        try:
          state = TaskState(str(completed.get("final_state") or TaskState.FAILED.value))
        except ValueError:
          state = TaskState.FAILED
        result = completed.get("result")
        error = completed.get("error")
        completed_at = completed.get("completed_at", completed_at)

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
        provider_name=registered.get("provider_name"),
        model=registered.get("model"),
        original_task_id=registered.get("original_task_id"),
        reconstructed_from_log=True,
      )
      self._tasks[task_id] = entry

    if max_suffix is not None:
      self._seq = max(self._seq, max_suffix + 1)
    self._auto_evict_completed()

  @property
  def inflight_count(self) -> int:
    return sum(1 for entry in self._tasks.values() if entry.state == TaskState.RUNNING)

  def _auto_evict_completed(self) -> None:
    limit = max(0, self._max_retained)
    if len(self._tasks) <= limit:
      return
    completed = [
      entry
      for entry in self.list_tasks()
      if entry.state in _TERMINAL_STATES
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
