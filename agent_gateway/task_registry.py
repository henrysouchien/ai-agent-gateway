from __future__ import annotations

import asyncio
import html
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol


class TaskState(Enum):
  PENDING = "pending"
  RUNNING = "running"
  COMPLETED = "completed"
  FAILED = "failed"
  KILLED = "killed"


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
  message_inbox: asyncio.Queue[str] = field(default_factory=asyncio.Queue)

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


_TERMINAL_STATES = {TaskState.COMPLETED, TaskState.FAILED, TaskState.KILLED}


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

  def register(self, task_type: str, agent_name: str | None = None, **metadata_kwargs: Any) -> TaskEntry:
    task_id = f"{self._id_prefix}_{self._seq}"
    self._seq += 1
    entry = TaskEntry(
      task_id=task_id,
      task_type=task_type,
      agent_name=agent_name,
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
