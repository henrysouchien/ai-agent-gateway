from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable

from .runner_state import budget_reason_suffix
from .task_registry import TaskState


@dataclass(frozen=True)
class BackgroundResultRequest:
  task_id: str
  wait: bool
  timeout: float


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
  if raw_timeout is not None and not isinstance(raw_timeout, (int, float)):
    return None, {"code": "invalid_input", "message": "timeout must be a number"}
  timeout = background_timeout_value(raw_timeout)
  return BackgroundResultRequest(task_id=task_id, wait=wait, timeout=timeout), None


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


async def background_result_tasks(
  *,
  task_entries: Callable[[], Iterable[Any]],
  wait: bool,
  timeout: float,
  wait_fn: Callable[..., Awaitable[Any]],
  payload: Callable[[Any], Dict[str, Any]],
) -> Dict[str, Any]:
  selected = list(task_entries())
  await wait_for_background_tasks(
    selected,
    wait=wait,
    timeout=timeout,
    wait_fn=wait_fn,
  )
  return {"tasks": [payload(bg_task) for bg_task in selected]}


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


async def drain_cancelled_background_tasks(
  running_entries: Iterable[Any],
  pending: Iterable[Any],
  *,
  kill_task: Callable[[str], None],
  gather_fn: Callable[..., Awaitable[Any]],
  wait_for_fn: Callable[..., Awaitable[Any]],
  timeout: float,
) -> None:
  kill_background_tasks(running_entries, kill_task=kill_task)
  await wait_for_fn(gather_fn(*pending, return_exceptions=True), timeout=timeout)


async def drain_still_pending_background_tasks(
  running_entries: Iterable[Any],
  pending: Iterable[Any],
  *,
  kill_task: Callable[[str], None],
  wait_fn: Callable[..., Awaitable[Any]],
  gather_fn: Callable[..., Awaitable[Any]],
  wait_for_fn: Callable[..., Awaitable[Any]],
  wait_timeout: float,
  drain_timeout: float,
) -> None:
  _done, still_pending = await wait_fn(pending, timeout=wait_timeout)
  if not still_pending:
    return
  kill_background_tasks_for_asyncio_tasks(
    running_entries,
    still_pending,
    kill_task=kill_task,
  )
  await wait_for_fn(gather_fn(*still_pending, return_exceptions=True), timeout=drain_timeout)


def background_task_limit_error(
  *,
  inflight_count: int,
  max_background_tasks: int,
) -> Dict[str, Any] | None:
  if inflight_count < max_background_tasks:
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
        "provider_name": metadata.get("provider_name", getattr(bg_task, "provider_name", None)),
        "model": metadata.get("model", getattr(bg_task, "model", None)),
        "original_task_id": getattr(bg_task, "original_task_id", None),
        "resumable": bool(metadata.get("resumable", False)),
        "resumed_as": resumed_as,
        "latest_resume_task_id": resumed_as[-1] if resumed_as else None,
        "message": "Background task was interrupted by a gateway restart before completion.",
      }
    )
    return payload

  if getattr(bg_task, "state", None) == TaskState.KILLED:
    payload["status"] = "killed"
    payload["elapsed_seconds"] = elapsed_seconds
    return payload

  if bg_task.completed:
    payload["elapsed_seconds"] = elapsed_seconds
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


def sub_agent_result_from_log_entries(
  entries: Iterable[Any],
  *,
  timed_out: bool,
  timeout: float | None,
  budget_exceeded_reason: str | None = None,
  original_task_id: str | None = None,
) -> Dict[str, Any]:
  text_parts: list[str] = []
  tool_calls_made: list[str] = []
  usage: Dict[str, Any] = {}
  error_msg: str | None = None
  budget_exceeded = False
  event_budget_exceeded_reason: str | None = None
  max_turns_hit = False
  for entry in entries:
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
      raw_reason = event.get("reason")
      if isinstance(raw_reason, str) and raw_reason:
        event_budget_exceeded_reason = raw_reason
    elif event_type == "max_turns_reached":
      max_turns_hit = True
    elif event_type == "error":
      error_msg = str(event.get("error", "Sub-agent error"))

  result: Dict[str, Any] = {
    "response": "".join(text_parts).strip(),
    "tools_used": tool_calls_made,
    "usage": usage,
  }
  if original_task_id is not None:
    result["original_task_id"] = original_task_id

  warnings: list[str] = []
  if timed_out:
    warnings.append(f"Sub-agent timed out after {timeout}s — partial results returned")
  elif error_msg:
    warnings.append(f"Sub-agent error: {error_msg}")
  if budget_exceeded:
    exceeded_reason = event_budget_exceeded_reason or budget_exceeded_reason
    warnings.append(f"Sub-agent stopped: budget limit reached{budget_reason_suffix(exceeded_reason)}")
  if max_turns_hit:
    warnings.append("Sub-agent stopped: max turns reached — partial results")
  if warnings:
    result["warning"] = "; ".join(warnings)
  return result


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
    "provider_name": metadata.get("provider_name", entry.provider_name),
    "model": metadata.get("model", entry.model),
  }
  if entry.original_task_id is not None:
    payload["original_task_id"] = entry.original_task_id
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


def task_registered_event_payload(
  entry: Any,
  *,
  correlation_payload: Dict[str, Any],
  agent_name: str | None,
  parent_session_id: str,
) -> Dict[str, Any]:
  return {
    "type": "task_registered",
    **correlation_payload,
    "agent_name": agent_name,
    "parent_session_id": parent_session_id,
    "metadata": dict(entry.metadata),
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


def background_task_provider_name(
  tool_input: Dict[str, Any],
  *,
  default_provider_name: str | None,
) -> str | None:
  raw_provider_name = tool_input.get("provider_name", tool_input.get("provider"))
  if isinstance(raw_provider_name, str) and raw_provider_name.strip():
    return raw_provider_name.strip()
  return default_provider_name


def background_task_model(
  tool_input: Dict[str, Any],
  *,
  auth_model: Any,
) -> str | None:
  raw_model = tool_input.get("model")
  if isinstance(raw_model, str) and raw_model.strip():
    return raw_model.strip()
  if isinstance(auth_model, str) and auth_model.strip():
    return auth_model.strip()
  return None


def parse_child_budget_usd(raw_child_budget: Any) -> float | None:
  if raw_child_budget is None or isinstance(raw_child_budget, bool):
    return None
  try:
    parsed_child_budget = float(raw_child_budget)
  except (TypeError, ValueError):
    return None
  if parsed_child_budget > 0:
    return parsed_child_budget
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
  provider_name: str | None,
  model: str | None,
  child_budget_usd: float | None,
  original_task_id: str | None,
  tool_input: Dict[str, Any],
) -> Dict[str, Any]:
  correlation: Dict[str, Any] = {
    "owner_runner_id": owner_runner_id,
    "owner_role": owner_role,
    "sub_agent_id": sub_agent_id,
    "parent_turn_id": parent_turn_id,
    "call_index": call_index,
    "task_type": "background",
    "provider_name": provider_name,
    "model": model,
  }
  if child_budget_usd is not None:
    correlation["child_budget_usd"] = child_budget_usd
  if original_task_id is not None:
    correlation["original_task_id"] = original_task_id
  if "resumable" in tool_input:
    correlation["resumable"] = bool(tool_input.get("resumable"))
  return correlation


def prepare_background_task_registration(
  entry: Any,
  *,
  tool_input: Dict[str, Any],
  default_provider_name: str | None,
  auth_model: Any,
  owner_runner_id: str | None,
  owner_role: str,
  sub_agent_id_for_call_index: Callable[[int], str],
  parent_turn_id: str | None,
  original_task_id: str | None,
) -> int:
  entry.provider_name = background_task_provider_name(
    tool_input,
    default_provider_name=default_provider_name,
  )
  entry.model = background_task_model(
    tool_input,
    auth_model=auth_model,
  )
  child_budget_usd = parse_child_budget_usd(tool_input.get("child_budget_usd"))
  call_index = background_task_call_index(entry.task_id)
  sub_agent_id = sub_agent_id_for_call_index(call_index)
  entry.metadata.update(
    background_task_registration_metadata(
      owner_runner_id=owner_runner_id,
      owner_role=owner_role,
      sub_agent_id=sub_agent_id,
      parent_turn_id=parent_turn_id,
      call_index=call_index,
      provider_name=entry.provider_name,
      model=entry.model,
      child_budget_usd=child_budget_usd,
      original_task_id=original_task_id,
      tool_input=tool_input,
    )
  )
  return call_index
