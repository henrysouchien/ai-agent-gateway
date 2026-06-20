from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import secrets
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

import httpx

from ._io import _atomic_write_json, _read_json_object
from ._provider_utils import _allowed_models_for_provider, _get_default_model_for_provider, _resolve_provider
from .autonomous_excel_dispatch import make_autonomous_message_excel_agent_handler
from .event_log import EventLog
from .excel_dispatch import make_message_excel_agent_tool_def
from .mcp_client import McpClientManager
from .multi_user.billing import SessionUsageSummary, UsageEvent
from .providers import ModelProvider
from .runner import AgentRunner, ToolResultContext
from .skills import SkillLoader
from .sub_agent import (
  make_get_background_result_handler,
  make_get_background_result_tool_def,
  make_send_message_handler,
  make_send_message_tool_def,
  make_run_agent_handler,
  make_run_agent_tool_def,
)
from .task_registry import CoordinatorConfig
from .tool_dispatcher import LocalToolHandler, ToolDispatcher, ToolInterceptor


log = logging.getLogger("agent_gateway.autonomous")
_STATE_JSON_MARKER = "## STATE_UPDATE_JSON"
_JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_RUN_SESSION_FORCE_CLOSE_SECONDS = 2.0
_RUN_SESSION_CANCEL_DRAIN_SECONDS = 5.0


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


@dataclass
class RunOutput:
  response: str
  tools_used: list[str]
  usage: dict[str, Any]
  error: str | None
  timed_out: bool
  budget_exceeded: bool = False
  max_turns_reached: bool = False
  operator_paused: bool = False
  max_tokens_reached: bool = False
  exit_reason: str | None = None
  post_run_guard: dict[str, Any] | None = None


@dataclass
class DeliveryConfig:
  on_complete: Callable[[RunOutput, dict[str, Any] | None], Awaitable[None] | None] | None = None
  telegram_bot_token: str | None = None
  telegram_chat_id: str | None = None
  telegram_label: str | None = None
  briefing_file: Path | str | None = None
  webhook_url: str | None = None
  format_message: Callable[[RunOutput, dict[str, Any] | None], str] | None = None


def _is_env_placeholder(value: str) -> bool:
  stripped = value.strip()
  return stripped.startswith("${") and stripped.endswith("}")


def _resolve_autonomous_mcp_gateway_api_key(user_id: str, user_email: str | None) -> str:
  try:
    from api.agent.autonomous.mcp_config import _resolve_gateway_api_key

    return str(_resolve_gateway_api_key(user_id, user_email)).strip()
  except ModuleNotFoundError:
    api_dir = Path(__file__).resolve().parents[3] / "api"
    if not api_dir.exists():
      raise
    api_dir_text = str(api_dir)
    if api_dir_text not in sys.path:
      sys.path.insert(0, api_dir_text)
    from agent.autonomous.mcp_config import _resolve_gateway_api_key

    return str(_resolve_gateway_api_key(user_id, user_email)).strip()


def _resolve_message_excel_agent_gateway_config(user_id: str) -> tuple[str, str]:
  gateway_url = os.getenv("GATEWAY_URL", "").strip()
  if not gateway_url or _is_env_placeholder(gateway_url):
    gateway_url = "https://localhost:8000"

  env_key = os.getenv("GATEWAY_API_KEY", "").strip()
  if env_key and not _is_env_placeholder(env_key):
    return gateway_url, env_key

  resolved_user_id = str(user_id or "").strip()
  user_email = os.getenv("RISK_MODULE_USER_EMAIL", "").strip() or None
  if not resolved_user_id:
    raise RuntimeError(
      "message_excel_agent registration requires GATEWAY_API_KEY or a user_id "
      "for api/agent/autonomous/mcp_config._resolve_gateway_api_key"
    )

  try:
    gateway_api_key = _resolve_autonomous_mcp_gateway_api_key(resolved_user_id, user_email)
  except SystemExit as exc:
    detail = str(exc) or "gateway API key resolver exited without a message"
    raise RuntimeError(
      "message_excel_agent registration requires GATEWAY_API_KEY or "
      "GATEWAY_USER_KEYS channel='mcp' entry for "
      f"user_id={resolved_user_id!r}, user_email={user_email!r}: {detail}"
    ) from exc
  except Exception as exc:
    raise RuntimeError(
      "message_excel_agent registration requires GATEWAY_API_KEY or "
      "api/agent/autonomous/mcp_config._resolve_gateway_api_key; "
      f"resolver failed for user_id={resolved_user_id!r}, user_email={user_email!r}: {exc}"
    ) from exc

  if not gateway_api_key:
    raise RuntimeError(
      "message_excel_agent registration requires a non-empty gateway API key from "
      "GATEWAY_API_KEY or api/agent/autonomous/mcp_config._resolve_gateway_api_key"
    )
  return gateway_url, gateway_api_key


def _extract_summary(text: str, limit: int = 1500) -> str:
  stripped = str(text or "").strip()
  if len(stripped) <= limit:
    return stripped
  return stripped[: limit - 3].rstrip() + "..."


def _ensure_string_list(value: Any) -> list[str]:
  if not isinstance(value, list):
    return []
  items: list[str] = []
  for item in value:
    if item is None:
      continue
    text = str(item).strip()
    if text:
      items.append(text)
  return items


def collect_run_output(event_log: EventLog, timed_out: bool) -> RunOutput:
  text_parts: list[str] = []
  tool_calls: list[str] = []
  usage: dict[str, Any] = {}
  error_msg: str | None = None
  budget_exceeded = False
  max_turns_reached = False
  operator_paused = False
  max_tokens_reached = False

  for entry in event_log.entries:
    event = entry.event
    event_type = event.get("type")
    if event_type == "stream_retry":
      text_parts.clear()
      tool_calls.clear()
    elif event_type == "text_delta":
      text_parts.append(str(event.get("text", "")))
    elif event_type == "tool_call_start":
      tool_calls.append(str(event.get("tool_name", "")))
    elif event_type == "stream_complete":
      event_usage = event.get("usage")
      if isinstance(event_usage, dict):
        usage = event_usage
    elif event_type == "budget_exceeded":
      budget_exceeded = True
    elif event_type == "max_turns_reached":
      max_turns_reached = True
    elif event_type == "max_tokens_reached":
      max_tokens_reached = True
    elif event_type == "assistant_message" and event.get("stop_reason") == "max_tokens":
      max_tokens_reached = True
    elif event_type == "operator_pause":
      operator_paused = True
    elif event_type == "interrupted" and event.get("reason") == "operator_pause":
      operator_paused = True
    elif event_type == "error":
      error_msg = str(event.get("error", "Autonomous run encountered an error"))

  return RunOutput(
    response="".join(text_parts).strip(),
    tools_used=tool_calls,
    usage=usage,
    error=error_msg,
    timed_out=timed_out,
    budget_exceeded=budget_exceeded,
    max_turns_reached=max_turns_reached,
    operator_paused=operator_paused,
    max_tokens_reached=max_tokens_reached,
  )


async def _force_close_runner(runner: Any, *, timeout: float) -> None:
  force_close = getattr(runner, "force_close", None)
  if not callable(force_close):
    return
  try:
    signature = inspect.signature(force_close)
  except (TypeError, ValueError):
    accepts_timeout = True
  else:
    accepts_timeout = (
      "timeout" in signature.parameters
      or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    )
  if accepts_timeout:
    close_result = force_close(timeout=timeout)
  else:
    close_result = force_close()
  if inspect.isawaitable(close_result):
    await asyncio.wait_for(close_result, timeout=max(0.1, timeout + 0.5))


async def _drain_cancelled_run_task(task: asyncio.Task[None], *, timeout: float) -> bool:
  if task.done():
    try:
      await task
    except asyncio.CancelledError:
      return True
    except Exception as exc:
      log.warning("Autonomous run raised while draining after timeout: %s", exc)
      return True
    return True

  done, _pending = await asyncio.wait({task}, timeout=timeout)
  if task not in done:
    return False
  try:
    await task
  except asyncio.CancelledError:
    return True
  except Exception as exc:
    log.warning("Autonomous run raised while draining after timeout: %s", exc)
  return True


def _consume_late_run_task_result(task: asyncio.Task[None]) -> None:
  try:
    task.result()
  except asyncio.CancelledError:
    return
  except Exception as exc:
    log.warning("Autonomous run raised after timeout return: %s", exc)


async def run_session(
  runner: AgentRunner,
  event_log: EventLog,
  *,
  model: str,
  max_turns: int,
  timeout_seconds: float | None,
  initial_message: str,
  system_prompt: str | list[tuple[str, bool]],
) -> RunOutput:
  timed_out = False
  error_msg: str | None = None
  coro = runner.run(
    messages=[{"role": "user", "content": initial_message}],
    system_prompt=system_prompt,
    model_override=model,
    max_turns=max_turns,
  )
  run_task: asyncio.Task[None] | None = None
  try:
    if timeout_seconds is not None and timeout_seconds > 0:
      run_task = asyncio.create_task(coro)
      done, _pending = await asyncio.wait({run_task}, timeout=timeout_seconds)
      if run_task in done:
        await run_task
      else:
        timed_out = True
        log.warning("Autonomous run timed out after %ss", timeout_seconds)
        run_task.cancel()
        try:
          await _force_close_runner(runner, timeout=_RUN_SESSION_FORCE_CLOSE_SECONDS)
        except Exception as exc:
          log.warning("Autonomous runner force-close after timeout failed: %s", exc)
        drained = await _drain_cancelled_run_task(run_task, timeout=_RUN_SESSION_CANCEL_DRAIN_SECONDS)
        if not drained:
          log.warning(
            "Autonomous run cancellation did not drain within %ss after timeout",
            _RUN_SESSION_CANCEL_DRAIN_SECONDS,
          )
          run_task.add_done_callback(_consume_late_run_task_result)
    else:
      await coro
  except asyncio.CancelledError:
    if run_task is not None and not run_task.done():
      run_task.cancel()
      try:
        await _force_close_runner(runner, timeout=_RUN_SESSION_FORCE_CLOSE_SECONDS)
      except Exception as exc:
        log.warning("Autonomous runner force-close after cancellation failed: %s", exc)
      drained = await _drain_cancelled_run_task(run_task, timeout=_RUN_SESSION_CANCEL_DRAIN_SECONDS)
      if not drained:
        run_task.add_done_callback(_consume_late_run_task_result)
    raise
  except Exception as exc:
    error_msg = f"{type(exc).__name__}: {exc}"
    log.error("Autonomous run failed: %s", error_msg, exc_info=True)

  output = collect_run_output(event_log, timed_out=timed_out)
  if error_msg and not output.error:
    output.error = error_msg
  return output


def run_output_exit_code(run_output: RunOutput) -> int:
  if run_output.timed_out:
    return 124
  if run_output.exit_reason == "post_run_guard_failed":
    return 1
  if run_output.error:
    return 1
  if run_output.budget_exceeded:
    return 2
  if run_output.max_turns_reached:
    return 3
  if run_output.max_tokens_reached:
    return 4
  return 0


def run_output_outcome(run_output: RunOutput) -> str:
  if run_output.timed_out:
    return "timeout"
  if run_output.exit_reason:
    return run_output.exit_reason
  if run_output.error:
    return "error"
  if run_output.budget_exceeded:
    return "budget_exceeded"
  if run_output.max_turns_reached:
    return "max_turns"
  if run_output.max_tokens_reached:
    return "max_tokens"
  if run_output.operator_paused:
    return "operator_pause"
  return "success"


def mark_post_run_guard_failure(
  run_output: RunOutput,
  *,
  guard: str,
  message: str,
  details: dict[str, Any] | None = None,
) -> None:
  payload = {
    "guard": guard,
    "message": message,
  }
  if details:
    payload.update(details)
  run_output.error = message
  run_output.exit_reason = "post_run_guard_failed"
  run_output.post_run_guard = payload


def load_state(
  state_dir: str | Path,
  state_file: str = "state.json",
) -> dict[str, Any]:
  return _read_json_object(Path(state_dir) / state_file)


def save_state(
  state_dir: str | Path,
  state: dict[str, Any],
  state_file: str = "state.json",
) -> None:
  _atomic_write_json(Path(state_dir) / state_file, dict(state))


def extract_state_update(text: str) -> dict[str, Any]:
  if not str(text or "").strip():
    return {}

  section = text
  marker_idx = text.rfind(_STATE_JSON_MARKER)
  if marker_idx >= 0:
    section = text[marker_idx:]

  matches = list(_JSON_FENCE_RE.finditer(section))
  for match in reversed(matches):
    candidate = match.group(1)
    try:
      payload = json.loads(candidate)
    except json.JSONDecodeError:
      continue
    if isinstance(payload, dict):
      return payload
  return {}


def build_state_payload(
  previous_state: dict[str, Any],
  model_state: dict[str, Any],
  run_output: RunOutput,
  model_name: str = "",
  briefing_file: str = "",
  connected_servers: Sequence[str] | None = None,
  active_servers: Sequence[str] | None = None,
  extract_summary_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
  state: dict[str, Any] = {}
  if isinstance(previous_state, dict):
    state.update(previous_state)
  if isinstance(model_state, dict):
    state.update(model_state)

  summary_fn = extract_summary_fn or _extract_summary
  connected_server_names = sorted({str(name) for name in (connected_servers or []) if str(name).strip()})
  active_server_names = sorted({str(name) for name in (active_servers or []) if str(name).strip()})

  state["last_run"] = datetime.now(tz=timezone.utc).isoformat()
  state["model"] = model_name
  state["briefing_file"] = briefing_file
  state["timed_out"] = run_output.timed_out
  state["budget_exceeded"] = run_output.budget_exceeded
  state["max_turns_reached"] = run_output.max_turns_reached
  state["max_tokens_reached"] = run_output.max_tokens_reached
  state["operator_paused"] = run_output.operator_paused
  state["connected_servers"] = connected_server_names
  state["active_servers"] = active_server_names
  state["tools_used"] = sorted({name for name in run_output.tools_used if name})
  state["usage"] = run_output.usage
  state["last_summary"] = summary_fn(run_output.response)

  if run_output.error:
    state["error"] = run_output.error
  else:
    state.pop("error", None)

  alerts = _ensure_string_list(state.get("alerts"))
  next_session = _ensure_string_list(state.get("next_session"))
  if alerts:
    state["alerts"] = alerts
  if next_session:
    state["next_session"] = next_session

  return state


def format_run_summary(
  run_output: RunOutput,
  label: str | None = None,
  state: dict[str, Any] | None = None,
  format_state_fn: Callable[[dict[str, Any]], str] | None = None,
) -> str:
  status = "timed out" if run_output.timed_out else "completed"
  if run_output.exit_reason == "post_run_guard_failed" and not run_output.timed_out:
    status = "post-run guard failed"
  elif run_output.error and not run_output.timed_out:
    status = "failed"
  elif run_output.budget_exceeded:
    status = "budget exceeded"
  elif run_output.max_turns_reached:
    status = "max turns reached"
  elif run_output.max_tokens_reached:
    status = "max tokens reached"
  elif run_output.operator_paused:
    status = "operator paused"

  usage = run_output.usage if isinstance(run_output.usage, dict) else {}
  in_tokens = usage.get("input_tokens", "?")
  out_tokens = usage.get("output_tokens", "?")
  est_cost = usage.get("estimated_cost", "?")
  tools_used = sorted({name for name in run_output.tools_used if name})
  tools_preview = ", ".join(tools_used[:8]) if tools_used else "none"
  if len(tools_used) > 8:
    tools_preview += f", ... (+{len(tools_used) - 8})"

  lines = [
    label or "Autonomous run",
    f"Status: {status}",
  ]
  if isinstance(state, dict) and state.get("briefing_file"):
    lines.append(f"Briefing: {state['briefing_file']}")
  lines.extend([
    f"Usage: in={in_tokens} out={out_tokens} est_cost={est_cost}",
    f"Tools: {tools_preview}",
  ])

  if state and format_state_fn is not None:
    formatted_state = str(format_state_fn(state) or "").strip()
    if formatted_state:
      lines.extend(["", *formatted_state.splitlines()])

  summary = _extract_summary(run_output.response, limit=1200)
  if summary:
    lines.extend(["", "Summary:", summary])
  if run_output.exit_reason:
    lines.extend(["", f"Exit reason: {run_output.exit_reason}"])
  if run_output.post_run_guard:
    guard_name = run_output.post_run_guard.get("guard")
    if guard_name:
      lines.append(f"Post-run guard: {guard_name}")
  if run_output.error:
    lines.extend(["", f"Error: {run_output.error}"])

  message = "\n".join(lines)
  if len(message) > 3900:
    message = message[:3897] + "..."
  return message


def split_messages(text: str, max_len: int = 4096) -> list[str]:
  if not text:
    return []

  chunks: list[str] = []
  current_lines: list[str] = []
  current_len = 0

  def _flush_current() -> None:
    nonlocal current_lines, current_len
    if not current_lines:
      return
    chunks.append("".join(current_lines))
    current_lines = []
    current_len = 0

  for line in text.splitlines(keepends=True):
    line_len = len(line)
    if line_len > max_len:
      _flush_current()
      for start in range(0, line_len, max_len):
        chunks.append(line[start : start + max_len])
      continue

    if current_len + line_len > max_len:
      _flush_current()

    current_lines.append(line)
    current_len += line_len

  _flush_current()
  return chunks


async def send_telegram(message: str, bot_token: str, chat_id: str) -> None:
  url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
  payload = {
    "chat_id": chat_id,
    "text": message,
    "disable_web_page_preview": True,
  }
  async with httpx.AsyncClient(timeout=10.0) as client:
    response = await client.post(url, json=payload)
    response.raise_for_status()


async def send_telegram_file(path: Path | str, bot_token: str, chat_id: str) -> None:
  briefing_path = Path(path)
  if not briefing_path.exists():
    return

  try:
    content = briefing_path.read_text(encoding="utf-8")
  except (OSError, UnicodeDecodeError):
    return

  if not content.strip():
    return

  for chunk in split_messages(content):
    await send_telegram(chunk, bot_token, chat_id)


async def send_webhook(
  url: str,
  payload: dict[str, Any],
  headers: dict[str, str] | None = None,
) -> None:
  async with httpx.AsyncClient(timeout=10.0) as client:
    response = await client.post(url, json=payload, headers=headers)
    response.raise_for_status()


async def deliver(
  config: DeliveryConfig,
  run_output: RunOutput,
  state: dict[str, Any] | None = None,
) -> None:
  message = (
    config.format_message(run_output, state)
    if config.format_message is not None
    else format_run_summary(run_output, label=config.telegram_label, state=state)
  )

  bot_token = str(config.telegram_bot_token or "").strip()
  chat_id = str(config.telegram_chat_id or "").strip()
  if bot_token and chat_id:
    try:
      await send_telegram(message, bot_token, chat_id)
      if config.briefing_file is not None:
        await send_telegram_file(config.briefing_file, bot_token, chat_id)
    except Exception as exc:
      log.warning("Telegram delivery failed (non-fatal): %s", exc)

  if config.webhook_url:
    try:
      await send_webhook(
        config.webhook_url,
        {
          "run_output": asdict(run_output),
          "state": dict(state or {}),
          "outcome": run_output_outcome(run_output),
          "summary": message,
        },
      )
    except Exception as exc:
      log.warning("Webhook delivery failed (non-fatal): %s", exc)

  if config.on_complete is not None:
    try:
      result = config.on_complete(run_output, state)
      if inspect.isawaitable(result):
        await result
    except Exception as exc:
      log.warning("Completion callback failed (non-fatal): %s", exc)


async def run_autonomous(
  system_prompt: str | list[tuple[str, bool]],
  initial_message: str,
  *,
  provider: str | ModelProvider = "anthropic",
  model: str | None = None,
  api_key: str | None = None,
  auth_token: str | None = None,
  auth_config: dict[str, Any] | None = None,
  provider_config: dict[str, Any] | None = None,
  max_tokens: int = 16_000,
  mcp_servers: dict[str, dict[str, Any]] | None = None,
  mcp_config_path: str | Path | None = None,
  mcp_session_inject_servers: set[str] | None = None,
  mcp_timeout_overrides: dict[str, int] | None = None,
  tool_handlers: dict[str, LocalToolHandler] | None = None,
  tool_definitions: list[dict[str, Any]] | None = None,
  skills_dir: str | Path | None = None,
  skills_excluded_tools: set[str] | None = None,
  outputs_dir: str | Path | None = None,
  needs_approval: Callable[..., bool] | None = None,
  excluded_tools: set[str] | None = None,
  interceptors: Sequence[ToolInterceptor] | None = None,
  max_turns: int = 80,
  timeout_seconds: float | None = None,
  max_budget_usd: float | None = None,
  # None by default: thinking-turn duration is unpredictable, so a wall-clock
  # per-turn cap races the runner's event-gap stall guard (which retries) and
  # terminally kills slow-first-token turns (ACUI-25). Liveness = stall guard;
  # runaway bounds = max_turns / max_budget_usd / timeout_seconds.
  per_turn_timeout: float | None = None,
  client_timeout: float = 90.0,
  max_concurrent_sub_agents: int | None = None,
  compaction_instructions: str | None = None,
  state_dir: str | Path | None = None,
  state_file: str = "state.json",
  delivery: DeliveryConfig | None = None,
  on_usage: Callable[[UsageEvent], Awaitable[Any] | Any] | None = None,
  on_session_summary: Callable[[SessionUsageSummary], Awaitable[Any] | Any] | None = None,
  on_tool_result: Callable[[ToolResultContext], Awaitable[Any] | Any] | None = None,
  on_tool_timing: Callable[..., None] | None = None,
  session_id: str | None = None,
  user_id: str,
  billing_mode: str,
  rate_table_version: str,
  coordinator: CoordinatorConfig | None = None,
) -> RunOutput:
  """Run a headless agent to completion without an HTTP server.

  Use for cron jobs, batch tasks, or as the building block for HeartbeatLoop.
  Supports MCP tools, local tool handlers, skills/sub-agents, state persistence,
  and delivery (Telegram, webhook, or callback) on completion.
  The default execution control is turn-based; pass `timeout_seconds` only for
  callers that need an explicit wall-clock SLA, and set `max_budget_usd` for
  production cost control.
  """
  provider_instance, _provider_name, resolved_auth_config = _resolve_provider(
    provider,
    model,
    api_key,
    auth_token,
    provider_config,
    auth_config=auth_config,
    max_tokens=max_tokens,
  )
  resolved_model = str(resolved_auth_config.get("model") or _get_default_model_for_provider(_provider_name))
  allowed_models = _allowed_models_for_provider(provider_instance, resolved_model)
  sid = str(session_id or f"autonomous-{secrets.token_hex(8)}")
  skill_loader = SkillLoader(skills_dir) if skills_dir else None
  excel_dispatch_config: tuple[str, str] | None = None
  excel_dispatch_enabled = os.getenv("EXCEL_ORCHESTRATION_DEV", "").strip() == "1"
  if excel_dispatch_enabled and "message_excel_agent" not in (tool_handlers or {}):
    # Non-fatal: a globally-set dev flag must not crash a non-Excel autonomous run.
    # If the gateway URL/key can't be resolved, log and skip registering the tool.
    try:
      excel_dispatch_config = _resolve_message_excel_agent_gateway_config(user_id)
    except Exception as exc:
      log.warning("message_excel_agent registration skipped (gateway config unresolved): %s", exc)
      excel_dispatch_config = None

  mcp_client: McpClientManager | None = None
  connected_servers: set[str] = set()
  active_servers: set[str] = set()
  if mcp_servers or mcp_config_path:
    builtin_names = set((tool_handlers or {}).keys())
    if excel_dispatch_config is not None:
      builtin_names.add("message_excel_agent")
    if skills_dir and "run_agent" not in builtin_names:
      builtin_names |= {"run_agent", "get_background_result", "send_message"}
    mcp_client = McpClientManager(
      inline_servers=mcp_servers,
      config_path=mcp_config_path,
      builtin_tool_names=builtin_names,
      timeout_overrides=mcp_timeout_overrides,
    )

  try:
    if mcp_client is not None:
      await mcp_client.startup()
      connected_servers = set(mcp_client.get_server_names())
      active_servers = set(connected_servers)

    local_handlers = dict(tool_handlers or {})
    extra_tool_defs = list(tool_definitions or [])
    runner_ref: list[Any] = [None]

    if excel_dispatch_config is not None and "message_excel_agent" not in local_handlers:
      gateway_url, gateway_api_key = excel_dispatch_config
      local_handlers["message_excel_agent"] = make_autonomous_message_excel_agent_handler(
        gateway_url=gateway_url,
        gateway_api_key=gateway_api_key,
      )
      if not any(definition.get("name") == "message_excel_agent" for definition in extra_tool_defs):
        extra_tool_defs.append(make_message_excel_agent_tool_def())

    if skill_loader is not None and "run_agent" not in local_handlers:
      local_handlers["run_agent"] = make_run_agent_handler(
        runner_ref,
        skill_loader=skill_loader,
        mcp_client=mcp_client or _NullMcpClient(),
        needs_approval=needs_approval,
        mcp_session_inject_servers=mcp_session_inject_servers,
        local_tool_handlers=local_handlers,
        excluded_tools=skills_excluded_tools,
        outputs_dir=Path(outputs_dir) if outputs_dir is not None else None,
        default_model=resolved_model,
        allowed_models=allowed_models,
        coordinator_config=coordinator,
      )
      if "get_background_result" not in local_handlers:
        local_handlers["get_background_result"] = make_get_background_result_handler(runner_ref)
      if "send_message" not in local_handlers:
        local_handlers["send_message"] = make_send_message_handler(runner_ref)
      if not any(definition.get("name") == "run_agent" for definition in extra_tool_defs):
        extra_tool_defs.append(make_run_agent_tool_def(skill_loader))
      if not any(definition.get("name") == "get_background_result" for definition in extra_tool_defs):
        extra_tool_defs.append(make_get_background_result_tool_def())
      if not any(definition.get("name") == "send_message" for definition in extra_tool_defs):
        extra_tool_defs.append(make_send_message_tool_def())

    def _get_tool_defs() -> list[dict[str, Any]]:
      defs: list[dict[str, Any]] = []
      if mcp_client is not None:
        defs.extend(mcp_client.get_tool_definitions())
      defs.extend(extra_tool_defs)
      return defs

    async def _combined_on_tool_result(ctx: ToolResultContext):
      if on_tool_result is None:
        return None
      result = on_tool_result(ctx)
      if inspect.isawaitable(result):
        return await result
      return result

    event_log = EventLog()
    dispatcher = ToolDispatcher(
      mcp_client=mcp_client or _NullMcpClient(),
      local_tool_handlers=local_handlers,
      needs_approval=needs_approval,
      event_log=event_log,
      interceptors=interceptors,
      session_id=sid,
      should_avoid_permission_prompts=True,
      mcp_session_inject_servers=mcp_session_inject_servers,
      get_tool_definitions=_get_tool_defs,
    )
    runner = AgentRunner(
      event_log=event_log,
      dispatcher=dispatcher,
      session_id=sid,
      provider=provider_instance,
      auth_config=resolved_auth_config,
      client_timeout=client_timeout,
      max_tokens_override=max_tokens,
      per_turn_timeout=per_turn_timeout,
      mcp_client=mcp_client,
      loaded_mcp_servers=active_servers,
      excluded_tools=excluded_tools,
      get_tool_definitions=_get_tool_defs,
      on_tool_result=_combined_on_tool_result,
      on_usage=on_usage,
      on_session_summary=on_session_summary,
      on_tool_timing=on_tool_timing,
      user_id=user_id,
      billing_mode=billing_mode,
      rate_table_version=rate_table_version,
      max_budget_usd=max_budget_usd,
      max_concurrent_sub_agents=max_concurrent_sub_agents,
      compaction_instructions=compaction_instructions,
      coordinator=coordinator,
    )
    runner_ref[0] = runner

    previous_state = load_state(state_dir, state_file=state_file) if state_dir is not None else {}
    output = await run_session(
      runner,
      event_log,
      model=resolved_model,
      max_turns=max_turns,
      timeout_seconds=timeout_seconds,
      initial_message=initial_message,
      system_prompt=system_prompt,
    )

    current_state: dict[str, Any] | None = None
    if state_dir is not None or delivery is not None:
      briefing_file = ""
      if delivery is not None and delivery.briefing_file is not None:
        briefing_file = str(delivery.briefing_file)
      current_state = build_state_payload(
        previous_state=previous_state,
        model_state=extract_state_update(output.response),
        run_output=output,
        model_name=resolved_model,
        briefing_file=briefing_file,
        connected_servers=connected_servers,
        active_servers=active_servers,
      )

    if state_dir is not None and current_state is not None:
      save_state(state_dir, current_state, state_file=state_file)

    if delivery is not None:
      await deliver(delivery, output, current_state)

    return output
  finally:
    if mcp_client is not None:
      await mcp_client.shutdown()


def run_autonomous_sync(
  system_prompt: str | list[tuple[str, bool]],
  initial_message: str,
  **kwargs: Any,
) -> RunOutput:
  try:
    asyncio.get_running_loop()
  except RuntimeError:
    return asyncio.run(run_autonomous(system_prompt, initial_message, **kwargs))
  raise RuntimeError("run_autonomous_sync cannot run inside an active asyncio loop.")


__all__ = [
  "DeliveryConfig",
  "RunOutput",
  "build_state_payload",
  "collect_run_output",
  "deliver",
  "extract_state_update",
  "format_run_summary",
  "load_state",
  "mark_post_run_guard_failure",
  "run_autonomous",
  "run_autonomous_sync",
  "run_output_exit_code",
  "run_output_outcome",
  "run_session",
  "save_state",
  "send_telegram",
  "send_telegram_file",
  "send_webhook",
  "split_messages",
]
