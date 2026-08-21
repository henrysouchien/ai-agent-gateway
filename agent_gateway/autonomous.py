from __future__ import annotations

import asyncio
import inspect
import logging
import os
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

import httpx

from ._io import _atomic_write_json, _read_json_object
from .agent_session_log import AgentSessionLog, slugify
from .autonomous_output import (
  RunOutput as RunOutput,
  _JSON_FENCE_RE as _JSON_FENCE_RE,
  _STATE_JSON_MARKER as _STATE_JSON_MARKER,
  _ensure_string_list as _ensure_string_list,
  _extract_summary as _extract_summary,
  build_state_payload as _build_state_payload,
  collect_run_output as _collect_run_output,
  extract_state_update as _extract_state_update,
  format_run_summary as _format_run_summary,
  mark_post_run_guard_failure as _mark_post_run_guard_failure,
  run_output_exit_code as _run_output_exit_code,
  run_output_outcome as _run_output_outcome,
)
from .capability_execution import (
  BoundCapabilityExecution,
  CapabilityExecutionResolver,
)
from .event_log import EventLog
from .mcp_client import McpClientManager
from .operation_catalog import AgentOperationCatalog
from .multi_user.billing import SessionUsageSummary, UsageEvent
from .openai_history_fence import (
  OPENAI_SESSION_EPOCH_ENV,
  scope_provider_session_id,
)
from .approval_policy import RunContext
from .runner import AgentRunner, ToolResultContext
from .runner_session_lifecycle import WriterLeaseAlreadyHeldError
from .skill_lifecycle import drain_owned_lifecycle_task
from .skills import SkillLoader
from .session import GatewaySession
from .sub_agent import (
  make_get_background_result_handler,
  make_get_background_result_tool_def,
  make_send_message_handler,
  make_send_message_tool_def,
  make_run_agent_handler,
  make_run_agent_tool_def,
)
from .agent_result_content import make_get_agent_result_content_tool_def
from .task_registry import CoordinatorConfig
from .tool_dispatcher import LocalToolHandler, ToolDispatcher, ToolInterceptor


log = logging.getLogger("agent_gateway.autonomous")
_RUN_SESSION_FORCE_CLOSE_SECONDS = 2.0
_RUN_SESSION_CANCEL_DRAIN_SECONDS = 5.0
_SESSION_LOG_BASE_DIR_ENV = "AGENT_SESSION_LOG_BASE_DIR"


def _trusted_autonomous_mcp_policy(
  *,
  mcp_servers: dict[str, dict[str, Any]] | None,
  mcp_config_path: str | Path | None,
  trusted_allowed_servers: set[str] | None,
  trusted_server_aliases: dict[str, str] | None,
) -> tuple[set[str] | None, dict[str, str]]:
  """Validate launcher-owned MCP admission policy before any server can start."""

  if not mcp_servers and not mcp_config_path:
    return None, {}
  if trusted_allowed_servers is None:
    raise ValueError(
      "run_autonomous requires trusted_mcp_allowed_servers when MCP "
      "configuration is supplied"
    )

  aliases = dict(trusted_server_aliases or {})
  for alias, canonical_name in aliases.items():
    if not isinstance(alias, str) or not alias.strip():
      raise ValueError("trusted MCP server aliases must have non-empty string names")
    if not isinstance(canonical_name, str) or not canonical_name.strip():
      raise ValueError("trusted MCP server aliases must target non-empty string names")

  allowed_servers: set[str] = set()
  for server_name in trusted_allowed_servers:
    if not isinstance(server_name, str) or not server_name.strip():
      raise ValueError("trusted MCP allowed servers must be non-empty strings")
    allowed_servers.add(server_name)

  canonical_allowed_servers = {
    aliases.get(server_name, server_name)
    for server_name in allowed_servers
  }
  disallowed_inline_servers: list[str] = []
  for server_name in (mcp_servers or {}):
    if not isinstance(server_name, str) or not server_name.strip():
      raise ValueError("inline MCP server names must be non-empty strings")
    if aliases.get(server_name, server_name) not in canonical_allowed_servers:
      disallowed_inline_servers.append(server_name)
  if disallowed_inline_servers:
    raise ValueError(
      "inline MCP server(s) absent from the trusted allowlist: "
      f"{', '.join(sorted(disallowed_inline_servers))}"
    )

  return allowed_servers, aliases


def _resolve_session_log_base_dir(
  configured: str | Path | None,
) -> Path:
  """Resolve durable storage for direct autonomous runs without cwd drift."""

  raw = str(configured or "").strip()
  if not raw:
    raw = os.getenv(_SESSION_LOG_BASE_DIR_ENV, "").strip()
    if raw:
      env_path = Path(raw).expanduser()
      if not env_path.is_absolute():
        raise ValueError(
          f"{_SESSION_LOG_BASE_DIR_ENV} must be an absolute path"
        )
      return env_path.resolve(strict=False)
  if raw:
    return Path(raw).expanduser().resolve(strict=False)
  return Path("~/.cache/agent-gateway/agent-sessions").expanduser()


def _autonomous_session_log(
  session: GatewaySession,
  *,
  session_id: str,
  base_dir: Path,
  supplied: AgentSessionLog | None,
) -> AgentSessionLog:
  existing = getattr(session, "agent_session_log", None)
  if supplied is not None and not isinstance(supplied, AgentSessionLog):
    raise TypeError("agent_session_log must be AgentSessionLog")
  if existing is not None and not isinstance(existing, AgentSessionLog):
    raise TypeError(
      "GatewaySession.agent_session_log must be AgentSessionLog"
    )
  if supplied is not None and existing is not None and supplied is not existing:
    raise ValueError(
      "agent_session_log does not match GatewaySession.agent_session_log"
    )
  if supplied is not None:
    session_log = supplied
  elif existing is not None:
    session_log = existing
  else:
    path = (
      base_dir
      / "autonomous"
      / (
        f"agentsess_{slugify(session_id)}_"
        f"{slugify(session.user_id)}.jsonl"
      )
    )
    session_log = AgentSessionLog(
      path=path,
      gateway_session=session,
    )
  setattr(session, "agent_session_log", session_log)
  return session_log


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


@dataclass
class DeliveryConfig:
  on_complete: Callable[[RunOutput, dict[str, Any] | None], Awaitable[None] | None] | None = None
  telegram_bot_token: str | None = None
  telegram_chat_id: str | None = None
  telegram_label: str | None = None
  briefing_file: Path | str | None = None
  webhook_url: str | None = None
  format_message: Callable[[RunOutput, dict[str, Any] | None], str] | None = None


def collect_run_output(event_log: EventLog, timed_out: bool) -> RunOutput:
  return _collect_run_output(event_log, timed_out, run_output_cls=RunOutput)


def run_output_exit_code(run_output: RunOutput) -> int:
  return _run_output_exit_code(run_output)


def run_output_outcome(run_output: RunOutput) -> str:
  return _run_output_outcome(run_output)


def mark_post_run_guard_failure(
  run_output: RunOutput,
  *,
  guard: str,
  message: str,
  details: dict[str, Any] | None = None,
) -> None:
  _mark_post_run_guard_failure(run_output, guard=guard, message=message, details=details)


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
  return _extract_state_update(text, state_json_marker=_STATE_JSON_MARKER, json_fence_re=_JSON_FENCE_RE)


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
  return _build_state_payload(
    previous_state=previous_state,
    model_state=model_state,
    run_output=run_output,
    model_name=model_name,
    briefing_file=briefing_file,
    connected_servers=connected_servers,
    active_servers=active_servers,
    extract_summary_fn=extract_summary_fn or _extract_summary,
    ensure_string_list_fn=_ensure_string_list,
  )


def format_run_summary(
  run_output: RunOutput,
  label: str | None = None,
  state: dict[str, Any] | None = None,
  format_state_fn: Callable[[dict[str, Any]], str] | None = None,
) -> str:
  return _format_run_summary(
    run_output,
    label=label,
    state=state,
    format_state_fn=format_state_fn,
    extract_summary_fn=_extract_summary,
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


def _top_level_skill_enrolled(runner: Any) -> bool:
  return bool(getattr(runner, "top_level_skill_enrolled", False))


def _set_server_terminal_cause(
  runner: Any,
  cause: str,
) -> bool:
  setter = getattr(runner, "set_server_terminal_cause", None)
  if not callable(setter):
    return False
  return bool(setter(cause))


async def _drain_enrolled_run_settlement(
  runner: Any,
  run_task: asyncio.Task[None],
) -> None:
  run_failure: Exception | None = None
  try:
    await drain_owned_lifecycle_task(run_task)
  except asyncio.CancelledError:
    pass
  except Exception as exc:
    run_failure = exc
  waiter = getattr(
    runner,
    "wait_for_top_level_skill_settlement",
    None,
  )
  if not callable(waiter):
    raise RuntimeError(
      "Enrolled top-level skill runner is missing its settlement "
      "handshake"
    )
  settlement_task = asyncio.create_task(
    waiter(),
    name="top-level-skill:settlement-handshake",
  )
  try:
    await drain_owned_lifecycle_task(settlement_task)
  except Exception as settlement_failure:
    if run_failure is not None:
      raise run_failure from settlement_failure
    raise


async def _wait_for_enrolled_run_and_settlement(
  runner: Any,
  run_task: asyncio.Task[None],
) -> None:
  run_failure: Exception | None = None
  try:
    await asyncio.shield(run_task)
  except asyncio.CancelledError:
    raise
  except Exception as exc:
    run_failure = exc
  try:
    await _wait_for_enrolled_settlement(runner)
  except Exception as settlement_failure:
    if run_failure is not None:
      raise run_failure from settlement_failure
    raise


async def _wait_for_enrolled_settlement(runner: Any) -> None:
  waiter = getattr(
    runner,
    "wait_for_top_level_skill_settlement",
    None,
  )
  if not callable(waiter):
    raise RuntimeError(
      "Enrolled top-level skill runner is missing its settlement "
      "handshake"
    )
  settlement_task = asyncio.create_task(
    waiter(),
    name="top-level-skill:settlement-handshake",
  )
  try:
    await asyncio.shield(settlement_task)
  except asyncio.CancelledError:
    await drain_owned_lifecycle_task(settlement_task)
    raise


async def run_session(
  runner: AgentRunner,
  event_log: EventLog,
  *,
  max_turns: int,
  timeout_seconds: float | None,
  initial_message: str,
  system_prompt: str | list[tuple[str, bool]] | None,
) -> RunOutput:
  timed_out = False
  error_msg: str | None = None
  lease_skip = False
  enrolled_top_level_skill = _top_level_skill_enrolled(runner)
  coro = runner.run(
    messages=[{"role": "user", "content": initial_message}],
    system_prompt=system_prompt,
    max_turns=max_turns,
  )
  run_task: asyncio.Task[None] | None = None
  try:
    if timeout_seconds is not None and timeout_seconds > 0:
      run_task = asyncio.create_task(coro)
      done, _pending = await asyncio.wait({run_task}, timeout=timeout_seconds)
      if run_task in done:
        if enrolled_top_level_skill:
          await _wait_for_enrolled_run_and_settlement(
            runner,
            run_task,
          )
        else:
          await run_task
      else:
        log.warning("Autonomous run timed out after %ss", timeout_seconds)
        cause_accepted = _set_server_terminal_cause(
          runner,
          "timeout",
        )
        if enrolled_top_level_skill and not cause_accepted:
          await _wait_for_enrolled_run_and_settlement(
            runner,
            run_task,
          )
        else:
          timed_out = True
          run_task.cancel()
          try:
            await _force_close_runner(
              runner,
              timeout=_RUN_SESSION_FORCE_CLOSE_SECONDS,
            )
          except Exception as exc:
            log.warning(
              "Autonomous runner force-close after timeout failed: %s",
              exc,
            )
          if enrolled_top_level_skill:
            await _drain_enrolled_run_settlement(
              runner,
              run_task,
            )
            current_task = asyncio.current_task()
            if (
              current_task is not None
              and current_task.cancelling()
            ):
              raise asyncio.CancelledError
          else:
            drained = await _drain_cancelled_run_task(
              run_task,
              timeout=_RUN_SESSION_CANCEL_DRAIN_SECONDS,
            )
            if not drained:
              log.warning(
                "Autonomous run cancellation did not drain within %ss "
                "after timeout",
                _RUN_SESSION_CANCEL_DRAIN_SECONDS,
              )
              run_task.add_done_callback(
                _consume_late_run_task_result
              )
    else:
      if enrolled_top_level_skill:
        run_task = asyncio.create_task(coro)
        await _wait_for_enrolled_run_and_settlement(
          runner,
          run_task,
        )
      else:
        await coro
  except asyncio.CancelledError:
    if run_task is not None and enrolled_top_level_skill:
      cause_resolver = getattr(
        runner,
        "classify_server_cancellation_cause",
        None,
      )
      cause = (
        cause_resolver()
        if callable(cause_resolver)
        else "caller_cancellation"
      )
      cause_accepted = _set_server_terminal_cause(runner, cause)
      if not run_task.done() and cause_accepted:
        run_task.cancel()
        try:
          await _force_close_runner(
            runner,
            timeout=_RUN_SESSION_FORCE_CLOSE_SECONDS,
          )
        except Exception as exc:
          log.warning(
            "Autonomous runner force-close after cancellation failed: %s",
            exc,
          )
      await _drain_enrolled_run_settlement(runner, run_task)
    elif run_task is not None and not run_task.done():
      run_task.cancel()
      try:
        await _force_close_runner(
          runner,
          timeout=_RUN_SESSION_FORCE_CLOSE_SECONDS,
        )
      except Exception as exc:
        log.warning(
          "Autonomous runner force-close after cancellation failed: %s",
          exc,
        )
      drained = await _drain_cancelled_run_task(
        run_task,
        timeout=_RUN_SESSION_CANCEL_DRAIN_SECONDS,
      )
      if not drained:
        run_task.add_done_callback(_consume_late_run_task_result)
    raise
  except WriterLeaseAlreadyHeldError as exc:
    lease_skip = True
    error_msg = f"{type(exc).__name__}: {exc}"
    log.info("Autonomous run skipped: %s", error_msg)
  except Exception as exc:
    if enrolled_top_level_skill:
      raise
    error_msg = f"{type(exc).__name__}: {exc}"
    log.error("Autonomous run failed: %s", error_msg, exc_info=True)

  output = collect_run_output(event_log, timed_out=timed_out)
  if timed_out and output.error == "missing_terminal_event":
    output.error = None
  if error_msg and output.error in {None, "missing_terminal_event"}:
    output.error = error_msg
  if lease_skip:
    output.exit_reason = "writer_lease_already_held"
  return output


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
  capability_execution: BoundCapabilityExecution,
  session: GatewaySession,
  mcp_servers: dict[str, dict[str, Any]] | None = None,
  mcp_config_path: str | Path | None = None,
  trusted_mcp_allowed_servers: set[str] | None = None,
  trusted_mcp_server_aliases: dict[str, str] | None = None,
  mcp_session_inject_servers: set[str] | None = None,
  mcp_meta_inject_servers: frozenset[str] | None = None,
  mcp_timeout_overrides: dict[str, int] | None = None,
  tool_handlers: dict[str, LocalToolHandler] | None = None,
  tool_definitions: list[dict[str, Any]] | None = None,
  skills_dir: str | Path | None = None,
  operation_catalog: AgentOperationCatalog | None = None,
  skills_excluded_tools: set[str] | None = None,
  outputs_dir: str | Path | None = None,
  agent_session_log: AgentSessionLog | None = None,
  session_log_base_dir: str | Path | None = None,
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
  max_concurrent_sub_agents: int | None = 4,
  compaction_instructions: str | None = None,
  state_dir: str | Path | None = None,
  state_file: str = "state.json",
  delivery: DeliveryConfig | None = None,
  on_usage: Callable[[UsageEvent], Awaitable[Any] | Any] | None = None,
  on_session_summary: Callable[[SessionUsageSummary], Awaitable[Any] | Any] | None = None,
  on_tool_result: Callable[[ToolResultContext], Awaitable[Any] | Any] | None = None,
  on_tool_timing: Callable[..., None] | None = None,
  session_id: str | None = None,
  top_level_skill_name: str | None = None,
  skill_run_id: str | None = None,
  user_id: str,
  billing_mode: str,
  rate_table_version: str,
  coordinator: CoordinatorConfig | None = None,
  capability_execution_resolver: CapabilityExecutionResolver | None = None,
  commercial_usage_producer: Any | None = None,
) -> RunOutput:
  """Execute one already-resolved headless session-driver execution.

  Capability, provider, credential, model, and effort selection must happen
  before this execution-only helper is called.

  Use for autonomous/cron jobs or as the building block for HeartbeatLoop.
  Supports MCP tools, local tool handlers, skills/sub-agents, state persistence,
  and delivery (Telegram, webhook, or callback) on completion.
  Inline or file-backed MCP configuration requires a launcher-owned
  `trusted_mcp_allowed_servers`; skill metadata is not an admission policy.
  The default execution control is turn-based; pass `timeout_seconds` only for
  callers that need an explicit wall-clock SLA, and set `max_budget_usd` for
  production cost control.
  Standalone top-level skill callers must bind both `top_level_skill_name` and
  a unique `skill_run_id`; these values are trusted runtime context, not model input.
  """
  if skills_dir is not None and operation_catalog is not None:
    raise ValueError(
      "run_autonomous accepts either skills_dir or operation_catalog, not both"
    )
  if type(session) is not GatewaySession:
    raise TypeError("run_autonomous requires an exact GatewaySession")
  if not isinstance(capability_execution, BoundCapabilityExecution):
    raise TypeError(
      "run_autonomous requires a BoundCapabilityExecution"
    )
  capability_execution.validate()
  capability_bind = capability_execution.bind
  if capability_bind.capability_id != "session.driver":
    raise ValueError(
      "run_autonomous requires a session.driver capability execution"
    )
  if capability_bind.run_mode not in {"autonomous", "cron"}:
    raise ValueError(
      "run_autonomous requires an autonomous or cron capability execution"
    )
  resolved_auth_config = dict(capability_execution.auth_config)
  if top_level_skill_name is None:
    if skill_run_id is not None:
      raise ValueError(
        "run_autonomous skill_run_id requires top_level_skill_name"
      )
    trusted_skill_name = None
    trusted_skill_run_id = None
    policy_bundle_hash = "unknown"
  else:
    if (
      type(top_level_skill_name) is not str
      or not top_level_skill_name
      or top_level_skill_name != top_level_skill_name.strip()
    ):
      raise ValueError(
        "run_autonomous top_level_skill_name must be canonical non-empty text"
      )
    if (
      type(skill_run_id) is not str
      or not skill_run_id
      or skill_run_id != skill_run_id.strip()
    ):
      raise ValueError(
        "run_autonomous top-level skill requires a canonical skill_run_id"
      )
    policy_bundle_hash = str(
      getattr(session.approval_policy, "policy_bundle_hash", "")
    ).strip()
    if (
      len(policy_bundle_hash) != 64
      or any(character not in "0123456789abcdef" for character in policy_bundle_hash)
    ):
      raise ValueError(
        "run_autonomous top-level skill requires a trusted policy bundle"
      )
    trusted_skill_name = top_level_skill_name
    trusted_skill_run_id = skill_run_id
  max_tokens = resolved_auth_config.get("max_tokens")
  if (
    isinstance(max_tokens, bool)
    or not isinstance(max_tokens, int)
    or max_tokens <= 0
  ):
    raise ValueError(
      "capability execution max_tokens must be a positive integer"
    )
  runtime_interceptors = tuple(interceptors or ())
  resolved_mcp_allowed_servers, resolved_mcp_server_aliases = (
    _trusted_autonomous_mcp_policy(
      mcp_servers=mcp_servers,
      mcp_config_path=mcp_config_path,
      trusted_allowed_servers=trusted_mcp_allowed_servers,
      trusted_server_aliases=trusted_mcp_server_aliases,
    )
  )
  provider_name_for_scope = capability_bind.provider
  resolved_model = capability_bind.upstream_model
  durable_session = state_dir is not None or session_id is not None
  sid = str(session_id or f"autonomous-{secrets.token_hex(8)}")
  sid = scope_provider_session_id(
    sid,
    provider=provider_name_for_scope,
    durable=durable_session,
    openai_epoch=(
      os.environ.get(OPENAI_SESSION_EPOCH_ENV)
      if provider_name_for_scope == "openai"
      else None
    ),
  )
  run_context = RunContext(
    user_id=session.user_id,
    request_id=str(getattr(session, "request_id", "") or sid),
    session_id=sid,
    run_id=trusted_skill_run_id,
    profile="autonomous",
    channel=str(session.channel or "autonomous"),
    skill=trusted_skill_name,
    decider_role=session.role,
    policy_bundle_hash=policy_bundle_hash,
  )
  session.run_context = run_context
  skill_loader = SkillLoader(skills_dir) if skills_dir else None
  operation_source = (
    operation_catalog if operation_catalog is not None else skill_loader
  )
  durable_session_log = (
    _autonomous_session_log(
      session,
      session_id=sid,
      base_dir=_resolve_session_log_base_dir(session_log_base_dir),
      supplied=agent_session_log,
    )
    if operation_source is not None
    else None
  )
  if (
    operation_source is not None
    and "run_agent" not in (tool_handlers or {})
    and capability_execution_resolver is None
  ):
    raise ValueError(
      "run_autonomous requires capability_execution_resolver before "
      "registering the run_agent child surface"
    )
  mcp_client: McpClientManager | None = None
  connected_servers: set[str] = set()
  active_servers: set[str] = set()
  if mcp_servers or mcp_config_path:
    builtin_names = set((tool_handlers or {}).keys())
    if operation_source is not None and "run_agent" not in builtin_names:
      builtin_names |= {
        "run_agent",
        "get_agent_result_content",
        "get_background_result",
        "send_message",
      }
    mcp_client = McpClientManager(
      allowed_servers=resolved_mcp_allowed_servers,
      inline_servers=mcp_servers,
      config_path=mcp_config_path,
      builtin_tool_names=builtin_names,
      timeout_overrides=mcp_timeout_overrides,
      server_aliases=resolved_mcp_server_aliases,
    )

  try:
    if mcp_client is not None:
      await mcp_client.startup()
      connected_servers = set(mcp_client.get_server_names())
      active_servers = set(connected_servers)

    local_handlers = dict(tool_handlers or {})
    extra_tool_defs = list(tool_definitions or [])
    runner_ref: list[Any] = [None]

    if operation_source is not None and "run_agent" not in local_handlers:
      local_handlers["run_agent"] = make_run_agent_handler(
        runner_ref,
        parent_session=session,
        skill_loader=skill_loader,
        operation_catalog=operation_catalog,
        mcp_client=mcp_client or _NullMcpClient(),
        needs_approval=needs_approval,
        interceptors=runtime_interceptors,
        mcp_session_inject_servers=mcp_session_inject_servers,
        local_tool_handlers=local_handlers,
        excluded_tools=skills_excluded_tools,
        outputs_dir=Path(outputs_dir) if outputs_dir is not None else None,
        capability_execution_resolver=capability_execution_resolver,
        coordinator_config=coordinator,
      )
      if "get_background_result" not in local_handlers:
        local_handlers["get_background_result"] = make_get_background_result_handler(runner_ref)
      if "send_message" not in local_handlers:
        local_handlers["send_message"] = make_send_message_handler(runner_ref)
      if not any(definition.get("name") == "run_agent" for definition in extra_tool_defs):
        extra_tool_defs.append(make_run_agent_tool_def(operation_source))
      if not any(definition.get("name") == "get_background_result" for definition in extra_tool_defs):
        extra_tool_defs.append(make_get_background_result_tool_def())
      if not any(definition.get("name") == "get_agent_result_content" for definition in extra_tool_defs):
        extra_tool_defs.append(make_get_agent_result_content_tool_def())
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
      interceptors=runtime_interceptors,
      session_id=sid,
      should_avoid_permission_prompts=True,
      mcp_session_inject_servers=mcp_session_inject_servers,
      mcp_meta_inject_servers=mcp_meta_inject_servers,
      user_id=session.user_id,
      risk_user_id=session.risk_user_id,
      channel=session.channel,
      role=session.role,
      session=session,
      run_context=run_context,
      get_tool_definitions=_get_tool_defs,
    )
    runner = AgentRunner(
      event_log=event_log,
      dispatcher=dispatcher,
      session_id=sid,
      capability_execution=capability_execution,
      allow_stub_response=False,
      client_timeout=client_timeout,
      max_tokens_override=max_tokens,
      per_turn_timeout=per_turn_timeout,
      mcp_client=mcp_client,
      mcp_activation_fold=session.mcp_activation_fold,
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
      skill_run_id=trusted_skill_run_id,
      compaction_instructions=compaction_instructions,
      coordinator=coordinator,
      agent_session_log=durable_session_log,
      commercial_usage_producer=commercial_usage_producer,
    )
    runner_ref[0] = runner

    previous_state = load_state(state_dir, state_file=state_file) if state_dir is not None else {}
    output = await run_session(
      runner,
      event_log,
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
  *,
  capability_execution: BoundCapabilityExecution,
  session: GatewaySession,
  **kwargs: Any,
) -> RunOutput:
  removed_selection_fields = sorted(
    {
      "api_key",
      "auth_config",
      "auth_token",
      "bound_auth_config",
      "capability_bind",
      "execution_transport",
      "max_tokens",
      "model",
      "provider",
      "provider_config",
    }.intersection(kwargs)
  )
  if removed_selection_fields:
    fields = ", ".join(removed_selection_fields)
    raise TypeError(
      "run_autonomous_sync no longer accepts raw execution selection "
      f"fields: {fields}"
    )
  try:
    asyncio.get_running_loop()
  except RuntimeError:
    return asyncio.run(
        run_autonomous(
          system_prompt,
          initial_message,
          capability_execution=capability_execution,
          session=session,
          **kwargs,
      )
    )
  raise RuntimeError("run_autonomous_sync cannot run inside an active asyncio loop.")


__all__ = [
  "DeliveryConfig",
  "RunOutput",
  "WriterLeaseAlreadyHeldError",
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
