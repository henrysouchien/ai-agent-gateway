from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Union

from .event_log import EventLog
from .mcp_client import McpClientManager
from .tool_dispatcher import ToolDispatcher


log = logging.getLogger("claude_gateway.runner")
MODEL_CONTEXT_LIMIT = 200_000
CONTEXT_WARNING_PCT = 80
STREAM_STALL_TIMEOUT = 30  # max seconds between stream events before watchdog cancels
STREAM_RETRY_MAX = 2
STREAM_RETRY_DELAY = 2.0
_OAUTH_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


def _estimate_tokens(text: str) -> int:
  """Rough token estimate: ~4 chars per token for English text + JSON overhead."""
  return max(1, len(text) // 4)


def _thinking_param(model: str, max_tokens: int) -> dict | None:
  """Return the correct thinking parameter for the given model, or None to omit."""
  # Models that support adaptive thinking (no budget_tokens needed)
  if any(tag in model for tag in ("sonnet-4-6", "opus-4-6")):
    return {"type": "adaptive"}

  # Models that support enabled thinking (require budget_tokens)
  if any(tag in model for tag in ("sonnet-4-5", "opus-4-5", "sonnet-4")):
    budget_tokens = min(10000, max_tokens - 1024)
    if budget_tokens >= 1024:
      return {"type": "enabled", "budget_tokens": budget_tokens}
    return None  # max_tokens too low for thinking

  # Unknown/unsupported models — omit thinking
  return None


def _format_exc(exc: Exception) -> str:
  parts = [f"{type(exc).__name__}: {repr(exc)}"]
  seen = {id(exc)}
  cause = exc.__cause__
  while cause is not None and id(cause) not in seen:
    parts.append(f"caused by {type(cause).__name__}: {repr(cause)}")
    seen.add(id(cause))
    cause = cause.__cause__
  return " | ".join(parts)


def _is_retryable_stream_error(exc: Exception) -> bool:
  try:
    import httpx
  except ImportError:
    return False

  try:
    from anthropic import APIConnectionError, APIStatusError
  except ImportError:
    return False

  if isinstance(exc, APIStatusError):
    return False
  if isinstance(exc, APIConnectionError):
    return True
  if isinstance(exc, (httpx.TransportError, httpx.StreamError)):
    return True
  return False


@dataclass
class ToolResultContext:
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
  excluded_tools: Set[str]
  system_prompt: str | None = None
  max_turns: int = 15
  model: str | None = None


OnToolResult = Callable[[ToolResultContext], Awaitable[List[Dict[str, Any]] | None]]
OnUsage = Callable[[Dict[str, Any]], None]
OnToolTiming = Callable[[str, str, str | None, int, bool, int], None]


class AgentRunner:
  def __init__(
    self,
    event_log: EventLog,
    dispatcher: ToolDispatcher,
    session_id: str,
    *,
    auth_config: Dict[str, Any] | None = None,
    client_timeout: float | None = None,
    max_tokens_override: int | None = None,
    per_turn_timeout: float | None = None,
    mcp_client: McpClientManager | None = None,
    loaded_mcp_servers: Set[str] | None = None,
    excluded_tools: Set[str] | None = None,
    get_tool_definitions: Callable[[], List[Dict[str, Any]]] | None = None,
    on_tool_result: OnToolResult | None = None,
    on_usage: OnUsage | None = None,
    on_tool_timing: OnToolTiming | None = None,
    sub_agent_config: SubAgentConfig | None = None,
  ) -> None:
    self._log = event_log
    self._dispatcher = dispatcher
    self._full_session_id = session_id or "no-session"
    self._sid = self._full_session_id[:12]
    self._auth_config = dict(auth_config or {})
    self._client_timeout = client_timeout
    self._max_tokens_override = max_tokens_override
    self._per_turn_timeout = per_turn_timeout
    self._mcp_client = mcp_client
    self._loaded_mcp_servers = loaded_mcp_servers if loaded_mcp_servers is not None else set()
    self._excluded_tools = set(excluded_tools or set())
    self._get_tool_definitions = get_tool_definitions
    self._on_tool_result = on_tool_result
    self._on_usage = on_usage
    self._on_tool_timing = on_tool_timing
    self._sub_agent_config = sub_agent_config
    self._active_client: Any | None = None

  def _append(self, event: Dict[str, Any]) -> None:
    self._log.append(event)

  @staticmethod
  def _annotate_result(result: Any, tool_name: str = "") -> Any:
    """Add _runner_warning to generic results with detectable anomalies."""
    if not isinstance(result, dict):
      return result

    warnings: List[str] = []

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

  @staticmethod
  def _has_active_credential(config: Dict[str, Any]) -> bool:
    if config.get("auth_mode", "api") == "oauth":
      return bool(config.get("auth_token"))
    return bool(config.get("api_key"))

  async def _stub_stream(self, messages: List[Dict[str, Any]]) -> AsyncIterator[str]:
    last_user = next((msg for msg in reversed(messages) if msg.get("role") == "user"), {})
    prompt = last_user.get("content") or "your request"
    response = f"Stub response (no Anthropic credential configured). You asked: {prompt}"
    for token in response.split():
      yield token + " "
      await asyncio.sleep(0.05)

  def _create_client(self, timeout: float | None = None) -> Any:
    from anthropic import AsyncAnthropic
    import httpx

    config = self._auth_config
    mode = str(config.get("auth_mode", "api")).strip().lower()
    client_kwargs: Dict[str, Any] = {}
    actual_timeout = timeout if timeout is not None else self._client_timeout
    if actual_timeout is not None:
      client_kwargs["timeout"] = httpx.Timeout(timeout=actual_timeout, connect=5.0)
    if mode == "oauth":
      from anthropic import Omit
      return AsyncAnthropic(
        api_key="",
        auth_token=str(config.get("auth_token", "")),
        default_headers={
          "X-Api-Key": Omit(),
          "anthropic-beta": "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14",
          "user-agent": "claude-cli/2026.3.14",
          "x-app": "cli",
        },
        **client_kwargs,
      )
    return AsyncAnthropic(
      api_key=str(config.get("api_key", "")),
      auth_token="",
      **client_kwargs,
    )

  def _set_client(self, client: Any) -> None:
    """Track the active httpx client for cleanup on cancellation."""
    self._active_client = client

  async def _close_client(self, client: Any, timeout: float = 2.0) -> None:
    """Close a client and clear tracking if it's the active one."""
    if self._active_client is client:
      self._active_client = None
    try:
      await asyncio.wait_for(client.aclose(), timeout=timeout)
    except Exception:
      pass

  async def force_close(self, timeout: float = 2.0) -> None:
    """Force-close the active client, if any. Safe to call multiple times."""
    client = self._active_client
    if client is not None:
      self._active_client = None
      try:
        await asyncio.wait_for(client.aclose(), timeout=timeout)
      except Exception:
        pass

  async def spawn_sub_agent(
    self,
    task: str,
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    dispatcher: ToolDispatcher,
    excluded_tools: Set[str] | None = None,
    max_turns: int | None,
    timeout: float | None,
    client_timeout: float = 90,
    per_turn_timeout: float | None = None,
    max_tokens: int = 32000,
    call_index: int = 0,
    on_sub_event: Optional[Callable[[Dict[str, Any], str], None]] = None,
  ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    if self._sub_agent_config is not None:
      if model is None:
        model = self._sub_agent_config.model
      if system_prompt is None:
        system_prompt = self._sub_agent_config.system_prompt
      if excluded_tools is None:
        excluded_tools = set(self._sub_agent_config.excluded_tools)

    sub_session_id = f"sub{call_index}:{self._sid}"
    original_on_event = getattr(self._log, "_on_event", None)

    def _composed_on_event(event: Dict[str, Any], session_id: str) -> None:
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
      auth_config=self._auth_config,
      client_timeout=client_timeout,
      max_tokens_override=max_tokens,
      per_turn_timeout=per_turn_timeout if per_turn_timeout is not None else self._per_turn_timeout,
      mcp_client=self._mcp_client,
      loaded_mcp_servers=self._loaded_mcp_servers,
      excluded_tools=excluded_tools or set(),
      get_tool_definitions=self._get_tool_definitions,
      on_tool_result=self._on_tool_result,
      on_usage=self._on_usage,
      on_tool_timing=self._on_tool_timing,
      sub_agent_config=self._sub_agent_config,
    )

    timed_out = False
    try:
      await asyncio.wait_for(
        sub_runner.run(
          messages=[{"role": "user", "content": task}],
          system_prompt=system_prompt,
          model_override=model,
          max_turns=max_turns,
        ),
        timeout=timeout,
      )
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

  def _call_on_usage(self, usage_payload: Dict[str, Any]) -> None:
    if self._on_usage is None:
      return
    try:
      self._on_usage(usage_payload)
    except Exception as exc:
      log.warning("[%s] on_usage hook failed (non-fatal): %s", self._sid, exc)

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
    self._append(
      {
        "type": "tool_call_start",
        "tool_call_id": tool_id,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "execution_location": "backend",
        "call_index": call_index,
      }
    )

    tool_t0 = time.time()
    server = self._mcp_client.get_server_for_tool(tool_name) if self._mcp_client is not None else None
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
        result, error = await self._dispatcher.dispatch(
          tool_id,
          tool_name,
          tool_input,
          call_index=call_index,
        )

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
      self._append(
        {
          "type": "tool_call_complete",
          "tool_call_id": tool_id,
          "tool_name": tool_name,
          "result": result,
          "error": error,
          "duration_ms": duration_ms,
          "server": server,
        }
      )
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
    try:
      config = {
        "auth_mode": str(self._auth_config.get("auth_mode", "api")).strip().lower(),
        "api_key": str(self._auth_config.get("api_key", "")),
        "auth_token": str(self._auth_config.get("auth_token", "")),
        "model": str(self._auth_config.get("model", "claude-sonnet-4-6")),
        "max_tokens": int(self._auth_config.get("max_tokens", 16000)),
        "thinking": bool(self._auth_config.get("thinking", True)),
      }
      if model_override:
        config["model"] = model_override

      if not self._has_active_credential(config):
        async for delta in self._stub_stream(messages):
          self._append({"type": "text_delta", "text": delta})
        self._append({"type": "stream_complete", "usage": {}})
        return

      try:
        client = self._create_client(timeout=self._client_timeout)
        self._set_client(client)
      except Exception:
        async for delta in self._stub_stream(messages):
          self._append({"type": "text_delta", "text": delta})
        self._append({"type": "stream_complete", "usage": {}})
        return

      cached_tools = self._default_tool_definitions()
      if self._excluded_tools:
        cached_tools = [tool for tool in cached_tools if tool["name"] not in self._excluded_tools]

      is_oauth = str(config.get("auth_mode", "api")).strip().lower() == "oauth"
      system_blocks = None
      # OAuth requires Claude Code identity as the first system block
      if is_oauth:
        system_blocks = [{"type": "text", "text": _OAUTH_IDENTITY}]
      if system_prompt:
        if isinstance(system_prompt, list):
          if system_blocks is None:
            system_blocks = []
          for block_text, should_cache in system_prompt:
            if not block_text:
              continue
            block: Dict[str, Any] = {"type": "text", "text": block_text}
            if should_cache:
              block["cache_control"] = {"type": "ephemeral"}
            system_blocks.append(block)
          if not system_blocks:
            system_blocks = None
        else:
          if system_blocks is None:
            system_blocks = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
          else:
            system_blocks.append({"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}})

      max_tokens = self._max_tokens_override if self._max_tokens_override is not None else config["max_tokens"]
      if config["thinking"] and max_tokens >= 2048:
        thinking_param = _thinking_param(config["model"], max_tokens)
      else:
        thinking_param = None

      base_kwargs: Dict[str, Any] = {
        "model": config["model"],
        "max_tokens": max_tokens,
        "messages": messages,
        "tools": cached_tools,
      }
      if system_blocks:
        base_kwargs["system"] = system_blocks
      if thinking_param is not None:
        base_kwargs["thinking"] = thinking_param
        log.info("[%s] Thinking enabled | %s | max_tokens=%d", self._sid, thinking_param, max_tokens)
      else:
        if not config["thinking"]:
          log.info("[%s] Thinking disabled | thinking=false", self._sid)
        elif max_tokens < 2048:
          log.info("[%s] Thinking disabled | max_tokens=%d too low (need >=2048)", self._sid, max_tokens)
        else:
          log.info("[%s] Thinking disabled | model=%s not supported", self._sid, config["model"])

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
      total_input_tokens = 0
      total_output_tokens = 0
      total_cache_creation_tokens = 0
      total_cache_read_tokens = 0
      turn_count = 0
      tools_used: List[str] = []

      current_messages = list(messages)

      while True:
        turn_count += 1
        if max_turns is not None and turn_count > max_turns:
          log.warning("[%s] Max turns (%d) reached, stopping", self._sid, max_turns)
          self._append({"type": "text_delta", "text": "\n\n[Sub-agent reached maximum turn limit]"})
          break
        turn_t0 = time.time()
        first_token_t: Optional[float] = None

        full_text = ""
        tool_uses: List[Tuple[str, str, Dict[str, Any]]] = []
        stop_reason: Optional[str] = None
        current_tool_id: Optional[str] = None
        current_tool_name: Optional[str] = None
        current_tool_json = ""
        current_block_type = None
        current_thinking_text = ""
        current_signature = ""
        current_text_block = ""
        content_blocks: List[Dict[str, Any]] = []

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

        last_event_at = time.monotonic()

        async def _do_stream() -> None:
          nonlocal total_input_tokens, total_output_tokens, total_cache_creation_tokens
          nonlocal total_cache_read_tokens, full_text, tool_uses, content_blocks
          nonlocal stop_reason, first_token_t, current_tool_id, current_tool_name
          nonlocal current_tool_json, current_block_type, current_thinking_text
          nonlocal current_signature, current_text_block, last_event_at
          log.debug("[%s] Turn %d stream open", self._sid, turn_count)
          async with client.messages.stream(**{**base_kwargs, "messages": current_messages}) as stream:
            async for event in stream:
              last_event_at = time.monotonic()
              event_type = getattr(event, "type", None)

              if event_type == "message_start":
                msg = getattr(event, "message", None)
                if msg:
                  usage = getattr(msg, "usage", None)
                  if usage:
                    input_tokens = getattr(usage, "input_tokens", 0)
                    total_input_tokens += input_tokens
                    total_cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0)
                    total_cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0)
                    if turn_count == 1:
                      log.info(
                        "[%s] Cache | read=%d create=%d uncached=%d",
                        self._sid,
                        getattr(usage, "cache_read_input_tokens", 0),
                        getattr(usage, "cache_creation_input_tokens", 0),
                        input_tokens,
                      )
                    if turn_count == 1 and input_tokens > 0:
                      msgs_chars = len(json.dumps(current_messages, default=str))
                      total_chars = system_chars + tools_chars + msgs_chars
                      if total_chars > 0:
                        pct_system = round(system_chars / total_chars * 100)
                        pct_tools = round(tools_chars / total_chars * 100)
                        pct_messages = round(msgs_chars / total_chars * 100)
                        tok_system = round(input_tokens * system_chars / total_chars)
                        tok_tools = round(input_tokens * tools_chars / total_chars)
                        tok_messages = input_tokens - tok_system - tok_tools
                        log.info(
                          "[%s] Token breakdown | system=%d (%d%%) tools=%d (%d%%) messages=%d (%d%%) | total=%d",
                          self._sid,
                          tok_system,
                          pct_system,
                          tok_tools,
                          pct_tools,
                          tok_messages,
                          pct_messages,
                          input_tokens,
                          extra={
                            "data": {
                              "event": "token_breakdown",
                              "session_id": self._sid,
                              "turn": turn_count,
                              "input_tokens": input_tokens,
                              "est_system_tokens": tok_system,
                              "est_tools_tokens": tok_tools,
                              "est_messages_tokens": tok_messages,
                              "pct_system": pct_system,
                              "pct_tools": pct_tools,
                              "pct_messages": pct_messages,
                            }
                          },
                        )

              elif event_type == "content_block_start":
                block = getattr(event, "content_block", None)
                if block:
                  block_type = getattr(block, "type", None)
                  if block_type == "tool_use":
                    current_tool_id = getattr(block, "id", None)
                    current_tool_name = getattr(block, "name", None)
                    current_tool_json = ""
                    current_block_type = "tool_use"
                  elif block_type == "thinking":
                    current_thinking_text = ""
                    current_signature = ""
                    current_block_type = "thinking"
                  elif block_type == "text":
                    current_text_block = ""
                    current_block_type = "text"

              elif event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta:
                  delta_type = getattr(delta, "type", None)
                  if delta_type == "text_delta":
                    text = getattr(delta, "text", "")
                    if text:
                      if first_token_t is None:
                        first_token_t = time.time()
                      self._append({"type": "text_delta", "text": text})
                      full_text += text
                      current_text_block += text
                  elif delta_type == "input_json_delta":
                    partial = getattr(delta, "partial_json", "")
                    if partial:
                      current_tool_json += partial
                  elif delta_type == "thinking_delta":
                    thinking_text = getattr(delta, "thinking", "")
                    if thinking_text:
                      self._append({"type": "thinking_delta", "text": thinking_text})
                      current_thinking_text += thinking_text
                  elif delta_type == "signature_delta":
                    sig = getattr(delta, "signature", "")
                    if sig:
                      current_signature += sig

              elif event_type == "content_block_stop":
                if current_block_type == "thinking":
                  content_blocks.append(
                    {
                      "type": "thinking",
                      "thinking": current_thinking_text,
                      "signature": current_signature,
                    }
                  )
                  log.info("[%s] Thinking block complete | %d chars", self._sid, len(current_thinking_text))
                  current_block_type = None
                elif current_block_type == "tool_use" and current_tool_id is not None:
                  try:
                    tool_input = json.loads(current_tool_json) if current_tool_json else {}
                  except json.JSONDecodeError:
                    tool_input = {}
                  tool_uses.append((current_tool_id, current_tool_name or "tool", tool_input))
                  content_blocks.append(
                    {
                      "type": "tool_use",
                      "id": current_tool_id,
                      "name": current_tool_name,
                      "input": tool_input,
                    }
                  )
                  current_tool_id = None
                  current_tool_name = None
                  current_tool_json = ""
                  current_block_type = None
                elif current_block_type == "text":
                  if current_text_block:
                    content_blocks.append({"type": "text", "text": current_text_block})
                  current_text_block = ""
                  current_block_type = None

              elif event_type == "message_delta":
                delta = getattr(event, "delta", None)
                if delta:
                  stop_reason = getattr(delta, "stop_reason", None)
                usage = getattr(event, "usage", None)
                if usage:
                  total_output_tokens += getattr(usage, "output_tokens", 0)

        # -- Stream watchdog: detect stalls and enforce turn timeout --
        _guard_reason: str | None = None

        async def _stream_guard(task: asyncio.Task) -> None:
          nonlocal _guard_reason
          stall_limit = STREAM_STALL_TIMEOUT
          turn_limit = self._per_turn_timeout
          start = time.monotonic()
          while not task.done():
            await asyncio.sleep(2.0)
            now = time.monotonic()
            stall = now - last_event_at
            if stall > stall_limit:
              _guard_reason = f"no stream events for {stall:.0f}s"
              log.error("[%s] Turn %d watchdog: %s", self._sid, turn_count, _guard_reason)
              task.cancel()
              return
            if turn_limit is not None and (now - start) > turn_limit:
              _guard_reason = f"turn timeout after {now - start:.0f}s"
              log.error("[%s] Turn %d watchdog: %s", self._sid, turn_count, _guard_reason)
              task.cancel()
              return

        stream_error: Exception | None = None
        for _attempt in range(1 + STREAM_RETRY_MAX):
          if _attempt > 0:
            full_text = ""
            tool_uses = []
            content_blocks = []
            stop_reason = None
            first_token_t = None
            current_tool_id = None
            current_tool_name = None
            current_tool_json = ""
            current_block_type = None
            current_thinking_text = ""
            current_signature = ""
            current_text_block = ""
            last_event_at = time.monotonic()

            await self._close_client(client, timeout=2.0)
            client = self._create_client(timeout=self._client_timeout)
            self._set_client(client)

            log.warning(
              "[%s] Stream retry %d/%d on turn %d after %s",
              self._sid,
              _attempt,
              STREAM_RETRY_MAX,
              turn_count,
              _format_exc(stream_error) if stream_error is not None else "unknown error",
            )
            await asyncio.sleep(STREAM_RETRY_DELAY)

          _tokens_snapshot = (
            total_input_tokens,
            total_output_tokens,
            total_cache_creation_tokens,
            total_cache_read_tokens,
          )
          _guard_reason = None
          stream_task = asyncio.create_task(_do_stream())
          guard_task = asyncio.create_task(_stream_guard(stream_task))
          try:
            done, pending = await asyncio.wait(
              {stream_task, guard_task},
              return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
              t.cancel()
            # Give cancelled tasks time to clean up; escalate if stuck
            if pending:
              _, stuck = await asyncio.wait(pending, timeout=5.0)
              if stuck:
                log.warning(
                  "[%s] Turn %d: cancelled task stuck, force-closing client",
                  self._sid,
                  turn_count,
                )
                await self._close_client(client, timeout=2.0)
                # Drain remaining tasks after force-close
                await asyncio.wait(stuck, timeout=2.0)
            # Re-raise stream exception if stream completed with error
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
            (
              total_input_tokens,
              total_output_tokens,
              total_cache_creation_tokens,
              total_cache_read_tokens,
            ) = _tokens_snapshot

            if _guard_reason:
              log.error(
                "[%s] Stream watchdog on turn %d after %.1fs (%s): %s",
                self._sid,
                turn_count,
                time.time() - turn_t0,
                _guard_reason,
                _format_exc(exc),
              )
              self._append({"type": "error", "error": f"Stream watchdog: {_guard_reason}"})
              await self._close_client(client, timeout=5.0)
              return

            if not _is_retryable_stream_error(exc):
              formatted_exc = _format_exc(exc)
              log.error(
                "[%s] Stream error on turn %d after %.1fs (non-retryable): %s",
                self._sid,
                turn_count,
                time.time() - turn_t0,
                formatted_exc,
              )
              self._append({"type": "error", "error": formatted_exc})
              await self._close_client(client, timeout=5.0)
              return

            formatted_exc = _format_exc(exc)
            log.warning(
              "[%s] Transient stream error on turn %d after %.1fs (attempt %d/%d): %s",
              self._sid,
              turn_count,
              time.time() - turn_t0,
              _attempt + 1,
              1 + STREAM_RETRY_MAX,
              formatted_exc,
            )
            if _attempt < STREAM_RETRY_MAX:
              self._append({"type": "stream_retry", "attempt": _attempt, "error": formatted_exc})
              continue
          else:
            stream_error = None

            if _guard_reason:
              self._append({"type": "error", "error": f"Stream watchdog: {_guard_reason}"})
              await self._close_client(client, timeout=5.0)
              return

            break

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
          return
        log.debug("[%s] Turn %d stream end", self._sid, turn_count)

        turn_elapsed = time.time() - turn_t0
        ttft = (first_token_t - turn_t0) if first_token_t else None
        text_len = len(full_text)
        tool_names = [tool[1] for tool in tool_uses]
        text_preview = full_text[:150].replace("\n", " ") if full_text else ""

        log.info(
          "[%s] Turn %d complete | %.1fs | TTFT=%.2fs | text=%d chars | tools=%s | stop=%s | response=%s",
          self._sid,
          turn_count,
          turn_elapsed,
          ttft if ttft is not None else -1,
          text_len,
          tool_names or "none",
          stop_reason,
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
              "stop_reason": stop_reason,
            }
          },
        )

        if not tool_uses:
          break

        assistant_content = list(content_blocks)
        current_messages.append({"role": "assistant", "content": assistant_content})

        content_blocks = []
        full_text = ""

        tool_results_content: List[Dict[str, Any]] = []
        i = 0
        run_agent_seq = 0
        while i < len(tool_uses):
          tool_id, tool_name, tool_input = tool_uses[i]
          if tool_name == "run_agent" and "run_agent" not in self._excluded_tools:
            batch: List[Tuple[int, str, str, Dict[str, Any]]] = []
            call_indices: List[int] = []
            while i < len(tool_uses):
              batch_tool_id, batch_tool_name, batch_tool_input = tool_uses[i]
              if batch_tool_name != "run_agent" or "run_agent" in self._excluded_tools:
                break
              batch.append((i, batch_tool_id, batch_tool_name, batch_tool_input))
              call_indices.append(run_agent_seq)
              run_agent_seq += 1
              i += 1

            results = await asyncio.gather(
              *[
                self._execute_single_tool(
                  batch_tool_id,
                  batch_tool_name,
                  batch_tool_input,
                  base_kwargs,
                  call_index=call_index,
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

        if stop_reason == "end_turn":
          break

      total_elapsed = time.time() - chat_t0
      cache_status = "miss"
      if total_cache_read_tokens > 0:
        cache_status = f"hit ({total_cache_read_tokens} tokens cached)"
      elif total_cache_creation_tokens > 0:
        cache_status = f"write ({total_cache_creation_tokens} tokens written)"

      uncached_input = max(0, total_input_tokens - total_cache_read_tokens - total_cache_creation_tokens)
      cost = (
        uncached_input * 3.00 / 1_000_000
        + total_cache_read_tokens * 0.30 / 1_000_000
        + total_cache_creation_tokens * 3.75 / 1_000_000
        + total_output_tokens * 15.00 / 1_000_000
      )

      log.info(
        "[%s] Chat done | %.1fs total | %d turns | tools=%s | tokens in=%d out=%d | cache=%s | cost=$%.4f",
        self._sid,
        total_elapsed,
        turn_count,
        tools_used or "none",
        total_input_tokens,
        total_output_tokens,
        cache_status,
        cost,
        extra={
          "data": {
            "event": "chat_done",
            "session_id": self._sid,
            "elapsed_s": round(total_elapsed, 1),
            "turns": turn_count,
            "tools": tools_used,
            "tokens_in": total_input_tokens,
            "tokens_out": total_output_tokens,
            "cache_read": total_cache_read_tokens,
            "cache_write": total_cache_creation_tokens,
            "cost": round(cost, 4),
          }
        },
      )

      usage_payload = {
        "session_id": self._full_session_id,
        "turns": turn_count,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cache_read_input_tokens": total_cache_read_tokens,
        "cache_creation_input_tokens": total_cache_creation_tokens,
        "estimated_cost": round(cost, 4),
      }
      self._call_on_usage(usage_payload)

      self._append(
        {
          "type": "stream_complete",
          "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cache_creation_input_tokens": total_cache_creation_tokens,
            "cache_read_input_tokens": total_cache_read_tokens,
            "estimated_cost": round(cost, 4),
          },
        }
      )

      await self._close_client(client, timeout=5.0)
    finally:
      await self.force_close()
