from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .event_log import EventLog
from .providers.agent_sdk import AgentSDKConfig, estimate_cost, _validate_sdk_version
from .runner import ToolResultContext


log = logging.getLogger("agent_gateway.sdk_runner")

OnToolResult = Callable[[ToolResultContext], Awaitable[List[Dict[str, Any]] | None]]
OnUsage = Callable[[Dict[str, Any]], None]
OnToolTiming = Callable[[str, str, str | None, int, bool, int], None]


@dataclass
class ToolCallInfo:
  tool_call_id: str
  tool_name: str
  tool_input: Dict[str, Any]
  started_at: float


@dataclass
class _ActiveToolUse:
  tool_call_id: str
  tool_name: str
  input_json: str = ""
  raw_block: Any = None


def _as_plain_dict(value: Any) -> Any:
  if value is None:
    return None
  if isinstance(value, dict):
    return {key: _as_plain_dict(item) for key, item in value.items()}
  if isinstance(value, list):
    return [_as_plain_dict(item) for item in value]
  if hasattr(value, "model_dump"):
    try:
      return _as_plain_dict(value.model_dump())
    except Exception:
      pass
  if hasattr(value, "__dict__"):
    return {
      key: _as_plain_dict(item)
      for key, item in vars(value).items()
      if not key.startswith("_")
    }
  return value


def _get_attr(value: Any, key: str, default: Any = None) -> Any:
  if isinstance(value, Mapping):
    return value.get(key, default)
  return getattr(value, key, default)


def _as_dict(value: Any) -> Dict[str, Any]:
  plain = _as_plain_dict(value)
  if isinstance(plain, dict):
    return plain
  return {}


def _extract_text(value: Any) -> str:
  if value is None:
    return ""
  if isinstance(value, str):
    return value
  if isinstance(value, list):
    chunks: List[str] = []
    for item in value:
      if isinstance(item, str):
        if item:
          chunks.append(item)
        continue
      text = _get_attr(item, "text")
      if isinstance(text, str) and text:
        chunks.append(text)
        continue
      item_type = _get_attr(item, "type")
      if item_type == "text":
        block_text = _get_attr(item, "text")
        if isinstance(block_text, str) and block_text:
          chunks.append(block_text)
    return "\n".join(chunks).strip()
  return str(value)


def _parse_result_payload(value: Any) -> Any:
  if isinstance(value, (dict, list)):
    return _as_plain_dict(value)
  if isinstance(value, str):
    stripped = value.strip()
    if not stripped:
      return ""
    try:
      return json.loads(stripped)
    except json.JSONDecodeError:
      return value
  if value is None:
    return None
  return _as_plain_dict(value)


def _summarize_error_payload(value: Any) -> str:
  parsed = _parse_result_payload(value)
  if isinstance(parsed, dict):
    inner = parsed.get("error")
    if isinstance(inner, dict):
      message = inner.get("message")
      if isinstance(message, str) and message.strip():
        return message.strip()
    message = parsed.get("message")
    if isinstance(message, str) and message.strip():
      return message.strip()
    if parsed.get("success") is False:
      return "success=false"
    status = parsed.get("status")
    if isinstance(status, str) and status.strip():
      return f"status={status.strip()}"
    return json.dumps(parsed, default=str)
  if isinstance(parsed, list):
    return json.dumps(parsed, default=str)
  return str(parsed or "Tool failed")


def _join_system_prompt(system_prompt: str | List[Tuple[str, bool]] | None) -> str:
  if system_prompt is None:
    return ""
  if isinstance(system_prompt, str):
    return system_prompt
  return "\n\n".join(text for text, _should_cache in system_prompt if text)


def _server_for_tool(tool_name: str) -> str | None:
  if not tool_name.startswith("mcp__"):
    return None
  parts = tool_name.split("__", 2)
  if len(parts) < 3 or not parts[1]:
    return None
  return parts[1]


class AgentSDKRunner:
  """Run a conversation through the Anthropic agent SDK.

  This is an alternative to `AgentRunner` when you want to delegate tool-loop
  execution to the pinned SDK while keeping the same gateway HTTP surface.
  """

  def __init__(
    self,
    event_log: EventLog,
    session_id: str,
    *,
    sdk_config: AgentSDKConfig,
    system_prompt: str,
    disallowed_tools: list[str] | None = None,
    mcp_server_configs: dict | None = None,
    max_turns: int | None = None,
    on_usage: Callable[..., Any] | None = None,
    on_tool_result: Callable[..., Any] | None = None,
    on_tool_timing: Callable[..., Any] | None = None,
  ) -> None:
    self._log = event_log
    self._session_id = session_id or "no-session"
    self._sid = self._session_id[:12]
    self._sdk_config = sdk_config
    self._system_prompt = system_prompt
    self._disallowed_tools = list(disallowed_tools or sdk_config.disallowed_tools)
    self._mcp_server_configs = dict(mcp_server_configs or {})
    self._max_turns = max_turns
    self._on_usage = on_usage
    self._on_tool_result = on_tool_result
    self._on_tool_timing = on_tool_timing
    self._pending_tool_calls: Dict[str, ToolCallInfo] = {}
    self._active_tool_use: _ActiveToolUse | None = None
    self._query_iter: Any = None
    self._usage: Dict[str, Any] = {
      "input_tokens": 0,
      "output_tokens": 0,
      "cache_creation_input_tokens": 0,
      "cache_read_input_tokens": 0,
    }
    self._num_turns = 0
    self._stream_terminal_emitted = False
    self._effective_model = sdk_config.model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6").strip()

  def _append(self, event: Dict[str, Any]) -> None:
    self._log.append(event)

  async def _call_on_usage(self, usage_payload: Dict[str, Any]) -> None:
    if self._on_usage is None:
      return
    try:
      result = self._on_usage(usage_payload)
      if inspect.isawaitable(result):
        await result
    except Exception as exc:
      log.warning("[%s] on_usage hook failed (non-fatal): %s", self._sid, exc)

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
        self._session_id,
        tool_name,
        server,
        duration_ms,
        is_error,
        result_bytes,
      )
    except Exception as exc:
      log.warning("[%s] on_tool_timing hook failed (non-fatal): %s", self._sid, exc)

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

  def _build_prompt(self, messages: List[Dict[str, Any]]) -> str:
    normalized: List[Tuple[str, str]] = []
    for message in messages:
      role = str(message.get("role") or "user").strip().lower() or "user"
      content = str(message.get("content") or "")
      if not content:
        continue
      normalized.append((role, content))

    if not normalized:
      return ""
    if len(normalized) == 1 and normalized[0][0] == "user":
      return normalized[0][1]

    transcript = "\n\n".join(f"{role.upper()}: {content}" for role, content in normalized[:-1])
    last_role, last_content = normalized[-1]
    if last_role == "user":
      if transcript:
        return (
          "You are continuing a stateless conversation. Use the transcript below as prior context.\n\n"
          f"{transcript}\n\n"
          "Respond to the latest user message below.\n\n"
          f"USER: {last_content}"
        )
      return last_content

    combined = "\n\n".join(f"{role.upper()}: {content}" for role, content in normalized)
    return (
      "You are continuing a stateless conversation. Use the transcript below as prior context and continue appropriately.\n\n"
      f"{combined}"
    )

  def _make_result_entry(
    self,
    tool_call_id: str,
    result: Any | None,
    error: Dict[str, Any] | None,
  ) -> Dict[str, Any]:
    if error is not None:
      return {
        "type": "tool_result",
        "tool_use_id": tool_call_id,
        "content": json.dumps({"error": error}, default=str),
        "is_error": True,
      }
    entry = {
      "type": "tool_result",
      "tool_use_id": tool_call_id,
      "content": json.dumps(result, default=str),
    }
    if isinstance(result, dict) and (result.get("success") is False or result.get("status") == "error"):
      entry["is_error"] = True
    return entry

  def _format_additional_context(
    self,
    *,
    tool_name: str,
    result_entry: Dict[str, Any],
    extra_blocks: Sequence[Dict[str, Any]],
  ) -> str | None:
    parts: List[str] = []
    parsed = _parse_result_payload(result_entry.get("content"))
    if isinstance(parsed, dict):
      warning = parsed.get("_runner_warning") or parsed.get("warning")
      if isinstance(warning, str) and warning.strip():
        parts.append(f"WARNING: {warning.strip()}")

      if result_entry.get("is_error") is True:
        summary = _summarize_error_payload(parsed)
        parts.append(
          f"ERROR: The previous tool call ({tool_name}) returned a structured error: {summary}. "
          "Treat this result as a failure."
        )

    for block in extra_blocks:
      block_type = str(block.get("type") or "")
      if block_type == "text":
        text = str(block.get("text") or "").strip()
        if text:
          parts.append(text)
        continue
      parts.append(json.dumps(block, default=str))

    if not parts:
      return None
    return "\n\n".join(part for part in parts if part)

  async def _build_hook_additional_context(
    self,
    *,
    tool_call_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
    result: Any | None,
    error: Dict[str, Any] | None,
  ) -> str | None:
    pending = self._pending_tool_calls.get(tool_call_id)
    duration_ms = int((time.time() - pending.started_at) * 1000) if pending is not None else 0
    result_entry = self._make_result_entry(tool_call_id, result, error)
    extra_blocks = await self._call_on_tool_result(
      ToolResultContext(
        tool_name=tool_name,
        tool_input=dict(tool_input),
        result=result,
        error=error,
        duration_ms=duration_ms,
        tool_call_id=tool_call_id,
        session_id=self._session_id,
        server=_server_for_tool(tool_name),
        result_entry=result_entry,
      )
    )
    additional_context = self._format_additional_context(
      tool_name=tool_name,
      result_entry=result_entry,
      extra_blocks=extra_blocks,
    )
    if additional_context:
      log.info("[%s] Injecting additionalContext for %s", self._sid, tool_name)
    return additional_context

  async def _post_tool_use_hook(
    self,
    input_data: Dict[str, Any],
    tool_use_id: str | None,
    _context: Any,
  ) -> Dict[str, Any]:
    tool_name = str(input_data.get("tool_name") or input_data.get("name") or "")
    tool_input = _as_dict(input_data.get("tool_input") or input_data.get("input"))
    result = _parse_result_payload(
      input_data.get("result", input_data.get("tool_result", input_data.get("output")))
    )
    additional_context = await self._build_hook_additional_context(
      tool_call_id=str(tool_use_id or input_data.get("tool_use_id") or ""),
      tool_name=tool_name,
      tool_input=tool_input,
      result=result,
      error=None,
    )
    if not additional_context:
      return {}
    return {
      "hookSpecificOutput": {
        "hookEventName": "PostToolUse",
        "additionalContext": additional_context,
      }
    }

  async def _post_tool_use_failure_hook(
    self,
    input_data: Dict[str, Any],
    tool_use_id: str | None,
    _context: Any,
  ) -> Dict[str, Any]:
    tool_name = str(input_data.get("tool_name") or input_data.get("name") or "")
    tool_input = _as_dict(input_data.get("tool_input") or input_data.get("input"))
    error_message = _summarize_error_payload(
      input_data.get("error", input_data.get("result", input_data.get("message")))
    )
    error = {
      "code": str(input_data.get("code") or "tool_error"),
      "message": error_message,
    }
    additional_context = await self._build_hook_additional_context(
      tool_call_id=str(tool_use_id or input_data.get("tool_use_id") or ""),
      tool_name=tool_name,
      tool_input=tool_input,
      result=None,
      error=error,
    )
    if not additional_context:
      return {}
    return {
      "hookSpecificOutput": {
        "hookEventName": "PostToolUseFailure",
        "additionalContext": additional_context,
      }
    }

  def _build_hooks(self, hook_matcher_cls: Any) -> Dict[str, List[Any]]:
    hooks: Dict[str, List[Any]] = {}
    if self._on_tool_result is not None:
      hooks["PostToolUse"] = [hook_matcher_cls(hooks=[self._post_tool_use_hook])]
      hooks["PostToolUseFailure"] = [hook_matcher_cls(hooks=[self._post_tool_use_failure_hook])]
    return hooks

  def _handle_stream_event(self, raw_event: Dict[str, Any]) -> None:
    event_type = str(raw_event.get("type") or "")
    if event_type == "content_block_start":
      block = _as_dict(raw_event.get("content_block"))
      if str(block.get("type") or "") == "tool_use":
        self._active_tool_use = _ActiveToolUse(
          tool_call_id=str(block.get("id") or ""),
          tool_name=str(block.get("name") or "tool"),
          raw_block=block,
        )
      return

    if event_type == "content_block_delta":
      delta = _as_dict(raw_event.get("delta"))
      delta_type = str(delta.get("type") or "")
      if delta_type == "text_delta":
        text = str(delta.get("text") or "")
        if text:
          self._append({"type": "text_delta", "text": text})
        return
      if delta_type == "thinking_delta":
        thinking_text = str(delta.get("thinking") or "")
        if thinking_text:
          self._append({"type": "thinking_delta", "text": thinking_text})
        return
      if delta_type == "input_json_delta" and self._active_tool_use is not None:
        partial_json = str(delta.get("partial_json") or "")
        if partial_json:
          self._active_tool_use.input_json += partial_json
        return
      return

    if event_type == "content_block_stop" and self._active_tool_use is not None:
      raw_block = _as_dict(self._active_tool_use.raw_block)
      tool_input: Dict[str, Any] = {}
      if self._active_tool_use.input_json:
        try:
          parsed = json.loads(self._active_tool_use.input_json)
          if isinstance(parsed, dict):
            tool_input = parsed
        except json.JSONDecodeError:
          tool_input = {}
      elif isinstance(raw_block.get("input"), dict):
        tool_input = dict(raw_block.get("input") or {})

      tool_call_id = self._active_tool_use.tool_call_id
      tool_name = self._active_tool_use.tool_name
      self._pending_tool_calls[tool_call_id] = ToolCallInfo(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_input=tool_input,
        started_at=time.time(),
      )
      self._append(
        {
          "type": "tool_call_start",
          "tool_call_id": tool_call_id,
          "tool_name": tool_name,
          "tool_input": tool_input,
        }
      )
      self._active_tool_use = None

  def _extract_tool_result_blocks(self, message: Any) -> List[Dict[str, Any]]:
    content = _get_attr(message, "content")
    parent_tool_use_id = _get_attr(message, "parent_tool_use_id")
    tool_use_result = _get_attr(message, "tool_use_result")

    blocks: List[Dict[str, Any]] = []
    if isinstance(content, list):
      for raw_block in content:
        block = _as_dict(raw_block)
        if str(block.get("type") or "") == "tool_result":
          blocks.append(block)

    if not blocks and parent_tool_use_id and tool_use_result is not None:
      blocks.append(
        {
          "type": "tool_result",
          "tool_use_id": str(parent_tool_use_id),
          "content": tool_use_result,
        }
      )
    return blocks

  def _normalize_tool_result(
    self,
    block: Dict[str, Any],
  ) -> Tuple[Any | None, Dict[str, Any] | None]:
    parsed = _parse_result_payload(block.get("content"))
    if bool(block.get("is_error")):
      if isinstance(parsed, dict):
        nested = parsed.get("error")
        if isinstance(nested, dict):
          return None, nested
        return None, {
          "code": str(parsed.get("code") or "tool_error"),
          "message": _summarize_error_payload(parsed),
        }
      return None, {"code": "tool_error", "message": _summarize_error_payload(parsed)}
    return parsed, None

  def _complete_tool_call(
    self,
    tool_call_id: str,
    *,
    result: Any | None = None,
    error: Dict[str, Any] | None = None,
    synthetic: bool = False,
  ) -> None:
    info = self._pending_tool_calls.pop(tool_call_id, None)
    if info is None:
      return

    server = _server_for_tool(info.tool_name)
    duration_ms = int((time.time() - info.started_at) * 1000)
    result_bytes = len(json.dumps(result, default=str)) if result is not None else 0
    event = {
      "type": "tool_call_complete",
      "tool_call_id": tool_call_id,
      "tool_name": info.tool_name,
      "result": result,
      "error": error,
      "duration_ms": duration_ms,
      "server": server,
    }
    if synthetic:
      log.info("[%s] Synthetic tool completion for %s (%s)", self._sid, info.tool_name, tool_call_id)
    self._append(event)
    self._call_on_tool_timing(
      tool_name=info.tool_name,
      server=server,
      duration_ms=duration_ms,
      is_error=error is not None,
      result_bytes=result_bytes,
    )

  def _flush_pending_tool_calls(self) -> None:
    for tool_call_id in list(self._pending_tool_calls.keys()):
      self._complete_tool_call(tool_call_id, synthetic=True)

  def _handle_user_message(self, message: Any) -> None:
    for block in self._extract_tool_result_blocks(message):
      tool_call_id = str(block.get("tool_use_id") or "")
      if not tool_call_id:
        continue
      result, error = self._normalize_tool_result(block)
      self._complete_tool_call(tool_call_id, result=result, error=error)

    # Message boundary flush: keep UI from showing long-lived pending tools when the
    # SDK consumes tool results internally and omits explicit tool_result blocks.
    self._flush_pending_tool_calls()

  def _handle_system_message(self, message: Any) -> None:
    subtype = str(_get_attr(message, "subtype") or "")
    data = _as_dict(_get_attr(message, "data"))
    if subtype == "init":
      statuses = data.get("mcp_servers")
      log.info("[%s] SDK init | mcp_servers=%s", self._sid, statuses if statuses is not None else data)
      return
    log.info("[%s] SDK system message | subtype=%s data=%s", self._sid, subtype or "unknown", data)

  def _handle_assistant_message(self, message: Any) -> None:
    error = _get_attr(message, "error")
    if error:
      log.warning("[%s] SDK assistant message error: %s", self._sid, error)

  def _emit_stream_complete(self) -> None:
    if self._stream_terminal_emitted:
      return

    cost_usd = float(self._usage.get("estimated_cost") or 0.0)
    usage = {
      "input_tokens": int(self._usage.get("input_tokens") or 0),
      "output_tokens": int(self._usage.get("output_tokens") or 0),
      "cache_creation_input_tokens": int(self._usage.get("cache_creation_input_tokens") or 0),
      "cache_read_input_tokens": int(self._usage.get("cache_read_input_tokens") or 0),
      "estimated_cost": round(cost_usd, 4),
    }
    self._append({"type": "stream_complete", "usage": usage})
    self._stream_terminal_emitted = True

  async def _close_query_iterator(self) -> None:
    iterator = self._query_iter
    self._query_iter = None
    if iterator is None:
      return
    close_fn = getattr(iterator, "aclose", None)
    if close_fn is not None:
      try:
        await close_fn()
      except Exception:
        pass
      return
    close_fn = getattr(iterator, "close", None)
    if close_fn is None:
      return
    try:
      maybe_awaitable = close_fn()
      if asyncio.iscoroutine(maybe_awaitable):
        await maybe_awaitable
    except Exception:
      pass

  def _update_usage(self, usage: Any, *, total_cost_usd: float | None = None, num_turns: int | None = None) -> None:
    usage_dict = _as_dict(usage)
    self._usage["input_tokens"] = int(usage_dict.get("input_tokens") or self._usage.get("input_tokens") or 0)
    self._usage["output_tokens"] = int(usage_dict.get("output_tokens") or self._usage.get("output_tokens") or 0)
    self._usage["cache_creation_input_tokens"] = int(
      usage_dict.get("cache_creation_input_tokens") or self._usage.get("cache_creation_input_tokens") or 0
    )
    self._usage["cache_read_input_tokens"] = int(
      usage_dict.get("cache_read_input_tokens") or self._usage.get("cache_read_input_tokens") or 0
    )
    if total_cost_usd is None:
      estimated = estimate_cost(
        self._effective_model,
        int(self._usage["input_tokens"]),
        int(self._usage["output_tokens"]),
        cache_read_tokens=int(self._usage["cache_read_input_tokens"]),
        cache_creation_tokens=int(self._usage["cache_creation_input_tokens"]),
      )
      total_cost_usd = estimated.total
    self._usage["estimated_cost"] = float(total_cost_usd or 0.0)
    if num_turns is not None:
      self._num_turns = int(num_turns)

  async def _emit_usage_hook(self) -> None:
    usage_payload = {
      "session_id": self._session_id,
      "turns": self._num_turns,
      "input_tokens": int(self._usage.get("input_tokens") or 0),
      "output_tokens": int(self._usage.get("output_tokens") or 0),
      "cache_read_input_tokens": int(self._usage.get("cache_read_input_tokens") or 0),
      "cache_creation_input_tokens": int(self._usage.get("cache_creation_input_tokens") or 0),
      "estimated_cost": round(float(self._usage.get("estimated_cost") or 0.0), 4),
    }
    await self._call_on_usage(usage_payload)

  async def run(
    self,
    messages: list[dict],
    system_prompt: str | None = None,
    model_override: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    _validate_sdk_version()
    try:
      import claude_agent_sdk
    except ImportError as exc:
      self._append({"type": "error", "error": "claude-agent-sdk dependency is required"})
      raise RuntimeError("claude-agent-sdk dependency is required") from exc

    prompt = self._build_prompt(messages)
    effective_system_prompt = _join_system_prompt(system_prompt or self._system_prompt)
    effective_model = str(model_override or self._sdk_config.model or self._effective_model).strip()
    if effective_model:
      self._effective_model = effective_model

    hooks = self._build_hooks(getattr(claude_agent_sdk, "HookMatcher"))
    options_kwargs: Dict[str, Any] = {
      "system_prompt": effective_system_prompt or None,
      "mcp_servers": dict(self._mcp_server_configs),
      "permission_mode": "bypassPermissions",
      "continue_conversation": False,
      "max_turns": max_turns if max_turns is not None else self._max_turns,
      "max_budget_usd": self._sdk_config.max_budget_usd,
      "disallowed_tools": list(self._disallowed_tools),
      "model": effective_model or None,
      "cwd": str(self._sdk_config.cwd) if self._sdk_config.cwd is not None else None,
      "include_partial_messages": True,
      "hooks": hooks or None,
    }
    options_kwargs = {key: value for key, value in options_kwargs.items() if value is not None}
    options = getattr(claude_agent_sdk, "ClaudeAgentOptions")(**options_kwargs)

    original_api_key = os.environ.get("ANTHROPIC_API_KEY")
    original_auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    os.environ["ANTHROPIC_API_KEY"] = self._sdk_config.api_key
    os.environ["ANTHROPIC_AUTH_TOKEN"] = ""

    try:
      query_iter = getattr(claude_agent_sdk, "query")(prompt=prompt, options=options)
      self._query_iter = query_iter
      async for message in query_iter:
        if hasattr(message, "event"):
          self._handle_stream_event(_as_dict(getattr(message, "event")))
          continue

        if hasattr(message, "duration_ms") and hasattr(message, "num_turns"):
          self._update_usage(
            _get_attr(message, "usage"),
            total_cost_usd=_get_attr(message, "total_cost_usd"),
            num_turns=_get_attr(message, "num_turns"),
          )
          await self._emit_usage_hook()
          self._flush_pending_tool_calls()
          self._emit_stream_complete()
          continue

        if hasattr(message, "subtype") and hasattr(message, "data"):
          self._handle_system_message(message)
          continue

        if hasattr(message, "model") and hasattr(message, "content"):
          self._handle_assistant_message(message)
          continue

        if hasattr(message, "content"):
          self._handle_user_message(message)

      self._flush_pending_tool_calls()
      self._emit_stream_complete()
    except asyncio.CancelledError:
      await self._close_query_iterator()
      self._flush_pending_tool_calls()
      self._emit_stream_complete()
    except Exception as exc:
      await self._close_query_iterator()
      self._flush_pending_tool_calls()
      self._append({"type": "error", "error": str(exc)})
      self._stream_terminal_emitted = True
      raise
    finally:
      await self._close_query_iterator()
      if original_api_key is None:
        os.environ.pop("ANTHROPIC_API_KEY", None)
      else:
        os.environ["ANTHROPIC_API_KEY"] = original_api_key
      if original_auth_token is None:
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
      else:
        os.environ["ANTHROPIC_AUTH_TOKEN"] = original_auth_token


__all__ = ["AgentSDKRunner"]
