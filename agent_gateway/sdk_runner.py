from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .approval_policy import (
  ApprovalDecision as PolicyApprovalDecision,
  ApprovalPolicy,
  ApprovalRequest as PolicyApprovalRequest,
  RunContext,
  apply_decision_to_request,
  build_approval_request,
  call_policy_safely,
  sha256_args,
  utc_now,
)
from . import approval_settings
from .event_log import EventLog
from .multi_user.billing import SessionUsageSummary, UsageEvent, _UsageAggregator, normalize_identity
from .product_config import gateway_product_id
from .providers.agent_sdk import AgentSDKConfig, estimate_cost, _validate_sdk_version
from .runner import (
  ToolResultContext,
  _ACTIVE_SKILL_DENY_RESULT_KEY,
  _ACTIVE_SKILL_REPORT_DOORS_RESULT_KEY,
  _REPORT_DOOR_CLEAR_SUCCESS_STATUSES,
  _detect_user_id_param,
)
from .session_recap import emit_recap_then_terminal
from .skill_context import clear_current_skill, current_skill
from .tool_dispatcher import RELAY_POLICY_DENIED_MESSAGE, RELAY_POLICY_DENIED_SUB_CODE
from .tool_display import resolve_display
from .tool_result_semantics import classify_semantic_tool_error


log = logging.getLogger("agent_gateway.sdk_runner")

OnToolResult = Callable[[ToolResultContext], Awaitable[List[Dict[str, Any]] | None]]
OnUsage = Callable[[UsageEvent], Awaitable[None] | None]
OnSessionSummary = Callable[[SessionUsageSummary], Awaitable[None] | None]
OnToolTiming = Callable[..., None]


def _approval_queue_timeout_seconds(expiry_seconds: float | int | None) -> float:
  return min(float(expiry_seconds or 600), approval_settings.approval_wait_seconds())


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


class _PromptMessages:
  def __init__(self, text: str) -> None:
    self.text = text

  def __contains__(self, needle: object) -> bool:
    return isinstance(needle, str) and needle in self.text

  def __str__(self) -> str:
    return self.text

  async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
    yield {
      "type": "user",
      "message": {
        "role": "user",
        "content": [{"type": "text", "text": self.text}],
      },
    }


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


def _redact_tool_input_for_event(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
  try:
    from agent.shared.tool_redaction import get_audit_hmac_secret, redact_tool_input

    return redact_tool_input(tool_name, tool_input, deployment_secret=get_audit_hmac_secret())
  except Exception:
    return dict(tool_input)


def _policy_tool_name(tool_name: str) -> str:
  if tool_name.startswith("mcp__"):
    parts = tool_name.split("__", 2)
    if len(parts) == 3:
      return parts[2]
  return tool_name


_PATCH_OP_RAW_INPUT_TOOLS = frozenset({
  "apply_patch_ops",
  "preview_patch_ops",
  "publish_price_target",
})


def _should_escrow_raw_tool_input(tool_name: str) -> bool:
  return _policy_tool_name(tool_name) in _PATCH_OP_RAW_INPUT_TOOLS


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
    on_session_summary: Callable[..., Any] | None = None,
    on_late_usage_event: Callable[..., Any] | None = None,
    on_tool_result: Callable[..., Any] | None = None,
    on_tool_timing: Callable[..., Any] | None = None,
    _parent_aggregator: _UsageAggregator | None = None,
    session: Any | None = None,
    store: Any | None = None,
    policy: ApprovalPolicy | None = None,
    run_context: RunContext | None = None,
    skill_run_id: str | None = None,
    workspace_dir: str | None = None,
    started_at: float | None = None,
    emit_session_recap: bool = True,
  ) -> None:
    self._log = event_log
    self._session_id = session_id or "no-session"
    self._sid = self._session_id[:12]
    self._session_started_at = float(started_at if started_at is not None else time.time())
    self._emit_session_recap = bool(emit_session_recap)
    self._sdk_config = sdk_config
    self._system_prompt = system_prompt
    self._disallowed_tools = list(disallowed_tools or sdk_config.disallowed_tools)
    self._mcp_server_configs = dict(mcp_server_configs or {})
    self._max_turns = max_turns
    self._on_usage = on_usage
    self._on_session_summary = on_session_summary
    self._on_late_usage_event = on_late_usage_event
    self._on_tool_result = on_tool_result
    self._on_tool_timing = on_tool_timing
    self._on_tool_timing_accepts_user_id = _detect_user_id_param(on_tool_timing)
    self._pending_tool_calls: Dict[str, ToolCallInfo] = {}
    self._active_tool_use: _ActiveToolUse | None = None
    self._active_skill_deny: set[str] = set()
    self._active_skill_report_doors: dict[str, str] = {}
    self._query_iter: Any = None
    self._usage: Dict[str, Any] = {
      "input_tokens": 0,
      "output_tokens": 0,
      "cache_creation_input_tokens": 0,
      "cache_read_input_tokens": 0,
    }
    self._num_turns = 0
    self._stream_terminal_emitted = False
    self._effective_model = sdk_config.model or os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7").strip()
    self._request_id = str(sdk_config.request_id or uuid.uuid4())
    (
      self._usage_user_id,
      self._rate_table_version,
      self._billing_mode,
      self._channel,
    ) = normalize_identity(
      sdk_config.user_id,
      sdk_config.rate_table_version,
      sdk_config.billing_mode,
      sdk_config.channel,
    )
    self._parent_aggregator = _parent_aggregator
    self._aggregator = _parent_aggregator or _UsageAggregator(
      user_id=self._usage_user_id,
      session_id=self._session_id,
      request_id=self._request_id,
      channel=self._channel,
    )
    self._summary_emitted = False
    self._session = session
    self._approval_store = store or getattr(session, "approval_store", None)
    self._approval_policy = policy or getattr(session, "approval_policy", None)
    self._run_context = run_context
    self._skill_run_id = skill_run_id
    self._workspace_dir = workspace_dir

  def _append(self, event: Dict[str, Any]) -> None:
    payload = dict(event)
    pid = gateway_product_id()
    if pid is not None:
      payload["product_id"] = pid
    if payload.get("type") in {"stream_complete", "error"}:
      emit_recap_then_terminal(
        self._log,
        payload,
        session_id=self._session_id,
        started_at=self._session_started_at,
        emit_recap=self._emit_session_recap,
      )
      return
    self._log.append(payload)

  async def _call_on_usage(self, usage_event: UsageEvent) -> None:
    recorded = await self._aggregator.record(usage_event)
    await self._aggregator.set_turns(self._num_turns)
    if not recorded or self._summary_emitted:
      log.warning("[%s] Usage event arrived after session summary emission: %s", self._sid, usage_event.event_id)
      await self._call_on_late_usage_event(usage_event)
      return
    if self._on_usage is None:
      return
    try:
      result = self._on_usage(usage_event)
      if inspect.isawaitable(result):
        await result
    except Exception as exc:
      log.warning("[%s] on_usage hook failed (non-fatal): %s", self._sid, exc)

  async def _call_on_late_usage_event(self, usage_event: UsageEvent) -> None:
    if self._on_late_usage_event is None:
      return
    try:
      result = self._on_late_usage_event(usage_event)
      if inspect.isawaitable(result):
        await result
    except Exception as exc:
      log.warning("[%s] on_late_usage_event hook failed (non-fatal): %s", self._sid, exc)

  async def _call_on_session_summary(self, summary: SessionUsageSummary) -> None:
    if self._on_session_summary is None:
      return
    try:
      result = self._on_session_summary(summary)
      if inspect.isawaitable(result):
        await result
    except Exception as exc:
      log.warning("[%s] on_session_summary hook failed (non-fatal): %s", self._sid, exc)

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
      if self._on_tool_timing_accepts_user_id:
        self._on_tool_timing(
          self._session_id,
          tool_name,
          server,
          duration_ms,
          is_error,
          result_bytes,
          user_id=self._usage_user_id,
        )
      else:
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
    if classify_semantic_tool_error(result) is not None:
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
      if block.get("_event_only"):
        continue
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
        skill_run_id=self._skill_run_id,
        workspace_dir=self._workspace_dir,
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

  def _effective_disallowed_tools(self) -> set[str]:
    if not self._active_skill_deny:
      return set(self._disallowed_tools)
    return set(self._disallowed_tools) | set(self._active_skill_deny)

  def _activate_skill_deny(self, tool_names: Any) -> None:
    if isinstance(tool_names, str):
      candidates = [tool_names]
    elif isinstance(tool_names, (list, tuple, set, frozenset)):
      candidates = list(tool_names)
    else:
      return
    denied = {normalized for name in candidates if (normalized := str(name or "").strip())}
    self._active_skill_deny = denied

  def _activate_skill_report_doors(self, value: Any) -> None:
    if isinstance(value, dict):
      self._active_skill_report_doors = {
        str(tool_name).strip(): str(skill_name).strip()
        for tool_name, skill_name in value.items()
        if str(tool_name).strip() and str(skill_name).strip()
      }
      return
    if value is not None:
      self._active_skill_report_doors = {}

  def _clear_active_skill_if_report_door_completed(
    self,
    *,
    tool_name: str,
    result: Any,
    error: Dict[str, Any] | None,
  ) -> bool:
    if error is not None:
      return False
    normalized_tool_name = str(tool_name or "").strip()
    expected_skill = self._active_skill_report_doors.get(normalized_tool_name)
    if not expected_skill:
      return False
    if not (
      isinstance(result, dict)
      and "subcommand" in result
      and str(result.get("mutation_mode") or "").strip() == "preview"
    ):
      return False
    if str(result.get("status") or "").strip().lower() not in _REPORT_DOOR_CLEAR_SUCCESS_STATUSES:
      return False
    if current_skill() != expected_skill:
      return False
    clear_current_skill()
    self._active_skill_deny.clear()
    self._active_skill_report_doors.clear()
    return True

  def _consume_private_tool_result_fields(self, result: Any, *, tool_name: str | None = None) -> Any:
    if isinstance(result, dict):
      self._activate_skill_report_doors(result.pop(_ACTIVE_SKILL_REPORT_DOORS_RESULT_KEY, None))
      self._activate_skill_deny(result.pop(_ACTIVE_SKILL_DENY_RESULT_KEY, None))
      if tool_name is not None:
        self._clear_active_skill_if_report_door_completed(
          tool_name=tool_name,
          result=result,
          error=None,
        )
    return result

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
    result = self._consume_private_tool_result_fields(result, tool_name=tool_name)
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
      if _should_escrow_raw_tool_input(tool_name):
        record_raw_tool_input = getattr(self._log, "record_raw_tool_input", None)
        if callable(record_raw_tool_input):
          record_raw_tool_input(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_input=tool_input,
          )
      redacted_tool_input = _redact_tool_input_for_event(tool_name, tool_input)
      display = resolve_display(tool_name, redacted_tool_input)
      tool_start_event = {
        "type": "tool_call_start",
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "tool_input": redacted_tool_input,
      }
      if display is not None:
        tool_start_event["display"] = display
      self._append(tool_start_event)
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
    return self._consume_private_tool_result_fields(parsed), None

  def _complete_tool_call(
    self,
    tool_call_id: str,
    *,
    result: Any | None = None,
    error: Dict[str, Any] | None = None,
    synthetic: bool = False,
    outcome: str | None = None,
  ) -> None:
    info = self._pending_tool_calls.pop(tool_call_id, None)
    if info is None:
      return

    server = _server_for_tool(info.tool_name)
    duration_ms = int((time.time() - info.started_at) * 1000)
    result_bytes = len(json.dumps(result, default=str)) if result is not None else 0
    semantic_error = classify_semantic_tool_error(result) if error is None else None
    event = {
      "type": "tool_call_complete",
      "tool_call_id": tool_call_id,
      "tool_name": info.tool_name,
      "result": result,
      "error": error,
      "duration_ms": duration_ms,
      "server": server,
      "is_error": error is not None or semantic_error is not None,
    }
    if semantic_error is not None:
      event["semantic_error"] = dict(semantic_error)
    self._clear_active_skill_if_report_door_completed(
      tool_name=info.tool_name,
      result=result,
      error=error,
    )
    if synthetic:
      log.info("[%s] Synthetic tool completion for %s (%s)", self._sid, info.tool_name, tool_call_id)
    self._append(event)
    self._call_on_tool_timing(
      tool_name=info.tool_name,
      server=server,
      duration_ms=duration_ms,
      is_error=event["is_error"],
      result_bytes=result_bytes,
    )

  def _flush_pending_tool_calls(self, *, outcome: str | None = None) -> None:
    for tool_call_id in list(self._pending_tool_calls.keys()):
      self._complete_tool_call(tool_call_id, synthetic=True, outcome=outcome)

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

  def _approval_lifecycle_configured(self) -> bool:
    return self._approval_store is not None and self._approval_policy is not None and self._session is not None

  def _resolve_run_context(self) -> RunContext:
    if self._run_context is not None:
      return self._run_context
    return RunContext(
      user_id=str(self._usage_user_id or getattr(self._session, "user_id", "") or "unknown"),
      request_id=self._request_id,
      session_id=self._session_id,
      profile="chat",
      channel=str(self._channel or getattr(self._session, "channel", None) or "web"),
      decider_role=str(getattr(self._session, "role", "owner") or "owner"),
      policy_bundle_hash=str(getattr(self._approval_policy, "policy_bundle_hash", "unknown")),
      model_id=self._effective_model,
    )

  def _resolve_tool_class(self, tool_name: str) -> str:
    policy_tool = _policy_tool_name(tool_name)
    try:
      from agent.shared.server_policies import get_tool_class

      server = _server_for_tool(tool_name)
      if server:
        cls = get_tool_class(server, policy_tool)
        if cls is not None:
          return cls
    except Exception:
      pass
    return "state_write"

  def _redact_for_approval_request(self, tool_name: str, tool_input: Dict[str, Any]) -> tuple[dict[str, Any], str]:
    try:
      from agent.shared.tool_redaction import get_audit_hmac_secret, get_audit_hmac_key_id, hmac_value, redact_tool_input

      secret = get_audit_hmac_secret()
      key_id = get_audit_hmac_key_id()
      return (
        redact_tool_input(tool_name, tool_input, deployment_secret=secret, key_id=key_id),
        hmac_value(tool_input, deployment_secret=secret, key_id=key_id),
      )
    except Exception:
      return {}, sha256_args(tool_input)

  async def _await_user_approval_via_pending_tools(
    self,
    request: PolicyApprovalRequest,
    decision: PolicyApprovalDecision,
    *,
    nonce: str,
  ) -> dict[str, Any] | None:
    session = self._session
    if session is None:
      return None
    approval_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    session.pending_tools[request.tool_call_id] = {
      "approval_id": request.approval_id,
      "nonce": nonce,
      "requested_at": int(time.time()),
      "status": "approval_pending",
      "tool_name": request.tool_name,
      "resolved_qualifier": "",
    }
    session.approval_queues[request.tool_call_id] = approval_queue
    approval_event = {
      "type": "tool_approval_request",
      "tool_call_id": request.tool_call_id,
      "approval_id": request.approval_id,
      "nonce": nonce,
      "tool_name": request.tool_name,
      "tool_input": request.tool_args_redacted,
      "resolved_qualifier": "",
      "reason": decision.reason,
      "allow_persistent_approval": decision.allow_persistent_grant,
      "ts": time.time(),
    }
    self._append(approval_event)
    session_log = getattr(session, "agent_session_log", None)
    if session_log is not None:
      try:
        await session_log.append(approval_event)
      except Exception:
        log.warning("Failed to persist approval request event for %s", request.tool_call_id, exc_info=True)
    try:
      return await asyncio.wait_for(
        approval_queue.get(),
        timeout=max(0.1, _approval_queue_timeout_seconds(decision.expiry_seconds)),
      )
    except asyncio.TimeoutError:
      store = self._approval_store
      if store is not None:
        try:
          latest = await store.get(request.approval_id)
          if latest is not None and latest.state == "pending_user":
            await store.transition_state(
              latest.approval_id,
              "expired",
              expected_state_version=latest.state_version,
              decision_reason="Timed out waiting for user approval",
            )
        except Exception:
          log.warning("Failed to expire timed-out approval request %s", request.approval_id, exc_info=True)
      return None
    finally:
      session.pending_tools.pop(request.tool_call_id, None)
      session.approval_queues.pop(request.tool_call_id, None)

  async def _can_use_tool_callback(self, tool_name: str, input_data: dict[str, Any], _context: Any) -> Any:
    import claude_agent_sdk

    allow_cls = getattr(claude_agent_sdk, "PermissionResultAllow")
    deny_cls = getattr(claude_agent_sdk, "PermissionResultDeny")
    if tool_name in self._effective_disallowed_tools():
      return deny_cls(message=f"Tool '{tool_name}' is not available in this context")
    if not self._approval_lifecycle_configured():
      return allow_cls()
    store = self._approval_store
    policy = self._approval_policy
    assert store is not None and policy is not None
    run_context = self._resolve_run_context()
    active_skill = current_skill()
    if active_skill and run_context.skill is None:
      run_context = replace(run_context, skill=active_skill)
    policy_tool = _policy_tool_name(tool_name)
    redacted, args_hash = self._redact_for_approval_request(tool_name, input_data)
    request = build_approval_request(
      tool_call_id=f"sdk-{uuid.uuid4().hex}",
      tool_name=policy_tool,
      tool_class=self._resolve_tool_class(tool_name),
      tool_args_redacted=redacted,
      args_hash=args_hash,
      run_context=run_context,
    )
    await store.create(request)
    raw_args = dict(input_data)
    try:
      decision = await call_policy_safely(policy, request, raw_args, run_context)
    finally:
      raw_args.clear()
      del raw_args
    request = apply_decision_to_request(request, decision)
    await store.update_request(request)

    if decision.outcome == "auto_approve":
      request = await store.transition_state(request.approval_id, "auto_approved", expected_state_version=request.state_version)
      await policy.on_resolve(request=request)
      if decision.modified_tool_args is not None:
        return allow_cls(updated_input=decision.modified_tool_args)
      return allow_cls()
    if decision.outcome == "auto_deny":
      request = await store.transition_state(request.approval_id, "auto_denied", expected_state_version=request.state_version)
      await policy.on_resolve(request=request)
      return deny_cls(message=decision.reason)
    if decision.outcome == "route_external":
      await store.transition_state(
        request.approval_id,
        "routed_external",
        route_target=decision.route_target,
        route_target_type=decision.route_target_type,
        expires_at=utc_now() + timedelta(seconds=decision.expiry_seconds or 600),
        expected_state_version=request.state_version,
      )
      return deny_cls(message="external approval route is pending")

    request = await store.transition_state(
      request.approval_id,
      "pending_user",
      route_target_type="pending_tools",
      expires_at=utc_now() + timedelta(seconds=decision.expiry_seconds or 600),
      expected_state_version=request.state_version,
    )
    approval = await self._await_user_approval_via_pending_tools(request, decision, nonce=os.urandom(8).hex())
    if approval and approval.get("approved"):
      if decision.modified_tool_args is not None:
        return allow_cls(updated_input=decision.modified_tool_args)
      return allow_cls()
    if approval and approval.get("denied_by") == "relay_policy":
      return deny_cls(message=f"[{RELAY_POLICY_DENIED_SUB_CODE}] {RELAY_POLICY_DENIED_MESSAGE}")
    return deny_cls(message="user denied")

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

  async def on_disconnect(self) -> None:
    try:
      await self._close_query_iterator()
    except Exception as exc:
      log.warning("[%s] query iterator close on disconnect failed (non-fatal): %s", self._sid, exc)

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
    usage_event = UsageEvent(
      user_id=self._usage_user_id,
      session_id=self._session_id,
      request_id=self._request_id,
      parent_turn_id=None,
      timestamp=time.time(),
      model=self._effective_model,
      provider="agent-sdk",
      input_tokens=int(self._usage.get("input_tokens") or 0),
      output_tokens=int(self._usage.get("output_tokens") or 0),
      cache_read_tokens=int(self._usage.get("cache_read_input_tokens") or 0),
      cache_creation_tokens=int(self._usage.get("cache_creation_input_tokens") or 0),
      cost_usd=float(self._usage.get("estimated_cost") or 0.0),
      rate_table_version=self._rate_table_version,
      billing_mode=self._billing_mode,
      channel=self._channel,
    )
    await self._call_on_usage(usage_event)

  async def run(
    self,
    messages: list[dict],
    system_prompt: str | None = None,
    model_override: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    if self._summary_emitted:
      raise RuntimeError("AgentSDKRunner is single-use; construct a new runner for subsequent runs")
    _validate_sdk_version()
    try:
      import claude_agent_sdk
    except ImportError as exc:
      self._append({"type": "error", "error": "claude-agent-sdk dependency is required"})
      raise RuntimeError("claude-agent-sdk dependency is required") from exc

    prompt_text = self._build_prompt(messages)
    prompt = _PromptMessages(prompt_text)
    effective_system_prompt = _join_system_prompt(system_prompt or self._system_prompt)
    effective_model = str(model_override or self._sdk_config.model or self._effective_model).strip()
    if effective_model:
      self._effective_model = effective_model

    hooks = self._build_hooks(getattr(claude_agent_sdk, "HookMatcher"))
    options_kwargs: Dict[str, Any] = {
      "system_prompt": effective_system_prompt or None,
      "mcp_servers": dict(self._mcp_server_configs),
      "continue_conversation": False,
      "max_turns": max_turns if max_turns is not None else self._max_turns,
      "max_budget_usd": self._sdk_config.max_budget_usd,
      "disallowed_tools": list(self._disallowed_tools),
      "model": effective_model or None,
      "cwd": str(self._sdk_config.cwd) if self._sdk_config.cwd is not None else None,
      "include_partial_messages": True,
      "hooks": hooks or None,
      "can_use_tool": self._can_use_tool_callback,
    }
    options_kwargs = {key: value for key, value in options_kwargs.items() if value is not None}
    options = getattr(claude_agent_sdk, "ClaudeAgentOptions")(**options_kwargs)

    original_api_key = os.environ.get("ANTHROPIC_API_KEY")
    original_auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    auth_mode = str(self._sdk_config.auth_mode or "api").strip().lower()
    if auth_mode == "oauth":
      os.environ["ANTHROPIC_API_KEY"] = ""
      os.environ["ANTHROPIC_AUTH_TOKEN"] = str(self._sdk_config.auth_token or "")
    else:
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
          self._flush_pending_tool_calls(outcome="success")
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

      self._flush_pending_tool_calls(outcome="success")
      self._emit_stream_complete()
    except asyncio.CancelledError:
      await self._close_query_iterator()
      self._flush_pending_tool_calls(outcome="cancelled")
      self._emit_stream_complete()
    except Exception as exc:
      await self._close_query_iterator()
      self._flush_pending_tool_calls(outcome="tool_error")
      self._append({"type": "error", "error": str(exc)})
      self._stream_terminal_emitted = True
      raise
    finally:
      clear_current_skill()
      self._active_skill_deny.clear()
      self._active_skill_report_doors.clear()
      if self._parent_aggregator is None:
        try:
          await self._aggregator.close()
          summary = await self._aggregator.snapshot(
            ended_at=time.time(),
            drain_complete=True,
            in_flight_task_count=0,
          )
          self._summary_emitted = True
          await self._call_on_session_summary(summary)
        except Exception as exc:
          log.warning("[%s] session summary emission failed (non-fatal): %s", self._sid, exc)
      else:
        self._summary_emitted = True
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
