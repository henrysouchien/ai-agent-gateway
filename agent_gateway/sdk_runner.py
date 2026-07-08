from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import replace
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Sequence

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
from .approval_enrichment import effective_trade_approval_expiry_seconds, enrich_trade_approval_args
from .event_log import EventLog
from .multi_user.billing import SessionUsageSummary, UsageEvent, _UsageAggregator, normalize_identity
from .policy_imports import resolve_server_policy_tool_class
from .product_config import gateway_product_id
from .providers.agent_sdk import AgentSDKConfig, estimate_cost, _validate_sdk_version
from .runner import (
  ToolResultContext,
  _ACTIVE_SKILL_DENY_RESULT_KEY,
  _ACTIVE_SKILL_REPORT_DOORS_RESULT_KEY,
  _REPORT_DOOR_CLEAR_SUCCESS_STATUSES,
  _detect_user_id_param,
)
from .runner_introspection import detect_keyword_param as _detect_keyword_param
from .runner_skill_gate import is_report_door_clear_event as _is_report_door_clear_event
from .session_recap import emit_recap_then_terminal
from .skill_context import clear_current_skill, current_skill
from . import sdk_runner_approval as _sdk_runner_approval
from . import sdk_runner_context as _sdk_runner_context
from . import sdk_runner_helpers as _sdk_runner_helpers
from .sdk_runner_stream import ToolCallInfo, _ActiveToolUse, _SDKRunnerStreamMixin
from .tool_dispatcher import RELAY_POLICY_DENIED_MESSAGE, RELAY_POLICY_DENIED_SUB_CODE
from .tool_display import resolve_display  # noqa: F401 - compatibility alias for stream helpers
from .tool_result_semantics import classify_semantic_tool_error  # noqa: F401 - compatibility alias for helpers


log = logging.getLogger("agent_gateway.sdk_runner")

OnToolResult = Callable[[ToolResultContext], Awaitable[List[Dict[str, Any]] | None]]
OnUsage = Callable[[UsageEvent], Awaitable[None] | None]
OnSessionSummary = Callable[[SessionUsageSummary], Awaitable[None] | None]
OnToolTiming = Callable[..., None]


_PATCH_OP_RAW_INPUT_TOOLS = _sdk_runner_helpers.PATCH_OP_RAW_INPUT_TOOLS
_as_dict = _sdk_runner_helpers.as_dict
_as_plain_dict = _sdk_runner_helpers.as_plain_dict
_extract_text = _sdk_runner_helpers.extract_text
_get_attr = _sdk_runner_helpers.get_attr
_join_system_prompt = _sdk_runner_helpers.join_system_prompt
_parse_result_payload = _sdk_runner_helpers.parse_result_payload
_policy_owner_mismatch = _sdk_runner_helpers.policy_owner_mismatch
_policy_tool_name = _sdk_runner_helpers.policy_tool_name
_redact_tool_input_for_event = _sdk_runner_helpers.redact_tool_input_for_event
_server_for_tool = _sdk_runner_helpers.server_for_tool
_should_escrow_raw_tool_input = _sdk_runner_helpers.should_escrow_raw_tool_input
_summarize_error_payload = _sdk_runner_helpers.summarize_error_payload
ToolCallInfo.__module__ = __name__
_ActiveToolUse.__module__ = __name__


def _approval_queue_timeout_seconds(expiry_seconds: float | int | None) -> float:
  return _sdk_runner_approval.approval_queue_timeout_seconds(
    expiry_seconds,
    approval_wait_seconds_fn=approval_settings.approval_wait_seconds,
  )


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


class AgentSDKRunner(_SDKRunnerStreamMixin):
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
    allowed_tools: list[str] | None = None,
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
    batch_id: int | str | None = None,
    started_at: float | None = None,
    emit_session_recap: bool = True,
    context_surfaces: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]] | None = None,
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
    self._allowed_tools = list(allowed_tools or [])
    self._max_turns = max_turns
    self._on_usage = on_usage
    self._on_session_summary = on_session_summary
    self._on_late_usage_event = on_late_usage_event
    self._on_tool_result = on_tool_result
    self._on_tool_timing = on_tool_timing
    self._on_tool_timing_accepts_user_id = _detect_user_id_param(on_tool_timing)
    self._on_tool_timing_accepts_context_surfaces = _detect_keyword_param(on_tool_timing, "context_surfaces")
    self._on_tool_timing_accepts_tool_call_id = _detect_keyword_param(on_tool_timing, "tool_call_id")
    self._on_tool_timing_accepts_request_id = _detect_keyword_param(on_tool_timing, "request_id")
    self._context_surfaces_provider = context_surfaces if callable(context_surfaces) else None
    self._context_surfaces_static = self._normalize_context_surfaces(None if callable(context_surfaces) else context_surfaces)
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
      rate_table_version=self._rate_table_version,
      billing_mode=self._billing_mode,
    )
    self._summary_emitted = False
    self._session = session
    self._approval_store = store or getattr(session, "approval_store", None)
    self._approval_policy = policy or getattr(session, "approval_policy", None)
    self._run_context = run_context
    self._skill_run_id = skill_run_id
    self._workspace_dir = workspace_dir
    self._batch_id = str(batch_id).strip() if batch_id is not None and str(batch_id).strip() else None

  @staticmethod
  def _normalize_context_surfaces(surfaces: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return _sdk_runner_context.normalize_context_surfaces(surfaces)

  def _context_surface_records(self) -> list[dict[str, Any]]:
    return _sdk_runner_context.context_surface_records(self, logger=log)

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
    await _sdk_runner_context.call_on_usage(self, usage_event, logger=log)

  async def _call_on_late_usage_event(self, usage_event: UsageEvent) -> None:
    await _sdk_runner_context.call_on_late_usage_event(self, usage_event, logger=log)

  async def _call_on_session_summary(self, summary: SessionUsageSummary) -> None:
    await _sdk_runner_context.call_on_session_summary(self, summary, logger=log)

  def _call_on_tool_timing(
    self,
    *,
    tool_name: str,
    server: str | None,
    duration_ms: int,
    is_error: bool,
    result_bytes: int,
    tool_call_id: str | None = None,
    request_id: str | None = None,
  ) -> None:
    _sdk_runner_context.call_on_tool_timing(
      self,
      tool_name=tool_name,
      server=server,
      duration_ms=duration_ms,
      is_error=is_error,
      result_bytes=result_bytes,
      tool_call_id=tool_call_id,
      request_id=request_id,
      logger=log,
    )

  async def _call_on_tool_result(self, ctx: ToolResultContext) -> List[Dict[str, Any]]:
    return await _sdk_runner_context.call_on_tool_result(self, ctx, logger=log)

  def _build_prompt(self, messages: List[Dict[str, Any]]) -> str:
    return _sdk_runner_context.build_prompt(messages)

  def _make_result_entry(
    self,
    tool_call_id: str,
    result: Any | None,
    error: Dict[str, Any] | None,
  ) -> Dict[str, Any]:
    return _sdk_runner_context.make_result_entry(tool_call_id, result, error)

  def _format_additional_context(
    self,
    *,
    tool_name: str,
    result_entry: Dict[str, Any],
    extra_blocks: Sequence[Dict[str, Any]],
  ) -> str | None:
    return _sdk_runner_context.format_additional_context(
      tool_name=tool_name,
      result_entry=result_entry,
      extra_blocks=extra_blocks,
    )

  async def _build_hook_additional_context(
    self,
    *,
    tool_call_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
    result: Any | None,
    error: Dict[str, Any] | None,
  ) -> str | None:
    return await _sdk_runner_context.build_hook_additional_context(
      self,
      tool_call_id=tool_call_id,
      tool_name=tool_name,
      tool_input=tool_input,
      result=result,
      error=error,
      logger=log,
    )

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
    if not _is_report_door_clear_event(
      {
        "type": "tool_call_complete",
        "tool_name": normalized_tool_name,
        "result": result,
        "error": None,
      },
      expected_skill=expected_skill,
      success_statuses=_REPORT_DOOR_CLEAR_SUCCESS_STATUSES,
    ):
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

  def _approval_lifecycle_configured(self) -> bool:
    return _sdk_runner_approval.approval_lifecycle_configured(
      store=self._approval_store,
      policy=self._approval_policy,
      session=self._session,
    )

  def _resolve_run_context(self) -> RunContext:
    return _sdk_runner_approval.resolve_run_context(
      run_context=self._run_context,
      usage_user_id=self._usage_user_id,
      session=self._session,
      approval_policy=self._approval_policy,
      request_id=self._request_id,
      session_id=self._session_id,
      channel=self._channel,
      effective_model=self._effective_model,
    )

  def _resolve_tool_class(self, tool_name: str) -> str:
    return _sdk_runner_approval.resolve_tool_class(
      tool_name,
      policy_tool_name_fn=_policy_tool_name,
      server_for_tool_fn=_server_for_tool,
      resolve_server_policy_tool_class_fn=resolve_server_policy_tool_class,
    )

  def _redact_for_approval_request(self, tool_name: str, tool_input: Dict[str, Any]) -> tuple[dict[str, Any], str]:
    return _sdk_runner_approval.redact_for_approval_request(
      tool_name,
      tool_input,
      sha256_args_fn=sha256_args,
    )

  async def _await_user_approval_via_pending_tools(
    self,
    request: PolicyApprovalRequest,
    decision: PolicyApprovalDecision,
    *,
    nonce: str,
  ) -> dict[str, Any] | None:
    return await _sdk_runner_approval.await_user_approval_via_pending_tools(
      session=self._session,
      approval_store=self._approval_store,
      request=request,
      decision=decision,
      nonce=nonce,
      append_event_fn=self._append,
      timeout_seconds=_approval_queue_timeout_seconds(decision.expiry_seconds),
      log=log,
      time_fn=time.time,
    )

  async def _can_use_tool_callback(self, tool_name: str, input_data: dict[str, Any], _context: Any) -> Any:
    return await _sdk_runner_approval.can_use_tool_callback(
      self,
      tool_name,
      input_data,
      _context,
      policy_owner_mismatch_fn=_policy_owner_mismatch,
      policy_tool_name_fn=_policy_tool_name,
      current_skill_fn=current_skill,
      replace_fn=replace,
      enrich_trade_approval_args_fn=enrich_trade_approval_args,
      build_approval_request_fn=build_approval_request,
      call_policy_safely_fn=call_policy_safely,
      apply_decision_to_request_fn=apply_decision_to_request,
      effective_trade_approval_expiry_seconds_fn=effective_trade_approval_expiry_seconds,
      approval_wait_seconds_fn=approval_settings.approval_wait_seconds,
      utc_now_fn=utc_now,
      uuid_hex_fn=lambda: uuid.uuid4().hex,
      os_urandom_fn=os.urandom,
      relay_policy_denied_sub_code=RELAY_POLICY_DENIED_SUB_CODE,
      relay_policy_denied_message=RELAY_POLICY_DENIED_MESSAGE,
    )

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
      "allowed_tools": list(self._allowed_tools),
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
            context_surfaces=self._context_surface_records(),
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
