from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import logging
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, Mapping

from . import sdk_runner_helpers as _sdk_runner_helpers
from model_authority.binding import validate_reported_identity
from .event_adapter import TERMINAL_DISPOSITIONS, TerminalDisposition
from .runner_session_events import (
  build_tool_call_complete_event as _build_tool_call_complete_event,
)
from .tool_display import resolve_display
from .tool_dispatch_classification import (
  build_dispatch_record_from_sources as _build_dispatch_record_from_sources,
  resolve_dispatch_entry as _resolve_dispatch_entry,
)
from .tool_result_semantics import classify_semantic_tool_error
from .providers.anthropic import _server_tool_unit_deltas
from .runner_tool_audit import get_tool_risk_value
from .secret_boundary import SANITIZATION_FAILED, sanitize_boundary_value


if TYPE_CHECKING:
  from .capability_execution import BoundCapabilityExecution
  from .event_log import EventLog
  from .tool_dispatch_classification import DispatchEntry, ToolResultSettlement


log = logging.getLogger("agent_gateway.sdk_runner")
_PARENT_MODULE = "agent_gateway.sdk_runner"


@dataclass
class ToolCallInfo:
  tool_call_id: str
  tool_name: str
  tool_input: Dict[str, Any]
  started_at: float
  redacted_tool_input: Dict[str, Any]


@dataclass
class _ActiveToolUse:
  tool_call_id: str
  tool_name: str
  input_json: str = ""
  raw_block: Any = None


def _parent_attr(name: str, fallback: Any) -> Any:
  parent = sys.modules.get(_PARENT_MODULE)
  if parent is not None and hasattr(parent, name):
    return getattr(parent, name)
  return fallback


def _as_dict(value: Any) -> Dict[str, Any]:
  return _parent_attr("_as_dict", _sdk_runner_helpers.as_dict)(value)


def _get_attr(value: Any, key: str, default: Any = None) -> Any:
  return _parent_attr("_get_attr", _sdk_runner_helpers.get_attr)(value, key, default)


def _server_for_tool(tool_name: str) -> str | None:
  return _parent_attr("_server_for_tool", _sdk_runner_helpers.server_for_tool)(tool_name)


def _should_escrow_raw_tool_input(tool_name: str) -> bool:
  return _parent_attr(
    "_should_escrow_raw_tool_input",
    _sdk_runner_helpers.should_escrow_raw_tool_input,
  )(tool_name)


def _time() -> float:
  return _parent_attr("time", time).time()


def _is_accepted_ui_blocks_result(tool_name: str, result: Any, error: Any) -> bool:
  if tool_name != "emit_ui_blocks" or error is not None or not isinstance(result, dict):
    return False
  accepted = result.get("accepted")
  return isinstance(accepted, dict) and isinstance(accepted.get("ui_blocks_id"), str)


class _SDKRunnerStreamMixin:
  if TYPE_CHECKING:
    _capability_execution: BoundCapabilityExecution
    _log: EventLog
    _pending_sdk_usage_deltas: list[dict[str, Any]]
    _pending_tool_calls: Dict[str, ToolCallInfo]
    _request_id: str
    _sid: str
    _usage: Dict[str, Any]

    def _append(self, event: Dict[str, Any]) -> None: ...

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
    ) -> None: ...

    def _clear_active_skill_if_report_door_completed(
      self,
      *,
      tool_name: str,
      result: Any,
      error: Dict[str, Any] | None,
    ) -> bool: ...

    def _redact_sdk_tool_input_for_event(
      self,
      tool_name: str,
      tool_input: Dict[str, Any],
    ) -> Dict[str, Any]: ...

    def _settle_sdk_tool_result(
      self,
      tool_name: str,
      tool_input: Mapping[str, object],
      dispatch_entry: DispatchEntry,
      result: object,
      error: Mapping[str, object] | None,
      semantic_error: Mapping[str, object] | None,
    ) -> ToolResultSettlement: ...

  def _handle_stream_event(self, raw_event: Dict[str, Any]) -> None:
    event_type = str(raw_event.get("type") or "")
    if event_type == "message_start":
      message = _as_dict(raw_event.get("message"))
      usage = _as_dict(message.get("usage"))
      bind = self._capability_execution.bind
      reported_model = str(message.get("model") or "").strip() or None
      if reported_model is not None:
        reported_model = validate_reported_identity(
          bind,
          reported_model,
          registry=self._capability_execution.registry,
        )
        self._usage["provider_reported_model"] = reported_model
      if getattr(self, "_commercial_usage_producer", None) is None:
        return
      self._sdk_provider_call_usage = {
        "capability_bind": bind.to_json(),
        "provider_reported_model": reported_model,
        "input_tokens": int(usage.get("input_tokens") or 0),
        "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
        "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens") or 0),
        "output_tokens": 0,
        "provider_unit_deltas": _server_tool_unit_deltas(usage),
      }
      return

    if event_type == "message_delta":
      if getattr(self, "_commercial_usage_producer", None) is None:
        return
      usage = _as_dict(raw_event.get("usage"))
      current = dict(getattr(self, "_sdk_provider_call_usage", None) or {})
      bind = self._capability_execution.bind
      current.setdefault("capability_bind", bind.to_json())
      current.setdefault("input_tokens", 0)
      current.setdefault("cache_read_input_tokens", 0)
      current.setdefault("cache_creation_input_tokens", 0)
      current["output_tokens"] = int(usage.get("output_tokens") or 0)
      current["provider_unit_deltas"] = (
        _server_tool_unit_deltas(usage)
        or dict(current.get("provider_unit_deltas") or {})
      )
      self._pending_sdk_usage_deltas.append(current)
      self._sdk_provider_call_usage = None
      return

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
        if text and not getattr(self, "_suppress_text_after_accepted_ui_blocks", False):
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
      started_at = _time()
      if _should_escrow_raw_tool_input(tool_name):
        record_raw_tool_input = getattr(self._log, "record_raw_tool_input", None)
        if callable(record_raw_tool_input):
          record_raw_tool_input(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_input=tool_input,
          )
      redacted_tool_input = self._redact_sdk_tool_input_for_event(
        tool_name,
        tool_input,
      )
      self._pending_tool_calls[tool_call_id] = ToolCallInfo(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_input=tool_input,
        started_at=started_at,
        redacted_tool_input=copy.deepcopy(redacted_tool_input),
      )
      display = resolve_display(tool_name, redacted_tool_input)
      tool_start_event = {
        "type": "tool_call_start",
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "tool_input": copy.deepcopy(redacted_tool_input),
      }
      if display is not None:
        tool_start_event["display"] = display
      self._append(tool_start_event)
      self._active_tool_use = None

  def _complete_tool_call(
    self,
    tool_call_id: str,
    *,
    executed_tool_input: Dict[str, Any],
    result: Any | None = None,
    error: Dict[str, Any] | None = None,
  ) -> Dict[str, Any]:
    info = self._pending_tool_calls.pop(tool_call_id)

    server = _server_for_tool(info.tool_name)
    duration_ms = int((_time() - info.started_at) * 1000)
    result_bytes = len(json.dumps(result, default=str)) if result is not None else 0
    semantic_error = classify_semantic_tool_error(result) if error is None else None
    original_tool_name = info.tool_name
    origin = "local"
    if server is not None:
      if info.tool_name.startswith("mcp__"):
        original_tool_name = info.tool_name.split("__", 2)[2]
      origin = "mcp"
    dispatch_entry = _resolve_dispatch_entry(
      info.tool_name,
      origin=origin,
      server=server,
      original_tool_name=original_tool_name,
    )
    settlement = self._settle_sdk_tool_result(
      info.tool_name,
      executed_tool_input,
      dispatch_entry,
      result,
      error,
      semantic_error,
    )
    event = _build_tool_call_complete_event(
      tool_call_id=tool_call_id,
      tool_name=info.tool_name,
      result=result,
      error=error,
      duration_ms=duration_ms,
      server=server,
      dispatch=_build_dispatch_record_from_sources(
        entry=dispatch_entry,
        outcome=settlement.outcome,
        sources=settlement.sources,
        attempts=1,
      ),
      semantic_error=semantic_error,
    )
    if semantic_error is None and _is_accepted_ui_blocks_result(info.tool_name, result, error):
      self._suppress_text_after_accepted_ui_blocks = True
    self._clear_active_skill_if_report_door_completed(
      tool_name=info.tool_name,
      result=result,
      error=error,
    )
    self._append(event)
    self._call_on_tool_timing(
      tool_name=info.tool_name,
      server=server,
      duration_ms=duration_ms,
      is_error=event["is_error"],
      result_bytes=result_bytes,
      tool_call_id=tool_call_id,
      request_id=self._request_id,
    )
    return event

  def _interrupt_tool_call(self, tool_call_id: str, *, reason: str) -> None:
    info = self._pending_tool_calls.pop(tool_call_id, None)
    if info is None:
      return
    discovered_at = _time()
    duration_ms = int((discovered_at - info.started_at) * 1000)
    server = _server_for_tool(info.tool_name)
    self._append({
      "type": "tool_call_interrupted",
      "tool_call_id": tool_call_id,
      "tool_name": info.tool_name,
      "tool_input": copy.deepcopy(info.redacted_tool_input),
      "original_started_at": info.started_at,
      "discovered_at": discovered_at,
      "reason": reason,
      "tool_risk": get_tool_risk_value(info.tool_name),
      "role": "writer",
    })
    self._call_on_tool_timing(
      tool_name=info.tool_name,
      server=server,
      duration_ms=duration_ms,
      is_error=True,
      result_bytes=0,
      tool_call_id=tool_call_id,
      request_id=self._request_id,
    )

  def _flush_pending_tool_calls(self, *, outcome: str) -> None:
    for tool_call_id in list(self._pending_tool_calls.keys()):
      self._interrupt_tool_call(tool_call_id, reason=outcome)

  def _handle_system_message(self, message: Any) -> None:
    subtype = str(_get_attr(message, "subtype") or "")
    data = _as_dict(_get_attr(message, "data"))
    safe_subtype = sanitize_boundary_value(
      subtype or "unknown",
      sink="sdk_system_message_subtype_log",
      boundary=getattr(self, "_secret_boundary", None),
    )
    safe_data = sanitize_boundary_value(
      data,
      sink="sdk_system_message_data_log",
      boundary=getattr(self, "_secret_boundary", None),
    )
    if not isinstance(safe_subtype, str):
      safe_subtype = SANITIZATION_FAILED
    if subtype == "init":
      statuses = safe_data.get("mcp_servers") if isinstance(safe_data, dict) else None
      log.info(
        "[%s] SDK init | mcp_servers=%s",
        self._sid,
        statuses if statuses is not None else safe_data,
      )
      return
    log.info(
      "[%s] SDK system message | subtype=%s data=%s",
      self._sid,
      safe_subtype,
      safe_data,
    )

  def _handle_assistant_message(self, message: Any) -> None:
    error = _get_attr(message, "error")
    if error:
      safe_error = sanitize_boundary_value(
        error,
        sink="sdk_assistant_error_log",
        boundary=getattr(self, "_secret_boundary", None),
      )
      log.warning("[%s] SDK assistant message error: %s", self._sid, safe_error)

  def _emit_stream_complete(
    self,
    *,
    terminal_disposition: TerminalDisposition = "completed",
    reason: str | None = None,
  ) -> None:
    if self._stream_terminal_emitted:
      return
    if terminal_disposition not in TERMINAL_DISPOSITIONS:
      raise ValueError(
        f"terminal_disposition must be one of {sorted(TERMINAL_DISPOSITIONS)}"
      )
    if terminal_disposition == "interrupted" and not str(reason or "").strip():
      raise ValueError("interrupted stream_complete requires a reason")

    cost_usd = float(self._usage.get("estimated_cost") or 0.0)
    bind = self._capability_execution.bind
    usage = {
      "capability_bind": bind.to_json(),
      "provider_reported_model": self._usage.get("provider_reported_model"),
      "input_tokens": int(self._usage.get("input_tokens") or 0),
      "output_tokens": int(self._usage.get("output_tokens") or 0),
      "cache_creation_input_tokens": int(self._usage.get("cache_creation_input_tokens") or 0),
      "cache_read_input_tokens": int(self._usage.get("cache_read_input_tokens") or 0),
      "estimated_cost": round(cost_usd, 4),
    }
    terminal_event = {
      "type": "stream_complete",
      "terminal_disposition": terminal_disposition,
      "usage": usage,
    }
    if terminal_disposition == "interrupted":
      terminal_event["reason"] = str(reason).strip()
    self._append(terminal_event)
    self._stream_terminal_emitted = True


__all__ = [
  "ToolCallInfo",
  "_ActiveToolUse",
  "_SDKRunnerStreamMixin",
]
