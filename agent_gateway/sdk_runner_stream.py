from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import sys
import time
from typing import Any, Dict, List, Tuple

from . import sdk_runner_helpers as _sdk_runner_helpers
from .tool_display import resolve_display
from .tool_result_semantics import classify_semantic_tool_error


log = logging.getLogger("agent_gateway.sdk_runner")
_PARENT_MODULE = "agent_gateway.sdk_runner"


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


def _parent_attr(name: str, fallback: Any) -> Any:
  parent = sys.modules.get(_PARENT_MODULE)
  if parent is not None and hasattr(parent, name):
    return getattr(parent, name)
  return fallback


def _as_dict(value: Any) -> Dict[str, Any]:
  return _parent_attr("_as_dict", _sdk_runner_helpers.as_dict)(value)


def _get_attr(value: Any, key: str, default: Any = None) -> Any:
  return _parent_attr("_get_attr", _sdk_runner_helpers.get_attr)(value, key, default)


def _parse_result_payload(value: Any) -> Any:
  return _parent_attr("_parse_result_payload", _sdk_runner_helpers.parse_result_payload)(value)


def _summarize_error_payload(value: Any) -> str:
  return _parent_attr("_summarize_error_payload", _sdk_runner_helpers.summarize_error_payload)(value)


def _server_for_tool(tool_name: str) -> str | None:
  return _parent_attr("_server_for_tool", _sdk_runner_helpers.server_for_tool)(tool_name)


def _should_escrow_raw_tool_input(tool_name: str) -> bool:
  return _parent_attr(
    "_should_escrow_raw_tool_input",
    _sdk_runner_helpers.should_escrow_raw_tool_input,
  )(tool_name)


def _redact_tool_input_for_event(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
  return _parent_attr(
    "_redact_tool_input_for_event",
    _sdk_runner_helpers.redact_tool_input_for_event,
  )(tool_name, tool_input)


def _resolve_display(tool_name: str, tool_input: Dict[str, Any]) -> Any:
  return _parent_attr("resolve_display", resolve_display)(tool_name, tool_input)


def _classify_semantic_tool_error(result: Any) -> Dict[str, Any] | None:
  return _parent_attr("classify_semantic_tool_error", classify_semantic_tool_error)(result)


def _time() -> float:
  return _parent_attr("time", time).time()


class _SDKRunnerStreamMixin:
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
        started_at=_time(),
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
      display = _resolve_display(tool_name, redacted_tool_input)
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
    duration_ms = int((_time() - info.started_at) * 1000)
    result_bytes = len(json.dumps(result, default=str)) if result is not None else 0
    semantic_error = _classify_semantic_tool_error(result) if error is None else None
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
      tool_call_id=tool_call_id,
      request_id=self._request_id,
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


__all__ = [
  "ToolCallInfo",
  "_ActiveToolUse",
  "_SDKRunnerStreamMixin",
]
