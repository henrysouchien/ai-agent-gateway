from __future__ import annotations

import json
from typing import Any

import httpx

from .base import StreamEvent
from .codex_helpers import (
  _ResponsesStreamState,
  _convert_messages,
  _convert_tools,
  _map_event as _map_responses_event,
  _parse_sse,
  _system_prompt_text,
)


DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_INSTRUCTIONS = "Follow the user's instructions."
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


def resolve_responses_url(base_url: str | None) -> str:
  normalized = str(base_url or DEFAULT_XAI_BASE_URL).strip().rstrip("/")
  if normalized.endswith("/responses"):
    return normalized
  return f"{normalized}/responses"


def build_headers(token: str, additional_headers: dict[str, Any] | None = None) -> dict[str, str]:
  headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "text/event-stream",
    "Content-Type": "application/json",
  }
  for key, value in (additional_headers or {}).items():
    headers[str(key)] = str(value)
  return headers


def map_event(event: dict[str, Any], state: _ResponsesStreamState) -> list[StreamEvent]:
  """Map Responses events while preserving xAI provider identity in failures."""
  event_type = event.get("type")
  if event_type == "error":
    detail = str(event.get("message") or event.get("code") or json.dumps(event, separators=(",", ":")))
    raise RuntimeError(f"xAI error: {detail}")
  if event_type == "response.failed":
    response = event.get("response") if isinstance(event.get("response"), dict) else {}
    error = response.get("error") if isinstance(response.get("error"), dict) else {}
    detail = str(error.get("message") or "xAI response failed")
    raise RuntimeError(f"xAI response failed: {detail}")
  had_active_tool = state.current_block_type == "tool_use"
  mapped = _map_responses_event(event, state)
  # xAI may deliver a complete function call in output_item.added without
  # separate argument delta events. Preserve the normal start/delta/end
  # gateway lifecycle in that case.
  if event_type == "response.output_item.added":
    item = event.get("item") if isinstance(event.get("item"), dict) else {}
    arguments = str(item.get("arguments") or "")
    if item.get("type") == "function_call" and arguments:
      mapped.append(StreamEvent(type="tool_use_delta", tool_input_json=arguments))
  elif event_type == "response.output_item.done" and not had_active_tool:
    item = event.get("item") if isinstance(event.get("item"), dict) else {}
    if item.get("type") == "function_call":
      call_id = str(item.get("call_id") or "")
      item_id = str(item.get("id") or "")
      tool_id = f"{call_id}|{item_id}" if item_id else call_id
      arguments = str(item.get("arguments") or "{}")
      mapped = [
        StreamEvent(
          type="tool_use_start",
          tool_id=tool_id,
          tool_name=str(item.get("name") or ""),
          raw_block={"type": "tool_use", "id": tool_id, "name": str(item.get("name") or "")},
        ),
        StreamEvent(type="tool_use_delta", tool_input_json=arguments),
        *mapped,
      ]
      state.saw_tool_use = True
  return mapped


async def parse_error_response(response: httpx.Response) -> str:
  raw = await response.aread()
  raw_text = raw.decode("utf-8", errors="replace")
  message = raw_text or response.reason_phrase or "Request failed"
  try:
    parsed = json.loads(raw_text)
  except json.JSONDecodeError:
    parsed = None
  if isinstance(parsed, dict):
    error = parsed.get("error")
    if isinstance(error, dict):
      message = str(error.get("message") or error.get("code") or message)
  return f"xAI API error (status={response.status_code}): {message}"


__all__ = [
  "DEFAULT_INSTRUCTIONS",
  "DEFAULT_XAI_BASE_URL",
  "RETRYABLE_STATUSES",
  "_ResponsesStreamState",
  "_convert_messages",
  "_convert_tools",
  "_parse_sse",
  "_system_prompt_text",
  "build_headers",
  "map_event",
  "parse_error_response",
  "resolve_responses_url",
]
