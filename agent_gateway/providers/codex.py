from __future__ import annotations

import asyncio
import base64
import json
import platform
import re
import time
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator
from weakref import WeakKeyDictionary

import httpx

from .base import ModelInfo, ModelProvider, StreamEvent, ThinkingLevel, truncate_to_last_compaction


DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
JWT_CLAIM_PATH = "https://api.openai.com/auth"
DEFAULT_INSTRUCTIONS = "Follow the user's instructions."
_BETA_HEADER = "responses=experimental"
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_RETRYABLE_RE = re.compile(
  r"rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused",
  re.IGNORECASE,
)
_CODEX_RESPONSE_STATUSES = {
  "completed",
  "incomplete",
  "failed",
  "cancelled",
  "queued",
  "in_progress",
}
_TOOL_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_SURROGATE_RE = re.compile(r"[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]")


_MODEL_INFO_BY_TAG: list[tuple[tuple[str, ...], ModelInfo]] = [
  (
    ("gpt-5.5",),
    ModelInfo(
      id="gpt-5.5",
      provider="codex",
      context_window=1_050_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
      input_cost_per_mtok=5.00,
      output_cost_per_mtok=30.00,
      cache_read_cost_per_mtok=0.50,
    ),
  ),
  (
    ("gpt-5.1",),
    ModelInfo(
      id="gpt-5.1",
      provider="codex",
      context_window=272_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
      input_cost_per_mtok=1.25,
      output_cost_per_mtok=10.0,
      cache_read_cost_per_mtok=0.125,
    ),
  ),
  (
    ("gpt-5.1-codex-max",),
    ModelInfo(
      id="gpt-5.1-codex-max",
      provider="codex",
      context_window=272_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
      input_cost_per_mtok=1.25,
      output_cost_per_mtok=10.0,
      cache_read_cost_per_mtok=0.125,
    ),
  ),
  (
    ("gpt-5.1-codex-mini",),
    ModelInfo(
      id="gpt-5.1-codex-mini",
      provider="codex",
      context_window=272_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
      input_cost_per_mtok=0.25,
      output_cost_per_mtok=2.0,
      cache_read_cost_per_mtok=0.025,
    ),
  ),
  (
    ("gpt-5.2",),
    ModelInfo(
      id="gpt-5.2",
      provider="codex",
      context_window=272_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
      input_cost_per_mtok=1.75,
      output_cost_per_mtok=14.0,
      cache_read_cost_per_mtok=0.175,
    ),
  ),
  (
    ("gpt-5.2-codex",),
    ModelInfo(
      id="gpt-5.2-codex",
      provider="codex",
      context_window=272_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
      input_cost_per_mtok=1.75,
      output_cost_per_mtok=14.0,
      cache_read_cost_per_mtok=0.175,
    ),
  ),
  (
    ("gpt-5.3-codex",),
    ModelInfo(
      id="gpt-5.3-codex",
      provider="codex",
      context_window=272_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
      input_cost_per_mtok=1.75,
      output_cost_per_mtok=14.0,
      cache_read_cost_per_mtok=0.175,
    ),
  ),
  (
    ("gpt-5.3-codex-spark",),
    ModelInfo(
      id="gpt-5.3-codex-spark",
      provider="codex",
      context_window=128_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=False,
      input_cost_per_mtok=0.0,
      output_cost_per_mtok=0.0,
      cache_read_cost_per_mtok=0.0,
    ),
  ),
  (
    ("gpt-5.4",),
    ModelInfo(
      id="gpt-5.4",
      provider="codex",
      context_window=272_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
      input_cost_per_mtok=2.5,
      output_cost_per_mtok=15.0,
      cache_read_cost_per_mtok=0.25,
    ),
  ),
]


@dataclass
class _ResponsesStreamState:
  message_started: bool = False
  current_item: dict[str, Any] | None = None
  current_block_type: str | None = None
  current_text: str = ""
  current_thinking: str = ""
  current_tool_id: str = ""
  current_tool_name: str = ""
  current_tool_json: str = ""
  saw_tool_use: bool = False


def _config_base_url(config: dict[str, Any]) -> str | None:
  for key in ("base_url", "baseURL", "api_base_url", "api_base"):
    value = config.get(key)
    if value:
      return str(value)
  return None


def _model_matches_tag(model_id: str, tag: str) -> bool:
  candidates = [model_id, model_id.rsplit("/", 1)[-1]]
  return any(candidate == tag or candidate.startswith(f"{tag}-") for candidate in candidates)


def _credential_token(config: dict[str, Any]) -> str:
  mode = str(config.get("auth_mode", "")).strip().lower()
  auth_token = str(config.get("auth_token", "") or "").strip()
  api_key = str(config.get("api_key", "") or "").strip()
  if mode == "oauth":
    return auth_token or api_key
  if mode == "api":
    return api_key or auth_token
  return auth_token or api_key


def _json_dumps_compact(value: Any) -> str:
  return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _encode_text_signature_v1(message_id: str, phase: str | None = None) -> str:
  payload: dict[str, Any] = {"v": 1, "id": message_id}
  if phase:
    payload["phase"] = phase
  return _json_dumps_compact(payload)


def _parse_text_signature(signature: str) -> dict[str, Any] | None:
  if not signature:
    return None
  if signature.startswith("{"):
    try:
      parsed = json.loads(signature)
    except json.JSONDecodeError:
      parsed = None
    if isinstance(parsed, dict) and parsed.get("v") == 1 and isinstance(parsed.get("id"), str):
      phase = parsed.get("phase")
      if phase in {"commentary", "final_answer"}:
        return {"id": parsed["id"], "phase": phase}
      return {"id": parsed["id"]}
  return {"id": signature}


def _short_hash(text: str) -> str:
  def _imul(a: int, b: int) -> int:
    return ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) & 0xFFFFFFFF

  def _to_base36(value: int) -> str:
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value == 0:
      return "0"
    digits: list[str] = []
    while value:
      value, rem = divmod(value, 36)
      digits.append(chars[rem])
    return "".join(reversed(digits))

  h1 = 0xDEADBEEF
  h2 = 0x41C6CE57
  for char in text:
    code = ord(char)
    h1 = _imul(h1 ^ code, 2654435761)
    h2 = _imul(h2 ^ code, 1597334677)
  h1 = (_imul(h1 ^ (h1 >> 16), 2246822507) ^ _imul(h2 ^ (h2 >> 13), 3266489909)) & 0xFFFFFFFF
  h2 = (_imul(h2 ^ (h2 >> 16), 2246822507) ^ _imul(h1 ^ (h1 >> 13), 3266489909)) & 0xFFFFFFFF
  return f"{_to_base36(h2)}{_to_base36(h1)}"


def _sanitize_surrogates(text: str) -> str:
  return _SURROGATE_RE.sub("", str(text))


def _system_prompt_text(system_prompt: str | list[tuple[str, bool]] | None) -> str:
  if system_prompt is None:
    return ""
  if isinstance(system_prompt, list):
    return "\n\n".join(text for text, _should_cache in system_prompt if text)
  return str(system_prompt)


def _resolve_codex_url(base_url: str | None) -> str:
  raw = str(base_url or DEFAULT_CODEX_BASE_URL).strip() or DEFAULT_CODEX_BASE_URL
  normalized = raw.rstrip("/")
  if normalized.endswith("/codex/responses"):
    return normalized
  if normalized.endswith("/codex"):
    return f"{normalized}/responses"
  return f"{normalized}/codex/responses"


def _normalize_codex_status(status: Any) -> str | None:
  if not isinstance(status, str):
    return None
  return status if status in _CODEX_RESPONSE_STATUSES else None


def _map_reasoning_effort(level: ThinkingLevel) -> str | None:
  if level == ThinkingLevel.NONE:
    return None
  if level == ThinkingLevel.MINIMAL:
    return "minimal"
  if level == ThinkingLevel.LOW:
    return "low"
  if level == ThinkingLevel.MEDIUM:
    return "medium"
  return "high"


def _clamp_reasoning_effort(model_id: str, effort: str) -> str:
  identifier = model_id.rsplit("/", 1)[-1]
  if (
    identifier.startswith("gpt-5.2")
    or identifier.startswith("gpt-5.3")
    or identifier.startswith("gpt-5.4")
    or identifier.startswith("gpt-5.5")
  ) and effort == "minimal":
    return "low"
  if identifier == "gpt-5.1-codex-mini":
    return "high" if effort == "high" else "medium"
  return effort


def _same_model_message(message: dict[str, Any], model_info: ModelInfo) -> bool:
  return (
    str(message.get("provider", "")) == model_info.provider
    and str(message.get("model", "")) == model_info.id
  )


def _synthetic_tool_result(tool_id: str, tool_name: str) -> dict[str, Any]:
  return {
    "type": "tool_result",
    "tool_use_id": tool_id,
    "content": json.dumps({"error": {"code": "missing_tool_result", "message": "No result provided"}}),
    "is_error": True,
    "tool_name": tool_name,
  }


def _normalize_responses_tool_call_id(tool_id: str) -> str:
  raw = str(tool_id or "")
  if "|" not in raw:
    return raw
  call_id, item_id = raw.split("|", 1)
  normalized_call_id = _TOOL_ID_RE.sub("_", call_id)
  normalized_item_id = _TOOL_ID_RE.sub("_", item_id)
  if not normalized_item_id.startswith("fc"):
    normalized_item_id = f"fc_{normalized_item_id}"
  normalized_call_id = normalized_call_id[:64].rstrip("_")
  normalized_item_id = normalized_item_id[:64].rstrip("_")
  return f"{normalized_call_id}|{normalized_item_id}"


def _assistant_text_block(text: str, msg_index: int, block: dict[str, Any]) -> dict[str, Any]:
  parsed_signature = _parse_text_signature(str(block.get("textSignature") or ""))
  message_id = str(parsed_signature.get("id") or "") if isinstance(parsed_signature, dict) else ""
  if not message_id:
    message_id = f"msg_{msg_index}"
  elif len(message_id) > 64:
    message_id = f"msg_{_short_hash(message_id)}"
  output = {
    "type": "message",
    "role": "assistant",
    "content": [{"type": "output_text", "text": _sanitize_surrogates(text), "annotations": []}],
    "status": "completed",
    "id": message_id,
  }
  phase = parsed_signature.get("phase") if isinstance(parsed_signature, dict) else None
  if isinstance(phase, str):
    output["phase"] = phase
  return output


def _tool_result_output(content: Any, *, supports_vision: bool) -> Any:
  if isinstance(content, list):
    text_result = "\n".join(
      str(block.get("text", ""))
      for block in content
      if isinstance(block, dict) and block.get("type") == "text"
    )
    has_images = any(isinstance(block, dict) and block.get("type") == "image" for block in content)
    has_text = bool(text_result)
    if has_images and supports_vision:
      content_parts: list[dict[str, Any]] = []
      if has_text:
        content_parts.append({"type": "input_text", "text": _sanitize_surrogates(text_result)})
      for block in content:
        if not isinstance(block, dict) or block.get("type") != "image":
          continue
        mime_type = block.get("mime_type") or block.get("media_type") or block.get("mimeType")
        data = block.get("data_base64") or block.get("data")
        if mime_type and data:
          content_parts.append(
            {
              "type": "input_image",
              "detail": "auto",
              "image_url": f"data:{mime_type};base64,{data}",
            }
          )
      return content_parts
    return _sanitize_surrogates(text_result if has_text else "(see attached image)")

  if isinstance(content, str):
    return _sanitize_surrogates(content)
  if isinstance(content, (dict, list)):
    return _sanitize_surrogates(json.dumps(content, default=str))
  if content is None:
    return ""
  return _sanitize_surrogates(str(content))


def _parse_streaming_json(partial_json: str) -> dict[str, Any]:
  if not partial_json or not partial_json.strip():
    return {}
  try:
    parsed = json.loads(partial_json)
  except json.JSONDecodeError:
    return {}
  return parsed if isinstance(parsed, dict) else {}


def _extract_account_id(token: str) -> str:
  try:
    parts = str(token).split(".")
    if len(parts) != 3:
      raise ValueError("Invalid token")
    payload_part = parts[1]
    payload_part += "=" * (-len(payload_part) % 4)
    payload = json.loads(base64.urlsafe_b64decode(payload_part.encode("ascii")).decode("utf-8"))
    account_id = payload.get(JWT_CLAIM_PATH, {}).get("chatgpt_account_id")
    if not account_id:
      raise ValueError("No account ID in token")
    return str(account_id)
  except Exception as exc:  # pragma: no cover - exact branch covered by tests
    raise ValueError("Failed to extract accountId from token") from exc


def _build_headers(
  init_headers: dict[str, Any] | None,
  additional_headers: dict[str, Any] | None,
  account_id: str,
  token: str,
  session_id: str | None = None,
) -> dict[str, str]:
  headers = {str(key): str(value) for key, value in (init_headers or {}).items()}
  headers["Authorization"] = f"Bearer {token}"
  headers["chatgpt-account-id"] = account_id
  headers["OpenAI-Beta"] = _BETA_HEADER
  headers["originator"] = "pi"
  headers["User-Agent"] = f"pi ({platform.system().lower()} {platform.release()}; {platform.machine()})"
  headers["accept"] = "text/event-stream"
  headers["content-type"] = "application/json"
  for key, value in (additional_headers or {}).items():
    headers[str(key)] = str(value)
  if session_id:
    headers["session_id"] = str(session_id)
  return headers


def _convert_tools(tools: list[dict[str, Any]], *, strict: bool | None = False) -> list[dict[str, Any]]:
  converted: list[dict[str, Any]] = []
  for tool in tools:
    converted.append(
      {
        "type": "function",
        "name": str(tool.get("name", "")),
        "description": str(tool.get("description", "")),
        "parameters": tool.get("parameters") or tool.get("input_schema") or {"type": "object", "properties": {}},
        "strict": strict,
      }
    )
  return converted


def _convert_messages(messages: list[dict[str, Any]], model_info: ModelInfo) -> list[dict[str, Any]]:
  converted: list[dict[str, Any]] = []
  msg_index = 0

  for message in messages:
    role = message.get("role")

    if role == "user":
      content = message.get("content")
      if isinstance(content, str):
        converted.append(
          {
            "role": "user",
            "content": [{"type": "input_text", "text": _sanitize_surrogates(content)}],
          }
        )
        msg_index += 1
        continue

      if not isinstance(content, list):
        if content is not None:
          converted.append(
            {
              "role": "user",
              "content": [{"type": "input_text", "text": _sanitize_surrogates(str(content))}],
            }
          )
        msg_index += 1
        continue

      pending_user_parts: list[dict[str, Any]] = []
      for block in content:
        if not isinstance(block, dict):
          continue
        block_type = block.get("type")
        if block_type == "text":
          pending_user_parts.append({"type": "input_text", "text": _sanitize_surrogates(str(block.get("text", "")))})
          continue
        if block_type == "image" and model_info.supports_vision:
          mime_type = block.get("mime_type") or block.get("media_type") or block.get("mimeType")
          data = block.get("data_base64") or block.get("data")
          if mime_type and data:
            pending_user_parts.append(
              {
                "type": "input_image",
                "detail": "auto",
                "image_url": f"data:{mime_type};base64,{data}",
              }
            )
          continue
        if block_type != "tool_result":
          continue
        if pending_user_parts:
          converted.append({"role": "user", "content": pending_user_parts})
          pending_user_parts = []
        tool_use_id = str(block.get("tool_use_id") or "")
        call_id = tool_use_id.split("|", 1)[0]
        converted.append(
          {
            "type": "function_call_output",
            "call_id": call_id,
            "output": _tool_result_output(block.get("content"), supports_vision=model_info.supports_vision),
          }
        )

      if pending_user_parts:
        converted.append({"role": "user", "content": pending_user_parts})
      msg_index += 1
      continue

    if role == "assistant":
      content = message.get("content")
      if not isinstance(content, list):
        text = str(content or "")
        if text:
          converted.append(_assistant_text_block(text, msg_index, {}))
        msg_index += 1
        continue

      output: list[dict[str, Any]] = []
      is_different_model = (
        str(message.get("provider", "")) == model_info.provider
        and str(message.get("model", "")) != model_info.id
      )
      for block in content:
        if not isinstance(block, dict):
          continue
        block_type = block.get("type")
        if block_type == "thinking":
          thinking_signature = block.get("thinkingSignature") or block.get("signature")
          if thinking_signature:
            try:
              reasoning_item = json.loads(str(thinking_signature))
            except json.JSONDecodeError:
              reasoning_item = None
            if isinstance(reasoning_item, dict):
              output.append(reasoning_item)
          continue
        if block_type == "text":
          output.append(_assistant_text_block(str(block.get("text", "")), msg_index, block))
          continue
        if block_type not in {"tool_use", "server_tool_use"}:
          continue
        tool_call_id = str(block.get("id") or "")
        call_id, item_id = (tool_call_id.split("|", 1) + [""])[:2]
        function_call = {
          "type": "function_call",
          "call_id": call_id,
          "name": str(block.get("name", "")),
          "arguments": json.dumps(block.get("input", {}), default=str),
        }
        if item_id and not (is_different_model and item_id.startswith("fc_")):
          function_call["id"] = item_id
        output.append(function_call)

      if output:
        converted.extend(output)
      msg_index += 1
      continue

    if role == "tool":
      tool_call_id = str(message.get("tool_call_id") or "")
      call_id = tool_call_id.split("|", 1)[0]
      converted.append(
        {
          "type": "function_call_output",
          "call_id": call_id,
          "output": _tool_result_output(message.get("content"), supports_vision=model_info.supports_vision),
        }
      )
      msg_index += 1
      continue

    msg_index += 1

  return converted


def _parse_sse(buffer: str) -> tuple[list[dict[str, Any]], str]:
  events: list[dict[str, Any]] = []
  idx = buffer.find("\n\n")
  while idx != -1:
    chunk = buffer[:idx]
    buffer = buffer[idx + 2 :]
    data_lines = [line[5:].strip() for line in chunk.split("\n") if line.startswith("data:")]
    if data_lines:
      data = "\n".join(data_lines).strip()
      if data and data != "[DONE]":
        try:
          parsed = json.loads(data)
        except json.JSONDecodeError:
          parsed = None
        if isinstance(parsed, dict):
          events.append(parsed)
    idx = buffer.find("\n\n")
  return events, buffer


def _map_stop_reason(response: dict[str, Any], *, saw_tool_use: bool) -> str:
  status = _normalize_codex_status(response.get("status"))
  if status == "failed":
    return "error"
  if status == "incomplete":
    details = response.get("incomplete_details") or {}
    reason = str(details.get("reason") or "")
    if reason == "tool_use":
      return "tool_use"
    if reason == "max_output_tokens":
      return "max_tokens"
    return "max_tokens"
  if saw_tool_use:
    return "tool_use"
  return "end_turn"


def _map_event(event: dict[str, Any], state: _ResponsesStreamState) -> list[StreamEvent]:
  event_type = event.get("type")
  if not isinstance(event_type, str):
    return []

  if event_type == "error":
    code = str(event.get("code") or "")
    message = str(event.get("message") or "")
    detail = message or code or _json_dumps_compact(event)
    raise RuntimeError(f"Codex error: {detail}")

  if event_type == "response.failed":
    response = event.get("response") or {}
    if not isinstance(response, dict):
      response = {}
    error = response.get("error") or {}
    message = str(error.get("message") or "")
    raise RuntimeError(message or "Codex response failed")

  if event_type in {"response.done", "response.completed"}:
    response = event.get("response") or {}
    if not isinstance(response, dict):
      response = {}
    normalized_response = dict(response)
    normalized_response["status"] = _normalize_codex_status(response.get("status"))
    usage = normalized_response.get("usage") or {}
    if not isinstance(usage, dict):
      usage = {}
    input_details = usage.get("input_tokens_details") or {}
    cached_tokens = int(input_details.get("cached_tokens") or 0) if isinstance(input_details, dict) else 0
    input_tokens = max(0, int(usage.get("input_tokens") or 0) - cached_tokens)
    output_tokens = int(usage.get("output_tokens") or 0)
    events: list[StreamEvent] = []
    if not state.message_started:
      state.message_started = True
      events.append(
        StreamEvent(
          type="message_start",
          input_tokens=input_tokens,
          cache_read_tokens=cached_tokens,
        )
      )
    events.append(StreamEvent(type="usage_update", output_tokens=output_tokens))
    events.append(
      StreamEvent(
        type="message_end",
        stop_reason=_map_stop_reason(normalized_response, saw_tool_use=state.saw_tool_use),
      )
    )
    return events

  if event_type == "response.output_item.added":
    item = event.get("item") or {}
    if not isinstance(item, dict):
      return []
    state.current_item = item
    item_type = item.get("type")
    if item_type == "reasoning":
      state.current_block_type = "thinking"
      state.current_thinking = ""
      return []
    if item_type == "message":
      state.current_block_type = "text"
      state.current_text = ""
      return []
    if item_type == "function_call":
      state.current_block_type = "tool_use"
      state.current_tool_name = str(item.get("name") or "")
      call_id = str(item.get("call_id") or "")
      item_id = str(item.get("id") or "")
      state.current_tool_id = f"{call_id}|{item_id}" if item_id else call_id
      state.current_tool_json = str(item.get("arguments") or "")
      state.saw_tool_use = True
      return [
        StreamEvent(
          type="tool_use_start",
          tool_id=state.current_tool_id,
          tool_name=state.current_tool_name,
          raw_block={"type": "tool_use", "id": state.current_tool_id, "name": state.current_tool_name},
        )
      ]
    return []

  if event_type == "response.reasoning_summary_part.added":
    if state.current_item and state.current_item.get("type") == "reasoning":
      summary = state.current_item.setdefault("summary", [])
      if isinstance(summary, list):
        part = event.get("part") or {}
        if isinstance(part, dict):
          summary.append(part)
    return []

  if event_type == "response.reasoning_summary_text.delta":
    if state.current_block_type != "thinking" or not state.current_item or state.current_item.get("type") != "reasoning":
      return []
    summary = state.current_item.setdefault("summary", [])
    if not isinstance(summary, list) or not summary or not isinstance(summary[-1], dict):
      return []
    delta = str(event.get("delta") or "")
    state.current_thinking += delta
    summary[-1]["text"] = f"{summary[-1].get('text', '')}{delta}"
    return [StreamEvent(type="thinking_delta", thinking_text=delta)]

  if event_type == "response.reasoning_summary_part.done":
    if state.current_block_type != "thinking" or not state.current_item or state.current_item.get("type") != "reasoning":
      return []
    summary = state.current_item.setdefault("summary", [])
    if not isinstance(summary, list) or not summary or not isinstance(summary[-1], dict):
      return []
    summary[-1]["text"] = f"{summary[-1].get('text', '')}\n\n"
    state.current_thinking += "\n\n"
    return [StreamEvent(type="thinking_delta", thinking_text="\n\n")]

  if event_type == "response.content_part.added":
    if not state.current_item or state.current_item.get("type") != "message":
      return []
    part = event.get("part") or {}
    if not isinstance(part, dict) or part.get("type") not in {"output_text", "refusal"}:
      return []
    content = state.current_item.setdefault("content", [])
    if isinstance(content, list):
      content.append(part)
    return []

  if event_type == "response.output_text.delta":
    if state.current_block_type != "text" or not state.current_item or state.current_item.get("type") != "message":
      return []
    content = state.current_item.get("content")
    if not isinstance(content, list) or not content or not isinstance(content[-1], dict):
      return []
    last_part = content[-1]
    if last_part.get("type") != "output_text":
      return []
    delta = str(event.get("delta") or "")
    state.current_text += delta
    last_part["text"] = f"{last_part.get('text', '')}{delta}"
    return [StreamEvent(type="text_delta", text=delta)]

  if event_type == "response.refusal.delta":
    if state.current_block_type != "text" or not state.current_item or state.current_item.get("type") != "message":
      return []
    content = state.current_item.get("content")
    if not isinstance(content, list) or not content or not isinstance(content[-1], dict):
      return []
    last_part = content[-1]
    if last_part.get("type") != "refusal":
      return []
    delta = str(event.get("delta") or "")
    state.current_text += delta
    last_part["refusal"] = f"{last_part.get('refusal', '')}{delta}"
    return [StreamEvent(type="text_delta", text=delta)]

  if event_type == "response.function_call_arguments.delta":
    if state.current_block_type != "tool_use" or not state.current_item or state.current_item.get("type") != "function_call":
      return []
    delta = str(event.get("delta") or "")
    state.current_tool_json += delta
    return [StreamEvent(type="tool_use_delta", tool_input_json=delta)]

  if event_type == "response.function_call_arguments.done":
    if state.current_block_type == "tool_use" and state.current_item and state.current_item.get("type") == "function_call":
      state.current_tool_json = str(event.get("arguments") or "")
    return []

  if event_type == "response.output_item.done":
    item = event.get("item") or {}
    if not isinstance(item, dict):
      return []
    item_type = item.get("type")
    if item_type == "reasoning" and state.current_block_type == "thinking":
      summary = item.get("summary")
      if isinstance(summary, list):
        state.current_thinking = "\n\n".join(
          str(part.get("text", ""))
          for part in summary
          if isinstance(part, dict)
        )
      signature = _json_dumps_compact(item)
      raw_block = {
        "type": "thinking",
        "thinking": state.current_thinking,
        "signature": signature,
        "thinkingSignature": signature,
      }
      state.current_block_type = None
      state.current_item = None
      return [
        StreamEvent(
          type="thinking_end",
          thinking_text=state.current_thinking,
          signature=signature,
          raw_block=raw_block,
        )
      ]
    if item_type == "message" and state.current_block_type == "text":
      content = item.get("content") or []
      if isinstance(content, list):
        state.current_text = "".join(
          str(part.get("text") if part.get("type") == "output_text" else part.get("refusal", ""))
          for part in content
          if isinstance(part, dict)
        )
      text_signature = _encode_text_signature_v1(str(item.get("id") or ""), str(item.get("phase") or "") or None)
      raw_block = {
        "type": "text",
        "text": state.current_text,
        "textSignature": text_signature,
      }
      state.current_block_type = None
      state.current_item = None
      return [StreamEvent(type="text_end", text=state.current_text, raw_block=raw_block)]
    if item_type == "function_call":
      args_source = state.current_tool_json or str(item.get("arguments") or "{}")
      tool_input = _parse_streaming_json(args_source)
      call_id = str(item.get("call_id") or "")
      item_id = str(item.get("id") or "")
      tool_id = f"{call_id}|{item_id}" if item_id else call_id
      try:
        from agent.shared.tool_redaction import get_audit_hmac_secret, redact_tool_input

        redacted_tool_input = redact_tool_input(
          str(item.get("name") or ""),
          tool_input,
          deployment_secret=get_audit_hmac_secret(),
        )
      except Exception:
        redacted_tool_input = tool_input
      raw_block = {
        "type": "tool_use",
        "id": tool_id,
        "name": str(item.get("name") or ""),
        "input": redacted_tool_input,
      }
      state.current_block_type = None
      state.current_item = None
      state.current_tool_id = ""
      state.current_tool_name = ""
      state.current_tool_json = ""
      return [
        StreamEvent(
          type="tool_use_end",
          tool_id=tool_id,
          tool_name=str(item.get("name") or ""),
          tool_input_json=args_source,
          tool_input=tool_input,
          raw_block=raw_block,
        )
      ]
    return []

  return []


async def _parse_error_response(response: httpx.Response) -> dict[str, str | None]:
  raw = await response.aread()
  raw_text = raw.decode("utf-8", errors="replace")
  message = raw_text or response.reason_phrase or "Request failed"
  friendly_message: str | None = None
  try:
    parsed = json.loads(raw_text)
  except json.JSONDecodeError:
    parsed = None
  if isinstance(parsed, dict):
    error = parsed.get("error") or {}
    if isinstance(error, dict):
      code = str(error.get("code") or error.get("type") or "")
      if re.search(r"usage_limit_reached|usage_not_included|rate_limit_exceeded", code, re.IGNORECASE) or response.status_code == 429:
        plan_type = str(error.get("plan_type") or "").lower()
        plan = f" ({plan_type} plan)" if plan_type else ""
        resets_at = error.get("resets_at")
        mins: int | None = None
        if isinstance(resets_at, (int, float)):
          mins = max(0, round((float(resets_at) - time.time()) / 60))
        when = f" Try again in ~{mins} min." if mins is not None else ""
        friendly_message = f"You have hit your ChatGPT usage limit{plan}.{when}".strip()
      message = str(error.get("message") or friendly_message or message)
  return {"message": message, "friendly_message": friendly_message}


class CodexProvider(ModelProvider):
  """`ModelProvider` implementation for the ChatGPT Codex responses backend."""

  name = "codex"

  def __init__(self) -> None:
    self._client_state: WeakKeyDictionary[httpx.AsyncClient, dict[str, Any]] = WeakKeyDictionary()

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(_credential_token(config))

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    token = _credential_token(config)
    if not token:
      raise RuntimeError("No Codex token configured")

    account_id = _extract_account_id(token)
    base_url = _config_base_url(config)
    endpoint_url = _resolve_codex_url(base_url)

    client_kwargs: dict[str, Any] = {}
    effective_timeout = timeout if timeout is not None else 120.0
    client_kwargs["timeout"] = httpx.Timeout(timeout=effective_timeout, connect=10.0)
    client = httpx.AsyncClient(**client_kwargs)
    self._client_state[client] = {
      "token": token,
      "account_id": account_id,
      "endpoint_url": endpoint_url,
      "init_headers": dict(config.get("headers") or {}) if isinstance(config.get("headers"), dict) else {},
    }
    return client

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    if client is None or not hasattr(client, "aclose"):
      return
    self._client_state.pop(client, None)
    try:
      await asyncio.wait_for(client.aclose(), timeout=timeout)
    except Exception:
      pass

  def get_model_info(self, model: str) -> ModelInfo:
    model_id = str(model or "").strip()
    if not model_id:
      raise ValueError("Model is required")
    for tags, info in _MODEL_INFO_BY_TAG:
      if any(_model_matches_tag(model_id, tag) for tag in tags):
        return replace(info, id=model_id)
    if "gpt-5" in model_id:
      return ModelInfo(
        id=model_id,
        provider=self.name,
        context_window=272_000,
        max_output_tokens=128_000,
        supports_thinking=True,
        supports_vision=True,
      )
    return ModelInfo(id=model_id, provider=self.name)

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    thinking_level: ThinkingLevel = ThinkingLevel.HIGH,
    **kwargs: Any,
  ) -> dict[str, Any]:
    model_info = self.get_model_info(model)
    normalized_messages = self.normalize_messages(messages, model_info)

    params: dict[str, Any] = {
      "model": model,
      "store": False,
      "stream": True,
      "input": _convert_messages(normalized_messages, model_info),
      "text": {"verbosity": str(kwargs.get("text_verbosity") or "medium")},
      "include": ["reasoning.encrypted_content"],
      "tool_choice": "auto",
      "parallel_tool_calls": True,
    }

    system_text = _system_prompt_text(system_prompt).strip()
    params["instructions"] = _sanitize_surrogates(system_text or DEFAULT_INSTRUCTIONS)

    session_id = str(kwargs.get("session_id") or kwargs.get("_session_id") or "")
    if session_id:
      params["prompt_cache_key"] = session_id
      params["_session_id"] = session_id

    headers = kwargs.get("headers")
    if isinstance(headers, dict) and headers:
      params["_headers"] = dict(headers)

    base_url = kwargs.get("base_url")
    if base_url:
      params["_endpoint_url"] = _resolve_codex_url(str(base_url))

    if tools:
      params["tools"] = _convert_tools(tools, strict=None)

    reasoning_effort = _map_reasoning_effort(thinking_level)
    if model_info.supports_thinking and reasoning_effort is not None:
      params["reasoning"] = {
        "effort": _clamp_reasoning_effort(model, reasoning_effort),
        "summary": str(kwargs.get("reasoning_summary") or "auto"),
      }

    return params

  def normalize_messages(self, messages: list[dict[str, Any]], model_info: ModelInfo) -> list[dict[str, Any]]:
    tool_call_id_map: dict[str, str] = {}
    transformed: list[dict[str, Any]] = []

    for message in messages:
      role = message.get("role")
      if role == "assistant":
        if str(message.get("stop_reason", "")) in {"error", "aborted"}:
          continue

        is_same_model = _same_model_message(message, model_info)
        content = message.get("content")
        if not isinstance(content, list):
          transformed.append(dict(message))
          continue

        next_content: list[dict[str, Any]] = []
        for block in content:
          if not isinstance(block, dict):
            continue
          block_type = block.get("type")
          if block_type == "thinking":
            if block.get("redacted"):
              if is_same_model:
                next_content.append(dict(block))
              continue
            thinking_text = str(block.get("thinking", ""))
            thinking_signature = str(block.get("thinkingSignature") or block.get("signature") or "")
            if is_same_model and thinking_signature:
              next_block = dict(block)
              next_block["thinkingSignature"] = thinking_signature
              next_content.append(next_block)
              continue
            if not thinking_text.strip():
              continue
            if is_same_model:
              next_content.append(dict(block))
            else:
              next_content.append({"type": "text", "text": thinking_text})
            continue
          if block_type == "text":
            if is_same_model:
              next_content.append(dict(block))
            else:
              next_content.append({"type": "text", "text": str(block.get("text", ""))})
            continue
          if block_type in {"tool_use", "server_tool_use"}:
            next_block = dict(block)
            if not is_same_model:
              next_block.pop("thought_signature", None)
              next_block.pop("thoughtSignature", None)
              original_id = str(block.get("id") or "")
              normalized_id = _normalize_responses_tool_call_id(original_id)
              if normalized_id and normalized_id != original_id:
                tool_call_id_map[original_id] = normalized_id
                next_block["id"] = normalized_id
            next_content.append(next_block)
            continue
          next_content.append(dict(block))

        next_message = dict(message)
        next_message["content"] = next_content
        transformed.append(next_message)
        continue

      if role == "user":
        content = message.get("content")
        if isinstance(content, list):
          next_message = dict(message)
          next_content: list[dict[str, Any]] = []
          for block in content:
            if not isinstance(block, dict):
              continue
            if block.get("type") == "tool_result":
              next_block = dict(block)
              tool_use_id = str(block.get("tool_use_id") or "")
              normalized_id = tool_call_id_map.get(tool_use_id)
              if normalized_id:
                next_block["tool_use_id"] = normalized_id
              next_content.append(next_block)
            else:
              next_content.append(dict(block))
          next_message["content"] = next_content
          transformed.append(next_message)
          continue

      transformed.append(dict(message))

    result: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []
    existing_tool_result_ids: set[str] = set()

    for message in transformed:
      role = message.get("role")
      if role == "assistant":
        if pending_tool_calls:
          for tool_call in pending_tool_calls:
            tool_id = str(tool_call.get("id") or "")
            if tool_id and tool_id not in existing_tool_result_ids:
              result.append(
                {
                  "role": "user",
                  "content": [_synthetic_tool_result(tool_id, str(tool_call.get("name") or ""))],
                }
              )
          pending_tool_calls = []
          existing_tool_result_ids = set()

        content = message.get("content")
        if isinstance(content, list):
          pending_tool_calls = [
            dict(block)
            for block in content
            if isinstance(block, dict) and block.get("type") in {"tool_use", "server_tool_use"}
          ]
        else:
          pending_tool_calls = []
        result.append(message)
        continue

      if role == "user":
        content = message.get("content")
        tool_result_blocks = [
          block
          for block in content
          if isinstance(block, dict) and block.get("type") == "tool_result"
        ] if isinstance(content, list) else []
        if pending_tool_calls and tool_result_blocks:
          existing_tool_result_ids = {str(block.get("tool_use_id") or "") for block in tool_result_blocks}
          missing = [
            _synthetic_tool_result(str(tool_call.get("id") or ""), str(tool_call.get("name") or ""))
            for tool_call in pending_tool_calls
            if str(tool_call.get("id") or "") not in existing_tool_result_ids
          ]
          if missing:
            next_message = dict(message)
            next_message["content"] = [*list(content or []), *missing]
            result.append(next_message)
          else:
            result.append(message)
          pending_tool_calls = []
          existing_tool_result_ids = set()
          continue
        if pending_tool_calls:
          for tool_call in pending_tool_calls:
            tool_id = str(tool_call.get("id") or "")
            if tool_id and tool_id not in existing_tool_result_ids:
              result.append(
                {
                  "role": "user",
                  "content": [_synthetic_tool_result(tool_id, str(tool_call.get("name") or ""))],
                }
              )
          pending_tool_calls = []
          existing_tool_result_ids = set()
        result.append(message)
        continue

      result.append(message)

    return truncate_to_last_compaction(result, compaction_as_text=True)

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
    if not isinstance(client, httpx.AsyncClient):
      raise RuntimeError("CodexProvider requires an httpx.AsyncClient")

    state = self._client_state.get(client)
    if state is None:
      raise RuntimeError("Codex client state is missing")

    request_params = dict(params)
    session_id = str(request_params.pop("_session_id", "") or "")
    additional_headers = request_params.pop("_headers", None)
    endpoint_url = str(request_params.pop("_endpoint_url", state["endpoint_url"]))

    headers = _build_headers(
      state.get("init_headers"),
      additional_headers if isinstance(additional_headers, dict) else None,
      str(state["account_id"]),
      str(state["token"]),
      session_id or None,
    )

    async with client.stream("POST", endpoint_url, json=request_params, headers=headers) as response:
      if response.status_code != 200:
        info = await _parse_error_response(response)
        raise httpx.HTTPStatusError(str(info["message"] or "Request failed"), request=response.request, response=response)

      stream_state = _ResponsesStreamState()
      buffer = ""
      async for chunk in response.aiter_text():
        buffer += chunk
        events, buffer = _parse_sse(buffer)
        for event in events:
          for mapped_event in _map_event(event, stream_state):
            yield mapped_event

  def is_retryable_error(self, exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
      return exc.response.status_code in _RETRYABLE_STATUSES
    if isinstance(exc, (httpx.TransportError, httpx.StreamError)):
      return True
    return bool(_RETRYABLE_RE.search(str(exc)))


__all__ = [
  "CodexProvider",
  "_build_headers",
  "_convert_messages",
  "_convert_tools",
  "_extract_account_id",
  "_map_event",
  "_parse_sse",
  "_resolve_codex_url",
]
