from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..openai_history_fence import REASONING_SIGNATURE_MARKER, TEXT_SIGNATURE_MARKER
from .base import ModelInfo, StreamEvent


_TOOL_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_MAX_TOOL_ID_LEN = 64
_SURROGATE_RE = re.compile(r"[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]")
_RESPONSE_STATUSES = frozenset({"completed", "incomplete", "failed", "cancelled", "queued", "in_progress"})


def _responses_compat(
  *,
  effort_values: tuple[str, ...] = (),
  effort_default: str = "none",
  summary: bool = False,
  function_tools: bool = True,
) -> dict[str, Any]:
  return {
    "supportsResponsesStreaming": True,
    "supportsResponsesFunctionTools": function_tools,
    "supportsResponsesReasoningSummary": summary,
    "supportsReasoningEffort": bool(effort_values),
    "reasoningEffortValues": effort_values,
    "reasoningEffortDefault": effort_default,
  }


_GPT56_VALUES = ("none", "low", "medium", "high", "xhigh", "max")
_GPT55_VALUES = ("none", "low", "medium", "high", "xhigh")

_MODEL_INFO_BY_TAG: list[tuple[tuple[str, ...], ModelInfo]] = [
  *[
    (
      (model_id,),
      ModelInfo(
        id=model_id,
        provider="openai",
        context_window=1_050_000,
        max_output_tokens=128_000,
        supports_thinking=True,
        supports_vision=True,
        input_cost_per_mtok=input_cost,
        output_cost_per_mtok=output_cost,
        cache_read_cost_per_mtok=cache_cost,
        compat=_responses_compat(effort_values=_GPT56_VALUES, effort_default="medium", summary=True),
      ),
    )
    for model_id, input_cost, output_cost, cache_cost in (
      ("gpt-5.6-sol", 5.00, 30.00, 0.50),
      ("gpt-5.6-terra", 2.50, 15.00, 0.25),
      ("gpt-5.6-luna", 1.00, 6.00, 0.10),
      ("gpt-5.6", 5.00, 30.00, 0.50),
    )
  ],
  (
    ("gpt-5.5",),
    ModelInfo(
      id="gpt-5.5",
      provider="openai",
      context_window=1_050_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
      input_cost_per_mtok=5.00,
      output_cost_per_mtok=30.00,
      cache_read_cost_per_mtok=0.50,
      compat=_responses_compat(effort_values=_GPT55_VALUES, effort_default="medium", summary=True),
    ),
  ),
  (
    ("gpt-5.4",),
    ModelInfo(
      id="gpt-5.4",
      provider="openai",
      context_window=1_050_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
      input_cost_per_mtok=2.50,
      output_cost_per_mtok=15.00,
      cache_read_cost_per_mtok=0.25,
      compat=_responses_compat(effort_values=_GPT55_VALUES, effort_default="none", summary=True),
    ),
  ),
]


def _field(value: Any, name: str, default: Any = None) -> Any:
  if value is None:
    return default
  if isinstance(value, dict):
    return value.get(name, default)
  return getattr(value, name, default)


def _to_plain_dict(value: Any) -> Any:
  if value is None:
    return None
  if hasattr(value, "model_dump"):
    return _to_plain_dict(value.model_dump())
  if isinstance(value, dict):
    return {str(key): _to_plain_dict(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_to_plain_dict(item) for item in value]
  if hasattr(value, "__dict__"):
    return {key: _to_plain_dict(item) for key, item in vars(value).items() if not key.startswith("_")}
  return value


def _json_dumps(value: Any) -> str:
  return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _sanitize_text(value: Any) -> str:
  return _SURROGATE_RE.sub("", str(value or ""))


def _model_matches_tag(model_id: str, tag: str) -> bool:
  candidates = (model_id, model_id.rsplit("/", 1)[-1])
  return any(candidate == tag or candidate.startswith(f"{tag}-") for candidate in candidates)


def _normalize_id(value: Any, *, fallback: str = "call") -> str:
  normalized = _TOOL_ID_RE.sub("_", str(value or "").strip()).strip("_") or fallback
  return normalized[:_MAX_TOOL_ID_LEN]


def _normalize_tool_call_id(tool_id: str) -> str:
  call_id, separator, item_id = str(tool_id or "").partition("|")
  normalized_call = _normalize_id(call_id)
  if not separator or not item_id:
    return normalized_call
  normalized_item = _normalize_id(item_id, fallback="fc")
  return f"{normalized_call}|{normalized_item}"


def _same_model_message(message: dict[str, Any], model_info: ModelInfo) -> bool:
  return str(message.get("provider") or "") == model_info.provider and str(message.get("model") or "") == model_info.id


def _system_prompt_text(system_prompt: str | list[tuple[str, bool]] | None) -> str:
  if system_prompt is None:
    return ""
  if isinstance(system_prompt, list):
    return "\n\n".join(text for text, _cache in system_prompt if text)
  return str(system_prompt)


def _stringify_tool_result_content(content: Any) -> str:
  if isinstance(content, str):
    return _sanitize_text(content)
  if isinstance(content, list):
    texts = [str(block.get("text", "")) for block in content if isinstance(block, dict) and block.get("type") == "text"]
    return _sanitize_text("\n".join(texts) if texts else "(see attached image)")
  if isinstance(content, (dict, list)):
    return _sanitize_text(json.dumps(content, default=str))
  return _sanitize_text(content)


def _synthetic_tool_result(tool_id: str, tool_name: str) -> dict[str, Any]:
  return {
    "type": "tool_result",
    "tool_use_id": tool_id,
    "content": json.dumps({"error": {"code": "missing_tool_result", "message": "No result provided"}}),
    "is_error": True,
    "tool_name": tool_name,
  }


def _is_tool_result_message(message: dict[str, Any]) -> bool:
  content = message.get("content")
  return (
    message.get("role") == "user"
    and isinstance(content, list)
    and bool(content)
    and all(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)
  )


def _contains_tool_history(messages: list[dict[str, Any]]) -> bool:
  return any(
    isinstance(block, dict) and block.get("type") in {"tool_use", "server_tool_use", "tool_result"}
    for message in messages
    for block in (message.get("content") if isinstance(message.get("content"), list) else [])
  )


def convert_openai_response_tools(
  tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  """Convert gateway tool definitions to exact OpenAI Responses wire tools."""
  return [
    {
      "type": "function",
      "name": str(tool.get("name") or ""),
      "description": str(tool.get("description") or ""),
      "parameters": tool.get("input_schema") or tool.get("parameters") or {"type": "object", "properties": {}},
      "strict": False,
    }
    for tool in tools
  ]


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Backward-compatible private seam used by existing provider callers."""
  return convert_openai_response_tools(tools)


def _parse_signature(signature: Any, marker: str) -> dict[str, Any] | None:
  try:
    payload = json.loads(str(signature or ""))
  except (json.JSONDecodeError, TypeError, ValueError):
    return None
  if not isinstance(payload, dict) or payload.get("v") != marker:
    return None
  return payload


def _encode_text_signature(item: dict[str, Any]) -> str:
  payload: dict[str, Any] = {
    "v": TEXT_SIGNATURE_MARKER,
    "item_id": str(item.get("id") or ""),
    "status": str(item.get("status") or "completed"),
  }
  phase = item.get("phase")
  if phase in {"commentary", "final_answer"}:
    payload["phase"] = phase
  return _json_dumps(payload)


def _validated_text_identity(signature: Any) -> dict[str, str] | None:
  payload = _parse_signature(signature, TEXT_SIGNATURE_MARKER)
  if payload is None:
    return None
  item_id = payload.get("item_id")
  status = payload.get("status")
  if not isinstance(item_id, str) or not item_id or status not in _RESPONSE_STATUSES:
    return None
  result = {"id": item_id, "status": str(status)}
  if payload.get("phase") in {"commentary", "final_answer"}:
    result["phase"] = str(payload["phase"])
  return result


def _encode_reasoning_signature(item: dict[str, Any]) -> str:
  validated = _validated_reasoning_item(item)
  if validated is None:
    return ""
  return _json_dumps({"v": REASONING_SIGNATURE_MARKER, "item": validated})


def _validated_reasoning_item(value: Any) -> dict[str, Any] | None:
  if not isinstance(value, dict) or value.get("type") != "reasoning":
    return None
  item_id = value.get("id")
  if not isinstance(item_id, str) or not item_id:
    return None
  result: dict[str, Any] = {"type": "reasoning", "id": item_id}
  status = value.get("status")
  if status in _RESPONSE_STATUSES:
    result["status"] = status
  encrypted = value.get("encrypted_content")
  if isinstance(encrypted, str) and encrypted:
    result["encrypted_content"] = encrypted
  summary = value.get("summary")
  if isinstance(summary, list):
    safe_summary = []
    for part in summary:
      if isinstance(part, dict) and part.get("type") == "summary_text" and isinstance(part.get("text"), str):
        safe_summary.append({"type": "summary_text", "text": _sanitize_text(part["text"])})
    result["summary"] = safe_summary
  return result


def _reasoning_from_signature(signature: Any) -> dict[str, Any] | None:
  payload = _parse_signature(signature, REASONING_SIGNATURE_MARKER)
  return _validated_reasoning_item(payload.get("item")) if payload is not None else None


def _convert_messages(messages: list[dict[str, Any]], model_info: ModelInfo) -> list[dict[str, Any]]:
  converted: list[dict[str, Any]] = []
  for message in messages:
    role = message.get("role")
    content = message.get("content")
    if role == "user":
      if isinstance(content, str):
        converted.append({"role": "user", "content": [{"type": "input_text", "text": _sanitize_text(content)}]})
        continue
      if not isinstance(content, list):
        if content is not None:
          converted.append({"role": "user", "content": [{"type": "input_text", "text": _sanitize_text(content)}]})
        continue
      user_parts: list[dict[str, Any]] = []
      for block in content:
        if not isinstance(block, dict):
          continue
        if block.get("type") == "text":
          user_parts.append({"type": "input_text", "text": _sanitize_text(block.get("text"))})
        elif block.get("type") == "image" and model_info.supports_vision:
          media_type = block.get("media_type") or block.get("mime_type") or block.get("mimeType")
          data = block.get("data_base64") or block.get("data")
          if media_type and data:
            user_parts.append({"type": "input_image", "detail": "auto", "image_url": f"data:{media_type};base64,{data}"})
        elif block.get("type") == "tool_result":
          if user_parts:
            converted.append({"role": "user", "content": user_parts})
            user_parts = []
          call_id = _normalize_tool_call_id(str(block.get("tool_use_id") or "")).split("|", 1)[0]
          converted.append({
            "type": "function_call_output",
            "call_id": call_id,
            "output": _stringify_tool_result_content(block.get("content")),
          })
      if user_parts:
        converted.append({"role": "user", "content": user_parts})
      continue

    if role == "assistant":
      if not isinstance(content, list):
        text = _sanitize_text(content)
        if text:
          converted.append({"role": "assistant", "content": text})
        continue
      for block in content:
        if not isinstance(block, dict):
          continue
        block_type = block.get("type")
        if block_type == "thinking":
          if _same_model_message(message, model_info):
            reasoning = _reasoning_from_signature(block.get("signature") or block.get("thinkingSignature"))
            if reasoning is not None:
              converted.append(reasoning)
              continue
          thinking = _sanitize_text(block.get("thinking"))
          if thinking and not model_info.supports_thinking:
            converted.append({"role": "assistant", "content": thinking})
        elif block_type == "text":
          text = _sanitize_text(block.get("text"))
          if not text:
            continue
          identity = _validated_text_identity(block.get("textSignature") or block.get("signature"))
          if identity is None:
            converted.append({"role": "assistant", "content": text})
          else:
            item: dict[str, Any] = {
              "type": "message",
              "role": "assistant",
              "id": identity["id"],
              "status": identity["status"],
              "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
            if "phase" in identity:
              item["phase"] = identity["phase"]
            converted.append(item)
        elif block_type in {"tool_use", "server_tool_use"}:
          tool_id = _normalize_tool_call_id(str(block.get("id") or ""))
          call_id, separator, item_id = tool_id.partition("|")
          item = {
            "type": "function_call",
            "call_id": call_id,
            "name": str(block.get("name") or ""),
            "arguments": json.dumps(block.get("input") or {}, default=str),
          }
          if separator and item_id:
            item["id"] = item_id
          converted.append(item)
      continue

    if role == "tool":
      call_id = _normalize_tool_call_id(str(message.get("tool_call_id") or "")).split("|", 1)[0]
      converted.append({
        "type": "function_call_output",
        "call_id": call_id,
        "output": _stringify_tool_result_content(content),
      })
  return converted


@dataclass
class _ResponsesStreamState:
  message_started: bool = False
  provider_reported_model: str | None = None
  current_item: dict[str, Any] | None = None
  current_block_type: str | None = None
  current_text: str = ""
  current_thinking: str = ""
  current_tool_json: str = ""
  saw_argument_delta: bool = False
  saw_tool_use: bool = False
  terminal_emitted: bool = False
  terminal_error: RuntimeError | None = None
  reported_output_tokens: int = 0
  reported_reasoning_tokens: int = 0


def _parse_tool_input(raw: str) -> dict[str, Any]:
  try:
    parsed = json.loads(raw or "{}")
  except json.JSONDecodeError:
    return {}
  return parsed if isinstance(parsed, dict) else {}


def _redacted_tool_input(name: str, value: dict[str, Any]) -> dict[str, Any]:
  try:
    from agent.shared.tool_redaction import get_audit_hmac_secret, redact_tool_input
    return redact_tool_input(name, value, deployment_secret=get_audit_hmac_secret())
  except Exception:
    from ..secret_boundary import sanitization_failure_tool_input

    return sanitization_failure_tool_input()


def _terminal_events(response: dict[str, Any], state: _ResponsesStreamState) -> list[StreamEvent]:
  if state.terminal_emitted:
    return []
  state.terminal_emitted = True
  status = str(response.get("status") or "completed")
  reported_model = str(response.get("model") or "").strip() or None
  if reported_model is not None:
    state.provider_reported_model = reported_model
  usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
  input_details = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), dict) else {}
  output_details = usage.get("output_tokens_details") if isinstance(usage.get("output_tokens_details"), dict) else {}
  cached = int(input_details.get("cached_tokens") or 0)
  total_input = int(usage.get("input_tokens") or 0)
  total_output = int(usage.get("output_tokens") or 0)
  total_reasoning = int(output_details.get("reasoning_tokens") or 0)
  events: list[StreamEvent] = []
  if not state.message_started:
    state.message_started = True
    events.append(StreamEvent(
      type="message_start",
      provider_reported_model=state.provider_reported_model,
      input_tokens=max(0, total_input - cached),
      cache_read_tokens=cached,
    ))
  output_delta = max(0, total_output - state.reported_output_tokens)
  reasoning_delta = max(0, total_reasoning - state.reported_reasoning_tokens)
  if output_delta or reasoning_delta:
    state.reported_output_tokens = total_output
    state.reported_reasoning_tokens = total_reasoning
    events.append(StreamEvent(type="usage_update", output_tokens=output_delta, reasoning_tokens=reasoning_delta))
  if state.saw_tool_use:
    stop_reason = "tool_use"
  elif status == "incomplete":
    details = response.get("incomplete_details") if isinstance(response.get("incomplete_details"), dict) else {}
    stop_reason = "max_tokens" if details.get("reason") == "max_output_tokens" else "error"
  elif status in {"failed", "cancelled"}:
    stop_reason = "error"
  else:
    stop_reason = "end_turn"
  events.append(StreamEvent(type="message_end", stop_reason=stop_reason))
  return events


def map_event(event_value: Any, state: _ResponsesStreamState) -> list[StreamEvent]:
  event = _to_plain_dict(event_value)
  if not isinstance(event, dict):
    return []
  event_type = event.get("type")
  response = event.get("response") if isinstance(event.get("response"), dict) else {}
  reported_model = str(response.get("model") or "").strip() or None
  if reported_model is not None:
    state.provider_reported_model = reported_model
  if event_type == "error":
    raise RuntimeError(f"OpenAI Responses error: {event.get('message') or event.get('code') or 'unknown error'}")
  if event_type == "response.failed":
    response = event.get("response") if isinstance(event.get("response"), dict) else {}
    error = response.get("error") if isinstance(response.get("error"), dict) else {}
    if "status" not in response:
      response = {**response, "status": "failed"}
    terminal_events = _terminal_events(response, state)
    state.terminal_error = RuntimeError(
      f"OpenAI Responses failed: {error.get('message') or 'unknown error'}"
    )
    return terminal_events
  if event_type in {"response.completed", "response.incomplete", "response.done"}:
    response = event.get("response") if isinstance(event.get("response"), dict) else {}
    if event_type == "response.incomplete" and "status" not in response:
      response = {**response, "status": "incomplete"}
    return _terminal_events(response, state)

  if event_type == "response.output_item.added":
    item = event.get("item") if isinstance(event.get("item"), dict) else {}
    state.current_item = dict(item)
    item_type = item.get("type")
    if item_type == "reasoning":
      state.current_block_type = "thinking"
      state.current_thinking = ""
    elif item_type == "message":
      state.current_block_type = "text"
      state.current_text = ""
    elif item_type == "function_call":
      state.current_block_type = "tool_use"
      state.current_tool_json = str(item.get("arguments") or "")
      state.saw_argument_delta = False
      state.saw_tool_use = True
      call_id = _normalize_id(item.get("call_id"))
      item_id = _normalize_id(item.get("id"), fallback="fc") if item.get("id") else ""
      tool_id = f"{call_id}|{item_id}" if item_id else call_id
      return [StreamEvent(
        type="tool_use_start", tool_id=tool_id, tool_name=str(item.get("name") or ""),
        raw_block={"type": "tool_use", "id": tool_id, "name": str(item.get("name") or "")},
      )]
    return []

  if event_type == "response.reasoning_summary_part.added":
    separator: list[StreamEvent] = []
    if state.current_item is not None and state.current_item.get("type") == "reasoning":
      part = event.get("part")
      if isinstance(part, dict):
        summary = state.current_item.setdefault("summary", [])
        if isinstance(summary, list):
          # The separator belongs BETWEEN parts, mirroring the "\n\n".join() that
          # rebuilds the durable thinking block at response.output_item.done.
          # Emitting it on part.done instead appends one after the FINAL part, so
          # the client stream ends "...second\n\n" while the durable block ends
          # "...second" -- the two representations disagree.
          if summary and state.current_block_type == "thinking":
            state.current_thinking += "\n\n"
            separator.append(StreamEvent(type="thinking_delta", thinking_text="\n\n"))
          summary.append(dict(part))
    return separator
  if event_type == "response.reasoning_summary_text.delta" and state.current_block_type == "thinking":
    delta = _sanitize_text(event.get("delta"))
    state.current_thinking += delta
    return [StreamEvent(type="thinking_delta", thinking_text=delta)] if delta else []
  if event_type == "response.reasoning_summary_part.done":
    # Separator is emitted on the NEXT part's .added event, not here -- see above.
    return []
  if event_type == "response.reasoning_summary_text.done":
    return []
  if event_type == "response.content_part.added":
    if state.current_item is not None and state.current_item.get("type") == "message":
      part = event.get("part")
      if isinstance(part, dict):
        content = state.current_item.setdefault("content", [])
        if isinstance(content, list):
          content.append(dict(part))
    return []
  if event_type in {"response.output_text.delta", "response.refusal.delta"} and state.current_block_type == "text":
    delta = _sanitize_text(event.get("delta"))
    state.current_text += delta
    return [StreamEvent(type="text_delta", text=delta)] if delta else []
  if event_type == "response.function_call_arguments.delta" and state.current_block_type == "tool_use":
    delta = str(event.get("delta") or "")
    # A complete snapshot may be present in output_item.added. Separate deltas
    # are authoritative, so replace that snapshot on the first delta.
    # `delta and` guards an empty delta from discarding a valid seeded snapshot.
    if delta and not state.saw_argument_delta:
      state.current_tool_json = ""
      state.saw_argument_delta = True
    state.current_tool_json += delta
    return [StreamEvent(type="tool_use_delta", tool_input_json=delta)] if delta else []
  if event_type == "response.function_call_arguments.done" and state.current_block_type == "tool_use":
    if isinstance(event.get("arguments"), str):
      state.current_tool_json = event["arguments"]
    return []
  if event_type != "response.output_item.done":
    return []

  item = event.get("item") if isinstance(event.get("item"), dict) else {}
  item_type = item.get("type")
  if item_type == "reasoning":
    summary = item.get("summary") if isinstance(item.get("summary"), list) else []
    thinking = "\n\n".join(str(part.get("text") or "") for part in summary if isinstance(part, dict))
    signature = _encode_reasoning_signature(item)
    state.current_block_type = None
    state.current_item = None
    if not signature:
      return []
    return [StreamEvent(
      type="thinking_end", thinking_text=thinking, signature=signature,
      raw_block={"type": "thinking", "thinking": thinking, "signature": signature, "thinkingSignature": signature},
    )]
  if item_type == "message":
    content = item.get("content") if isinstance(item.get("content"), list) else []
    text = "".join(
      str(part.get("text") if part.get("type") == "output_text" else part.get("refusal") or "")
      for part in content if isinstance(part, dict)
    ) or state.current_text
    signature = _encode_text_signature(item)
    state.current_block_type = None
    state.current_item = None
    return [StreamEvent(type="text_end", text=text, raw_block={"type": "text", "text": text, "textSignature": signature})]
  if item_type == "function_call":
    arguments = str(item.get("arguments") or state.current_tool_json or "{}")
    tool_input = _parse_tool_input(arguments)
    call_id = _normalize_id(item.get("call_id"))
    item_id = _normalize_id(item.get("id"), fallback="fc") if item.get("id") else ""
    tool_id = f"{call_id}|{item_id}" if item_id else call_id
    tool_name = str(item.get("name") or "")
    done_events: list[StreamEvent] = []
    if state.current_block_type != "tool_use":
      state.saw_tool_use = True
      done_events.append(StreamEvent(
        type="tool_use_start", tool_id=tool_id, tool_name=tool_name,
        raw_block={"type": "tool_use", "id": tool_id, "name": tool_name},
      ))
    if not state.saw_argument_delta and arguments:
      done_events.append(StreamEvent(type="tool_use_delta", tool_input_json=arguments))
    state.current_block_type = None
    state.current_item = None
    state.current_tool_json = ""
    done_events.append(StreamEvent(
      type="tool_use_end", tool_id=tool_id, tool_name=tool_name, tool_input_json=arguments,
      tool_input=tool_input,
      raw_block={"type": "tool_use", "id": tool_id, "name": tool_name, "input": _redacted_tool_input(tool_name, tool_input)},
    ))
    state.saw_argument_delta = False
    return done_events
  return []


__all__ = [
  "_MODEL_INFO_BY_TAG",
  "_ResponsesStreamState",
  "_contains_tool_history",
  "_convert_messages",
  "_convert_tools",
  "_field",
  "_is_tool_result_message",
  "_model_matches_tag",
  "_normalize_tool_call_id",
  "_responses_compat",
  "_same_model_message",
  "_stringify_tool_result_content",
  "_synthetic_tool_result",
  "_system_prompt_text",
  "_to_plain_dict",
  "map_event",
]
