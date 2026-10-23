from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, AsyncGenerator
from weakref import WeakKeyDictionary

import httpx

from model_authority.rates import RateTable
from model_authority.registry import AdapterRouteSupport
from model_authority.schema import OpenAIChatCompletionsCompat
from model_authority.thinking import EffortResolution, ThinkingLevel
from ..auth import CredentialFailureKind, ProviderCredentialFailure
from .base import (
  ModelInfo,
  ModelProvider,
  StreamEvent,
  _classify_provider_credential_failure,
  _error_code_from_exception,
  _is_context_length_exception,
  _response_text,
  _status_code_from_exception,
  admitted_entry,
  authority_rate_table,
  model_info_from_entry,
  truncate_to_last_compaction,
)

log = logging.getLogger(__name__)

# Every request goes to the local daemon. It serves two routes: `ollama.local`
# (inference on this host) and `ollama.daemon_cloud` (`<name>:<tag>` values whose
# tag is `cloud` or ends in `-cloud`, proxied to ollama.com after `ollama signin`).
# https://docs.ollama.com/api/authentication
_DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
_CLOUD_TAG_RE = re.compile(r"(?:-cloud|:cloud)\b", re.IGNORECASE)

_RETRYABLE_STATUSES = frozenset({408, 409})
_USAGE_LIMIT_RE = re.compile(
  r"\b(?:rate(?:\s+limit(?:ed|ing)?|\s+limited)|too many requests"
  r"|usage(?:\s+limit|\s+exhausted)|quota(?:\s+exhausted|\s+exceeded))\b",
  re.IGNORECASE,
)
_CONTENT_REFUSAL_RE = re.compile(
  r"\b(?:content[_\s-]*filter|content[_\s-]*policy|safety[_\s-]*violation|refused to)\b",
  re.IGNORECASE,
)
_BILLING_NOT_USAGE_RE = re.compile(
  r"\b(?:payment|credit(?:s)?|subscription|billing|insufficient[_\s-]+quota)\b",
  re.IGNORECASE,
)
_SURROGATE_RE = re.compile(
  r"[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]"
)
_FINISH_STOP_REASONS = {
  "stop": "end_turn",
  "length": "max_tokens",
  "tool_calls": "tool_use",
  "function_call": "tool_use",
  "content_filter": "refusal",
}


def _chat_completions_url(base_url: str) -> str:
  return f"{str(base_url).strip().rstrip('/')}/chat/completions"


def _sanitize_text(value: Any) -> str:
  return _SURROGATE_RE.sub("", str(value or ""))


def _json_dumps(value: Any) -> str:
  return json.dumps(value, default=str, ensure_ascii=False)


def _system_prompt_text(system_prompt: str | list[tuple[str, bool]] | None) -> str:
  if system_prompt is None:
    return ""
  if isinstance(system_prompt, list):
    return "\n\n".join(text for text, _cache in system_prompt if text)
  return str(system_prompt)


def _stringify_tool_result(content: Any) -> str:
  if isinstance(content, str):
    return _sanitize_text(content)
  if isinstance(content, list):
    texts = [
      str(block.get("text") or "")
      for block in content
      if isinstance(block, dict) and block.get("type") == "text"
    ]
    if texts:
      return _sanitize_text("\n".join(texts))
    return _sanitize_text(_json_dumps(content))
  if isinstance(content, dict):
    return _sanitize_text(_json_dumps(content))
  return _sanitize_text(content)


def _chat_compat(model_info: ModelInfo) -> OpenAIChatCompletionsCompat:
  compat = model_info.compat
  if not isinstance(compat, OpenAIChatCompletionsCompat):
    raise TypeError(
      f"model {model_info.id!r} carries no openai.chat_completions compat"
    )
  return compat


def _rate_table_for(provider: str, override: RateTable | None) -> RateTable:
  """This provider's authority table, or an empty table when the data slice has not priced it.

  A missing file is not a price. Lookup then takes the same zero-cost path a
  model with no row already takes.
  """

  try:
    return authority_rate_table(provider, override)
  except ValueError:
    if override is not None and override.providers.get(provider):
      return override
    log.warning("%s has no authority rate table; cost estimates stay zero", provider)
    return RateTable(
      version="0",
      source="absent",
      providers={provider: {}},
      _path=Path(provider),
    )


def _request_temperature(value: Any) -> float | None:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return None
  return float(value)


def _reasoning_fragment(param: str, value: str) -> dict[str, Any]:
  *parents, leaf = param.split(".")
  if not leaf:
    return {}
  fragment: dict[str, Any] = {leaf: value}
  for parent in reversed(parents):
    if not parent:
      continue
    fragment = {parent: fragment}
  return fragment


def _tool_function(tool: dict[str, Any]) -> dict[str, Any] | None:
  if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
    function = tool["function"]
    name = str(function.get("name") or "").strip()
    if not name:
      return None
    parameters = function.get("parameters") or {"type": "object", "properties": {}}
    return {
      "name": name,
      "description": str(function.get("description") or ""),
      "parameters": parameters,
    }
  name = str(tool.get("name") or "").strip()
  if not name:
    return None
  return {
    "name": name,
    "description": str(tool.get("description") or ""),
    "parameters": (
      tool.get("input_schema")
      or tool.get("parameters")
      or {"type": "object", "properties": {}}
    ),
  }


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
  converted: list[dict[str, Any]] = []
  for tool in tools:
    if not isinstance(tool, dict):
      continue
    function = _tool_function(tool)
    if function is None:
      continue
    converted.append({"type": "function", "function": function})
  return converted


def _named_tool_choice(value: Any) -> str | None:
  if isinstance(value, str):
    name = value.strip()
    if not name or name.lower() in {"auto", "none", "required", "any"}:
      return None
    return name
  if not isinstance(value, dict):
    return None
  function = value.get("function")
  if isinstance(function, dict):
    name = str(function.get("name") or "").strip()
    if name:
      return name
  name = str(value.get("name") or "").strip()
  return name or None


def _tool_choice_mode(value: Any) -> str:
  """``omit``, ``none``, ``required``, or ``named``.

  The chat-completions shim accepts only ``none`` and ``required``. ``auto``
  is omitted; a named choice is applied by filtering tools.
  """

  if value is None:
    return "omit"
  if isinstance(value, str):
    token = value.strip().lower()
    if token in {"", "auto"}:
      return "omit"
    if token == "none":
      return "none"
    if token in {"required", "any"}:
      return "required"
    return "named"
  if isinstance(value, dict):
    kind = str(value.get("type") or "").strip().lower()
    if kind == "none":
      return "none"
    if kind in {"required", "any"}:
      return "required"
    if kind in {"", "auto"} and _named_tool_choice(value) is None:
      return "omit"
    if _named_tool_choice(value):
      return "named"
  return "omit"


def _image_url(block: dict[str, Any]) -> str | None:
  direct = block.get("image_url") or block.get("url")
  if isinstance(direct, str) and direct.strip():
    return direct.strip()
  if isinstance(direct, dict):
    url = direct.get("url")
    if isinstance(url, str) and url.strip():
      return url.strip()
  data = block.get("data_base64") or block.get("data")
  if not isinstance(data, str) or not data.strip():
    return None
  payload = data.strip()
  if payload.startswith("data:"):
    return payload
  media_type = str(block.get("media_type") or block.get("mime_type") or "image/png")
  return f"data:{media_type};base64,{payload}"


def _assistant_wire(content: Any, *, replay_reasoning: bool) -> dict[str, Any]:
  text_parts: list[str] = []
  thinking_parts: list[str] = []
  tool_calls: list[dict[str, Any]] = []
  blocks = content if isinstance(content, list) else None
  if blocks is None:
    text = _sanitize_text(content)
    return {"role": "assistant", "content": text}
  for block in blocks:
    if not isinstance(block, dict):
      continue
    block_type = block.get("type")
    if block_type == "text":
      text = _sanitize_text(block.get("text"))
      if text:
        text_parts.append(text)
    elif block_type == "thinking":
      thinking = _sanitize_text(block.get("thinking"))
      if thinking:
        thinking_parts.append(thinking)
    elif block_type in {"tool_use", "server_tool_use"}:
      name = str(block.get("name") or "").strip()
      tool_id = str(block.get("id") or "").strip()
      if not name and not tool_id:
        continue
      raw_input = block.get("input")
      if isinstance(raw_input, str):
        arguments = raw_input
      elif raw_input is None:
        arguments = "{}"
      else:
        arguments = _json_dumps(raw_input)
      tool_calls.append({
        "id": tool_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
      })
  if thinking_parts and not replay_reasoning:
    text_parts = [*thinking_parts, *text_parts]
    thinking_parts = []
  message: dict[str, Any] = {
    "role": "assistant",
    "content": "\n".join(text_parts),
  }
  if tool_calls:
    message["tool_calls"] = tool_calls
  if thinking_parts and replay_reasoning:
    message["reasoning"] = "\n\n".join(thinking_parts)
  return message


def _user_wire(content: Any, *, supports_vision: bool) -> list[dict[str, Any]]:
  if not isinstance(content, list):
    text = _sanitize_text(content)
    return [{"role": "user", "content": text}] if text else []
  messages: list[dict[str, Any]] = []
  parts: list[dict[str, Any]] = []
  for block in content:
    if not isinstance(block, dict):
      continue
    block_type = block.get("type")
    if block_type == "tool_result":
      if parts:
        messages.append({"role": "user", "content": parts})
        parts = []
      tool_id = str(block.get("tool_use_id") or "").strip()
      tool_message: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": tool_id,
        "content": _stringify_tool_result(block.get("content")),
      }
      tool_name = str(block.get("tool_name") or block.get("name") or "").strip()
      if tool_name:
        tool_message["name"] = tool_name
      messages.append(tool_message)
      continue
    if block_type == "text":
      text = _sanitize_text(block.get("text"))
      if text:
        parts.append({"type": "text", "text": text})
      continue
    if block_type == "image" and supports_vision:
      url = _image_url(block)
      if url:
        parts.append({"type": "image_url", "image_url": {"url": url}})
  if parts:
    if len(parts) == 1 and parts[0].get("type") == "text":
      messages.append({"role": "user", "content": parts[0]["text"]})
    else:
      messages.append({"role": "user", "content": parts})
  return messages


def _dict_value(value: object) -> dict[str, Any]:
  if not isinstance(value, dict):
    return {}
  return {str(key): item for key, item in value.items()}


def _synthetic_tool_message(tool_call: dict[str, Any]) -> dict[str, Any]:
  function = _dict_value(tool_call.get("function"))
  return {
    "role": "tool",
    "tool_call_id": str(tool_call.get("id") or ""),
    "name": str(function.get("name") or ""),
    "content": _json_dumps({
      "error": {"code": "missing_tool_result", "message": "No result provided"},
    }),
  }


def _convert_messages(
  messages: list[dict[str, Any]],
  *,
  system_prompt: str | list[tuple[str, bool]] | None,
  model_info: ModelInfo,
) -> list[dict[str, Any]]:
  replay_reasoning = bool(model_info.supports_thinking)
  converted: list[dict[str, Any]] = []
  system = _sanitize_text(_system_prompt_text(system_prompt)).strip()
  if system:
    converted.append({"role": "system", "content": system})
  pending_calls: list[dict[str, Any]] = []
  for message in messages:
    if not isinstance(message, dict):
      continue
    role = message.get("role")
    if pending_calls and role != "user":
      converted.extend(_synthetic_tool_message(call) for call in pending_calls)
      pending_calls = []
    if role == "system":
      text = _sanitize_text(message.get("content")).strip()
      if text:
        converted.append({"role": "system", "content": text})
      continue
    if role == "assistant":
      wire = _assistant_wire(message.get("content"), replay_reasoning=replay_reasoning)
      pending_calls = list(wire.get("tool_calls") or [])
      if message.get("stop_reason") == "compaction" and wire.get("content") and not pending_calls:
        converted.append({"role": "user", "content": wire["content"]})
        continue
      if wire.get("content") or pending_calls or wire.get("reasoning"):
        converted.append(wire)
      continue
    if role == "tool":
      pending_calls = []
      converted.append({
        "role": "tool",
        "tool_call_id": str(message.get("tool_call_id") or message.get("tool_use_id") or ""),
        "name": str(message.get("name") or message.get("tool_name") or ""),
        "content": _stringify_tool_result(message.get("content")),
      })
      continue
    if role != "user":
      continue
    user_messages = _user_wire(
      message.get("content"),
      supports_vision=bool(model_info.supports_vision),
    )
    if pending_calls:
      answered = {
        str(item.get("tool_call_id") or "")
        for item in user_messages
        if item.get("role") == "tool"
      }
      missing = [
        call for call in pending_calls if str(call.get("id") or "") not in answered
      ]
      converted.extend(_synthetic_tool_message(call) for call in missing)
      pending_calls = []
    converted.extend(user_messages)
  if pending_calls:
    converted.extend(_synthetic_tool_message(call) for call in pending_calls)
  return converted


def _caller_asked_for_structured_output(kwargs: dict[str, Any]) -> tuple[bool, Any]:
  response_format = kwargs.get("response_format")
  schema = None
  for key in ("json_schema", "output_schema", "response_schema"):
    value = kwargs.get(key)
    if isinstance(value, dict):
      schema = value
      break
  asked = (
    response_format is not None
    or schema is not None
    or kwargs.get("structured_output") is True
  )
  return asked, schema if schema is not None else response_format


def _response_format(compat: OpenAIChatCompletionsCompat, kwargs: dict[str, Any]) -> dict[str, Any] | None:
  asked, supplied = _caller_asked_for_structured_output(kwargs)
  if not asked or compat.json_mode == "none":
    return None
  if compat.json_mode == "json_object":
    return {"type": "json_object"}
  schema = supplied if isinstance(supplied, dict) and supplied.get("type") != "json_schema" else None
  if isinstance(supplied, dict):
    nested = supplied.get("json_schema")
    if isinstance(nested, dict) and isinstance(nested.get("schema"), dict):
      schema = nested["schema"]
    elif isinstance(supplied.get("schema"), dict) and supplied.get("type") != "object":
      schema = supplied["schema"]
    elif schema is None and supplied.get("type") not in {"json_schema", "json_object"}:
      schema = supplied
  if not isinstance(schema, dict):
    return None
  return {"type": "json_schema", "json_schema": {"schema": schema}}


def _apply_tools(params: dict[str, Any], tools: list[dict[str, Any]], tool_choice: Any) -> None:
  converted = _convert_tools(tools)
  mode = _tool_choice_mode(tool_choice)
  if mode == "named":
    name = _named_tool_choice(tool_choice)
    converted = [
      tool for tool in converted
      if str((tool.get("function") or {}).get("name") or "") == name
    ]
    mode = "required"
  if not converted and mode != "required":
    return
  params["tools"] = converted
  if mode == "none":
    params["tool_choice"] = "none"
  elif mode == "required":
    params["tool_choice"] = "required"


def _parse_sse(buffer: str) -> tuple[list[Any], str]:
  events: list[Any] = []
  while True:
    sep = buffer.find("\n\n")
    sep_len = 2
    cr = buffer.find("\r\n\r\n")
    if cr != -1 and (sep == -1 or cr < sep):
      sep = cr
      sep_len = 4
    if sep == -1:
      return events, buffer
    block, buffer = buffer[:sep], buffer[sep + sep_len :]
    data_lines = [
      line[5:].lstrip()
      for line in block.splitlines()
      if line.startswith("data:")
    ]
    if not data_lines:
      continue
    data = "\n".join(data_lines).strip()
    if not data:
      continue
    if data == "[DONE]":
      events.append("[DONE]")
      continue
    try:
      parsed = json.loads(data)
    except json.JSONDecodeError:
      continue
    events.append(parsed)
  return events, buffer


def _delta_text(content: Any) -> str:
  if content is None or isinstance(content, bool):
    return ""
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts: list[str] = []
    for part in content:
      if isinstance(part, str):
        parts.append(part)
      elif isinstance(part, dict):
        parts.append(str(part.get("text") or ""))
    return "".join(parts)
  return ""


def _accumulate_arguments(current: str, fragment: str) -> str:
  if not fragment:
    return current
  if not current:
    return fragment
  return current + fragment


class _ToolCallAccumulator:
  def __init__(self, index: int) -> None:
    self.index = index
    self.tool_id = ""
    self.name = ""
    self.arguments = ""
    self.started = False
    self.delta_sent = False

  def apply(self, raw: dict[str, Any]) -> str:
    tool_id = str(raw.get("id") or "").strip()
    if tool_id:
      self.tool_id = tool_id
    function = _dict_value(raw.get("function"))
    name = str(function.get("name") or "").strip()
    if name:
      self.name = name
    arguments = function.get("arguments")
    previous = self.arguments
    if isinstance(arguments, dict):
      self.arguments = _json_dumps(arguments)
      if self.arguments == previous:
        return ""
      if self.arguments.startswith(previous) and previous:
        return self.arguments[len(previous) :]
      return self.arguments
    fragment = "" if arguments is None else str(arguments)
    self.arguments = _accumulate_arguments(previous, fragment)
    return fragment


def _parse_tool_input(arguments: str) -> dict[str, Any]:
  if not arguments.strip():
    return {}
  try:
    parsed = json.loads(arguments)
  except json.JSONDecodeError:
    return {}
  return parsed if isinstance(parsed, dict) else {}


def _usage_counts(usage: dict[str, Any]) -> tuple[int, int, int, int, int]:
  prompt = int(usage.get("prompt_tokens") or 0)
  completion = int(usage.get("completion_tokens") or 0)
  details = usage.get("prompt_tokens_details")
  cached = 0
  if isinstance(details, dict):
    cached = int(details.get("cached_tokens") or 0)
  completion_details = usage.get("completion_tokens_details")
  reasoning = 0
  if isinstance(completion_details, dict):
    reasoning = int(completion_details.get("reasoning_tokens") or 0)
  return prompt, max(0, prompt - cached), cached, completion, reasoning


def _stop_reason(finish_reason: str, *, saw_tool_use: bool) -> str:
  mapped = _FINISH_STOP_REASONS.get(finish_reason, "end_turn")
  if saw_tool_use and mapped == "end_turn":
    return "tool_use"
  return mapped


def _error_message(response: httpx.Response) -> str:
  text = response.text
  message = text
  try:
    parsed = json.loads(text) if text else None
  except json.JSONDecodeError:
    parsed = None
  if isinstance(parsed, dict):
    error = parsed.get("error")
    if isinstance(error, dict):
      message = str(error.get("message") or error.get("type") or text)
    elif isinstance(error, str) and error:
      message = error
  detail = str(message or "").strip() or response.reason_phrase or "Request failed"
  return f"ollama API error (status={response.status_code}): {detail}"


def _exception_text(exc: Exception) -> str:
  return " ".join(part for part in (str(exc), _response_text(exc)) if part).strip()


def _is_cloud_model(model: str) -> bool:
  """A daemon cloud tag: ``name:cloud`` or a tag ending in ``-cloud``."""
  text = model.strip().lower()
  if not text:
    return False
  tag = text.rsplit(":", 1)[-1]
  return tag == "cloud" or tag.endswith("-cloud")


def _request_model(exc: Exception) -> str:
  stamped = getattr(exc, "ollama_model", None)
  if isinstance(stamped, str) and stamped.strip():
    return stamped.strip()
  request = getattr(exc, "request", None)
  content = getattr(request, "content", b"") if request is not None else b""
  if isinstance(content, bytes):
    content = content.decode("utf-8", errors="replace")
  if isinstance(content, str) and content.strip():
    try:
      parsed = json.loads(content)
    except json.JSONDecodeError:
      parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("model"), str):
      return str(parsed["model"]).strip()
  return ""


def _is_cloud_tag_failure(exc: Exception) -> bool:
  if _is_cloud_model(_request_model(exc)):
    return True
  return _CLOUD_TAG_RE.search(_exception_text(exc)) is not None


def _is_content_refusal(exc: Exception) -> bool:
  code = str(_error_code_from_exception(exc) or "")
  searchable = f"{_exception_text(exc)} {code}"
  return bool(_CONTENT_REFUSAL_RE.search(searchable) or code in {"content_filter", "refusal"})


def _is_usage_limit_only(exc: Exception) -> bool:
  if _is_context_length_exception(exc) or _is_content_refusal(exc):
    return False
  searchable = _exception_text(exc)
  if not _USAGE_LIMIT_RE.search(searchable):
    return False
  return _BILLING_NOT_USAGE_RE.search(searchable) is None


def _retry_after_reset_at(exc: Exception) -> float | None:
  raw = getattr(exc, "retry_after", None)
  if raw is None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
      raw = headers.get("retry-after")
  if raw is None:
    return None
  text = str(raw).strip()
  if not text:
    return None
  try:
    seconds = float(text)
  except ValueError:
    seconds = None
  if seconds is not None:
    if seconds < 0:
      return None
    return time.time() + seconds
  try:
    parsed = parsedate_to_datetime(text)
  except (TypeError, ValueError, OverflowError, IndexError):
    return None
  return parsed.timestamp()


def _credential_failure(
  provider: str,
  exc: Exception,
  *,
  kind: CredentialFailureKind,
  status_code: int | None,
  reset_at: float | None = None,
) -> ProviderCredentialFailure:
  return ProviderCredentialFailure(
    provider=provider,
    kind=kind,
    status_code=status_code,
    error_code=_error_code_from_exception(exc),
    message=_exception_text(exc),
    reset_at=reset_at,
  )


# Native routes (/api/version) are not under the OpenAI /v1 prefix.
_DAEMON_ORIGIN = "http://127.0.0.1:11434"
_PRESENCE_TIMEOUT_SECONDS = 0.5
_presence_cache: dict[str, bool] = {}


def _reset_presence_cache() -> None:
  _presence_cache.clear()


def _daemon_present(config: Mapping[str, object]) -> bool:
  """True when ``GET {base}/api/version`` is 2xx. Cached per base; never raises."""
  configured = str(config.get("base_url") or "").strip().rstrip("/")
  base = configured or _DAEMON_ORIGIN
  while base.endswith("/v1"):
    base = base[: -len("/v1")].rstrip("/")
  if not base:
    base = _DAEMON_ORIGIN
  cached = _presence_cache.get(base)
  if cached is not None:
    return cached
  url = f"{base}/api/version"
  raw_transport = config.get("_transport")
  transport = raw_transport if isinstance(raw_transport, httpx.BaseTransport) else None
  present = False
  try:
    if transport is not None:
      with httpx.Client(transport=transport, timeout=_PRESENCE_TIMEOUT_SECONDS) as client:
        present = client.get(url).is_success
    else:
      response = httpx.get(url, timeout=_PRESENCE_TIMEOUT_SECONDS)
      try:
        present = response.is_success
      finally:
        response.close()
  except Exception:
    present = False
  _presence_cache[base] = present
  return present


class OllamaLocalProvider(ModelProvider):
  """Keyless loopback chat-completions client.

  Cloud tags (``name:cloud`` or a tag ending in ``-cloud``) are the same
  daemon after ``ollama signin``. Request facts come from the bound entry's
  ``OpenAIChatCompletionsCompat``. The wire is ``POST {base}/chat/completions``
  with ``stream`` and ``stream_options.include_usage``. ``num_ctx`` and
  ``keep_alive`` are not request fields.
  """

  name = "ollama"
  supports_tool_choice_none = True

  @classmethod
  def adapter_route_support(cls) -> AdapterRouteSupport:
    return AdapterRouteSupport(
      adapter="ollama.chat_completions",
      provider="ollama",
      protocol_profiles=frozenset({"chat_completions.standard"}),
      routes=frozenset({"ollama.local", "ollama.daemon_cloud"}),
    )

  def __init__(self, *, rate_table: RateTable | None = None) -> None:
    self._rate_table = _rate_table_for(self.name, rate_table)
    self._client_state: WeakKeyDictionary[httpx.AsyncClient, dict[str, Any]] = WeakKeyDictionary()

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return _daemon_present(config)

  def _base_url(self, config: dict[str, Any]) -> str:
    configured = str(config.get("base_url") or "").strip()
    return configured or _DEFAULT_BASE_URL


  def _prepared_headers(self, config: dict[str, Any]) -> dict[str, str]:
    extra = config.get("headers")
    headers = {
      "Accept": "text/event-stream",
      "Content-Type": "application/json",
    }
    if isinstance(extra, dict):
      for key, value in extra.items():
        if str(key).lower() == "authorization":
          continue
        headers[str(key)] = str(value)
    api_key = str(config.get("api_key") or "").strip()
    if api_key:
      headers["Authorization"] = f"Bearer {api_key}"
    return headers

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    api_key = str(config.get("api_key") or "").strip()
    client_kwargs: dict[str, Any] = {
      "timeout": httpx.Timeout(timeout=timeout, connect=10.0),
    }
    transport = config.get("_transport")
    if isinstance(transport, httpx.AsyncBaseTransport):
      client_kwargs["transport"] = transport
    client = httpx.AsyncClient(**client_kwargs)
    self._client_state[client] = {
      "endpoint_url": _chat_completions_url(self._base_url(config)),
      "headers": self._prepared_headers(config),
      "send_authorization": bool(api_key),
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
    support = self.adapter_route_support()
    if support is None:
      raise TypeError(f"{type(self).__name__} declares no adapter route")
    model_id, entry = admitted_entry(support, model)
    compat = entry.compat
    if not isinstance(compat, OpenAIChatCompletionsCompat):
      raise TypeError(f"{entry.key}.compat is not the openai.chat_completions key set")
    control = compat.reasoning_control
    supports_thinking = bool(
      control is not None and any(value != "none" for value in control.values)
    )
    info = model_info_from_entry(
      model_id,
      entry,
      self._rate_table,
      supports_thinking=supports_thinking,
    )
    # Local inference sends nothing until this host has prefilled the prompt. A
    # daemon-proxied `:cloud` tag (route ollama.daemon_cloud) prefills on ollama.com
    # and keeps the hung-peer bound of any cloud stream.
    info.first_progress_after_prefill = entry.route == "ollama.local"
    return info

  def resolve_effort(
    self,
    *,
    requested: ThinkingLevel,
    model: str,
    model_info: ModelInfo,
    max_tokens: int,
    **request_context: Any,
  ) -> EffortResolution:
    del max_tokens, request_context
    control = _chat_compat(model_info).reasoning_control
    if control is None or not control.values:
      if requested != ThinkingLevel.NONE:
        raise ValueError(
          f"{self.name} model {model!r} does not support reasoning effort "
          f"{requested.value!r}; unsupported effort is refused, never clamped"
        )
      return EffortResolution(requested, ThinkingLevel.NONE, False, {})
    supported = tuple(ThinkingLevel(value) for value in control.values)
    if requested not in supported:
      raise ValueError(
        f"{self.name} model {model!r} does not support reasoning effort "
        f"{requested.value!r} (supported: "
        f"{', '.join(level.value for level in supported)}); unsupported "
        "effort is refused, never clamped"
      )
    wire_value = (
      control.off_value
      if requested == ThinkingLevel.NONE and control.off_value is not None
      else requested.value
    )
    fragment = _reasoning_fragment(control.param, str(wire_value))
    thinking_on = requested != ThinkingLevel.NONE and wire_value != control.off_value
    return EffortResolution(
      requested=requested,
      effective=requested,
      thinking_enabled_effective=thinking_on,
      payload_fragments={"reasoning_request": fragment} if fragment else {},
    )

  def normalize_messages(
    self,
    messages: list[dict[str, Any]],
    model_info: ModelInfo,
  ) -> list[dict[str, Any]]:
    del model_info
    return truncate_to_last_compaction(list(messages), compaction_as_text=True)

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
    del thinking_level
    model_info = self.get_model_info(model)
    compat = _chat_compat(model_info)
    effort_resolution = kwargs.get("effort_resolution")
    if not isinstance(effort_resolution, EffortResolution):
      raise ValueError(
        f"{self.name} build_request_params requires an admitted EffortResolution; "
        "it does not resolve effort locally"
      )
    params: dict[str, Any] = {
      "model": model_info.id,
      "messages": _convert_messages(
        self.normalize_messages(messages, model_info),
        system_prompt=system_prompt,
        model_info=model_info,
      ),
      "stream": True,
      "stream_options": {"include_usage": True},
      compat.output_token_param: max_tokens,
    }
    temperature = _request_temperature(kwargs.get("temperature"))
    if compat.sampling_required:
      params["temperature"] = 1.0 if temperature is None else temperature
    elif compat.accepts_temperature and temperature is not None:
      params["temperature"] = temperature
    if compat.reasoning_control is not None:
      fragment = effort_resolution.payload_fragments.get("reasoning_request")
      if isinstance(fragment, dict) and fragment:
        params.update(fragment)
    response_format = _response_format(compat, kwargs)
    if response_format is not None:
      params["response_format"] = response_format
    _apply_tools(params, tools, kwargs.get("tool_choice"))
    headers = kwargs.get("headers")
    if isinstance(headers, dict) and headers:
      params["_headers"] = dict(headers)
    base_url = kwargs.get("base_url")
    if base_url:
      params["_endpoint_url"] = _chat_completions_url(str(base_url))
    return params

  async def stream(
    self,
    client: Any,
    params: dict[str, Any],
  ) -> AsyncGenerator[StreamEvent, None]:
    if not isinstance(client, httpx.AsyncClient):
      raise RuntimeError(f"{type(self).__name__} requires an httpx.AsyncClient")
    state = self._client_state.get(client)
    if state is None:
      raise RuntimeError(f"{self.name} client state is missing")
    request_params = dict(params)
    additional_headers = request_params.pop("_headers", None)
    endpoint_url = str(request_params.pop("_endpoint_url", state["endpoint_url"]))
    headers = dict(state["headers"])
    if isinstance(additional_headers, dict):
      for key, value in additional_headers.items():
        if str(key).lower() == "authorization":
          continue
        headers[str(key)] = str(value)
    if not state.get("send_authorization"):
      headers.pop("Authorization", None)
      headers.pop("authorization", None)
    async with client.stream("POST", endpoint_url, json=request_params, headers=headers) as response:
      if response.status_code != 200:
        await response.aread()
        error = httpx.HTTPStatusError(
          _error_message(response),
          request=response.request,
          response=response,
        )
        setattr(error, "ollama_model", str(request_params.get("model") or ""))
        retry_after = response.headers.get("retry-after")
        if retry_after:
          setattr(error, "retry_after", retry_after)
        raise error
      async for event in self._map_sse(response):
        yield event

  async def _map_sse(self, response: httpx.Response) -> AsyncGenerator[StreamEvent, None]:
    tools: dict[int, _ToolCallAccumulator] = {}
    buffer = ""
    message_started = False
    provider_model: str | None = None
    text = ""
    thinking = ""
    text_open = False
    thinking_open = False
    saw_tool_use = False
    finish_reason = ""
    usage: dict[str, Any] | None = None
    done = False

    def start_events() -> list[StreamEvent]:
      nonlocal message_started
      if message_started:
        return []
      message_started = True
      return [StreamEvent(
        type="message_start",
        provider_reported_model=provider_model,
      )]

    async for chunk in response.aiter_text():
      buffer += chunk
      parsed, buffer = _parse_sse(buffer)
      for item in parsed:
        if item == "[DONE]":
          done = True
          break
        if not isinstance(item, dict):
          continue
        if isinstance(item.get("error"), dict) and "choices" not in item:
          error = item["error"]
          detail = str(error.get("message") or error.get("type") or "ollama stream error")
          raise RuntimeError(f"ollama stream error: {detail}")
        model_name = item.get("model")
        if isinstance(model_name, str) and model_name.strip():
          provider_model = model_name.strip()
        raw_usage = item.get("usage")
        if isinstance(raw_usage, dict):
          usage = raw_usage
        choices = item.get("choices")
        if not isinstance(choices, list) or not choices:
          continue
        choice = choices[0] if isinstance(choices[0], dict) else {}
        raw_finish = choice.get("finish_reason")
        if isinstance(raw_finish, str) and raw_finish:
          finish_reason = raw_finish
        delta = _dict_value(choice.get("delta"))
        if not delta and not finish_reason:
          continue
        for event in start_events():
          yield event
        reasoning = delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
          if text_open:
            yield StreamEvent(
              type="text_end",
              text=text,
              raw_block={"type": "text", "text": text},
            )
            text_open = False
          thinking += reasoning
          thinking_open = True
          yield StreamEvent(type="thinking_delta", thinking_text=reasoning)
        content = _delta_text(delta.get("content"))
        if content:
          if thinking_open:
            yield StreamEvent(
              type="thinking_end",
              thinking_text=thinking,
              raw_block={"type": "thinking", "thinking": thinking},
            )
            thinking_open = False
          text += content
          text_open = True
          yield StreamEvent(type="text_delta", text=content)
        raw_calls = delta.get("tool_calls")
        if isinstance(raw_calls, list):
          for raw in raw_calls:
            if not isinstance(raw, dict):
              continue
            index = raw.get("index")
            tool_index = index if isinstance(index, int) and not isinstance(index, bool) else 0
            accumulator = tools.get(tool_index)
            if accumulator is None:
              accumulator = _ToolCallAccumulator(tool_index)
              tools[tool_index] = accumulator
            fragment = accumulator.apply(raw)
            if accumulator.tool_id and not accumulator.started:
              accumulator.started = True
              saw_tool_use = True
              yield StreamEvent(
                type="tool_use_start",
                tool_id=accumulator.tool_id,
                tool_name=accumulator.name,
                raw_block={
                  "type": "tool_use",
                  "id": accumulator.tool_id,
                  "name": accumulator.name,
                },
              )
            if fragment:
              accumulator.delta_sent = True
              saw_tool_use = True
              yield StreamEvent(type="tool_use_delta", tool_input_json=fragment)
        if done:
          break
      if done:
        break

    for event in start_events():
      yield event
    if thinking_open:
      yield StreamEvent(
        type="thinking_end",
        thinking_text=thinking,
        raw_block={"type": "thinking", "thinking": thinking},
      )
    if text_open:
      yield StreamEvent(
        type="text_end",
        text=text,
        raw_block={"type": "text", "text": text},
      )
    for tool_index in sorted(tools):
      accumulator = tools[tool_index]
      tool_id = accumulator.tool_id or f"call_{tool_index}"
      tool_input = _parse_tool_input(accumulator.arguments)
      arguments = accumulator.arguments or "{}"
      if not accumulator.started:
        yield StreamEvent(
          type="tool_use_start",
          tool_id=tool_id,
          tool_name=accumulator.name,
          raw_block={"type": "tool_use", "id": tool_id, "name": accumulator.name},
        )
      if not accumulator.delta_sent and arguments:
        yield StreamEvent(type="tool_use_delta", tool_input_json=arguments)
      saw_tool_use = True
      yield StreamEvent(
        type="tool_use_end",
        tool_id=tool_id,
        tool_name=accumulator.name,
        tool_input_json=arguments,
        tool_input=tool_input,
        raw_block={
          "type": "tool_use",
          "id": tool_id,
          "name": accumulator.name,
          "input": tool_input,
        },
      )
    if isinstance(usage, dict):
      prompt, uncached, cached, completion, reasoning_tokens = _usage_counts(usage)
      yield StreamEvent(
        type="usage_update",
        input_tokens=uncached,
        output_tokens=completion,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cached,
      )
      request_input: int | None = prompt
    else:
      request_input = None
    yield StreamEvent(
      type="message_end",
      stop_reason=_stop_reason(finish_reason, saw_tool_use=saw_tool_use),
      stop_details={"finish_reason": finish_reason} if finish_reason else None,
      request_input_tokens=request_input,
    )

  def is_retryable_error(self, exc: Exception) -> bool:
    status = _status_code_from_exception(exc)
    if status is not None:
      return status in _RETRYABLE_STATUSES or 500 <= status <= 599
    return isinstance(exc, (httpx.TransportError, httpx.StreamError))

  def is_context_length_error(self, exc: Exception) -> bool:
    return _is_context_length_exception(exc)

  def classify_credential_failure(self, exc: Exception) -> ProviderCredentialFailure | None:
    status = _status_code_from_exception(exc)
    if status == 400 or _is_context_length_exception(exc) or _is_content_refusal(exc):
      return None
    # An unsigned-in daemon answers a cloud tag with 401/403. That is auth,
    # not a usage limit and not a billing denial.
    if status in {401, 403} and _is_cloud_tag_failure(exc):
      return _credential_failure(self.name, exc, kind="auth", status_code=status)
    if status == 429 or _is_usage_limit_only(exc):
      return _credential_failure(
        self.name,
        exc,
        kind="rate_limit",
        status_code=status,
        reset_at=_retry_after_reset_at(exc),
      )
    if status == 401:
      return _credential_failure(self.name, exc, kind="auth", status_code=status)
    if status == 403:
      return _credential_failure(self.name, exc, kind="billing", status_code=status)
    # 429 already returned as rate_limit. Do not defer it.
    return _classify_provider_credential_failure(provider=self.name, exc=exc)

__all__ = ["OllamaLocalProvider"]
