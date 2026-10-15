from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Mapping
from weakref import WeakKeyDictionary

import httpx

from model_authority.rates import RateTable
from model_authority.registry import AdapterRouteSupport
from model_authority.schema import GoogleGenerateContentCompat
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
  registry_effort_values,
)
from .credential_pool import rotate_provider_credential
from .google_tool_schema import sanitize_tool_schema

log = logging.getLogger(__name__)

# Cloud Code Assist streaming, not the public Gemini API. The envelope, the
# ``User-Agent`` and the ``{response: <generateContent chunk>}`` unwrap follow
# omp's antigravity path (google-gemini-cli.ts, gemini-headers.ts).
DEFAULT_ANTIGRAVITY_BASE_URL = "https://daily-cloudcode-pa.googleapis.com"
# Antigravity client's identity per the 2026-10-10 ruling. omp's
# ``getAntigravityUserAgent`` sends ``antigravity/hub/<version>`` and the
# backend gates newer models on that version. ``2.19.1`` is
# ``DEFAULT_ANTIGRAVITY_VERSION`` in gemini-headers.ts. os/arch/cl are that
# function's captured defaults; this module does not read the environment.
DEFAULT_ANTIGRAVITY_CLIENT_VERSION = "2.19.1"
_ANTIGRAVITY_OS = "darwin"
_ANTIGRAVITY_ARCH = "arm64"
_ANTIGRAVITY_CL = "963137146"
_BODY_USER_AGENT = "antigravity"
_REQUEST_TYPE = "agent"
# omp ``SKIP_THOUGHT_SIGNATURE``. Cloud Code Assist accepts this on an unsigned
# first function call and rejects a Gemini 3 turn that omits it.
_SKIP_THOUGHT_SIGNATURE = "skip_thought_signature_validator"
_SKIP_THOUGHT_FIELD = "requires_skip_thought_signature_on_first_function_call"
_ABSENT = object()
_INT63_MASK = (1 << 63) - 1
_SESSION_RANDOM_BOUND = 9_000_000_000_000_000_000
_RETRYABLE_STATUSES = frozenset({408, 409})
_CONTENT_ERROR_REASONS = frozenset({
  "SAFETY",
  "BLOCKLIST",
  "PROHIBITED_CONTENT",
  "RECITATION",
})
_FAILURE_FINISH_REASONS = frozenset({
  "LANGUAGE",
  "OTHER",
  "SPII",
  "MALFORMED_FUNCTION_CALL",
  "MALFORMED_RESPONSE",
  "IMAGE_SAFETY",
  "IMAGE_PROHIBITED_CONTENT",
  "IMAGE_OTHER",
  "IMAGE_RECITATION",
  "NO_IMAGE",
  "UNEXPECTED_TOOL_CALL",
  "TOO_MANY_TOOL_CALLS",
  "MISSING_THOUGHT_SIGNATURE",
  "ESCALATION",
  "PUP_LIMITED_DISABLED",
})
_ADJUSTABLE_HARM_CATEGORIES = (
  "HARM_CATEGORY_HARASSMENT",
  "HARM_CATEGORY_HATE_SPEECH",
  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
  "HARM_CATEGORY_DANGEROUS_CONTENT",
)
_SURROGATE_RE = re.compile(
  r"[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]"
)
_BASE64_SIGNATURE_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
_GOOGLE_STATUS_RE = re.compile(r'"status"\s*:\s*"([A-Z0-9_]+)"')
_USAGE_LIMIT_RE = re.compile(
  r"\b("
  r"resource[\s_-]*exhausted"
  r"|exceeded your current quota"
  r"|quota[\s_-]*(?:exhausted|exceeded|limit)"
  r"|rate[\s_-]*limit(?:ed|ing)?"
  r"|too many requests"
  r"|usage[\s_-]*limit"
  r")\b",
  re.IGNORECASE,
)
_CONTENT_REFUSAL_RE = re.compile(
  r"\b("
  r"safety"
  r"|blocklist"
  r"|prohibited[\s_-]*content"
  r"|recitation"
  r"|content[\s_-]*(?:filter|policy)"
  r"|prompt was blocked"
  r"|blocked due to"
  r")\b",
  re.IGNORECASE,
)


def _sanitize(value: Any) -> str:
  return _SURROGATE_RE.sub("\ufffd", str(value if value is not None else ""))


def _field(obj: dict[str, Any], *names: str) -> Any:
  for name in names:
    if name in obj:
      return obj[name]
  return None


def _credential(value: Any) -> str:
  return str(value or "").strip()


def _header_value(value: str, *, name: str) -> str:
  if any(char in value for char in "\r\n"):
    raise RuntimeError(f"google_antigravity {name} is not a single header value")
  return value


def _drop_header(headers: dict[str, Any], name: str) -> None:
  for key in list(headers):
    if str(key).lower() == name.lower():
      headers.pop(key, None)


def _bound_base_url(config: dict[str, Any]) -> str:
  configured = str(config.get("base_url") or "").strip()
  return configured.rstrip("/") or DEFAULT_ANTIGRAVITY_BASE_URL


def _client_version(config: dict[str, Any]) -> str:
  configured = str(config.get("client_version") or "").strip()
  version = configured or DEFAULT_ANTIGRAVITY_CLIENT_VERSION
  return _header_value(version, name="client_version")


def _user_agent(version: str) -> str:
  return (
    f"antigravity/hub/{version} "
    f"(aidev_client; os_type={_ANTIGRAVITY_OS}; arch={_ANTIGRAVITY_ARCH}; cl={_ANTIGRAVITY_CL})"
  )


def _stream_url(base_url: str) -> str:
  return f"{base_url.rstrip('/')}/v1internal:streamGenerateContent?alt=sse"


def _compat(model_info: ModelInfo) -> GoogleGenerateContentCompat:
  compat = model_info.compat
  if not isinstance(compat, GoogleGenerateContentCompat):
    raise TypeError(
      f"model {model_info.id!r} carries no google.generate_content compat"
    )
  return compat


def _model_info(model: str, rate_table: RateTable) -> ModelInfo:
  model_id, entry = admitted_entry(GoogleAntigravityProvider.adapter_route_support(), model)
  compat = entry.compat
  if not isinstance(compat, GoogleGenerateContentCompat):
    raise TypeError(f"{entry.key}.compat is not the google.generate_content key set")
  supports_thinking = any(value != "none" for value in registry_effort_values(entry))
  thinking_mode = None
  if supports_thinking and compat.thinking_control is not None:
    leaf = compat.thinking_control.field.rsplit(".", 1)[-1]
    thinking_mode = "budget" if leaf == "thinkingBudget" else "adaptive"
  return model_info_from_entry(
    model_id,
    entry,
    rate_table,
    supports_thinking=supports_thinking,
    thinking_mode=thinking_mode,
  )


def _rate_table_for(provider: str, override: RateTable | None) -> RateTable:
  """This provider's authority table, or an empty table when it is not priced yet."""

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


def _assign_dotted(root: dict[str, Any], field_path: str, value: Any) -> None:
  parts = [part for part in str(field_path or "").split(".") if part]
  if not parts:
    return
  cursor = root
  for key in parts[:-1]:
    child = cursor.get(key)
    if not isinstance(child, dict):
      child = {}
      cursor[key] = child
    cursor = child
  cursor[parts[-1]] = value


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
  for key, value in src.items():
    current = dst.get(key)
    if isinstance(current, dict) and isinstance(value, dict):
      _deep_merge(current, value)
    else:
      dst[key] = value


def _system_text(system_prompt: str | list[tuple[str, bool]] | None) -> str:
  if system_prompt is None:
    return ""
  if isinstance(system_prompt, list):
    return "\n\n".join(_sanitize(text) for text, _cache in system_prompt if text).strip()
  return _sanitize(system_prompt).strip()


def _safety_settings(value: str | None) -> list[dict[str, Any]] | None:
  text = str(value or "").strip()
  if not text:
    return None
  if text[:1] in "[{":
    try:
      parsed = json.loads(text)
    except json.JSONDecodeError:
      parsed = None
    if isinstance(parsed, list):
      return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
      nested = parsed.get("safetySettings")
      if isinstance(nested, list):
        return [item for item in nested if isinstance(item, dict)]
      return [parsed]
  return [
    {"category": category, "threshold": text}
    for category in _ADJUSTABLE_HARM_CATEGORIES
  ]


def _same_model(message: dict[str, Any], model_info: ModelInfo) -> bool:
  recorded = str(message.get("model") or "").strip()
  if recorded.startswith("models/"):
    recorded = recorded[len("models/"):]
  current = model_info.id[len("models/"):] if model_info.id.startswith("models/") else model_info.id
  return bool(recorded) and recorded == current


def _thought_signature(block: dict[str, Any]) -> str:
  for key in ("thoughtSignature", "thought_signature", "signature", "thinkingSignature"):
    value = block.get(key)
    if isinstance(value, str) and value:
      return value
  return ""


def _valid_thought_signature(value: str) -> str:
  # Google TYPE_BYTES signatures are base64. The skip sentinel is not, so a
  # replayed sentinel counts as unsigned and is re-applied only on the first call.
  if not value or len(value) % 4 != 0:
    return ""
  if _BASE64_SIGNATURE_RE.fullmatch(value) is None:
    return ""
  return value


def _skip_thought_on_first(compat: GoogleGenerateContentCompat) -> bool:
  """omp ``requiresSkipThoughtSignatureOnFirstFunctionCall``.

  Schema 2.3's family compat does not carry the key. An explicit value wins.
  Until the key exists this Code Assist route sends the sentinel: Gemini 3
  rejects an unsigned first function call, and a model id is not a fact.
  """

  raw = getattr(compat, _SKIP_THOUGHT_FIELD, _ABSENT)
  if raw is _ABSENT:
    return True
  return raw is True


def _json_dumps(value: Any) -> str:
  return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _parse_object(value: Any) -> dict[str, Any] | None:
  if isinstance(value, dict):
    return value
  if isinstance(value, str):
    try:
      parsed = json.loads(value)
    except json.JSONDecodeError:
      return None
    if isinstance(parsed, dict):
      return parsed
  return None


def _function_response_body(content: Any, *, is_error: bool) -> dict[str, Any]:
  parsed = _parse_object(content)
  if parsed is not None:
    if is_error and "error" not in parsed:
      return {"error": parsed}
    return parsed
  if isinstance(content, list):
    texts = [
      _sanitize(block.get("text"))
      for block in content
      if isinstance(block, dict) and block.get("type") == "text"
    ]
    text = "\n".join(part for part in texts if part)
    payload: Any = text or content
  else:
    payload = _sanitize(content) if content is not None else ""
  if is_error:
    return {"error": payload}
  if isinstance(payload, dict):
    return payload
  return {"result": payload}


def _args_json(args: Any) -> str:
  if isinstance(args, str):
    return args
  if args is None:
    return ""
  return _json_dumps(args)


def _args_dict(args: Any, raw: str) -> dict[str, Any]:
  if isinstance(args, dict):
    return args
  parsed = _parse_object(raw)
  return parsed if parsed is not None else {}


def _synthetic_tool_result(tool_id: str, tool_name: str) -> dict[str, Any]:
  return {
    "type": "tool_result",
    "tool_use_id": tool_id,
    "tool_name": tool_name,
    "content": _json_dumps({
      "error": {"code": "missing_tool_result", "message": "No result provided"},
    }),
    "is_error": True,
  }


def _is_tool_result_block(block: Any) -> bool:
  return isinstance(block, dict) and block.get("type") == "tool_result"


def _tool_result_ids(message: dict[str, Any]) -> set[str]:
  content = message.get("content")
  if not isinstance(content, list):
    return set()
  return {
    str(block.get("tool_use_id") or "")
    for block in content
    if _is_tool_result_block(block) and block.get("tool_use_id")
  }


def _image_part(block: dict[str, Any]) -> dict[str, Any] | None:
  mime = block.get("mimeType") or block.get("mime_type") or block.get("media_type")
  data = block.get("data") or block.get("data_base64")
  if not mime or not data:
    return None
  return {"inlineData": {"mimeType": str(mime), "data": str(data)}}


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
  declarations = []
  for tool in tools:
    if not isinstance(tool, dict):
      continue
    parameters = tool.get("input_schema") or tool.get("parameters") or {
      "type": "object",
      "properties": {},
    }
    declarations.append({
      "name": str(tool.get("name") or ""),
      "description": str(tool.get("description") or ""),
      "parameters": sanitize_tool_schema(parameters, {}),
    })
  if not declarations:
    return []
  return [{"functionDeclarations": declarations}]


def _function_call_part(block: dict[str, Any], *, signature: str) -> dict[str, Any]:
  call: dict[str, Any] = {
    "name": str(block.get("name") or ""),
    "args": block.get("input") if isinstance(block.get("input"), dict) else {},
  }
  tool_id = str(block.get("id") or "")
  if tool_id:
    call["id"] = tool_id
  part: dict[str, Any] = {"functionCall": call}
  if signature:
    part["thoughtSignature"] = signature
  return part


def _function_response_part(
  block: dict[str, Any],
  *,
  names: dict[str, str],
) -> dict[str, Any]:
  tool_id = str(block.get("tool_use_id") or block.get("tool_call_id") or block.get("id") or "")
  name = str(block.get("tool_name") or block.get("name") or names.get(tool_id) or "")
  response: dict[str, Any] = {
    "name": name,
    "response": _function_response_body(block.get("content"), is_error=bool(block.get("is_error"))),
  }
  if tool_id:
    response["id"] = tool_id
  return {"functionResponse": response}


def _user_parts(
  content: Any,
  *,
  model_info: ModelInfo,
  names: dict[str, str],
) -> list[dict[str, Any]]:
  if isinstance(content, str):
    text = _sanitize(content)
    return [{"text": text}] if text else []
  if not isinstance(content, list):
    if content is None:
      return []
    text = _sanitize(content)
    return [{"text": text}] if text else []
  parts: list[dict[str, Any]] = []
  for block in content:
    if not isinstance(block, dict):
      continue
    block_type = block.get("type")
    if block_type == "text":
      text = _sanitize(block.get("text"))
      if text:
        parts.append({"text": text})
    elif block_type == "image" and model_info.supports_vision:
      image = _image_part(block)
      if image is not None:
        parts.append(image)
    elif block_type == "tool_result":
      parts.append(_function_response_part(block, names=names))
  return parts


def _model_parts(
  message: dict[str, Any],
  *,
  model_info: ModelInfo,
  names: dict[str, str],
  skip_on_first: bool,
) -> list[dict[str, Any]]:
  content = message.get("content")
  same_model = _same_model(message, model_info) or not str(message.get("model") or "").strip()
  if isinstance(content, str):
    text = _sanitize(content)
    return [{"text": text}] if text else []
  if not isinstance(content, list):
    if content is None:
      return []
    text = _sanitize(content)
    return [{"text": text}] if text else []
  parts: list[dict[str, Any]] = []
  first_call = True
  for block in content:
    if not isinstance(block, dict):
      continue
    block_type = block.get("type")
    if block_type == "text":
      text = _sanitize(block.get("text"))
      if not text:
        continue
      part: dict[str, Any] = {"text": text}
      signature = _valid_thought_signature(_thought_signature(block)) if same_model else ""
      if signature:
        part["thoughtSignature"] = signature
      parts.append(part)
    elif block_type == "thinking":
      thinking = _sanitize(block.get("thinking"))
      if not thinking and not _thought_signature(block):
        continue
      if same_model:
        part = {"text": thinking, "thought": True}
        signature = _valid_thought_signature(_thought_signature(block))
        if signature:
          part["thoughtSignature"] = signature
        parts.append(part)
      elif thinking:
        parts.append({"text": thinking})
    elif block_type in {"tool_use", "server_tool_use"}:
      tool_id = str(block.get("id") or "")
      if tool_id:
        names[tool_id] = str(block.get("name") or names.get(tool_id) or "")
      kept = _valid_thought_signature(_thought_signature(block)) if same_model else ""
      # Signed-first parallel turns leave later unsigned calls bare. The
      # sentinel is only for a turn whose first function call is itself unsigned.
      if not kept and first_call and skip_on_first:
        kept = _SKIP_THOUGHT_SIGNATURE
      first_call = False
      parts.append(_function_call_part(block, signature=kept))
  return parts


def _group_function_responses(parts: list[dict[str, Any]], *, after_model: bool) -> list[dict[str, Any]]:
  responses = [part for part in parts if "functionResponse" in part]
  others = [part for part in parts if "functionResponse" not in part]
  if not responses or not others:
    return parts
  if after_model or (parts and "functionResponse" in parts[0]):
    return responses + others
  return others + responses


def _merge_contents(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
  merged: list[dict[str, Any]] = []
  for content in contents:
    if not content.get("parts"):
      continue
    if merged and merged[-1]["role"] == content["role"]:
      merged[-1]["parts"].extend(content["parts"])
    else:
      merged.append({"role": content["role"], "parts": list(content["parts"])})
  previous = ""
  for content in merged:
    if content["role"] == "user":
      content["parts"] = _group_function_responses(content["parts"], after_model=previous == "model")
    previous = str(content["role"])
  return [content for content in merged if content.get("parts")]


def _convert_contents(
  messages: list[dict[str, Any]],
  model_info: ModelInfo,
  *,
  skip_on_first: bool,
) -> list[dict[str, Any]]:
  names: dict[str, str] = {}
  contents: list[dict[str, Any]] = []
  for message in messages:
    if not isinstance(message, dict):
      continue
    role = str(message.get("role") or "")
    if role == "assistant":
      parts = _model_parts(
        message,
        model_info=model_info,
        names=names,
        skip_on_first=skip_on_first,
      )
      if parts:
        contents.append({"role": "model", "parts": parts})
      continue
    if role in {"user", "tool"}:
      if role == "tool":
        parts = [_function_response_part(message, names=names)]
      else:
        parts = _user_parts(message.get("content"), model_info=model_info, names=names)
      if parts:
        contents.append({"role": "user", "parts": parts})
  return _merge_contents(contents)


def _place_system(
  contents: list[dict[str, Any]],
  system_text: str,
  placement: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
  if not system_text:
    return contents, None
  if placement == "first_user_turn":
    placed = [dict(content, parts=list(content["parts"])) for content in contents]
    previous = ""
    for content in placed:
      if content["role"] == "user":
        content["parts"] = [{"text": system_text}, *content["parts"]]
        content["parts"] = _group_function_responses(
          content["parts"],
          after_model=previous == "model",
        )
        return placed, None
      previous = str(content["role"])
    return [{"role": "user", "parts": [{"text": system_text}]}, *placed], None
  # Antigravity tags systemInstruction with role "user". The caller's text is
  # otherwise unmodified.
  return contents, {"role": "user", "parts": [{"text": system_text}]}


def _signed_session_id(text: str) -> str:
  digest = hashlib.sha256(text.encode("utf-8")).digest()
  value = int.from_bytes(digest[:8], "big") & _INT63_MASK
  return f"-{value}"


def _random_session_id() -> str:
  return f"-{secrets.randbelow(_SESSION_RANDOM_BOUND)}"


def _first_user_text(contents: list[dict[str, Any]]) -> str:
  for content in contents:
    if content.get("role") != "user":
      continue
    parts = content.get("parts")
    if not isinstance(parts, list):
      continue
    for part in parts:
      if not isinstance(part, dict):
        continue
      text = part.get("text")
      if isinstance(text, str) and text.strip():
        return text
  return ""


def _envelope(
  *,
  project_id: str,
  model: str,
  inner: dict[str, Any],
) -> dict[str, Any]:
  """Cloud Code Assist envelope: project, requestId, userAgent, requestType, labels.

  Without a stored session this matches omp's no-state fallback: ephemeral
  agent and trajectory ids, step 2, and a session id derived from the first
  user text. ``labels.model_enum`` is a per-wire-id telemetry token; this
  adapter does not branch on model id, so it is omitted.
  """

  agent_id = str(uuid.uuid4())
  trajectory_id = str(uuid.uuid4())
  step = 2
  raw_text = inner.pop("_session_text", None)
  text = raw_text if isinstance(raw_text, str) else ""
  session_id = _signed_session_id(text) if text.strip() else _random_session_id()
  request: dict[str, Any] = dict(inner)
  request["sessionId"] = session_id
  request["labels"] = {
    "last_step_index": str(step - 1),
    "trajectory_id": trajectory_id,
    "used_claude": "false",
    "used_claude_conservative": "false",
  }
  return {
    "project": project_id,
    "requestId": f"agent/{agent_id}/{int(time.time() * 1000)}/{trajectory_id}/{step}",
    "request": request,
    "model": model,
    "userAgent": _BODY_USER_AGENT,
    "requestType": _REQUEST_TYPE,
  }


def _parse_sse(buffer: str) -> tuple[list[Any], str]:
  events: list[Any] = []
  normalized = buffer.replace("\r\n", "\n").replace("\r", "\n")
  idx = normalized.find("\n\n")
  while idx != -1:
    chunk = normalized[:idx]
    normalized = normalized[idx + 2:]
    data_lines = []
    for line in chunk.split("\n"):
      if line.startswith("data:"):
        data_lines.append(line[5:].strip())
    if data_lines:
      data = "\n".join(data_lines).strip()
      if data and data != "[DONE]":
        try:
          parsed = json.loads(data)
        except json.JSONDecodeError:
          parsed = None
        if parsed is not None:
          events.append(parsed)
    idx = normalized.find("\n\n")
  return events, normalized


def _searchable(exc: Exception) -> str:
  code = _error_code_from_exception(exc) or ""
  return " ".join(part for part in (str(exc), _response_text(exc), code) if part)


def _google_status(exc: Exception) -> str | None:
  body = _response_text(exc) or str(exc)
  match = _GOOGLE_STATUS_RE.search(body)
  if match:
    return match.group(1)
  try:
    parsed = json.loads(body)
  except json.JSONDecodeError:
    return None
  if not isinstance(parsed, dict):
    return None
  error = parsed.get("error")
  if isinstance(error, dict):
    status = error.get("status")
    if isinstance(status, str) and status:
      return status
  return None


def _is_content_refusal(exc: Exception) -> bool:
  return _CONTENT_REFUSAL_RE.search(_searchable(exc)) is not None


def _is_usage_limit_only(exc: Exception) -> bool:
  """True when the body is a rate/usage/quota exhaustion and nothing else."""

  status = _status_code_from_exception(exc)
  if status in {400, 401, 403}:
    return False
  if _is_context_length_exception(exc) or _is_content_refusal(exc):
    return False
  google_status = _google_status(exc)
  if google_status == "RESOURCE_EXHAUSTED":
    return True
  if google_status in {"PERMISSION_DENIED", "UNAUTHENTICATED", "INVALID_ARGUMENT"}:
    return False
  return _USAGE_LIMIT_RE.search(_searchable(exc)) is not None


def _retry_after_raw(exc: Exception) -> str | None:
  response = getattr(exc, "response", None)
  headers = getattr(response, "headers", None)
  if headers is None:
    headers = getattr(exc, "headers", None)
  getter = getattr(headers, "get", None)
  if callable(getter):
    raw = getter("retry-after")
    text = str(raw).strip() if raw is not None else ""
    if text:
      return text
  attached = getattr(exc, "retry_after", None)
  if isinstance(attached, str) and attached.strip():
    return attached.strip()
  return None


def _retry_after_reset_at(exc: Exception, *, now: float | None = None) -> float | None:
  """Epoch seconds from ``Retry-After`` (delta-seconds or HTTP-date), else None."""

  raw = _retry_after_raw(exc)
  if raw is None:
    return None
  moment = time.time() if now is None else now
  try:
    seconds = float(raw)
  except ValueError:
    seconds = None
  if seconds is not None:
    if seconds < 0:
      return None
    return moment + seconds
  try:
    parsed = parsedate_to_datetime(raw)
  except (TypeError, ValueError, OverflowError, IndexError):
    return None
  if parsed.tzinfo is None:
    parsed = parsed.replace(tzinfo=timezone.utc)
  return parsed.timestamp()


def _failure(
  exc: Exception,
  *,
  kind: CredentialFailureKind,
  status: int | None,
) -> ProviderCredentialFailure:
  google_status = _google_status(exc)
  message = " ".join(part for part in (str(exc), _response_text(exc)) if part).strip()
  return ProviderCredentialFailure(
    provider=GoogleAntigravityProvider.name,
    kind=kind,
    status_code=status,
    error_code=google_status or _error_code_from_exception(exc),
    message=message,
    reset_at=_retry_after_reset_at(exc) if kind == "rate_limit" else None,
  )


class _BodyResponse:
  def __init__(self, status_code: int, text: str, headers: Any = None) -> None:
    self.status_code = status_code
    self.text = text
    self.headers = headers or {}


class _GeminiStatusError(Exception):
  """An in-band error event on an accepted (200) stream.

  ``mid_stream`` is true when the backend had already delivered progress on
  this attempt before aborting: the request shape was accepted, so the
  attempt is retried like a dropped connection. The first event of a stream
  being an error is the backend's answer to the request itself.
  """

  def __init__(
    self, message: str, *, status_code: int, body: str, headers: Any = None, mid_stream: bool = False,
  ) -> None:
    super().__init__(message)
    self.status_code = status_code
    self.response = _BodyResponse(status_code, body, headers)
    self.mid_stream = mid_stream


def _raise_error_payload(
  payload: dict[str, Any], *, headers: Any = None, state: "_StreamState | None" = None,
) -> None:
  error = payload.get("error")
  error_obj = error if isinstance(error, dict) else {}
  status_code = error_obj.get("code")
  if not isinstance(status_code, int) or isinstance(status_code, bool):
    status_code = 400
  status = str(error_obj.get("status") or "")
  message = str(error_obj.get("message") or status or "google_antigravity API error")
  body = json.dumps(payload)
  raise _GeminiStatusError(
    f"google_antigravity API error (status={status_code}): {message}",
    status_code=status_code,
    body=body,
    headers=headers,
    mid_stream=state is not None and state.started,
  )


def _unwrap_generate_content(
  payload: dict[str, Any], state: "_StreamState | None" = None,
) -> dict[str, Any] | None:
  """Return the generateContent chunk inside ``{response: ...}``.

  An in-band ``error`` with no chunk is raised. A bare generateContent object
  is accepted so a caller that already unwrapped still maps.
  """

  response = payload.get("response")
  error = payload.get("error")
  if isinstance(error, dict) and not isinstance(response, dict):
    _raise_error_payload(payload, state=state)
  if isinstance(response, dict):
    if "error" in response and "candidates" not in response:
      _raise_error_payload(response, state=state)
    return response
  if any(
    key in payload
    for key in ("candidates", "usageMetadata", "usage_metadata", "promptFeedback", "prompt_feedback")
  ):
    return payload
  return None


async def _error_message(response: httpx.Response) -> str:
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
      message = str(error.get("message") or error.get("status") or message)
    elif isinstance(error, str) and error:
      message = error
  return f"google_antigravity API error (status={response.status_code}): {message}"


def _map_stop(finish: str, block: str, *, saw_tool: bool) -> str:
  finish_u = finish.strip().upper()
  block_u = block.strip().upper()
  if finish_u in _CONTENT_ERROR_REASONS or block_u in _CONTENT_ERROR_REASONS:
    return "error"
  if block_u and block_u != "BLOCK_REASON_UNSPECIFIED":
    return "error"
  if finish_u in _FAILURE_FINISH_REASONS:
    return "error"
  if saw_tool and finish_u != "MAX_TOKENS":
    return "tool_use"
  if finish_u == "MAX_TOKENS":
    return "max_tokens"
  if saw_tool:
    return "tool_use"
  if finish_u in {"", "STOP", "FINISH_REASON_UNSPECIFIED"}:
    return "end_turn"
  return "error"


@dataclass
class _OpenText:
  kind: str
  text: str = ""
  signature: str = ""


@dataclass
class _OpenTool:
  tool_id: str
  name: str
  args: Any = None
  args_json: str = ""
  signature: str = ""
  started: bool = False


@dataclass
class _StreamState:
  started: bool = False
  model: str | None = None
  text: _OpenText | None = None
  tools: list[_OpenTool] = field(default_factory=list)
  tool_index: int = 0
  saw_tool: bool = False
  finish_reason: str = ""
  block_reason: str = ""
  safety_ratings: Any = None
  prompt_tokens: int | None = None
  cached_tokens: int = 0
  candidates_tokens: int = 0
  thoughts_tokens: int = 0
  reported_input: int = 0
  reported_cache: int = 0
  reported_output: int = 0
  reported_reasoning: int = 0
  terminal: bool = False


def _int_field(value: Any) -> int:
  if isinstance(value, bool) or value is None:
    return 0
  try:
    return int(value)
  except (TypeError, ValueError):
    return 0


def _usage_events(state: _StreamState, *, starting: bool) -> list[StreamEvent]:
  events: list[StreamEvent] = []
  input_total = 0
  if state.prompt_tokens is not None:
    input_total = max(0, state.prompt_tokens - state.cached_tokens)
  if starting and not state.started:
    state.started = True
    state.reported_input = input_total
    state.reported_cache = state.cached_tokens
    events.append(StreamEvent(
      type="message_start",
      provider_reported_model=state.model,
      input_tokens=input_total,
      cache_read_tokens=state.cached_tokens,
    ))
    return events
  if not state.started:
    return _usage_events(state, starting=True)
  output_total = state.candidates_tokens + state.thoughts_tokens
  input_delta = max(0, input_total - state.reported_input)
  cache_delta = max(0, state.cached_tokens - state.reported_cache)
  output_delta = max(0, output_total - state.reported_output)
  reasoning_delta = max(0, state.thoughts_tokens - state.reported_reasoning)
  if input_delta or cache_delta or output_delta or reasoning_delta:
    state.reported_input = max(state.reported_input, input_total)
    state.reported_cache = state.cached_tokens
    state.reported_output = output_total
    state.reported_reasoning = state.thoughts_tokens
    events.append(StreamEvent(
      type="usage_update",
      input_tokens=input_delta,
      cache_read_tokens=cache_delta,
      output_tokens=output_delta,
      reasoning_tokens=reasoning_delta,
    ))
  return events


def _close_text(state: _StreamState) -> list[StreamEvent]:
  current = state.text
  if current is None:
    return []
  state.text = None
  if current.kind == "thinking":
    block: dict[str, Any] = {"type": "thinking", "thinking": current.text}
    if current.signature:
      block["signature"] = current.signature
      block["thoughtSignature"] = current.signature
    return [StreamEvent(
      type="thinking_end",
      thinking_text=current.text,
      signature=current.signature,
      raw_block=block,
    )]
  block = {"type": "text", "text": current.text}
  if current.signature:
    block["thoughtSignature"] = current.signature
  return [StreamEvent(type="text_end", text=current.text, raw_block=block)]


def _append_text(state: _StreamState, kind: str, delta: str, signature: str) -> list[StreamEvent]:
  events: list[StreamEvent] = []
  if state.text is not None and state.text.kind != kind:
    events.extend(_close_text(state))
  if state.text is None:
    state.text = _OpenText(kind=kind)
  state.text.text += delta
  if signature:
    state.text.signature = signature
  if not delta:
    return events
  if kind == "thinking":
    events.append(StreamEvent(type="thinking_delta", thinking_text=delta))
  else:
    events.append(StreamEvent(type="text_delta", text=delta))
  return events


def _upsert_tool(state: _StreamState, call: dict[str, Any], signature: str) -> list[StreamEvent]:
  tool_id = str(call.get("id") or "")
  name = str(call.get("name") or "")
  args = call.get("args")
  args_json = _args_json(args)
  existing = next((tool for tool in state.tools if tool_id and tool.tool_id == tool_id), None)
  events: list[StreamEvent] = []
  if existing is None:
    state.tool_index += 1
    if not tool_id:
      tool_id = f"fc-{state.tool_index}"
    existing = _OpenTool(tool_id=tool_id, name=name, signature=signature)
    state.tools.append(existing)
    state.saw_tool = True
  if name and not existing.name:
    existing.name = name
  if signature and not existing.signature:
    existing.signature = signature
  if not existing.started:
    existing.started = True
    raw = {"type": "tool_use", "id": existing.tool_id, "name": existing.name}
    if existing.signature:
      raw["thoughtSignature"] = existing.signature
    events.append(StreamEvent(
      type="tool_use_start",
      tool_id=existing.tool_id,
      tool_name=existing.name,
      raw_block=raw,
    ))
  if args_json and args_json != existing.args_json:
    if existing.args_json and args_json.startswith(existing.args_json):
      delta = args_json[len(existing.args_json):]
    else:
      delta = args_json
    existing.args = args
    existing.args_json = args_json
    if delta:
      events.append(StreamEvent(type="tool_use_delta", tool_input_json=delta))
  return events


def _close_tools(state: _StreamState) -> list[StreamEvent]:
  events: list[StreamEvent] = []
  for tool in state.tools:
    if not tool.started:
      continue
    tool_input = _args_dict(tool.args, tool.args_json)
    raw: dict[str, Any] = {
      "type": "tool_use",
      "id": tool.tool_id,
      "name": tool.name,
      "input": tool_input,
    }
    if tool.signature:
      raw["thoughtSignature"] = tool.signature
    events.append(StreamEvent(
      type="tool_use_end",
      tool_id=tool.tool_id,
      tool_name=tool.name,
      tool_input_json=tool.args_json,
      tool_input=tool_input,
      raw_block=raw,
    ))
    tool.started = False
  return events


def _read_usage(state: _StreamState, payload: dict[str, Any]) -> None:
  usage = _field(payload, "usageMetadata", "usage_metadata")
  if not isinstance(usage, dict):
    return
  prompt = _field(usage, "promptTokenCount", "prompt_token_count")
  if prompt is not None:
    state.prompt_tokens = _int_field(prompt)
  state.cached_tokens = _int_field(_field(usage, "cachedContentTokenCount", "cached_content_token_count"))
  state.candidates_tokens = _int_field(_field(usage, "candidatesTokenCount", "candidates_token_count"))
  state.thoughts_tokens = _int_field(_field(usage, "thoughtsTokenCount", "thoughts_token_count"))
  reported = str(_field(payload, "modelVersion", "model_version") or "").strip()
  if reported:
    state.model = reported


def _apply_payload(state: _StreamState, payload: dict[str, Any]) -> list[StreamEvent]:
  if "error" in payload and "candidates" not in payload:
    _raise_error_payload(payload, state=state)
  events: list[StreamEvent] = []
  _read_usage(state, payload)
  feedback = _field(payload, "promptFeedback", "prompt_feedback")
  if isinstance(feedback, dict):
    block = str(_field(feedback, "blockReason", "block_reason") or "")
    if block:
      state.block_reason = block
    ratings = _field(feedback, "safetyRatings", "safety_ratings")
    if ratings is not None:
      state.safety_ratings = ratings
  if not state.started:
    events.extend(_usage_events(state, starting=True))
  candidates = payload.get("candidates")
  candidate = candidates[0] if isinstance(candidates, list) and candidates else {}
  if not isinstance(candidate, dict):
    candidate = {}
  finish = str(_field(candidate, "finishReason", "finish_reason") or "")
  if finish:
    state.finish_reason = finish
  ratings = _field(candidate, "safetyRatings", "safety_ratings")
  if ratings is not None:
    state.safety_ratings = ratings
  content = candidate.get("content") if isinstance(candidate.get("content"), dict) else {}
  parts = content.get("parts") if isinstance(content, dict) else None
  if isinstance(parts, list):
    for part in parts:
      if not isinstance(part, dict):
        continue
      call = _field(part, "functionCall", "function_call")
      if isinstance(call, dict):
        events.extend(_close_text(state))
        signature = str(_field(part, "thoughtSignature", "thought_signature") or "")
        events.extend(_upsert_tool(state, call, signature))
        continue
      text = part.get("text")
      if not isinstance(text, str) or not text:
        continue
      kind = "thinking" if part.get("thought") is True else "text"
      signature = str(_field(part, "thoughtSignature", "thought_signature") or "")
      events.extend(_append_text(state, kind, text, signature))
  events.extend(_usage_events(state, starting=False))
  return events


def _finalize(state: _StreamState) -> list[StreamEvent]:
  if state.terminal:
    return []
  state.terminal = True
  events: list[StreamEvent] = []
  if not state.started:
    events.extend(_usage_events(state, starting=True))
  events.extend(_close_text(state))
  events.extend(_close_tools(state))
  events.extend(_usage_events(state, starting=False))
  details: dict[str, Any] = {}
  if state.finish_reason:
    details["finish_reason"] = state.finish_reason
  if state.block_reason:
    details["block_reason"] = state.block_reason
  if state.safety_ratings is not None:
    details["safety_ratings"] = state.safety_ratings
  events.append(StreamEvent(
    type="message_end",
    stop_reason=_map_stop(state.finish_reason, state.block_reason, saw_tool=state.saw_tool),
    stop_details=details or None,
    provider_reported_model=state.model,
    request_input_tokens=state.prompt_tokens,
  ))
  return events


def _request_headers(state: dict[str, Any], extra: Any) -> dict[str, str]:
  merged: dict[str, Any] = dict(state.get("headers") or {})
  if isinstance(extra, dict):
    merged.update(extra)
  _drop_header(merged, "authorization")
  merged.update(state.get("auth_headers") or {})
  merged["Content-Type"] = "application/json"
  merged["Accept"] = "text/event-stream"
  if not any(str(key).lower() == "user-agent" for key in merged):
    merged["User-Agent"] = state.get("user_agent") or _user_agent(DEFAULT_ANTIGRAVITY_CLIENT_VERSION)
  return {str(key): str(value) for key, value in merged.items()}


class GoogleAntigravityProvider(ModelProvider):
  """Gemini over Cloud Code Assist (Antigravity). Not the public Gemini API."""

  name = "google_antigravity"

  @classmethod
  def adapter_route_support(cls) -> AdapterRouteSupport:
    return AdapterRouteSupport(
      adapter="google_antigravity.code_assist",
      provider="google_antigravity",
      protocol_profiles=frozenset({"generate_content.standard"}),
      routes=frozenset({"google.antigravity"}),
    )

  def __init__(self, *, rate_table: RateTable | None = None) -> None:
    self._rate_table = _rate_table_for(self.name, rate_table)
    self._client_state: WeakKeyDictionary[httpx.AsyncClient, dict[str, Any]] = WeakKeyDictionary()

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(_credential(config.get("auth_token")) and _credential(config.get("project_id")))

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    project_id = _credential(config.get("project_id"))
    if not project_id:
      raise RuntimeError(
        "google_antigravity project_id is absent; run `agent auth login gemini`"
      )
    token = _header_value(_credential(config.get("auth_token")), name="auth_token")
    if not token:
      raise RuntimeError("No google_antigravity credential configured")
    version = _client_version(config)
    client_kwargs: dict[str, Any] = {
      "timeout": httpx.Timeout(timeout=timeout, connect=10.0),
    }
    transport = config.get("_transport")
    if isinstance(transport, httpx.AsyncBaseTransport):
      client_kwargs["transport"] = transport
    client = httpx.AsyncClient(**client_kwargs)
    extra_headers = config.get("headers")
    self._client_state[client] = {
      "auth_headers": {"Authorization": f"Bearer {token}"},
      "base_url": _bound_base_url(config),
      "headers": dict(extra_headers) if isinstance(extra_headers, dict) else {},
      "project_id": project_id,
      "user_agent": _user_agent(version),
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
    return _model_info(model, self._rate_table)

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
    supported = tuple(ThinkingLevel(value) for value in model_info.effort_values)
    if not supported:
      if requested != ThinkingLevel.NONE:
        raise ValueError(
          f"google_antigravity model {model!r} does not support reasoning effort "
          f"{requested.value!r}; unsupported effort is refused, never repaired"
        )
      return EffortResolution(requested, ThinkingLevel.NONE, False, {})
    if requested not in supported:
      raise ValueError(
        f"google_antigravity model {model!r} does not support reasoning effort "
        f"{requested.value!r} (supported: "
        f"{', '.join(level.value for level in supported)}); unsupported "
        "effort is refused, never clamped"
      )
    fragments: dict[str, Any] = {}
    control = _compat(model_info).thinking_control
    if control is not None and requested == ThinkingLevel.NONE and control.off_value is not None:
      _assign_dotted(fragments, control.field, control.off_value)
    elif control is not None and requested != ThinkingLevel.NONE and control.off_value is None:
      _assign_dotted(fragments, control.field, requested.value)
    return EffortResolution(
      requested=requested,
      effective=requested,
      thinking_enabled_effective=requested != ThinkingLevel.NONE,
      payload_fragments=fragments,
    )

  def normalize_messages(self, messages: list[dict[str, Any]], model_info: ModelInfo) -> list[dict[str, Any]]:
    del model_info
    normalized: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    seen: set[str] = set()
    for message in messages:
      if not isinstance(message, dict):
        continue
      role = str(message.get("role") or "")
      if role == "assistant" and str(message.get("stop_reason") or "") in {"error", "aborted"}:
        continue
      if pending and role == "tool":
        tool_id = str(message.get("tool_use_id") or message.get("tool_call_id") or "")
        if tool_id:
          seen.add(tool_id)
      if pending and role == "user":
        seen.update(_tool_result_ids(message))
      if pending and role not in {"user", "tool"}:
        missing = [
          _synthetic_tool_result(str(block.get("id") or ""), str(block.get("name") or ""))
          for block in pending
          if str(block.get("id") or "") not in seen
        ]
        if missing:
          normalized.append({"role": "user", "content": missing})
        pending = []
        seen = set()
      normalized.append(message)
      if role != "assistant":
        continue
      content = message.get("content")
      if not isinstance(content, list):
        pending = []
        seen = set()
        continue
      pending = [
        block for block in content
        if isinstance(block, dict) and block.get("type") in {"tool_use", "server_tool_use"}
      ]
      seen = set()
    if pending:
      missing = [
        _synthetic_tool_result(str(block.get("id") or ""), str(block.get("name") or ""))
        for block in pending
        if str(block.get("id") or "") not in seen
      ]
      if missing:
        normalized.append({"role": "user", "content": missing})
    return normalized

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
    compat = _compat(model_info)
    effort_resolution = kwargs.get("effort_resolution")
    if not isinstance(effort_resolution, EffortResolution):
      raise ValueError(
        "google_antigravity build_request_params requires an admitted EffortResolution; "
        "it does not resolve effort locally"
      )
    normalized = self.normalize_messages(messages, model_info)
    contents = _convert_contents(
      normalized,
      model_info,
      skip_on_first=_skip_thought_on_first(compat),
    )
    placement = compat.system_instruction_placement or "system_instruction"
    contents, system_instruction = _place_system(contents, _system_text(system_prompt), placement)
    params: dict[str, Any] = {
      "model": model,
      "contents": contents,
      "generationConfig": {"maxOutputTokens": max_tokens},
    }
    params["_session_text"] = _first_user_text(contents)
    if system_instruction is not None:
      params["systemInstruction"] = system_instruction
    converted_tools = _convert_tools(tools)
    if converted_tools:
      params["tools"] = converted_tools
      # Antigravity's default tool mode. An omitted toolConfig is not AUTO.
      params["toolConfig"] = {"functionCallingConfig": {"mode": "VALIDATED"}}
    safety = _safety_settings(compat.safety_settings_default)
    if safety:
      params["safetySettings"] = safety
    _deep_merge(params, dict(effort_resolution.payload_fragments))
    if effort_resolution.thinking_enabled_effective:
      _assign_dotted(params, "generationConfig.thinkingConfig.includeThoughts", True)
    headers = kwargs.get("headers")
    if isinstance(headers, dict) and headers:
      params["_headers"] = dict(headers)
    if kwargs.get("base_url"):
      params["_endpoint_base"] = str(kwargs["base_url"]).strip().rstrip("/")
    return params

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
    if not isinstance(client, httpx.AsyncClient):
      raise RuntimeError("GoogleAntigravityProvider requires an httpx.AsyncClient")
    state = self._client_state.get(client)
    if state is None:
      raise RuntimeError("google_antigravity client state is missing")
    request_params = dict(params)
    extra_headers = request_params.pop("_headers", None)
    base_url = str(request_params.pop("_endpoint_base", state["base_url"]) or state["base_url"])
    model = str(request_params.pop("model", "") or "")
    project_id = str(state.get("project_id") or "")
    if not project_id:
      raise RuntimeError(
        "google_antigravity project_id is absent; run `agent auth login gemini`"
      )
    envelope = _envelope(project_id=project_id, model=model, inner=request_params)
    endpoint = _stream_url(base_url)
    headers = _request_headers(state, extra_headers)
    stream_state = _StreamState()
    async with client.stream("POST", endpoint, json=envelope, headers=headers) as response:
      if response.status_code != 200:
        message = await _error_message(response)
        raise httpx.HTTPStatusError(message, request=response.request, response=response)
      buffer = ""
      saw_event = False
      async for chunk in response.aiter_text():
        buffer += chunk
        events, buffer = _parse_sse(buffer)
        for event in events:
          saw_event = True
          payloads = event if isinstance(event, list) else [event]
          for payload in payloads:
            if not isinstance(payload, dict):
              continue
            inner = _unwrap_generate_content(payload, stream_state)
            if inner is None:
              continue
            for mapped in _apply_payload(stream_state, inner):
              yield mapped
      if buffer.strip():
        events, _rest = _parse_sse(buffer + "\n\n")
        if not events and buffer.strip().startswith("{"):
          try:
            parsed = json.loads(buffer)
          except json.JSONDecodeError:
            parsed = None
          if isinstance(parsed, dict):
            events = [parsed]
        for event in events:
          saw_event = True
          payloads = event if isinstance(event, list) else [event]
          for payload in payloads:
            if not isinstance(payload, dict):
              continue
            inner = _unwrap_generate_content(payload, stream_state)
            if inner is None:
              continue
            for mapped in _apply_payload(stream_state, inner):
              yield mapped
      if not saw_event and not stream_state.terminal:
        raise RuntimeError("google_antigravity stream ended without a generateContent event")
    for mapped in _finalize(stream_state):
      yield mapped

  def is_retryable_error(self, exc: Exception) -> bool:
    if isinstance(exc, _GeminiStatusError) and exc.mid_stream:
      # The backend accepted the request and streamed progress, then aborted
      # (PAYC :8256 2026-10-13: 400 INVALID_ARGUMENT after 9.8 s of thinking;
      # the byte-identical request replayed 200). That is a dropped attempt,
      # not an invalid request.
      return True
    status = _status_code_from_exception(exc)
    if status in _RETRYABLE_STATUSES or (isinstance(status, int) and 500 <= status <= 599):
      return True
    return isinstance(exc, (httpx.TransportError, httpx.StreamError))

  def is_context_length_error(self, exc: Exception) -> bool:
    return _is_context_length_exception(exc)

  def classify_credential_failure(self, exc: Exception) -> ProviderCredentialFailure | None:
    status = _status_code_from_exception(exc)
    if status == 429 or _is_usage_limit_only(exc):
      return _failure(exc, kind="rate_limit", status=status if status is not None else 429)
    if status == 401:
      return _failure(exc, kind="auth", status=401)
    if status == 403:
      return _failure(exc, kind="billing", status=403)
    if status == 400 or _is_context_length_exception(exc) or _is_content_refusal(exc):
      return None
    return _classify_provider_credential_failure(provider=self.name, exc=exc)

  def next_credential(
    self,
    config: Mapping[str, Any],
    failure: ProviderCredentialFailure,
  ) -> dict[str, Any] | None:
    from . import google_antigravity_oauth  # noqa: F401

    return rotate_provider_credential(
      self.name,
      config,
      failure_kind=failure.kind,
      reset_at=failure.reset_at,
      status_code=failure.status_code,
    )


__all__ = ["GoogleAntigravityProvider"]
