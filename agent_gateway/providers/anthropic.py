from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, replace
from typing import Any, AsyncGenerator, Dict, Literal, Mapping

from ..auth import ProviderCredentialFailure
from ..rate_limit import get_global_token_bucket
from model_authority.rates import RateTable
from model_authority.registry import AdapterRouteSupport
from model_authority.schema import AnthropicMessagesCompat
from model_authority.thinking import EffortResolution, ThinkingLevel, clamp_effort
from .base import (
  ModelInfo,
  ModelProvider,
  StreamEvent,
  _BILLING_PATTERNS,
  _credential_reset_at,
  _projected_epoch_seconds,
  admitted_entry,
  authority_rate_table,
  model_info_from_entry,
  truncate_to_last_compaction,
)
from .anthropic_helpers import (
  _COMMON_BETA_SLUGS as _COMMON_BETA_SLUGS,
  _COMPACTION_BETA_SLUG as _COMPACTION_BETA_SLUG,
  _ERROR_REDACTION as _ERROR_REDACTION,
  _MAX_ERROR_DETAIL_LEN as _MAX_ERROR_DETAIL_LEN,
  _MAX_TOOL_ID_LEN as _MAX_TOOL_ID_LEN,
  _OAUTH_BETA_SLUGS as _OAUTH_BETA_SLUGS,
  _OAUTH_IDENTITY as _OAUTH_IDENTITY,
  _SENSITIVE_ERROR_KEY_RE as _SENSITIVE_ERROR_KEY_RE,
  _SENSITIVE_ERROR_VALUE_RES as _SENSITIVE_ERROR_VALUE_RES,
  _STRUCTURED_OUTPUTS_BETA_SLUG as _STRUCTURED_OUTPUTS_BETA_SLUG,
  _TOOL_ID_RE as _TOOL_ID_RE,
  _exception_body as _exception_body,
  _exception_status_code as _exception_status_code,
  _format_anthropic_rejection_detail as _format_anthropic_rejection_detail,
  _has_tool_result_block as _has_tool_result_block,
  _normalize_tool_call_id as _normalize_tool_call_id,
  _redact_error_body as _redact_error_body,
  _response_header as _response_header,
  _same_model_message as _same_model_message,
  _stream_request_context as _stream_request_context,
  _synthetic_tool_result as _synthetic_tool_result,
  _thinking_param as _thinking_param,
  _to_plain_dict as _to_plain_dict,
  _truncate_error_detail as _truncate_error_detail,
)
from . import anthropic_quota
from .anthropic_oauth import (
  ANTHROPIC_DEPLOYMENT_API_KEY_FIELD,
  CREDENTIAL_AFFINITY_KEY_FIELD,
  CREDENTIAL_PREPARED_FIELD,
  UNDECLARED_LIMIT_BLOCK_SECONDS,
  AnthropicCredential,
  AnthropicCredentialSources,
  AnthropicRequestBinding,
  anthropic_request_binding,
  anthropic_pool_config,
  record_anthropic_success,
  ensure_fresh_anthropic_credential,
  resolve_anthropic_credentials,
  rotate_anthropic_credential_source,
  select_anthropic_credential_source,
)


log = logging.getLogger("agent_gateway.providers.anthropic")
_DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
# The model family the bound request serves ("opus", "sonnet", ...), carried in
# the auth material `prepare_credential` returns so a rotation after a failed
# stream ranks siblings for the same family. Opaque and not a secret.
CREDENTIAL_MODEL_FAMILY_FIELD = "credential_model_family"
# Failures a sibling credential can actually answer: a spent subscription window
# and a credential this account can no longer authenticate. Billing and request
# rejections say the same thing on every sibling, so they are not rotated.
_ROTATABLE_CREDENTIAL_FAILURES = frozenset({"auth", "rate_limit"})
_MESSAGE_CACHE_CONTROL = {"type": "ephemeral"}
_MESSAGE_CACHEABLE_BLOCK_TYPES = frozenset({
  "bash_code_execution_tool_result",
  "code_execution_tool_result",
  "container_upload",
  "document",
  "image",
  "search_result",
  "server_tool_use",
  "text",
  "text_editor_code_execution_tool_result",
  "tool_reference",
  "tool_result",
  "tool_search_tool_result",
  "tool_use",
  "web_fetch_tool_result",
  "web_search_tool_result",
})
_MessageCacheBreakpointMode = Literal["normal", "fork"]


def _is_cacheable_message_block(block: Any) -> bool:
  if not isinstance(block, dict):
    return False
  block_type = block.get("type")
  if block_type not in _MESSAGE_CACHEABLE_BLOCK_TYPES:
    return False
  if block_type == "text":
    return bool(str(block.get("text", "")))
  return True


def _message_cache_marker_locations(
  messages: list[dict[str, Any]],
) -> list[tuple[int, int]]:
  locations: list[tuple[int, int]] = []
  for message_index, message in enumerate(messages):
    content = message.get("content")
    if not isinstance(content, list):
      continue
    locations.extend(
      (message_index, block_index)
      for block_index, block in enumerate(content)
      if isinstance(block, dict) and "cache_control" in block
    )
  return locations


def _assert_message_cache_breakpoint_placement(
  messages: list[dict[str, Any]],
  *,
  expected_location: tuple[int, int] | None,
  mode: _MessageCacheBreakpointMode,
) -> None:
  if messages and mode == "normal":
    assert messages[-1].get("role") == "user", (
      "final Anthropic request message must have user role"
    )
  marker_locations = _message_cache_marker_locations(messages)
  if expected_location is None:
    assert marker_locations == [], (
      f"expected no message cache marker, found {marker_locations}"
    )
    return

  assert marker_locations == [expected_location], (
    f"expected one message cache marker at {expected_location}, found {marker_locations}"
  )
  if mode == "fork":
    content = messages[expected_location[0]].get("content")
    assert isinstance(content, list)
    assert _is_cacheable_message_block(content[expected_location[1]]), (
      "fork message cache marker must remain on a cacheable pinned block"
    )
    return

  final_message_index = len(messages) - 1
  assert expected_location[0] == final_message_index, (
    "message cache marker must be on the final message"
  )
  final_content = messages[final_message_index].get("content")
  assert isinstance(final_content, list)
  cacheable_indices = [
    index
    for index, block in enumerate(final_content)
    if _is_cacheable_message_block(block)
  ]
  assert cacheable_indices and expected_location[1] == cacheable_indices[-1], (
    "message cache marker must be on the final message's last cacheable block"
  )


def _normalize_bare_string_message_content(
  message: dict[str, Any],
) -> Any:
  content = message.get("content")
  if isinstance(content, str):
    content = [{"type": "text", "text": content}]
    message["content"] = content
  return content


def _place_message_cache_breakpoint(
  messages: list[dict[str, Any]],
  *,
  mode: _MessageCacheBreakpointMode,
  allow_marker: bool = True,
  pinned_position: tuple[int, int] | None = None,
) -> tuple[list[dict[str, Any]], tuple[int, int] | None]:
  """Copy messages and authoritatively place the selected cache marker arm."""
  if mode not in {"normal", "fork"}:
    raise ValueError(f"unsupported message cache breakpoint mode: {mode}")

  cleaned_messages: list[dict[str, Any]] = []
  for message in messages:
    next_message = dict(message)
    next_message.pop("cache_control", None)
    content = message.get("content")
    if isinstance(content, list):
      next_content: list[Any] = []
      for block in content:
        if isinstance(block, dict):
          next_block = dict(block)
          next_block.pop("cache_control", None)
          next_content.append(next_block)
        else:
          next_content.append(block)
      next_message["content"] = next_content
    cleaned_messages.append(next_message)

  if mode == "fork":
    if pinned_position is None:
      raise ValueError("fork cache breakpoint mode requires a pinned position")
    message_index, block_index = pinned_position
    if (
      not allow_marker
      or message_index >= len(cleaned_messages)
      or message_index < 0
    ):
      raise ValueError(
        "fork cache breakpoint cannot place its required pinned marker"
      )
    content = _normalize_bare_string_message_content(
      cleaned_messages[message_index]
    )
    if (
      not isinstance(content, list)
      or block_index < 0
      or block_index >= len(content)
      or not _is_cacheable_message_block(content[block_index])
    ):
      raise ValueError(
        "fork cache breakpoint pinned position is not a cacheable block"
      )
    marked_block = dict(content[block_index])
    marked_block["cache_control"] = dict(_MESSAGE_CACHE_CONTROL)
    content[block_index] = marked_block
    return cleaned_messages, pinned_position

  if not cleaned_messages:
    log.debug("Skipping Anthropic message cache breakpoint: request has no messages")
    return cleaned_messages, None

  final_message = cleaned_messages[-1]
  if final_message.get("role") != "user":
    log.debug(
      "Skipping Anthropic message cache breakpoint: final message role is not user"
    )
    return cleaned_messages, None

  final_content = _normalize_bare_string_message_content(final_message)

  if not isinstance(final_content, list):
    log.debug(
      "Skipping Anthropic message cache breakpoint: final message has no cacheable block"
    )
    return cleaned_messages, None

  marker_index = next(
    (
      index
      for index in range(len(final_content) - 1, -1, -1)
      if _is_cacheable_message_block(final_content[index])
    ),
    None,
  )
  if marker_index is None:
    log.debug(
      "Skipping Anthropic message cache breakpoint: final message has no cacheable block"
    )
    return cleaned_messages, None

  if not allow_marker:
    log.debug(
      "Skipping Anthropic message cache breakpoint: request already carries four "
      "explicit cache breakpoints"
    )
    return cleaned_messages, None

  marked_block = dict(final_content[marker_index])
  marked_block["cache_control"] = dict(_MESSAGE_CACHE_CONTROL)
  final_content[marker_index] = marked_block
  return cleaned_messages, (len(cleaned_messages) - 1, marker_index)


def _explicit_cache_breakpoint_count(params: dict[str, Any]) -> int:
  system = params.get("system")
  tools = params.get("tools")
  messages = params.get("messages")
  return (
    sum(
      1
      for block in system if isinstance(block, dict) and "cache_control" in block
    )
    if isinstance(system, list)
    else 0
  ) + (
    sum(
      1
      for tool in tools if isinstance(tool, dict) and "cache_control" in tool
    )
    if isinstance(tools, list)
    else 0
  ) + (
    len(_message_cache_marker_locations(messages))
    if isinstance(messages, list)
    else 0
  )


def _bound_anthropic_base_url(config: dict[str, Any]) -> str:
  configured = {
    str(config[key]).strip()
    for key in ("base_url", "baseURL")
    if str(config.get(key) or "").strip()
  }
  if len(configured) > 1:
    raise ValueError("conflicting Anthropic base_url and baseURL values")
  return next(iter(configured), _DEFAULT_ANTHROPIC_BASE_URL)


# A quota 429 is a spent window; a concurrency 429 and an overload are not, and
# neither says anything about the account's quota.
_OVERLOAD_RE = re.compile(r"\boverloaded(?:_error)?\b", re.IGNORECASE)
# The projected rate-limit fields (`api/credentials.py`
# `_anthropic_rate_limit_projection`) and the response headers they come from.
# A failure carries them as attributes when a sanitizer or `stream` projected
# them, and as headers on the raw SDK exception's response.
_RATE_LIMIT_PROJECTION_HEADERS = {
  "rate_limit_representative_claim": "anthropic-ratelimit-unified-representative-claim",
  "rate_limit_unified_status": "anthropic-ratelimit-unified-status",
  "rate_limit_unified_reset": "anthropic-ratelimit-unified-reset",
  "rate_limit_5h_status": "anthropic-ratelimit-unified-5h-status",
  "rate_limit_5h_reset": "anthropic-ratelimit-unified-5h-reset",
  "rate_limit_7d_status": "anthropic-ratelimit-unified-7d-status",
  "rate_limit_7d_reset": "anthropic-ratelimit-unified-7d-reset",
  "retry_after": "retry-after",
}
_WINDOW_STATUS_FIELDS = (
  "rate_limit_5h_status", "rate_limit_7d_status", "rate_limit_unified_status",
)
_COOLDOWN_REASONS = {"rate_limit": "quota", "billing": "billing", "auth": "auth"}


@dataclass(frozen=True)
class _LimitEvidence:
  """What a failed response said about an exhausted quota window."""

  scope: str
  reset_at: float | None


def _limit_evidence(exc: Exception, *, now: float) -> _LimitEvidence | None:
  """Quota evidence on a failure, from the projection the sanitizer attached or the raw response.

  A window the limiter reports as not allowed, or a representative claim when no
  window status is reported at all, is evidence; a response whose reported
  windows are all allowed (a concurrency 429) is not.
  """

  response = getattr(exc, "response", None)
  headers = getattr(response, "headers", None)
  windows = anthropic_quota.parse_rate_limit_headers(headers, now=now) if headers is not None else ()
  exhausted_windows = tuple(
    window for window in windows if anthropic_quota.window_exhausted(window.status)
  )
  unified_status = _projected_rate_limit_field(exc, response, "rate_limit_unified_status").lower()
  reported = [
    _projected_rate_limit_field(exc, response, field).lower() for field in _WINDOW_STATUS_FIELDS
  ]
  reported.extend((window.status or "").strip().lower() for window in windows)
  reported = [status for status in reported if status]
  claim = _projected_rate_limit_field(exc, response, "rate_limit_representative_claim")
  exhausted = any(anthropic_quota.window_exhausted(status) for status in reported)
  if not exhausted and (reported or not claim):
    return None
  if claim:
    # The representative claim is the limit that decided the rejection.
    scope = anthropic_quota.claim_scope(claim)
  else:
    scopes = {window.scope for window in exhausted_windows}
    scope = next(iter(scopes)) if len(scopes) == 1 else "shared"
  candidates = [
    _credential_reset_at(exc, now=now),
    *(window.reset_at for window in exhausted_windows),
  ]
  if anthropic_quota.window_exhausted(unified_status):
    candidates.append(_projected_epoch_seconds(
      _projected_rate_limit_field(exc, response, "rate_limit_unified_reset"), now=now,
    ))
  resets = [reset for reset in candidates if reset is not None]
  return _LimitEvidence(scope=scope, reset_at=max(resets) if resets else None)


def _has_billing_text(failure: ProviderCredentialFailure) -> bool:
  text = f"{failure.message} {failure.error_code or ''}"
  return any(pattern.search(text) for pattern in _BILLING_PATTERNS)


def _projected_rate_limit_field(exc: Exception, response: Any, field: str) -> str:
  value = getattr(exc, field, None)
  text = "" if value is None or isinstance(value, bool) else str(value).strip()
  return text or _response_header(response, _RATE_LIMIT_PROJECTION_HEADERS[field]).strip()


def _carry_rate_limit_evidence(target: Exception, source: Exception) -> None:
  """Keep a rejected response's limit evidence on the error that replaces it.

  `stream` raises a sanitized rejection in place of the SDK error, whose response
  is what says whether a 403 or 429 was a spent quota; the classifier reads it
  from the same projected fields.
  """

  response = getattr(source, "response", None)
  status_code = _exception_status_code(source)
  if status_code is not None:
    setattr(target, "status_code", status_code)
  for field in _RATE_LIMIT_PROJECTION_HEADERS:
    text = _projected_rate_limit_field(source, response, field)
    if text:
      setattr(target, field, text)


def _ingest_rate_limit_headers(binding: AnthropicRequestBinding | None, headers: Any) -> None:
  """Record the quota windows a response's headers describe against its exact binding.

  Telemetry only: a header or sidecar problem is logged by reason and never
  reaches inference.
  """

  if binding is None or headers is None:
    return
  try:
    now = time.time()
    windows = anthropic_quota.parse_rate_limit_headers(headers, now=now)
    if windows:
      anthropic_quota.record_observation(
        binding.sidecar_store_path, binding.locator, windows, now=now,
      )
  except Exception as exc:
    log.warning("Anthropic quota header ingestion failed (%s)", type(exc).__name__)


def _record_stream_success(binding: AnthropicRequestBinding | None) -> None:
  """Remember which account served a completed stream for this session's affinity."""

  if binding is None:
    return
  try:
    record_anthropic_success(binding)
  except Exception as exc:
    log.warning("Anthropic affinity record failed (%s)", type(exc).__name__)


def _credential_material(
  selected: AnthropicCredential,
  config: Mapping[str, Any],
) -> dict[str, Any]:
  """Auth material for a selected or rotated member, in the shape the runner merges."""

  is_api_key = selected.record is not None and selected.record.credential_kind == "api_key"
  material: dict[str, Any] = {
    "auth_mode": "api" if is_api_key else "oauth",
    "api_key": selected.token if is_api_key else "",
    "auth_token": "" if is_api_key else selected.token,
    "credential_source": selected.source_metadata(),
    CREDENTIAL_PREPARED_FIELD: selected.prepared,
  }
  if ANTHROPIC_DEPLOYMENT_API_KEY_FIELD in config:
    material[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD] = config[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD]
  return material


def _server_tool_unit_deltas(usage: Any) -> dict[str, int]:
  if usage is None:
    return {}
  server_tool_use = (
    usage.get("server_tool_use") if isinstance(usage, dict)
    else getattr(usage, "server_tool_use", None)
  )
  raw = _to_plain_dict(server_tool_use)
  if not isinstance(raw, dict):
    return {}
  known = {"web_search_requests", "web_fetch_requests"}
  unknown_populated = {
    key: value
    for key, value in raw.items()
    if key not in known
    and value is not None
    and not (type(value) in (int, float) and value == 0)
  }
  if unknown_populated:
    raise ValueError(
      f"unrecognized separately billed Anthropic server-tool units: {sorted(unknown_populated)}"
    )
  counts: dict[str, int] = {}
  for key in known:
    value = raw.get(key)
    if value is None:
      continue
    if type(value) is not int or value < 0:
      raise ValueError(f"invalid Anthropic server-tool unit count: {key}")
    if value > 0:
      counts[key.removesuffix("_requests")] = value
  return {
    key: counts[key]
    for key in sorted(counts)
  }


def _usage_int(usage: Any, key: str) -> int:
  if usage is None:
    return 0
  value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, 0)
  try:
    return int(value or 0)
  except (TypeError, ValueError):
    return 0


def _usage_iterations(usage: Any) -> list[Any]:
  if usage is None:
    return []
  iterations = usage.get("iterations") if isinstance(usage, dict) else getattr(usage, "iterations", None)
  plain = _to_plain_dict(iterations)
  return plain if isinstance(plain, list) else []


def _anthropic_prompt_usage(usage: Any) -> dict[str, int]:
  """Read only reported prompt components; absent input is not measured zero."""
  components: dict[str, int] = {}
  for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
    value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
    if value is not None:
      components[key] = int(value)
  return components


def _anthropic_usage_totals(usage: Any) -> dict[str, Any]:
  iterations = _usage_iterations(usage)
  if not iterations:
    return {
      "input_tokens": _usage_int(usage, "input_tokens"),
      "output_tokens": _usage_int(usage, "output_tokens"),
      "cache_creation_input_tokens": _usage_int(usage, "cache_creation_input_tokens"),
      "cache_read_input_tokens": _usage_int(usage, "cache_read_input_tokens"),
      "provider_unit_deltas": _server_tool_unit_deltas(usage),
    }

  provider_unit_deltas: dict[str, int] = {}
  totals = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "provider_unit_deltas": provider_unit_deltas,
  }
  for iteration in iterations:
    totals["input_tokens"] += _usage_int(iteration, "input_tokens")
    totals["output_tokens"] += _usage_int(iteration, "output_tokens")
    totals["cache_creation_input_tokens"] += _usage_int(iteration, "cache_creation_input_tokens")
    totals["cache_read_input_tokens"] += _usage_int(iteration, "cache_read_input_tokens")
    for operation, count in _server_tool_unit_deltas(iteration).items():
      provider_unit_deltas[operation] = int(provider_unit_deltas.get(operation, 0) or 0) + int(count)
  return totals


def _unit_delta(
  totals: dict[str, int],
  reported: dict[str, int],
) -> dict[str, int]:
  delta = {
    key: max(0, int(value or 0) - int(reported.get(key, 0) or 0))
    for key, value in totals.items()
    if int(value or 0) > int(reported.get(key, 0) or 0)
  }
  reported.update(totals)
  return delta


def prepare_anthropic_tools(
  tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  """Transform strict tool schemas for Anthropic's constrained decoder."""

  if not any(tool.get("strict") is True for tool in tools):
    return tools

  try:
    from anthropic import transform_schema
  except ImportError as exc:
    raise RuntimeError(
      "anthropic dependency is required to prepare strict Anthropic tools"
    ) from exc

  prepared: list[dict[str, Any]] = []
  for tool in tools:
    if tool.get("strict") is not True:
      prepared.append(tool)
      continue
    input_schema = tool.get("input_schema")
    if not isinstance(input_schema, dict):
      raise ValueError("Strict Anthropic tools require an object input_schema")
    strict_tool = dict(tool)
    strict_tool["input_schema"] = transform_schema(input_schema)
    prepared.append(strict_tool)
  return prepared


def _prepare_anthropic_tools(
  tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  """Backward-compatible private seam used by existing provider callers."""
  return prepare_anthropic_tools(tools)


# A tool_result entry names the tools its call made callable under this key
# (provider-neutral: other providers' transcripts ignore it). The rendering
# below turns it into `tool_reference` blocks.
_EXPOSED_TOOLS_KEY = "exposed_tools"


def referenced_tool_names(messages: list[dict[str, Any]]) -> frozenset[str]:
  """Names the visible history exposes through its tool_result entries."""

  return frozenset(
    name
    for message in messages
    if message.get("role") == "user" and isinstance(message.get("content"), list)
    for block in message["content"]
    if isinstance(block, dict)
    and block.get("type") == "tool_result"
    for name in block.get(_EXPOSED_TOOLS_KEY) or ()
  )


def _prefix_stable_tools(
  tools: list[dict[str, Any]],
  anchor: list[dict[str, Any]],
  referenced: frozenset[str],
  framing_tool: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
  """Lay the request tools out so later additions never touch the cached prefix.

  Anthropic's cache prefix is tools, then system, then messages, so any change
  to the non-deferred tool list rewrites everything after it.  The anchor (the
  wire definitions the conversation's first request sent) is re-sent unchanged
  and in order.  Every other tool follows it, and whether it is deferred is a
  pure function of the visible history: a tool is sent ``defer_loading`` only
  when a ``tool_reference`` for it renders from ``referenced`` (the API expands
  that reference into the definition), and is otherwise sent plain so the model
  can see it.  Order: anchor, plain tools by name, deferred tools by name.  A
  deferred tool cannot carry ``cache_control``.
  """

  anchor_names = {tool["name"] for tool in anchor}
  others = sorted(
    (
      {
        key: value
        for key, value in tool.items()
        if key not in {"cache_control", "defer_loading"}
      }
      for tool in tools
      if tool["name"] not in anchor_names
    ),
    key=lambda tool: tool["name"],
  )
  plain = [tool for tool in others if tool["name"] not in referenced]
  deferred = [tool for tool in others if tool["name"] in referenced]
  if (
    framing_tool is not None
    and framing_tool.get("name")
    and framing_tool["name"] not in anchor_names
    and not any(t["name"] == framing_tool["name"] for t in others)
  ):
    clean_framing = {
      key: value for key, value in framing_tool.items()
      if key not in {"cache_control", "defer_loading"}
    }
    deferred.append(clean_framing)
  return [
    *anchor,
    *_prepare_anthropic_tools(plain),
    *(
      {**tool, "defer_loading": True}
      for tool in _prepare_anthropic_tools(deferred)
    ),
  ]


def _render_tool_references(
  messages: list[dict[str, Any]],
  deferred_names: set[str],
) -> list[dict[str, Any]]:
  """Render each tool_result's exposed tools as ``tool_reference`` blocks.

  A reference to a tool the request does not carry as deferred is a 400, so
  only names in ``deferred_names`` are rendered.  The entry key is always
  removed: the API rejects it.  The API also rejects a tool_result that mixes
  ``tool_reference`` blocks with any other content ("Tool definitions/code
  execution functions cannot be mixed with other content", observed live
  2026-10-13), so a referencing tool_result carries the references alone and
  its own result moves, unchanged and labelled by ``tool_use_id``, to the end
  of the same user message.  The layout is a pure function of the stored
  entry, so every later request re-sends the same bytes.
  """

  rendered: list[dict[str, Any]] = []
  for message in messages:
    content = message.get("content")
    if message.get("role") != "user" or not isinstance(content, list):
      rendered.append(message)
      continue
    next_content: list[Any] = []
    moved_results: list[Any] = []
    for block in content:
      if not (
        isinstance(block, dict)
        and block.get("type") == "tool_result"
        and _EXPOSED_TOOLS_KEY in block
      ):
        next_content.append(block)
        continue
      next_block = {
        key: value for key, value in block.items() if key != _EXPOSED_TOOLS_KEY
      }
      references = [
        {"type": "tool_reference", "tool_name": name}
        for name in block[_EXPOSED_TOOLS_KEY]
        if name in deferred_names
      ]
      if references:
        result_content = next_block.get("content")
        if isinstance(result_content, str):
          parts = [{"type": "text", "text": result_content}] if result_content else []
        else:
          parts = list(result_content or [])
        next_block["content"] = references
        if parts:
          moved_results.append({
            "type": "text",
            "text": f"Result of tool call {next_block.get('tool_use_id')}:",
          })
          moved_results.extend(parts)
      next_content.append(next_block)
    rendered.append({**message, "content": [*next_content, *moved_results]})
  return rendered


# `thinking.effort_control` names how a request carries effort; the request's
# thinking parameter follows from it (see `_thinking_param`).
_THINKING_MODE_BY_EFFORT_CONTROL: dict[str, Literal["adaptive", "budget", "none"]] = {
  "output_config": "adaptive",
  "budget": "budget",
  "none": "none",
}


def _messages_compat(model_info: ModelInfo) -> AnthropicMessagesCompat:
  compat = model_info.compat
  if not isinstance(compat, AnthropicMessagesCompat):
    raise TypeError(f"Anthropic model {model_info.id!r} carries no anthropic.messages compat")
  return compat


def _messages_model_info(model: str, rate_table: RateTable) -> ModelInfo:
  model_id, entry = admitted_entry(AnthropicProvider.adapter_route_support(), model)
  compat = entry.compat
  if not isinstance(compat, AnthropicMessagesCompat):
    raise TypeError(f"{entry.key}.compat is not the anthropic.messages key set")
  thinking_mode = _THINKING_MODE_BY_EFFORT_CONTROL[compat.thinking.effort_control]
  return model_info_from_entry(
    model_id,
    entry,
    rate_table,
    supports_thinking=thinking_mode != "none",
    thinking_mode=thinking_mode,
    supports_native_compaction=compat.native_compaction,
  )


class AnthropicProvider(ModelProvider):
  """`ModelProvider` implementation for Anthropic's Messages API.

  Supports API key and OAuth-style auth, prompt caching blocks, tool use, and
  Anthropic thinking/compaction features when the selected model allows them.
  """

  name = "anthropic"

  @classmethod
  def adapter_route_support(cls) -> AdapterRouteSupport:
    # Protocol facts about this implementation: the public Messages API with
    # plain requests (`messages.standard`) and adaptive/budgeted thinking
    # requests (`messages.adaptive`, see `resolve_effort`/`_thinking_param`).
    # The Risk-local `anthropic.sdk.messages` adapter is a different
    # implementation in a different serving process and is not declared here.
    return AdapterRouteSupport(
      adapter="anthropic.messages",
      provider="anthropic",
      protocol_profiles=frozenset({"messages.standard", "messages.adaptive"}),
      routes=frozenset({"anthropic.public"}),
    )

  def __init__(self, *, rate_table: RateTable | None = None):
    self._rate_table = authority_rate_table(self.name, rate_table)

  @staticmethod
  def thinking_param(model: str, max_tokens: int) -> dict[str, Any] | None:
    # The model authority is the single owner of which model ids are
    # admitted; _messages_model_info raises for anything it does not know.
    return _thinking_param(
      _messages_model_info(model, authority_rate_table(AnthropicProvider.name)),
      max_tokens,
    )

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    if str(config.get("auth_mode", "api")).strip().lower() == "oauth":
      return bool(str(config.get("auth_token") or "").strip())
    return bool(str(config.get("api_key") or "").strip())

  def classify_credential_failure(self, exc: Exception) -> ProviderCredentialFailure | None:
    """The generic mapping, with Anthropic's quota evidence deciding what parks.

    A 429 parks an account only when it reports an exhausted window; a
    concurrency 429 and an overload are transient and keep the runner's
    bounded backoff. A 403 is a quota signal only when it carries that same
    evidence; any other 403 keeps the generic mapping unchanged.
    """

    failure = super().classify_credential_failure(exc)
    if failure is None:
      return None
    if failure.status_code == 529 or _OVERLOAD_RE.search(
      f"{failure.message} {failure.error_code or ''}"
    ):
      return None
    evidence = _limit_evidence(exc, now=time.time())
    if failure.kind == "rate_limit":
      if evidence is None:
        return None
      return replace(failure, reset_at=evidence.reset_at, limit_scope=evidence.scope)
    if failure.kind == "billing" and failure.status_code in {403, 429}:
      if evidence is not None and (failure.status_code == 429 or not _has_billing_text(failure)):
        return replace(
          failure, kind="rate_limit", reset_at=evidence.reset_at, limit_scope=evidence.scope,
        )
    return failure

  async def prepare_credential(
    self,
    config: Mapping[str, Any],
    *,
    model: str,
    session_id: str | None,
  ) -> dict[str, Any] | None:
    """Rank the caller's authorized pool for this model and session, before the client binds.

    Only a resolver-issued (automatic) `credential_source` is a hint into the
    pool; an exact caller-supplied one, and a config with no source, keep the
    credential they carry. Stale or cold subscription accounts are probed for
    usage within one bounded batch; the probe is telemetry, so a failure of it
    leaves ordered selection. A successful refresh preflight marks the returned
    material prepared, letting `create_client` skip its own refresh. An all-blocked
    fallback is not refresh-prepared and still refreshes at the client boundary.
    """

    source = config.get("credential_source")
    if not isinstance(source, Mapping) or not source.get("automatic"):
      return None
    # The automatic source is an affinity hint, not a pin: resolve the whole
    # pool exactly as rotation does.
    pool_config = anthropic_pool_config(config)
    sources = await asyncio.to_thread(resolve_anthropic_credentials, pool_config)
    if not any(member.auth_mechanism == "oauth_bearer" for member in sources.credentials):
      return None
    await self._probe_usage(sources, pool_config)
    selected = await asyncio.to_thread(
      select_anthropic_credential_source,
      sources,
      placement=str(config.get("placement") or "local"),
      model=model,
      session_id=session_id,
    )
    if selected is None:
      return None
    material = _credential_material(selected, config)
    material[CREDENTIAL_AFFINITY_KEY_FIELD] = anthropic_quota.affinity_key(
      sources.user_scope, session_id,
    )
    family = anthropic_quota.model_family(model)
    if family is not None:
      material[CREDENTIAL_MODEL_FAMILY_FIELD] = family
    if all(config.get(key) == value for key, value in material.items()):
      return None
    return material

  async def _probe_usage(
    self,
    sources: AnthropicCredentialSources,
    config: dict[str, Any],
  ) -> None:
    """Refresh the usage view of stale or cold subscription accounts, within the batch budget."""

    base_url = _bound_anthropic_base_url(config)
    if not anthropic_quota.usage_probe_allowed(base_url):
      return
    now = time.time()
    try:
      due: dict[str, AnthropicCredential] = {}
      for member in sources.credentials:
        if member.auth_mechanism != "oauth_bearer":
          continue
        # An access token about to expire would 401 and mark the account's
        # usage unavailable for an hour; its own refresh comes first.
        if member.record is not None and member.record.needs_refresh(now=now):
          continue
        if anthropic_quota.claim_usage_probe(
          member.sidecar_store_path,
          member.locator,
          generation=member.generation_id or None,
          now=now,
        ):
          due[member.locator.id] = member
      if not due:
        return
      results = await anthropic_quota.probe_usage_batch(
        [
          anthropic_quota.ProbeTarget(
            locator=member.locator,
            bearer=member.token,
            generation=member.generation_id or None,
            base_url=base_url,
          )
          for member in due.values()
        ],
        now=now,
      )
      for locator_id, result in results.items():
        member = due[locator_id]
        if result.windows:
          anthropic_quota.record_observation(
            member.sidecar_store_path, member.locator, result.windows, now=now,
          )
        anthropic_quota.record_probe_state(member.sidecar_store_path, member.locator, result.probe)
    except Exception as exc:
      log.warning("Anthropic usage probe failed (%s); selecting without it", type(exc).__name__)

  def next_credential(
    self,
    config: Mapping[str, Any],
    failure: ProviderCredentialFailure,
  ) -> dict[str, Any] | None:
    """The next unblocked sibling credential for the same bound model.

    Limits and denials park the rejected API key or OAuth seat, for the scope
    the limiter named, until the reset the provider reported. The sibling
    retains its own authentication mode; the caller rebuilds its client from
    that material and the same `model_key`. With no sibling left the run keeps
    the existing rate-limit retry.
    """

    mode = str(config.get("auth_mode", "api")).strip().lower()
    # A seat's `billing` denial is the organisation refusing the client and
    # parks nothing; an API key's is its credits spent, which the seats
    # behind it are there to absorb (Henry 2026-10-13).
    rotatable = _ROTATABLE_CREDENTIAL_FAILURES | ({"billing"} if mode == "api" else frozenset())
    if failure.kind not in rotatable:
      return None
    current = str(config.get("auth_token" if mode == "oauth" else "api_key") or "").strip()
    if not current:
      return None
    # Rotation resolves the pool, not the exact frozen-generation bind: drop
    # `credential_source` so this is the caller's whole pool (user store + org),
    # and the sibling the resolver names is a genuinely eligible generation.
    # This is also what lets a bound generation that was retired at its owner
    # rotate: the pool path neither raises for the stale bind nor rewrites it.
    pool_config = anthropic_pool_config(config)
    sources = resolve_anthropic_credentials(pool_config)
    if failure.reset_at is None:
      log.info(
        "Anthropic %s failure (status=%s) reported no reset window; parking this "
        "credential for %.0fs and rotating",
        failure.kind,
        failure.status_code,
        UNDECLARED_LIMIT_BLOCK_SECONDS,
      )
    blocked_until = (
      failure.reset_at
      if failure.reset_at is not None
      else time.time() + UNDECLARED_LIMIT_BLOCK_SECONDS
    )
    selected = rotate_anthropic_credential_source(
      sources,
      current=current,
      until=blocked_until,
      model=config.get(CREDENTIAL_MODEL_FAMILY_FIELD),
      scope=failure.limit_scope or "shared",
      reason=_COOLDOWN_REASONS.get(failure.kind, failure.kind),
    )
    if selected is None:
      log.warning(
        "Anthropic %s failure with no unblocked sibling credential in a pool of %d",
        failure.kind,
        len(sources.credentials),
      )
      return None
    rotated = _credential_material(selected, config)
    for field in (CREDENTIAL_AFFINITY_KEY_FIELD, CREDENTIAL_MODEL_FAMILY_FIELD):
      if field in config:
        rotated[field] = config[field]
    return rotated

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    try:
      from anthropic import AsyncAnthropic
      import httpx2
    except ImportError as exc:
      raise RuntimeError("anthropic dependency is required to use AnthropicProvider") from exc

    mode = str(config.get("auth_mode", "api")).strip().lower()
    credential_field = "auth_token" if mode == "oauth" else "api_key"
    credential = str(config.get(credential_field) or "").strip()
    if not credential:
      raise RuntimeError(f"No Anthropic {mode} credential configured")
    # A prepared config carries the bearer selection just refreshed and bound;
    # refreshing here again could send a bearer the binding does not describe.
    if mode == "oauth" and not config.get(CREDENTIAL_PREPARED_FIELD):
      # The bound token may have been selected hours ago (a long session, a
      # captured run). An account enrolled through this product's own login
      # refreshes here, at the one boundary where the credential becomes a
      # client; an environment token has no refresh grant and is untouched.
      credential = ensure_fresh_anthropic_credential(credential, config=config)
    # Responses and failures of this client are attributed to the bearer it
    # actually sends, not to whatever `config` names after another process
    # refreshes.
    binding = anthropic_request_binding(config, credential)

    auth_header = "Authorization" if mode == "oauth" else "X-Api-Key"
    auth_value = f"Bearer {credential}" if mode == "oauth" else credential

    class BoundAsyncAnthropic(AsyncAnthropic):
      def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hank_anthropic_binding = binding
        # The SDK imports ambient headers during construction, including copies.
        # Drop auth overrides before default_headers and pre-request validation.
        self._custom_headers = {
          name: value for name, value in self._custom_headers.items()
          if name.lower() not in {"x-api-key", "authorization"}
        }

      async def _prepare_request(self, request: httpx2.Request) -> None:
        # This SDK hook runs after options are logged and all header layers merge.
        # Bind on the HTTP request so ambient headers lose without logging secrets.
        request.headers.pop("x-api-key", None)
        request.headers.pop("authorization", None)
        request.headers[auth_header] = auth_value

    client_kwargs: Dict[str, Any] = {
      "base_url": _bound_anthropic_base_url(config),
      # The gateway owns provider retry policy: runner_stream_turn's bounded
      # STREAM_RETRY_MAX loop plus this provider's next_credential rotation.
      # The SDK's own retry (anthropic 1.6.0 DEFAULT_MAX_RETRIES=2) is a second
      # writer: _calculate_retry_timeout obeys the 429's retry-after verbatim
      # (capped only at 4_294_967 s) and sleeps it inside the request, so a
      # reset reported in hours never surfaces as RateLimitError and the turn
      # dies on the watchdog instead of reporting the limit. Zero retries here
      # hands 429/5xx straight to is_retryable_error, the sanitizer, and the
      # credential pool.
      "max_retries": 0,
    }
    if timeout is not None:
      client_kwargs["timeout"] = httpx2.Timeout(timeout=timeout, connect=5.0)

    if mode == "oauth":
      oauth_headers = {
        "anthropic-beta": ",".join([*_OAUTH_BETA_SLUGS, *_COMMON_BETA_SLUGS]),
        "user-agent": "claude-cli/2026.3.14",
        "x-app": "cli",
      }

      return BoundAsyncAnthropic(
        api_key="",
        auth_token=credential,
        default_headers=oauth_headers,
        **client_kwargs,
      )

    api_headers: Dict[str, Any] = {}
    if _COMMON_BETA_SLUGS:
      api_headers["anthropic-beta"] = ",".join(_COMMON_BETA_SLUGS)
    return BoundAsyncAnthropic(
      api_key=credential,
      auth_token="",
      default_headers=api_headers,
      **client_kwargs,
    )

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    if client is None:
      return
    closer = getattr(client, "aclose", None) or getattr(client, "close", None)
    if closer is None:
      return
    try:
      result = closer()
      if asyncio.iscoroutine(result):
        await asyncio.wait_for(result, timeout=timeout)
    except Exception:
      pass

  def get_model_info(self, model: str) -> ModelInfo:
    return _messages_model_info(model, self._rate_table)

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
    auth_mode = str(kwargs.get("auth_mode", "api")).strip().lower()
    system_blocks: list[dict[str, Any]] | None = None
    if auth_mode == "oauth":
      system_blocks = [{"type": "text", "text": _OAUTH_IDENTITY}]
    model_info = self.get_model_info(model)

    if system_prompt:
      if isinstance(system_prompt, list):
        if system_blocks is None:
          system_blocks = []
        for block_text, should_cache in system_prompt:
          if not block_text:
            continue
          block: Dict[str, Any] = {"type": "text", "text": block_text}
          if should_cache:
            block["cache_control"] = {"type": "ephemeral"}
          system_blocks.append(block)
        if not system_blocks:
          system_blocks = None
      else:
        block = {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
        if system_blocks is None:
          system_blocks = [block]
        else:
          system_blocks.append(block)

    tool_anchor = kwargs.get("tool_anchor")
    raw_framing = kwargs.get("framing_tool")
    framing_tool = (
      dict(raw_framing)
      if isinstance(raw_framing, dict) and raw_framing.get("name")
      else None
    )
    referenced = referenced_tool_names(messages)
    if tool_anchor is None:
      prepared_tools = _prepare_anthropic_tools(tools)
      if (
        framing_tool is not None
        and not any(t.get("name") == framing_tool["name"] for t in prepared_tools)
      ):
        clean_framing = {
          key: value for key, value in framing_tool.items()
          if key not in {"cache_control", "defer_loading"}
        }
        prepared_tools.append({**clean_framing, "defer_loading": True})
    else:
      prepared_tools = _prefix_stable_tools(
        tools,
        tool_anchor,
        referenced,
        framing_tool=framing_tool,
      )
    messages = _render_tool_references(
      messages,
      {
        tool["name"]
        for tool in prepared_tools
        if tool.get("defer_loading") is True
      },
    )
    preceding_breakpoint_count = (
      sum(
        1
        for block in system_blocks
        if isinstance(block, dict) and "cache_control" in block
      )
      if system_blocks
      else 0
    ) + sum(
      1
      for tool in prepared_tools
      if isinstance(tool, dict) and "cache_control" in tool
    )
    fork_mode = bool(kwargs.get("fork_mode", False))
    raw_fork_marker = kwargs.get("fork_marker_position")
    fork_marker_position = (
      tuple(raw_fork_marker)
      if isinstance(raw_fork_marker, (list, tuple))
      and len(raw_fork_marker) == 2
      else None
    )
    request_messages, message_marker_location = _place_message_cache_breakpoint(
      messages,
      mode="fork" if fork_mode else "normal",
      allow_marker=preceding_breakpoint_count < 4,
      pinned_position=fork_marker_position,
    )

    params: Dict[str, Any] = {
      "model": model,
      "max_tokens": max_tokens,
      "messages": request_messages,
      "tools": prepared_tools,
      "_provider_auth_mode": auth_mode,
    }
    if system_blocks:
      params["system"] = system_blocks

    effort_resolution = kwargs.get("effort_resolution")
    if not isinstance(effort_resolution, EffortResolution):
      effort_resolution = self.resolve_effort(
        requested=thinking_level,
        model=model,
        model_info=model_info,
        max_tokens=max_tokens,
        auth_mode=auth_mode,
      )
    params.update(effort_resolution.payload_fragments)
    if prepared_tools:
      forcing = _messages_compat(model_info).tool_choice_forcing
      may_force = forcing == "any" or (
        forcing == "thinking_off_only" and not effort_resolution.thinking_enabled_effective
      )
      params["tool_choice"] = (
        {"type": "tool", "name": prepared_tools[0]["name"]}
        if len(prepared_tools) == 1 and may_force
        else {"type": "auto"}
      )

    compaction_trigger = kwargs.get("compaction_trigger")
    if compaction_trigger is not None and model_info.supports_native_compaction:
      compact_edit: Dict[str, Any] = {
        "type": "compact_20260112",
        "trigger": {"type": "input_tokens", "value": compaction_trigger},
        "pause_after_compaction": False,
      }
      compaction_instructions = kwargs.get("compaction_instructions")
      if compaction_instructions is not None:
        compact_edit["instructions"] = compaction_instructions
      params["context_management"] = {"edits": [compact_edit]}

    if __debug__:
      try:
        _assert_message_cache_breakpoint_placement(
          request_messages,
          expected_location=message_marker_location,
          mode="fork" if fork_mode else "normal",
        )
        assert _explicit_cache_breakpoint_count(params) <= 4, (
          "Anthropic requests may carry at most four explicit cache breakpoints"
        )
      except AssertionError as exc:
        log.error("Anthropic cache breakpoint invariant failed: %s", exc)

    return params

  def resolve_effort(
    self,
    *,
    requested: ThinkingLevel,
    model: str,
    model_info: ModelInfo,
    max_tokens: int,
    **request_context: Any,
  ) -> EffortResolution:
    del model, request_context
    thinking = _messages_compat(model_info).thinking
    default_effort = ThinkingLevel(thinking.effort_when_omitted)
    omitted_on = thinking.default_when_omitted == "on"
    if requested == ThinkingLevel.NONE:
      if thinking.off_value is not None:
        return EffortResolution(
          requested, ThinkingLevel.NONE, False, {"thinking": {"type": thinking.off_value}},
        )
      return EffortResolution(
        requested, default_effort if omitted_on else ThinkingLevel.NONE, omitted_on, {},
      )
    if not model_info.supports_thinking or model_info.thinking_mode == "none":
      return EffortResolution(requested, ThinkingLevel.NONE, False, {})

    if model_info.thinking_mode == "budget":
      budget = _thinking_param(model_info, max_tokens)
      return EffortResolution(
        requested,
        ThinkingLevel.HIGH if budget is not None else ThinkingLevel.NONE,
        budget is not None,
        {"thinking": budget} if budget is not None else {},
      )

    supported = tuple(
      ThinkingLevel(value) for value in model_info.effort_values if value != ThinkingLevel.NONE.value
    )


    normalized = ThinkingLevel.LOW if requested == ThinkingLevel.MINIMAL else requested
    effective = clamp_effort(normalized, supported) if supported else ThinkingLevel.NONE
    if max_tokens < 2048:
      return EffortResolution(
        requested,
        default_effort if omitted_on else ThinkingLevel.NONE,
        omitted_on,
        {},
      )
    thinking_fragment = _thinking_param(model_info, max_tokens)
    fragments: dict[str, Any] = {"thinking": thinking_fragment} if thinking_fragment is not None else {}
    if thinking.effort_control == "output_config" and effective != ThinkingLevel.NONE:
      fragments["output_config"] = {"effort": effective.value}
    return EffortResolution(requested, effective, effective != ThinkingLevel.NONE, fragments)

  def normalize_messages(self, messages: list[dict[str, Any]], model_info: ModelInfo) -> list[dict[str, Any]]:
    tool_id_map: dict[str, str] = {}
    transformed: list[dict[str, Any]] = []

    for message in messages:
      role = message.get("role")
      if role == "assistant":
        if str(message.get("stop_reason", "")) in {"error", "aborted"}:
          continue

        is_same_model = _same_model_message(message, model_info)
        content = message.get("content")
        if not isinstance(content, list):
          transformed.append({"role": "assistant", "content": content})
          continue

        next_content: list[dict[str, Any]] = []
        for block in content:
          if not isinstance(block, dict):
            continue

          block_type = block.get("type")
          if block_type == "thinking":
            thinking_text = str(block.get("thinking", ""))
            if is_same_model:
              next_content.append(dict(block))
            elif thinking_text.strip():
              # The signature is bound to the model that produced it, so a
              # different model gets the reasoning narrative as handoff
              # context, marked as such rather than as its own prior words.
              prior_model = str(message.get("model", "")) or "a previous model"
              next_content.append({
                "type": "text",
                "text": f"[Reasoning summary from {prior_model}]\n{thinking_text}",
              })
            continue

          if block_type in {"tool_use", "server_tool_use"}:
            next_block = dict(block)
            if not is_same_model:
              # Thought signatures belong to the model that produced them;
              # Anthropic rejects a foreign one as an extra tool_use input.
              next_block.pop("thought_signature", None)
              next_block.pop("thoughtSignature", None)
            tool_id = str(block.get("id", ""))
            if tool_id:
              normalized_id = _normalize_tool_call_id(tool_id)
              if normalized_id != tool_id:
                tool_id_map[tool_id] = normalized_id
                next_block["id"] = normalized_id
            next_content.append(next_block)
            continue

          if block_type == "text":
            next_content.append({"type": "text", "text": str(block.get("text", ""))})
            continue

          next_content.append(dict(block))

        next_message = {"role": "assistant", "content": next_content}
        transformed.append(next_message)
        continue

      if role == "user":
        content = message.get("content")
        if isinstance(content, list):
          next_content: list[dict[str, Any]] = []
          for block in content:
            if not isinstance(block, dict):
              continue
            if block.get("type") == "tool_result":
              next_block = dict(block)
              next_block.pop("tool_name", None)
              tool_use_id = str(block.get("tool_use_id", ""))
              normalized_id = tool_id_map.get(tool_use_id)
              if normalized_id is not None:
                next_block["tool_use_id"] = normalized_id
              next_content.append(next_block)
            else:
              next_content.append(dict(block))
          transformed.append({"role": "user", "content": next_content})
        else:
          transformed.append({"role": "user", "content": content})
        continue

      transformed.append({"role": role, "content": message.get("content")})

    result: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []
    existing_tool_result_ids: set[str] = set()

    for message in transformed:
      role = message.get("role")
      if role == "assistant":
        if pending_tool_calls:
          missing = [
            _synthetic_tool_result(str(block.get("id", "")), str(block.get("name", "")))
            for block in pending_tool_calls
            if str(block.get("id", "")) not in existing_tool_result_ids
          ]
          if missing:
            result.append({"role": "user", "content": missing})
          pending_tool_calls = []
          existing_tool_result_ids = set()

        content = message.get("content")
        if isinstance(content, list):
          pending_tool_calls = [
            dict(block)
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use"
          ]
        else:
          pending_tool_calls = []

        result.append(message)
        continue

      if role == "user" and pending_tool_calls and _has_tool_result_block(message):
        content = list(message.get("content") or [])
        pending_ids = {str(block.get("id", "")) for block in pending_tool_calls}
        filtered_content: list[Any] = []
        existing_tool_result_ids = {
          str(block.get("tool_use_id", ""))
          for block in content
          if (
            isinstance(block, dict)
            and block.get("type") == "tool_result"
            and str(block.get("tool_use_id", "")) in pending_ids
          )
        }
        for block in content:
          if not isinstance(block, dict) or block.get("type") != "tool_result":
            filtered_content.append(block)
            continue
          if str(block.get("tool_use_id", "")) in pending_ids:
            filtered_content.append(block)
        missing = [
          _synthetic_tool_result(str(block.get("id", "")), str(block.get("name", "")))
          for block in pending_tool_calls
          if str(block.get("id", "")) not in existing_tool_result_ids
        ]
        if missing:
          next_message = dict(message)
          next_message["content"] = [*filtered_content, *missing]
          result.append(next_message)
        else:
          next_message = dict(message)
          next_message["content"] = filtered_content
          result.append(next_message)
        pending_tool_calls = []
        existing_tool_result_ids = set()
        continue

      if pending_tool_calls:
        missing = [
          _synthetic_tool_result(str(block.get("id", "")), str(block.get("name", "")))
          for block in pending_tool_calls
          if str(block.get("id", "")) not in existing_tool_result_ids
        ]
        if missing:
          result.append({"role": "user", "content": missing})
        pending_tool_calls = []
        existing_tool_result_ids = set()

      if role == "user" and _has_tool_result_block(message):
        content = [
          block
          for block in list(message.get("content") or [])
          if not (isinstance(block, dict) and block.get("type") == "tool_result")
        ]
        if content:
          next_message = dict(message)
          next_message["content"] = content
          result.append(next_message)
        continue

      result.append(message)

    if pending_tool_calls:
      missing = [
        _synthetic_tool_result(str(block.get("id", "")), str(block.get("name", "")))
        for block in pending_tool_calls
        if str(block.get("id", "")) not in existing_tool_result_ids
      ]
      if missing:
        result.append({"role": "user", "content": missing})

    return truncate_to_last_compaction(
      result,
      compaction_as_text=not model_info.supports_native_compaction,
    )

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
    bucket = get_global_token_bucket()
    provider_wait_ms = 0
    if bucket is not None:
      waited_seconds = await bucket.acquire(
        estimated_tokens=int(params.get("max_tokens") or 0),
      )
      provider_wait_ms = round(waited_seconds * 1000)

    stream_params = dict(params)
    # Attribution follows the bearer this client actually sends (see create_client).
    binding: AnthropicRequestBinding | None = getattr(client, "hank_anthropic_binding", None)
    auth_mode = str(stream_params.pop("_provider_auth_mode", "api")).strip().lower()
    use_compaction = "context_management" in stream_params
    request_tools = stream_params.get("tools")
    use_structured_outputs = (
      isinstance(request_tools, list)
      and any(
        isinstance(tool, dict) and tool.get("strict") is True
        for tool in request_tools
      )
    )
    betas: list[str] = []
    stop_reason = ""
    stop_details: dict[str, Any] | None = None
    current_block_type: str | None = None
    current_tool_id: str | None = None
    current_tool_name: str | None = None
    current_tool_json = ""
    current_tool_block: Any = None
    current_text = ""
    current_thinking = ""
    current_signature = ""
    current_compaction: Any = None
    message_start_usage = {
      "input_tokens": 0,
      "cache_creation_input_tokens": 0,
      "cache_read_input_tokens": 0,
    }
    reported_usage_update = {
      "input_tokens": 0,
      "output_tokens": 0,
      "cache_creation_input_tokens": 0,
      "cache_read_input_tokens": 0,
    }
    reported_provider_unit_deltas: dict[str, int] = {}
    request_prompt_usage: dict[str, int] = {}
    request_has_iterations = False

    try:
      if use_compaction or use_structured_outputs:
        betas = list(_COMMON_BETA_SLUGS)
        if auth_mode == "oauth":
          betas = [*_OAUTH_BETA_SLUGS, *betas]
        if use_structured_outputs:
          betas.append(_STRUCTURED_OUTPUTS_BETA_SLUG)
        if use_compaction:
          betas.append(_COMPACTION_BETA_SLUG)
        stream_cm = client.beta.messages.stream(**stream_params, betas=betas)
      else:
        stream_cm = client.messages.stream(**stream_params)

      async with stream_cm as stream_obj:
        _ingest_rate_limit_headers(
          binding, getattr(getattr(stream_obj, "response", None), "headers", None),
        )
        async for event in stream_obj:
          event_type = getattr(event, "type", None)

          if event_type == "ping":
            yield StreamEvent(type="heartbeat")
            continue

          if event_type == "message_start":
            message = getattr(event, "message", None)
            usage = getattr(message, "usage", None) if message is not None else None
            request_prompt_usage = _anthropic_prompt_usage(usage)
            reported_model = str(
              (getattr(message, "model", "") if message is not None else "")
              or ""
            ).strip() or None
            message_start_usage = {
              "input_tokens": getattr(usage, "input_tokens", 0) if usage is not None else 0,
              "cache_creation_input_tokens": (
                getattr(usage, "cache_creation_input_tokens", 0) if usage is not None else 0
              ),
              "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) if usage is not None else 0,
            }
            yield StreamEvent(
              type="message_start",
              provider_reported_model=reported_model,
              input_tokens=message_start_usage["input_tokens"],
              output_tokens=getattr(usage, "output_tokens", 0) if usage is not None else 0,
              cache_creation_tokens=message_start_usage["cache_creation_input_tokens"],
              cache_read_tokens=message_start_usage["cache_read_input_tokens"],
              # Anthropic reports separately billed server-tool units on the
              # final usage snapshot; avoid counting the same call twice.
              provider_units=0,
              provider_wait_ms=provider_wait_ms,
            )
            continue

          if event_type == "content_block_start":
            block = getattr(event, "content_block", None)
            block_type = getattr(block, "type", None)
            if block_type == "tool_use":
              current_block_type = "tool_use"
              current_tool_id = getattr(block, "id", None)
              current_tool_name = getattr(block, "name", None)
              current_tool_json = ""
              current_tool_block = block
              caller = _to_plain_dict(getattr(block, "caller", None))
              yield StreamEvent(
                type="tool_use_start",
                tool_id=str(current_tool_id or ""),
                tool_name=str(current_tool_name or ""),
                raw_block=_to_plain_dict(block),
                caller=caller if isinstance(caller, dict) else None,
              )
            elif block_type == "thinking":
              current_block_type = "thinking"
              current_thinking = ""
              current_signature = ""
              yield StreamEvent(type="stream_progress")
            elif block_type == "text":
              current_block_type = "text"
              current_text = ""
            elif block_type == "compaction":
              current_block_type = "compaction"
              current_compaction = None
            continue

          if event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            delta_type = getattr(delta, "type", None)
            if delta_type == "text_delta":
              text = getattr(delta, "text", "")
              if text:
                current_text += text
                yield StreamEvent(type="text_delta", text=text)
            elif delta_type == "input_json_delta":
              partial = getattr(delta, "partial_json", "")
              if partial:
                current_tool_json += partial
                yield StreamEvent(type="tool_use_delta", tool_input_json=partial)
            elif delta_type == "thinking_delta":
              thinking_text = getattr(delta, "thinking", "")
              if thinking_text:
                current_thinking += thinking_text
                yield StreamEvent(type="thinking_delta", thinking_text=thinking_text)
            elif delta_type == "signature_delta":
              signature = getattr(delta, "signature", "")
              if signature:
                current_signature += signature
                yield StreamEvent(type="stream_progress")
            elif delta_type == "compaction_delta":
              current_compaction = getattr(delta, "content", None)
              yield StreamEvent(type="stream_progress")
            continue

          if event_type == "content_block_stop":
            if current_block_type == "text":
              yield StreamEvent(
                type="text_end",
                text=current_text,
                raw_block={"type": "text", "text": current_text},
              )
              current_text = ""
              current_block_type = None
            elif current_block_type == "thinking":
              block = {
                "type": "thinking",
                "thinking": current_thinking,
                "signature": current_signature,
              }
              yield StreamEvent(
                type="thinking_end",
                thinking_text=current_thinking,
                signature=current_signature,
                raw_block=block,
              )
              current_thinking = ""
              current_signature = ""
              current_block_type = None
            elif current_block_type == "tool_use" and current_tool_id is not None:
              try:
                tool_input = json.loads(current_tool_json) if current_tool_json else {}
              except json.JSONDecodeError:
                tool_input = {}
              block = _to_plain_dict(current_tool_block)
              if not isinstance(block, dict):
                block = {"type": "tool_use", "id": current_tool_id, "name": current_tool_name}
              block["input"] = tool_input
              yield StreamEvent(
                type="tool_use_end",
                tool_id=str(current_tool_id),
                tool_name=str(current_tool_name or ""),
                tool_input_json=current_tool_json,
                tool_input=tool_input,
                raw_block=block,
                caller=_to_plain_dict(getattr(current_tool_block, "caller", None)),
              )
              current_tool_id = None
              current_tool_name = None
              current_tool_json = ""
              current_tool_block = None
              current_block_type = None
            elif current_block_type == "compaction":
              yield StreamEvent(
                type="compaction",
                text=current_compaction,
                raw_block={"type": "compaction", "content": current_compaction},
              )
              current_compaction = None
              current_block_type = None
            continue

          if event_type == "message_delta":
            delta = getattr(event, "delta", None)
            stop_reason = str(getattr(delta, "stop_reason", "") or "")
            raw_stop_details = _to_plain_dict(getattr(delta, "stop_details", None))
            stop_details = raw_stop_details if isinstance(raw_stop_details, dict) else None
            usage = getattr(event, "usage", None)
            if usage is not None:
              iterations = _usage_iterations(usage)
              if iterations:
                # Each snapshot's last invocation owns context, even when its
                # required input measurement is absent. Never retain the prefix.
                request_prompt_usage = _anthropic_prompt_usage(iterations[-1])
                request_has_iterations = True
              elif not request_has_iterations:
                # Scalar deltas refer to the same invocation; output-only
                # deltas leave its prompt intact, including explicit zeroes.
                request_prompt_usage.update(_anthropic_prompt_usage(usage))
              usage_totals = _anthropic_usage_totals(usage)
              current_totals = {
                "input_tokens": max(
                  0,
                  int(usage_totals["input_tokens"])
                  - int(message_start_usage["input_tokens"]),
                ),
                "output_tokens": int(usage_totals["output_tokens"]),
                "cache_creation_input_tokens": max(
                  0,
                  int(usage_totals["cache_creation_input_tokens"])
                  - int(message_start_usage["cache_creation_input_tokens"]),
                ),
                "cache_read_input_tokens": max(
                  0,
                  int(usage_totals["cache_read_input_tokens"])
                  - int(message_start_usage["cache_read_input_tokens"]),
                ),
              }
              usage_delta = _unit_delta(current_totals, reported_usage_update)
              provider_unit_deltas = _unit_delta(
                {
                  str(operation): int(count)
                  for operation, count in dict(usage_totals.get("provider_unit_deltas") or {}).items()
                },
                reported_provider_unit_deltas,
              )
              yield StreamEvent(
                type="usage_update",
                input_tokens=usage_delta.get("input_tokens", 0),
                output_tokens=usage_delta.get("output_tokens", 0),
                cache_creation_tokens=usage_delta.get("cache_creation_input_tokens", 0),
                cache_read_tokens=usage_delta.get("cache_read_input_tokens", 0),
                provider_unit_deltas=provider_unit_deltas,
              )
    except Exception as exc:
      _ingest_rate_limit_headers(binding, getattr(getattr(exc, "response", None), "headers", None))
      if self.is_retryable_error(exc):
        raise
      detail = _format_anthropic_rejection_detail(exc)
      if detail is None:
        raise
      context = _stream_request_context(
        stream_params,
        auth_mode=auth_mode,
        use_compaction=use_compaction,
        betas=betas,
      )
      log.warning("Anthropic request rejected during stream: %s; %s", detail, context)
      rejection = RuntimeError(f"Anthropic request rejected (stage=stream): {detail}; {context}")
      _carry_rate_limit_evidence(rejection, exc)
      raise rejection from None

    _record_stream_success(binding)
    yield StreamEvent(
      type="message_end",
      stop_reason=stop_reason,
      stop_details=stop_details,
      request_input_tokens=(
        sum(request_prompt_usage.values()) if "input_tokens" in request_prompt_usage else None
      ),
    )

  def is_retryable_error(self, exc: Exception) -> bool:
    try:
      import httpx2
      from anthropic import APIConnectionError, APIStatusError
    except ImportError:
      return False

    if isinstance(exc, APIStatusError):
      status_code = getattr(exc, "status_code", None)
      if status_code is None:
        response = getattr(exc, "response", None)
        if response is not None:
          status_code = getattr(response, "status_code", None)
      return status_code in {200, 429} or (isinstance(status_code, int) and 500 <= status_code < 600)
    if isinstance(exc, APIConnectionError):
      return True
    if isinstance(exc, (httpx2.TransportError, httpx2.StreamError)):
      return True
    return False
