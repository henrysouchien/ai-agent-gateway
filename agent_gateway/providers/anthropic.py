from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from typing import Any, AsyncIterator, Dict, Literal

from ..rate_limit import get_global_token_bucket
from ..rates import RateTable, UnknownModelError, load_rate_table
from .base import CostEstimate, ModelInfo, ModelProvider, StreamEvent, ThinkingLevel, truncate_to_last_compaction
from ..model_registry import AdapterRouteSupport
from ..thinking import EffortResolution, clamp_effort
from .anthropic_helpers import (
  _COMMON_BETA_SLUGS as _COMMON_BETA_SLUGS,
  _COMPACTION_BETA_SLUG as _COMPACTION_BETA_SLUG,
  _ERROR_REDACTION as _ERROR_REDACTION,
  _MAX_ERROR_DETAIL_LEN as _MAX_ERROR_DETAIL_LEN,
  _MAX_TOOL_ID_LEN as _MAX_TOOL_ID_LEN,
  _MODEL_INFO_BY_TAG as _MODEL_INFO_BY_TAG,
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
  _model_info_for_model as _model_info_for_model,
  _model_matches_tag as _model_matches_tag,
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


log = logging.getLogger("agent_gateway.providers.anthropic")
_DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
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
  return [
    (message_index, block_index)
    for message_index, message in enumerate(messages)
    for block_index, block in enumerate(
      message.get("content") if isinstance(message.get("content"), list) else []
    )
    if isinstance(block, dict) and "cache_control" in block
  ]


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


def _server_tool_units(usage: Any) -> int:
  """Compatibility helper for callers that require exactly one unit kind."""
  deltas = _server_tool_unit_deltas(usage)
  if len(deltas) > 1:
    raise ValueError("multiple separately billed Anthropic unit kinds require distinct events")
  return next(iter(deltas.values()), 0)


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
    self._rate_table = rate_table or load_rate_table()

  @staticmethod
  def thinking_param(model: str, max_tokens: int) -> dict[str, Any] | None:
    model_id = str(model or "").strip()
    if not model_id.startswith("claude"):
      return None
    return _thinking_param(_model_info_for_model(model_id), max_tokens)

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    if str(config.get("auth_mode", "api")).strip().lower() == "oauth":
      return bool(str(config.get("auth_token") or "").strip())
    return bool(str(config.get("api_key") or "").strip())

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    try:
      from anthropic import AsyncAnthropic, Omit
      import httpx
    except ImportError as exc:
      raise RuntimeError("anthropic dependency is required to use AnthropicProvider") from exc

    mode = str(config.get("auth_mode", "api")).strip().lower()
    credential_field = "auth_token" if mode == "oauth" else "api_key"
    credential = str(config.get(credential_field) or "").strip()
    if not credential:
      raise RuntimeError(f"No Anthropic {mode} credential configured")

    client_kwargs: Dict[str, Any] = {
      "base_url": _bound_anthropic_base_url(config),
    }
    if timeout is not None:
      client_kwargs["timeout"] = httpx.Timeout(timeout=timeout, connect=5.0)

    if mode == "oauth":
      oauth_headers = {
        "X-Api-Key": Omit(),
        "anthropic-beta": ",".join([*_OAUTH_BETA_SLUGS, *_COMMON_BETA_SLUGS]),
        "user-agent": "claude-cli/2026.3.14",
        "x-app": "cli",
      }

      return AsyncAnthropic(
        api_key="",
        auth_token=credential,
        default_headers=oauth_headers,
        **client_kwargs,
      )

    api_headers: Dict[str, Any] = {
      "Authorization": Omit(),
    }
    if _COMMON_BETA_SLUGS:
      api_headers["anthropic-beta"] = ",".join(_COMMON_BETA_SLUGS)
    return AsyncAnthropic(
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
    model_id = str(model or "").strip()
    if not model_id:
      raise ValueError("Model is required")
    if not model_id.startswith("claude"):
      raise ValueError(f"AnthropicProvider does not recognize model: {model_id}")
    model_info = _model_info_for_model(model_id)
    try:
      rates = self._rate_table.lookup(self.name, model_id)
    except UnknownModelError:
      return model_info
    return replace(
      model_info,
      context_window=rates.context_window or 200_000,
      max_output_tokens=rates.max_tokens or 16_384,
      input_cost_per_mtok=rates.input_cost_per_mtok,
      output_cost_per_mtok=rates.output_cost_per_mtok,
      cache_read_cost_per_mtok=rates.cache_read_cost_per_mtok,
      cache_write_cost_per_mtok=rates.cache_write_cost_per_mtok,
    )

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

    prepared_tools = _prepare_anthropic_tools(tools)
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
    if not model_info.supports_thinking or model_info.thinking_mode == "none":
      return EffortResolution(requested, ThinkingLevel.NONE, False, {})

    compat = dict(model_info.compat or {})
    if model_info.thinking_mode == "budget":
      if requested == ThinkingLevel.NONE:
        return EffortResolution(requested, ThinkingLevel.NONE, False, {})
      thinking = _thinking_param(model_info, max_tokens)
      return EffortResolution(
        requested,
        ThinkingLevel.HIGH if thinking is not None else ThinkingLevel.NONE,
        thinking is not None,
        {"thinking": thinking} if thinking is not None else {},
      )

    disable = str(compat.get("thinking_disable", "omit"))
    default_effort = ThinkingLevel(str(compat.get("thinking_default_effort", "none")))
    omitted_on = str(compat.get("thinking_default_when_omitted", "off")) == "on"
    supported = tuple(ThinkingLevel(str(value)) for value in compat.get("effort_values", ()))

    if requested == ThinkingLevel.NONE:
      if disable == "disabled":
        return EffortResolution(requested, ThinkingLevel.NONE, False, {"thinking": {"type": "disabled"}})
      if disable == "unsupported":
        return EffortResolution(requested, default_effort, True, {})
      return EffortResolution(requested, ThinkingLevel.NONE, False, {})

    normalized = ThinkingLevel.LOW if requested == ThinkingLevel.MINIMAL else requested
    effective = clamp_effort(normalized, supported) if supported else ThinkingLevel.NONE
    if max_tokens < 2048:
      return EffortResolution(
        requested,
        default_effort if omitted_on else ThinkingLevel.NONE,
        omitted_on,
        {},
      )
    fragments: dict[str, Any] = {"thinking": {"type": "adaptive"}}
    if compat.get("supports_output_config_effort") and effective != ThinkingLevel.NONE:
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
              next_content.append({"type": "text", "text": thinking_text})
            continue

          if block_type in {"tool_use", "server_tool_use"}:
            next_block = dict(block)
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

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
    bucket = get_global_token_bucket()
    if bucket is not None:
      await bucket.acquire(estimated_tokens=int(params.get("max_tokens") or 0))

    stream_params = dict(params)
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
        async for event in stream_obj:
          event_type = getattr(event, "type", None)

          if event_type == "ping":
            yield StreamEvent(type="heartbeat")
            continue

          if event_type == "message_start":
            message = getattr(event, "message", None)
            usage = getattr(message, "usage", None) if message is not None else None
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
              try:
                from agent.shared.tool_redaction import get_audit_hmac_secret, redact_tool_input

                block["input"] = redact_tool_input(
                  str(current_tool_name or ""),
                  tool_input,
                  deployment_secret=get_audit_hmac_secret(),
                )
              except Exception:
                from ..secret_boundary import sanitization_failure_tool_input

                block["input"] = sanitization_failure_tool_input()
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
            usage = getattr(event, "usage", None)
            if usage is not None:
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
      raise RuntimeError(f"Anthropic request rejected (stage=stream): {detail}; {context}") from None

    yield StreamEvent(type="message_end", stop_reason=stop_reason)

  def is_retryable_error(self, exc: Exception) -> bool:
    try:
      import httpx
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
    if isinstance(exc, (httpx.TransportError, httpx.StreamError)):
      return True
    return False

  def estimate_cost(
    self,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
  ) -> CostEstimate:
    return super().estimate_cost(
      model,
      input_tokens,
      output_tokens,
      cache_read_tokens=cache_read_tokens,
      cache_creation_tokens=cache_creation_tokens,
    )
