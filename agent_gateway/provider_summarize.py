from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .capability_binding import validate_reported_identity
from .capability_execution import BoundCapabilityExecution
from .thinking import parse_effort


DEFAULT_SUMMARY_SYSTEM_PROMPT = "You write cumulative narrative summaries of autonomous analyst sessions."


@dataclass(frozen=True)
class ProviderSummarizeResult:
  text: str
  usage: dict[str, Any]
  saw_tool_use: bool = False


def _empty_usage() -> dict[str, Any]:
  return {
    "input_tokens": 0,
    "output_tokens": 0,
    "reasoning_tokens_observed": 0,
    "provider_units": 0,
    "provider_unit_deltas": {},
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
  }


def _apply_provider_unit_deltas(usage: dict[str, Any], deltas: dict[str, int] | None) -> None:
  accumulated = usage.setdefault("provider_unit_deltas", {})
  if not isinstance(accumulated, dict):
    accumulated = {}
    usage["provider_unit_deltas"] = accumulated
  for operation, count in (deltas or {}).items():
    accumulated[str(operation)] = int(accumulated.get(str(operation), 0) or 0) + int(count or 0)


def _add_usage_from_event(usage: dict[str, Any], event: Any) -> None:
  event_type = getattr(event, "type", "")
  if event_type == "message_start":
    usage["input_tokens"] = int(usage.get("input_tokens", 0) or 0) + int(getattr(event, "input_tokens", 0) or 0)
    usage["cache_creation_input_tokens"] = (
      int(usage.get("cache_creation_input_tokens", 0) or 0)
      + int(getattr(event, "cache_creation_tokens", 0) or 0)
    )
    usage["cache_read_input_tokens"] = (
      int(usage.get("cache_read_input_tokens", 0) or 0)
      + int(getattr(event, "cache_read_tokens", 0) or 0)
    )
    usage["provider_units"] = int(usage.get("provider_units", 0) or 0) + int(getattr(event, "provider_units", 0) or 0)
    _apply_provider_unit_deltas(usage, getattr(event, "provider_unit_deltas", None))
    return

  if event_type == "usage_update":
    usage["input_tokens"] = int(usage.get("input_tokens", 0) or 0) + int(getattr(event, "input_tokens", 0) or 0)
    usage["cache_creation_input_tokens"] = (
      int(usage.get("cache_creation_input_tokens", 0) or 0)
      + int(getattr(event, "cache_creation_tokens", 0) or 0)
    )
    usage["cache_read_input_tokens"] = (
      int(usage.get("cache_read_input_tokens", 0) or 0)
      + int(getattr(event, "cache_read_tokens", 0) or 0)
    )
    usage["output_tokens"] = int(usage.get("output_tokens", 0) or 0) + int(getattr(event, "output_tokens", 0) or 0)
    usage["reasoning_tokens_observed"] = (
      int(usage.get("reasoning_tokens_observed", 0) or 0)
      + int(getattr(event, "reasoning_tokens", 0) or 0)
    )
    usage["provider_units"] = int(usage.get("provider_units", 0) or 0) + int(getattr(event, "provider_units", 0) or 0)
    _apply_provider_unit_deltas(usage, getattr(event, "provider_unit_deltas", None))


async def provider_summarize(
  *,
  capability_execution: BoundCapabilityExecution,
  messages: list[dict[str, Any]],
  system_prompt: str | list[tuple[str, bool]] | None,
  tools: list[dict[str, Any]] | None = None,
  max_tokens: int,
  timeout: float = 60.0,
  request_kwargs: dict[str, Any] | None = None,
) -> ProviderSummarizeResult:
  if not isinstance(capability_execution, BoundCapabilityExecution):
    raise TypeError(
      "provider summarize requires a BoundCapabilityExecution"
    )
  capability_execution.validate()
  provider = capability_execution.provider
  model = capability_execution.bind.upstream_model
  requested_effort = parse_effort(
    capability_execution.bind.effort,
    field_name="capability_execution.effort",
  )
  if requested_effort is None:
    raise ValueError("provider summarize requires an explicitly bound effort")

  run_config = dict(capability_execution.auth_config)
  run_config["auth_mode"] = str(run_config.get("auth_mode", "api")).strip().lower() or "api"
  if not provider.has_active_credential(run_config):
    raise RuntimeError(f"No active credential configured for provider={provider.name}")

  model_info = provider.get_model_info(model)
  token_limit = int(max_tokens)
  if token_limit <= 0:
    raise ValueError("provider summarize max_tokens must be positive")
  token_limit = min(token_limit, int(getattr(model_info, "max_output_tokens", token_limit) or token_limit))
  effort_resolution = provider.resolve_effort(
    requested=requested_effort,
    model=model,
    model_info=model_info,
    max_tokens=token_limit,
    auth_mode=run_config.get("auth_mode"),
    base_url=run_config.get("base_url") or run_config.get("baseURL"),
    compat=run_config.get("compat"),
  )
  if (
    effort_resolution.requested != requested_effort
    or effort_resolution.effective != requested_effort
  ):
    raise ValueError(
      (
        f"provider summarize cannot preserve bound effort "
        f"{requested_effort.value!r} for {provider.name}:{model} "
        f"with max_tokens={token_limit}"
      )
    )
  client = provider.create_client(run_config, timeout=float(run_config.get("client_timeout") or timeout))
  try:
    kwargs = dict(request_kwargs or {})
    kwargs.update(
      {
        "auth_mode": run_config["auth_mode"],
        "base_url": run_config.get("base_url") or run_config.get("baseURL"),
        "compat": run_config.get("compat"),
        "compaction_trigger": None,
        "compaction_instructions": None,
      }
    )
    params = provider.build_request_params(
      model=model,
      messages=messages,
      system_prompt=system_prompt,
      tools=list(tools or []),
      max_tokens=token_limit,
      thinking_level=requested_effort,
      effort_resolution=effort_resolution,
      **kwargs,
    )

    pieces: list[str] = []
    fallback_blocks: list[str] = []
    usage = _empty_usage()
    bind = capability_execution.bind
    usage.update({
      "capability_bind": bind.receipt(),
    })
    saw_tool_use = False
    async for event in provider.stream(client, params):
      event_type = getattr(event, "type", "")
      reported_model = getattr(event, "provider_reported_model", None)
      if event_type == "message_start" and reported_model is not None:
        usage["provider_reported_model"] = validate_reported_identity(
          bind,
          reported_model,
          registry=capability_execution.registry,
        )
      _add_usage_from_event(usage, event)
      if event_type == "text_delta":
        text = str(getattr(event, "text", "") or "")
        if text:
          pieces.append(text)
      elif event_type == "text_end" and isinstance(getattr(event, "raw_block", None), dict):
        block_text = str(event.raw_block.get("text") or "").strip()
        if block_text:
          fallback_blocks.append(block_text)
      elif event_type in {"tool_use_start", "tool_use_end"}:
        saw_tool_use = True

    summary_text = "".join(pieces).strip()
    if not summary_text:
      summary_text = "\n".join(fallback_blocks).strip()
    if not summary_text:
      raise RuntimeError("Summary generation returned empty text")
    return ProviderSummarizeResult(text=summary_text, usage=usage, saw_tool_use=saw_tool_use)
  finally:
    await provider.close_client(client, timeout=5.0)


async def _provider_summarize(
  prompt_text: str,
  *,
  capability_execution: BoundCapabilityExecution,
  system_prompt: str | list[tuple[str, bool]] | None = DEFAULT_SUMMARY_SYSTEM_PROMPT,
  max_tokens: int = 4096,
) -> str:
  result = await provider_summarize(
    capability_execution=capability_execution,
    messages=[{"role": "user", "content": prompt_text}],
    system_prompt=system_prompt,
    tools=[],
    max_tokens=max_tokens,
  )
  return result.text


__all__ = [
  "DEFAULT_SUMMARY_SYSTEM_PROMPT",
  "ProviderSummarizeResult",
  "_provider_summarize",
  "provider_summarize",
]
