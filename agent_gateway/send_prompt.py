from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from .capability_binding import (
  require_capability_execution_bind,
  validate_reported_identity,
)
from .capability_execution import (
  BoundCapabilityExecution,
)
from .commercial_usage import CommercialUsageProducer
from .multi_user.billing import UsageEvent, normalize_identity
from .thinking import EffortResolution


log = logging.getLogger("agent_gateway.send_prompt")


def _call_usage_callback(
  cb: Callable[..., Any],
  usage_event: UsageEvent,
) -> None:
  cb(usage_event)


def _prepare_bound_execution(
  capability_execution: BoundCapabilityExecution,
) -> tuple[dict[str, Any], int, EffortResolution]:
  if not isinstance(capability_execution, BoundCapabilityExecution):
    raise TypeError("send_prompt requires a BoundCapabilityExecution")
  capability_execution.validate()

  capability_bind = capability_execution.bind
  provider = capability_execution.provider
  config = dict(capability_execution.auth_config)
  max_tokens = config.get("max_tokens")
  if (
    isinstance(max_tokens, bool)
    or not isinstance(max_tokens, int)
    or max_tokens <= 0
  ):
    raise ValueError(
      "bound_auth_config.max_tokens must be a positive integer"
    )

  model_info = provider.get_model_info(capability_bind.upstream_model)
  model_max_output = int(getattr(model_info, "max_output_tokens", 0) or 0)
  if model_max_output > 0 and max_tokens > model_max_output:
    max_tokens = model_max_output

  effort_resolution = require_capability_execution_bind(
    capability_bind,
    provider=provider,
    auth_config=config,
  )
  return config, max_tokens, effort_resolution


async def send_prompt(
  prompt: str,
  *,
  capability_execution: BoundCapabilityExecution,
  user_id: str,
  system_prompt: str | list[tuple[str, bool]] | None = None,
  client_timeout: float = 180.0,
  session_id: str | None = None,
  request_id: str | None = None,
  rate_table_version: str | None = None,
  billing_mode: str | None = None,
  channel: str | None = None,
  on_usage: Callable[..., None] | None = None,
  commercial_usage_producer: CommercialUsageProducer | None = None,
) -> str:
  """Execute one already-resolved capability bind and return plain text.

  Capability, provider, credential, model, and effort selection must happen
  before this execution-only helper is called. The exact provider adapter and
  bound credential snapshot are validated before a provider client is created.
  """
  sid = str(session_id or "send-prompt")
  normalized_request_id = str(request_id or "").strip()
  normalized_rate_version = str(rate_table_version or "").strip()
  normalized_billing_mode = str(billing_mode or "").strip()
  normalized_channel = str(channel or "").strip()
  if commercial_usage_producer is not None and (
    not normalized_request_id
    or not normalized_rate_version
    or not normalized_billing_mode
    or not normalized_channel
  ):
    raise ValueError(
      "commercial send_prompt requires request_id, rate_table_version, billing_mode, and channel"
    )
  usage_user_id, resolved_rate_version, resolved_billing_mode, resolved_channel = normalize_identity(
    user_id,
    normalized_rate_version or "unknown",
    normalized_billing_mode or "byok",
    normalized_channel or None,
  )
  usage_request_id = normalized_request_id or sid
  if commercial_usage_producer is not None:
    commercial_guard = getattr(commercial_usage_producer, "assert_work_allowed", None)
    if callable(commercial_guard):
      commercial_guard(resolved_billing_mode)
  config, max_tokens, effort_resolution = _prepare_bound_execution(
    capability_execution
  )
  capability_bind = capability_execution.bind
  provider = capability_execution.provider
  model = capability_bind.upstream_model
  thinking_level = effort_resolution.requested
  client: Any = None
  text_parts: list[str] = []
  input_tokens = 0
  output_tokens = 0
  cache_read_tokens = 0
  cache_creation_tokens = 0
  reasoning_tokens = 0
  provider_unit_deltas: dict[str, int] = {}
  provider_reported_model: str | None = None
  stop_reason = ""

  async def _emit_usage(usage_state: str) -> None:
    if on_usage is None and commercial_usage_producer is None:
      return
    cost = provider.estimate_cost(
      model,
      input_tokens,
      output_tokens,
      cache_read_tokens=cache_read_tokens,
      cache_creation_tokens=cache_creation_tokens,
    )
    usage_event = UsageEvent(
      user_id=usage_user_id,
      session_id=sid,
      request_id=usage_request_id,
      parent_turn_id=None,
      timestamp=time.time(),
      model=model,
      provider=capability_bind.provider,
      input_tokens=input_tokens,
      output_tokens=output_tokens,
      reasoning_tokens_observed=reasoning_tokens,
      provider_unit_deltas=provider_unit_deltas or None,
      cache_read_tokens=cache_read_tokens,
      cache_creation_tokens=cache_creation_tokens,
      cost_usd=cost.total,
      rate_table_version=resolved_rate_version,
      billing_mode=resolved_billing_mode,
      channel=resolved_channel,
      capability_bind=capability_bind.receipt(),
      provider_reported_model=provider_reported_model,
    )
    if commercial_usage_producer is not None:
      await commercial_usage_producer.emit(usage_event, usage_state=usage_state)
    if on_usage is not None:
      _call_usage_callback(on_usage, usage_event)

  try:
    client = provider.create_client(config=config, timeout=client_timeout)
    params = provider.build_request_params(
      model=model,
      messages=[{"role": "user", "content": prompt}],
      system_prompt=system_prompt,
      tools=[],
      max_tokens=max_tokens,
      thinking_level=thinking_level,
      effort_resolution=effort_resolution,
      auth_mode=config.get("auth_mode"),
    )

    try:
      async for event in provider.stream(client, params):
        if event.type == "text_delta":
          text = str(event.text or "")
          if text:
            text_parts.append(text)
          continue

        if event.type == "message_start":
          input_tokens += int(event.input_tokens or 0)
          cache_read_tokens += int(event.cache_read_tokens or 0)
          cache_creation_tokens += int(event.cache_creation_tokens or 0)
          if event.provider_reported_model is not None:
            provider_reported_model = validate_reported_identity(
              capability_bind,
              event.provider_reported_model,
              registry=capability_execution.registry,
            )
          if event.provider_unit_deltas:
            provider_unit_deltas.update(event.provider_unit_deltas)
          continue

        if event.type == "usage_update":
          output_tokens += int(event.output_tokens or 0)
          reasoning_tokens += int(event.reasoning_tokens or 0)
          if event.provider_unit_deltas:
            provider_unit_deltas.update(event.provider_unit_deltas)
          continue

        if event.type == "message_end":
          stop_reason = str(event.stop_reason or "")
    except asyncio.CancelledError:
      if input_tokens or output_tokens or cache_read_tokens or cache_creation_tokens:
        await _emit_usage("canceled")
      raise
    except Exception:
      if input_tokens or output_tokens or cache_read_tokens or cache_creation_tokens:
        await _emit_usage("failed_billable")
      raise

    await _emit_usage("succeeded")

    log.info(
      "[%s] Prompt done | model=%s | stop=%s | tokens in=%d out=%d | cache read=%d create=%d",
      sid,
      model,
      stop_reason or "unknown",
      input_tokens,
      output_tokens,
      cache_read_tokens,
      cache_creation_tokens,
      extra={
        "data": {
          "event": "send_prompt_done",
          "session_id": sid,
          "model": model,
          "stop_reason": stop_reason,
          "tokens_in": input_tokens,
          "tokens_out": output_tokens,
          "cache_read": cache_read_tokens,
          "cache_write": cache_creation_tokens,
        }
      },
    )

    return "".join(text_parts).strip()
  finally:
    await provider.close_client(client)


def send_prompt_sync(
  prompt: str,
  *,
  capability_execution: BoundCapabilityExecution,
  **kwargs: Any,
) -> str:
  """Run `send_prompt()` from synchronous code."""
  try:
    asyncio.get_running_loop()
  except RuntimeError:
    return asyncio.run(
      send_prompt(
        prompt,
        capability_execution=capability_execution,
        **kwargs,
      )
    )
  raise RuntimeError("send_prompt_sync cannot run inside an active asyncio loop.")
