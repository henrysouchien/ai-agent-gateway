from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from ._provider_utils import resolve_auth_config
from .multi_user.billing import UsageEvent, normalize_identity
from .providers.anthropic import AnthropicProvider
from .providers.base import ThinkingLevel


log = logging.getLogger("agent_gateway.send_prompt")


def _call_usage_callback(
  cb: Callable[..., Any],
  usage_event: UsageEvent,
) -> None:
  cb(usage_event)


async def send_prompt(
  prompt: str,
  *,
  model: str,
  user_id: str,
  system_prompt: str | list[tuple[str, bool]] | None = None,
  max_tokens: int = 4096,
  thinking: bool = False,
  auth_config: dict[str, Any] | None = None,
  client_timeout: float = 180.0,
  session_id: str | None = None,
  on_usage: Callable[..., None] | None = None,
) -> str:
  """Send one prompt through `AnthropicProvider` and return plain text.

  This helper is useful when you want provider normalization and usage tracking
  without standing up the full gateway server. It supports the same cached
  system prompt block format as `create_agent()`.
  """
  provider = AnthropicProvider()
  config = resolve_auth_config(
    auth_config=auth_config,
    model=model,
    max_tokens=max_tokens,
    thinking=thinking,
  )
  if not provider.has_active_credential(config):
    raise RuntimeError("No Anthropic credentials found")

  thinking_level = ThinkingLevel.HIGH if thinking else ThinkingLevel.NONE
  sid = str(session_id or "send-prompt")
  usage_user_id, rate_table_version, billing_mode, channel = normalize_identity(user_id, "unknown", "byok", None)
  client: Any = None
  text_parts: list[str] = []
  input_tokens = 0
  output_tokens = 0
  cache_read_tokens = 0
  cache_creation_tokens = 0
  stop_reason = ""

  try:
    client = provider.create_client(config=config, timeout=client_timeout)
    params = provider.build_request_params(
      model=model,
      messages=[{"role": "user", "content": prompt}],
      system_prompt=system_prompt,
      tools=[],
      max_tokens=max_tokens,
      thinking_level=thinking_level,
      auth_mode=config["auth_mode"],
    )

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
        continue

      if event.type == "usage_update":
        output_tokens += int(event.output_tokens or 0)
        continue

      if event.type == "message_end":
        stop_reason = str(event.stop_reason or "")

    if on_usage is not None:
      usage_event = UsageEvent(
        user_id=usage_user_id,
        session_id=sid,
        request_id=sid,
        parent_turn_id=None,
        timestamp=time.time(),
        model=model,
        provider=provider.name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cost_usd=0.0,
        rate_table_version=rate_table_version,
        billing_mode=billing_mode,
        channel=channel,
      )
      _call_usage_callback(on_usage, usage_event)

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


def send_prompt_sync(prompt: str, **kwargs: Any) -> str:
  """Run `send_prompt()` from synchronous code."""
  try:
    asyncio.get_running_loop()
  except RuntimeError:
    return asyncio.run(send_prompt(prompt, **kwargs))
  raise RuntimeError("send_prompt_sync cannot run inside an active asyncio loop.")
