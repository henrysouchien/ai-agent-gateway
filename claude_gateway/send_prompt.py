from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable

from .providers.anthropic import AnthropicProvider
from .providers.base import ThinkingLevel


log = logging.getLogger("claude_gateway.send_prompt")


def _infer_auth_config_from_env() -> dict[str, Any]:
  api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
  auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN", "").strip()

  if api_key:
    return {
      "auth_mode": "api",
      "api_key": api_key,
      "auth_token": "",
    }
  if auth_token:
    return {
      "auth_mode": "oauth",
      "api_key": "",
      "auth_token": auth_token,
    }
  raise RuntimeError("No Anthropic credentials found")


def _prepare_auth_config(
  auth_config: dict[str, Any] | None,
  *,
  model: str,
  max_tokens: int,
  thinking: bool,
) -> dict[str, Any]:
  config = dict(auth_config) if auth_config is not None else _infer_auth_config_from_env()
  auth_mode = str(config.get("auth_mode", "")).strip().lower()
  if not auth_mode:
    if str(config.get("api_key", "")).strip():
      auth_mode = "api"
    elif str(config.get("auth_token", "")).strip():
      auth_mode = "oauth"
    else:
      raise RuntimeError("No Anthropic credentials found")

  config["auth_mode"] = auth_mode
  config["model"] = model
  config["max_tokens"] = max_tokens
  config["thinking"] = thinking
  return config


async def send_prompt(
  prompt: str,
  *,
  model: str,
  system_prompt: str | list[tuple[str, bool]] | None = None,
  max_tokens: int = 4096,
  thinking: bool = False,
  auth_config: dict[str, Any] | None = None,
  client_timeout: float = 180.0,
  session_id: str | None = None,
  on_usage: Callable[[int, int, int, int], None] | None = None,
) -> str:
  """Send one prompt through `AnthropicProvider` and return plain text.

  This helper is useful when you want provider normalization and usage tracking
  without standing up the full gateway server. It supports the same cached
  system prompt block format as `create_agent()`.
  """
  provider = AnthropicProvider()
  config = _prepare_auth_config(
    auth_config,
    model=model,
    max_tokens=max_tokens,
    thinking=thinking,
  )
  if not provider.has_active_credential(config):
    raise RuntimeError("No Anthropic credentials found")

  thinking_level = ThinkingLevel.HIGH if thinking else ThinkingLevel.NONE
  sid = str(session_id or "send-prompt")
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
      on_usage(input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens)

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
