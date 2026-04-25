from __future__ import annotations

import asyncio
from collections import OrderedDict
import inspect
import logging
import time
from typing import Any, Callable, Literal
import warnings
import weakref

from ._provider_utils import resolve_auth_config
from .multi_user.billing import UsageEvent
from .providers.anthropic import AnthropicProvider
from .providers.base import ThinkingLevel


log = logging.getLogger("agent_gateway.send_prompt")

UsageCallbackKind = Literal["legacy", "modern"]
_CLASSIFIED_CALLBACKS: weakref.WeakKeyDictionary[Any, UsageCallbackKind] = weakref.WeakKeyDictionary()
_WARNED_CALLBACKS: weakref.WeakSet[Any] = weakref.WeakSet()
_CLASSIFIED_CALLBACK_IDS: OrderedDict[int, UsageCallbackKind] = OrderedDict()
_WARNED_CALLBACK_IDS: OrderedDict[int, None] = OrderedDict()
_FALLBACK_CACHE_LIMIT = 1024


def _cache_get(cb: Callable[..., Any]) -> UsageCallbackKind | None:
  try:
    return _CLASSIFIED_CALLBACKS.get(cb)
  except TypeError:
    value = _CLASSIFIED_CALLBACK_IDS.get(id(cb))
    if value is not None:
      _CLASSIFIED_CALLBACK_IDS.move_to_end(id(cb))
    return value


def _cache_set(cb: Callable[..., Any], kind: UsageCallbackKind) -> None:
  try:
    _CLASSIFIED_CALLBACKS[cb] = kind
    return
  except TypeError:
    key = id(cb)
    _CLASSIFIED_CALLBACK_IDS[key] = kind
    _CLASSIFIED_CALLBACK_IDS.move_to_end(key)
    while len(_CLASSIFIED_CALLBACK_IDS) > _FALLBACK_CACHE_LIMIT:
      _CLASSIFIED_CALLBACK_IDS.popitem(last=False)


def _mark_warned(cb: Callable[..., Any]) -> bool:
  try:
    if cb in _WARNED_CALLBACKS:
      return False
    _WARNED_CALLBACKS.add(cb)
    return True
  except TypeError:
    key = id(cb)
    if key in _WARNED_CALLBACK_IDS:
      _WARNED_CALLBACK_IDS.move_to_end(key)
      return False
    _WARNED_CALLBACK_IDS[key] = None
    while len(_WARNED_CALLBACK_IDS) > _FALLBACK_CACHE_LIMIT:
      _WARNED_CALLBACK_IDS.popitem(last=False)
    return True


def _warn_once(cb: Callable[..., Any], message: str, *, deprecation: bool = False) -> None:
  if not _mark_warned(cb):
    return
  if deprecation:
    warnings.warn(message, DeprecationWarning, stacklevel=3)
  else:
    log.warning(message)


def _classify_usage_callback(cb: Callable[..., Any]) -> UsageCallbackKind:
  cached = _cache_get(cb)
  if cached is not None:
    return cached

  try:
    signature = inspect.signature(cb)
  except (TypeError, ValueError):
    _warn_once(
      cb,
      "could not introspect callback; assuming modern signature. If this is a 4-int legacy callback, "
      "migrate to UsageEvent or wrap it with functools.wraps for introspection.",
    )
    _cache_set(cb, "modern")
    return "modern"

  params = list(signature.parameters.values())
  positional = [
    param
    for param in params
    if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
  ]
  if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params):
    _warn_once(cb, "callback uses *args; treating as modern. Legacy 4-int *args callbacks must migrate.")
    _cache_set(cb, "modern")
    return "modern"

  required = [param for param in positional if param.default is inspect.Parameter.empty]
  if len(required) == 4:
    _warn_once(
      cb,
      "send_prompt(on_usage=...) 4-int callbacks are deprecated and will be removed in v0.9.0. "
      "Use `def on_usage(event: UsageEvent): ...` for per-call usage, or aggregate with "
      "SessionUsageSummary in AgentRunner session hooks.",
      deprecation=True,
    )
    _cache_set(cb, "legacy")
    return "legacy"
  if len(required) == 1 or (len(required) == 0 and positional):
    _cache_set(cb, "modern")
    return "modern"

  _warn_once(cb, f"unusual on_usage callback signature {signature}; treating as modern UsageEvent callback.")
  _cache_set(cb, "modern")
  return "modern"


def _call_usage_callback(
  cb: Callable[..., Any],
  usage_event: UsageEvent,
) -> None:
  if _classify_usage_callback(cb) == "legacy":
    cb(
      usage_event.input_tokens,
      usage_event.output_tokens,
      usage_event.cache_read_tokens,
      usage_event.cache_creation_tokens,
    )
    return
  cb(usage_event)


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
        user_id="_default",
        session_id=sid,
        request_id=sid,
        parent_turn_id=None,
        timestamp=time.time(),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cost_usd=0.0,
        rate_table_version="unknown",
        billing_mode="byok",
        channel=None,
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
