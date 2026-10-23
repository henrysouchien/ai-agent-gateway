"""Request id and rate-limit headers off a successful provider HTTP response.

Diagnostics for `StreamEvent.provider_request_id` / `rate_limit` on message_start.
Every adapter (Anthropic, Responses-family, chat-completions, Gemini, Ollama) uses
`response_diagnostics`; no adapter keeps its own copy.
"""

from __future__ import annotations

from typing import Any

_REQUEST_ID_HEADERS = ("request-id", "x-request-id", "anthropic-request-id")
_RATE_LIMIT_PREFIXES = ("x-ratelimit-", "ratelimit-", "anthropic-ratelimit-")


def response_diagnostics(headers: Any) -> tuple[str | None, dict[str, str] | None]:
  """`(request_id, rate_limit)` from a response header mapping.

  Names are lowercased and values verbatim; an unreadable mapping yields
  nothing rather than failing the stream.
  """

  if headers is None:
    return None, None
  try:
    lowered = {str(name).lower(): str(value) for name, value in headers.items()}
  except Exception:
    return None, None
  request_id = next(
    (stripped for name in _REQUEST_ID_HEADERS if (stripped := lowered.get(name, "").strip())),
    None,
  )
  rate_limit = {
    name: value
    for name, value in lowered.items()
    if name.startswith(_RATE_LIMIT_PREFIXES) or name == "retry-after"
  }
  return request_id, rate_limit or None
