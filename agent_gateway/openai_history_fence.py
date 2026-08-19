"""OpenAI Responses history markers and durable session epoch helpers.

These primitives are defined by
`docs/design/gateway-openai-responses-migration-plan.md` (section 10, "Durable
history and rollback fence"). They are intentionally self-contained. The
current runtime is Responses-native. The marker predicate now routes
durable persistence and distinguishes legacy Chat history from native replay;
it is not a rejecting fence in this release.

Two independent concerns live here:

1. **History version fence.** The Responses release persists version markers that
   a rolled-back Chat release cannot interpret. `contains_openai_responses_history`
   detects those markers for native replay/persistence routing. The current
   runtime never attempts a Responses -> Chat translation.
2. **Durable session epoch.** `scope_provider_session_id` namespaces durable
   OpenAI session identifiers by epoch so a rollback quarantines old logs by
   namespace rather than deleting or rewriting them.

Non-OpenAI providers are never affected: their identifiers are returned
byte-for-byte unchanged and `OPENAI_SESSION_EPOCH` is never consulted.
"""

from __future__ import annotations

import re
from typing import Any

# --- History version markers (section 10) ---------------------------------

REASONING_SIGNATURE_MARKER = "openai.responses.reasoning.v1"
TEXT_SIGNATURE_MARKER = "openai.responses.text.v1"

DURABLE_HISTORY_VERSION_KEY = "openai_history_version"
RESPONSES_HISTORY_VERSION = "responses-v1"
_RESPONSES_VERSION_PREFIX = "responses-"

_SIGNATURE_MARKERS = frozenset({REASONING_SIGNATURE_MARKER, TEXT_SIGNATURE_MARKER})

# Bound the walk so a hostile or pathological payload cannot exhaust the stack.
_MAX_SCAN_DEPTH = 12

# --- Durable session epoch (section 10) -----------------------------------

OPENAI_SESSION_EPOCH_ENV = "OPENAI_SESSION_EPOCH"
_EPOCH_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,31}$")
_SCOPE_SEPARATOR = "--openai-"
_OPENAI_PROVIDER = "openai"


class OpenAISessionEpochError(RuntimeError):
  """Raised when durable OpenAI execution has no valid session epoch.

  Fail-closed: raised before a model client or durable log is opened.
  """

  requires_new_session = False


def _iter_strings(payload: Any, depth: int = 0) -> Any:
  """Yield every string reachable in a nested mapping/sequence payload."""
  if depth > _MAX_SCAN_DEPTH:
    return
  if isinstance(payload, str):
    yield payload
    return
  if isinstance(payload, dict):
    for key, value in payload.items():
      if isinstance(key, str):
        yield key
      yield from _iter_strings(value, depth + 1)
    return
  if isinstance(payload, (list, tuple, set, frozenset)):
    for item in payload:
      yield from _iter_strings(item, depth + 1)


def _declares_responses_history(payload: Any, depth: int = 0) -> bool:
  """True when a mapping declares a Responses durable-history version."""
  if depth > _MAX_SCAN_DEPTH:
    return False
  if isinstance(payload, dict):
    version = payload.get(DURABLE_HISTORY_VERSION_KEY)
    if isinstance(version, str) and version.strip().lower().startswith(_RESPONSES_VERSION_PREFIX):
      return True
    return any(_declares_responses_history(value, depth + 1) for value in payload.values())
  if isinstance(payload, (list, tuple, set, frozenset)):
    return any(_declares_responses_history(item, depth + 1) for item in payload)
  return False


def contains_openai_responses_history(payload: Any) -> bool:
  """Detect OpenAI Responses-native history markers in messages or events.

  Accepts a single message/event or any nested collection of them. Returns True
  when a Responses reasoning/text signature marker or a Responses durable
  history version is present. The Responses runtime uses this for routing;
  the retained Chat rollback artifact uses it to fail closed.
  """
  if payload is None:
    return False
  if _declares_responses_history(payload):
    return True
  for text in _iter_strings(payload):
    for marker in _SIGNATURE_MARKERS:
      if marker in text:
        return True
  return False


def scope_provider_session_id(
  base_session_id: str,
  *,
  provider: str,
  durable: bool,
  openai_epoch: str | None,
) -> str:
  """Namespace a durable OpenAI session identifier by epoch.

  Contract (section 10):

  - ``provider != "openai"`` returns ``base_session_id`` byte-for-byte unchanged
    and never reads the epoch.
  - Non-durable interactive sessions keep their server-generated identifier and
    do not require an epoch.
  - Durable OpenAI execution requires a non-blank epoch matching
    ``[a-z0-9][a-z0-9._-]{0,31}`` and fails closed when unset or invalid.
  - Output is deterministically ``base_session_id + "--openai-" + epoch``.
    Re-applying the same epoch is idempotent; a different existing epoch suffix
    is rejected.
  """
  if str(provider or "").strip().lower() != _OPENAI_PROVIDER:
    return base_session_id
  if not durable:
    return base_session_id

  epoch = str(openai_epoch or "").strip()
  if not epoch:
    raise OpenAISessionEpochError(
      f"Durable OpenAI execution requires {OPENAI_SESSION_EPOCH_ENV} to be set."
    )
  if not _EPOCH_PATTERN.match(epoch):
    raise OpenAISessionEpochError(
      f"Invalid {OPENAI_SESSION_EPOCH_ENV} value {epoch!r}; "
      "expected [a-z0-9][a-z0-9._-]{0,31}."
    )

  if _SCOPE_SEPARATOR in base_session_id:
    existing = base_session_id.rsplit(_SCOPE_SEPARATOR, 1)[1]
    if existing == epoch:
      return base_session_id
    raise OpenAISessionEpochError(
      f"Session identifier is already scoped to OpenAI epoch {existing!r}; "
      f"refusing to re-scope to {epoch!r}."
    )

  return f"{base_session_id}{_SCOPE_SEPARATOR}{epoch}"


__all__ = [
  "DURABLE_HISTORY_VERSION_KEY",
  "OPENAI_SESSION_EPOCH_ENV",
  "OpenAISessionEpochError",
  "REASONING_SIGNATURE_MARKER",
  "RESPONSES_HISTORY_VERSION",
  "TEXT_SIGNATURE_MARKER",
  "contains_openai_responses_history",
  "scope_provider_session_id",
]
