"""Total classification of one tool dispatch, and the retry decision.

Three status dialects reach the tool boundary today — ``status == "error"`` /
``success is False`` (the semantic classifier), ``status == "success"`` (the
corpus/parser extraction chain) and ``status == "ok"`` (the vendor and
computation partial extractors).  Each was read by a different consumer, and
nothing settled a single normalized answer, which is how a rate-limited vendor
payload could reach the citation minting path and mint a source
(``ToolResultContext`` carries no ``semantic_error``, and the vendor extractor
success-checks only ``gsheets_read_range``).

This module absorbs all three into one total function.  ``classify_tool_outcome``
answers with a single normalized outcome, and source identities are extracted
**only inside the ``ok`` arm** — the 429-minting hole closes by sequencing, not
by another guard.

``classify_semantic_tool_error`` is re-exported here so the runner derives the
event's ``semantic_error`` payload from this module; the payload itself stays
on the event (D-B1-4) with ``dispatch.outcome`` riding beside it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import random
import re
from typing import Any, Literal

from agent_workflow_contracts import CatalogToolEntry

from .capability_resolution import (
  canonical_dispatch_tool_name,
  lookup_catalog_entry,
)
from .tool_dispatch_source_identity import read_source_identities
from .tool_result_semantics import (
  classify_semantic_tool_error,
  is_semantic_tool_error,
  status_error_has_detail,
)


DispatchOutcome = Literal[
  "ok",
  "cancelled",
  "error_timeout",
  "error_rate_limited",
  "error_transport",
  "error_semantic",
]

OUTCOME_OK = "ok"
OUTCOME_CANCELLED = "cancelled"
OUTCOME_TIMEOUT = "error_timeout"
OUTCOME_RATE_LIMITED = "error_rate_limited"
OUTCOME_TRANSPORT = "error_transport"
OUTCOME_SEMANTIC = "error_semantic"

ALL_DISPATCH_OUTCOMES: frozenset[str] = frozenset(
  {
    OUTCOME_OK,
    OUTCOME_CANCELLED,
    OUTCOME_TIMEOUT,
    OUTCOME_RATE_LIMITED,
    OUTCOME_TRANSPORT,
    OUTCOME_SEMANTIC,
  }
)

# Only these outcomes are ever retried; everything else is a settled answer.
RETRYABLE_OUTCOMES: frozenset[str] = frozenset(
  {OUTCOME_TRANSPORT, OUTCOME_TIMEOUT, OUTCOME_RATE_LIMITED}
)

_RATE_LIMIT_ERROR_CODES = frozenset({"rate_limited", "broker_rate_limited"})
_TIMEOUT_ERROR_CODES = frozenset({"tool_timeout", "timeout"})
_CANCELLED_ERROR_CODES = frozenset({"cancelled"})
_TRANSPORT_ERROR_CODES = frozenset(
  {
    "internal_error",
    "transport_error",
    "mcp_transport_error",
    "connection_error",
    "server_unavailable",
    "sheets_unavailable",
  }
)
# Structural refusals: the call was decided, not attempted.  Never retried.
_POLICY_ERROR_CODES = frozenset(
  {
    "tool_excluded",
    "mcp_tool_not_allowed",
    "role_policy_denied",
    "tool_not_advertised",
    "invalid_tool_input_schema",
    "invalid_input",
  }
)
_POLICY_ERROR_CODE_PREFIXES = ("planned_write_",)

_RATE_LIMIT_TEXT_RE = re.compile(
  r"\b(?:429|too many requests|rate[ _-]?limit(?:ed|ing)?|overload(?:ed)?)\b",
  re.IGNORECASE,
)


_CLASSIFICATION_REGEXES: tuple[Any, ...] | None = None
_STATUS_CODE_REGEX: Any = None


def _classification_regexes() -> tuple[tuple[Any, ...], Any]:
  """Reuse ``retry.py``'s classification regexes (never its run machinery).

  Imported lazily: ``agent_gateway.retry`` pulls in the autonomous run
  surface, which imports the runner this module is dispatched from.
  """

  global _CLASSIFICATION_REGEXES, _STATUS_CODE_REGEX
  if _CLASSIFICATION_REGEXES is None:
    from .retry import _STATUS_CODE_RE, _TRANSIENT_ERROR_PATTERNS

    _CLASSIFICATION_REGEXES = tuple(_TRANSIENT_ERROR_PATTERNS)
    _STATUS_CODE_REGEX = _STATUS_CODE_RE
  return _CLASSIFICATION_REGEXES, _STATUS_CODE_REGEX


@dataclass(frozen=True, slots=True)
class DispatchEntry:
  """Everything the boundary knows about one tool before it dispatches.

  ``catalog_entry`` is the platform catalog's description of this tool
  (D-B1-2's end state): the declaration table is construction-only input to
  ``snapshot_platform_catalog`` and has no dispatch-time reader.
  """

  tool_name: str
  canonical_name: str
  route_id: str
  catalog_entry: CatalogToolEntry | None = None

  @property
  def effect(self) -> str | None:
    return self.catalog_entry.effect if self.catalog_entry is not None else None

  @property
  def idempotent(self) -> bool | None:
    return self.catalog_entry.idempotent if self.catalog_entry is not None else None


def build_route_id(
  *,
  tool_name: str,
  server: str | None = None,
  provider_id: str | None = None,
) -> str:
  """Name the route a call actually took, from facts already local."""

  segments: list[str] = []
  server_text = str(server or "").strip()
  segments.append(f"mcp:{server_text}" if server_text else "local")
  provider_text = str(provider_id or "").strip()
  if provider_text:
    segments.append(f"provider:{provider_text}")
  segments.append(str(tool_name or ""))
  return "/".join(segments)


def resolve_dispatch_entry(
  tool_name: str,
  *,
  server: str | None = None,
  provider_id: str | None = None,
) -> DispatchEntry:
  return DispatchEntry(
    tool_name=str(tool_name or ""),
    canonical_name=canonical_dispatch_tool_name(tool_name),
    route_id=build_route_id(
      tool_name=tool_name,
      server=server,
      provider_id=provider_id,
    ),
    catalog_entry=lookup_catalog_entry(tool_name),
  )


def _error_text(payload: Mapping[str, Any] | None) -> str:
  if not isinstance(payload, Mapping):
    return ""
  parts: list[str] = []
  for key in ("code", "sub_code", "message", "detail", "reason", "status"):
    value = payload.get(key)
    if value is not None and str(value).strip():
      parts.append(str(value).strip())
  return " ".join(parts)


def _is_rate_limited(code: str, sub_code: str, text: str) -> bool:
  if code in _RATE_LIMIT_ERROR_CODES or sub_code in _RATE_LIMIT_ERROR_CODES:
    return True
  return bool(_RATE_LIMIT_TEXT_RE.search(text))


def _is_transport(code: str, sub_code: str, text: str) -> bool:
  if code in _TRANSPORT_ERROR_CODES or sub_code in _TRANSPORT_ERROR_CODES:
    return True
  transient_patterns, status_code_re = _classification_regexes()
  match = status_code_re.search(text)
  if match is not None and match.group(1).startswith("5"):
    return True
  return any(pattern.search(text) for pattern in transient_patterns)


def _is_policy_refusal(code: str) -> bool:
  if code in _POLICY_ERROR_CODES:
    return True
  return any(code.startswith(prefix) for prefix in _POLICY_ERROR_CODE_PREFIXES)


def _classify_failure_payload(payload: Mapping[str, Any] | None) -> str:
  """Normalize one failure payload (dispatcher error or semantic error)."""

  code = str((payload or {}).get("code") or "").strip().lower()
  sub_code = str((payload or {}).get("sub_code") or "").strip().lower()
  text = _error_text(payload)
  if code in _CANCELLED_ERROR_CODES:
    return OUTCOME_CANCELLED
  if code in _TIMEOUT_ERROR_CODES or sub_code in _TIMEOUT_ERROR_CODES:
    return OUTCOME_TIMEOUT
  if _is_rate_limited(code, sub_code, text):
    return OUTCOME_RATE_LIMITED
  if _is_policy_refusal(code):
    return OUTCOME_SEMANTIC
  if _is_transport(code, sub_code, text):
    return OUTCOME_TRANSPORT
  return OUTCOME_SEMANTIC


def _success_signal_satisfied(
  signal: Mapping[str, Any] | None,
  result: Any,
) -> bool | None:
  """Evaluate a declared success signal.

  ``None`` means the signal is inapplicable — no declaration, or a payload
  shape the signal cannot speak about — which keeps today's behavior
  (D-B1-5).
  """

  if signal is None:
    return None
  if not isinstance(result, Mapping):
    return None
  kind = str(signal.get("kind") or "")
  if kind != "status_equals":
    return None
  field = str(signal.get("field") or "status")
  values = tuple(str(value) for value in (signal.get("values") or ()))
  raw = result.get(field)
  if raw is None:
    return False
  return str(raw).strip().lower() in {value.strip().lower() for value in values}


def classify_tool_outcome(
  entry: DispatchEntry | None,
  result: Any,
  error: Mapping[str, Any] | None,
  semantic_error: Mapping[str, Any] | None = None,
) -> str:
  """Settle one normalized outcome for a completed tool dispatch.

  Total over the three status dialects: the dispatcher's ``error`` dict, the
  result-borne semantic failure, and the tool's declared success signal.
  """

  if isinstance(error, Mapping):
    return _classify_failure_payload(error)

  failure = semantic_error
  if failure is None:
    failure = classify_semantic_tool_error(result)
  if isinstance(failure, Mapping):
    return _classify_failure_payload(failure)

  described = entry.catalog_entry if entry is not None else None
  signal = described.success_signal if described is not None else None
  satisfied = _success_signal_satisfied(signal, result)
  if satisfied is False:
    return OUTCOME_SEMANTIC
  return OUTCOME_OK


def extract_source_identities(
  entry: DispatchEntry | None,
  result: Any,
) -> tuple[Mapping[str, Any], ...]:
  """Extract the source identities a successful payload carries.

  ``build_dispatch_record`` only reaches this on the ``ok`` arm — that
  sequencing is what closes the 429-minting hole. The declared success signal
  is re-checked here so the helper is honest standalone too: a failed payload
  yields no identities no matter who asks.
  """

  described = entry.catalog_entry if entry is not None else None
  if described is None:
    return ()
  if _success_signal_satisfied(described.success_signal, result) is False:
    return ()
  return read_source_identities(described.source_identity, result)


def build_dispatch_record(
  *,
  entry: DispatchEntry | None,
  result: Any,
  error: Mapping[str, Any] | None,
  semantic_error: Mapping[str, Any] | None = None,
  attempts: int = 1,
  retries_exhausted: bool = False,
) -> dict[str, Any]:
  """Build the unconditional ``dispatch`` block for one tool call.

  ``sources`` is plural and stays empty unless the outcome is ``ok`` — the
  sequencing that closes the 429-minting hole.
  """

  outcome = classify_tool_outcome(entry, result, error, semantic_error)
  sources: tuple[Mapping[str, Any], ...] = ()
  if outcome == OUTCOME_OK:
    sources = extract_source_identities(entry, result)
  record: dict[str, Any] = {
    "outcome": outcome,
    "attempts": int(attempts),
    "route_id": entry.route_id if entry is not None else "",
    "sources": [dict(source) for source in sources],
  }
  if retries_exhausted:
    record["retries_exhausted"] = True
  return record


# --------------------------------------------------------------------------
# B-2 retry
# --------------------------------------------------------------------------


DEFAULT_MAX_TOOL_RETRIES = 2
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_MAX_SECONDS = 8.0


@dataclass(frozen=True, slots=True)
class RetryPolicy:
  """Bounded, jittered retry at the dispatch site.

  Not ``retry.RetryConfig``: that governs whole autonomous runs with 30–600 s
  backoff steps.  Only the classification regexes are shared.
  """

  max_retries: int = DEFAULT_MAX_TOOL_RETRIES
  base_delay_seconds: float = _BACKOFF_BASE_SECONDS
  max_delay_seconds: float = _BACKOFF_MAX_SECONDS
  # Retry never re-enters an approval round-trip.
  allow_when_approval_required: bool = False


DEFAULT_TOOL_RETRY_POLICY = RetryPolicy()


def retry_eligible(entry: DispatchEntry | None) -> bool:
  """Reads retry by default; writes never, regardless of annotation."""

  if entry is None or entry.catalog_entry is None:
    return False
  if entry.effect != "read":
    return False
  return entry.idempotent is not False


def retry_decision(
  entry: DispatchEntry | None,
  outcome: str,
  attempt: int,
  policy: RetryPolicy = DEFAULT_TOOL_RETRY_POLICY,
  *,
  needs_approval: bool = False,
  aborted: bool = False,
  wall_clock_exhausted: bool = False,
) -> str:
  """Return ``"retry"`` or ``"settle"`` for one completed attempt.

  ``attempt`` is 1-based: the first dispatch is attempt 1, so at most
  ``policy.max_retries`` further attempts follow.
  """

  if outcome not in RETRYABLE_OUTCOMES:
    return "settle"
  if aborted or wall_clock_exhausted:
    return "settle"
  if needs_approval and not policy.allow_when_approval_required:
    return "settle"
  if not retry_eligible(entry):
    return "settle"
  if attempt >= 1 + max(0, int(policy.max_retries)):
    return "settle"
  return "retry"


def retry_backoff_seconds(
  attempt: int,
  policy: RetryPolicy = DEFAULT_TOOL_RETRY_POLICY,
  *,
  rng: random.Random | None = None,
) -> float:
  """Jittered exponential backoff for the completed 1-based ``attempt``."""

  exponent = max(0, int(attempt) - 1)
  ceiling = min(
    policy.max_delay_seconds,
    policy.base_delay_seconds * (2**exponent),
  )
  source = rng if rng is not None else random
  return round(source.uniform(0.0, max(0.0, ceiling)), 4)


__all__ = [
  "ALL_DISPATCH_OUTCOMES",
  "DEFAULT_MAX_TOOL_RETRIES",
  "DEFAULT_TOOL_RETRY_POLICY",
  "DispatchEntry",
  "DispatchOutcome",
  "OUTCOME_CANCELLED",
  "OUTCOME_OK",
  "OUTCOME_RATE_LIMITED",
  "OUTCOME_SEMANTIC",
  "OUTCOME_TIMEOUT",
  "OUTCOME_TRANSPORT",
  "RETRYABLE_OUTCOMES",
  "RetryPolicy",
  "build_dispatch_record",
  "build_route_id",
  "classify_semantic_tool_error",
  "classify_tool_outcome",
  "extract_source_identities",
  "is_semantic_tool_error",
  "resolve_dispatch_entry",
  "retry_backoff_seconds",
  "retry_decision",
  "retry_eligible",
  "status_error_has_detail",
]
