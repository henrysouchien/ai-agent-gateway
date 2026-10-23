"""Machine-wide spend ledger metering for upstream LLM provider calls.

Every upstream provider call the gateway starts writes exactly one
``api_spend_events`` row at call completion through
``model_authority.spend_ledger.record_api_spend``: success, provider error with
or without usage, cancellation (client disconnect, watchdog stall) and an
Agent SDK query. ``ProviderCallMeter`` is the one owner of that row; the three
paths that start a provider stream (``runner_stream_turn``, ``send_prompt``,
``provider_summarize``) wrap ``provider.stream`` with ``ProviderCallMeter.stream``
and the Agent SDK runner calls ``record`` once per query.

Token counts reuse ``runner_usage``'s normalization (``input_tokens`` is
uncached input, cache read/write and reasoning are separate) and the shadow cost
is the provider adapter's own ``estimate_cost`` for that call: ``None`` when the
adapter has no rate for the model, never a second price. What kind of money the
call is (``payg_api_key``, ``oauth_subscription``, ``plan_api_key``,
``local_or_free``) comes from ``model_authority.spend_ledger.classify_auth_kind``
over the credential's wire mechanism, never from the commercial ``billing_mode``.

A PAYG call is checked against the shared ceilings before it starts
(``ProviderCallMeter.enforce``); a refusal writes a zero-token ``budget_blocked``
row and raises ``SpendBudgetBlocked``, which no retry, rotation or fallback path
catches. OAuth and local credentials are never checked. Telemetry never blocks
or fails the call it records.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncGenerator, Mapping

from model_authority.spend_ledger import (
  PAYG,
  AuthKind,
  SpendLedgerUnavailable,
  SpendStatus,
  classify_auth_kind,
  fingerprint_credential,
  record_api_spend,
  resolve_attribution,
)
from model_authority.spend_settings import (
  SpendBreach,
  SpendCeilingExceeded,
  SpendSettingsError,
  enforce_spend_ceilings,
)

from .auth import derive_auth_mechanism_and_entitlement
from .providers.base import _status_code_from_exception
from .runner_usage import (
  apply_message_start_usage,
  apply_usage_update,
  empty_usage_totals,
  estimate_usage_cost,
)

if TYPE_CHECKING:
  from .providers.base import ModelProvider, StreamEvent

log = logging.getLogger("agent_gateway.spend_metering")

_SPEND_COMMAND = "python3 scripts/hank_spend.py"


class SpendBudgetBlocked(RuntimeError):
  """A PAYG call was refused before it started: a spend ceiling or the ledger said stop.

  Not a provider failure: the runner's retry, credential rotation and model
  fallback paths never see it. ``breaches`` names every breached ceiling; it is
  empty when the ledger or settings could not be read (a PAYG call never runs
  on an unreadable ledger).
  """

  def __init__(self, message: str, *, breaches: tuple[SpendBreach, ...] = ()) -> None:
    super().__init__(message)
    self.breaches = breaches


@dataclass(frozen=True)
class SpendCredential:
  auth_kind: AuthKind
  fingerprint: str | None


def spend_credential(provider_name: str, config: Mapping[str, Any]) -> SpendCredential:
  """The credential kind and fingerprint of the material ``config`` carries."""
  mechanism, _ = derive_auth_mechanism_and_entitlement({**config, "provider": provider_name})
  api_key = str(config.get("api_key") or "").strip()
  auth_token = str(config.get("auth_token") or "").strip()
  secret = (auth_token or api_key) if mechanism == "oauth_bearer" else (api_key or auth_token)
  if mechanism == "none":
    secret = ""
  return SpendCredential(
    auth_kind=classify_auth_kind(provider_name, mechanism),
    fingerprint=fingerprint_credential(secret) if secret else None,
  )


def _describe_breach(breach: SpendBreach) -> str:
  if breach.ceiling == "freeze_payg":
    return "freeze_payg is on (every PAYG call is stopped)"
  scope = f" [{breach.scope}]" if breach.scope else ""
  return f"{breach.ceiling}{scope}: ${breach.used:.2f} spent today of ${breach.limit:.2f}"


def _blocked_message(breaches: tuple[SpendBreach, ...], provider: str, model: str) -> str:
  named = "; ".join(_describe_breach(breach) for breach in breaches)
  return (
    f"Hank spend ceiling refused this {provider}/{model} call before it started "
    f"({named}). Adjust with `{_SPEND_COMMAND} set <setting> <value>` "
    f"(for example `set freeze_payg false`); `{_SPEND_COMMAND} settings` lists every ceiling."
  )


def _failure_status(provider: ModelProvider | None, exc: BaseException) -> tuple[SpendStatus, int | None, bool]:
  """(ledger status, http status, whether the provider says the request reached upstream)."""
  failure = None
  if provider is not None and isinstance(exc, Exception):
    try:
      failure = provider.classify_credential_failure(exc)
    except Exception:
      failure = None
  http_status = failure.status_code if failure is not None else None
  if http_status is None and isinstance(exc, Exception):
    http_status = _status_code_from_exception(exc)
  retryable = False
  if provider is not None and isinstance(exc, Exception):
    try:
      retryable = bool(provider.is_retryable_error(exc))
    except Exception:
      retryable = False
  rate_limited = (failure is not None and failure.kind == "rate_limit") or http_status == 429
  reached_upstream = failure is not None or http_status is not None or retryable
  return ("rate_limit" if rate_limited else "error"), http_status, reached_upstream


class ProviderCallMeter:
  """Writes the one ledger row for one upstream provider call.

  Build one per call attempt (a retry or a credential rotation is a new
  attempt with its own meter) from the credential material and model actually
  used. ``provider`` is the adapter that prices and classifies the call; the
  Agent SDK path has none and passes its own ``shadow_cost_usd`` to ``record``.
  """

  def __init__(
    self,
    *,
    provider: ModelProvider | None,
    provider_name: str,
    model: str,
    model_key: str | None,
    config: Mapping[str, Any],
    call_site: str,
    session_id: str | None = None,
    request_id: str | None = None,
    run_id: str | None = None,
    stage_id: str | None = None,
    skill: str | None = None,
    attrs: Mapping[str, object] | None = None,
  ) -> None:
    self._provider = provider
    self.provider_name = provider_name
    self.model = model
    self._model_key = model_key
    # An enclosing ``spend_context(call_site=...)`` names the real caller (an eval
    # jury, an investment tool) and wins over this path's own default label.
    self._call_site = resolve_attribution("call_site") or call_site
    self._attribution = {
      key: value
      for key, value in (
        ("session_id", session_id),
        ("request_id", request_id),
        ("run_id", run_id),
        ("stage_id", stage_id),
        ("skill", skill),
      )
      if value
    }
    self._attrs = dict(attrs or {})
    self.credential = spend_credential(provider_name, config)

  def _credential_fields(self) -> dict[str, Any]:
    fields: dict[str, Any] = {"auth_kind": self.credential.auth_kind}
    if self.credential.fingerprint is not None:
      fields["credential_fingerprint"] = self.credential.fingerprint
    return fields

  def enforce(self) -> None:
    """Refuse a PAYG call that breaches a ceiling, before it starts.

    Raises ``SpendBudgetBlocked`` after writing a zero-token ``budget_blocked``
    row. A ledger or settings file that cannot be read refuses the call too: a
    PAYG caller never treats an unreadable ledger as unlimited.
    """
    if self.credential.auth_kind != PAYG:
      return
    breaches: tuple[SpendBreach, ...] = ()
    try:
      enforce_spend_ceilings(
        provider=self.provider_name,
        run_id=self._attribution.get("run_id"),
        **self._credential_fields(),
      )
      return
    except SpendCeilingExceeded as exc:
      breaches = tuple(exc.breaches)
      message = _blocked_message(breaches, self.provider_name, self.model)
      error: BaseException = exc
    except (SpendLedgerUnavailable, SpendSettingsError) as exc:
      message = (
        f"Hank spend ledger refused this {self.provider_name}/{self.model} call before it "
        f"started: {exc}. PAYG calls do not run while the spend ledger or settings cannot be "
        f"read; inspect with `{_SPEND_COMMAND} settings`."
      )
      error = exc
    self._record(
      tokens={},
      status="budget_blocked",
      error=error,
      api_calls=0,
      attrs={"breaches": [_breach_json(breach) for breach in breaches]},
    )
    raise SpendBudgetBlocked(message, breaches=breaches) from error

  def record(
    self,
    *,
    tokens: Mapping[str, int],
    status: SpendStatus = "ok",
    error: BaseException | None = None,
    duration_ms: float | None = None,
    shadow_cost_usd: float | None,
    api_calls: int = 1,
    http_status: int | None = None,
    attrs: Mapping[str, object] | None = None,
  ) -> None:
    """Write the row for one completed call; never raises."""
    self._record(
      tokens=tokens,
      status=status,
      error=error,
      duration_ms=duration_ms,
      shadow_cost_usd=shadow_cost_usd,
      api_calls=api_calls,
      http_status=http_status,
      attrs=attrs,
    )

  def _record(
    self,
    *,
    tokens: Mapping[str, int],
    status: SpendStatus,
    error: BaseException | None = None,
    duration_ms: float | None = None,
    shadow_cost_usd: float | None = 0.0,
    api_calls: int = 1,
    http_status: int | None = None,
    attrs: Mapping[str, object] | None = None,
  ) -> None:
    try:
      row_attrs = {**self._attrs, **(attrs or {})}
      if isinstance(error, (asyncio.CancelledError, GeneratorExit)):
        row_attrs["canceled"] = True
      fields: dict[str, Any] = {
        **self._attribution,
        **self._credential_fields(),
      }
      if http_status is not None:
        fields["http_status"] = http_status
      record_api_spend(
        api_category="llm",
        provider=self.provider_name,
        model_or_endpoint=self.model,
        model_key=self._model_key,
        call_site=self._call_site,
        input_tokens=int(tokens.get("input_tokens", 0)),
        cache_write_tokens=int(tokens.get("cache_write_tokens", 0)),
        cache_read_tokens=int(tokens.get("cache_read_tokens", 0)),
        output_tokens=int(tokens.get("output_tokens", 0)),
        reasoning_tokens=int(tokens.get("reasoning_tokens", 0)),
        api_calls=api_calls,
        duration_ms=duration_ms,
        status=status,
        error_type=None if error is None else type(error).__name__,
        shadow_cost_usd=shadow_cost_usd,
        attrs=row_attrs or None,
        **fields,
      )
    except Exception as exc:  # telemetry never fails the call it records
      log.warning("spend ledger row not written (%s: %s)", type(exc).__name__, exc)

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
    """``provider.stream`` plus exactly one ledger row when the stream ends.

    The row is written when the generator is finished or closed: normal end,
    provider error (with or without usage), cancellation and watchdog stall.
    A failure before any request could have been sent (no event, no usage and
    nothing a provider would call an upstream error) writes nothing.
    """
    provider = self._provider
    if provider is None:
      raise TypeError("ProviderCallMeter.stream requires a provider adapter")
    totals = empty_usage_totals()
    started = time.monotonic()
    saw_event = False
    completed = False
    failure: BaseException | None = None
    reported_model: str | None = None
    try:
      async for event in provider.stream(client, params):
        saw_event = True
        event_type = event.type
        if event_type == "message_start":
          apply_message_start_usage(
            totals,
            input_tokens=event.input_tokens,
            cache_creation_tokens=event.cache_creation_tokens,
            cache_read_tokens=event.cache_read_tokens,
            provider_units=event.provider_units,
            provider_unit_deltas=event.provider_unit_deltas,
            provider_wait_ms=event.provider_wait_ms,
          )
          if event.provider_reported_model is not None:
            reported_model = event.provider_reported_model
        elif event_type == "usage_update":
          apply_usage_update(
            totals,
            input_tokens=event.input_tokens,
            output_tokens=event.output_tokens,
            reasoning_tokens=event.reasoning_tokens,
            cache_creation_tokens=event.cache_creation_tokens,
            cache_read_tokens=event.cache_read_tokens,
            provider_units=event.provider_units,
            provider_unit_deltas=event.provider_unit_deltas,
          )
        elif event_type == "message_end":
          completed = True
        yield event
      completed = True
    except BaseException as exc:
      failure = exc
      raise
    finally:
      self._record_stream(
        totals,
        started=started,
        saw_event=saw_event,
        failure=None if completed else failure,
        reported_model=reported_model,
      )

  def _record_stream(
    self,
    totals: dict[str, Any],
    *,
    started: float,
    saw_event: bool,
    failure: BaseException | None,
    reported_model: str | None,
  ) -> None:
    try:
      tokens = {
        "input_tokens": int(totals["input_tokens"]),
        "cache_write_tokens": int(totals["cache_creation_input_tokens"]),
        "cache_read_tokens": int(totals["cache_read_input_tokens"]),
        "output_tokens": int(totals["output_tokens"]),
        "reasoning_tokens": int(totals.get("reasoning_tokens_observed", 0) or 0),
      }
      has_tokens = any(tokens.values())
      status: SpendStatus = "ok"
      http_status: int | None = None
      if failure is not None:
        status, http_status, reached_upstream = _failure_status(self._provider, failure)
        interrupted = isinstance(failure, (asyncio.CancelledError, GeneratorExit))
        if not (has_tokens or saw_event or interrupted or reached_upstream):
          # Local failure before a request could have been sent: nothing was spent.
          return
      attrs: dict[str, object] = {}
      units = totals.get("provider_unit_deltas")
      if units:
        attrs["provider_unit_deltas"] = dict(units)
      if reported_model:
        attrs["provider_reported_model"] = reported_model
      self._record(
        tokens=tokens,
        status=status,
        error=failure,
        duration_ms=(time.monotonic() - started) * 1000.0,
        shadow_cost_usd=self._shadow_cost(totals, has_tokens),
        http_status=http_status,
        attrs=attrs,
      )
    except Exception as exc:
      log.warning("spend ledger row not written (%s: %s)", type(exc).__name__, exc)

  def _shadow_cost(self, totals: dict[str, Any], has_tokens: bool) -> float | None:
    """The adapter's own estimate for this call; ``None`` when it has no rate."""
    if not has_tokens:
      return 0.0
    try:
      cost = estimate_usage_cost(self._provider, self.model, totals)
    except Exception:
      return None
    if not getattr(cost, "cost_known", True):
      return None
    return float(cost.total)


def _breach_json(breach: SpendBreach) -> dict[str, object]:
  return {
    "ceiling": breach.ceiling,
    "scope": breach.scope,
    "limit": breach.limit,
    "used": breach.used,
  }


__all__ = [
  "ProviderCallMeter",
  "SpendBudgetBlocked",
  "SpendCredential",
  "spend_credential",
]
