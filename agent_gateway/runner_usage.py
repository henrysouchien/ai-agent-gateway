from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Callable, Dict

from .multi_user.billing import SessionUsageSummary, UsageEvent, write_dlq
from .commercial_usage import CommercialUsageProducer

USAGE_TOKEN_KEYS = (
  "input_tokens",
  "output_tokens",
  "reasoning_tokens_observed",
  "provider_units",
  "cache_read_input_tokens",
  "cache_creation_input_tokens",
)


@dataclass(frozen=True)
class UsageDeltaState:
  usage: Dict[str, Any]
  has_tokens: bool


def empty_usage_totals() -> Dict[str, Any]:
  return {
    "input_tokens": 0,
    "output_tokens": 0,
    "reasoning_tokens_observed": 0,
    "provider_units": 0,
    "provider_unit_deltas": {},
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
  }


def usage_snapshot(usage_totals: Dict[str, Any]) -> Dict[str, Any]:
  """Detach usage state from the mutable aggregate, including named units."""
  snapshot = dict(usage_totals)
  if "provider_unit_deltas" in snapshot:
    snapshot["provider_unit_deltas"] = dict(
      usage_totals.get("provider_unit_deltas") or {}
    )
  return snapshot


def usage_has_tokens(usage_totals: Dict[str, Any]) -> bool:
  return any(int(usage_totals.get(key, 0) or 0) > 0 for key in USAGE_TOKEN_KEYS) or any(
    int(value or 0) > 0
    for value in (usage_totals.get("provider_unit_deltas") or {}).values()
  )


def usage_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
  delta = {
    key: max(0, int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0))
    for key in USAGE_TOKEN_KEYS
  }
  before_units = before.get("provider_unit_deltas") or {}
  after_units = after.get("provider_unit_deltas") or {}
  delta["provider_unit_deltas"] = {
    key: max(0, int(value or 0) - int(before_units.get(key, 0) or 0))
    for key, value in after_units.items()
    if int(value or 0) > int(before_units.get(key, 0) or 0)
  }
  for key in (
    "capability_bind",
    "provider_reported_model",
  ):
    value = after.get(key)
    if value is not None:
      delta[key] = value
  return delta


def usage_delta_state(before: Dict[str, Any], after: Dict[str, Any]) -> UsageDeltaState:
  delta = usage_delta(before, after)
  return UsageDeltaState(
    usage=delta,
    has_tokens=usage_has_tokens(delta),
  )


def record_compaction(aggregator: Any) -> None:
  """Record one live compaction, rejecting events after summary closure."""
  if not aggregator.record_compaction_nowait():
    raise RuntimeError(
      "session usage aggregator closed before compaction was recorded"
    )


def apply_message_start_usage(
  usage_totals: Dict[str, int],
  *,
  input_tokens: int,
  cache_creation_tokens: int,
  cache_read_tokens: int,
  provider_units: int = 0,
  provider_unit_deltas: dict[str, int] | None = None,
) -> Dict[str, int]:
  usage_totals["input_tokens"] += input_tokens
  usage_totals["cache_creation_input_tokens"] += cache_creation_tokens
  usage_totals["cache_read_input_tokens"] += cache_read_tokens
  usage_totals["provider_units"] = int(usage_totals.get("provider_units", 0) or 0) + provider_units
  accumulated_units = usage_totals.setdefault("provider_unit_deltas", {})
  for operation, count in (provider_unit_deltas or {}).items():
    accumulated_units[operation] = int(accumulated_units.get(operation, 0) or 0) + int(count)
  return usage_totals


def apply_usage_update(
  usage_totals: Dict[str, int],
  *,
  input_tokens: int = 0,
  output_tokens: int,
  reasoning_tokens: int = 0,
  cache_creation_tokens: int = 0,
  cache_read_tokens: int = 0,
  provider_units: int = 0,
  provider_unit_deltas: dict[str, int] | None = None,
) -> Dict[str, int]:
  usage_totals["input_tokens"] += input_tokens
  usage_totals["cache_creation_input_tokens"] += cache_creation_tokens
  usage_totals["cache_read_input_tokens"] += cache_read_tokens
  usage_totals["output_tokens"] += output_tokens
  usage_totals["reasoning_tokens_observed"] = (
    int(usage_totals.get("reasoning_tokens_observed", 0) or 0) + reasoning_tokens
  )
  usage_totals["provider_units"] = int(usage_totals.get("provider_units", 0) or 0) + provider_units
  accumulated_units = usage_totals.setdefault("provider_unit_deltas", {})
  for operation, count in (provider_unit_deltas or {}).items():
    accumulated_units[operation] = int(accumulated_units.get(operation, 0) or 0) + int(count)
  return usage_totals


def turn_usage_payload(
  usage_totals: Dict[str, Any],
  *,
  estimated_cost: float | None = None,
) -> Dict[str, Any]:
  payload = usage_snapshot(usage_totals)
  if estimated_cost is not None:
    payload["estimated_cost"] = round(estimated_cost, 4)
  return payload


def estimate_usage_cost(provider: Any, model: str, usage_totals: Dict[str, Any]) -> Any:
  uncached_input = max(0, usage_totals["input_tokens"])
  return provider.estimate_cost(
    model,
    uncached_input,
    usage_totals["output_tokens"],
    cache_read_tokens=usage_totals["cache_read_input_tokens"],
    cache_creation_tokens=usage_totals["cache_creation_input_tokens"],
  )


def build_usage_event(
  *,
  user_id: str,
  session_id: str,
  request_id: str | None,
  parent_turn_id: str | None,
  timestamp: float,
  model: str,
  provider_name: str | None,
  usage_totals: Dict[str, Any],
  cost_total: float,
  rate_table_version: str,
  billing_mode: str,
  channel: str | None,
) -> UsageEvent:
  bind_receipt = usage_totals.get("capability_bind")
  if not isinstance(bind_receipt, dict):
    raise ValueError("provider-call usage requires a capability bind receipt")
  if provider_name is None:
    raise ValueError("provider-call usage requires a provider projection")
  return UsageEvent(
    user_id=user_id,
    session_id=session_id,
    request_id=request_id,
    parent_turn_id=parent_turn_id,
    timestamp=timestamp,
    model=model,
    provider=provider_name,
    capability_bind=dict(bind_receipt),
    provider_reported_model=(
      str(usage_totals["provider_reported_model"])
      if usage_totals.get("provider_reported_model") is not None else None
    ),
    input_tokens=int(usage_totals["input_tokens"]),
    output_tokens=int(usage_totals["output_tokens"]),
    reasoning_tokens_observed=int(usage_totals.get("reasoning_tokens_observed", 0) or 0),
    provider_units=int(usage_totals.get("provider_units", 0) or 0) or None,
    provider_unit_deltas=dict(usage_totals.get("provider_unit_deltas") or {}) or None,
    cache_read_tokens=int(usage_totals["cache_read_input_tokens"]),
    cache_creation_tokens=int(usage_totals["cache_creation_input_tokens"]),
    cost_usd=float(cost_total),
    rate_table_version=rate_table_version,
    billing_mode=billing_mode,
    channel=channel,
  )


async def call_late_usage_event_hook(
  on_late_usage_event: Callable[[UsageEvent], Any] | None,
  usage_event: UsageEvent,
  *,
  log_session_id: str,
  logger: Any,
) -> None:
  if on_late_usage_event is None:
    return
  try:
    result = on_late_usage_event(usage_event)
    if inspect.isawaitable(result):
      await result
  except Exception as exc:
    logger.error(
      "[%s] on_late_usage_event hook failed (non-fatal) | exception_type=%s",
      log_session_id,
      type(exc).__name__,
    )


async def call_session_summary_hook(
  on_session_summary: Callable[[SessionUsageSummary], Any] | None,
  summary: SessionUsageSummary,
  *,
  log_session_id: str,
  logger: Any,
  commercial_usage_producer: CommercialUsageProducer | None = None,
  emit_metric: Callable[[str, int], None] | None = None,
) -> None:
  if commercial_usage_producer is not None:
    reconcile = getattr(commercial_usage_producer, "reconcile", None)
    if callable(reconcile):
      try:
        await reconcile(summary)
      except Exception as exc:
        logger.error(
          "[%s] commercial usage reconciliation failed | exception_type=%s",
          log_session_id,
          type(exc).__name__,
        )
        if emit_metric is not None:
          emit_metric("gateway.commercial_usage_reconciliation_error", 1)
  if on_session_summary is None:
    return
  try:
    result = on_session_summary(summary)
    if inspect.isawaitable(result):
      await result
  except Exception as exc:
    logger.error(
      "[%s] on_session_summary hook failed (non-fatal) | exception_type=%s",
      log_session_id,
      type(exc).__name__,
    )


async def call_usage_event_hook(
  aggregator: Any,
  usage_event: UsageEvent,
  *,
  is_summary_emitted: Callable[[], bool],
  on_usage: Callable[[UsageEvent], Any] | None,
  on_late_usage_event: Callable[[UsageEvent], Any] | None,
  emit_metric: Callable[[str, int], None],
  dlq_path: Path | None,
  log_session_id: str,
  logger: Any,
  commercial_usage_producer: CommercialUsageProducer | None = None,
  usage_state: str = "succeeded",
) -> None:
  if commercial_usage_producer is not None:
    await commercial_usage_producer.emit(usage_event, usage_state=usage_state)
  recorded = await aggregator.record(usage_event)
  if not recorded or is_summary_emitted():
    if commercial_usage_producer is not None:
      mark_late = getattr(commercial_usage_producer, "mark_late", None)
      if callable(mark_late):
        try:
          late_result = mark_late(usage_event.event_id)
          if inspect.isawaitable(late_result):
            await late_result
        except Exception as exc:
          logger.error(
            "[%s] late commercial reconciliation failed | exception_type=%s",
            log_session_id,
            type(exc).__name__,
          )
          emit_metric("gateway.commercial_usage_reconciliation_error", 1)
    logger.warning("[%s] Usage event arrived after session summary emission: %s", log_session_id, usage_event.event_id)
    await call_late_usage_event_hook(
      on_late_usage_event,
      usage_event,
      log_session_id=log_session_id,
      logger=logger,
    )
    return
  if on_usage is None:
    return
  try:
    result = on_usage(usage_event)
    if inspect.isawaitable(result):
      await result
  except Exception as exc:
    logger.error(
      "[%s] on_usage hook failed (non-fatal) | exception_type=%s",
      log_session_id,
      type(exc).__name__,
    )
    emit_metric("gateway.usage_event_dropped", 1)
    if dlq_path is not None:
      try:
        write_dlq(usage_event, dlq_path)
      except Exception as dlq_exc:
        logger.error(
          "[%s] usage DLQ write failed (non-fatal) | exception_type=%s",
          log_session_id,
          type(dlq_exc).__name__,
        )
