"""Default-off production of canonical commercial provider-call usage events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import inspect
import json
import re
from typing import Any, Awaitable, Callable, Literal
from uuid import NAMESPACE_URL, uuid5

from jsonschema import Draft202012Validator

from .commercial_claims import VerifiedCommercialClaim
from .commercial_contract import (
  canonical_usage_payload_sha256,
  packaged_contract_directory,
)
from .commercial_work_start import CommercialWorkStartContext
from .multi_user.billing import UsageEvent


_CODE = re.compile(r"^[a-z][a-z0-9._:-]{0,127}$")
CommercialUsageSink = Callable[[list[dict[str, Any]]], Awaitable[None] | None]


@dataclass(frozen=True)
class CommercialUsageLineage:
  source_product: str
  workflow_run_id: str
  funding_route_id: str
  operation: str
  reservation_id: str | None = None
  capability_id: str | None = None

  def __post_init__(self) -> None:
    for name in ("source_product",):
      value = getattr(self, name)
      if not isinstance(value, str) or not _CODE.fullmatch(value):
        raise ValueError(f"commercial usage {name} is invalid")
    for name in ("workflow_run_id", "funding_route_id", "operation"):
      value = getattr(self, name)
      if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError(f"commercial usage {name} is invalid")
    if self.reservation_id is not None and (
      not isinstance(self.reservation_id, str)
      or not self.reservation_id
      or len(self.reservation_id) > 512
    ):
      raise ValueError("commercial usage reservation_id is invalid")
    for name in ("capability_id",):
      value = getattr(self, name)
      if value is not None and (not isinstance(value, str) or not _CODE.fullmatch(value)):
        raise ValueError(f"commercial usage {name} is invalid")


class CommercialUsageProducer:
  """Convert provider-call deltas to V1 or verified-work-start Usage V2."""

  def __init__(
    self,
    *,
    enabled: bool,
    claim: VerifiedCommercialClaim | None,
    lineage: CommercialUsageLineage | None,
    sink: CommercialUsageSink | None,
    work_start: CommercialWorkStartContext | None = None,
    reconciliation_tracker: Any | None = None,
    on_reconciliation: Callable[[Any], Awaitable[None] | None] | None = None,
  ) -> None:
    self._enabled = enabled
    if work_start is not None:
      if not isinstance(work_start, CommercialWorkStartContext):
        raise TypeError("commercial Usage V2 requires verified work-start context")
      authorization = work_start.authorization
      derived_lineage = CommercialUsageLineage(
        source_product="hank-agent-gateway",
        workflow_run_id=str(authorization.workflow_run_id),
        funding_route_id=str(authorization.funding_route_id),
        operation=authorization.operation,
        reservation_id=str(authorization.reservation_id)
        if authorization.reservation_id is not None else None,
        capability_id=authorization.capability_id,
      )
      if claim is not None and claim != work_start.claim:
        raise ValueError("commercial usage claim differs from verified work start")
      if lineage is not None and lineage != derived_lineage:
        raise ValueError("commercial usage lineage differs from verified work start")
      claim = work_start.claim
      lineage = derived_lineage
    self._claim = claim
    self._lineage = lineage
    self._work_start = work_start
    self._sink = sink
    self._reconciliation_tracker = reconciliation_tracker
    self._on_reconciliation = on_reconciliation
    self._reconciliation_summary: Any | None = None
    schema_name = (
      "commercial-usage-event-v2.schema.json"
      if work_start is not None else "commercial-usage-event.schema.json"
    )
    schema = packaged_contract_directory() / schema_name
    self._validator = Draft202012Validator(json.loads(schema.read_text()))
    if enabled and (claim is None or lineage is None or sink is None):
      raise ValueError("enabled commercial usage requires claim, lineage, and durable sink")

  @property
  def enabled(self) -> bool:
    return self._enabled

  def assert_work_allowed(self, billing_mode: Literal["byok", "metered"]) -> None:
    if not self._enabled or self._sink is None:
      return
    guard = getattr(self._sink, "assert_work_allowed", None)
    if callable(guard):
      guard(billing_mode)

  async def mark_late(self, root_source_event_id: str) -> Any | None:
    if self._reconciliation_tracker is not None:
      self._reconciliation_tracker.mark_late(root_source_event_id)
      if self._reconciliation_summary is not None:
        return await self._publish_reconciliation(self._reconciliation_summary)
    return None

  async def reconcile(self, summary: Any) -> Any | None:
    self._ensure_reconciliation_tracker(
      request_id=str(summary.request_id), session_id=str(summary.session_id)
    )
    if self._reconciliation_tracker is None:
      return None
    self._reconciliation_summary = summary
    return await self._publish_reconciliation(summary)

  def _ensure_reconciliation_tracker(
    self, *, request_id: str, session_id: str
  ) -> None:
    if self._reconciliation_tracker is not None:
      return
    claim = self._claim
    lineage = self._lineage
    if claim is None or lineage is None:
      return
    from .usage_reconciliation import CommercialUsageReconciliationTracker

    self._reconciliation_tracker = CommercialUsageReconciliationTracker(
      request_id=request_id,
      session_id=session_id,
      environment=claim.environment,
      source_product=lineage.source_product,
      execution_context_id=str(claim.context_id),
      workflow_run_id=lineage.workflow_run_id,
    )

  async def _publish_reconciliation(self, summary: Any) -> Any:
    report = self._reconciliation_tracker.compare(summary)
    if self._on_reconciliation is not None:
      result = self._on_reconciliation(report)
      if inspect.isawaitable(result):
        await result
    return report

  async def emit(
    self,
    event: UsageEvent,
    *,
    usage_state: Literal[
      "succeeded", "failed_billable", "failed_unbilled", "canceled"
    ] = "succeeded",
  ) -> dict[str, Any] | None:
    if not self._enabled:
      return None
    claim = self._claim
    lineage = self._lineage
    sink = self._sink
    if claim is None or lineage is None or sink is None:
      raise RuntimeError("commercial usage producer lost required configuration")
    if (
      not isinstance(event.event_id, str)
      or not event.event_id.strip()
      or not isinstance(event.request_id, str)
      or not event.request_id.strip()
    ):
      raise ValueError("commercial usage source and request identity are required")
    if event.provider is None or not _CODE.fullmatch(event.provider):
      raise ValueError("commercial usage provider is required")
    if event.channel is None or not _CODE.fullmatch(event.channel):
      raise ValueError("commercial usage channel is required")
    if event.reasoning_tokens_observed is not None and (
      event.reasoning_tokens_observed < 0
      or event.reasoning_tokens_observed > event.output_tokens
    ):
      raise ValueError("reasoning tokens must be an informational output subset")
    if event.billing_mode == "metered" and lineage.reservation_id is None:
      raise ValueError("Hank-funded commercial usage requires reservation lineage")
    if event.input_tokens < 0 or event.cache_read_tokens < 0 or event.cache_creation_tokens < 0:
      raise ValueError("commercial usage token counts cannot be negative")
    if event.provider_unit_deltas and Decimal(str(event.separately_billed_tool_cost_usd)) != 0:
      raise ValueError("typed provider units require allocated, not aggregate, tool cost")
    if event.provider_unit_deltas and event.provider_units is not None:
      raise ValueError("aggregate provider units cannot coexist with typed unit deltas")
    if self._work_start is not None:
      authorization = self._work_start.authorization
      if (
        event.request_id != authorization.request_id
        or event.session_id != authorization.session_id
        or event.provider != authorization.provider
        or event.billing_mode != authorization.billing_mode
      ):
        raise ValueError(
          "commercial usage event differs from verified work-start facts"
        )
    occurred_at = datetime.fromtimestamp(event.timestamp, timezone.utc).isoformat().replace(
      "+00:00", "Z"
    )
    payload: dict[str, Any] = {
      "schema_version": 2 if self._work_start is not None else 1,
      "source_product": lineage.source_product,
      "source_event_id": event.event_id,
      "environment": claim.environment,
      "occurred_at": occurred_at,
      "execution_context_id": str(claim.context_id),
      "request_id": event.request_id,
      "session_id": event.session_id,
      "parent_turn_id": event.parent_turn_id,
      "workflow_run_id": lineage.workflow_run_id,
      "reservation_id": lineage.reservation_id,
      "funding_route_id": lineage.funding_route_id,
      "channel": event.channel,
      "provider": event.provider,
      "operation": lineage.operation,
      "model": event.model or None,
      "capability_id": lineage.capability_id,
      "usage_state": usage_state,
      "uncached_input_tokens": event.input_tokens,
      "billable_output_tokens": event.output_tokens,
      "reasoning_tokens_observed": event.reasoning_tokens_observed,
      "cache_write_tokens": event.cache_creation_tokens,
      "cache_read_tokens": event.cache_read_tokens,
      "is_batch": event.is_batch,
      "provider_units": event.provider_units,
      "separately_billed_tool_cost_usd": str(event.separately_billed_tool_cost_usd),
      "producer_estimated_cost_usd": str(Decimal(str(event.cost_usd))),
      "provider_reported_cost_usd": (
        str(event.provider_reported_cost_usd)
        if event.provider_reported_cost_usd is not None else None
      ),
      "cost_observation_kind": (
        "provider_response"
        if event.provider_reported_cost_usd is not None else "producer_estimate"
      ),
      "producer_rate_version": event.rate_table_version,
      "shadow_rate_version": claim.shadow_rate_version,
      "raw_billing_mode": event.billing_mode,
    }
    if self._work_start is not None:
      authorization = self._work_start.authorization
      payload.update({
        "workflow_attempt_group_id": str(
          authorization.workflow_attempt_group_id
        ),
        "workflow_attempt_number": authorization.workflow_attempt_number,
        "retry_of_workflow_run_id": str(
          authorization.retry_of_workflow_run_id
        ) if authorization.retry_of_workflow_run_id is not None else None,
        "workflow_attempt_kind": authorization.workflow_attempt_kind,
        "work_authorization_id": str(authorization.authorization_id),
      })
    payloads = [payload]
    for operation, count in sorted((event.provider_unit_deltas or {}).items()):
      if not _CODE.fullmatch(operation) or not isinstance(count, int) or count <= 0:
        raise ValueError("commercial provider unit delta is invalid")
      unit_payload = dict(payload)
      unit_payload.update({
        "source_event_id": str(uuid5(NAMESPACE_URL, f"{event.event_id}:provider-unit:{operation}")),
        "operation": operation,
        "uncached_input_tokens": 0,
        "billable_output_tokens": 0,
        "reasoning_tokens_observed": None,
        "cache_write_tokens": 0,
        "cache_read_tokens": 0,
        "provider_units": count,
        "producer_estimated_cost_usd": None,
        "provider_reported_cost_usd": None,
        "separately_billed_tool_cost_usd": "0",
        "cost_observation_kind": "unknown",
      })
      payloads.append(unit_payload)

    for candidate in payloads:
      candidate["source_payload_sha256"] = canonical_usage_payload_sha256(candidate)
      errors = sorted(self._validator.iter_errors(candidate), key=lambda error: list(error.path))
      if errors:
        raise ValueError(f"commercial usage event is invalid: {errors[0].message}")
    result = sink(payloads)
    if inspect.isawaitable(result):
      result = await result
    self._ensure_reconciliation_tracker(
      request_id=event.request_id, session_id=event.session_id
    )
    if self._reconciliation_tracker is not None:
      durability = result if result in {"outbox", "emergency_spool", "lost"} else "outbox"
      self._reconciliation_tracker.record_batch(payloads, durability=durability)
    return payload
