"""Default-off production of canonical commercial provider-call usage events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import inspect
import json
import logging
import re
from typing import Any, Awaitable, Callable, Protocol
from uuid import NAMESPACE_URL, uuid5

from jsonschema import Draft202012Validator
from model_authority.bind import CapabilityBind

from hank_contracts.commercial_claims import VerifiedCommercialClaim
from .commercial_contract import (
  canonical_usage_payload_sha256,
  packaged_usage_v3_contract_directory,
)
from hank_contracts.commercial_work import CommercialWorkStartContext
from .multi_user.billing import (
  BillingMode,
  SessionUsageSummary,
  UsageEvent,
  UsageState,
)
from hank_contracts.usage import (
  CommercialUsageLineage as CommercialUsageLineage,
  CommercialUsageReconciliationReport,
  UsageDurability,
)
from .usage_reconciliation import CommercialUsageReconciliationTracker

logger = logging.getLogger(__name__)

_CODE = re.compile(r"^[a-z][a-z0-9._:-]{0,127}$")
_AUTHENTICATED_SUBJECT = re.compile(r"^user:[1-9][0-9]*$")
CommercialUsageSink = Callable[
  [list[dict[str, Any]]],
  UsageDurability | Awaitable[UsageDurability],
]

class CommercialUsageReconciler(Protocol):
  async def reconcile(
    self,
    summary: SessionUsageSummary,
  ) -> CommercialUsageReconciliationReport | None: ...


@dataclass(frozen=True)
class CommercialUsageObservation:
  """Authenticated-owner context for observation-only Usage V3.

  It asserts no commercial claim, workflow, funding route, reservation, rate
  policy or work authorization: only the server-configured identity of the
  gateway, the inference operation, and the authenticated Risk user.
  """

  source_product: str
  environment: str
  operation: str
  authenticated_subject: str

  def __post_init__(self) -> None:
    if not isinstance(self.source_product, str) or not _CODE.fullmatch(self.source_product):
      raise ValueError("observation usage source product is invalid")
    if not isinstance(self.environment, str) or not _CODE.fullmatch(self.environment):
      raise ValueError("observation usage environment is invalid")
    if not isinstance(self.operation, str) or not _CODE.fullmatch(self.operation):
      raise ValueError("observation usage operation is invalid")
    if (
      not isinstance(self.authenticated_subject, str)
      or not _AUTHENTICATED_SUBJECT.fullmatch(self.authenticated_subject)
    ):
      raise ValueError(
        "observation usage subject must be user:<positive Risk user id>"
      )


class CommercialUsageProducer:
  """Convert provider-call deltas to authoritative commercial Usage V3."""

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
    observation: CommercialUsageObservation | None = None,
  ) -> None:
    self._enabled = enabled
    if observation is not None:
      if not isinstance(observation, CommercialUsageObservation):
        raise TypeError("observation usage requires an observation context")
      if claim is not None or lineage is not None or work_start is not None:
        raise ValueError(
          "observation usage cannot carry a commercial claim, lineage or work start"
        )
      if reconciliation_tracker is not None:
        raise ValueError("observation usage has no commercial reconciliation")
    self._observation = observation
    if work_start is not None:
      if not isinstance(work_start, CommercialWorkStartContext):
        raise TypeError("commercial Usage V3 requires verified work-start context")
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
    self._reconciliation_tracker: CommercialUsageReconciliationTracker | None = (
      reconciliation_tracker
    )
    self._on_reconciliation = on_reconciliation
    self._reconciliation_summary: Any | None = None
    schema = (
      packaged_usage_v3_contract_directory()
      / "commercial-usage-event-v3.schema.json"
    )
    self._validator = Draft202012Validator(json.loads(schema.read_text()))
    if enabled and (
      sink is None
      or (observation is None and (claim is None or lineage is None))
    ):
      raise ValueError("enabled commercial usage requires claim, lineage, and durable sink")

  @property
  def enabled(self) -> bool:
    return self._enabled

  def assert_work_allowed(self, billing_mode: BillingMode) -> None:
    # Observation never stops work: an unavailable sink is handled by its
    # outbox/spool, not by refusing the turn.
    if not self._enabled or self._sink is None or self._observation is not None:
      return
    guard = getattr(self._sink, "assert_work_allowed", None)
    if callable(guard):
      guard(billing_mode)

  async def mark_late(self, root_source_event_id: str) -> Any | None:
    tracker = self._reconciliation_tracker
    if tracker is not None:
      tracker.mark_late(root_source_event_id)
      if self._reconciliation_summary is not None:
        return await self._publish_reconciliation(
          self._reconciliation_summary, tracker
        )
    return None

  async def reconcile(self, summary: Any) -> Any | None:
    self._ensure_reconciliation_tracker(
      request_id=str(summary.request_id), session_id=str(summary.session_id)
    )
    tracker = self._reconciliation_tracker
    if tracker is None:
      return None
    self._reconciliation_summary = summary
    return await self._publish_reconciliation(summary, tracker)

  def _ensure_reconciliation_tracker(
    self, *, request_id: str, session_id: str
  ) -> None:
    if self._reconciliation_tracker is not None:
      return
    claim = self._claim
    lineage = self._lineage
    if claim is None or lineage is None:
      return

    self._reconciliation_tracker = CommercialUsageReconciliationTracker(
      request_id=request_id,
      session_id=session_id,
      environment=claim.environment,
      source_product=lineage.source_product,
      execution_context_id=str(claim.context_id),
      workflow_run_id=lineage.workflow_run_id,
    )

  async def _publish_reconciliation(
    self, summary: Any, tracker: CommercialUsageReconciliationTracker
  ) -> Any:
    report = tracker.compare(summary)
    if self._on_reconciliation is not None:
      result = self._on_reconciliation(report)
      if inspect.isawaitable(result):
        await result
    return report

  async def emit(
    self,
    event: UsageEvent,
    *,
    usage_state: UsageState = "succeeded",
  ) -> dict[str, Any] | None:
    if not self._enabled:
      return None
    claim = self._claim
    lineage = self._lineage
    observation = self._observation
    sink = self._sink
    if sink is None:
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
    if event.capability_bind is None:
      raise ValueError("commercial Usage V3 requires capability bind JSON")
    # The bind is the billed identity. UsageEvent already bills a differing
    # model/provider projection as the bind and keeps it in projection_divergence;
    # a work-start lineage naming another capability is logged, and the event is
    # published under the bind (the closed Usage V3 contract has no divergence field).
    bind = CapabilityBind.from_json(event.capability_bind)
    if lineage is not None and lineage.capability_id != bind.capability_id:
      logger.warning(
        "commercial usage %s: lineage capability %s differs from capability bind %s; "
        "billed as the bind",
        event.event_id,
        lineage.capability_id,
        bind.capability_id,
      )
    if event.channel is None or not _CODE.fullmatch(event.channel):
      raise ValueError("commercial usage channel is required")
    if event.reasoning_tokens_observed is not None and (
      event.reasoning_tokens_observed < 0
      or event.reasoning_tokens_observed > event.output_tokens
    ):
      raise ValueError("reasoning tokens must be an informational output subset")
    # Money: Hank-funded usage without a budget reservation is unbillable spend.
    if (
      lineage is not None
      and event.billing_mode == "metered"
      and lineage.reservation_id is None
    ):
      raise ValueError(
        "Hank-funded commercial usage requires reservation lineage: start the "
        "work under a metered work authorization whose reservation_id names "
        "the risk_module budget reservation (minted by its budget reserve "
        "when it issues that authorization)"
      )
    if event.input_tokens < 0 or event.cache_read_tokens < 0 or event.cache_creation_tokens < 0:
      raise ValueError("commercial usage token counts cannot be negative")
    if event.provider_unit_deltas and Decimal(str(event.separately_billed_tool_cost_usd)) != 0:
      raise ValueError("typed provider units require allocated, not aggregate, tool cost")
    if event.provider_unit_deltas and event.provider_units is not None:
      raise ValueError("aggregate provider units cannot coexist with typed unit deltas")
    occurred_at = datetime.fromtimestamp(event.timestamp, timezone.utc).isoformat().replace(
      "+00:00", "Z"
    )
    if observation is not None:
      # Observation asserts no commercial identity: every commercial lineage
      # key is an explicit null and the authenticated subject names the owner.
      identity: dict[str, Any] = {
        "source_product": observation.source_product,
        "environment": observation.environment,
        "execution_context_id": None,
        "workflow_run_id": None,
        "reservation_id": None,
        "funding_route_id": None,
        "operation": observation.operation,
        "shadow_rate_version": None,
        "authenticated_subject": observation.authenticated_subject,
      }
    else:
      if claim is None or lineage is None:
        raise RuntimeError("commercial usage producer lost required configuration")
      identity = {
        "source_product": lineage.source_product,
        "environment": claim.environment,
        "execution_context_id": str(claim.context_id),
        "workflow_run_id": lineage.workflow_run_id,
        "reservation_id": lineage.reservation_id,
        "funding_route_id": lineage.funding_route_id,
        "operation": lineage.operation,
        "shadow_rate_version": claim.shadow_rate_version,
      }
    payload: dict[str, Any] = {
      "schema_version": 3,
      "source_product": identity["source_product"],
      "source_event_id": event.event_id,
      "environment": identity["environment"],
      "occurred_at": occurred_at,
      "execution_context_id": identity["execution_context_id"],
      "request_id": event.request_id,
      "session_id": event.session_id,
      "parent_turn_id": event.parent_turn_id,
      "workflow_run_id": identity["workflow_run_id"],
      "reservation_id": identity["reservation_id"],
      "funding_route_id": identity["funding_route_id"],
      "channel": event.channel,
      "provider": bind.provider,
      "operation": identity["operation"],
      "model": bind.upstream_model,
      "capability_id": bind.capability_id,
      "capability_bind": bind.to_json(),
      "provider_reported_model": event.provider_reported_model,
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
      "shadow_rate_version": identity["shadow_rate_version"],
      "raw_billing_mode": event.billing_mode,
      "workflow_attempt_group_id": None,
      "workflow_attempt_number": None,
      "retry_of_workflow_run_id": None,
      "workflow_attempt_kind": None,
      "work_authorization_id": None,
    }
    if "authenticated_subject" in identity:
      payload["authenticated_subject"] = identity["authenticated_subject"]
      # Risk stores these observations at eight decimal places. Normalize only
      # this shape before hashing; existing commercial payload bytes stay exact.
      for field in ("producer_estimated_cost_usd", "provider_reported_cost_usd"):
        value = payload[field]
        if value is not None:
          payload[field] = format(
            Decimal(value).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP),
            "f",
          )
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
    if event.selection_provenance is not None:
      payload["selection_provenance"] = event.selection_provenance
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
    # rulings-index WA-ACCOUNT-BUDGET: one ledger per work start, fed only here.
    ledger = None if self._work_start is None else getattr(
      self._work_start, "spend_ledger", None
    )
    if ledger is not None:
      ledger.record_usd(payload.get("producer_estimated_cost_usd"))
    result = sink(payloads)
    if inspect.isawaitable(result):
      result = await result
    self._ensure_reconciliation_tracker(
      request_id=event.request_id, session_id=event.session_id
    )
    if self._reconciliation_tracker is not None:
      self._reconciliation_tracker.record_batch(payloads, durability=result)
    return payload
