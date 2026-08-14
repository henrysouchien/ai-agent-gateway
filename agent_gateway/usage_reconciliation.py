"""Reconciliation-only comparison of commercial deltas and legacy summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
import threading
from typing import Any, Literal
from uuid import UUID

from agent_workflow_contracts import CapabilityBind

from .multi_user.billing import SessionUsageSummary


@dataclass(frozen=True)
class CommercialUsageReconciliationReport:
  environment: str
  source_product: str
  request_id: str
  session_id: str
  execution_context_id: str
  workflow_run_id: str
  status: Literal["match", "mismatch", "incomplete"]
  commercial_event_count: int
  provider_call_event_count: int
  durable_provider_call_event_count: int
  separate_unit_event_count: int
  expected_provider_call_count: int
  provider_call_count_delta: int
  missing_event_id_count: int
  missing_source_event_ids: tuple[str, ...]
  emergency_spooled_event_count: int
  durability_lost_event_count: int
  conflicting_event_id_count: int
  conflicting_source_event_ids: tuple[str, ...]
  late_event_count: int
  late_source_event_ids: tuple[str, ...]
  observed_source_event_ids: tuple[str, ...]
  summary_usage_event_ids: tuple[str, ...]
  event_lines: tuple[dict[str, Any], ...]
  commercial_input_tokens: int
  summary_input_tokens: int
  input_token_delta: int
  commercial_output_tokens: int
  summary_output_tokens: int
  output_token_delta: int
  commercial_cache_read_tokens: int
  summary_cache_read_tokens: int
  cache_read_token_delta: int
  commercial_cache_write_tokens: int
  summary_cache_write_tokens: int
  cache_write_token_delta: int
  reasoning_tokens_observed: int
  provider_units: Decimal
  commercial_producer_estimate_usd: Decimal
  summary_estimate_usd: Decimal
  estimate_delta_usd: Decimal
  drain_complete: bool
  in_flight_task_count: int
  summary_started_at: float
  summary_ended_at: float
  summary_emitted_as_cost_event: Literal[False] = False
  workflow_attempt_group_id: str | None = None
  workflow_attempt_number: int | None = None
  retry_of_workflow_run_id: str | None = None
  workflow_attempt_kind: str | None = None
  work_authorization_id: str | None = None
  source_usage_schema_version: Literal[1, 2, 3] = 1

  @property
  def evidence_schema_version(self) -> Literal[1, 2, 3]:
    if self.source_usage_schema_version == 3:
      return 3
    return 2 if self.workflow_attempt_group_id is not None else 1

  def as_dict(self) -> dict[str, Any]:
    value = asdict(self)
    for field in (
      "provider_units", "commercial_producer_estimate_usd",
      "summary_estimate_usd", "estimate_delta_usd",
    ):
      value[field] = str(value[field])
    for field in (
      "missing_source_event_ids", "late_source_event_ids",
      "conflicting_source_event_ids", "observed_source_event_ids",
      "summary_usage_event_ids", "event_lines",
    ):
      value[field] = list(value[field])
    if self.evidence_schema_version != 3:
      value.pop("source_usage_schema_version")
    if self.evidence_schema_version == 1:
      for field in (
        "workflow_attempt_group_id", "workflow_attempt_number",
        "retry_of_workflow_run_id", "workflow_attempt_kind",
        "work_authorization_id",
      ):
        value.pop(field)
    return value


class CommercialUsageReconciliationTracker:
  """Request-scoped parity tracker; it has no commercial persistence sink."""

  def __init__(
    self,
    *,
    request_id: str,
    session_id: str,
    environment: str | None = None,
    source_product: str | None = None,
    execution_context_id: str | None = None,
    workflow_run_id: str | None = None,
  ) -> None:
    if not request_id or not session_id:
      raise ValueError("commercial reconciliation identity is required")
    self._request_id = request_id
    self._session_id = session_id
    self._environment = str(environment or "").strip()
    self._source_product = str(source_product or "").strip()
    self._execution_context_id = str(execution_context_id or "").strip()
    self._workflow_run_id = str(workflow_run_id or "").strip()
    self._lock = threading.Lock()
    self._payloads: dict[str, dict[str, Any]] = {}
    self._roots: dict[str, set[str]] = {}
    self._late_ids: set[str] = set()
    self._conflicts: set[str] = set()
    self._durability: dict[str, str] = {}
    self._parent_ids: set[str] = set()
    self._unit_ids: set[str] = set()
    self._source_schema_version: Literal[1, 2, 3] | None = None
    self._attempt_identity: tuple[
      str, int, str | None, str, str | None
    ] | None = None

  def record_batch(
    self,
    payloads: list[dict[str, Any]],
    *,
    durability: Literal["outbox", "emergency_spool", "lost"] = "outbox",
  ) -> None:
    if not payloads:
      return
    root = str(payloads[0].get("source_event_id") or "")
    batch_ids: set[str] = set()
    with self._lock:
      staged_environment = self._environment
      staged_source_product = self._source_product
      staged_execution_context_id = self._execution_context_id
      staged_workflow_run_id = self._workflow_run_id
      staged_schema_version = self._source_schema_version
      staged_attempt_identity = self._attempt_identity
      staged_payloads = dict(self._payloads)
      staged_roots = {key: set(value) for key, value in self._roots.items()}
      staged_conflicts = set(self._conflicts)
      staged_durability = dict(self._durability)
      staged_parent_ids = set(self._parent_ids)
      staged_unit_ids = set(self._unit_ids)
      for payload in payloads:
        if (
          payload.get("request_id") != self._request_id
          or payload.get("session_id") != self._session_id
        ):
          raise ValueError("commercial reconciliation payload identity mismatch")
        source_identity = (
          str(payload.get("environment") or "").strip(),
          str(payload.get("source_product") or "").strip(),
          str(payload.get("execution_context_id") or "").strip(),
          str(payload.get("workflow_run_id") or "").strip(),
        )
        if not all(source_identity):
          raise ValueError("commercial reconciliation source lineage is missing")
        configured = (
          staged_environment,
          staged_source_product,
          staged_execution_context_id,
          staged_workflow_run_id,
        )
        if any(configured) and configured != source_identity:
          raise ValueError("commercial reconciliation source lineage mismatch")
        source_schema_version = int(payload.get("schema_version") or 1)
        if source_schema_version not in {1, 2, 3}:
          raise ValueError("commercial reconciliation source schema is unsupported")
        if (
          staged_schema_version is not None
          and staged_schema_version != source_schema_version
        ):
          raise ValueError("commercial reconciliation cannot mix usage schema versions")
        if source_schema_version == 3:
          try:
            bind = CapabilityBind.from_receipt(payload.get("capability_bind"))
          except (TypeError, ValueError) as exc:
            raise ValueError(
              "commercial reconciliation capability bind is invalid"
            ) from exc
          if payload.get("capability_bind") != bind.receipt():
            raise ValueError(
              "commercial reconciliation capability bind is not canonical"
            )
          if (
            payload.get("provider") != bind.provider
            or payload.get("model") != bind.upstream_model
            or payload.get("capability_id") != bind.capability_id
          ):
            raise ValueError(
              "commercial reconciliation identity projection mismatch"
            )
          reported = payload.get("provider_reported_model")
          if reported is not None and (
            not isinstance(reported, str) or not reported.strip()
          ):
            raise ValueError(
              "commercial reconciliation provider-reported model is invalid"
            )
        attempt_fields = (
          "workflow_attempt_group_id",
          "workflow_attempt_number",
          "retry_of_workflow_run_id",
          "workflow_attempt_kind",
          "work_authorization_id",
        )
        attempt_field_count = sum(field in payload for field in attempt_fields)
        if source_schema_version == 2 and attempt_field_count != len(attempt_fields):
          raise ValueError("commercial reconciliation attempt identity is incomplete")
        if source_schema_version == 3 and attempt_field_count != len(attempt_fields):
          raise ValueError("commercial reconciliation attempt identity is incomplete")
        attempt_identity = None
        group_value = payload.get("workflow_attempt_group_id")
        if source_schema_version == 3 and group_value is None:
          if any(payload.get(field) is not None for field in attempt_fields):
            raise ValueError("commercial reconciliation attempt identity is invalid")
        elif attempt_field_count:
          group_id = str(payload.get("workflow_attempt_group_id") or "").strip()
          work_authorization_value = payload.get("work_authorization_id")
          work_authorization_id = (
            str(work_authorization_value).strip()
            if work_authorization_value is not None else None
          )
          attempt_kind = str(payload.get("workflow_attempt_kind") or "").strip()
          attempt_number = payload.get("workflow_attempt_number")
          retry_value = payload.get("retry_of_workflow_run_id")
          retry_of = str(retry_value).strip() if retry_value is not None else None
          try:
            canonical_group = str(UUID(group_id))
            canonical_authorization = (
              str(UUID(work_authorization_id))
              if work_authorization_id is not None else None
            )
            canonical_retry = str(UUID(retry_of)) if retry_of is not None else None
            if source_schema_version == 3:
              for field in (
                "execution_context_id", "workflow_run_id", "funding_route_id",
              ):
                value = str(payload.get(field) or "").strip()
                if value != str(UUID(value)):
                  raise ValueError
              reservation_value = payload.get("reservation_id")
              if reservation_value is not None:
                reservation = str(reservation_value).strip()
                if reservation != str(UUID(reservation)):
                  raise ValueError
          except (ValueError, AttributeError, TypeError):
            raise ValueError(
              "commercial reconciliation attempt identity is invalid"
            ) from None
          if (
            group_id != canonical_group
            or work_authorization_id != canonical_authorization
            or (retry_of is not None and retry_of != canonical_retry)
            or type(attempt_number) is not int
            or attempt_number <= 0
            or attempt_kind not in {"initial", "user_retry", "automatic_retry"}
            or (
              source_schema_version == 3
              and source_identity[1] == "hank-agent-gateway"
              and work_authorization_id is None
            )
            or (
              source_schema_version == 3
              and source_identity[1] == "risk-module-direct"
              and work_authorization_id is not None
            )
            or (
              attempt_kind == "initial"
              and (
                attempt_number != 1
                or retry_of is not None
                or group_id != source_identity[3]
              )
            )
            or (
              attempt_kind != "initial"
              and (
                attempt_number <= 1
                or retry_of is None
                or group_id == source_identity[3]
                or retry_of == source_identity[3]
              )
            )
          ):
            raise ValueError(
              "commercial reconciliation attempt identity is invalid"
            )
          attempt_identity = (
            group_id, attempt_number, retry_of, attempt_kind,
            work_authorization_id,
          )
          if (
            staged_attempt_identity is not None
            and staged_attempt_identity != attempt_identity
          ):
            raise ValueError("commercial reconciliation attempt identity mismatch")
        if (
          source_schema_version == 3
          and staged_schema_version == 3
          and (staged_attempt_identity is None) != (attempt_identity is None)
        ):
          raise ValueError("commercial reconciliation attempt identity mismatch")
        (
          staged_environment,
          staged_source_product,
          staged_execution_context_id,
          staged_workflow_run_id,
        ) = source_identity
        staged_schema_version = source_schema_version
        if attempt_identity is not None:
          staged_attempt_identity = attempt_identity
        event_id = str(payload.get("source_event_id") or "").strip()
        digest = str(payload.get("source_payload_sha256") or "").strip()
        if not event_id or not digest:
          raise ValueError("commercial reconciliation source identity is missing")
        existing = staged_payloads.get(event_id)
        if existing is not None and existing.get("source_payload_sha256") != digest:
          staged_conflicts.add(event_id)
          continue
        staged_payloads.setdefault(event_id, dict(payload))
        durability_rank = {"lost": 0, "emergency_spool": 1, "outbox": 2}
        existing_durability = staged_durability.get(event_id)
        if (
          existing_durability is None
          or durability_rank[durability] > durability_rank[existing_durability]
        ):
          staged_durability[event_id] = durability
        batch_ids.add(event_id)
      if root:
        staged_roots.setdefault(root, set()).update(batch_ids)
        staged_parent_ids.add(root)
        staged_unit_ids.update(batch_ids - {root})
      self._environment = staged_environment
      self._source_product = staged_source_product
      self._execution_context_id = staged_execution_context_id
      self._workflow_run_id = staged_workflow_run_id
      self._source_schema_version = staged_schema_version
      self._attempt_identity = staged_attempt_identity
      self._payloads = staged_payloads
      self._roots = staged_roots
      self._conflicts = staged_conflicts
      self._durability = staged_durability
      self._parent_ids = staged_parent_ids
      self._unit_ids = staged_unit_ids

  def mark_late(self, root_source_event_id: str) -> None:
    with self._lock:
      self._late_ids.update(
        self._roots.get(root_source_event_id, {root_source_event_id})
      )

  def compare(self, summary: SessionUsageSummary) -> CommercialUsageReconciliationReport:
    if summary.request_id != self._request_id or summary.session_id != self._session_id:
      raise ValueError("commercial reconciliation summary identity mismatch")
    with self._lock:
      payloads = [dict(payload) for payload in self._payloads.values()]
      late_ids = tuple(sorted(self._late_ids))
      conflict_count = len(self._conflicts)
      conflict_ids = tuple(sorted(self._conflicts))
      durability = dict(self._durability)
      parent_ids = set(self._parent_ids)
      unit_ids = set(self._unit_ids)
      observed_ids = tuple(sorted(self._payloads))
      source_identity = (
        self._environment,
        self._source_product,
        self._execution_context_id,
        self._workflow_run_id,
      )
      source_schema_version = self._source_schema_version
      attempt_identity = self._attempt_identity
    if not all(source_identity):
      raise ValueError("commercial reconciliation source lineage is unavailable")
    provider_calls = [
      payload for payload in payloads
      if str(payload.get("source_event_id")) in parent_ids
    ]
    unit_events = [
      payload for payload in payloads
      if str(payload.get("source_event_id")) in unit_ids
    ]

    def token_total(name: str) -> int:
      return sum(int(payload.get(name) or 0) for payload in payloads)

    commercial_input = token_total("uncached_input_tokens")
    commercial_output = token_total("billable_output_tokens")
    commercial_cache_read = token_total("cache_read_tokens")
    commercial_cache_write = token_total("cache_write_tokens")
    reasoning = token_total("reasoning_tokens_observed")
    provider_units = sum(
      (Decimal(str(payload.get("provider_units") or 0)) for payload in payloads),
      Decimal("0"),
    )
    producer_estimate = sum(
      (
        Decimal(str(payload["producer_estimated_cost_usd"]))
        for payload in payloads
        if payload.get("producer_estimated_cost_usd") is not None
      ),
      Decimal("0"),
    )
    summary_estimate = Decimal(str(summary.cost))
    durable_provider_calls = [
      payload for payload in provider_calls
      if durability.get(str(payload.get("source_event_id"))) != "lost"
    ]
    expected_event_ids = set(summary.usage_event_ids)
    durable_parent_ids = {
      str(payload.get("source_event_id")) for payload in durable_provider_calls
    }
    missing_ids = tuple(sorted(expected_event_ids - durable_parent_ids))
    expected_count = int(summary.usage_event_count)
    missing = max(len(missing_ids), expected_count - len(durable_provider_calls), 0)
    event_count_delta = len(provider_calls) - expected_count
    emergency_count = sum(1 for value in durability.values() if value == "emergency_spool")
    lost_count = sum(1 for value in durability.values() if value == "lost")
    deltas = (
      commercial_input - int(summary.input_tokens),
      commercial_output - int(summary.output_tokens),
      commercial_cache_read - int(summary.cache_read_tokens),
      commercial_cache_write - int(summary.cache_creation_tokens),
    )
    estimate_delta = producer_estimate - summary_estimate
    is_match = (
      not any(deltas)
      and abs(estimate_delta) <= Decimal("0.00000001")
      and missing == 0
      and event_count_delta == 0
      and durable_parent_ids == expected_event_ids
      and conflict_count == 0
      and lost_count == 0
      and not late_ids
    )
    status: Literal["match", "mismatch", "incomplete"] = (
      "incomplete" if not summary.drain_complete or summary.in_flight_task_count else
      "match" if is_match else "mismatch"
    )
    event_lines = tuple(
      {
        "source_event_id": str(payload.get("source_event_id")),
        "source_payload_sha256": str(payload.get("source_payload_sha256")),
        "event_kind": (
          "provider_call"
          if str(payload.get("source_event_id")) in parent_ids
          else "separate_unit"
        ),
        "durability": durability.get(
          str(payload.get("source_event_id")), "lost"
        ),
        "late": str(payload.get("source_event_id")) in set(late_ids),
        "occurred_at": str(payload.get("occurred_at") or ""),
        "uncached_input_tokens": int(payload.get("uncached_input_tokens") or 0),
        "billable_output_tokens": int(payload.get("billable_output_tokens") or 0),
        "cache_read_tokens": int(payload.get("cache_read_tokens") or 0),
        "cache_write_tokens": int(payload.get("cache_write_tokens") or 0),
        "reasoning_tokens_observed": (
          int(payload["reasoning_tokens_observed"])
          if payload.get("reasoning_tokens_observed") is not None else None
        ),
        "provider_units": (
          str(payload["provider_units"])
          if payload.get("provider_units") is not None else None
        ),
        "producer_estimated_cost_usd": (
          str(payload["producer_estimated_cost_usd"])
          if payload.get("producer_estimated_cost_usd") is not None else None
        ),
        **(
          {
            "source_schema_version": source_schema_version,
            "workflow_attempt_group_id": str(
              payload.get("workflow_attempt_group_id")
            ) if payload.get("workflow_attempt_group_id") is not None else None,
            "workflow_attempt_number": (
              int(payload["workflow_attempt_number"])
              if payload.get("workflow_attempt_number") is not None else None
            ),
            "retry_of_workflow_run_id": payload.get("retry_of_workflow_run_id"),
            "workflow_attempt_kind": (
              str(payload["workflow_attempt_kind"])
              if payload.get("workflow_attempt_kind") is not None else None
            ),
            "work_authorization_id": (
              str(payload["work_authorization_id"])
              if payload.get("work_authorization_id") is not None else None
            ),
            **(
              {
                "capability_bind": dict(payload["capability_bind"]),
                "provider_reported_model": payload.get(
                  "provider_reported_model"
                ),
                "provider": str(payload["provider"]),
                "model": str(payload["model"]),
                "capability_id": str(payload["capability_id"]),
              }
              if source_schema_version == 3 else {}
            ),
          }
          if source_schema_version in {2, 3} else {}
        ),
      }
      for payload in sorted(payloads, key=lambda value: str(value.get("source_event_id")))
    )
    return CommercialUsageReconciliationReport(
      environment=source_identity[0],
      source_product=source_identity[1],
      request_id=self._request_id,
      session_id=self._session_id,
      execution_context_id=source_identity[2],
      workflow_run_id=source_identity[3],
      status=status,
      commercial_event_count=len(payloads),
      provider_call_event_count=len(provider_calls),
      durable_provider_call_event_count=len(durable_provider_calls),
      separate_unit_event_count=len(unit_events),
      expected_provider_call_count=expected_count,
      provider_call_count_delta=event_count_delta,
      missing_event_id_count=missing,
      missing_source_event_ids=missing_ids,
      emergency_spooled_event_count=emergency_count,
      durability_lost_event_count=lost_count,
      conflicting_event_id_count=conflict_count,
      conflicting_source_event_ids=conflict_ids,
      late_event_count=len(late_ids),
      late_source_event_ids=late_ids,
      observed_source_event_ids=observed_ids,
      summary_usage_event_ids=tuple(sorted(expected_event_ids)),
      event_lines=event_lines,
      commercial_input_tokens=commercial_input,
      summary_input_tokens=int(summary.input_tokens),
      input_token_delta=deltas[0],
      commercial_output_tokens=commercial_output,
      summary_output_tokens=int(summary.output_tokens),
      output_token_delta=deltas[1],
      commercial_cache_read_tokens=commercial_cache_read,
      summary_cache_read_tokens=int(summary.cache_read_tokens),
      cache_read_token_delta=deltas[2],
      commercial_cache_write_tokens=commercial_cache_write,
      summary_cache_write_tokens=int(summary.cache_creation_tokens),
      cache_write_token_delta=deltas[3],
      reasoning_tokens_observed=reasoning,
      provider_units=provider_units,
      commercial_producer_estimate_usd=producer_estimate,
      summary_estimate_usd=summary_estimate,
      estimate_delta_usd=estimate_delta,
      drain_complete=bool(summary.drain_complete),
      in_flight_task_count=int(summary.in_flight_task_count),
      summary_started_at=float(summary.started_at),
      summary_ended_at=float(summary.ended_at),
      workflow_attempt_group_id=(attempt_identity[0] if attempt_identity else None),
      workflow_attempt_number=(attempt_identity[1] if attempt_identity else None),
      retry_of_workflow_run_id=(attempt_identity[2] if attempt_identity else None),
      workflow_attempt_kind=(attempt_identity[3] if attempt_identity else None),
      work_authorization_id=(attempt_identity[4] if attempt_identity else None),
      source_usage_schema_version=source_schema_version or 1,
    )
