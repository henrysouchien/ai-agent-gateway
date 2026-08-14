from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Any


class CostAccumulator:
  """Running cost tracker shared across parent and sub-agent runners."""

  def __init__(self, budget: float) -> None:
    self.budget = budget
    self._total = 0.0

  def add(self, cost: float) -> None:
    self._total += cost

  @property
  def total(self) -> float:
    return self._total

  @property
  def exceeded(self) -> bool:
    return self._total >= self.budget


class ChildCostAccumulator(CostAccumulator):
  """Child-local budget tracker that forwards spend to the parent."""

  def __init__(self, parent: CostAccumulator | None, budget: float) -> None:
    super().__init__(budget)
    self._parent = parent

  def add(self, cost: float) -> None:
    super().add(cost)
    if self._parent is not None:
      self._parent.add(cost)

  @property
  def _local_exceeded(self) -> bool:
    return self.total >= self.budget

  @property
  def exceeded(self) -> bool:
    return self._local_exceeded or bool(self._parent is not None and self._parent.exceeded)

  @property
  def exceeded_reason(self) -> str | None:
    if self._local_exceeded:
      return "child_budget"
    if self._parent is not None and self._parent.exceeded:
      return "parent_budget"
    return None

  @property
  def effective_total(self) -> float:
    if not self._local_exceeded and self._parent is not None and self._parent.exceeded:
      return self._parent.total
    return self.total

  @property
  def effective_budget(self) -> float:
    if not self._local_exceeded and self._parent is not None and self._parent.exceeded:
      return self._parent.budget
    return self.budget


class ObservationOnlyCostAccumulator(CostAccumulator):
  """Record child spend against an optional telemetry threshold.

  Observation-only child accounting deliberately does not forward spend into
  a potentially enforcing parent accumulator. The shared usage aggregator and
  the child result remain the authoritative rollups for workflow accounting.
  Crossing ``observation_threshold_usd`` never makes ``exceeded`` true and
  never participates in provider-request admission.
  """

  def __init__(
    self,
    observation_threshold_usd: float | None = None,
  ) -> None:
    if observation_threshold_usd is not None and (
      isinstance(observation_threshold_usd, bool)
      or not isinstance(observation_threshold_usd, (int, float))
      or not math.isfinite(float(observation_threshold_usd))
      or float(observation_threshold_usd) <= 0
    ):
      raise ValueError(
        "cost_observation_threshold_usd must be a finite positive number"
      )
    # AgentRunner treats ``budget`` as hard execution authority. Keep it null;
    # the observational threshold has a distinct, non-authoritative field.
    self.budget = None
    self.observation_threshold_usd = (
      float(observation_threshold_usd)
      if observation_threshold_usd is not None
      else None
    )
    self._total = 0.0

  @property
  def exceeded(self) -> bool:
    return False

  @property
  def exceeded_reason(self) -> None:
    return None

  @property
  def observation_threshold_crossed(self) -> bool:
    threshold = self.observation_threshold_usd
    return threshold is not None and self.total >= threshold


@dataclass(frozen=True)
class BudgetExceededState:
  total_cost: float
  budget: float
  reason: Any
  reason_suffix: str


@dataclass(frozen=True)
class BudgetCostProgress:
  incremental_cost: float
  last_reported_cost: float
  exceeded_state: BudgetExceededState | None


@dataclass(frozen=True)
class ProviderRequestBudgetAdmission:
  """Hard cost admission for one provider request."""

  max_output_tokens: int | None
  projected_max_cost: float
  remaining_budget: float
  denied_state: BudgetExceededState | None


class ProviderRequestBudgetError(RuntimeError):
  """Raised when provider request cost cannot be bounded safely."""


def budget_reason_suffix(reason: Any) -> str:
  if reason == "child_budget":
    return " (child budget)"
  if reason == "parent_budget":
    return " (parent budget)"
  return ""


def budget_exceeded_state(cost_accumulator: Any) -> BudgetExceededState | None:
  if not cost_accumulator.exceeded:
    return None
  reason = getattr(cost_accumulator, "exceeded_reason", None)
  return BudgetExceededState(
    total_cost=getattr(cost_accumulator, "effective_total", cost_accumulator.total),
    budget=getattr(cost_accumulator, "effective_budget", cost_accumulator.budget),
    reason=reason,
    reason_suffix=budget_reason_suffix(reason),
  )


def budget_cost_progress(
  cost_accumulator: Any,
  *,
  running_total: float,
  last_reported_cost: float,
) -> BudgetCostProgress:
  incremental_cost = max(0.0, running_total - last_reported_cost)
  if incremental_cost:
    cost_accumulator.add(incremental_cost)
  return BudgetCostProgress(
    incremental_cost=incremental_cost,
    last_reported_cost=running_total,
    exceeded_state=budget_exceeded_state(cost_accumulator),
  )


def _remaining_budget(
  cost_accumulator: Any,
) -> tuple[float, float, str | None] | None:
  if isinstance(cost_accumulator, ObservationOnlyCostAccumulator):
    return None
  if isinstance(cost_accumulator, ChildCostAccumulator):
    candidates = [
      (
        max(0.0, float(cost_accumulator.budget) - cost_accumulator.total),
        float(cost_accumulator.budget),
        "child_budget",
      )
    ]
    parent = cost_accumulator._parent
    if parent is not None:
      candidates.append((
        max(0.0, float(parent.budget) - parent.total),
        float(parent.budget),
        "parent_budget",
      ))
    return min(candidates, key=lambda item: item[0])
  budget = getattr(cost_accumulator, "budget", None)
  total = getattr(cost_accumulator, "total", None)
  if (
    isinstance(budget, bool)
    or not isinstance(budget, int | float)
    or not math.isfinite(float(budget))
    or isinstance(total, bool)
    or not isinstance(total, int | float)
    or not math.isfinite(float(total))
  ):
    raise ProviderRequestBudgetError(
      "provider request budget authority is invalid"
    )
  return max(0.0, float(budget) - float(total)), float(budget), None


def _project_provider_request_cost(
  provider: Any,
  *,
  model: str,
  estimated_input_tokens: int,
  max_output_tokens: int,
) -> float:
  try:
    uncached_estimate = provider.estimate_cost(
      model,
      estimated_input_tokens,
      max_output_tokens,
      cache_read_tokens=0,
      cache_creation_tokens=0,
    )
    cache_write_estimate = provider.estimate_cost(
      model,
      0,
      max_output_tokens,
      cache_read_tokens=0,
      cache_creation_tokens=estimated_input_tokens,
    )
    total = max(
      float(uncached_estimate.total),
      float(cache_write_estimate.total),
    )
  except Exception as exc:
    raise ProviderRequestBudgetError(
      "provider request cost could not be bounded"
    ) from exc
  if not math.isfinite(total) or total < 0:
    raise ProviderRequestBudgetError(
      "provider request cost could not be bounded"
    )
  return total


def admit_provider_request_budget(
  cost_accumulator: Any | None,
  *,
  provider: Any,
  model: str,
  estimated_input_tokens: int,
  requested_max_output_tokens: int,
) -> ProviderRequestBudgetAdmission:
  """Cap one request to the remaining hard budget before provider transport.

  Input tokens are conservatively priced as uncached. Output tokens are
  bounded by the request's provider-enforced ``max_tokens`` value.
  """

  if (
    isinstance(estimated_input_tokens, bool)
    or not isinstance(estimated_input_tokens, int)
    or estimated_input_tokens < 0
    or isinstance(requested_max_output_tokens, bool)
    or not isinstance(requested_max_output_tokens, int)
    or requested_max_output_tokens <= 0
  ):
    raise ProviderRequestBudgetError(
      "provider request cost could not be bounded"
    )
  remaining = (
    _remaining_budget(cost_accumulator)
    if cost_accumulator is not None
    else None
  )
  if remaining is None:
    return ProviderRequestBudgetAdmission(
      max_output_tokens=requested_max_output_tokens,
      projected_max_cost=0.0,
      remaining_budget=math.inf,
      denied_state=None,
    )
  remaining_usd, effective_budget, reason = remaining
  minimum_cost = _project_provider_request_cost(
    provider,
    model=model,
    estimated_input_tokens=estimated_input_tokens,
    max_output_tokens=1,
  )
  if minimum_cost > remaining_usd:
    return ProviderRequestBudgetAdmission(
      max_output_tokens=None,
      projected_max_cost=minimum_cost,
      remaining_budget=remaining_usd,
      denied_state=BudgetExceededState(
        total_cost=effective_budget,
        budget=effective_budget,
        reason=reason,
        reason_suffix=budget_reason_suffix(reason),
      ),
    )
  full_cost = _project_provider_request_cost(
    provider,
    model=model,
    estimated_input_tokens=estimated_input_tokens,
    max_output_tokens=requested_max_output_tokens,
  )
  if full_cost <= remaining_usd:
    return ProviderRequestBudgetAdmission(
      max_output_tokens=requested_max_output_tokens,
      projected_max_cost=full_cost,
      remaining_budget=remaining_usd,
      denied_state=None,
    )
  low = 1
  high = requested_max_output_tokens
  while low < high:
    candidate = (low + high + 1) // 2
    candidate_cost = _project_provider_request_cost(
      provider,
      model=model,
      estimated_input_tokens=estimated_input_tokens,
      max_output_tokens=candidate,
    )
    if candidate_cost <= remaining_usd:
      low = candidate
    else:
      high = candidate - 1
  projected = _project_provider_request_cost(
    provider,
    model=model,
    estimated_input_tokens=estimated_input_tokens,
    max_output_tokens=low,
  )
  return ProviderRequestBudgetAdmission(
    max_output_tokens=low,
    projected_max_cost=projected,
    remaining_budget=remaining_usd,
    denied_state=None,
  )


__all__ = [
  "BudgetCostProgress",
  "BudgetExceededState",
  "ProviderRequestBudgetAdmission",
  "ProviderRequestBudgetError",
  "ChildCostAccumulator",
  "CostAccumulator",
  "ObservationOnlyCostAccumulator",
  "budget_cost_progress",
  "budget_exceeded_state",
  "budget_reason_suffix",
  "admit_provider_request_budget",
]
