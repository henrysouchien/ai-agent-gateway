from __future__ import annotations

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


__all__ = [
  "BudgetCostProgress",
  "BudgetExceededState",
  "ChildCostAccumulator",
  "CostAccumulator",
  "budget_cost_progress",
  "budget_exceeded_state",
  "budget_reason_suffix",
]
