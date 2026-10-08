"""Run cost measurement: what a run spent, never what it may spend.

The soft-budget ladder (``runner_soft_budget``) counts turns only; cost is a
measurement on the run record (``stream_complete.usage.estimated_cost``, the
usage events, ``UsageObservation``), and this accumulator is the runner's
running total of it.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Any


class CostAccumulator:
  """A run's recorded spend, with an optional telemetry threshold.

  Crossing ``observation_threshold_usd`` only sets
  ``observation_threshold_crossed``; it never admits, interrupts or gates
  anything.
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
    self.observation_threshold_usd = (
      float(observation_threshold_usd)
      if observation_threshold_usd is not None
      else None
    )
    self._total = 0.0

  def add(self, cost: float) -> None:
    self._total += cost

  @property
  def total(self) -> float:
    return self._total

  @property
  def observation_threshold_crossed(self) -> bool:
    threshold = self.observation_threshold_usd
    return threshold is not None and self.total >= threshold


@dataclass(frozen=True)
class BudgetCostProgress:
  incremental_cost: float
  last_reported_cost: float


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
  )


__all__ = [
  "BudgetCostProgress",
  "CostAccumulator",
  "budget_cost_progress",
]
