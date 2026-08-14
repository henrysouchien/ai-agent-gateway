from __future__ import annotations

import math
from typing import Any


DEFAULT_COST_OBSERVATION_THRESHOLD_USD = 50.0


class CostObservationThresholdError(ValueError):
  """Invalid observational cost-threshold input."""

  def __init__(
    self,
    code: str,
    message: str,
    *,
    source: str,
    value: Any = None,
  ) -> None:
    super().__init__(message)
    self.code = code
    self.source = source
    self.value = value

  def receipt(self) -> dict[str, Any]:
    receipt: dict[str, Any] = {
      "code": self.code,
      "message": str(self),
      "source": self.source,
    }
    if self.value is not None:
      receipt["value"] = self.value
    return receipt


def _positive_finite_threshold(value: Any, *, source: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise CostObservationThresholdError(
      "invalid_cost_observation_threshold",
      f"{source} cost observation threshold must be a finite positive number",
      source=source,
      value=value,
    )
  normalized = float(value)
  if not math.isfinite(normalized) or normalized <= 0:
    raise CostObservationThresholdError(
      "invalid_cost_observation_threshold",
      f"{source} cost observation threshold must be a finite positive number",
      source=source,
      value=value,
    )
  return normalized


def resolve_cost_observation_threshold_usd(
  *,
  call_threshold_usd: Any = None,
  configured_default_threshold_usd: Any = (
    DEFAULT_COST_OBSERVATION_THRESHOLD_USD
  ),
) -> float | None:
  """Resolve an optional telemetry threshold, never an execution cap.

  The threshold can drive cost telemetry or a runaway warning. Crossing it
  cannot deny provider requests, interrupt a child, or alter task settlement.
  """

  if call_threshold_usd is not None:
    return _positive_finite_threshold(call_threshold_usd, source="call")
  if configured_default_threshold_usd is None:
    return None
  return _positive_finite_threshold(
    configured_default_threshold_usd,
    source="configured_default",
  )


__all__ = [
  "CostObservationThresholdError",
  "DEFAULT_COST_OBSERVATION_THRESHOLD_USD",
  "resolve_cost_observation_threshold_usd",
]
