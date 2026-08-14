from __future__ import annotations

import math

import pytest

from agent_gateway.sub_agent_cost_observation import (
  CostObservationThresholdError,
  DEFAULT_COST_OBSERVATION_THRESHOLD_USD,
  resolve_cost_observation_threshold_usd,
)


def test_cost_observation_threshold_prefers_call_then_default() -> None:
  assert resolve_cost_observation_threshold_usd(
    call_threshold_usd=1.25,
    configured_default_threshold_usd=(
      DEFAULT_COST_OBSERVATION_THRESHOLD_USD
    ),
  ) == 1.25
  assert (
    resolve_cost_observation_threshold_usd()
    == DEFAULT_COST_OBSERVATION_THRESHOLD_USD
  )


def test_cost_observation_threshold_is_optional() -> None:
  assert resolve_cost_observation_threshold_usd(
    configured_default_threshold_usd=None,
  ) is None


@pytest.mark.parametrize(
  "value",
  [True, "5", 0, -1, math.inf, -math.inf, math.nan],
)
def test_cost_observation_threshold_rejects_invalid_value(
  value: object,
) -> None:
  with pytest.raises(CostObservationThresholdError) as exc_info:
    resolve_cost_observation_threshold_usd(
      call_threshold_usd=value,
      configured_default_threshold_usd=5.0,
    )

  assert exc_info.value.code == "invalid_cost_observation_threshold"
  assert exc_info.value.source == "call"


def test_cost_observation_threshold_receipt_does_not_claim_enforcement() -> None:
  with pytest.raises(CostObservationThresholdError) as exc_info:
    resolve_cost_observation_threshold_usd(call_threshold_usd=0)

  receipt = exc_info.value.receipt()
  assert receipt["code"] == "invalid_cost_observation_threshold"
  assert "budget" not in receipt["message"]
  assert "cap" not in receipt["message"]
