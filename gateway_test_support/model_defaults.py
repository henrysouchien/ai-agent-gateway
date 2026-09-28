"""Packaged model-selection defaults, read from the shipped authority.

Tests that exercise *default* resolution derive their expectations here so a
catalog revision changes one frozen pin
(``tests/test_model_registry_artifacts.py``), not every consumer.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_gateway.model_registry import (
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)


@dataclass(frozen=True)
class CapabilityDefault:
  capability_id: str
  model_key: str
  upstream_model: str
  effort: str
  registry_revision: str
  policy_revision: str


def capability_default(capability_id: str) -> CapabilityDefault:
  """Packaged policy default for a capability whose default kind is 'model'; raises for inherit_parent."""
  default = INITIAL_MODEL_SELECTION_POLICY.capabilities[capability_id].default
  if default.kind != "model" or default.model_key is None or default.effort is None:
    raise ValueError(
      f"{capability_id} has no packaged model default (kind={default.kind!r})"
    )
  return CapabilityDefault(
    capability_id=capability_id,
    model_key=default.model_key,
    upstream_model=INITIAL_MODEL_REGISTRY.require(default.model_key).upstream_model,
    effort=default.effort,
    registry_revision=INITIAL_MODEL_REGISTRY.revision,
    policy_revision=INITIAL_MODEL_SELECTION_POLICY.revision,
  )


SESSION_DRIVER = capability_default("session.driver")
QUANT_WORKER = capability_default("investment.quant_worker")
