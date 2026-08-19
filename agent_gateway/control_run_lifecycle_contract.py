from __future__ import annotations

import json
from typing import Any

from .control_run_lifecycle import (
  CONTROL_CHAT_MESSAGEABLE_RUN_STATES,
  CONTROL_RUN_CONTRACT_VERSION,
  CONTROL_RUN_STATE_CLASSIFICATION,
)


def control_run_lifecycle_contract_payload() -> dict[str, Any]:
  """Project the producer-owned lifecycle facts into the public contract."""
  ordered_states = list(CONTROL_RUN_STATE_CLASSIFICATION)
  return {
    "contract_version": CONTROL_RUN_CONTRACT_VERSION,
    "states": ordered_states,
    "classification": {
      state: dict(classification)
      for state, classification in CONTROL_RUN_STATE_CLASSIFICATION.items()
    },
    "chat_messageable_states": [
      state
      for state in ordered_states
      if state in CONTROL_CHAT_MESSAGEABLE_RUN_STATES
    ],
  }


def render_control_run_lifecycle_contract() -> str:
  """Render deterministic checked-in JSON without performing file I/O."""
  return json.dumps(
    control_run_lifecycle_contract_payload(),
    indent=2,
    ensure_ascii=False,
  ) + "\n"


__all__ = [
  "control_run_lifecycle_contract_payload",
  "render_control_run_lifecycle_contract",
]
