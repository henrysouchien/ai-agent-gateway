from __future__ import annotations

import json
import os
from pathlib import Path
from typing import get_args

import pytest

from agent_gateway.control_run_lifecycle import (
  CONTROL_ACTIVE_RUN_STATES,
  CONTROL_CANCELLABLE_RUN_STATES,
  CONTROL_CHAT_MESSAGEABLE_RUN_STATES,
  CONTROL_RESUMABLE_RUN_STATES,
  CONTROL_RUN_STATE_CLASSIFICATION,
  CONTROL_TERMINAL_RUN_STATES,
)
from agent_gateway.control_plane.runs_models import AutonomousRunState, ChatRunState


RISK_MODULE_ROOT = Path(
  os.environ.get("RISK_MODULE_ROOT", "/Users/henrychien/Documents/Jupyter/risk_module")
)
CONTRACT_SCHEMA_PATH = RISK_MODULE_ROOT / "docs" / "interfaces" / "agent-control-contracts.schema.json"


def _ordered_states(states: frozenset[str]) -> list[str]:
  return [state for state in CONTROL_RUN_STATE_CLASSIFICATION if state in states]


@pytest.mark.skipif(
  not CONTRACT_SCHEMA_PATH.exists(),
  reason="risk_module checked-in Agent Control contract schema bundle is unavailable",
)
def test_control_run_lifecycle_matches_risk_module_contract_bundle() -> None:
  bundle = json.loads(CONTRACT_SCHEMA_PATH.read_text(encoding="utf-8"))
  lifecycle = bundle["run_lifecycle"]

  assert lifecycle["states"] == list(CONTROL_RUN_STATE_CLASSIFICATION)
  assert lifecycle["classification"] == {
    state: dict(classification)
    for state, classification in CONTROL_RUN_STATE_CLASSIFICATION.items()
  }
  assert lifecycle["active_states"] == _ordered_states(CONTROL_ACTIVE_RUN_STATES)
  assert lifecycle["terminal_states"] == _ordered_states(CONTROL_TERMINAL_RUN_STATES)
  assert lifecycle["resumable_states"] == _ordered_states(CONTROL_RESUMABLE_RUN_STATES)
  assert lifecycle["cancellable_states"] == _ordered_states(CONTROL_CANCELLABLE_RUN_STATES)
  assert lifecycle["chat_messageable_states"] == _ordered_states(CONTROL_CHAT_MESSAGEABLE_RUN_STATES)


@pytest.mark.skipif(
  not CONTRACT_SCHEMA_PATH.exists(),
  reason="risk_module checked-in Agent Control contract schema bundle is unavailable",
)
def test_control_run_response_models_expose_only_contract_states() -> None:
  bundle = json.loads(CONTRACT_SCHEMA_PATH.read_text(encoding="utf-8"))
  contract_states = set(bundle["run_lifecycle"]["states"])

  assert set(get_args(ChatRunState)) == contract_states
  assert set(get_args(AutonomousRunState)) == contract_states
  assert "budget_limited" in contract_states
  assert "blocked" not in contract_states
  assert "remediating" not in contract_states


@pytest.mark.skipif(
  not CONTRACT_SCHEMA_PATH.exists(),
  reason="risk_module checked-in Agent Control contract schema bundle is unavailable",
)
def test_projected_control_event_shapes_are_covered_by_risk_module_contract_bundle() -> None:
  bundle = json.loads(CONTRACT_SCHEMA_PATH.read_text(encoding="utf-8"))
  models = bundle["models"]

  assert bundle["contract_versions"]["event"] == "control-event-v1"
  assert bundle["route_schemas"]["control_events"]["GET /control/events"] == {
    "projected_envelope": "ControlEventEnvelopeContract",
    "event_payload": "ControlEventPayloadContract",
    "known_event_payloads": [
      "RunStateChangedControlEventContract",
      "TextDeltaControlEventContract",
      "ParentMessageSentControlEventContract",
      "RunResumedControlEventContract",
      "RunResumedFromControlEventContract",
      "EventsDroppedControlEventContract",
      "ReplayTruncatedControlEventContract",
    ],
    "future_event_payload": "FutureControlEventContract",
  }
  assert models["RunStateChangedControlEventContract"]["properties"]["state"]["enum"] == list(
    CONTROL_RUN_STATE_CLASSIFICATION
  )
  assert {
    "resumed_run_id",
    "resumed_task_id",
    "request_id",
  }.issubset(models["RunResumedControlEventContract"]["properties"])
  assert {
    "resumed_from",
    "resumed_from_task_id",
    "request_id",
  }.issubset(models["RunResumedFromControlEventContract"]["properties"])
  assert "dropped_before_seq" in models["ReplayTruncatedControlEventContract"]["properties"]
  assert "dropped_through_seq" in models["EventsDroppedControlEventContract"]["properties"]
