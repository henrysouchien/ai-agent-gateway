from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
import subprocess
import sys
from typing import get_args

from agent_gateway.control_run_lifecycle import (
  CONTROL_RUN_CONTRACT_VERSION,
  CONTROL_RUN_STATE_CLASSIFICATION,
  ControlRunState,
)
from agent_gateway.control_run_lifecycle_contract import (
  control_run_lifecycle_contract_payload,
  render_control_run_lifecycle_contract,
)
from agent_gateway.control_plane.runs_models import AutonomousRunState, ChatRunState


ROOT = Path(__file__).resolve().parents[4]
CONTRACT_RESOURCE = resources.files("agent_gateway").joinpath(
  "contracts/control-run-v1/control_run_lifecycle.json"
)


def test_checked_in_control_run_lifecycle_contract_has_no_generation_drift() -> None:
  result = subprocess.run(
    [
      sys.executable,
      str(ROOT / "scripts" / "generate_control_run_lifecycle_contract.py"),
      "--check",
    ],
    cwd=ROOT,
    text=True,
    capture_output=True,
    check=False,
  )

  assert result.returncode == 0, result.stderr


def test_control_run_lifecycle_contract_check_fails_on_local_drift(
  tmp_path: Path,
) -> None:
  stale_output = tmp_path / "control_run_lifecycle.json"
  stale_output.write_text("{}\n", encoding="utf-8")

  result = subprocess.run(
    [
      sys.executable,
      str(ROOT / "scripts" / "generate_control_run_lifecycle_contract.py"),
      "--check",
      "--output",
      str(stale_output),
    ],
    cwd=ROOT,
    text=True,
    capture_output=True,
    check=False,
  )

  assert result.returncode == 1
  assert "stale control-run lifecycle contract" in result.stderr


def test_packaged_control_run_lifecycle_contract_matches_python_owner() -> None:
  resource_text = CONTRACT_RESOURCE.read_text(encoding="utf-8")
  payload = json.loads(resource_text)

  assert resource_text == render_control_run_lifecycle_contract()
  assert payload == control_run_lifecycle_contract_payload()
  assert payload["contract_version"] == CONTROL_RUN_CONTRACT_VERSION


def test_control_run_response_literals_match_python_owner() -> None:
  contract_states = set(CONTROL_RUN_STATE_CLASSIFICATION)

  assert set(get_args(ControlRunState)) == contract_states
  assert set(get_args(ChatRunState)) == contract_states
  assert set(get_args(AutonomousRunState)) == contract_states
  assert "budget_limited" in contract_states
  assert "blocked" not in contract_states
  assert "remediating" not in contract_states
