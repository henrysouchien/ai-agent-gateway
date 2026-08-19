from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def test_installed_wheel_runs_with_packaged_redaction_and_raw_schema_boundary(
  tmp_path: Path,
) -> None:
  package_root = Path(__file__).resolve().parents[1]
  wheel_dir = tmp_path / "wheel"
  wheel_dir.mkdir()
  builder_python = Path(sys.base_prefix) / (
    "python.exe" if os.name == "nt" else "bin/python3"
  )
  subprocess.run(
    [
      str(builder_python),
      "-m",
      "pip",
      "wheel",
      "--no-deps",
      "--wheel-dir",
      str(wheel_dir),
      str(package_root),
    ],
    check=True,
    capture_output=True,
    text=True,
  )
  wheel = next(wheel_dir.glob("ai_agent_gateway-*.whl"))
  installed = tmp_path / "installed"
  subprocess.run(
    [
      str(builder_python),
      "-m",
      "pip",
      "install",
      "--no-deps",
      "--disable-pip-version-check",
      "--target",
      str(installed),
      str(wheel),
    ],
    check=True,
    capture_output=True,
    text=True,
  )

  probe = Path(__file__).parent / "fixtures" / "standalone_redaction_probe.py"
  isolated_env = os.environ.copy()
  isolated_env.pop("PYTHONPATH", None)
  completed = subprocess.run(
    [sys.executable, "-I", str(probe), str(installed), str(tmp_path / "state")],
    cwd=tmp_path,
    env=isolated_env,
    check=True,
    capture_output=True,
    text=True,
  )
  observed = json.loads(completed.stdout)

  assert Path(observed["agent_gateway_file"]).is_relative_to(installed)
  assert observed["fallback_redactor_module"] == "agent_gateway.tool_redaction"
  assert observed["fallback_handler_exact"] is True
  assert observed["fallback_history_has_fields"] is True
  assert observed["fallback_surface_has_redaction"] is True
  assert observed["fallback_surface_has_secret"] is False
  assert observed["fallback_surface_has_tombstone"] is False
  assert observed["logs_have_secret"] is False
  assert observed["broken_valid_handler_exact"] is True
  assert observed["broken_valid_surface_has_tombstone"] is True
  assert observed["broken_surface_has_secret"] is False
  assert observed["broken_malformed_handler_count"] == 0
  assert len(observed["broken_malformed_validation_errors"]) == 1
  assert observed["broken_malformed_validation_errors"][0]["code"] == (
    "invalid_tool_input_schema"
  )
  assert observed["broken_malformed_validation_errors"][0]["details"] == {
    "missing": ["symbol", "start_date", "end_date"],
    "tool_name": "data_historical_prices",
    "type_errors": [],
    "unexpected": [],
  }
