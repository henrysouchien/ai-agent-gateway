from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile


def test_installed_wheel_runs_with_packaged_redaction_and_raw_schema_boundary(
  tmp_path: Path,
) -> None:
  source_root = Path(__file__).resolve().parents[1]
  package_root = tmp_path / "agent-gateway"
  package_root.mkdir()
  for filename in ("pyproject.toml", "README.md", "LICENSE"):
    shutil.copy2(source_root / filename, package_root / filename)
  shutil.copytree(
    source_root / "agent_gateway",
    package_root / "agent_gateway",
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
  )
  contracts_source = source_root.parent / "agent-workflow-contracts"
  contracts_root = tmp_path / "agent-workflow-contracts"
  contracts_root.mkdir()
  shutil.copy2(contracts_source / "pyproject.toml", contracts_root / "pyproject.toml")
  shutil.copytree(
    contracts_source / "agent_workflow_contracts",
    contracts_root / "agent_workflow_contracts",
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
  )
  wheel_dir = tmp_path / "wheel"
  wheel_dir.mkdir()
  build_env = os.environ.copy()
  build_env["PIP_CACHE_DIR"] = str(tmp_path / "pip-cache")
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
      str(contracts_root),
    ],
    check=True,
    env=build_env,
    capture_output=True,
    text=True,
  )
  wheel = next(wheel_dir.glob("ai_agent_gateway-*.whl"))
  contracts_wheel = next(wheel_dir.glob("agent_workflow_contracts-*.whl"))
  with ZipFile(wheel) as archive:
    assert not any(
      name.startswith("agent_workflow_contracts/") for name in archive.namelist()
    )
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
      str(contracts_wheel),
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
  assert observed["packaged_handler_exact"] is True
  assert observed["packaged_history_has_fields"] is True
  assert observed["packaged_surface_has_redaction"] is True
  assert observed["packaged_surface_has_secret"] is False
  assert observed["packaged_surface_has_tombstone"] is False
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
