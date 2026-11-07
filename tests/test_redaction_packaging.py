from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

def _offline_wheel_python(tmp_path: Path) -> Path:
  """A pip that builds these wheels without an index.

  Build isolation fetches setuptools. The EC2 gate host has no index: `pip wheel`
  on the base interpreter exited 1 after about six minutes, and the JUnit did not
  keep the captured stderr. A fresh venv has no checkout pip.conf. The caller's
  setuptools and wheel are copied in, as the model-authority smoke test does,
  and the build uses `--no-build-isolation`.
  """
  venv = tmp_path / "build-venv"
  created = subprocess.run(
    [sys.executable, "-m", "venv", str(venv)],
    check=False, capture_output=True, text=True,
  )
  assert created.returncode == 0, created.stderr
  python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
  site = Path(subprocess.run(
    [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
    check=True, capture_output=True, text=True,
  ).stdout.strip())
  for name in ("setuptools", "wheel"):
    distribution = importlib.metadata.distribution(name)
    for file in distribution.files or ():
      if not file.parts or file.parts[0] == "..":
        continue
      source = Path(str(distribution.locate_file(file)))
      if not source.is_file():
        continue
      target = site / file
      target.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(source, target)
  return python


def _wheel_env(tmp_path: Path) -> dict[str, str]:
  environment = {
    name: value for name, value in os.environ.items()
    if name != "PYTHONPATH" and not name.startswith("PIP_")
  }
  environment["PIP_NO_INDEX"] = "1"
  environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
  environment["PIP_CACHE_DIR"] = str(tmp_path / "pip-cache")
  return environment


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
  builder_python = _offline_wheel_python(tmp_path)
  build_env = _wheel_env(tmp_path)
  built = subprocess.run(
    [
      str(builder_python),
      "-I",
      "-m",
      "pip",
      "wheel",
      "--no-deps",
      "--no-build-isolation",
      "--wheel-dir",
      str(wheel_dir),
      str(package_root),
      str(contracts_root),
    ],
    check=False,
    env=build_env,
    capture_output=True,
    text=True,
  )
  assert built.returncode == 0, built.stderr
  wheel = next(wheel_dir.glob("ai_agent_gateway-*.whl"))
  contracts_wheel = next(wheel_dir.glob("agent_workflow_contracts-*.whl"))
  with ZipFile(wheel) as archive:
    assert not any(
      name.startswith("agent_workflow_contracts/") for name in archive.namelist()
    )
  installed = tmp_path / "installed"
  installed_wheel = subprocess.run(
    [
      str(builder_python),
      "-I",
      "-m",
      "pip",
      "install",
      "--no-deps",
      "--no-index",
      "--disable-pip-version-check",
      "--target",
      str(installed),
      str(wheel),
      str(contracts_wheel),
    ],
    check=False,
    env=build_env,
    capture_output=True,
    text=True,
  )
  assert installed_wheel.returncode == 0, installed_wheel.stderr

  probe = Path(__file__).parent / "fixtures" / "standalone_redaction_probe.py"
  isolated_env = os.environ.copy()
  isolated_env.pop("PYTHONPATH", None)
  completed = subprocess.run(
    [sys.executable, "-I", str(probe), str(installed), str(tmp_path / "state")],
    cwd=tmp_path,
    env=isolated_env,
    check=False,
    capture_output=True,
    text=True,
  )
  assert completed.returncode == 0, completed.stderr
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
