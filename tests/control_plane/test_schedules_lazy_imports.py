from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_schedules_module_imports_with_stripped_gateway_pythonpath() -> None:
  repo_root = Path(__file__).resolve().parents[4]
  env = os.environ.copy()
  env["PYTHONPATH"] = os.pathsep.join(
    [
      str(repo_root / "api"),
      str(repo_root / "packages" / "agent-gateway"),
    ]
  )

  completed = subprocess.run(
    [sys.executable, "-c", "import agent_gateway.control_plane.schedules; print('OK')"],
    cwd="/tmp",
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=30,
  )

  assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
  assert completed.stdout.strip() == "OK"
  assert "ModuleNotFoundError" not in completed.stderr
