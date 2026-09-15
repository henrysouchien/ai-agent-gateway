from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile


def test_canvas_resources_resolve_from_installed_wheel(tmp_path: Path) -> None:
  source_root = Path(__file__).resolve().parents[1]
  package_root = tmp_path / "agent-gateway"
  package_root.mkdir()
  for filename in ("pyproject.toml", "README.md", "LICENSE"):
    shutil.copy2(source_root / filename, package_root / filename)
  for package_name in ("agent_gateway", "agent_workflow_contracts"):
    shutil.copytree(
      source_root / package_name,
      package_root / package_name,
      ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
  wheel_dir = tmp_path / "wheel"
  wheel_dir.mkdir()
  build_env = os.environ.copy()
  build_env["PIP_CACHE_DIR"] = str(tmp_path / "pip-cache")
  subprocess.run(
    [
      sys.executable,
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
    env=build_env,
    text=True,
  )
  wheel = next(wheel_dir.glob("ai_agent_gateway-*.whl"))
  with ZipFile(wheel) as archive:
    names = set(archive.namelist())
  required = {
    "agent_gateway/contracts/canvas-kit-v1/canvas_kit_manifest.v1.json",
    "agent_gateway/contracts/canvas-kit-v1/types/node_modules/@hank/canvas-kit/index.d.ts",
    "agent_gateway/contracts/canvas-kit-v1/types/node_modules/@hank/canvas-kit/components/index.d.ts",
    "agent_gateway/contracts/canvas-kit-v1/types/node_modules/@hank/canvas-kit/fmt.d.ts",
    "agent_gateway/canvas_build/.node-version",
    "agent_gateway/canvas_build/package.json",
    "agent_gateway/canvas_build/package-lock.json",
    "agent_gateway/canvas_build/node_checksums.json",
    "agent_gateway/canvas_build/build.mjs",
    "agent_gateway/canvas_build/policy.mjs",
  }
  assert required <= names
  assert "agent_gateway/contracts/canvas-kit-v1/digest.json" not in names
  assert not any(name.startswith("agent_gateway/canvas_build/node_modules/") for name in names)
