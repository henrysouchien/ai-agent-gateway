from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile


def test_canvas_resources_resolve_from_installed_wheel(tmp_path: Path) -> None:
  package_root = Path(__file__).resolve().parents[1]
  wheel_dir = tmp_path / "wheel"
  wheel_dir.mkdir()
  subprocess.run(
    [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheel_dir), str(package_root)],
    check=True, capture_output=True, text=True,
  )
  wheel = next(wheel_dir.glob("ai_agent_gateway-*.whl"))
  with ZipFile(wheel) as archive:
    names = set(archive.namelist())
  required = {
    "agent_gateway/contracts/canvas-kit-v1/canvas_kit_manifest.v1.json",
    "agent_gateway/contracts/canvas-kit-v1/types/node_modules/@hank/canvas-kit/index.d.ts",
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
