from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import threading

import pytest

from agent_gateway import canvas_build_environment as canvas


def _provision_script() -> str:
  return (
    canvas.packaged_build_directory() / "provision.sh"
  ).read_text(encoding="utf-8")


def _external_runtime(tmp_path: Path) -> Path:
  runtime = tmp_path / "canvas-build"
  runtime.mkdir()
  packaged = canvas.packaged_build_directory()
  for relative_path in canvas.CANVAS_RUNTIME_SUPPORT_FILES:
    shutil.copyfile(packaged / relative_path, runtime / relative_path)
  tools = runtime / "node_modules" / ".bin"
  tools.mkdir(parents=True)
  for name in ("tsc", "esbuild"):
    path = tools / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
  return runtime


def test_external_runtime_support_files_are_verified_before_tool_versions(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  runtime = _external_runtime(tmp_path)
  expected_node = (
    canvas.packaged_build_directory() / ".node-version"
  ).read_text(encoding="utf-8").strip()
  monkeypatch.setattr(
    canvas.canvas_kit_contract,
    "pinned_versions",
    lambda: {"typescript": "fixture-tsc", "esbuild": "fixture-esbuild"},
  )
  monkeypatch.setattr(
    canvas,
    "_run_version",
    lambda argv, **_kwargs: (
      f"v{expected_node}"
      if argv[-1] == "--version" and Path(argv[0]).name == "node"
      else "Version fixture-tsc"
      if Path(argv[0]).name == "tsc"
      else "fixture-esbuild"
    ),
  )

  preflight = canvas.preflight_canvas_build_environment({
    "CANVAS_BUILD_DIR": str(runtime),
    "CANVAS_NODE_BINARY": "node",
  })

  assert preflight is not None
  assert preflight.build_dir == runtime.resolve()


def test_preflight_is_single_flight_and_cached_per_process_configuration(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  runtime = _external_runtime(tmp_path)
  expected_node = (
    canvas.packaged_build_directory() / ".node-version"
  ).read_text(encoding="utf-8").strip()
  expected_versions = canvas.canvas_kit_contract.pinned_versions()
  first_probe_started = threading.Event()
  release_first_probe = threading.Event()
  calls: list[tuple[str, ...]] = []

  def fake_run_version(argv: list[str], *, env: dict[str, str]) -> str:
    del env
    calls.append(tuple(argv))
    if len(calls) == 1:
      first_probe_started.set()
      assert release_first_probe.wait(timeout=5)
    name = Path(argv[0]).name
    if name not in {"tsc", "esbuild"}:
      return f"v{expected_node}"
    if name == "tsc":
      return f"Version {expected_versions['typescript']}"
    return expected_versions["esbuild"]

  monkeypatch.setattr(canvas, "_run_version", fake_run_version)
  env = {
    "CANVAS_BUILD_DIR": str(runtime),
    "CANVAS_NODE_BINARY": "node",
  }
  with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [
      executor.submit(canvas.preflight_canvas_build_environment, env)
      for _ in range(8)
    ]
    assert first_probe_started.wait(timeout=5)
    release_first_probe.set()
    results = [future.result(timeout=5) for future in futures]

  assert len(calls) == 3
  assert results[0] is not None
  assert all(result is results[0] for result in results)

  changed_config_result = canvas.preflight_canvas_build_environment({
    **env,
    "CANVAS_NODE_BINARY": "/alternate-canvas-runtime/node",
  })

  assert len(calls) == 6
  assert changed_config_result is not None
  assert changed_config_result is not results[0]


@pytest.mark.parametrize("relative_path", canvas.CANVAS_RUNTIME_SUPPORT_FILES)
def test_external_runtime_rejects_support_file_drift(
  tmp_path: Path,
  relative_path: str,
) -> None:
  runtime = _external_runtime(tmp_path)
  (runtime / relative_path).write_bytes(b"tampered\n")

  with pytest.raises(
    RuntimeError,
    match=rf"{relative_path.replace('.', r'\.')}: differs from packaged canonical file",
  ):
    canvas.preflight_canvas_build_environment({
      "CANVAS_BUILD_DIR": str(runtime),
      "CANVAS_NODE_BINARY": "node",
    })


def test_build_executes_policy_and_bundle_from_preflight_runtime(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  runtime = _external_runtime(tmp_path)
  commands: list[list[str]] = []

  def fake_run(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stage: str,
  ) -> subprocess.CompletedProcess[bytes]:
    del env
    commands.append(argv)
    if stage == "bundle":
      Path(argv[-2]).write_bytes(b"bundle")
    return subprocess.CompletedProcess(argv, 0, b"", b"")

  monkeypatch.setattr(canvas, "_run_build_command", fake_run)
  preflight = canvas.CanvasBuildPreflight(
    build_dir=runtime,
    node=Path("node"),
    tsc=runtime / "node_modules" / ".bin" / "tsc",
    esbuild=runtime / "node_modules" / ".bin" / "esbuild",
    toolchain_version="fixture",
  )

  result = canvas.build_canvas_bundle(
    "import React from 'react'; export default function Canvas(){return <div/>;}",
    preflight,
  )

  assert result == b"bundle"
  assert commands[0][1] == str(runtime / "policy.mjs")
  assert commands[-1][1] == str(runtime / "build.mjs")


def test_typecheck_diagnostic_names_expected_component_prop_shape() -> None:
  source = """import { SectionHeader } from '@hank/canvas-kit';
export default function Example() {
  return <SectionHeader>Wrong child shape</SectionHeader>;
}
"""
  raw = (
    "source.tsx(3,11): error TS2322: Type '{ children: string; }' "
    "is not assignable to type '{ title: string; }'."
  )

  diagnostics = canvas._typecheck_diagnostics(raw, source)

  assert diagnostics[0]["code"] == "TS2322"
  assert "Expected SectionHeader props:" in diagnostics[0]["repair_hint"]
  assert "title: string" in diagnostics[0]["repair_hint"]
  assert "children" not in diagnostics[0]["repair_hint"].split(
    "Expected SectionHeader props:", 1
  )[1]


def test_provision_bounds_node_download_and_replaces_the_pinned_prefix() -> None:
  script = _provision_script()

  for option in (
    "--connect-timeout 10",
    "--max-time 120",
    "--speed-limit 1024",
    "--speed-time 30",
    "--retry 3",
    "--retry-delay 2",
    "--retry-max-time 180",
    "--retry-connrefused",
  ):
    assert option in script

  stage = (
    'NODE_STAGE="$(mktemp -d '
    '"$INSTALL_PARENT/.$EXPECTED_INSTALL_BASENAME.new.XXXXXXXX")"'
  )
  extract = (
    'tar -xJf "$DOWNLOAD_DIR/$ARCHIVE" '
    '--strip-components=1 -C "$NODE_STAGE"'
  )
  remove = 'rm -rf -- "$INSTALL_PREFIX"'
  promote = 'mv "$NODE_STAGE" "$INSTALL_PREFIX"'
  assert stage in script
  assert extract in script
  assert remove in script
  assert promote in script
  assert script.index(extract) < script.index(remove) < script.index(promote)
  assert '-C "$INSTALL_PREFIX"' not in script
