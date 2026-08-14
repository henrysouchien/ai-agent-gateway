from __future__ import annotations

import os
import asyncio
from pathlib import Path

import pytest

from agent_gateway.canvas_artifact_pipeline import emit_canvas_artifact
from agent_gateway.canvas_build_environment import (
  CanvasBuildFailure, CanvasBuildPreflight, build_canvas_bundle_async,
  check_module_policy, packaged_build_directory,
  preflight_canvas_build_environment, sanitized_subprocess_env,
)
from agent_gateway.canvas_kit_contract import packaged_contract_directory


def _preflight_or_skip():
  build_dir = packaged_build_directory()
  try:
    value = preflight_canvas_build_environment({
      **os.environ, "CANVAS_BUILD_DIR": str(build_dir),
    })
  except RuntimeError as exc:
    pytest.skip(f"canvas_toolchain_unavailable: {exc}")
  assert value is not None
  return value


def test_module_policy_rejects_contract_fixtures_with_exact_codes() -> None:
  fixtures = packaged_contract_directory() / "fixtures"
  for name, code in (
    ("forbidden-import.tsx", "import_allowlist"),
    ("module-scope-side-effect.tsx", "module_scope_side_effect"),
    ("forbidden-identifier.tsx", "forbidden_identifier"),
  ):
    with pytest.raises(CanvasBuildFailure) as caught:
      check_module_policy((fixtures / name).read_text())
    assert caught.value.stage == "module_policy"
    assert caught.value.diagnostics[0]["code"] == code
    assert {"line", "column", "code", "message", "repair_hint"} <= set(caught.value.diagnostics[0])


def test_env_sanitization_strips_node_and_npm_injection() -> None:
  clean = sanitized_subprocess_env({
    "PATH": "/bin", "LANG": "C", "NODE_PATH": "/evil",
    "NODE_OPTIONS": "--require=/evil", "npm_config_registry": "https://evil",
    "npm_config_cache": "/evil", "SECRET": "nope",
  })
  assert clean["PATH"] == "/bin"
  assert clean["NO_PROXY"] == clean["no_proxy"] == "*"
  assert not ({"NODE_PATH", "NODE_OPTIONS", "npm_config_registry", "npm_config_cache", "SECRET"} & set(clean))


def test_valid_contract_fixture_bundles_deterministically_without_bare_imports() -> None:
  preflight = _preflight_or_skip()
  source = (packaged_contract_directory() / "fixtures" / "valid-canvas.tsx").read_text()
  from agent_gateway.canvas_build_environment import build_canvas_bundle
  first = build_canvas_bundle(source, preflight)
  second = build_canvas_bundle(source, preflight)
  assert first == second
  assert b"HankCanvasRuntime.register" in first
  assert b"require(" not in first and b" from \"" not in first


@pytest.mark.parametrize(
  ("fixture", "stage", "code"),
  [
    ("type-error.tsx", "typecheck", "TS2322"),
    ("oversize-source.tsx", "size_cap", "source_size_cap_exceeded"),
    ("oversize-bundle.tsx", "bundle_size_cap", "bundle_size_cap_exceeded"),
  ],
)
def test_contract_negative_fixtures_hit_exact_first_failure(tmp_path: Path, fixture: str, stage: str, code: str) -> None:
  preflight = _preflight_or_skip() if stage in {"typecheck", "bundle_size_cap"} else object()
  source = (packaged_contract_directory() / "fixtures" / fixture).read_text()
  result = emit_canvas_artifact(
    workspace_dir=tmp_path, preflight=preflight, title="Fixture", purpose="exploration",
    summary="Fixture", tsx_source=source, copy_as_markdown="fallback",
    source_skill="fixture", skill_run_id="fixture-run",
  )
  failure = result["validation_failed"]
  assert failure["stage"] == stage
  assert failure["diagnostics"][0]["code"] == code


def test_valid_fixture_writes_sidecar_and_emits_event(tmp_path: Path) -> None:
  preflight = _preflight_or_skip()
  events = []
  source = (packaged_contract_directory() / "fixtures" / "valid-canvas.tsx").read_text()
  result = emit_canvas_artifact(
    workspace_dir=tmp_path, preflight=preflight, title="Valid", purpose="exploration",
    summary="Valid fixture", tsx_source=source, copy_as_markdown="fallback",
    source_skill="fixture", skill_run_id="fixture-run", emit_event=events.append,
  )
  assert result["status"] == "ok"
  assert events[0]["skill"] == "_canvas"
  assert events[0]["contract_name"] == "CanvasArtifact"
  assert (tmp_path / result["artifact_path"]).is_file()


def test_subprocess_cancellation_cleans_emission_tempdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  fake_node = tmp_path / "slow-node"
  fake_node.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
  fake_node.chmod(0o755)
  monkeypatch.setenv("TMPDIR", str(tmp_path))
  preflight = CanvasBuildPreflight(
    build_dir=tmp_path, node=fake_node, tsc=fake_node, esbuild=fake_node,
    toolchain_version="test",
  )

  async def scenario() -> None:
    task = asyncio.create_task(build_canvas_bundle_async(
      "import React from 'react'; export default function Canvas(){return <div/>;}",
      preflight,
    ))
    for _ in range(100):
      if list(tmp_path.glob("hank-canvas-build-*")): break
      await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await task

  asyncio.run(scenario())
  assert list(tmp_path.glob("hank-canvas-build-*")) == []
