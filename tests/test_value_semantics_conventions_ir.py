from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[3]
PACKAGE_SOURCE = ROOT / "packages" / "value-semantics-core"


@pytest.fixture
def vendored_package(tmp_path: Path) -> tuple[Path, Path]:
  vendor_root = tmp_path / "vendored" / "value-semantics-core"
  shutil.copytree(PACKAGE_SOURCE, vendor_root)
  run_dir = tmp_path / "isolated-cwd"
  run_dir.mkdir()
  return vendor_root, run_dir


def _subprocess_env(hash_seed: str | None = None) -> dict[str, str]:
  env = os.environ.copy()
  env.pop("PYTHONPATH", None)
  env["PYTHONNOUSERSITE"] = "1"
  if hash_seed is not None:
    env["PYTHONHASHSEED"] = hash_seed
  return env


def test_vendored_package_imports_with_ai_repository_absent(
  vendored_package: tuple[Path, Path],
) -> None:
  vendor_root, run_dir = vendored_package
  script = textwrap.dedent(
    """
    import importlib.util
    from pathlib import Path
    import sys

    vendor_root = Path(sys.argv[1]).resolve()
    repository_root = Path(sys.argv[2]).resolve()
    sys.path.insert(0, str(vendor_root))

    for entry in sys.path:
      if not entry:
        raise SystemExit("empty sys.path entry would expose the subprocess CWD")
      resolved = Path(entry).resolve()
      if resolved == repository_root or repository_root in resolved.parents:
        raise SystemExit(f"AI repository leaked onto sys.path: {resolved}")

    if importlib.util.find_spec("schema") is not None:
      raise SystemExit("AI-repository schema package unexpectedly importable")

    from value_semantics_core.conventions_ir_v1 import CONVENTIONS_IR_V1

    assert CONVENTIONS_IR_V1.numeric_scales[-1] == "billions"
    assert Path(sys.modules["value_semantics_core"].__file__).resolve().is_relative_to(
      vendor_root
    )
    """
  )

  result = subprocess.run(
    [
      sys.executable,
      "-I",
      "-c",
      script,
      str(vendor_root),
      str(ROOT),
    ],
    cwd=run_dir,
    env=_subprocess_env(),
    capture_output=True,
    text=True,
    check=False,
  )

  assert result.returncode == 0, result.stderr


def test_ir_canonical_json_is_hash_seed_independent(
  vendored_package: tuple[Path, Path],
) -> None:
  vendor_root, run_dir = vendored_package
  script = textwrap.dedent(
    """
    from dataclasses import asdict
    import json
    from pathlib import Path
    import sys

    vendor_root = Path(sys.argv[1]).resolve()
    repository_root = Path(sys.argv[2]).resolve()
    run_dir = Path.cwd().resolve()
    retained_stdlib_paths = []
    for entry in sys.path:
      if not entry:
        continue
      resolved = Path(entry).resolve()
      if resolved == repository_root or repository_root in resolved.parents:
        continue
      if resolved == run_dir or run_dir in resolved.parents:
        continue
      retained_stdlib_paths.append(entry)
    sys.path[:] = [str(vendor_root), *retained_stdlib_paths]

    from value_semantics_core.conventions_ir_v1 import CONVENTIONS_IR_V1

    sys.stdout.write(
      json.dumps(
        asdict(CONVENTIONS_IR_V1),
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
      )
    )
    """
  )

  outputs: list[bytes] = []
  for seed in ("0", "1", "42", "random"):
    result = subprocess.run(
      [sys.executable, "-c", script, str(vendor_root), str(ROOT)],
      cwd=run_dir,
      env=_subprocess_env(seed),
      capture_output=True,
      check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    outputs.append(result.stdout)

  assert outputs
  assert all(output == outputs[0] for output in outputs[1:])
