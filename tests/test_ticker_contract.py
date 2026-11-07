from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from agent_gateway.artifact_paths import (
  ArtifactPathError,
  normalize_ticker_for_artifact_request,
  ticker_artifact_paths_for_request,
)
from agent_workflow_contracts.ticker_contract import (
  TICKER_CONTRACT_SEMANTIC_MANIFEST,
  TICKER_INPUT_CONTRACT,
  TICKER_NORMALIZATION_ALGORITHM_ID,
  normalize_contract_ticker,
)


def test_packaged_contract_owns_canonical_ticker_and_typed_identity() -> None:
  assert normalize_contract_ticker(" brk-b ") == "BRKB"
  assert TICKER_INPUT_CONTRACT.namespace == "workflow"
  assert TICKER_INPUT_CONTRACT.name == "ticker"
  assert TICKER_INPUT_CONTRACT.version == "1.0"
  assert TICKER_INPUT_CONTRACT.digest == (
    "sha256:34000d3fc4e0ad11f187f6e2b1e89ff4fa0e95da664311bc9cce5210825ae831"
  )
  assert dict(TICKER_CONTRACT_SEMANTIC_MANIFEST)["normalizer"] == (
    TICKER_NORMALIZATION_ALGORITHM_ID
  )
  assert "." not in TICKER_NORMALIZATION_ALGORITHM_ID


def test_standalone_gateway_import_does_not_require_repo_schema(
  tmp_path: Path,
) -> None:
  package_root = Path(__file__).resolve().parents[1]
  env = dict(os.environ)
  env["PYTHONPATH"] = str(package_root)
  result = subprocess.run(
    [
      sys.executable,
      "-c",
      (
        "import builtins\n"
        "real_import = builtins.__import__\n"
        "def guarded(name, globals=None, locals=None, fromlist=(), level=0):\n"
        "    if level == 0 and (name == 'schema' or name.startswith('schema.')):\n"
        "        raise ImportError('repo schema forbidden')\n"
        "    return real_import(name, globals, locals, fromlist, level)\n"
        "builtins.__import__ = guarded\n"
        "import agent_gateway.sub_agent\n"
        "from agent_workflow_contracts.ticker_contract import normalize_contract_ticker\n"
        "assert normalize_contract_ticker('QCOM') == 'QCOM'\n"
      ),
    ],
    cwd=tmp_path,
    env=env,
    capture_output=True,
    text=True,
    check=False,
    timeout=60,
  )
  assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
  ("raw", "expected"),
  (
    ("NYSE:AAPL", "AAPL"),
    ("NASDAQ:MSFT", "MSFT"),
    ("NYSE:BRK.B", "BRKB"),
    ("AAPL", "AAPL"),
    ("TAEE11.SA", "TAEE11"),
    ("0700.HK", "0700"),
  ),
)
def test_gateway_reader_ticker_agrees_with_contract_normalizer(raw: str, expected: str) -> None:
  assert normalize_ticker_for_artifact_request(raw) == expected
  assert normalize_contract_ticker(raw) == expected


@pytest.mark.parametrize("raw", ("A..B", "../X", "A/B", "EFC-PC", "ABCDEFGHIJKLMNOP"))
def test_gateway_reader_ticker_refuses_what_the_contract_refuses(raw: str) -> None:
  with pytest.raises(ArtifactPathError, match="invalid ticker path component"):
    normalize_ticker_for_artifact_request(raw)


def test_gateway_exchange_prefixed_ticker_reads_the_unprefixed_artifact_directory(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(
    "agent_gateway.artifact_paths.user_workspace_root", lambda user_id: tmp_path.resolve()
  )
  artifact_dir = tmp_path / "artifacts" / "AAPL" / "earnings-preview"
  artifact_dir.mkdir(parents=True)
  (artifact_dir / "a1.json").write_text("{}", encoding="utf-8")

  prefixed = ticker_artifact_paths_for_request("1", ticker="NYSE:AAPL")
  bare = ticker_artifact_paths_for_request("1", ticker="AAPL")

  assert prefixed == bare
  assert prefixed


@pytest.mark.parametrize("raw", ("BVMF:TAEE11.SA", "TAEE11.SA", "SEHK:0700.HK", "NYSE:BRK.B"))
def test_gateway_raw_and_normalized_forms_resolve_same_artifact(
  raw: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  from agent_gateway.artifact_paths import artifact_json_path_for_request

  monkeypatch.setattr("agent_gateway.artifact_paths.user_workspace_root", lambda _: tmp_path)
  normalized = normalize_contract_ticker(raw)
  path = tmp_path / "artifacts" / normalized / "earnings-preview" / "a1.json"
  path.parent.mkdir(parents=True)
  path.write_text('{"retained": true}', encoding="utf-8")
  for requested in (raw, normalized):
    resolved = artifact_json_path_for_request(
      "1", ticker=requested, skill="earnings-preview", artifact_id="a1",
    )
    assert resolved.path == path
    assert resolved.path.read_text(encoding="utf-8") == '{"retained": true}'
    assert ticker_artifact_paths_for_request("1", ticker=requested)["earnings-preview"][0].path == path
