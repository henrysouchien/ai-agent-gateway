from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

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
