"""Fixtures for the outbound-send production gate (``agent_gateway.production_identity``).

Both conftests (``tests/`` and ``packages/agent-gateway/tests/``) import these, so every
test starts as a production-runtime process and sender tests run. A test of the hold asks
for ``dev_runtime``, which puts the code file back where a checkout run has it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _production_send_authority(monkeypatch):
  """Default every test to a production-runtime layout so send paths run.

  The guard reads only the code file's location and the login home; this fixture
  moves both inputs under a throwaway shared runtime root.
  """
  from agent_gateway import production_identity

  root = Path(os.getenv("TMPDIR", "/tmp")) / "ai-excel-addin-test-production-runtime"
  monkeypatch.setattr(production_identity, "_shared_runtime_root", lambda: root)
  monkeypatch.setattr(
    production_identity,
    "_CODE_FILE",
    root
    / "versions"
    / "ai-test"
    / "ai-excel-addin"
    / "packages"
    / "agent-gateway"
    / "agent_gateway"
    / "production_identity.py",
  )
  return root


@pytest.fixture
def dev_runtime(_production_send_authority, monkeypatch):
  """Run the test as a checkout or isolated-instance process (not production)."""
  from agent_gateway import production_identity

  monkeypatch.setattr(production_identity, "_CODE_FILE", Path(production_identity.__file__).resolve())
  return _production_send_authority
