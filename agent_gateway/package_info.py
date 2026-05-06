from __future__ import annotations

import os
from importlib import metadata
from typing import Any

PACKAGE_NAME = "ai-agent-gateway"
CONTRACT_CREDENTIAL_REFRESH_V1 = "credential-refresh-v1"
CONTRACTS = frozenset({CONTRACT_CREDENTIAL_REFRESH_V1})


def _package_version() -> str:
  try:
    return metadata.version(PACKAGE_NAME)
  except metadata.PackageNotFoundError:
    return "0+unknown"


__version__ = _package_version()
SOURCE_COMMIT = os.getenv("AGENT_GATEWAY_SOURCE_COMMIT", "").strip() or None


def package_health() -> dict[str, Any]:
  return {
    "name": PACKAGE_NAME,
    "version": __version__,
    "source_commit": SOURCE_COMMIT,
    "contracts": sorted(CONTRACTS),
  }


__all__ = [
  "CONTRACTS",
  "CONTRACT_CREDENTIAL_REFRESH_V1",
  "PACKAGE_NAME",
  "SOURCE_COMMIT",
  "__version__",
  "package_health",
]
