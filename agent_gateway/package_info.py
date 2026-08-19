from __future__ import annotations

import os
from importlib import metadata
from typing import Any

from .control_run_lifecycle import CONTROL_RUN_CONTRACT_VERSION


PACKAGE_NAME = "ai-agent-gateway"
CONTRACT_CREDENTIAL_REFRESH_V1 = "credential-refresh-v1"
CONTRACT_AUTONOMOUS_OPERATOR_MESSAGES_V1 = "autonomous-operator-messages-v1"
CONTRACT_CONTROL_CHAT_CONTINUATION_V1 = "control-chat-continuation-v1"
CONTRACT_CONTROL_RUN_V1 = CONTROL_RUN_CONTRACT_VERSION
CONTRACT_CHAT_ATTACHMENTS_V1 = "chat-attachments-v1"
CONTRACT_INVESTMENT_SELECTED_CONTENT_V1 = "investment-selected-content-v1"
CONTRACTS = frozenset({
  CONTRACT_CREDENTIAL_REFRESH_V1,
  CONTRACT_AUTONOMOUS_OPERATOR_MESSAGES_V1,
  CONTRACT_CONTROL_CHAT_CONTINUATION_V1,
  CONTRACT_CONTROL_RUN_V1,
})


def _package_version() -> str:
  try:
    return metadata.version(PACKAGE_NAME)
  except metadata.PackageNotFoundError:
    return "0+unknown"


__version__ = _package_version()
SOURCE_COMMIT = os.getenv("AGENT_GATEWAY_SOURCE_COMMIT", "").strip() or None


def package_health(*, additional_contracts: frozenset[str] = frozenset()) -> dict[str, Any]:
  return {
    "name": PACKAGE_NAME,
    "version": __version__,
    "source_commit": SOURCE_COMMIT,
    "source_commit_provenance": (
      "deployment_environment" if SOURCE_COMMIT is not None else None
    ),
    "contracts": sorted(CONTRACTS | additional_contracts),
  }


__all__ = [
  "CONTRACTS",
  "CONTRACT_AUTONOMOUS_OPERATOR_MESSAGES_V1",
  "CONTRACT_CONTROL_CHAT_CONTINUATION_V1",
  "CONTRACT_CONTROL_RUN_V1",
  "CONTRACT_CHAT_ATTACHMENTS_V1",
  "CONTRACT_INVESTMENT_SELECTED_CONTENT_V1",
  "CONTRACT_CREDENTIAL_REFRESH_V1",
  "PACKAGE_NAME",
  "SOURCE_COMMIT",
  "__version__",
  "package_health",
]
