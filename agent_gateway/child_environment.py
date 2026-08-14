from __future__ import annotations

from typing import Dict, Mapping


RUN_BASH_ENVIRONMENT = "run_bash"
CODE_EXECUTE_ENVIRONMENT = "code_execute"

_PROCESS_RUNTIME_ENV_NAMES = frozenset(
  {
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "PYTHONIOENCODING",
    "PYTHONPATH",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TEMP",
    "TMP",
    "TMPDIR",
    "TZ",
  }
)

_SIGNED_AGENT_CLAIM_ENV_NAMES = frozenset(
  {
    "AGENT_API_CLAIM_AUDIENCE",
    "AGENT_API_CLAIM_EXPIRY",
    "AGENT_API_CLAIM_ISSUED_AT",
    "AGENT_API_CLAIM_NONCE",
    "AGENT_API_CLAIM_SIGNATURE",
    "AGENT_API_CLAIM_USER_EMAIL",
    "AGENT_API_CLAIM_USER_ID",
  }
)

_AGENT_CORRELATION_ENV_NAMES = frozenset(
  {
    "AGENT_CONTROL_RUN_ID",
    "AGENT_TELEMETRY_REQUEST_ID",
    "AGENT_TELEMETRY_RUN_ID",
    "AGENT_TELEMETRY_TOOL_CALL_ID",
  }
)

_AGENT_API_ENV_NAMES = frozenset({"RISK_API_URL"})

CHILD_ENVIRONMENT_ALLOWLISTS = {
  RUN_BASH_ENVIRONMENT: (
    _PROCESS_RUNTIME_ENV_NAMES
    | _SIGNED_AGENT_CLAIM_ENV_NAMES
    | _AGENT_CORRELATION_ENV_NAMES
    | _AGENT_API_ENV_NAMES
  ),
  CODE_EXECUTE_ENVIRONMENT: (
    _PROCESS_RUNTIME_ENV_NAMES
    | _SIGNED_AGENT_CLAIM_ENV_NAMES
    | _AGENT_CORRELATION_ENV_NAMES
    | _AGENT_API_ENV_NAMES
    | frozenset(
      {
        "AGENT_CODE_EXECUTE_WORK_DIR",
        "HOME",
        "MPLBACKEND",
        "MPLCONFIGDIR",
        "PYTHONUNBUFFERED",
        "XDG_CACHE_HOME",
      }
    )
  ),
}


def filter_child_environment(
  env: Mapping[str, str],
  *,
  purpose: str,
) -> Dict[str, str]:
  """Return only the variables explicitly authorized for one child purpose.

  Callers must apply this as the final environment-building step. Adding a new
  gateway secret is safe by default because unknown names are never inherited.
  """
  try:
    allowed_names = CHILD_ENVIRONMENT_ALLOWLISTS[purpose]
  except KeyError as exc:
    raise ValueError(f"Unknown child environment purpose: {purpose}") from exc
  return {
    key: str(value)
    for key, value in env.items()
    if key in allowed_names
  }


__all__ = [
  "CHILD_ENVIRONMENT_ALLOWLISTS",
  "CODE_EXECUTE_ENVIRONMENT",
  "RUN_BASH_ENVIRONMENT",
  "filter_child_environment",
]
