from __future__ import annotations

from typing import Dict, Mapping

from .agent_session_log_layout import SESSION_LOG_LAYOUT_ENV
from .containment_settings import CONTAINMENT_ENV_NAMES
from .gateway_address import GATEWAY_BASE_URL_ENV


RUN_BASH_ENVIRONMENT = "run_bash"
CODE_EXECUTE_ENVIRONMENT = "code_execute"
AUTONOMOUS_CHILD_ENVIRONMENT = "autonomous_child"

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

# The runtime vocabulary an autonomous child process needs to boot the agent
# runtime against this gateway's paired generation and its writable state. It
# carries every containment setting name, so the child resolves the parent's
# ``ContainmentSettings`` at its own start.
_AUTONOMOUS_CHILD_PROCESS_ENV_NAMES = frozenset({
  "PATH",
  "PYTHONPATH",
  "PYTHONHOME",
  "VIRTUAL_ENV",
  "HOME",
  "USER",
  "LOGNAME",
  "SHELL",
  "LANG",
  "LC_ALL",
  "LC_CTYPE",
  "TZ",
  "TMPDIR",
  "SSL_CERT_FILE",
  "SSL_CERT_DIR",
  "REQUESTS_CA_BUNDLE",
  "NO_PROXY",
  "PRODUCT_ID",
  "ENVIRONMENT",
  "APP_ENV",
  "AGENT_GATEWAY_ENV",
  "HANK_MODEL_AUTHORITY_DIR",
  "HANK_LOCAL_FALLBACK_ENABLED",
  "HANK_MODEL_PRESET",
  "USER_DATA_DIR",
  "CORPUS_LOG_DIR",
  "CORPUS_STATE_DIR",
  "LOG_DIR",
  "GATEWAY_LOG_DIR",
  "MCP_CONFIG_TMP_DIR",
  "MCP_CONFIG_TEMPLATE",
  "JUPYTER_ROOT",
  "AI_EXCEL_ADDIN_DOTENV",
  "INVESTMENT_TOOLS_DOTENV",
  "RISK_MODULE_DOTENV",
  "MCP_STARTUP_CONCURRENCY",
  "MCP_STDIO_CONNECT_BACKOFF_S",
  "MCP_STDIO_CONNECT_RETRIES",
  "MCP_STDIO_CONNECT_STABILIZE_S",
  "LOCAL_GATEWAY_PYTHON",
  "LOCAL_GATEWAY_CONTROL_GENERATION",
  "LOCAL_GATEWAY_CONTROL_GENERATION_ID",
  "LOCAL_GATEWAY_CONTROL_STATE_ROOT",
  "LOCAL_GATEWAY_RUNTIME_ROOT",
  "LOCAL_GATEWAY_RUNTIME_MANIFEST",
  "LOCAL_GATEWAY_RUNTIME_VERSION_ROOT",
  GATEWAY_BASE_URL_ENV,
  "GATEWAY_SSL_VERIFY",
  "AGENT_API_CLAIM_TTL_SECONDS",
  "AGENT_GATEWAY_SKILLS_DIR",
  "AGENT_SESSION_LOG_BASE_DIR",
  SESSION_LOG_LAYOUT_ENV,
  "AGENT_SESSION_LOG_MAX_ACTIVE_BYTES",
  "AGENT_SDK_CWD",
  "MEMORY_ENABLED",
  "MEMORY_KEYWORD_WEIGHT",
  "MEMORY_VECTOR_WEIGHT",
  "MEMORY_MAX_PROMPT_TOKENS",
  "CITATION_SESSION_REGISTRY_RETENTION_DAYS",
  "EDGAR_USE_UPSTREAM_EQUIVALENCE",
  "RISK_API_URL",
  "RISK_CLIENT_PATH",
  "RISK_MODULE_RESOLVER_URL",
  "CODE_EXECUTE_DOCKER_IMAGE",
  "EDGAR_API_URL",
  "EDGAR_UPDATER_ROOT",
  # Launch references the gateway's own MCP topology names for sibling servers
  # (deploy/mcp.local.template.json). Each is a path or a public key — never a
  # credential: interpreter and executable paths, a state directory, and the
  # Ed25519 *verify* key the Investment server checks capability claims with
  # (its private key is a different name and is never admitted).
  "EDGAR_UPDATER_PYTHON",
  "INVESTMENT_TOOLS_PYTHON",
  "INVESTMENT_TOOLS_STATE_DIR",
  "GSHEETS_MCP_EXECUTABLE",
  "INVESTMENT_CAPABILITY_CLAIM_ED25519_PUBLIC_KEY",
  "EXCEL_MCP_BACKEND_URL",
  "MODEL_ENGINE_OVERRIDES_DIR",
  "GATEWAY_SURFACE_CONTENT_CAPTURE_ENABLED",
  "GATEWAY_SURFACE_CONTENT_PATH",
}) | CONTAINMENT_ENV_NAMES

# Profiles whose runtime limits and delivery destination are configured per
# profile. Every autonomous child receives every profile's limit names alike.
AUTONOMOUS_CHILD_PROFILES = frozenset({"analyst", "advisor", "research_producer"})

_AUTONOMOUS_CHILD_PROFILE_LIMIT_ENV_NAMES = frozenset(
  f"{profile.upper()}_AGENT_{suffix}"
  for profile in AUTONOMOUS_CHILD_PROFILES
  for suffix in (
    "MAX_TOKENS",
  )
)

# What the launcher computes or forwards as the child's own runtime contract;
# a sibling server's dotenv never supplies one of these names.
AUTONOMOUS_CHILD_RUNTIME_ENV_NAMES = (
  _AUTONOMOUS_CHILD_PROCESS_ENV_NAMES
  | _AUTONOMOUS_CHILD_PROFILE_LIMIT_ENV_NAMES
)

# Tool configuration the child's MCP servers read. One set for every
# autonomous child: containment is the OS boundary around the child, not which
# profile it runs as. A name the gateway does not carry may still reach the
# declaring server from its own dotenv.
_AUTONOMOUS_CHILD_TOOL_ENV_NAMES = frozenset({
  "FMP_API_KEY",
  "FMP_CACHE_DIR",
  "FRED_API_KEY",
  "ORATS_API_KEY",
  "EDGAR_API_KEY",
  "SEC_BUDGET_SITE",
  "SEC_USER_AGENT",
  "SEC_EXPECTED_MACHINE_IDENTITY",
  "DATABENTO_API_KEY",
  "ALERTS_GATEWAY_API_KEY",
  "IBKR_ENABLED",
  "IBKR_FLEX_ENABLED",
  "IBKR_FLEX_TOKEN",
  "IBKR_FLEX_QUERY_ID",
  "SCHWAB_CLIENT_ID",
  "SCHWAB_CLIENT_SECRET",
  "SNAPTRADE_CLIENT_ID",
  "SNAPTRADE_CONSUMER_KEY",
  "PLAID_CLIENT_ID",
  "PLAID_SECRET",
  "PLAID_ENV",
})

CHILD_ENVIRONMENT_ALLOWLISTS = {
  RUN_BASH_ENVIRONMENT: (
    _PROCESS_RUNTIME_ENV_NAMES
    | _SIGNED_AGENT_CLAIM_ENV_NAMES
    | _AGENT_CORRELATION_ENV_NAMES
    | _AGENT_API_ENV_NAMES
  ),
  CODE_EXECUTE_ENVIRONMENT: (
    _PROCESS_RUNTIME_ENV_NAMES
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
  AUTONOMOUS_CHILD_ENVIRONMENT: (
    AUTONOMOUS_CHILD_RUNTIME_ENV_NAMES
    | _AUTONOMOUS_CHILD_TOOL_ENV_NAMES
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
  "AUTONOMOUS_CHILD_ENVIRONMENT",
  "AUTONOMOUS_CHILD_PROFILES",
  "AUTONOMOUS_CHILD_RUNTIME_ENV_NAMES",
  "CHILD_ENVIRONMENT_ALLOWLISTS",
  "CODE_EXECUTE_ENVIRONMENT",
  "RUN_BASH_ENVIRONMENT",
  "filter_child_environment",
]
