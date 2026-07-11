from __future__ import annotations

import os
import re
import random
from pathlib import Path
from typing import Any, Callable, Mapping


UNSET = object()
ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
STREAMABLE_HTTP_TYPES = {"streamable-http", "streamable_http", "http", "streamable"}
SUPPORTED_SERVER_TYPES = {"stdio"} | STREAMABLE_HTTP_TYPES
DEFAULT_ENV_ALLOWLIST = {
  "PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR", "USER",
  "PYTHONPATH", "NODE_PATH", "VIRTUAL_ENV",
}
MCP_STDIO_CONNECT_RETRIES_ENV = "MCP_STDIO_CONNECT_RETRIES"
MCP_STDIO_CONNECT_BACKOFF_ENV = "MCP_STDIO_CONNECT_BACKOFF_S"
MCP_STDIO_CONNECT_STABILIZE_ENV = "MCP_STDIO_CONNECT_STABILIZE_S"
MCP_STARTUP_CONCURRENCY_ENV = "MCP_STARTUP_CONCURRENCY"
MCP_STDIO_CONNECT_RETRIES_DEFAULT = 3
MCP_STDIO_CONNECT_BACKOFF_DEFAULT = 0.4
MCP_STDIO_CONNECT_STABILIZE_DEFAULT = 0.5
MCP_STDIO_RETRYABLE_EXCEPTION_NAMES = {
  "BrokenPipeError",
  "BrokenResourceError",
  "ClosedResourceError",
  "ConnectionAbortedError",
  "ConnectionResetError",
  "EOFError",
  "EndOfStream",
}
MCP_STDIO_RETRYABLE_MESSAGE_MARKERS = (
  "connection closed",
  "connection is closed",
  "transport closed",
  "stream closed",
  "closed stream",
  "broken pipe",
  "connection reset",
)


def resolve_mcp_config_path(
  config_path: Path | str | None | object = UNSET,
  *,
  unset: object = UNSET,
  environ: Mapping[str, str] | None = None,
  home_factory: Callable[[], Path] | None = None,
) -> Path | None:
  env = os.environ if environ is None else environ
  if config_path is unset:
    env_path = env.get("MCP_CONFIG_PATH", "").strip()
    if env_path:
      return Path(env_path).expanduser()
    return (home_factory or Path.home)() / ".claude.json"
  if config_path is None:
    return None
  return Path(config_path).expanduser()


def build_mcp_env(
  server_env: Mapping[str, Any] | None,
  *,
  environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
  env_source = os.environ if environ is None else environ
  env = {k: v for k, v in env_source.items() if k in DEFAULT_ENV_ALLOWLIST}
  if not isinstance(server_env, dict):
    return env

  for key, value in server_env.items():
    if value is None:
      continue
    env[str(key)] = expand_env_refs(value, environ=env_source)
  return env


def expand_env_refs(value: Any, *, environ: Mapping[str, str] | None = None) -> str:
  env = os.environ if environ is None else environ
  raw = str(value)
  return ENV_REF_RE.sub(lambda match: env.get(match.group(1), ""), raw)


def build_http_headers(
  headers: Mapping[str, Any] | None,
  *,
  environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
  if not isinstance(headers, dict):
    return {}
  return {
    str(key): expand_env_refs(value, environ=environ)
    for key, value in headers.items()
    if value is not None
  }


def env_nonnegative_int(
  name: str,
  default: int,
  *,
  environ: Mapping[str, str] | None = None,
  logger: Any | None = None,
) -> int:
  env = os.environ if environ is None else environ
  raw = env.get(name)
  if raw is None or not raw.strip():
    return default
  try:
    return max(0, int(raw))
  except ValueError:
    if logger is not None:
      logger.warning("Ignoring invalid %s=%r; using %d", name, raw, default)
    return default


def env_nonnegative_float(
  name: str,
  default: float,
  *,
  environ: Mapping[str, str] | None = None,
  logger: Any | None = None,
) -> float:
  env = os.environ if environ is None else environ
  raw = env.get(name)
  if raw is None or not raw.strip():
    return default
  try:
    return max(0.0, float(raw))
  except ValueError:
    if logger is not None:
      logger.warning("Ignoring invalid %s=%r; using %.3f", name, raw, default)
    return default


def stdio_connect_retries(
  *,
  environ: Mapping[str, str] | None = None,
  logger: Any | None = None,
) -> int:
  return env_nonnegative_int(
    MCP_STDIO_CONNECT_RETRIES_ENV,
    MCP_STDIO_CONNECT_RETRIES_DEFAULT,
    environ=environ,
    logger=logger,
  )


def stdio_connect_retry_delay(
  attempt: int,
  *,
  environ: Mapping[str, str] | None = None,
  logger: Any | None = None,
  jitter_fn: Callable[[float, float], float] | None = None,
) -> float:
  base = env_nonnegative_float(
    MCP_STDIO_CONNECT_BACKOFF_ENV,
    MCP_STDIO_CONNECT_BACKOFF_DEFAULT,
    environ=environ,
    logger=logger,
  )
  if base <= 0:
    return 0.0
  capped_attempt = max(0, attempt - 1)
  jitter = random.uniform if jitter_fn is None else jitter_fn
  return (base * (2 ** capped_attempt)) + jitter(0.0, base)


def stdio_connect_stabilize_delay(
  *,
  environ: Mapping[str, str] | None = None,
  logger: Any | None = None,
) -> float:
  return env_nonnegative_float(
    MCP_STDIO_CONNECT_STABILIZE_ENV,
    MCP_STDIO_CONNECT_STABILIZE_DEFAULT,
    environ=environ,
    logger=logger,
  )


def startup_concurrency_limit(
  *,
  environ: Mapping[str, str] | None = None,
  logger: Any | None = None,
) -> int:
  return env_nonnegative_int(
    MCP_STARTUP_CONCURRENCY_ENV,
    0,
    environ=environ,
    logger=logger,
  )


def iter_exception_tree(exc: BaseException):
  yield exc
  children = getattr(exc, "exceptions", None)
  if not isinstance(children, (list, tuple)):
    return
  for child in children:
    if isinstance(child, BaseException):
      yield from iter_exception_tree(child)


def is_retryable_stdio_connect_error(exc: BaseException) -> bool:
  for candidate in iter_exception_tree(exc):
    type_name = type(candidate).__name__
    message = str(candidate).lower()
    if type_name == "McpError":
      if any(marker in message for marker in MCP_STDIO_RETRYABLE_MESSAGE_MARKERS):
        return True
      continue
    if type_name in MCP_STDIO_RETRYABLE_EXCEPTION_NAMES:
      return True
    if isinstance(candidate, OSError) and any(
      marker in message for marker in MCP_STDIO_RETRYABLE_MESSAGE_MARKERS
    ):
      return True
  return False


def is_retryable_stdio_startup_error(exc: BaseException) -> bool:
  """Startup-connect retryability: transient transport errors (same as the tool-call gate) PLUS
  transient connect timeouts. Timeouts are startup-only — a tool-call timeout must NOT trigger a
  reconnect/re-run (duplicate side effects)."""
  if is_retryable_stdio_connect_error(exc):
    return True
  return any(isinstance(candidate, TimeoutError) for candidate in iter_exception_tree(exc))


def safe_cache_name(name: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "server"


__all__ = [
  "DEFAULT_ENV_ALLOWLIST",
  "ENV_REF_RE",
  "MCP_STARTUP_CONCURRENCY_ENV",
  "MCP_STDIO_CONNECT_BACKOFF_DEFAULT",
  "MCP_STDIO_CONNECT_BACKOFF_ENV",
  "MCP_STDIO_CONNECT_RETRIES_DEFAULT",
  "MCP_STDIO_CONNECT_RETRIES_ENV",
  "MCP_STDIO_CONNECT_STABILIZE_DEFAULT",
  "MCP_STDIO_CONNECT_STABILIZE_ENV",
  "MCP_STDIO_RETRYABLE_EXCEPTION_NAMES",
  "MCP_STDIO_RETRYABLE_MESSAGE_MARKERS",
  "STREAMABLE_HTTP_TYPES",
  "SUPPORTED_SERVER_TYPES",
  "UNSET",
  "build_http_headers",
  "build_mcp_env",
  "env_nonnegative_float",
  "env_nonnegative_int",
  "expand_env_refs",
  "is_retryable_stdio_connect_error",
  "is_retryable_stdio_startup_error",
  "iter_exception_tree",
  "resolve_mcp_config_path",
  "safe_cache_name",
  "startup_concurrency_limit",
  "stdio_connect_retries",
  "stdio_connect_retry_delay",
  "stdio_connect_stabilize_delay",
]
