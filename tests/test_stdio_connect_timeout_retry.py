import asyncio
from types import SimpleNamespace

import pytest

from agent_gateway.mcp_client_config import (
  is_retryable_stdio_connect_error,
  is_retryable_stdio_startup_error,
)
from agent_gateway.mcp_client_connections import (
  McpConnectionRuntime,
  connect_stdio_with_retries,
)


def _runtime(
  *,
  retries,
  retry_delay=lambda _attempt: 0,
  retryable=is_retryable_stdio_startup_error,
) -> McpConnectionRuntime:
  return McpConnectionRuntime(
    startup_concurrency_limit=lambda: 0,
    startup_failure_from_exception=lambda exc: {
      "category": "startup_error",
      "message": str(exc),
      "retryable": False,
      "error_type": type(exc).__name__,
    },
    streamable_http_types=set(),
    stdio_connect_retries=retries,
    stdio_connect_retry_delay=retry_delay,
    stdio_connect_stabilize_delay=lambda: 0,
    is_retryable_stdio_startup_error=retryable,
    build_mcp_env=lambda _env: {},
    preflight_stdio_executable=lambda _command, _args, _env: None,
    build_http_headers=lambda _headers: {},
    parse_allowed_tools=lambda _value: None,
    safe_cache_name=lambda name: name,
    close_contexts=lambda _contexts: None,
    server_state_factory=lambda **kwargs: kwargs,
    stdio_server_parameters_factory=lambda **kwargs: kwargs,
    stdio_client_factory=lambda *_args, **_kwargs: None,
    client_session_factory=lambda *_args, **_kwargs: None,
    httpx_module=None,
    streamable_http_client_factory=None,
    json_file_key_value_factory=lambda _path: None,
    fastmcp_oauth_factory=None,
    path_factory=None,
    httpx_import_error=None,
    streamable_http_import_error=None,
    fastmcp_oauth_import_error=None,
    environ={},
    logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
  )


def test_stdio_startup_retryable_predicate_includes_timeouts_and_transport_errors() -> None:
  assert is_retryable_stdio_startup_error(TimeoutError()) is True
  assert is_retryable_stdio_startup_error(asyncio.TimeoutError()) is True
  assert is_retryable_stdio_startup_error(ExceptionGroup("startup", [TimeoutError()])) is True
  assert is_retryable_stdio_startup_error(BrokenPipeError()) is True
  assert is_retryable_stdio_startup_error(ValueError()) is False


def test_stdio_connect_retryable_predicate_stays_timeout_free_for_tool_calls() -> None:
  assert is_retryable_stdio_connect_error(TimeoutError()) is False
  assert is_retryable_stdio_connect_error(asyncio.TimeoutError()) is False


def test_stdio_connect_retries_startup_timeout_then_returns_success() -> None:
  sentinel = object()
  attempts: list[int] = []

  class _Manager:
    async def _connect_stdio(self, _name, _config):
      attempts.append(1)
      if len(attempts) == 1:
        raise asyncio.TimeoutError()
      return sentinel

  result = asyncio.run(
    connect_stdio_with_retries(
      _Manager(),
      "portfolio-reads-mcp",
      {"type": "stdio", "command": "python3"},
      _runtime(retries=lambda: 3),
    )
  )

  assert result is sentinel
  assert len(attempts) == 2


def test_stdio_connect_raises_after_all_startup_timeout_attempts() -> None:
  attempts: list[int] = []
  total_retries = 3

  class _Manager:
    async def _connect_stdio(self, _name, _config):
      attempts.append(1)
      raise asyncio.TimeoutError()

  with pytest.raises(asyncio.TimeoutError):
    asyncio.run(
      connect_stdio_with_retries(
        _Manager(),
        "portfolio-reads-mcp",
        {"type": "stdio", "command": "python3"},
        _runtime(retries=lambda: total_retries),
      )
    )

  assert len(attempts) == 1 + total_retries
