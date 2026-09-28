# ruff: noqa: E402

import asyncio
import sys
from pathlib import Path

import pytest
from mcp import MCPError
from mcp.types import (
  CONNECTION_CLOSED,
  INTERNAL_ERROR,
  INVALID_PARAMS,
  INVALID_REQUEST,
  METHOD_NOT_FOUND,
  PARSE_ERROR,
  REQUEST_TIMEOUT,
  CallToolResult,
)

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import mcp_client_config, mcp_client_errors
from agent_gateway.mcp_client import McpClientManager, _ConnectedServerState


@pytest.mark.parametrize(
  ("exc", "expected"),
  [
    (asyncio.TimeoutError("operation timed out"), "timeout"),
    (ConnectionError("connection refused"), "connection_error"),
    (RuntimeError("other"), "unknown"),
    (MCPError(REQUEST_TIMEOUT, "response deadline elapsed"), "timeout"),
    (MCPError(CONNECTION_CLOSED, "peer disappeared"), "connection_error"),
    (MCPError(PARSE_ERROR, "bad JSON"), "parse_error"),
    (MCPError(INVALID_REQUEST, "bad request"), "parse_error"),
    (MCPError(INVALID_PARAMS, "bad arguments"), "parse_error"),
    (MCPError(METHOD_NOT_FOUND, "unknown method"), "not_found"),
    (MCPError(INTERNAL_ERROR, "connection timed out parsing not found"), "unknown"),
  ],
)
def test_exception_classification_uses_protocol_codes(exc, expected) -> None:
  assert mcp_client_errors.classify_exception(exc, str(exc)) == expected


def test_tool_result_error_text_classification() -> None:
  mcp_error_cases = [
    ("filing not found", "not_found"),
    ("malformed payload", "parse_error"),
    ("request timed out", "timeout"),
    ("unexpected", "unknown"),
  ]
  for message, expected in mcp_error_cases:
    assert mcp_client_errors.classify_mcp_error(message) == expected


def test_startup_failure_helper_classifies_timeout_config_and_default() -> None:
  def never_retryable(_exc: BaseException) -> bool:
    return False

  assert mcp_client_errors.startup_failure_from_exception(
    asyncio.TimeoutError(),
    is_retryable_stdio_connect_error=never_retryable,
  ) == {
    "category": "transient_timeout",
    "retryable": True,
    "message": "TimeoutError",
    "error_type": "TimeoutError",
  }
  assert mcp_client_errors.startup_failure_from_exception(
    ValueError("missing url"),
    is_retryable_stdio_connect_error=never_retryable,
  ) == {
    "category": "config_error",
    "retryable": False,
    "message": "missing url",
    "error_type": "ValueError",
  }
  assert mcp_client_errors.startup_failure_from_exception(
    RuntimeError("boom"),
    is_retryable_stdio_connect_error=never_retryable,
  ) == {
    "category": "startup_error",
    "retryable": False,
    "message": "boom",
    "error_type": "RuntimeError",
  }


def test_startup_failure_helper_classifies_missing_executable_before_retryable() -> None:
  def always_retryable(_exc: BaseException) -> bool:
    return True

  assert mcp_client_errors.startup_failure_from_exception(
    mcp_client_errors.McpExecutableMissingError(
      "stdio executable does not exist: /deleted/venv/bin/gsheets-mcp"
    ),
    is_retryable_stdio_connect_error=always_retryable,
  ) == {
    "category": "executable_missing",
    "retryable": False,
    "message": "stdio executable does not exist: /deleted/venv/bin/gsheets-mcp",
    "error_type": "McpExecutableMissingError",
  }


@pytest.mark.parametrize("grouped", [False, True])
def test_sdk_request_timeout_is_transient_without_replay_permission(grouped) -> None:
  exc: Exception = MCPError(REQUEST_TIMEOUT, "response deadline elapsed")
  if grouped:
    exc = ExceptionGroup("transport task failed", [exc])

  failure = mcp_client_errors.startup_failure_from_exception(
    exc,
    is_retryable_stdio_connect_error=mcp_client_config.is_retryable_stdio_connect_error,
  )

  assert failure["category"] == "transient_timeout"
  assert failure["retryable"] is True
  assert mcp_client_config.is_retryable_stdio_startup_error(exc) is True
  assert mcp_client_config.is_retryable_stdio_connect_error(exc) is False


@pytest.mark.parametrize("allow_uncertain_replay", [False, True])
def test_sdk_timeout_does_not_replay_dispatched_mutation(
  monkeypatch, allow_uncertain_replay,
) -> None:
  async def scenario():
    mutations = []
    connections = []

    class Session:
      def __init__(self, generation):
        self.generation = generation

      async def call_tool(
        self,
        name: str,
        arguments: dict[str, object],
        *,
        read_timeout_seconds: float,
        meta: dict[str, object] | None = None,
      ) -> CallToolResult:
        mutations.append((self.generation, arguments["value"]))
        if self.generation == 1:
          raise MCPError(REQUEST_TIMEOUT, "response deadline elapsed")
        return CallToolResult(content=[], structured_content={"written": arguments["value"]})

    def state(generation):
      return _ConnectedServerState(
        name="documents-mcp",
        session=Session(generation),
        exit_contexts=[],
        tool_definitions=[{"name": "write_document", "description": "", "input_schema": {}}],
        tool_names={"write_document"},
        config={"type": "stdio", "command": "documents-mcp"},
      )

    async def connect(name, config):
      connections.append(name)
      return state(2)

    manager = McpClientManager(config_path=None)
    manager._servers = {"documents-mcp": state(1)}
    manager._tool_to_server = {"write_document": "documents-mcp"}
    monkeypatch.setattr(manager, "_connect_stdio_with_retries", connect)
    try:
      result, error = await manager.call_tool(
        "write_document", {"value": "first"},
        allow_uncertain_replay=allow_uncertain_replay,
      )
      assert result is None
      assert error is not None
      assert error["sub_code"] == "timeout"
      assert mutations == [(1, "first")]

      if allow_uncertain_replay:
        # A timeout must never enter the reconnect-and-replay path.
        assert connections == []
      else:
        assert connections == ["documents-mcp"]
        result, error = await manager.call_tool(
          "write_document", {"value": "second"}, allow_uncertain_replay=False,
        )
        assert error is None
        assert result == {"written": "second"}
        assert mutations == [(1, "first"), (2, "second")]
    finally:
      await manager.shutdown()

  asyncio.run(scenario())


def test_tool_error_names_an_exception_whose_string_is_empty(monkeypatch) -> None:
  import anyio

  async def scenario():
    class Session:
      async def call_tool(self, name, arguments, *, read_timeout_seconds, meta=None):
        raise anyio.ClosedResourceError()

    def state():
      return _ConnectedServerState(
        name="edgar-parser-mcp",
        session=Session(),
        exit_contexts=[],
        tool_definitions=[{"name": "get_filing_sections", "description": "", "input_schema": {}}],
        tool_names={"get_filing_sections"},
        config={"type": "stdio", "command": "edgar-parser-mcp"},
      )

    async def connect(name, config):
      return state()

    manager = McpClientManager(config_path=None)
    manager._servers = {"edgar-parser-mcp": state()}
    manager._tool_to_server = {"get_filing_sections": "edgar-parser-mcp"}
    monkeypatch.setattr(manager, "_connect_stdio_with_retries", connect)
    try:
      assert str(anyio.ClosedResourceError()) == ""
      result, error = await manager.call_tool(
        "get_filing_sections", {"ticker": "PCTY"}, allow_uncertain_replay=False,
      )
    finally:
      await manager.shutdown()
    assert result is None
    assert error == {
      "code": "tool_error",
      "sub_code": "unknown",
      "message": "ClosedResourceError",
    }

  asyncio.run(scenario())
