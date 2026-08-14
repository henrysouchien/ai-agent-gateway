from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.autonomous as autonomous  # noqa: E402
from agent_gateway import BoundCapabilityExecution, EventLog  # noqa: E402
from agent_gateway.session import GatewaySession  # noqa: E402
from tests.capability_execution_test_support import (  # noqa: E402
  stub_capability_execution_resolver,
)


def _run(coro: Any) -> Any:
  return asyncio.run(coro)


def _bound_execution() -> dict[str, Any]:
  resolver = stub_capability_execution_resolver(run_mode="autonomous")
  resolved = resolver.resolve("session.driver")
  execution = BoundCapabilityExecution(
    bind=resolved.bind,
    registry=resolved.registry,
    adapter=resolved.adapter,
    auth_config={**resolved.auth_config, "max_tokens": 16_000},
  )
  return {
    "capability_execution": execution,
    "capability_execution_resolver": resolver,
    "session": GatewaySession(
      session_id="autonomous-mcp-allowlist-test",
      api_key_hash="test",
      created_at=1,
      expires_at=2,
      user_id="alice",
      role="owner",
    ),
  }


def _run_kwargs() -> dict[str, Any]:
  return {
    **_bound_execution(),
    "user_id": "alice",
    "billing_mode": "byok",
    "rate_table_version": "unknown",
  }


@pytest.mark.parametrize(
  "mcp_configuration",
  [
    {"mcp_servers": {"browser": {"command": "never"}}},
    {"mcp_config_path": "/config/that/must/not/be/read.json"},
  ],
)
def test_mcp_configuration_without_trusted_allowlist_fails_before_manager_construction(
  monkeypatch: pytest.MonkeyPatch,
  mcp_configuration: dict[str, Any],
) -> None:
  class _UnexpectedMcpClientManager:
    def __init__(self, **_kwargs: Any) -> None:
      raise AssertionError("MCP manager must not be constructed")

  monkeypatch.setattr(autonomous, "McpClientManager", _UnexpectedMcpClientManager)

  with pytest.raises(ValueError, match="requires trusted_mcp_allowed_servers"):
    _run(
      autonomous.run_autonomous(
        "System",
        "Run",
        **_run_kwargs(),
        **mcp_configuration,
      )
    )


@pytest.mark.parametrize(
  ("inline_name", "server_aliases"),
  [
    ("browser", {}),
    ("legacy-browser", {"legacy-browser": "browser"}),
  ],
)
def test_disallowed_inline_names_and_aliases_fail_before_manager_construction(
  monkeypatch: pytest.MonkeyPatch,
  inline_name: str,
  server_aliases: dict[str, str],
) -> None:
  class _UnexpectedMcpClientManager:
    def __init__(self, **_kwargs: Any) -> None:
      raise AssertionError("MCP manager must not be constructed")

  monkeypatch.setattr(autonomous, "McpClientManager", _UnexpectedMcpClientManager)

  with pytest.raises(ValueError, match="absent from the trusted allowlist"):
    _run(
      autonomous.run_autonomous(
        "System",
        "Run",
        **_run_kwargs(),
        mcp_servers={inline_name: {"command": "never"}},
        trusted_mcp_allowed_servers={"portfolio-reads-mcp"},
        trusted_mcp_server_aliases=server_aliases,
      )
    )


@pytest.mark.parametrize(
  ("mcp_configuration", "expected_inline"),
  [
    (
      {"mcp_servers": {"portfolio-reads-mcp": {"command": "mocked"}}},
      {"portfolio-reads-mcp"},
    ),
    (
      {
        "mcp_servers": {"legacy-reads": {"command": "mocked"}},
        "trusted_mcp_server_aliases": {"legacy-reads": "portfolio-reads-mcp"},
      },
      {"legacy-reads"},
    ),
    ({"mcp_config_path": "/config/read-only-in-the-mock.json"}, set()),
  ],
)
def test_trusted_product_allowed_configuration_reaches_mocked_startup(
  monkeypatch: pytest.MonkeyPatch,
  mcp_configuration: dict[str, Any],
  expected_inline: set[str],
) -> None:
  captured: dict[str, Any] = {}

  class _FakeMcpClientManager:
    def __init__(self, **kwargs: Any) -> None:
      captured["manager_kwargs"] = kwargs
      self._inline = dict(kwargs.get("inline_servers") or {})

    async def startup(self) -> None:
      captured["startup"] = True

    async def shutdown(self) -> None:
      captured["shutdown"] = True

    def get_server_names(self) -> set[str]:
      return set(self._inline)

    def get_tool_definitions(self) -> list[dict[str, Any]]:
      return []

  class _StubDispatcher:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
      return None

  class _StubRunner:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
      return None

  async def _fake_run_session(
    _runner: Any,
    _event_log: EventLog,
    **_kwargs: Any,
  ) -> autonomous.RunOutput:
    return autonomous.RunOutput("ok", [], {}, None, False)

  monkeypatch.delenv("EXCEL_ORCHESTRATION_DEV", raising=False)
  monkeypatch.setattr(autonomous, "McpClientManager", _FakeMcpClientManager)
  monkeypatch.setattr(autonomous, "ToolDispatcher", _StubDispatcher)
  monkeypatch.setattr(autonomous, "AgentRunner", _StubRunner)
  monkeypatch.setattr(autonomous, "run_session", _fake_run_session)

  output = _run(
    autonomous.run_autonomous(
      "System",
      "Run",
      **_run_kwargs(),
      **mcp_configuration,
      trusted_mcp_allowed_servers={"portfolio-reads-mcp"},
    )
  )

  assert output.response == "ok"
  assert captured["manager_kwargs"]["allowed_servers"] == {"portfolio-reads-mcp"}
  assert captured["manager_kwargs"]["server_aliases"] == mcp_configuration.get(
    "trusted_mcp_server_aliases",
    {},
  )
  assert set(captured["manager_kwargs"].get("inline_servers") or {}) == expected_inline
  assert captured["startup"] is True
  assert captured["shutdown"] is True
