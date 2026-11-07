from __future__ import annotations

import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT / "api", ROOT / "packages" / "excel-mcp" / "python"):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from agent.interactive.tool_dispatcher import ExcelToolDispatcher  # noqa: E402
from agent_gateway import ToolDispatcher  # noqa: E402
from agent_gateway.mcp_client import McpClientManager  # noqa: E402
from agent_gateway.registry_scope import RegistryScope  # noqa: E402
from excel_mcp.relay import ChannelRegistry  # noqa: E402
from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND  # noqa: E402
from gateway_test_support.run_context_test_support import run_context_for_test  # noqa: E402


@pytest.fixture(params=[False, True], ids=["base", "excel"])
def dispatchers(request: pytest.FixtureRequest) -> tuple[ToolDispatcher, ToolDispatcher | ExcelToolDispatcher]:
  base = ToolDispatcher(
    mcp_client=McpClientManager(config_path=None),
    capability_bind=TEST_CAPABILITY_BIND,
    allowed_mcp_tools_by_server={"research": {"lookup"}},
    run_context=run_context_for_test(
      registry_scope=RegistryScope("run", "admission-reads", "/tmp/test-store"),
    ),
  )
  if not request.param:
    return base, base

  async def execute_addin(_request: Any) -> Any:
    raise AssertionError("admission reads must not execute tools")

  return base, ExcelToolDispatcher(
    base=base,
    channel_registry=ChannelRegistry(),
    execute_addin=execute_addin,
  )


def test_public_admission_reads_are_live_and_read_only(dispatchers: tuple[Any, Any]) -> None:
  base, dispatcher = dispatchers
  assert dispatcher.mcp_client is base.mcp_client
  allowed = dispatcher.allowed_mcp_tools_by_server
  assert allowed is base.allowed_mcp_tools_by_server
  assert allowed == {"research": {"lookup"}}
  allowed["research"].add("describe")
  assert dispatcher.allowed_mcp_tools_by_server["research"] == {"lookup", "describe"}
  for name in ("mcp_client", "allowed_mcp_tools_by_server"):
    with pytest.raises(AttributeError):
      setattr(dispatcher, name, None)


def test_bound_sequence_catalog_is_snapshotted(dispatchers: tuple[Any, Any]) -> None:
  base, dispatcher = dispatchers
  schema = {"type": "object", "properties": {"query": {"type": "string"}}}
  definition = MappingProxyType({"name": "lookup", "input_schema": schema})
  definitions = [definition]
  dispatcher.bind_tool_definitions(definitions)
  definitions.clear()
  assert dispatcher.tool_input_schema("lookup", "research") is schema
  first = base.get_tool_definitions()
  assert first == [definition]
  first.clear()
  assert base.get_tool_definitions() == [definition]
  assert dispatcher.tool_input_schema("missing", "research") == {}


def test_bound_callback_catalog_stays_live(dispatchers: tuple[Any, Any]) -> None:
  base, dispatcher = dispatchers
  definitions: list[dict[str, Any]] = []
  dispatcher.bind_tool_definitions(lambda: definitions)
  assert dispatcher.tool_input_schema("lookup", "research") == {}
  schema = {"type": "object", "properties": {"ticker": {"type": "string"}}}
  definitions.append({"name": "mcp__research__lookup", "inputSchema": schema})
  assert dispatcher.tool_input_schema("lookup", "research") is schema
  assert base.get_tool_definitions() == definitions
  definitions.clear()
  assert dispatcher.tool_input_schema("lookup", "research") == {}
