# ruff: noqa: E402

from datetime import timedelta
import sys
from pathlib import Path
from typing import Never

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.mcp_client import McpClientManager, _ServerState


class _UnusedMcpSession:
  async def call_tool(
    self,
    name: str,
    arguments: dict[str, object],
    *,
    read_timeout_seconds: timedelta,
    meta: dict[str, object] | None = None,
  ) -> Never:
    _ = name, arguments, read_timeout_seconds, meta
    raise AssertionError("MCP session is not used in this test")


def test_apply_collision_filtering_strips_hidden_input_fields_from_advertised_schemas() -> None:
  manager = McpClientManager(config_path=None, strip_input_fields={"_session_id", "internal_only"})
  manager._servers = {
    "browser": _ServerState(
      name="browser",
      session=_UnusedMcpSession(),
      exit_contexts=[],
      tool_definitions=[
        {
          "name": "browser_snapshot",
          "description": "Take a browser snapshot.",
          "input_schema": {
            "type": "object",
            "properties": {
              "selector": {"type": "string"},
              "_session_id": {"type": "string"},
              "internal_only": {"type": "boolean"},
            },
            "required": ["selector", "_session_id", "internal_only"],
          },
        }
      ],
      tool_names={"browser_snapshot"},
    )
  }

  manager._apply_collision_filtering()

  tool_defs = manager.get_tool_definitions()
  schema = tool_defs[0]["input_schema"]

  assert schema["properties"] == {"selector": {"type": "string"}}
  assert schema["required"] == ["selector"]
  assert manager.get_server_tool_definitions({"browser"}) == tool_defs
