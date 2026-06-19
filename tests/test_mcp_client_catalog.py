# ruff: noqa: E402

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.mcp_client as mcp_client_module
from agent_gateway.mcp_client import McpClientManager, _ServerState
from agent_gateway.mcp_client_catalog import apply_collision_filtering


class _CaptureLogger:
  def __init__(self) -> None:
    self.warnings: list[tuple[object, ...]] = []
    self.infos: list[tuple[object, ...]] = []

  def warning(self, message, *args) -> None:
    self.warnings.append((message, *args))

  def info(self, message, *args) -> None:
    self.infos.append((message, *args))


def test_apply_collision_filtering_handles_collisions_prefixes_and_hidden_fields() -> None:
  logger = _CaptureLogger()
  first_tool = {
    "name": "shared_tool",
    "description": "first",
    "input_schema": {
      "type": "object",
      "properties": {"visible": {}, "_session_id": {}},
      "required": ["visible", "_session_id"],
    },
  }
  servers = {
    "first": SimpleNamespace(
      tool_prefix="",
      tool_definitions=[
        {"name": "builtin_tool", "description": "collision", "input_schema": {}},
        first_tool,
      ],
      tool_names=set(),
    ),
    "second": SimpleNamespace(
      tool_prefix="",
      tool_definitions=[{"name": "shared_tool", "description": "duplicate", "input_schema": {}}],
      tool_names=set(),
    ),
    "prefixed": SimpleNamespace(
      tool_prefix="safe_",
      tool_definitions=[{"name": "shared_tool", "description": "prefixed", "input_schema": {}}],
      tool_names=set(),
    ),
  }

  result = apply_collision_filtering(
    servers=servers,
    builtin_tool_names={"builtin_tool"},
    strip_input_fields={"_session_id"},
    logger=logger,
  )

  assert [tool["name"] for tool in result.tool_definitions] == ["shared_tool", "safe_shared_tool"]
  assert result.tool_to_server == {"shared_tool": "first", "safe_shared_tool": "prefixed"}
  assert result.prefixed_to_original == {"safe_shared_tool": "shared_tool"}
  assert result.mcp_tool_names == {"shared_tool", "safe_shared_tool"}
  assert servers["first"].tool_definitions == [first_tool]
  assert servers["first"].tool_names == {"shared_tool"}
  assert servers["second"].tool_definitions == []
  assert servers["second"].tool_names == set()
  assert servers["prefixed"].tool_definitions[0]["name"] == "safe_shared_tool"
  assert first_tool["input_schema"]["properties"] == {"visible": {}}
  assert first_tool["input_schema"]["required"] == ["visible"]
  assert len(logger.warnings) == 2
  assert logger.warnings[0][1:3] == ("builtin_tool", "first")
  assert logger.warnings[1][1:4] == ("shared_tool", "second", "first")
  assert len(logger.infos) == 3


def test_parent_apply_collision_filtering_uses_parent_logger(monkeypatch) -> None:
  logger = _CaptureLogger()
  monkeypatch.setattr(mcp_client_module, "log", logger)
  manager = McpClientManager(config_path=None, builtin_tool_names={"builtin_tool"})
  manager._servers = {
    "gateway": _ServerState(
      name="gateway",
      session=object(),
      exit_contexts=[],
      tool_definitions=[
        {"name": "builtin_tool", "description": "collision", "input_schema": {}},
        {"name": "remote_tool", "description": "kept", "input_schema": {}},
      ],
      tool_names={"builtin_tool", "remote_tool"},
    ),
    "prefixed": _ServerState(
      name="prefixed",
      session=object(),
      exit_contexts=[],
      tool_definitions=[
        {"name": "remote_tool", "description": "prefixed", "input_schema": {}},
      ],
      tool_names={"remote_tool"},
      tool_prefix="safe_",
    ),
  }

  manager._apply_collision_filtering()

  assert manager.get_tool_definitions() == [
    {"name": "remote_tool", "description": "kept", "input_schema": {}},
    {"name": "safe_remote_tool", "description": "prefixed", "input_schema": {}},
  ]
  assert manager.is_mcp_tool("remote_tool") is True
  assert manager.is_mcp_tool("safe_remote_tool") is True
  assert manager.get_server_for_tool("remote_tool") == "gateway"
  assert manager.get_server_for_tool("safe_remote_tool") == "prefixed"
  assert manager.resolve_tool_name("prefixed", "remote_tool") == "safe_remote_tool"
  assert logger.warnings[0][1:3] == ("builtin_tool", "gateway")
  assert logger.infos[0][1:4] == ("gateway", 1, ["remote_tool"])
  assert logger.infos[1][1:4] == ("prefixed", 1, ["safe_remote_tool"])
