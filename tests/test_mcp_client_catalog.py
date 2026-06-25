# ruff: noqa: E402

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import policy_imports
import agent_gateway.mcp_client as mcp_client_module
from agent_gateway.mcp_client import McpClientManager, _ServerState
from agent_gateway.mcp_client_catalog import apply_collision_filtering
import api.agent.shared.server_policies as api_server_policies


class _CaptureLogger:
  def __init__(self) -> None:
    self.warnings: list[tuple[object, ...]] = []
    self.infos: list[tuple[object, ...]] = []
    self.errors: list[tuple[object, ...]] = []

  def warning(self, message, *args) -> None:
    self.warnings.append((message, *args))

  def info(self, message, *args) -> None:
    self.infos.append((message, *args))

  def error(self, message, *args) -> None:
    self.errors.append((message, *args))


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


def test_policy_owner_mismatch_hides_residual_runtime_tool(monkeypatch) -> None:
  logger = _CaptureLogger()
  monkeypatch.setattr(mcp_client_module, "log", logger)
  manager = McpClientManager(config_path=None)
  manager._servers = {
    "portfolio-mcp": _ServerState(
      name="portfolio-mcp",
      session=object(),
      exit_contexts=[],
      tool_definitions=[
        {"name": "execute_trade", "description": "stale residual", "input_schema": {}},
        {"name": "preview_trade", "description": "current residual", "input_schema": {}},
      ],
      tool_names={"execute_trade", "preview_trade"},
    ),
  }

  manager._apply_collision_filtering(
    policy_server_for_tool=lambda tool_name: (
      "portfolio-trades-mcp" if tool_name == "execute_trade" else "portfolio-mcp"
    )
  )

  assert manager.get_tool_definitions() == [
    {"name": "preview_trade", "description": "current residual", "input_schema": {}},
  ]
  assert manager.is_mcp_tool("execute_trade") is False
  assert manager.is_mcp_tool("preview_trade") is True
  assert manager.get_server_for_tool("execute_trade") is None
  assert manager.get_server_for_tool("preview_trade") == "portfolio-mcp"
  assert manager._servers["portfolio-mcp"].tool_names == {"preview_trade"}
  assert manager.get_startup_diagnostics()["portfolio-mcp"]["category"] == "policy_owner_mismatch"
  assert "execute_trade" in manager.get_startup_diagnostics()["portfolio-mcp"]["message"]
  assert logger.errors


def test_policy_owner_invariant_falls_back_to_api_import(monkeypatch) -> None:
  logger = _CaptureLogger()
  monkeypatch.setattr(mcp_client_module, "log", logger)

  def fake_import_module(name: str):
    if name == "agent.shared.server_policies":
      raise ModuleNotFoundError("No module named 'agent'", name="agent")
    if name == "api.agent.shared.server_policies":
      return api_server_policies
    raise AssertionError(f"unexpected import: {name}")

  monkeypatch.setattr(policy_imports.importlib, "import_module", fake_import_module)
  monkeypatch.setattr(
    api_server_policies,
    "get_server_for_policy_tool",
    lambda tool_name: "portfolio-trades-mcp" if tool_name == "execute_trade" else None,
  )
  manager = McpClientManager(config_path=None)
  manager._servers = {
    "portfolio-mcp": _ServerState(
      name="portfolio-mcp",
      session=object(),
      exit_contexts=[],
      tool_definitions=[
        {"name": "execute_trade", "description": "stale residual", "input_schema": {}},
      ],
      tool_names={"execute_trade"},
    ),
  }

  manager._apply_collision_filtering()

  assert manager.get_tool_definitions() == []
  assert manager.is_mcp_tool("execute_trade") is False
  assert manager.get_startup_diagnostics()["portfolio-mcp"]["category"] == "policy_owner_mismatch"


def test_policy_owner_invariant_raises_when_policy_import_dependency_breaks(
  monkeypatch,
) -> None:
  def fake_import_module(_name: str):
    raise ModuleNotFoundError("No module named 'broken_dependency'", name="broken_dependency")

  monkeypatch.setattr(policy_imports.importlib, "import_module", fake_import_module)
  manager = McpClientManager(config_path=None)
  manager._servers = {
    "portfolio-mcp": _ServerState(
      name="portfolio-mcp",
      session=object(),
      exit_contexts=[],
      tool_definitions=[
        {"name": "execute_trade", "description": "stale residual", "input_schema": {}},
      ],
      tool_names={"execute_trade"},
    ),
  }

  with pytest.raises(ModuleNotFoundError, match="broken_dependency"):
    manager._apply_collision_filtering()


def test_policy_owner_invariant_raises_when_api_policy_import_dependency_breaks(
  monkeypatch,
) -> None:
  def fake_import_module(name: str):
    if name == "agent.shared.server_policies":
      raise ModuleNotFoundError("No module named 'agent'", name="agent")
    if name == "api.agent.shared.server_policies":
      raise ModuleNotFoundError("No module named 'broken_dependency'", name="broken_dependency")
    raise AssertionError(f"unexpected import: {name}")

  monkeypatch.setattr(policy_imports.importlib, "import_module", fake_import_module)
  manager = McpClientManager(config_path=None)
  manager._servers = {
    "portfolio-mcp": _ServerState(
      name="portfolio-mcp",
      session=object(),
      exit_contexts=[],
      tool_definitions=[
        {"name": "execute_trade", "description": "stale residual", "input_schema": {}},
      ],
      tool_names={"execute_trade"},
    ),
  }

  with pytest.raises(ModuleNotFoundError, match="broken_dependency"):
    manager._apply_collision_filtering()


def test_policy_owner_invariant_uses_original_name_for_prefixed_tools(monkeypatch) -> None:
  logger = _CaptureLogger()
  monkeypatch.setattr(mcp_client_module, "log", logger)
  manager = McpClientManager(config_path=None)
  manager._servers = {
    "portfolio-trades-mcp": _ServerState(
      name="portfolio-trades-mcp",
      session=object(),
      exit_contexts=[],
      tool_definitions=[
        {"name": "execute_trade", "description": "split", "input_schema": {}},
      ],
      tool_names={"execute_trade"},
      tool_prefix="trades_",
    ),
  }

  manager._apply_collision_filtering(
    policy_server_for_tool=lambda tool_name: (
      "portfolio-trades-mcp" if tool_name == "execute_trade" else None
    )
  )

  assert manager.get_tool_definitions() == [
    {"name": "trades_execute_trade", "description": "split", "input_schema": {}},
  ]
  assert manager.get_server_for_tool("trades_execute_trade") == "portfolio-trades-mcp"
  assert manager.get_original_tool_name("trades_execute_trade") == "execute_trade"
  assert manager.get_original_tool_name("unknown_tool") == "unknown_tool"
  assert manager.resolve_tool_name("portfolio-trades-mcp", "execute_trade") == "trades_execute_trade"
  assert manager.get_startup_diagnostics() == {}
  assert logger.errors == []
