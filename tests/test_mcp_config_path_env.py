# ruff: noqa: E402

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import McpClientManager, create_agent
from agent_gateway.providers.agent_sdk import load_mcp_config_for_sdk


def _write_mcp_config(path: Path, server_name: str, command: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
    json.dumps({"mcpServers": {server_name: {"command": command}}}),
    encoding="utf-8",
  )


def _tiers(*server_names: str) -> dict:
  return {None: {"always": set(server_names), "defer": set()}}


def test_mcp_client_manager_uses_mcp_config_path_env(
  monkeypatch,
  tmp_path: Path,
) -> None:
  config_path = tmp_path / "env.json"
  monkeypatch.setenv("MCP_CONFIG_PATH", str(config_path))

  manager = McpClientManager()

  assert manager._config_path == config_path


def test_mcp_client_manager_unset_env_loads_no_file_backed_servers(
  monkeypatch,
  tmp_path: Path,
) -> None:
  monkeypatch.setenv("HOME", str(tmp_path))
  monkeypatch.delenv("MCP_CONFIG_PATH", raising=False)

  manager = McpClientManager()

  assert manager._config_path is None


def test_mcp_client_manager_explicit_config_path_wins_over_env(
  monkeypatch,
  tmp_path: Path,
) -> None:
  env_path = tmp_path / "env.json"
  explicit_path = tmp_path / "explicit.json"
  monkeypatch.setenv("MCP_CONFIG_PATH", str(env_path))

  manager = McpClientManager(config_path=explicit_path)

  assert manager._config_path == explicit_path


def test_mcp_client_manager_expands_tilde_from_env(
  monkeypatch,
  tmp_path: Path,
) -> None:
  monkeypatch.setenv("HOME", str(tmp_path))
  monkeypatch.setenv("MCP_CONFIG_PATH", "~/gateway.json")

  manager = McpClientManager()

  assert manager._config_path == tmp_path / "gateway.json"


def test_mcp_client_manager_blank_env_loads_no_file_backed_servers(
  monkeypatch,
  tmp_path: Path,
) -> None:
  monkeypatch.setenv("HOME", str(tmp_path))
  monkeypatch.setenv("MCP_CONFIG_PATH", " \t ")

  manager = McpClientManager()

  assert manager._config_path is None


def test_agent_sdk_loader_uses_mcp_config_path_env(
  monkeypatch,
  tmp_path: Path,
) -> None:
  config_path = tmp_path / "sdk-env.json"
  _write_mcp_config(config_path, "research", "env-cmd")
  monkeypatch.setenv("MCP_CONFIG_PATH", str(config_path))

  configs = load_mcp_config_for_sdk(None, _tiers("research"))

  assert configs == {"research": {"command": "env-cmd"}}


def test_agent_sdk_loader_unset_env_ignores_home_claude_json(
  monkeypatch,
  tmp_path: Path,
) -> None:
  config_path = tmp_path / ".claude.json"
  _write_mcp_config(config_path, "research", "default-cmd")
  monkeypatch.setenv("HOME", str(tmp_path))
  monkeypatch.delenv("MCP_CONFIG_PATH", raising=False)

  configs = load_mcp_config_for_sdk(None, _tiers("research"))

  assert configs == {}


def test_agent_sdk_loader_explicit_config_path_wins_over_env(
  monkeypatch,
  tmp_path: Path,
) -> None:
  env_path = tmp_path / "sdk-env.json"
  explicit_path = tmp_path / "sdk-explicit.json"
  _write_mcp_config(env_path, "research", "env-cmd")
  _write_mcp_config(explicit_path, "research", "explicit-cmd")
  monkeypatch.setenv("MCP_CONFIG_PATH", str(env_path))

  configs = load_mcp_config_for_sdk(None, _tiers("research"), config_path=explicit_path)

  assert configs == {"research": {"command": "explicit-cmd"}}


def test_agent_sdk_loader_expands_tilde_from_env(
  monkeypatch,
  tmp_path: Path,
) -> None:
  config_path = tmp_path / "sdk-home.json"
  _write_mcp_config(config_path, "research", "tilde-cmd")
  monkeypatch.setenv("HOME", str(tmp_path))
  monkeypatch.setenv("MCP_CONFIG_PATH", "~/sdk-home.json")

  configs = load_mcp_config_for_sdk(None, _tiers("research"))

  assert configs == {"research": {"command": "tilde-cmd"}}


def test_agent_sdk_loader_blank_env_ignores_home_claude_json(
  monkeypatch,
  tmp_path: Path,
) -> None:
  config_path = tmp_path / ".claude.json"
  _write_mcp_config(config_path, "research", "default-cmd")
  monkeypatch.setenv("HOME", str(tmp_path))
  monkeypatch.setenv("MCP_CONFIG_PATH", "  ")

  configs = load_mcp_config_for_sdk(None, _tiers("research"))

  assert configs == {}


def test_create_agent_uses_mcp_config_path_env(
  monkeypatch,
  tmp_path: Path,
) -> None:
  config_path = tmp_path / "easy-env.json"
  monkeypatch.setenv("MCP_CONFIG_PATH", str(config_path))

  app = create_agent("test")

  assert app.state.gateway_config.mcp_client._config_path == config_path


def test_create_agent_explicit_mcp_config_path_wins_over_env(
  monkeypatch,
  tmp_path: Path,
) -> None:
  env_path = tmp_path / "easy-env.json"
  explicit_path = tmp_path / "easy-explicit.json"
  monkeypatch.setenv("MCP_CONFIG_PATH", str(env_path))

  app = create_agent("test", mcp_config_path=explicit_path)

  assert app.state.gateway_config.mcp_client._config_path == explicit_path


def test_create_agent_expands_tilde_from_mcp_config_path_env(
  monkeypatch,
  tmp_path: Path,
) -> None:
  monkeypatch.setenv("HOME", str(tmp_path))
  monkeypatch.setenv("MCP_CONFIG_PATH", "~/easy.json")

  app = create_agent("test")

  assert app.state.gateway_config.mcp_client._config_path == tmp_path / "easy.json"


def test_create_agent_blank_mcp_config_path_env_is_unset_for_inline_servers(
  monkeypatch,
) -> None:
  monkeypatch.setenv("MCP_CONFIG_PATH", "  ")

  app = create_agent("test", mcp_servers={"inline": {"command": "inline-cmd"}})

  assert app.state.gateway_config.mcp_client._config_path is None
