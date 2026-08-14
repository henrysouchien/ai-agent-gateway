from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.mcp_client import McpClientManager, _ServerState  # noqa: E402
from agent_gateway.session import GatewaySession  # noqa: E402
from agent_gateway.tool_dispatcher import ToolDispatcher  # noqa: E402
import agent_gateway.mcp_client as mcp_client_module  # noqa: E402


SERVER = "research-corpus-mcp"
DYNAMIC_ENV = ["GATEWAY_API_KEY", "RISK_MODULE_USER_EMAIL"]


def _definition() -> _ServerState:
  return _ServerState(
    SERVER,
    SimpleNamespace(),
    [],
    [],
    {"thesis_list"},
    config={
      "command": "synthetic-research-server",
      "per_user": True,
      "per_user_env": DYNAMIC_ENV,
      "env": {"MCP_SUBPROCESS": "true"},
    },
  )


def _session(user_id: int, email: str) -> GatewaySession:
  return GatewaySession(
    session_id=f"session-{user_id}",
    api_key_hash="synthetic-hash",
    created_at=1,
    expires_at=2,
    user_id=str(user_id),
    user_email=email,
    risk_user_id=user_id,
    owner_user_id=str(user_id),
  )


def _manager(resolver) -> McpClientManager:
  manager = McpClientManager(
    config_path=None,
    per_user_env_resolver=resolver,
  )
  manager._servers = {SERVER: _definition()}
  manager._tool_to_server = {"thesis_list": SERVER}
  manager._mcp_tool_names = {"thesis_list"}
  return manager


def test_definition_process_strips_declared_user_authority(tmp_path, monkeypatch) -> None:
  config_path = tmp_path / "mcp.json"
  config_path.write_text(
    json.dumps({
      "mcpServers": {
        SERVER: {
          "command": "synthetic-research-server",
          "per_user": True,
          "per_user_env": DYNAMIC_ENV,
          "env": {
            "GATEWAY_API_KEY": "${GATEWAY_API_KEY}",
            "RISK_MODULE_USER_EMAIL": "${RISK_MODULE_USER_EMAIL}",
            "MCP_SUBPROCESS": "true",
          },
        }
      }
    }),
    encoding="utf-8",
  )
  monkeypatch.setattr(mcp_client_module, "MCP_IMPORT_ERROR", None)
  monkeypatch.setattr(
    mcp_client_module.os,
    "environ",
    {
      "GATEWAY_API_KEY": "synthetic-parent-ambient",
      "RISK_MODULE_USER_EMAIL": "operator@example.com",
    },
  )
  manager = McpClientManager(config_path=config_path)
  captured = {}

  async def scenario() -> None:
    async def connect(jobs):
      captured.update(jobs[0][1])
      return []

    manager._connect_startup_servers = connect
    await manager.startup()

  asyncio.run(scenario())

  assert captured["per_user"] is True
  assert captured["per_user_env"] == DYNAMIC_ENV
  assert captured["env"] == {"MCP_SUBPROCESS": "true"}


def test_authenticated_user_projection_completes_read_with_trusted_meta() -> None:
  captured = {}

  def resolver(server_name, user_id, user_email):
    assert (server_name, user_id, user_email) == (SERVER, "7", "user@example.com")
    return {
      "GATEWAY_API_KEY": "synthetic-user-projection",
      "RISK_MODULE_USER_EMAIL": "user@example.com",
    }

  manager = _manager(resolver)

  async def scenario() -> None:
    class Session:
      async def call_tool(self, name, tool_input, **kwargs):
        captured["call"] = (name, tool_input, kwargs)
        return SimpleNamespace(
          isError=False,
          structuredContent={"items": []},
          content=[],
        )

    async def connect(_name, config):
      captured["env"] = dict(config["env"])
      return _ServerState(
        SERVER,
        Session(),
        [object()],
        [],
        {"thesis_list"},
        config=config,
      )

    manager._connect_stdio_with_retries = connect
    result, error = await manager.call_tool(
      "thesis_list",
      {"limit": 1},
      meta={"user_id": "7", "channel": "cli"},
      gateway_session=_session(7, "user@example.com"),
    )
    assert error is None
    assert result == {"items": []}

  asyncio.run(scenario())

  assert captured["env"] == {
    "MCP_SUBPROCESS": "true",
    "GATEWAY_API_KEY": "synthetic-user-projection",
    "RISK_MODULE_USER_EMAIL": "user@example.com",
  }
  assert "GATEWAY_USER_KEYS" not in captured["env"]
  call_name, call_input, call_kwargs = captured["call"]
  assert call_name == "thesis_list"
  assert call_input == {"limit": 1}
  assert call_kwargs["meta"] == {"user_id": "7", "channel": "cli"}
  assert set(call_kwargs) == {"read_timeout_seconds", "meta"}


def test_missing_user_projection_fails_before_spawn() -> None:
  manager = _manager(lambda *_args: {})
  spawned = False

  async def scenario() -> None:
    nonlocal spawned

    async def connect(_name, _config):
      nonlocal spawned
      spawned = True
      raise AssertionError("missing authority must not spawn")

    manager._connect_stdio_with_retries = connect
    result, error = await manager.call_tool(
      "thesis_list",
      {},
      gateway_session=_session(7, "user@example.com"),
    )
    assert result is None
    assert error == {
      "code": "mcp_tool_error",
      "sub_code": "mcp_user_authority_unavailable",
      "message": "User-scoped MCP authority is incomplete.",
    }

  asyncio.run(scenario())
  assert spawned is False


def test_user_processes_isolate_and_credential_change_replaces_cached_child() -> None:
  projections = {
    "7": "synthetic-user-seven-v1",
    "8": "synthetic-user-eight-v1",
  }
  captured_envs = []

  def resolver(_server_name, user_id, user_email):
    return {
      "GATEWAY_API_KEY": projections[user_id],
      "RISK_MODULE_USER_EMAIL": str(user_email),
    }

  manager = _manager(resolver)
  drained = []

  async def scenario() -> None:
    async def connect(_name, config):
      captured_envs.append(dict(config["env"]))
      return _ServerState(
        SERVER,
        SimpleNamespace(),
        [object()],
        [],
        {"thesis_list"},
        config=config,
      )

    manager._connect_stdio_with_retries = connect
    manager._schedule_drain = drained.append
    subject_seven = mcp_client_module._PerUserGatewaySubject.from_gateway_session(
      _session(7, "seven@example.com")
    )
    subject_eight = mcp_client_module._PerUserGatewaySubject.from_gateway_session(
      _session(8, "eight@example.com")
    )

    first = await manager._get_per_user_server(SERVER, subject_seven)
    same = await manager._get_per_user_server(SERVER, subject_seven)
    other = await manager._get_per_user_server(SERVER, subject_eight)
    projections["7"] = "synthetic-user-seven-v2"
    replacement = await manager._get_per_user_server(SERVER, subject_seven)

    assert same is first
    assert other is not first
    assert replacement is not first
    assert len(drained) == 1

  asyncio.run(scenario())

  assert [env["GATEWAY_API_KEY"] for env in captured_envs] == [
    "synthetic-user-seven-v1",
    "synthetic-user-eight-v1",
    "synthetic-user-seven-v2",
  ]
  assert all("GATEWAY_USER_KEYS" not in env for env in captured_envs)


def test_dispatcher_preserves_meta_and_authenticated_session_for_user_server() -> None:
  async def scenario() -> None:
    manager = _manager(lambda *_args: {})
    gateway_session = _session(7, "user@example.com")
    captured = {}

    async def call_tool(name, tool_input, **kwargs):
      captured.update(name=name, tool_input=tool_input, kwargs=kwargs)
      return {"ok": True}, None

    manager.call_tool = call_tool
    dispatcher = ToolDispatcher(
      mcp_client=manager,
      session=gateway_session,
      session_id="session-7",
      user_id="7",
      risk_user_id=7,
      channel="cli",
      role="owner",
      mcp_meta_inject_servers=frozenset({SERVER}),
    )
    result, error = await dispatcher.dispatch(
      "call-1",
      "thesis_list",
      {},
      advertised_tool_names=frozenset({"thesis_list"}),
    )

    assert error is None
    assert result == {"ok": True}
    assert captured["kwargs"]["gateway_session"] is gateway_session
    assert captured["kwargs"]["meta"] == {
      "session_id": "session-7",
      "user_id": "7",
      "channel": "cli",
      "role": "owner",
    }

  asyncio.run(scenario())
