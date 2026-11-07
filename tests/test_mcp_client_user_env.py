from __future__ import annotations
from agent_gateway.registry_scope import RegistryScope
from gateway_test_support.run_context_test_support import run_context_for_test
from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND

import asyncio
import sys
from pathlib import Path
from typing import Mapping


PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.mcp_client import McpClientManager, _ServerState  # noqa: E402
from agent_gateway.mcp_client_connections import McpToolCallResult  # noqa: E402
from agent_gateway.session import GatewaySession  # noqa: E402
from agent_gateway.tool_dispatcher import ToolDispatcher  # noqa: E402
from gateway_test_support.host_policy import owner_session_host_policy


SERVER = "research-corpus-mcp"


class _UnusedClientSession:
  async def call_tool(
    self,
    name: str,
    arguments: Mapping[str, object],
    *,
    read_timeout_seconds: float | None,
    meta: Mapping[str, object] | None = None,
  ) -> McpToolCallResult:
    raise AssertionError(f"unexpected physical MCP call: {name}")

def _definition() -> _ServerState:
  return _ServerState(
    SERVER,
    _UnusedClientSession(),
    [],
    [],
    {"thesis_list"},
    config={
      "command": "synthetic-research-server",
      "per_user": True,
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
    role="owner",
  )


def _manager() -> McpClientManager:
  manager = McpClientManager(config_path=None)
  manager._servers = {SERVER: _definition()}
  manager._tool_to_server = {"thesis_list": SERVER}
  manager._mcp_tool_names = {"thesis_list"}
  return manager


def test_dispatcher_preserves_meta_and_authenticated_session_for_user_server(monkeypatch, owner_session_host_policy) -> None:
  async def scenario() -> None:
    manager = _manager()
    gateway_session = _session(7, "user@example.com")
    captured = {}

    async def call_tool(name, tool_input, **kwargs):
      captured.update(name=name, tool_input=tool_input, kwargs=kwargs)
      return {"ok": True}, None

    monkeypatch.setattr(manager, "call_tool", call_tool)
    dispatcher = ToolDispatcher(
      capability_bind=TEST_CAPABILITY_BIND,
      mcp_client=manager,
      session=gateway_session,
      session_id="session-7",
      user_id="7",
      risk_user_id=7,
      channel="cli",
      role="owner",
      mcp_meta_inject_servers=frozenset({SERVER}),
      run_context=run_context_for_test(registry_scope=RegistryScope('run', 'test-run', '/tmp/test-store')),
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
    assert "session_token" not in captured["kwargs"]["meta"]

  asyncio.run(scenario())



def test_user_server_expansion_uses_projected_env_not_ambient_pool() -> None:
  manager = _manager()
  config = {
    "command": "synthetic-research-server",
    "per_user": True,
    "run_scoped_env": ["RISK_API_URL"],
    "env": {
      "RISK_API_URL": "${RISK_API_URL}",
      "ANTHROPIC_AUTH_TOKEN": "${ANTHROPIC_AUTH_TOKEN}",
    },
  }
  manager._server_env_passthrough = {"research-corpus-mcp": {"ANTHROPIC_AUTH_TOKEN"}}
  ambient = {
    "RISK_API_URL": "https://risk.example.test",
    "ANTHROPIC_AUTH_TOKEN": "ambient-operator-seat",
    "ANTHROPIC_API_KEY": "ambient-payg",
  }
  denied, absent = manager._scoped_server_env(
    SERVER, config, deployment_environ=ambient,
  )
  assert denied["RISK_API_URL"] == "https://risk.example.test"
  assert "ANTHROPIC_AUTH_TOKEN" not in denied
  assert "ANTHROPIC_API_KEY" not in denied
  assert "ANTHROPIC_AUTH_TOKEN" in absent
  admitted, admitted_absent = manager._scoped_server_env(
    SERVER,
    config,
    deployment_environ=ambient,
    projected_env={"ANTHROPIC_AUTH_TOKEN": "run-owner-seat"},
  )
  assert admitted["ANTHROPIC_AUTH_TOKEN"] == "run-owner-seat"
  assert "ANTHROPIC_AUTH_TOKEN" not in admitted_absent
  assert "ambient-operator-seat" not in admitted.values()


def test_unbound_shared_startup_has_no_pool_and_owner_run_projects_its_seat(monkeypatch) -> None:
  import agent_gateway.mcp_client as mcp_client_module
  from hank_model_transport.providers.anthropic_oauth import ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES

  manager = McpClientManager(
    config_path=None,
    server_env_passthrough={"jobs-mcp": set(ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES) | {
      "GMAIL_TOKEN_PATH",
      "USER_DATA_DIR",
    }},
  )
  manager._servers = {
    "jobs-mcp": _ServerState(
      "jobs-mcp",
      _UnusedClientSession(),
      [],
      [],
      {"list_schedules"},
      config={"command": "jobs-server", "env": {"USER_DATA_DIR": "${USER_DATA_DIR}"}},
    ),
  }
  monkeypatch.setattr(
    mcp_client_module.os, "environ",
    {
      "ANTHROPIC_AUTH_TOKEN": "owner-seat",
      "USER_DATA_DIR": "/mnt/hank-data",
      "GMAIL_TOKEN_PATH": "/run/gmail-token",
    },
  )
  monkeypatch.setattr(mcp_client_module, "load_user_runtime_credentials", lambda _env: None)
  monkeypatch.setattr(
    mcp_client_module, "scope_admits_operator_seats",
    lambda scope, *, environ: str(scope) == "7",
  )

  assert manager._jobs_oauth_projection(mcp_client_module.os.environ) == {}
  config = manager._servers["jobs-mcp"].config
  assert config is not None
  shared, absent = manager._scoped_server_env(
    "jobs-mcp", config,
  )
  assert "ANTHROPIC_AUTH_TOKEN" in absent
  assert shared["USER_DATA_DIR"] == "/mnt/hank-data"
  assert shared["GMAIL_TOKEN_PATH"] == "/run/gmail-token"

  owner = _session(7, "owner@example.com")
  subject = manager._authenticated_jobs_subject(owner)
  assert subject is not None and subject.user_id == "7"
  run_config = manager._jobs_run_config(config, subject.user_id)
  assert run_config["_projected_env"] == {"ANTHROPIC_AUTH_TOKEN": "owner-seat"}
  assert "USER_DATA_DIR" not in run_config["_projected_env"]

  invite = _session(9, "invite@example.com")
  invite.role = "invite"
  assert manager._authenticated_jobs_subject(invite) is None
  other_owner = _session(8, "owner2@example.com")
  assert manager._authenticated_jobs_subject(other_owner) is None


def test_jobs_owner_run_transport_failure_returns_typed_error_and_evicts_child(monkeypatch) -> None:
  import time

  import agent_gateway.mcp_client as mcp_client_module
  from agent_gateway.mcp_client import _PerUserServerState
  from hank_model_transport.providers.anthropic_oauth import ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES

  async def scenario() -> None:
    manager = McpClientManager(
      config_path=None,
      server_env_passthrough={"jobs-mcp": set(ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES)},
    )
    definition = _ServerState(
      "jobs-mcp", _UnusedClientSession(), [], [], {"list_schedules"},
      config={"command": "jobs-server", "env": {}},
    )
    manager._servers = {"jobs-mcp": definition}
    manager._tool_to_server = {"list_schedules": "jobs-mcp"}
    manager._mcp_tool_names = {"list_schedules"}
    monkeypatch.setattr(
      mcp_client_module.os, "environ", {"ANTHROPIC_AUTH_TOKEN": "owner-seat"},
    )
    monkeypatch.setattr(mcp_client_module, "load_user_runtime_credentials", lambda _env: None)
    monkeypatch.setattr(
      mcp_client_module, "scope_admits_operator_seats",
      lambda scope, *, environ: str(scope) == "7",
    )
    child = _PerUserServerState(
      _ServerState(
        "jobs-mcp[user]", _UnusedClientSession(), [], [], {"list_schedules"},
        config={"command": "jobs-server"},
      ),
      time.time() + 3600,
      time.time(),
    )
    resolved: list[str] = []

    async def resolve(server, subject, **_kwargs):
      resolved.append(subject.user_id)
      manager._per_user_servers[(server, subject.user_id)] = child
      return child

    async def fail(**_kwargs):
      raise EOFError("transport failed with sensitive upstream detail")

    drained: list[tuple[object, str]] = []
    monkeypatch.setattr(manager, "_get_per_user_server", resolve)
    monkeypatch.setattr(manager, "_call_tool_once", fail)
    monkeypatch.setattr(
      manager, "_schedule_drain", lambda state, reason: drained.append((state, reason)),
    )

    result, error = await manager.call_tool(
      "list_schedules", {}, gateway_session=_session(7, "owner@example.com"),
    )

    assert result is None
    assert error is not None and error["code"] == "tool_error"
    assert error["sub_code"] == "unknown"
    assert resolved == ["7"]
    assert ("jobs-mcp", "7") not in manager._per_user_servers
    assert drained == [(child, "dispatch_transport_failure")]
    assert child.active_calls == 0

  asyncio.run(scenario())
