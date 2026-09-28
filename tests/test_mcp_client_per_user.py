# ruff: noqa: E402

import asyncio
import hashlib
import hmac
import json
import logging
import sys
import time
from pathlib import Path

import pytest

from typing import Mapping
ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))
from mcp.types import CallToolResult as _ToolResult, ContentBlock, TextContent

from agent_gateway.mcp_client import (
  McpClientManager,
  _ConnectedServerState,
  _PerUserGatewaySubject,
  _PerUserMcpError,
  _PerUserServerState,
  _ServerState,
)
from agent_gateway.session import GatewaySession
from agent_gateway.tool_dispatcher import ToolDispatcher
import agent_gateway.mcp_client as mcp_client_module
from gateway_test_support.host_policy import owner_session_host_policy


def _definition(config=None):
  return _ServerState("gsheets-mcp", _UnusedClientSession(), [], [], {"tool"}, config=config or {
    "command": "/venv/python",
    "args": ["server.py"],
    "per_user": True,
    "env": {"GSHEETS_TOKEN_MODE": "broker", "GSHEETS_HEADLESS": "1"},
  })


def _child(label):
  return _ConnectedServerState(
    label,
    _UnusedClientSession(),
    [object()],
    [],
    {"tool"},
    config={"type": "stdio"},
  )


_MISSING = object()


async def _state(
  manager,
  label,
  expires_at=None,
  last_used_at=None,
  *,
  active_calls=0,
  user_id="7",
  server_name="gsheets-mcp",
):
  """A per-user state shaped the way `_spawn_per_user_server` shapes one.

  The child's contexts are entered by its own host task, so a later close
  unwinds them from that task and not from whoever asked for the child.
  """
  child = _child(label)

  async def _connect(*_args, **_kwargs):
    return child

  host = mcp_client_module._PerUserChildHost(server_name, user_id)
  saved = manager.__dict__.get("_connect_stdio_with_retries", _MISSING)
  manager._connect_stdio_with_retries = _connect
  task = host.start(manager._host_per_user_child(host, {}))
  try:
    await asyncio.shield(host.ready)
  finally:
    if saved is _MISSING:
      del manager._connect_stdio_with_retries
    else:
      manager._connect_stdio_with_retries = saved
  manager._per_user_hosts.add(host)
  task.add_done_callback(lambda _task: manager._per_user_hosts.discard(host))
  now = time.time()
  return _PerUserServerState(
    child,
    now + 3600 if expires_at is None else expires_at,
    now if last_used_at is None else last_used_at,
    active_calls=active_calls,
    host=host,
  )


def _manager(operation="gsheets_read_range"):
  manager = McpClientManager(config_path=None)
  manager._servers = {"gsheets-mcp": _definition()}
  manager._tool_to_server = {"tool": "gsheets-mcp"}
  manager._prefixed_to_original = {"tool": operation}
  return manager


def _gateway_session(user_id: int = 7, *, owner_user_id: str | None = None) -> GatewaySession:
  return GatewaySession(
    session_id=f"session-{user_id}",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id=str(user_id),
    risk_user_id=user_id,
    owner_user_id=owner_user_id or str(user_id),
    role="owner",
  )


def _subject(user_id: int = 7) -> _PerUserGatewaySubject:
  return _PerUserGatewaySubject.from_gateway_session(_gateway_session(user_id))


async def _mint_ok(subject):
  return "tier-two", time.time() + 3600, "https://risk"




class _UnusedClientSession:
  async def call_tool(
    self,
    name: str,
    arguments: Mapping[str, object],
    *,
    read_timeout_seconds: float,
    meta: Mapping[str, object] | None = None,
  ) -> _ToolResult:
    raise AssertionError(f"unexpected physical MCP call: {name}")


def _tool_result(*, is_error=False, payload=None, structured_content=None) -> _ToolResult:
  content: list[ContentBlock] = []
  if payload is not None:
    content = [TextContent(type="text", text=json.dumps(payload))]
  return _ToolResult(
    is_error=is_error,
    structured_content=structured_content,
    content=content,
  )


def _sheets_error_result(
  *,
  operation="gsheets_read_range",
  code="broker_session_expired",
  message="The Google Sheets broker session expired.",
  outcome_state="not_started",
  retry_safe=True,
  retry_automatic=True,
  recovery=None,
):
  return _tool_result(
    is_error=True,
    structured_content={
      "status": "error",
      "operation": operation,
      "error": {
        "code": code,
        "message": message,
        "outcome": {
          "state": outcome_state,
          "phase": "authorize",
          "mutation_may_have_occurred": False,
        },
        "retry": {
          "safe": retry_safe,
          "automatic": retry_automatic,
          "action": "refresh_session",
          "retry_after_seconds": None,
        },
        "validation": None,
        "recovery": recovery,
      },
    },
  )


def test_definition_only_config_parses_per_user_without_token(tmp_path):
  config_path = tmp_path / "mcp.json"
  config_path.write_text(json.dumps({"mcpServers": {"gsheets-mcp": _definition().config}}))
  manager = McpClientManager(config_path=config_path)
  loaded = manager._read_claude_config()["mcpServers"]["gsheets-mcp"]
  assert loaded["per_user"] is True
  assert "GSHEETS_BROKER_SESSION_TOKEN" not in loaded["env"]


def test_startup_connects_definition_only_config_without_token(tmp_path):
  async def scenario():
    config_path = tmp_path / "mcp.json"
    config_path.write_text(json.dumps({"mcpServers": {"gsheets-mcp": _definition().config}}))
    manager = McpClientManager(config_path=config_path)
    captured = []

    async def connect(connect_jobs):
      captured.extend(connect_jobs)
      return []

    manager._connect_startup_servers = connect
    await manager.startup()
    assert captured[0][1]["per_user"] is True
    assert "GSHEETS_BROKER_SESSION_TOKEN" not in captured[0][1]["env"]

  asyncio.run(scenario())


def test_tier_one_http_contract_and_typed_404(monkeypatch):
  async def scenario():
    manager = _manager()
    captured = []

    class Response:
      status_code = 404

      def json(self):
        return {"error": "sheets_not_connected"}

    class Client:
      def __init__(self, **_kwargs):
        pass

      async def __aenter__(self):
        return self

      async def __aexit__(self, *_args):
        pass

      async def post(self, url, *, json, headers):
        captured.append({"url": url, "body": json, "headers": headers})
        return Response()

    monkeypatch.setenv("GOOGLE_SHEETS_BROKER_URL", "https://risk.example/")
    monkeypatch.setenv("GATEWAY_GOOGLE_SHEETS_BROKER_HMAC_KEY", "test-key")
    monkeypatch.setattr(mcp_client_module.httpx, "AsyncClient", Client)
    for _attempt in range(2):
      try:
        await manager._mint_gsheets_broker_session(_subject())
      except _PerUserMcpError as exc:
        assert exc.code == "sheets_not_connected"
      else:
        raise AssertionError("expected typed broker failure")
    first = captured[0]
    assert first["url"].endswith("/api/internal/google/sheets-broker-session")
    assert first["body"]["scopes"] == ["https://www.googleapis.com/auth/spreadsheets"]
    assert first["body"]["ttl_s"] == 3600
    assert len(first["body"]["request_id"]) == 32
    assert set(first["headers"]) == {"X-Resolver-Timestamp", "X-Resolver-Signature"}
    canonical_body = json.dumps(
      first["body"],
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=False,
    ).encode("utf-8")
    signed_message = (
      first["headers"]["X-Resolver-Timestamp"].encode("ascii")
      + b"\n"
      + canonical_body
    )
    expected_signature = hmac.new(b"test-key", signed_message, hashlib.sha256).hexdigest()
    assert first["headers"]["X-Resolver-Signature"] == expected_signature
    assert captured[0]["body"]["request_id"] != captured[1]["body"]["request_id"]

  asyncio.run(scenario())


def test_tier_one_hmac_uses_canonical_session_subject(monkeypatch):
  async def scenario():
    manager = _manager()
    captured = {}

    class Response:
      status_code = 200

      def json(self):
        return {
          "session_token": "tier-two",
          "expires_at": time.time() + 3600,
        }

    class Client:
      def __init__(self, **_kwargs):
        pass

      async def __aenter__(self):
        return self

      async def __aexit__(self, *_args):
        pass

      async def post(self, _url, *, json, headers):
        captured.update({"body": json, "headers": headers})
        return Response()

    monkeypatch.setenv("GOOGLE_SHEETS_BROKER_URL", "https://risk.example")
    monkeypatch.setenv("GATEWAY_GOOGLE_SHEETS_BROKER_HMAC_KEY", "test-key")
    monkeypatch.setattr(mcp_client_module.httpx, "AsyncClient", Client)
    await manager._mint_gsheets_broker_session(_subject())

    canonical_body = json.dumps(
      captured["body"],
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=False,
    ).encode("utf-8")
    assert captured["body"]["user_id"] == "7"
    signed_message = (
      captured["headers"]["X-Resolver-Timestamp"].encode("ascii")
      + b"\n"
      + canonical_body
    )
    expected_signature = hmac.new(b"test-key", signed_message, hashlib.sha256).hexdigest()
    assert captured["headers"]["X-Resolver-Signature"] == expected_signature

  asyncio.run(scenario())


def test_missing_identity_fails_closed_without_spawn():
  manager = _manager()
  result, error = asyncio.run(manager.call_tool("tool", {}))
  assert result is None
  assert error is not None
  assert error["sub_code"] == "missing_user_identity"
  assert error["data"]["operation"] == "gsheets_read_range"
  assert error["data"]["error"]["outcome"]["state"] == "not_started"
  assert error["data"]["error"]["retry"]["automatic"] is False
  assert manager._per_user_servers == {}


def test_session_owner_mismatch_fails_closed_without_broker_mint():
  manager = _manager()
  result, error = asyncio.run(
    manager.call_tool(
      "tool",
      {},
      gateway_session=_gateway_session(owner_user_id="8"),
    )
  )
  assert result is None
  assert error is not None
  assert error["sub_code"] == "missing_user_identity"
  assert manager._per_user_servers == {}


def test_dispatcher_passes_authenticated_session_to_per_user_mcp(monkeypatch, owner_session_host_policy):
  async def scenario():
    manager = _manager()
    manager._tool_to_server = {"gsheets_read_range": "gsheets-mcp"}
    manager._prefixed_to_original = {}
    manager._mcp_tool_names = {"gsheets_read_range"}
    session = _gateway_session()
    # This asserts identity plumbing on the owner path.
    session.role = "owner"
    captured = {}

    async def call_tool(name, tool_input, **kwargs):
      captured.update(name=name, tool_input=tool_input, kwargs=kwargs)
      return {"ok": True}, None

    monkeypatch.setattr(manager, "call_tool", call_tool)
    dispatcher = ToolDispatcher(
      mcp_client=manager,
      session=session,
      risk_user_id=7,
    )
    result, error = await dispatcher.dispatch(
      "call-1",
      "gsheets_read_range",
      {},
      advertised_tool_names=frozenset({"gsheets_read_range"}),
    )
    assert error is None
    assert result == {"ok": True}
    assert captured["kwargs"]["gateway_session"] is session
    assert "user_id" not in captured["kwargs"]

  asyncio.run(scenario())


def test_same_user_single_flight_and_different_users_isolate(monkeypatch):
  async def scenario():
    manager = _manager()
    spawned = []
    gate = asyncio.Event()

    async def spawn(server, user, broker_session=None):
      del server, broker_session
      spawned.append(user.user_id)
      await gate.wait()
      return await _state(manager, user.user_id, time.time() + 3600, time.time())

    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", _mint_ok)
    monkeypatch.setattr(manager, "_spawn_per_user_server", spawn)
    same = [
      asyncio.create_task(manager._get_per_user_server("gsheets-mcp", _subject()))
      for _ in range(2)
    ]
    await asyncio.sleep(0)
    gate.set()
    first, second = await asyncio.gather(*same)
    other = await manager._get_per_user_server("gsheets-mcp", _subject(8))
    assert first is second
    assert other is not first
    assert spawned == ["7", "8"]

  asyncio.run(scenario())


def test_spawn_env_contains_token_but_not_tier_one_key(monkeypatch):
  async def scenario():
    manager = _manager()
    captured = {}
    monkeypatch.setenv("GATEWAY_GOOGLE_SHEETS_BROKER_HMAC_KEY", "must-not-leak")

    async def mint(_user):
      return "tier-two", time.time() + 3600, "https://risk"

    async def connect(_name, config):
      captured.update(config["env"])
      return _child("spawned")

    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", mint)
    monkeypatch.setattr(manager, "_connect_stdio_with_retries", connect)
    await manager._spawn_per_user_server("gsheets-mcp", _subject())
    assert captured["GSHEETS_BROKER_SESSION_TOKEN"] == "tier-two"
    assert captured["GSHEETS_BROKER_URL"] == "https://risk"
    assert "GATEWAY_GOOGLE_SHEETS_BROKER_HMAC_KEY" not in captured

  asyncio.run(scenario())


def test_mint_failures_are_terminal_and_do_not_spawn(monkeypatch):
  async def scenario(code):
    manager = _manager()

    async def fail(_user):
      raise _PerUserMcpError(code)

    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", fail)
    result, error = await manager.call_tool(
      "tool", {}, gateway_session=_gateway_session()
    )
    assert result is None
    assert error is not None
    assert error["sub_code"] == code
    assert error["data"]["error"]["outcome"]["state"] == "not_started"
    assert error["data"]["error"]["outcome"]["mutation_may_have_occurred"] is False
    assert manager._per_user_servers == {}

  asyncio.run(scenario("sheets_not_connected"))
  asyncio.run(scenario("broker_rate_limited"))


def test_near_expiry_replaces_and_drains_old(monkeypatch):
  async def scenario():
    manager = _manager()
    old = await _state(manager, "old", time.time() + 1, time.time(), active_calls=1)
    manager._per_user_servers[("gsheets-mcp", "7")] = old
    replacement = await _state(manager, "new", time.time() + 3600, time.time())
    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", _mint_ok)
    monkeypatch.setattr(manager, "_spawn_per_user_server", lambda *_, **__: asyncio.sleep(0, result=replacement))
    closed = []
    monkeypatch.setattr(manager, "_close_contexts", lambda contexts: asyncio.sleep(0, result=closed.append(contexts)))
    current = await manager._get_per_user_server("gsheets-mcp", _subject())
    assert current is replacement
    assert old.draining is True
    await asyncio.sleep(0)
    assert closed == []
    old.active_calls = 0
    await asyncio.gather(*manager._drain_tasks)
    assert closed

  asyncio.run(scenario())


def test_dead_instance_respawns_and_acquisition_retires_no_sibling(monkeypatch):
  """Acquiring one child is not the moment to decide another one is idle.

  The idle rule has one owner, the periodic reaper. Running it inside an
  acquisition retired the very child the caller was asking for, and everything
  else a long turn had opened along with it.
  """

  async def scenario():
    manager = _manager()
    stale = await _state(manager, "stale", time.time() + 3600, 0)
    dead = await _state(manager, "dead", time.time() + 3600, time.time())
    dead.server.exit_contexts = []
    manager._per_user_servers[("gsheets-mcp", "1")] = stale
    manager._per_user_servers[("gsheets-mcp", "2")] = dead
    drained = []
    monkeypatch.setattr(manager, "_schedule_drain", lambda state, reason: drained.append((state, reason)))
    spawned = []

    async def spawn(_server, user, broker_session=None):
      del broker_session
      spawned.append(user.user_id)
      return await _state(manager, user.user_id, time.time() + 3600, time.time())

    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", _mint_ok)
    monkeypatch.setattr(manager, "_spawn_per_user_server", spawn)
    await manager._get_per_user_server("gsheets-mcp", _subject(2))
    assert "2" in spawned
    assert drained == [(dead, "dead_transport")]
    assert manager._per_user_servers[("gsheets-mcp", "1")] is stale
    assert stale.draining is False

  asyncio.run(scenario())


def test_concurrent_burst_respects_atomic_per_server_cap(monkeypatch):
  async def scenario():
    manager = _manager()
    monkeypatch.setattr(mcp_client_module, "PER_USER_INSTANCE_CAP", 2)
    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", _mint_ok)
    gate = asyncio.Event()
    connect_started = 0
    max_accounted = 0

    async def connect(_name, _config):
      nonlocal connect_started, max_accounted
      connect_started += 1
      accounted = sum(key[0] == "gsheets-mcp" for key in manager._per_user_servers)
      accounted += manager._per_user_spawn_reservations.get("gsheets-mcp", 0)
      max_accounted = max(max_accounted, accounted)
      await gate.wait()
      return _child(f"child-{connect_started}")

    monkeypatch.setattr(manager, "_connect_stdio_with_retries", connect)
    tasks = [
      asyncio.create_task(manager._get_per_user_server("gsheets-mcp", _subject(user)))
      for user in range(1, 4)
    ]
    while connect_started < 2:
      await asyncio.sleep(0)
    await asyncio.sleep(0)
    gate.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert sum(isinstance(result, _PerUserServerState) for result in results) == 2
    errors = [result for result in results if isinstance(result, _PerUserMcpError)]
    assert len(errors) == 1
    assert errors[0].code == "sheets_unavailable"
    assert max_accounted == 2
    assert sum(key[0] == "gsheets-mcp" for key in manager._per_user_servers) == 2
    assert manager._per_user_spawn_reservations == {}

  asyncio.run(scenario())


def test_force_respawn_reserves_slot_before_concurrent_new_user(monkeypatch):
  async def scenario():
    manager = _manager()
    monkeypatch.setattr(mcp_client_module, "PER_USER_INSTANCE_CAP", 1)
    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", _mint_ok)
    old = await _state(manager, "old", time.time() + 3600, time.time())
    manager._per_user_servers[("gsheets-mcp", "1")] = old
    spawn_started = asyncio.Event()
    spawn_gate = asyncio.Event()
    spawn_users = []
    max_accounted = 0

    async def spawn(_server, user, broker_session=None):
      nonlocal max_accounted
      del broker_session
      spawn_users.append(user.user_id)
      accounted = sum(key[0] == "gsheets-mcp" for key in manager._per_user_servers)
      accounted += manager._per_user_spawn_reservations.get("gsheets-mcp", 0)
      max_accounted = max(max_accounted, accounted)
      spawn_started.set()
      await spawn_gate.wait()
      return await _state(manager, f"replacement-{user.user_id}", time.time() + 3600, time.time())

    drained = []
    monkeypatch.setattr(manager, "_spawn_per_user_server", spawn)
    monkeypatch.setattr(manager, "_schedule_drain", lambda state, reason: drained.append((state, reason)))
    manager._ensure_per_user_reaper = lambda: None
    replacement_task = asyncio.create_task(
      manager._get_per_user_server("gsheets-mcp", _subject(1), force=True)
    )
    await spawn_started.wait()
    new_user_task = asyncio.create_task(
      manager._get_per_user_server("gsheets-mcp", _subject(2))
    )
    await asyncio.sleep(0)
    assert new_user_task.done()
    spawn_gate.set()
    replacement, new_user = await asyncio.gather(
      replacement_task,
      new_user_task,
      return_exceptions=True,
    )

    assert isinstance(replacement, _PerUserServerState)
    assert isinstance(new_user, _PerUserMcpError)
    assert new_user.code == "sheets_unavailable"
    assert spawn_users == ["1"]
    assert max_accounted == 1
    assert manager._per_user_servers == {("gsheets-mcp", "1"): replacement}
    assert manager._per_user_spawn_reservations == {}
    assert drained == [(old, "forced_refresh")]

  asyncio.run(scenario())


def test_failed_mint_at_capacity_evicts_nobody(monkeypatch):
  async def scenario():
    manager = _manager()
    monkeypatch.setattr(mcp_client_module, "PER_USER_INSTANCE_CAP", 1)
    healthy = await _state(manager, "healthy", time.time() + 3600, time.time())
    manager._per_user_servers[("gsheets-mcp", "1")] = healthy
    drains = []
    monkeypatch.setattr(manager, "_schedule_drain", lambda state, reason: drains.append((state, reason)))

    async def fail(_user):
      raise _PerUserMcpError("sheets_not_connected")

    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", fail)
    try:
      await manager._get_per_user_server("gsheets-mcp", _subject(2))
    except _PerUserMcpError as exc:
      assert exc.code == "sheets_not_connected"
    else:
      raise AssertionError("expected mint failure")
    assert manager._per_user_servers == {("gsheets-mcp", "1"): healthy}
    assert drains == []
    assert manager._per_user_spawn_reservations == {}

  asyncio.run(scenario())


def test_spawn_failure_releases_reservation(monkeypatch):
  async def scenario():
    manager = _manager()
    monkeypatch.setattr(mcp_client_module, "PER_USER_INSTANCE_CAP", 1)
    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", _mint_ok)
    attempts = 0

    async def connect(_name, _config):
      nonlocal attempts
      attempts += 1
      if attempts == 1:
        raise RuntimeError("spawn failed")
      return _child("healthy")

    monkeypatch.setattr(manager, "_connect_stdio_with_retries", connect)
    try:
      await manager._get_per_user_server("gsheets-mcp", _subject(1))
    except RuntimeError as exc:
      assert str(exc) == "spawn failed"
    else:
      raise AssertionError("expected spawn failure")
    assert manager._per_user_spawn_reservations == {}
    state = await manager._get_per_user_server("gsheets-mcp", _subject(2))
    assert state.server.name == "healthy"
    assert manager._per_user_spawn_reservations == {}

  asyncio.run(scenario())


def test_replacement_spawn_failure_restores_old_and_releases_reservation(monkeypatch):
  async def scenario():
    manager = _manager()
    monkeypatch.setattr(mcp_client_module, "PER_USER_INSTANCE_CAP", 1)
    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", _mint_ok)
    old = await _state(manager, "old", time.time() + 3600, time.time())
    manager._per_user_servers[("gsheets-mcp", "1")] = old
    attempts = 0

    async def spawn(_server, user, broker_session=None):
      nonlocal attempts
      del broker_session
      attempts += 1
      if attempts == 1:
        raise RuntimeError("replacement failed")
      return await _state(manager, f"healthy-{user.user_id}", time.time() + 3600, time.time())

    drained = []
    monkeypatch.setattr(manager, "_spawn_per_user_server", spawn)
    monkeypatch.setattr(manager, "_schedule_drain", lambda state, reason: drained.append((state, reason)))
    manager._ensure_per_user_reaper = lambda: None
    try:
      await manager._get_per_user_server("gsheets-mcp", _subject(1), force=True)
    except RuntimeError as exc:
      assert str(exc) == "replacement failed"
    else:
      raise AssertionError("expected replacement spawn failure")

    assert manager._per_user_servers == {("gsheets-mcp", "1"): old}
    assert old.draining is False
    assert manager._per_user_spawn_reservations == {}
    assert drained == []

    added = await manager._get_per_user_server("gsheets-mcp", _subject(2))
    assert manager._per_user_servers == {("gsheets-mcp", "2"): added}
    assert added.server.name == "healthy-2"
    assert manager._per_user_spawn_reservations == {}
    assert drained == [(old, "instance_cap_eviction")]

  asyncio.run(scenario())


def test_expired_broker_child_is_discarded_when_replacement_spawn_fails(monkeypatch):
  async def scenario():
    manager = _manager()
    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", _mint_ok)
    old = await _state(manager, "expired", time.time() + 3600, time.time())
    manager._per_user_servers[("gsheets-mcp", "1")] = old
    drained = []

    async def fail_spawn(*_args, **_kwargs):
      raise RuntimeError("replacement failed")

    monkeypatch.setattr(manager, "_spawn_per_user_server", fail_spawn)
    monkeypatch.setattr(manager, "_schedule_drain", lambda state, reason: drained.append((state, reason)))
    manager._ensure_per_user_reaper = lambda: None

    try:
      await manager._get_per_user_server(
        "gsheets-mcp",
        _subject(1),
        force=True,
        discard_current_on_failure=True,
      )
    except RuntimeError as exc:
      assert str(exc) == "replacement failed"
    else:
      raise AssertionError("expected replacement spawn failure")

    assert manager._per_user_servers == {}
    assert drained == [(old, "replacement_spawn_failed")]
    assert manager._per_user_spawn_reservations == {}

  asyncio.run(scenario())


def test_expired_broker_child_is_discarded_when_replacement_mint_fails(monkeypatch):
  async def scenario():
    manager = _manager()
    old = await _state(manager, "expired", time.time() + 3600, time.time())
    manager._per_user_servers[("gsheets-mcp", "1")] = old
    drained = []

    async def fail_mint(_user):
      raise _PerUserMcpError("sheets_unavailable", "broker unavailable")

    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", fail_mint)
    monkeypatch.setattr(manager, "_schedule_drain", lambda state, reason: drained.append((state, reason)))

    try:
      await manager._get_per_user_server(
        "gsheets-mcp",
        _subject(1),
        force=True,
        discard_current_on_failure=True,
      )
    except _PerUserMcpError as exc:
      assert exc.code == "sheets_unavailable"
    else:
      raise AssertionError("expected replacement mint failure")

    assert manager._per_user_servers == {}
    assert drained == [(old, "broker_mint_failed")]
    assert manager._per_user_spawn_reservations == {}

  asyncio.run(scenario())


def test_eviction_uses_lru_idle_instance_from_same_server(monkeypatch):
  async def scenario():
    manager = _manager()
    manager._servers["other-mcp"] = _definition()
    monkeypatch.setattr(mcp_client_module, "PER_USER_INSTANCE_CAP", 2)
    now = time.time()
    old = await _state(manager, "old", now + 3600, now - 30)
    newer = await _state(manager, "newer", now + 3600, now - 10)
    other = await _state(manager, "other", now + 3600, now - 100)
    manager._per_user_servers.update({
      ("gsheets-mcp", "1"): old,
      ("gsheets-mcp", "2"): newer,
      ("other-mcp", "1"): other,
    })
    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", _mint_ok)
    monkeypatch.setattr(manager, "_connect_stdio_with_retries", lambda *_: asyncio.sleep(0, result=_child("added")))
    drained = []
    monkeypatch.setattr(manager, "_schedule_drain", lambda state, reason: drained.append((state, reason)))

    await manager._get_per_user_server("gsheets-mcp", _subject(3))
    assert ("gsheets-mcp", "1") not in manager._per_user_servers
    assert manager._per_user_servers[("gsheets-mcp", "2")] is newer
    assert manager._per_user_servers[("other-mcp", "1")] is other
    assert drained == [(old, "instance_cap_eviction")]

  asyncio.run(scenario())


def test_periodic_reaper_drains_idle_instance_and_retires_lock(monkeypatch):
  async def scenario():
    manager = _manager()
    monkeypatch.setattr(mcp_client_module, "PER_USER_IDLE_REAP_SECONDS", 0.02)
    monkeypatch.setattr(mcp_client_module, "PER_USER_REAPER_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", _mint_ok)
    monkeypatch.setattr(manager, "_connect_stdio_with_retries", lambda *_: asyncio.sleep(0, result=_child("idle")))
    closed = asyncio.Event()

    async def close(_contexts):
      closed.set()

    monkeypatch.setattr(manager, "_close_contexts", close)
    state = await manager._get_per_user_server("gsheets-mcp", _subject())
    state.last_used_at = time.time() - 1
    await asyncio.wait_for(closed.wait(), timeout=1)
    assert ("gsheets-mcp", "7") not in manager._per_user_servers
    assert ("gsheets-mcp", "7") not in manager._per_user_spawn_locks
    assert manager._per_user_reaper_task is not None
    await manager.shutdown()
    assert manager._per_user_reaper_task is None

  asyncio.run(scenario())


def test_transport_exception_cleanup_uses_normalized_user_id(monkeypatch):
  async def scenario():
    manager = _manager()
    state = await _state(manager, "first", time.time() + 3600, time.time())
    resolved_users = []

    async def resolve(server, user, force=False):
      del force
      resolved_users.append(user.user_id)
      manager._per_user_servers[(server, user.user_id)] = state
      return state

    async def fail(**_kwargs):
      raise EOFError("transport failed with sensitive upstream detail")

    monkeypatch.setattr(manager, "_get_per_user_server", resolve)
    monkeypatch.setattr(manager, "_call_tool_once", fail)
    monkeypatch.setattr(manager, "_close_contexts", lambda *_: asyncio.sleep(0))
    result, error = await manager.call_tool(
      "tool", {}, gateway_session=_gateway_session()
    )
    assert result is None
    assert error is not None
    assert error["sub_code"] == "sheets_transport_error"
    assert error["message"] == "The Google Sheets connection was lost before a read result was received."
    assert "sensitive" not in json.dumps(error)
    assert resolved_users == ["7"]
    assert ("gsheets-mcp", "7") not in manager._per_user_servers
    await asyncio.gather(*manager._drain_tasks)

  asyncio.run(scenario())


def test_transport_failure_during_failed_replacement_does_not_restore_old(monkeypatch):
  async def scenario():
    manager = _manager("gsheets_write_range")
    old = await _state(manager, "old", time.time() + 3600, time.time())
    manager._per_user_servers[("gsheets-mcp", "7")] = old
    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", _mint_ok)
    manager._ensure_per_user_reaper = lambda: None
    call_started = asyncio.Event()
    fail_call = asyncio.Event()
    spawn_started = asyncio.Event()
    spawn_gate = asyncio.Event()
    close_started = asyncio.Event()
    close_gate = asyncio.Event()
    spawn_attempts = 0
    close_calls = 0

    async def invoke(**_kwargs):
      call_started.set()
      await fail_call.wait()
      raise EOFError("transport failed")

    async def spawn(_server, user, broker_session=None):
      nonlocal spawn_attempts
      del broker_session
      spawn_attempts += 1
      if spawn_attempts == 1:
        spawn_started.set()
        await spawn_gate.wait()
        raise RuntimeError("replacement failed")
      return await _state(manager, f"healthy-{user.user_id}", time.time() + 3600, time.time())

    async def close(_contexts):
      nonlocal close_calls
      close_calls += 1
      close_started.set()
      await close_gate.wait()

    monkeypatch.setattr(manager, "_call_tool_once", invoke)
    monkeypatch.setattr(manager, "_spawn_per_user_server", spawn)
    monkeypatch.setattr(manager, "_close_contexts", close)
    call_task = asyncio.create_task(
      manager.call_tool("tool", {}, gateway_session=_gateway_session())
    )
    await call_started.wait()
    replacement_task = asyncio.create_task(
      manager._get_per_user_server("gsheets-mcp", _subject(), force=True)
    )
    await spawn_started.wait()
    assert ("gsheets-mcp", "7") not in manager._per_user_servers

    fail_call.set()
    result, error = await call_task
    assert result is None
    assert error is not None
    assert error["sub_code"] == "mutation_outcome_uncertain"
    assert error["data"]["error"]["outcome"] == {
      "state": "uncertain",
      "phase": "dispatch",
      "mutation_may_have_occurred": True,
    }
    assert error["data"]["error"]["retry"]["safe"] is False
    assert error["data"]["error"]["retry"]["automatic"] is False
    assert old.draining is True
    await close_started.wait()
    assert close_calls == 1

    spawn_gate.set()
    try:
      await replacement_task
    except RuntimeError as exc:
      assert str(exc) == "replacement failed"
    else:
      raise AssertionError("expected replacement spawn failure")
    assert ("gsheets-mcp", "7") not in manager._per_user_servers
    assert manager._per_user_spawn_reservations == {}
    assert close_calls == 1

    healthy = await manager._get_per_user_server("gsheets-mcp", _subject())
    assert manager._per_user_servers[("gsheets-mcp", "7")] is healthy
    assert healthy.server.name == "healthy-7"
    assert manager._per_user_spawn_reservations == {}
    close_gate.set()
    await asyncio.gather(*manager._drain_tasks)
    assert close_calls == 1

  asyncio.run(scenario())


def test_stale_transport_failure_preserves_inserted_replacement(monkeypatch):
  async def scenario():
    manager = _manager("gsheets_write_range")
    old = await _state(manager, "old", time.time() + 3600, time.time())
    replacement = await _state(manager, "replacement", time.time() + 3600, time.time())
    manager._per_user_servers[("gsheets-mcp", "7")] = old
    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", _mint_ok)
    manager._ensure_per_user_reaper = lambda: None
    call_started = asyncio.Event()
    fail_call = asyncio.Event()
    close_calls = 0

    async def invoke(**_kwargs):
      call_started.set()
      await fail_call.wait()
      raise EOFError("stale transport failed")

    async def close(_contexts):
      nonlocal close_calls
      close_calls += 1

    monkeypatch.setattr(manager, "_call_tool_once", invoke)
    monkeypatch.setattr(manager, "_spawn_per_user_server", lambda *_, **__: asyncio.sleep(0, result=replacement))
    monkeypatch.setattr(manager, "_close_contexts", close)
    call_task = asyncio.create_task(
      manager.call_tool("tool", {}, gateway_session=_gateway_session())
    )
    await call_started.wait()

    current = await manager._get_per_user_server("gsheets-mcp", _subject(), force=True)
    assert current is replacement
    assert manager._per_user_servers[("gsheets-mcp", "7")] is replacement
    assert old.draining is True
    assert len(manager._drain_tasks) == 1

    fail_call.set()
    result, error = await call_task
    assert result is None
    assert error is not None
    assert error["sub_code"] == "mutation_outcome_uncertain"
    assert manager._per_user_servers[("gsheets-mcp", "7")] is replacement
    await asyncio.gather(*manager._drain_tasks)
    assert close_calls == 1

  asyncio.run(scenario())


def test_schedule_drain_is_idempotent(monkeypatch):
  async def scenario():
    manager = _manager()
    state = await _state(manager, "old", time.time() + 3600, time.time())
    close_calls = 0

    async def close(_contexts):
      nonlocal close_calls
      close_calls += 1

    monkeypatch.setattr(manager, "_close_contexts", close)
    manager._schedule_drain(state, "idle_reap")
    manager._schedule_drain(state, "idle_reap")
    assert state.draining is True
    assert len(manager._drain_tasks) == 1
    await asyncio.gather(*manager._drain_tasks)
    assert close_calls == 1

  asyncio.run(scenario())


def test_broker_session_expired_live_shape_respawns_and_retries_once(monkeypatch):
  async def scenario():
    manager = _manager()
    monkeypatch.setattr(manager, "_close_contexts", lambda *_: asyncio.sleep(0))
    first = await _state(manager, "first", time.time() + 3600, time.time())
    second = await _state(manager, "second", time.time() + 3600, time.time())
    calls = []

    async def resolve(_server, user, force=False, discard_current_on_failure=False):
      calls.append((user.user_id, force, discard_current_on_failure))
      return second if force else first

    async def invoke(**kwargs):
      label = kwargs["server"].name
      if label == "first":
        return _sheets_error_result()
      return _tool_result(structured_content={
        "status": "ok",
        "operation": "gsheets_read_range",
        "spreadsheet": "sheet-id",
        "range": "Data!A1:B2",
        "values": [[1, 2]],
      })

    monkeypatch.setattr(manager, "_get_per_user_server", resolve)
    monkeypatch.setattr(manager, "_call_tool_once", invoke)
    result, error = await manager.call_tool(
      "tool", {}, gateway_session=_gateway_session()
    )
    assert result == {
      "status": "ok",
      "operation": "gsheets_read_range",
      "spreadsheet": "sheet-id",
      "range": "Data!A1:B2",
      "values": [[1, 2]],
    }
    assert error is None
    assert calls == [("7", False, False), ("7", True, True)]

  asyncio.run(scenario())


def test_broker_session_expired_no_uncertain_replay_refreshes_future_only(monkeypatch):
  async def scenario():
    manager = _manager()
    monkeypatch.setattr(manager, "_close_contexts", lambda *_: asyncio.sleep(0))
    first = await _state(manager, "first", time.time() + 3600, time.time())
    second = await _state(manager, "second", time.time() + 3600, time.time())
    resolves = []
    sends = []

    async def resolve(_server, user, force=False, discard_current_on_failure=False):
      resolves.append((user.user_id, force, discard_current_on_failure))
      return second if force else first

    async def invoke(**kwargs):
      sends.append(kwargs["server"].name)
      return _sheets_error_result()

    monkeypatch.setattr(manager, "_get_per_user_server", resolve)
    monkeypatch.setattr(manager, "_call_tool_once", invoke)
    result, error = await manager.call_tool(
      "tool",
      {},
      gateway_session=_gateway_session(),
      allow_uncertain_replay=False,
    )

    assert result is None
    assert error is not None
    assert error["sub_code"] == "broker_session_expired"
    assert sends == ["first"]
    assert resolves == [("7", False, False), ("7", True, True)]

  asyncio.run(scenario())


def test_future_dispatch_reuses_refreshed_per_user_sheets_session(monkeypatch, owner_session_host_policy):
  async def scenario():
    manager = _manager()
    manager._tool_to_server = {"gsheets_read_range": "gsheets-mcp"}
    manager._prefixed_to_original = {}
    manager._mcp_tool_names = {"gsheets_read_range"}
    monkeypatch.setattr(manager, "_close_contexts", lambda *_: asyncio.sleep(0))
    first = await _state(manager, "first", time.time() + 3600, time.time())
    second = await _state(manager, "second", time.time() + 3600, time.time())
    current = first
    sends = []

    async def resolve(_server, _user, force=False, discard_current_on_failure=False):
      nonlocal current
      del discard_current_on_failure
      if force:
        current = second
      return current

    async def invoke(**kwargs):
      sends.append(kwargs["server"].name)
      if kwargs["server"].name == "first":
        return _sheets_error_result()
      return _tool_result(structured_content={
        "status": "ok",
        "operation": "gsheets_read_range",
        "spreadsheet": "sheet-id",
        "range": "Data!A1",
        "values": [[1]],
      })

    monkeypatch.setattr(manager, "_get_per_user_server", resolve)
    monkeypatch.setattr(manager, "_call_tool_once", invoke)
    dispatcher = ToolDispatcher(
      mcp_client=manager,
      local_tool_handlers={},
      role="owner",
      session=_gateway_session(),
      get_tool_definitions=lambda: [{"name": "gsheets_read_range"}],
      allowed_mcp_tools_by_server={"gsheets-mcp": {"gsheets_read_range"}},
    )

    first_result, first_error = await dispatcher.dispatch(
      "first-call",
      "gsheets_read_range",
      {},
      advertised_tool_names=frozenset({"gsheets_read_range"}),
      allow_uncertain_mcp_replay=False,
    )
    second_result, second_error = await dispatcher.dispatch(
      "second-call",
      "gsheets_read_range",
      {},
      advertised_tool_names=frozenset({"gsheets_read_range"}),
      allow_uncertain_mcp_replay=False,
    )

    assert first_result is None
    assert first_error is not None
    assert first_error["sub_code"] == "broker_session_expired"
    assert second_error is None
    assert isinstance(second_result, dict)
    assert second_result["status"] == "ok"
    assert sends == ["first", "second"]

  asyncio.run(scenario())


def test_broker_session_expired_second_failure_is_typed_after_one_respawn(monkeypatch):
  async def scenario():
    manager = _manager()
    monkeypatch.setattr(manager, "_close_contexts", lambda *_: asyncio.sleep(0))
    first = await _state(manager, "first", time.time() + 3600, time.time())
    second = await _state(manager, "second", time.time() + 3600, time.time())
    resolves = []

    async def resolve(_server, user, force=False, discard_current_on_failure=False):
      resolves.append((user.user_id, force, discard_current_on_failure))
      return second if force else first

    async def invoke(**kwargs):
      del kwargs
      return _sheets_error_result()

    monkeypatch.setattr(manager, "_get_per_user_server", resolve)
    monkeypatch.setattr(manager, "_call_tool_once", invoke)
    result, error = await manager.call_tool(
      "tool", {}, gateway_session=_gateway_session()
    )
    assert result is None
    assert error is not None
    assert error["code"] == "mcp_tool_error"
    assert error["sub_code"] == "broker_session_expired"
    assert error["data"]["error"]["retry"]["automatic"] is True
    assert resolves == [("7", False, False), ("7", True, True)]
    await asyncio.gather(*manager._drain_tasks)

  asyncio.run(scenario())


def test_broker_expiry_backstop_does_not_substring_match_arbitrary_text(monkeypatch):
  async def scenario():
    manager = _manager()
    state = await _state(manager, "first", time.time() + 3600, time.time())
    resolves = []

    async def resolve(_server, _user, force=False, discard_current_on_failure=False):
      del discard_current_on_failure
      resolves.append(force)
      return state

    monkeypatch.setattr(manager, "_get_per_user_server", resolve)
    monkeypatch.setattr(manager, "_call_tool_once", lambda **_: asyncio.sleep(0, result=_ToolResult(
      is_error=True,
      structured_content=None,
      content=[TextContent(type="text", text="untyped broker_session_expired note")],
    )))
    result, error = await manager.call_tool(
      "tool", {}, gateway_session=_gateway_session()
    )
    assert result is None
    assert error is not None
    assert error["code"] == "mcp_tool_error"
    assert resolves == [False]

  asyncio.run(scenario())


def test_broker_session_expiry_never_replays_mutation_but_replaces_child(monkeypatch):
  async def scenario():
    manager = _manager("gsheets_append_rows")
    first = await _state(manager, "first", time.time() + 3600, time.time())
    second = await _state(manager, "second", time.time() + 3600, time.time())
    resolves = []
    dispatches = []

    async def resolve(_server, user, force=False, discard_current_on_failure=False):
      resolves.append((user.user_id, force, discard_current_on_failure))
      return second if force else first

    async def invoke(**kwargs):
      dispatches.append(kwargs["server"].name)
      return _sheets_error_result(
        operation="gsheets_append_rows",
        retry_safe=True,
        retry_automatic=True,
      )

    monkeypatch.setattr(manager, "_get_per_user_server", resolve)
    monkeypatch.setattr(manager, "_call_tool_once", invoke)
    result, error = await manager.call_tool(
      "tool", {"values": [[1]]}, gateway_session=_gateway_session()
    )

    assert result is None
    assert error is not None
    assert error["sub_code"] == "broker_session_expired"
    assert dispatches == ["first"]
    assert resolves == [("7", False, False), ("7", True, True)]

  asyncio.run(scenario())


def test_broker_session_expiry_read_requires_every_automatic_retry_marker(monkeypatch):
  async def scenario():
    for expired_result in (
      _sheets_error_result(retry_safe=False),
      _sheets_error_result(retry_automatic=False),
      _sheets_error_result(outcome_state="unchanged"),
    ):
      manager = _manager("gsheets_read_range")
      first = await _state(manager, "first", time.time() + 3600, time.time())
      second = await _state(manager, "second", time.time() + 3600, time.time())
      dispatches = []
      replacements = []

      async def resolve(_server, _user, force=False, discard_current_on_failure=False):
        if force:
          replacements.append(discard_current_on_failure)
        return second if force else first

      async def invoke(**kwargs):
        dispatches.append(kwargs["server"].name)
        return expired_result

      monkeypatch.setattr(manager, "_get_per_user_server", resolve)
      monkeypatch.setattr(manager, "_call_tool_once", invoke)
      result, error = await manager.call_tool(
        "tool", {}, gateway_session=_gateway_session()
      )

      assert result is None
      assert error is not None
      assert error["sub_code"] == "broker_session_expired"
      assert dispatches == ["first"]
      assert replacements == [True]

  asyncio.run(scenario())


def test_structured_sheets_error_details_are_preserved_verbatim(monkeypatch):
  async def scenario():
    manager = _manager("gsheets_copy_spreadsheet")
    state = await _state(manager, "first", time.time() + 3600, time.time())
    recovery = {
      "kind": "copy_progress",
      "destination_spreadsheet": "destination-id",
      "confirmed_tabs": ["Data"],
      "remaining_tabs": ["Assumptions"],
    }

    async def resolve(_server, _user, force=False, discard_current_on_failure=False):
      del force, discard_current_on_failure
      return state

    monkeypatch.setattr(manager, "_get_per_user_server", resolve)
    monkeypatch.setattr(manager, "_call_tool_once", lambda **_: asyncio.sleep(
      0,
      result=_sheets_error_result(
        operation="gsheets_copy_spreadsheet",
        code="copy_partial",
        message="The destination exists but the copy did not finish.",
        outcome_state="partial",
        retry_safe=False,
        retry_automatic=False,
        recovery=recovery,
      ),
    ))

    result, error = await manager.call_tool(
      "tool", {}, gateway_session=_gateway_session()
    )

    assert result is None
    assert error is not None
    assert error["sub_code"] == "copy_partial"
    assert error["data"]["operation"] == "gsheets_copy_spreadsheet"
    assert error["data"]["error"]["outcome"]["state"] == "partial"
    assert error["data"]["error"]["recovery"] == recovery

  asyncio.run(scenario())


def test_structured_sheets_error_requires_matching_operation(monkeypatch):
  async def scenario():
    manager = _manager("gsheets_read_range")
    state = await _state(manager, "first", time.time() + 3600, time.time())
    resolves = []

    async def resolve(_server, _user, force=False, discard_current_on_failure=False):
      resolves.append((force, discard_current_on_failure))
      return state

    monkeypatch.setattr(manager, "_get_per_user_server", resolve)
    monkeypatch.setattr(manager, "_call_tool_once", lambda **_: asyncio.sleep(
      0,
      result=_sheets_error_result(operation="gsheets_write_range"),
    ))

    result, error = await manager.call_tool(
      "tool", {}, gateway_session=_gateway_session()
    )

    assert result is None
    assert error is not None
    assert error["sub_code"] == "invalid_sheets_error_contract"
    assert error["data"]["operation"] == "gsheets_read_range"
    assert resolves == [(False, False)]

  asyncio.run(scenario())


def test_structured_sheets_error_requires_complete_contract_shape(monkeypatch):
  async def scenario():
    manager = _manager("gsheets_read_range")
    state = await _state(manager, "first", time.time() + 3600, time.time())
    malformed = _sheets_error_result()
    assert isinstance(malformed.structured_content, dict)
    malformed_error = malformed.structured_content["error"]
    assert isinstance(malformed_error, dict)
    del malformed_error["recovery"]

    async def resolve(_server, _user, force=False, discard_current_on_failure=False):
      del force, discard_current_on_failure
      return state

    monkeypatch.setattr(manager, "_get_per_user_server", resolve)
    monkeypatch.setattr(manager, "_call_tool_once", lambda **_: asyncio.sleep(0, result=malformed))

    result, error = await manager.call_tool(
      "tool", {}, gateway_session=_gateway_session()
    )

    assert result is None
    assert error is not None
    assert error["sub_code"] == "invalid_sheets_error_contract"
    assert error["data"]["error"]["retry"]["automatic"] is False

  asyncio.run(scenario())


class _ScopeLikeContext:
  """An anyio-shaped context: exiting it cancels the task that *entered* it.

  This is the whole of the failure the per-user host exists to prevent. A
  `ClientSession`/`stdio_client` pair is a task group; unwinding it cancels its
  scope, and anyio delivers that cancellation to the entering task.
  """

  def __init__(self):
    self.entered_by = None
    self.exited_by = None

  async def __aenter__(self):
    self.entered_by = asyncio.current_task()
    return self

  async def __aexit__(self, *_exc_info):
    self.exited_by = asyncio.current_task()
    if self.entered_by is not None and self.entered_by is not self.exited_by:
      self.entered_by.cancel()
    return False


def _scope_connect(scope):
  async def connect(label, _config):
    await scope.__aenter__()
    return _ConnectedServerState(
      label,
      _UnusedClientSession(),
      [scope],
      [],
      {"tool"},
      config={"type": "stdio"},
    )

  return connect


def test_per_user_close_never_cancels_the_task_that_opened_the_child(monkeypatch):
  async def scenario():
    manager = _manager()
    manager._ensure_per_user_reaper = lambda: None
    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", _mint_ok)
    scope = _ScopeLikeContext()
    monkeypatch.setattr(manager, "_connect_stdio_with_retries", _scope_connect(scope))

    opened = asyncio.Event()
    release = asyncio.Event()
    held = {}

    async def turn():
      held["state"] = await manager._get_per_user_server("gsheets-mcp", _subject())
      opened.set()
      # The turn goes on living long after the dispatch that opened the child.
      await release.wait()
      return "turn finished on its own"

    turn_task = asyncio.create_task(turn())
    await asyncio.wait_for(opened.wait(), timeout=1)

    state = held["state"]
    assert manager._per_user_servers.pop(("gsheets-mcp", "7")) is state
    manager._schedule_drain(state, "idle_reap")
    await asyncio.wait_for(asyncio.gather(*manager._drain_tasks), timeout=1)

    assert scope.exited_by is not None
    assert scope.entered_by is scope.exited_by
    assert scope.entered_by is not turn_task
    assert turn_task.cancelled() is False
    release.set()
    assert await asyncio.wait_for(turn_task, timeout=1) == "turn finished on its own"

  asyncio.run(scenario())


class _RunawayClock:
  """`monotonic` leaps 30 s per reading; `time` stays real.

  A drain that consults a clock at all crosses any deadline within two
  iterations here, so this fixture fails a restored `PER_USER_DRAIN_TIMEOUT`
  and passes only a drain that waits on `active_calls` alone.
  """

  def __init__(self):
    self.elapsed = 0.0

  def monotonic(self):
    self.elapsed += 30.0
    return self.elapsed

  def time(self):
    return time.time()


def test_drain_waits_for_an_in_flight_call_with_no_deadline(monkeypatch):
  async def scenario():
    manager = _manager()
    monkeypatch.setattr(mcp_client_module, "PER_USER_DRAIN_POLL_SECONDS", 0)
    state = await _state(manager, "busy", active_calls=1)
    closed = []
    monkeypatch.setattr(
      manager,
      "_close_contexts",
      lambda contexts: asyncio.sleep(0, result=closed.append(contexts)),
    )
    monkeypatch.setattr(mcp_client_module, "time", _RunawayClock())
    manager._schedule_drain(state, "idle_reap")
    for _ in range(500):
      await asyncio.sleep(0)
    assert closed == []
    state.active_calls = 0
    await asyncio.wait_for(asyncio.gather(*manager._drain_tasks), timeout=1)
    assert len(closed) == 1

  asyncio.run(scenario())


def test_abandoned_startup_is_closed_by_its_host_and_never_published(monkeypatch):
  """A startup nobody is waiting for still ends in a closed child.

  The host runs the startup to its own end — cancelling it there would
  interrupt the very teardown that reaps the child — and then closes it
  without ever publishing it.
  """

  async def scenario():
    manager = _manager()
    manager._ensure_per_user_reaper = lambda: None
    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", _mint_ok)
    scope = _ScopeLikeContext()
    connecting = asyncio.Event()
    handshake = asyncio.Event()
    connect = _scope_connect(scope)

    async def slow_connect(label, config):
      connecting.set()
      await handshake.wait()
      return await connect(label, config)

    monkeypatch.setattr(manager, "_connect_stdio_with_retries", slow_connect)
    caller = asyncio.create_task(manager._get_per_user_server("gsheets-mcp", _subject()))
    await asyncio.wait_for(connecting.wait(), timeout=1)
    hosts = list(manager._per_user_hosts)
    assert len(hosts) == 1
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
      await caller

    handshake.set()
    await asyncio.wait_for(hosts[0].closed.wait(), timeout=1)
    assert scope.exited_by is hosts[0].task
    assert scope.entered_by is scope.exited_by
    await asyncio.sleep(0)
    assert manager._per_user_hosts == set()
    assert manager._per_user_servers == {}
    assert manager._per_user_spawn_reservations == {}

  asyncio.run(scenario())


def test_no_drain_deadline_constant_remains():
  assert not hasattr(mcp_client_module, "PER_USER_DRAIN_TIMEOUT_SECONDS")


def test_per_user_close_names_the_site_and_the_child(monkeypatch, caplog):
  async def scenario():
    manager = _manager()
    state = await _state(manager, "old", user_id="42")
    monkeypatch.setattr(manager, "_close_contexts", lambda _contexts: asyncio.sleep(0))
    manager._schedule_drain(state, "idle_reap")
    await asyncio.wait_for(asyncio.gather(*manager._drain_tasks), timeout=1)

  with caplog.at_level(logging.INFO, logger="agent_gateway.mcp_client"):
    asyncio.run(scenario())
  messages = [record.getMessage() for record in caplog.records]
  assert any(
    "per-user MCP child gsheets-mcp closed" in message
    and "user=42" in message
    and "site=idle_reap" in message
    for message in messages
  )


def test_abandoned_spawn_is_closed_by_its_own_host(monkeypatch):
  async def scenario():
    manager = _manager()
    manager._ensure_per_user_reaper = lambda: None
    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", _mint_ok)
    scope = _ScopeLikeContext()
    connecting = asyncio.Event()
    finish_connect = asyncio.Event()
    connect = _scope_connect(scope)

    async def slow_connect(label, config):
      connecting.set()
      await finish_connect.wait()
      return await connect(label, config)

    monkeypatch.setattr(manager, "_connect_stdio_with_retries", slow_connect)
    caller = asyncio.create_task(manager._get_per_user_server("gsheets-mcp", _subject()))
    await asyncio.wait_for(connecting.wait(), timeout=1)
    hosts = list(manager._per_user_hosts)
    assert len(hosts) == 1
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
      await caller
    finish_connect.set()
    await asyncio.wait_for(hosts[0].closed.wait(), timeout=1)
    assert scope.entered_by is scope.exited_by
    assert manager._per_user_servers == {}

  asyncio.run(scenario())


def test_a_child_retired_during_its_startup_is_never_published(monkeypatch):
  """Readiness must not hand a caller a child whose close has been asked for."""

  async def scenario():
    manager = _manager()
    manager._ensure_per_user_reaper = lambda: None
    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", _mint_ok)
    scope = _ScopeLikeContext()
    connecting = asyncio.Event()
    handshake = asyncio.Event()
    connect = _scope_connect(scope)

    async def slow_connect(label, config):
      connecting.set()
      await handshake.wait()
      return await connect(label, config)

    monkeypatch.setattr(manager, "_connect_stdio_with_retries", slow_connect)
    caller = asyncio.create_task(manager._get_per_user_server("gsheets-mcp", _subject()))
    await asyncio.wait_for(connecting.wait(), timeout=1)
    host = next(iter(manager._per_user_hosts))

    host.request_close()
    handshake.set()
    with pytest.raises(asyncio.CancelledError):
      await asyncio.wait_for(caller, timeout=1)
    await asyncio.wait_for(host.closed.wait(), timeout=1)
    assert scope.exited_by is host.task
    assert manager._per_user_servers == {}
    assert manager._per_user_reaper_task is None

  asyncio.run(scenario())


def test_abandoned_startup_holds_its_instance_cap_slot(monkeypatch):
  """A child that outlives its caller still occupies a slot.

  The caller releases its own reservation as it unwinds, but the child keeps
  starting up until its host lets it go. A slot freed before the process is
  gone is a slot the cap can hand out twice.
  """

  async def scenario():
    manager = _manager()
    manager._ensure_per_user_reaper = lambda: None
    monkeypatch.setattr(mcp_client_module, "PER_USER_INSTANCE_CAP", 1)
    monkeypatch.setattr(manager, "_mint_gsheets_broker_session", _mint_ok)
    scope = _ScopeLikeContext()
    connecting = asyncio.Event()
    handshake = asyncio.Event()
    connect = _scope_connect(scope)

    async def slow_connect(label, config):
      connecting.set()
      await handshake.wait()
      return await connect(label, config)

    monkeypatch.setattr(manager, "_connect_stdio_with_retries", slow_connect)
    caller = asyncio.create_task(manager._get_per_user_server("gsheets-mcp", _subject(1)))
    await asyncio.wait_for(connecting.wait(), timeout=1)
    host = next(iter(manager._per_user_hosts))
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
      await caller

    assert manager._per_user_spawn_reservations == {"gsheets-mcp": 1}
    with pytest.raises(_PerUserMcpError):
      await manager._get_per_user_server("gsheets-mcp", _subject(2))

    handshake.set()
    await asyncio.wait_for(host.closed.wait(), timeout=1)
    await asyncio.sleep(0)
    assert manager._per_user_spawn_reservations == {}
    assert manager._per_user_hosts == set()

  asyncio.run(scenario())


def test_shutdown_closes_every_live_per_user_child_from_its_host(monkeypatch):
  async def scenario():
    manager = _manager()
    state = await _state(manager, "live")
    manager._per_user_servers[("gsheets-mcp", "7")] = state
    closed = []
    monkeypatch.setattr(
      manager,
      "_close_contexts",
      lambda contexts: asyncio.sleep(0, result=closed.append(contexts)),
    )
    await manager.shutdown()
    assert any(contexts is state.server.exit_contexts for contexts in closed)
    assert manager._per_user_hosts == set()
    assert manager._per_user_servers == {}

  asyncio.run(scenario())
