# ruff: noqa: E402

import asyncio
import hashlib
import hmac
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.mcp_client import (
  McpClientManager,
  _PerUserGatewaySubject,
  _PerUserMcpError,
  _PerUserServerState,
  _ServerState,
)
from agent_gateway.session import GatewaySession
from agent_gateway.tool_dispatcher import ToolDispatcher
import agent_gateway.mcp_client as mcp_client_module


def _definition(config=None):
  return _ServerState("gsheets-mcp", SimpleNamespace(), [], [], {"tool"}, config=config or {
    "command": "/venv/python",
    "args": ["server.py"],
    "per_user": True,
    "env": {"GSHEETS_TOKEN_MODE": "broker", "GSHEETS_HEADLESS": "1"},
  })


def _child(label):
  return _ServerState(label, SimpleNamespace(), [object()], [], {"tool"}, config={"type": "stdio"})


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
  )


def _subject(user_id: int = 7) -> _PerUserGatewaySubject:
  return _PerUserGatewaySubject.from_gateway_session(_gateway_session(user_id))


async def _mint_ok(_user):
  return "tier-two", time.time() + 3600, "https://risk"


def _tool_result(*, is_error=False, payload=None, structured_content=None):
  content = None
  if payload is not None:
    content = [SimpleNamespace(text=json.dumps(payload))]
  return SimpleNamespace(
    isError=is_error,
    structuredContent=structured_content,
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

    async def connect(jobs):
      captured.extend(jobs)
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
  assert error["sub_code"] == "missing_user_identity"
  assert manager._per_user_servers == {}


def test_dispatcher_passes_authenticated_session_instead_of_identity_override():
  async def scenario():
    manager = _manager()
    manager._mcp_tool_names = {"tool"}
    session = _gateway_session()
    captured = {}

    async def call_tool(name, tool_input, **kwargs):
      captured.update(name=name, tool_input=tool_input, kwargs=kwargs)
      return {"ok": True}, None

    manager.call_tool = call_tool
    dispatcher = ToolDispatcher(
      mcp_client=manager,
      session=session,
      risk_user_id=7,
      mcp_identity_overrides={"gsheets-mcp": 999},
    )
    result, error = await dispatcher.dispatch(
      "call-1",
      "tool",
      {},
      advertised_tool_names=frozenset({"tool"}),
    )
    assert error is None
    assert result == {"ok": True}
    assert captured["kwargs"]["gateway_session"] is session
    assert "user_id" not in captured["kwargs"]

  asyncio.run(scenario())


def test_same_user_single_flight_and_different_users_isolate():
  async def scenario():
    manager = _manager()
    spawned = []
    gate = asyncio.Event()

    async def spawn(server, user, broker_session=None):
      del server, broker_session
      spawned.append(user.user_id)
      await gate.wait()
      return _PerUserServerState(_child(user.user_id), time.time() + 3600, time.time())

    manager._mint_gsheets_broker_session = _mint_ok
    manager._spawn_per_user_server = spawn
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

    manager._mint_gsheets_broker_session = mint
    manager._connect_stdio_with_retries = connect
    await manager._spawn_per_user_server("gsheets-mcp", _subject())
    assert captured["GSHEETS_BROKER_SESSION_TOKEN"] == "tier-two"
    assert captured["GSHEETS_BROKER_URL"] == "https://risk"
    assert "GATEWAY_GOOGLE_SHEETS_BROKER_HMAC_KEY" not in captured

  asyncio.run(scenario())


def test_mint_failures_are_terminal_and_do_not_spawn():
  async def scenario(code):
    manager = _manager()

    async def fail(_user):
      raise _PerUserMcpError(code)

    manager._mint_gsheets_broker_session = fail
    result, error = await manager.call_tool(
      "tool", {}, gateway_session=_gateway_session()
    )
    assert result is None
    assert error["sub_code"] == code
    assert error["data"]["error"]["outcome"]["state"] == "not_started"
    assert error["data"]["error"]["outcome"]["mutation_may_have_occurred"] is False
    assert manager._per_user_servers == {}

  asyncio.run(scenario("sheets_not_connected"))
  asyncio.run(scenario("broker_rate_limited"))


def test_near_expiry_replaces_and_drains_old():
  async def scenario():
    manager = _manager()
    old = _PerUserServerState(_child("old"), time.time() + 1, time.time(), active_calls=1)
    manager._per_user_servers[("gsheets-mcp", "7")] = old
    replacement = _PerUserServerState(_child("new"), time.time() + 3600, time.time())
    manager._mint_gsheets_broker_session = _mint_ok
    manager._spawn_per_user_server = lambda *_, **__: asyncio.sleep(0, result=replacement)
    closed = []
    manager._close_contexts = lambda contexts: asyncio.sleep(0, result=closed.append(contexts))
    current = await manager._get_per_user_server("gsheets-mcp", _subject())
    assert current is replacement
    assert old.draining is True
    await asyncio.sleep(0)
    assert closed == []
    old.active_calls = 0
    await asyncio.gather(*manager._drain_tasks)
    assert closed

  asyncio.run(scenario())


def test_idle_reap_and_dead_instance_respawn(monkeypatch):
  async def scenario():
    manager = _manager()
    stale = _PerUserServerState(_child("stale"), time.time() + 3600, 0)
    dead = _PerUserServerState(_child("dead"), time.time() + 3600, time.time())
    dead.server.exit_contexts = []
    manager._per_user_servers[("gsheets-mcp", "1")] = stale
    manager._per_user_servers[("gsheets-mcp", "2")] = dead
    manager._close_contexts = lambda _contexts: asyncio.sleep(0)
    spawned = []

    async def spawn(_server, user, broker_session=None):
      del broker_session
      spawned.append(user.user_id)
      return _PerUserServerState(_child(user.user_id), time.time() + 3600, time.time())

    manager._mint_gsheets_broker_session = _mint_ok
    manager._spawn_per_user_server = spawn
    await manager._get_per_user_server("gsheets-mcp", _subject(2))
    assert "2" in spawned
    assert ("gsheets-mcp", "1") not in manager._per_user_servers

  asyncio.run(scenario())


def test_concurrent_burst_respects_atomic_per_server_cap(monkeypatch):
  async def scenario():
    manager = _manager()
    monkeypatch.setattr(mcp_client_module, "PER_USER_INSTANCE_CAP", 2)
    manager._mint_gsheets_broker_session = _mint_ok
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

    manager._connect_stdio_with_retries = connect
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
    manager._mint_gsheets_broker_session = _mint_ok
    old = _PerUserServerState(_child("old"), time.time() + 3600, time.time())
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
      return _PerUserServerState(
        _child(f"replacement-{user.user_id}"), time.time() + 3600, time.time()
      )

    drained = []
    manager._spawn_per_user_server = spawn
    manager._schedule_drain = drained.append
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
    assert drained == [old]

  asyncio.run(scenario())


def test_failed_mint_at_capacity_evicts_nobody(monkeypatch):
  async def scenario():
    manager = _manager()
    monkeypatch.setattr(mcp_client_module, "PER_USER_INSTANCE_CAP", 1)
    healthy = _PerUserServerState(_child("healthy"), time.time() + 3600, time.time())
    manager._per_user_servers[("gsheets-mcp", "1")] = healthy
    drains = []
    manager._schedule_drain = drains.append

    async def fail(_user):
      raise _PerUserMcpError("sheets_not_connected")

    manager._mint_gsheets_broker_session = fail
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
    manager._mint_gsheets_broker_session = _mint_ok
    attempts = 0

    async def connect(_name, _config):
      nonlocal attempts
      attempts += 1
      if attempts == 1:
        raise RuntimeError("spawn failed")
      return _child("healthy")

    manager._connect_stdio_with_retries = connect
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
    manager._mint_gsheets_broker_session = _mint_ok
    old = _PerUserServerState(_child("old"), time.time() + 3600, time.time())
    manager._per_user_servers[("gsheets-mcp", "1")] = old
    attempts = 0

    async def spawn(_server, user, broker_session=None):
      nonlocal attempts
      del broker_session
      attempts += 1
      if attempts == 1:
        raise RuntimeError("replacement failed")
      return _PerUserServerState(
        _child(f"healthy-{user.user_id}"), time.time() + 3600, time.time()
      )

    drained = []
    manager._spawn_per_user_server = spawn
    manager._schedule_drain = drained.append
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
    assert drained == [old]

  asyncio.run(scenario())


def test_expired_broker_child_is_discarded_when_replacement_spawn_fails():
  async def scenario():
    manager = _manager()
    manager._mint_gsheets_broker_session = _mint_ok
    old = _PerUserServerState(_child("expired"), time.time() + 3600, time.time())
    manager._per_user_servers[("gsheets-mcp", "1")] = old
    drained = []

    async def fail_spawn(*_args, **_kwargs):
      raise RuntimeError("replacement failed")

    manager._spawn_per_user_server = fail_spawn
    manager._schedule_drain = drained.append
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
    assert drained == [old]
    assert manager._per_user_spawn_reservations == {}

  asyncio.run(scenario())


def test_expired_broker_child_is_discarded_when_replacement_mint_fails():
  async def scenario():
    manager = _manager()
    old = _PerUserServerState(_child("expired"), time.time() + 3600, time.time())
    manager._per_user_servers[("gsheets-mcp", "1")] = old
    drained = []

    async def fail_mint(_user):
      raise _PerUserMcpError("sheets_unavailable", "broker unavailable")

    manager._mint_gsheets_broker_session = fail_mint
    manager._schedule_drain = drained.append

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
    assert drained == [old]
    assert manager._per_user_spawn_reservations == {}

  asyncio.run(scenario())


def test_eviction_uses_lru_idle_instance_from_same_server(monkeypatch):
  async def scenario():
    manager = _manager()
    manager._servers["other-mcp"] = _definition()
    monkeypatch.setattr(mcp_client_module, "PER_USER_INSTANCE_CAP", 2)
    now = time.time()
    old = _PerUserServerState(_child("old"), now + 3600, now - 30)
    newer = _PerUserServerState(_child("newer"), now + 3600, now - 10)
    other = _PerUserServerState(_child("other"), now + 3600, now - 100)
    manager._per_user_servers.update({
      ("gsheets-mcp", "1"): old,
      ("gsheets-mcp", "2"): newer,
      ("other-mcp", "1"): other,
    })
    manager._mint_gsheets_broker_session = _mint_ok
    manager._connect_stdio_with_retries = lambda *_: asyncio.sleep(0, result=_child("added"))
    drained = []
    manager._schedule_drain = drained.append

    await manager._get_per_user_server("gsheets-mcp", _subject(3))
    assert ("gsheets-mcp", "1") not in manager._per_user_servers
    assert manager._per_user_servers[("gsheets-mcp", "2")] is newer
    assert manager._per_user_servers[("other-mcp", "1")] is other
    assert drained == [old]

  asyncio.run(scenario())


def test_periodic_reaper_drains_idle_instance_and_retires_lock(monkeypatch):
  async def scenario():
    manager = _manager()
    monkeypatch.setattr(mcp_client_module, "PER_USER_IDLE_REAP_SECONDS", 0.02)
    monkeypatch.setattr(mcp_client_module, "PER_USER_REAPER_INTERVAL_SECONDS", 0.005)
    manager._mint_gsheets_broker_session = _mint_ok
    manager._connect_stdio_with_retries = lambda *_: asyncio.sleep(0, result=_child("idle"))
    closed = asyncio.Event()

    async def close(_contexts):
      closed.set()

    manager._close_contexts = close
    state = await manager._get_per_user_server("gsheets-mcp", _subject())
    state.last_used_at = time.time() - 1
    await asyncio.wait_for(closed.wait(), timeout=1)
    assert ("gsheets-mcp", "7") not in manager._per_user_servers
    assert ("gsheets-mcp", "7") not in manager._per_user_spawn_locks
    assert manager._per_user_reaper_task is not None
    await manager.shutdown()
    assert manager._per_user_reaper_task is None

  asyncio.run(scenario())


def test_transport_exception_cleanup_uses_normalized_user_id():
  async def scenario():
    manager = _manager()
    state = _PerUserServerState(_child("first"), time.time() + 3600, time.time())
    resolved_users = []

    async def resolve(server, user, force=False):
      del force
      resolved_users.append(user.user_id)
      manager._per_user_servers[(server, user.user_id)] = state
      return state

    async def fail(**_kwargs):
      raise EOFError("transport failed with sensitive upstream detail")

    manager._get_per_user_server = resolve
    manager._call_tool_once = fail
    manager._close_contexts = lambda *_: asyncio.sleep(0)
    result, error = await manager.call_tool(
      "tool", {}, gateway_session=_gateway_session()
    )
    assert result is None
    assert error["sub_code"] == "sheets_transport_error"
    assert error["message"] == "The Google Sheets connection was lost before a read result was received."
    assert "sensitive" not in json.dumps(error)
    assert resolved_users == ["7"]
    assert ("gsheets-mcp", "7") not in manager._per_user_servers
    await asyncio.gather(*manager._drain_tasks)

  asyncio.run(scenario())


def test_transport_failure_during_failed_replacement_does_not_restore_old():
  async def scenario():
    manager = _manager("gsheets_write_range")
    old = _PerUserServerState(_child("old"), time.time() + 3600, time.time())
    manager._per_user_servers[("gsheets-mcp", "7")] = old
    manager._mint_gsheets_broker_session = _mint_ok
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
      return _PerUserServerState(
        _child(f"healthy-{user.user_id}"), time.time() + 3600, time.time()
      )

    async def close(_contexts):
      nonlocal close_calls
      close_calls += 1
      close_started.set()
      await close_gate.wait()

    manager._call_tool_once = invoke
    manager._spawn_per_user_server = spawn
    manager._close_contexts = close
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


def test_stale_transport_failure_preserves_inserted_replacement():
  async def scenario():
    manager = _manager("gsheets_write_range")
    old = _PerUserServerState(_child("old"), time.time() + 3600, time.time())
    replacement = _PerUserServerState(_child("replacement"), time.time() + 3600, time.time())
    manager._per_user_servers[("gsheets-mcp", "7")] = old
    manager._mint_gsheets_broker_session = _mint_ok
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

    manager._call_tool_once = invoke
    manager._spawn_per_user_server = lambda *_, **__: asyncio.sleep(0, result=replacement)
    manager._close_contexts = close
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
    assert error["sub_code"] == "mutation_outcome_uncertain"
    assert manager._per_user_servers[("gsheets-mcp", "7")] is replacement
    await asyncio.gather(*manager._drain_tasks)
    assert close_calls == 1

  asyncio.run(scenario())


def test_schedule_drain_is_idempotent():
  async def scenario():
    manager = _manager()
    state = _PerUserServerState(_child("old"), time.time() + 3600, time.time())
    close_calls = 0

    async def close(_contexts):
      nonlocal close_calls
      close_calls += 1

    manager._close_contexts = close
    manager._schedule_drain(state)
    manager._schedule_drain(state)
    assert state.draining is True
    assert len(manager._drain_tasks) == 1
    await asyncio.gather(*manager._drain_tasks)
    assert close_calls == 1

  asyncio.run(scenario())


def test_broker_session_expired_live_shape_respawns_and_retries_once():
  async def scenario():
    manager = _manager()
    manager._close_contexts = lambda *_: asyncio.sleep(0)
    first = _PerUserServerState(_child("first"), time.time() + 3600, time.time())
    second = _PerUserServerState(_child("second"), time.time() + 3600, time.time())
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

    manager._get_per_user_server = resolve
    manager._call_tool_once = invoke
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


def test_broker_session_expired_second_failure_is_typed_after_one_respawn():
  async def scenario():
    manager = _manager()
    manager._close_contexts = lambda *_: asyncio.sleep(0)
    first = _PerUserServerState(_child("first"), time.time() + 3600, time.time())
    second = _PerUserServerState(_child("second"), time.time() + 3600, time.time())
    resolves = []

    async def resolve(_server, user, force=False, discard_current_on_failure=False):
      resolves.append((user.user_id, force, discard_current_on_failure))
      return second if force else first

    async def invoke(**kwargs):
      del kwargs
      return _sheets_error_result()

    manager._get_per_user_server = resolve
    manager._call_tool_once = invoke
    result, error = await manager.call_tool(
      "tool", {}, gateway_session=_gateway_session()
    )
    assert result is None
    assert error["code"] == "mcp_tool_error"
    assert error["sub_code"] == "broker_session_expired"
    assert error["data"]["error"]["retry"]["automatic"] is True
    assert resolves == [("7", False, False), ("7", True, True)]
    await asyncio.gather(*manager._drain_tasks)

  asyncio.run(scenario())


def test_broker_expiry_backstop_does_not_substring_match_arbitrary_text():
  async def scenario():
    manager = _manager()
    state = _PerUserServerState(_child("first"), time.time() + 3600, time.time())
    resolves = []

    async def resolve(_server, _user, force=False, discard_current_on_failure=False):
      del discard_current_on_failure
      resolves.append(force)
      return state

    manager._get_per_user_server = resolve
    manager._call_tool_once = lambda **_: asyncio.sleep(0, result=SimpleNamespace(
      isError=True,
      structuredContent=None,
      content=[SimpleNamespace(text="untyped broker_session_expired note")],
    ))
    result, error = await manager.call_tool(
      "tool", {}, gateway_session=_gateway_session()
    )
    assert result is None
    assert error["code"] == "mcp_tool_error"
    assert resolves == [False]

  asyncio.run(scenario())


def test_broker_session_expiry_never_replays_mutation_but_replaces_child():
  async def scenario():
    manager = _manager("gsheets_append_rows")
    first = _PerUserServerState(_child("first"), time.time() + 3600, time.time())
    second = _PerUserServerState(_child("second"), time.time() + 3600, time.time())
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

    manager._get_per_user_server = resolve
    manager._call_tool_once = invoke
    result, error = await manager.call_tool(
      "tool", {"values": [[1]]}, gateway_session=_gateway_session()
    )

    assert result is None
    assert error["sub_code"] == "broker_session_expired"
    assert dispatches == ["first"]
    assert resolves == [("7", False, False), ("7", True, True)]

  asyncio.run(scenario())


def test_broker_session_expiry_read_requires_every_automatic_retry_marker():
  async def scenario():
    for overrides in (
      {"retry_safe": False},
      {"retry_automatic": False},
      {"outcome_state": "unchanged"},
    ):
      manager = _manager("gsheets_read_range")
      first = _PerUserServerState(_child("first"), time.time() + 3600, time.time())
      second = _PerUserServerState(_child("second"), time.time() + 3600, time.time())
      dispatches = []
      replacements = []

      async def resolve(_server, _user, force=False, discard_current_on_failure=False):
        if force:
          replacements.append(discard_current_on_failure)
        return second if force else first

      async def invoke(**kwargs):
        dispatches.append(kwargs["server"].name)
        return _sheets_error_result(**overrides)

      manager._get_per_user_server = resolve
      manager._call_tool_once = invoke
      result, error = await manager.call_tool(
        "tool", {}, gateway_session=_gateway_session()
      )

      assert result is None
      assert error["sub_code"] == "broker_session_expired"
      assert dispatches == ["first"]
      assert replacements == [True]

  asyncio.run(scenario())


def test_structured_sheets_error_details_are_preserved_verbatim():
  async def scenario():
    manager = _manager("gsheets_copy_spreadsheet")
    state = _PerUserServerState(_child("first"), time.time() + 3600, time.time())
    recovery = {
      "kind": "copy_progress",
      "destination_spreadsheet": "destination-id",
      "confirmed_tabs": ["Data"],
      "remaining_tabs": ["Assumptions"],
    }

    async def resolve(_server, _user, force=False, discard_current_on_failure=False):
      del force, discard_current_on_failure
      return state

    manager._get_per_user_server = resolve
    manager._call_tool_once = lambda **_: asyncio.sleep(
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
    )

    result, error = await manager.call_tool(
      "tool", {}, gateway_session=_gateway_session()
    )

    assert result is None
    assert error["sub_code"] == "copy_partial"
    assert error["data"]["operation"] == "gsheets_copy_spreadsheet"
    assert error["data"]["error"]["outcome"]["state"] == "partial"
    assert error["data"]["error"]["recovery"] == recovery

  asyncio.run(scenario())


def test_structured_sheets_error_requires_matching_operation():
  async def scenario():
    manager = _manager("gsheets_read_range")
    state = _PerUserServerState(_child("first"), time.time() + 3600, time.time())
    resolves = []

    async def resolve(_server, _user, force=False, discard_current_on_failure=False):
      resolves.append((force, discard_current_on_failure))
      return state

    manager._get_per_user_server = resolve
    manager._call_tool_once = lambda **_: asyncio.sleep(
      0,
      result=_sheets_error_result(operation="gsheets_write_range"),
    )

    result, error = await manager.call_tool(
      "tool", {}, gateway_session=_gateway_session()
    )

    assert result is None
    assert error["sub_code"] == "invalid_sheets_error_contract"
    assert error["data"]["operation"] == "gsheets_read_range"
    assert resolves == [(False, False)]

  asyncio.run(scenario())


def test_structured_sheets_error_requires_complete_contract_shape():
  async def scenario():
    manager = _manager("gsheets_read_range")
    state = _PerUserServerState(_child("first"), time.time() + 3600, time.time())
    malformed = _sheets_error_result()
    del malformed.structuredContent["error"]["recovery"]

    async def resolve(_server, _user, force=False, discard_current_on_failure=False):
      del force, discard_current_on_failure
      return state

    manager._get_per_user_server = resolve
    manager._call_tool_once = lambda **_: asyncio.sleep(0, result=malformed)

    result, error = await manager.call_tool(
      "tool", {}, gateway_session=_gateway_session()
    )

    assert result is None
    assert error["sub_code"] == "invalid_sheets_error_contract"
    assert error["data"]["error"]["retry"]["automatic"] is False

  asyncio.run(scenario())
