# ruff: noqa: E402

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.mcp_client as mcp_client_module
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.tool_dispatcher import ToolDispatcher


def _run(coro):
  return asyncio.run(coro)


class _FakeMcpClient:
  def __init__(self, server_name: str = "portfolio-mcp") -> None:
    self.server_name = server_name
    self.calls: list[dict[str, Any]] = []

  def is_mcp_tool(self, name: str) -> bool:
    return name == "portfolio_tool"

  def get_server_for_tool(self, name: str) -> str | None:
    return self.server_name if name == "portfolio_tool" else None

  async def call_tool(self, name: str, tool_input: dict[str, Any], meta: dict[str, Any] | None = None):
    self.calls.append({"name": name, "tool_input": tool_input, "meta": meta})
    return {"ok": True}, None


class _FakeSession:
  def __init__(self) -> None:
    self.calls: list[dict[str, Any]] = []

  async def call_tool(self, name: str, tool_input: dict[str, Any], *, read_timeout_seconds, meta=None):
    self.calls.append(
      {
        "name": name,
        "tool_input": tool_input,
        "read_timeout_seconds": read_timeout_seconds,
        "meta": meta,
      }
    )
    return SimpleNamespace(
      isError=False,
      structuredContent={"ok": True},
      content=None,
    )


@pytest.mark.parametrize("server_name", ["portfolio-mcp", "research-corpus-mcp"])
def test_tool_dispatcher_injects_user_id_into_mcp_meta(server_name: str) -> None:
  mcp = _FakeMcpClient(server_name=server_name)
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session_id="sess-1",
    user_id="alice",
    risk_user_id=42,
    channel="excel",
    role="invite",
    mcp_meta_inject_servers=frozenset({"portfolio-mcp", "research-corpus-mcp"}),
  )

  result, error = _run(dispatcher.dispatch("call-1", "portfolio_tool", {"ticker": "AAPL"}))

  assert error is None
  assert result == {"ok": True}
  assert mcp.calls == [
    {
      "name": "portfolio_tool",
      "tool_input": {"ticker": "AAPL"},
      "meta": {"session_id": "sess-1", "user_id": "42", "channel": "excel", "role": "invite"},
    }
  ]


def test_tool_dispatcher_injects_run_context_into_mcp_meta_when_present() -> None:
  mcp = _FakeMcpClient()
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session_id="sess-1",
    risk_user_id=42,
    channel="excel",
    role="invite",
    mcp_meta_inject_servers=frozenset({"portfolio-mcp"}),
  )

  result, error = _run(
    dispatcher.dispatch(
      "call-1",
      "portfolio_tool",
      {"ticker": "AAPL"},
      skill_run_id="skill-run-123",
      workspace_dir="/tmp/workspace",
    )
  )

  assert error is None
  assert result == {"ok": True}
  assert mcp.calls == [
    {
      "name": "portfolio_tool",
      "tool_input": {"ticker": "AAPL"},
      "meta": {
        "session_id": "sess-1",
        "user_id": "42",
        "channel": "excel",
        "role": "invite",
        "skill_run_id": "skill-run-123",
        "workspace_dir": "/tmp/workspace",
      },
    }
  ]


def test_tool_dispatcher_omits_run_context_from_mcp_meta_when_absent() -> None:
  mcp = _FakeMcpClient()
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session_id="sess-1",
    risk_user_id=42,
    channel="excel",
    role="invite",
    mcp_meta_inject_servers=frozenset({"portfolio-mcp"}),
  )

  result, error = _run(
    dispatcher.dispatch(
      "call-1",
      "portfolio_tool",
      {"ticker": "AAPL"},
    )
  )

  assert error is None
  assert result == {"ok": True}
  assert mcp.calls[0]["meta"] == {
    "session_id": "sess-1",
    "user_id": "42",
    "channel": "excel",
    "role": "invite",
  }
  assert "skill_run_id" not in mcp.calls[0]["meta"]
  assert "workspace_dir" not in mcp.calls[0]["meta"]


def test_tool_dispatcher_session_param_injection_still_works() -> None:
  mcp = _FakeMcpClient(server_name="session-param-server")
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session_id="sess-1",
    mcp_session_inject_servers={"session-param-server"},
  )

  result, error = _run(dispatcher.dispatch("call-1", "portfolio_tool", {"ticker": "AAPL"}))

  assert error is None
  assert result == {"ok": True}
  assert mcp.calls[0]["tool_input"] == {"ticker": "AAPL", "_session_id": "sess-1"}
  assert mcp.calls[0]["meta"] is None


def test_tool_dispatcher_fails_closed_without_user_id_in_strict_mode() -> None:
  mcp = _FakeMcpClient()
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session_id="sess-1",
    user_id=None,
    risk_user_id=None,
    mcp_meta_inject_servers=frozenset({"portfolio-mcp", "research-corpus-mcp"}),
    credentials_resolver_active=True,
  )

  with pytest.raises(RuntimeError, match="MCP meta user_id is required in strict mode"):
    _run(dispatcher.dispatch("call-1", "portfolio_tool", {"ticker": "AAPL"}))

  assert mcp.calls == []


def test_mcp_client_call_tool_forwards_meta_to_underlying_session() -> None:
  manager = McpClientManager(config_path=None)
  session = _FakeSession()
  manager._tool_to_server = {"portfolio_tool": "portfolio-mcp"}
  manager._prefixed_to_original = {"portfolio_tool": "portfolio_tool"}
  manager._servers = {
    "portfolio-mcp": SimpleNamespace(session=session),
  }

  result, error = _run(
    manager.call_tool(
      "portfolio_tool",
      {"ticker": "AAPL"},
      meta={"session_id": "sess-1", "user_id": "42", "channel": "excel", "role": "invite"},
    )
  )

  assert error is None
  assert result == {"ok": True}
  assert session.calls[0]["meta"] == {"session_id": "sess-1", "user_id": "42", "channel": "excel", "role": "invite"}


def test_mcp_client_call_tool_uses_per_tool_timeout_before_server_timeout() -> None:
  manager = McpClientManager(
    config_path=None,
    timeout_overrides={"portfolio-mcp": 120},
    tool_timeout_overrides={"portfolio-mcp.build_model": 300},
  )
  session = _FakeSession()
  manager._tool_to_server = {
    "build_model": "portfolio-mcp",
    "portfolio_summary": "portfolio-mcp",
  }
  manager._prefixed_to_original = {
    "build_model": "build_model",
    "portfolio_summary": "portfolio_summary",
  }
  manager._servers = {
    "portfolio-mcp": SimpleNamespace(session=session),
  }

  result, error = _run(manager.call_tool("build_model", {"research_file_id": 1}))
  assert error is None
  assert result == {"ok": True}
  assert session.calls[-1]["read_timeout_seconds"].total_seconds() == 300

  result, error = _run(manager.call_tool("portfolio_summary", {}))
  assert error is None
  assert result == {"ok": True}
  assert session.calls[-1]["read_timeout_seconds"].total_seconds() == 120


def test_mcp_client_call_tool_enforces_hard_timeout_when_sdk_cancel_is_slow(monkeypatch) -> None:
  monkeypatch.setattr(mcp_client_module, "_MCP_TOOL_CANCEL_GRACE_SECONDS", 0.01)
  manager = McpClientManager(
    config_path=None,
    tool_timeout_overrides={"portfolio-mcp.slow_tool": 0.01},
  )

  class _SlowCancellationSession:
    def __init__(self) -> None:
      self.calls: list[dict[str, Any]] = []
      self.cancelled = False

    async def call_tool(self, name: str, tool_input: dict[str, Any], *, read_timeout_seconds, meta=None):
      self.calls.append(
        {
          "name": name,
          "tool_input": tool_input,
          "read_timeout_seconds": read_timeout_seconds,
          "meta": meta,
        }
      )
      try:
        await asyncio.sleep(60)
      except asyncio.CancelledError:
        self.cancelled = True
        await asyncio.sleep(2)
      return SimpleNamespace(
        isError=False,
        structuredContent={"late": True},
        content=None,
      )

  session = _SlowCancellationSession()
  manager._tool_to_server = {"slow_tool": "portfolio-mcp"}
  manager._prefixed_to_original = {"slow_tool": "slow_tool"}
  manager._servers = {
    "portfolio-mcp": SimpleNamespace(session=session, config=None),
  }

  started = time.monotonic()
  result, error = _run(manager.call_tool("slow_tool", {"ticker": "MSFT"}))

  assert time.monotonic() - started < 0.5
  assert result is None
  assert error is not None
  assert error["sub_code"] == "timeout"
  assert "MCP tool slow_tool timed out after 0.01s" in error["message"]
  assert session.cancelled is True
  assert session.calls[0]["read_timeout_seconds"].total_seconds() == 0.01


def test_mcp_client_call_tool_cancels_sdk_task_when_caller_is_cancelled() -> None:
  manager = McpClientManager(config_path=None)

  class _CancellableSession:
    def __init__(self) -> None:
      self.started = asyncio.Event()
      self.cancelled = False

    async def call_tool(self, name: str, tool_input: dict[str, Any], *, read_timeout_seconds, meta=None):
      _ = name, tool_input, read_timeout_seconds, meta
      self.started.set()
      try:
        await asyncio.sleep(60)
      except asyncio.CancelledError:
        self.cancelled = True
        raise

  async def _run_cancel() -> _CancellableSession:
    session = _CancellableSession()
    manager._tool_to_server = {"slow_tool": "portfolio-mcp"}
    manager._prefixed_to_original = {"slow_tool": "slow_tool"}
    manager._servers = {
      "portfolio-mcp": SimpleNamespace(session=session, config=None),
    }
    task = asyncio.create_task(manager.call_tool("slow_tool", {"ticker": "MSFT"}))
    await session.started.wait()
    task.cancel()
    try:
      await task
    except asyncio.CancelledError:
      pass
    return session

  session = _run(_run_cancel())

  assert session.cancelled is True


def test_mcp_client_call_tool_preserves_caller_cancellation_racing_timeout_cleanup(monkeypatch) -> None:
  monkeypatch.setattr(mcp_client_module, "_MCP_TOOL_CANCEL_GRACE_SECONDS", 1.0)
  manager = McpClientManager(
    config_path=None,
    tool_timeout_overrides={"portfolio-mcp.slow_tool": 0.01},
  )

  async def _run_race() -> bool:
    caller_task = asyncio.current_task()
    assert caller_task is not None

    class _CallerCancellingSession:
      def __init__(self) -> None:
        self.cancelled = False

      async def call_tool(self, name: str, tool_input: dict[str, Any], *, read_timeout_seconds, meta=None):
        _ = name, tool_input, read_timeout_seconds, meta
        try:
          await asyncio.sleep(60)
        except asyncio.CancelledError:
          self.cancelled = True
          caller_task.cancel()
          raise

    session = _CallerCancellingSession()
    manager._tool_to_server = {"slow_tool": "portfolio-mcp"}
    manager._prefixed_to_original = {"slow_tool": "slow_tool"}
    manager._servers = {
      "portfolio-mcp": SimpleNamespace(session=session, config=None),
    }

    try:
      await manager.call_tool("slow_tool", {"ticker": "MSFT"})
    except asyncio.CancelledError:
      return session.cancelled
    return False

  assert _run(_run_race()) is True
