# ruff: noqa: E402

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.types import CallToolResult

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.mcp_client as mcp_client_module
from agent_gateway import AgentRunner, EventLog
from agent_gateway.dispatcher_factory import (
  GatewayDispatcherDeps,
  InvocationPrincipal,
  build_tool_dispatcher,
)
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.mcp_client_connections import McpClientSession
from agent_gateway.session import GatewaySession
from agent_gateway.tool_dispatcher import ToolDispatcher
from gateway_test_support.capability_execution_test_support import (
  stub_runner_capability_execution,
)
from gateway_test_support.host_policy import owner_session_host_policy
from gateway_test_support.mcp_meta_transport import (
  _FakeMcpClient,
  _dispatch,
  _portfolio_tool_def,
  _run,
)


@pytest.fixture(autouse=True)
def _transport_host_policy(owner_session_host_policy):
  # Sessionless MCP fixtures exercise role transport, not local-tool authority.
  owner_session_host_policy.invite_denies_local_tool = lambda _tool: True
  return owner_session_host_policy









class _FakeExcelDispatcher:
  def __init__(self, *, base: ToolDispatcher, **_kwargs: Any) -> None:
    self._base = base


def _build_interactive_dispatcher(
  session: GatewaySession,
  mcp: _FakeMcpClient,
) -> ToolDispatcher:
  wrapped = build_tool_dispatcher(
    GatewayDispatcherDeps(
      mcp_client=mcp,
      approval_store=None,
      approval_policy=None,
      mcp_meta_inject_servers=frozenset({"portfolio-reads-mcp"}),
    ),
    principal=InvocationPrincipal.from_session(session),
    profile="interactive",
    event_log=None,
    session_id=session.session_id,
    request_approval=None,
    needs_approval=lambda *_args: False,
    approved_tool_types=set(),
    local_tool_handlers={},
    get_tool_definitions=mcp.get_tool_definitions,
    session=session,
    credentials_resolver_active=bool(session.auth_config),
    _excel_tool_dispatcher_cls=_FakeExcelDispatcher,
  )
  assert isinstance(wrapped, _FakeExcelDispatcher)
  return wrapped._base

class _FakeSession:
  def __init__(self) -> None:
    self.calls: list[dict[str, Any]] = []

  async def call_tool(
    self,
    name: str,
    arguments: dict[str, object],
    *,
    read_timeout_seconds: float,
    meta: dict[str, object] | None = None,
  ):
    self.calls.append(
      {
        "name": name,
        "tool_input": arguments,
        "read_timeout_seconds": read_timeout_seconds,
        "meta": meta,
      }
    )
    return CallToolResult(
      is_error=False,
      structured_content={"ok": True},
      content=[],
    )


def _server_state(
  session: McpClientSession,
  *,
  name: str = "portfolio-reads-mcp",
) -> mcp_client_module._ServerState:
  return mcp_client_module._ServerState(
    name=name,
    session=session,
    exit_contexts=[],
    tool_definitions=[],
    tool_names=set(),
  )




def _dispatch_scope(**overrides: Any) -> dict[str, Any]:
  return {
    "kind": "portfolio",
    "source": "user_selected",
    "portfolio_name": "taxable_combined",
    "portfolio_id": "portfolio-123",
    "display_name": "Taxable Combined",
    **overrides,
  }




@pytest.mark.parametrize("server_name", ["portfolio-reads-mcp", "research-corpus-mcp"])
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
    mcp_meta_inject_servers=frozenset({"portfolio-reads-mcp", "research-corpus-mcp"}),
  )

  result, error = _run(_dispatch(dispatcher, "call-1", "portfolio_tool", {"ticker": "AAPL"}))

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
      },
    }
  ]


@pytest.mark.parametrize(
  ("entry_path", "channel"),
  [
    ("cli --user-id 1", "cli"),
    ("cli slug henry", "cli"),
    ("mcp_analyst harness", "mcp"),
  ],
)
def test_real_interactive_entry_paths_forward_canonical_risk_user_id(
  entry_path: str,
  channel: str,
) -> None:
  # /chat/init resolves both CLI inputs through the key's GATEWAY_USER_KEYS
  # identity. The mcp_analyst session sidecar identifies the same interactive
  # runtime builder with channel=mcp.
  session = GatewaySession(
    session_id=f"sess-{entry_path}",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id="henry",
    owner_user_id="1",
    raw_user_id="henry",
    user_slug="henry",
    risk_user_id=1,
    role="owner",
    channel=channel,
    auth_config={"provider": "anthropic"},
  )
  mcp = _FakeMcpClient(
    server_name="portfolio-reads-mcp",
    tool_name="get_current_model",
  )
  dispatcher = _build_interactive_dispatcher(session, mcp)

  result, error = _run(
    _dispatch(
      dispatcher,
      "call-1",
      "get_current_model",
      {"research_file_id": 1},
    )
  )

  assert error is None
  assert result == {"ok": True}
  assert mcp.calls[0]["meta"]["user_id"] == "1"


def test_non_positive_risk_user_id_is_not_serialized_as_mcp_identity() -> None:
  mcp = _FakeMcpClient(
    server_name="portfolio-reads-mcp",
    tool_name="get_current_model",
  )
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session_id="sess-legacy",
    user_id="henry",
    risk_user_id=0,
    channel="mcp",
    role="owner",
    mcp_meta_inject_servers=frozenset({"portfolio-reads-mcp"}),
  )

  result, error = _run(
    _dispatch(
      dispatcher,
      "call-1",
      "get_current_model",
      {"research_file_id": 1},
    )
  )

  assert error is None
  assert result == {"ok": True}
  assert mcp.calls[0]["meta"]["user_id"] is None



def test_cli_channel_research_corpus_meta_carries_caller_session_token() -> None:
  # CLI thesis_list must execute as the calling session: meta carries that
  # session's token so research-corpus-mcp never mints a second gateway session.
  session = GatewaySession(
    session_id="sess-cli",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id="henry",
    risk_user_id=1,
    role="owner",
    channel="cli",
    session_token="caller-session-token",
  )
  mcp = _FakeMcpClient(server_name="research-corpus-mcp", tool_name="thesis_list")
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session=session,
    session_id=session.session_id,
    user_id=session.user_id,
    risk_user_id=session.risk_user_id,
    channel=session.channel,
    role=session.role,
    mcp_meta_inject_servers=frozenset({"research-corpus-mcp"}),
  )

  result, error = _run(
    _dispatch(dispatcher, "call-1", "thesis_list", {"ticker": "PCTY", "limit": 10})
  )

  assert error is None
  assert result == {"ok": True}
  assert mcp.calls == [
    {
      "name": "thesis_list",
      "tool_input": {"ticker": "PCTY", "limit": 10},
      "meta": {
        "session_id": "sess-cli",
        "user_id": "1",
        "channel": "cli",
        "role": "owner",
        "session_token": "caller-session-token",
      },
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
    mcp_meta_inject_servers=frozenset({"portfolio-reads-mcp"}),
  )

  result, error = _run(
    _dispatch(
      dispatcher,
      "call-1",
      "portfolio_tool",
      {"ticker": "AAPL"},
      skill_run_id="skill-run-123",
      workspace_dir="/tmp/workspace",
      batch_id=23,
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
        "batch_id": "23",
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
    mcp_meta_inject_servers=frozenset({"portfolio-reads-mcp"}),
  )

  result, error = _run(
    _dispatch(
      dispatcher,
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
  assert "session_token" not in mcp.calls[0]["meta"]
  assert "skill_run_id" not in mcp.calls[0]["meta"]
  assert "workspace_dir" not in mcp.calls[0]["meta"]
  assert "batch_id" not in mcp.calls[0]["meta"]


def test_tool_dispatcher_defaults_portfolio_scope_for_portfolio_mcp_tool() -> None:
  mcp = _FakeMcpClient()
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session_id="sess-1",
    risk_user_id=42,
    channel="web",
    mcp_meta_inject_servers=frozenset({"portfolio-reads-mcp"}),
    session=SimpleNamespace(dispatch_scope=_dispatch_scope(), role="owner"),
    get_tool_definitions=lambda: [_portfolio_tool_def()],
  )

  result, error = _run(_dispatch(dispatcher, "call-1", "portfolio_tool", {"format": "agent"}))

  assert error is None
  assert result == {"ok": True}
  assert mcp.calls[0]["tool_input"] == {
    "format": "agent",
    "portfolio_id": "portfolio-123",
    "portfolio_name": "taxable_combined",
  }
  assert mcp.calls[0]["meta"]["user_id"] == "42"


def test_tool_dispatcher_defaults_portfolio_scope_for_split_portfolio_mcp_tool() -> None:
  mcp = _FakeMcpClient(server_name="portfolio-reads-mcp")
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session_id="sess-1",
    risk_user_id=42,
    channel="web",
    mcp_meta_inject_servers=frozenset({"portfolio-reads-mcp"}),
    session=SimpleNamespace(dispatch_scope=_dispatch_scope(), role="owner"),
    get_tool_definitions=lambda: [_portfolio_tool_def()],
  )

  result, error = _run(_dispatch(dispatcher, "call-1", "portfolio_tool", {"format": "agent"}))

  assert error is None
  assert result == {"ok": True}
  assert mcp.calls[0]["tool_input"] == {
    "format": "agent",
    "portfolio_id": "portfolio-123",
    "portfolio_name": "taxable_combined",
  }
  assert mcp.calls[0]["meta"]["user_id"] == "42"


def test_tool_dispatcher_preserves_explicit_portfolio_tool_input() -> None:
  mcp = _FakeMcpClient()
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session_id="sess-1",
    session=SimpleNamespace(dispatch_scope=_dispatch_scope(), role="owner"),
    get_tool_definitions=lambda: [_portfolio_tool_def()],
  )

  result, error = _run(
    _dispatch(dispatcher, "call-1", "portfolio_tool", {"portfolio_name": "explicit_portfolio"})
  )

  assert error is None
  assert result == {"ok": True}
  assert mcp.calls[0]["tool_input"] == {"portfolio_name": "explicit_portfolio"}


def test_tool_dispatcher_defaults_when_portfolio_tool_input_is_null_or_blank() -> None:
  mcp = _FakeMcpClient()
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session_id="sess-1",
    session=SimpleNamespace(dispatch_scope=_dispatch_scope(), role="owner"),
    get_tool_definitions=lambda: [_portfolio_tool_def()],
  )

  result, error = _run(
    _dispatch(
      dispatcher,
      "call-1",
      "portfolio_tool",
      {"format": "agent", "portfolio_id": None, "portfolio_name": ""},
    )
  )

  assert error is None
  assert result == {"ok": True}
  assert mcp.calls[0]["tool_input"] == {
    "format": "agent",
    "portfolio_id": "portfolio-123",
    "portfolio_name": "taxable_combined",
  }


def test_tool_dispatcher_skips_portfolio_default_when_schema_does_not_accept_it() -> None:
  mcp = _FakeMcpClient()
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session_id="sess-1",
    session=SimpleNamespace(dispatch_scope=_dispatch_scope(), role="owner"),
    get_tool_definitions=lambda: [
      _portfolio_tool_def(properties={"format": {"type": "string"}}),
    ],
  )

  result, error = _run(_dispatch(dispatcher, "call-1", "portfolio_tool", {"format": "agent"}))

  assert error is None
  assert result == {"ok": True}
  assert mcp.calls[0]["tool_input"] == {"format": "agent"}


def test_tool_dispatcher_uses_original_tool_name_for_prefixed_portfolio_default() -> None:
  tool_name = "mcp__portfolio-reads-mcp__get_positions"
  mcp = _FakeMcpClient(
    tool_name=tool_name,
    original_names={tool_name: "get_positions"},
  )
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session_id="sess-1",
    session=SimpleNamespace(dispatch_scope=_dispatch_scope(portfolio_id=None), role="owner"),
    get_tool_definitions=lambda: [
      _portfolio_tool_def(
        name="get_positions",
        properties={
          "format": {"type": "string"},
          "portfolio_name": {"type": "string"},
        },
      ),
    ],
  )

  result, error = _run(_dispatch(dispatcher, "call-1", tool_name, {"format": "agent"}))

  assert error is None
  assert result == {"ok": True}
  assert mcp.calls[0]["tool_input"] == {"format": "agent", "portfolio_name": "taxable_combined"}


def test_runner_tool_start_event_uses_effective_dispatch_scope_input() -> None:
  mcp = _FakeMcpClient(server_name="portfolio-reads-mcp")
  event_log = EventLog()
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session_id="sess-1",
    risk_user_id=42,
    channel="web",
    mcp_meta_inject_servers=frozenset({"portfolio-reads-mcp"}),
    session=SimpleNamespace(dispatch_scope=_dispatch_scope(), role="owner"),
    get_tool_definitions=lambda: [_portfolio_tool_def()],
  )
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=dispatcher,
    session_id="sess-1",
    capability_execution=stub_runner_capability_execution(
      provider=SimpleNamespace(name="stub"),
      auth_config={"api_key": "test-secret"},
      model="stub-model",
      effort="none",
    ),
    mcp_client=mcp,
    get_tool_definitions=lambda: [_portfolio_tool_def()],
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner._execute_single_tool(
    "call-1",
    "portfolio_tool",
    {"format": "agent"},
    {"_request_advertised_tool_names": frozenset({"portfolio_tool"})},
  ))

  start_events = [entry.event for entry in event_log.entries if entry.event.get("type") == "tool_call_start"]
  assert start_events
  assert start_events[0]["tool_input"] == {
    "format": "agent",
    "portfolio_id": "portfolio-123",
    "portfolio_name": "taxable_combined",
  }
  assert mcp.calls[0]["tool_input"] == start_events[0]["tool_input"]


def test_tool_dispatcher_session_param_injection_still_works() -> None:
  mcp = _FakeMcpClient(server_name="session-param-server")
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session_id="sess-1",
    mcp_session_inject_servers={"session-param-server"},
  )

  result, error = _run(_dispatch(dispatcher, "call-1", "portfolio_tool", {"ticker": "AAPL"}))

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
    mcp_meta_inject_servers=frozenset({"portfolio-reads-mcp", "research-corpus-mcp"}),
    credentials_resolver_active=True,
  )

  with pytest.raises(RuntimeError, match="MCP meta user_id is required in strict mode"):
    _run(_dispatch(dispatcher, "call-1", "portfolio_tool", {"ticker": "AAPL"}))

  assert mcp.calls == []


def test_mcp_client_call_tool_forwards_meta_to_underlying_session() -> None:
  manager = McpClientManager(config_path=None)
  session = _FakeSession()
  manager._tool_to_server = {"portfolio_tool": "portfolio-reads-mcp"}
  manager._prefixed_to_original = {"portfolio_tool": "portfolio_tool"}
  manager._servers = {
    "portfolio-reads-mcp": _server_state(session),
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
    timeout_overrides={"portfolio-reads-mcp": 120},
    tool_timeout_overrides={"portfolio-reads-mcp.build_model": 300},
  )
  session = _FakeSession()
  manager._tool_to_server = {
    "build_model": "portfolio-reads-mcp",
    "portfolio_summary": "portfolio-reads-mcp",
  }
  manager._prefixed_to_original = {
    "build_model": "build_model",
    "portfolio_summary": "portfolio_summary",
  }
  manager._servers = {
    "portfolio-reads-mcp": _server_state(session),
  }

  result, error = _run(manager.call_tool("build_model", {"research_file_id": 1}))
  assert error is None
  assert result == {"ok": True}
  assert session.calls[-1]["read_timeout_seconds"] == 300

  result, error = _run(manager.call_tool("portfolio_summary", {}))
  assert error is None
  assert result == {"ok": True}
  assert session.calls[-1]["read_timeout_seconds"] == 120


def test_mcp_client_call_tool_enforces_hard_timeout_when_sdk_cancel_is_slow(monkeypatch) -> None:
  monkeypatch.setattr(mcp_client_module, "_MCP_TOOL_CANCEL_GRACE_SECONDS", 0.01)
  manager = McpClientManager(
    config_path=None,
    tool_timeout_overrides={"portfolio-reads-mcp.slow_tool": 0.01},
  )

  class _SlowCancellationSession:
    def __init__(self) -> None:
      self.calls: list[dict[str, Any]] = []
      self.cancelled = False

    async def call_tool(
      self,
      name: str,
      arguments: dict[str, object],
      *,
      read_timeout_seconds: float,
      meta: dict[str, object] | None = None,
    ):
      self.calls.append(
        {
          "name": name,
          "tool_input": arguments,
          "read_timeout_seconds": read_timeout_seconds,
          "meta": meta,
        }
      )
      try:
        await asyncio.sleep(60)
      except asyncio.CancelledError:
        self.cancelled = True
        await asyncio.sleep(2)
      return CallToolResult(
        is_error=False,
        structured_content={"late": True},
        content=[],
      )

  session = _SlowCancellationSession()
  manager._tool_to_server = {"slow_tool": "portfolio-reads-mcp"}
  manager._prefixed_to_original = {"slow_tool": "slow_tool"}
  manager._servers = {
    "portfolio-reads-mcp": _server_state(session),
  }

  started = time.monotonic()
  result, error = _run(manager.call_tool("slow_tool", {"ticker": "MSFT"}))

  assert time.monotonic() - started < 0.5
  assert result is None
  assert error is not None
  assert error["sub_code"] == "timeout"
  assert "MCP tool slow_tool timed out after 0.01s" in error["message"]
  assert session.cancelled is True
  assert session.calls[0]["read_timeout_seconds"] == 0.01


def test_mcp_client_call_tool_cancels_sdk_task_when_caller_is_cancelled() -> None:
  manager = McpClientManager(config_path=None)

  class _CancellableSession:
    def __init__(self) -> None:
      self.started = asyncio.Event()
      self.cancelled = False

    async def call_tool(
      self,
      name: str,
      arguments: dict[str, object],
      *,
      read_timeout_seconds: float,
      meta: dict[str, object] | None = None,
    ):
      _ = name, arguments, read_timeout_seconds, meta
      self.started.set()
      try:
        await asyncio.sleep(60)
      except asyncio.CancelledError:
        self.cancelled = True
        raise
      return CallToolResult(
        is_error=False,
        structured_content={"late": True},
        content=[],
      )

  async def _run_cancel() -> _CancellableSession:
    session = _CancellableSession()
    manager._tool_to_server = {"slow_tool": "portfolio-reads-mcp"}
    manager._prefixed_to_original = {"slow_tool": "slow_tool"}
    manager._servers = {
      "portfolio-reads-mcp": _server_state(session),
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
    tool_timeout_overrides={"portfolio-reads-mcp.slow_tool": 0.01},
  )

  async def _run_race() -> bool:
    caller_task = asyncio.current_task()
    assert caller_task is not None

    class _CallerCancellingSession:
      def __init__(self) -> None:
        self.cancelled = False

      async def call_tool(
        self,
        name: str,
        arguments: dict[str, object],
        *,
        read_timeout_seconds: float,
        meta: dict[str, object] | None = None,
      ):
        _ = name, arguments, read_timeout_seconds, meta
        try:
          await asyncio.sleep(60)
        except asyncio.CancelledError:
          self.cancelled = True
          caller_task.cancel()
          raise
        return CallToolResult(
          is_error=False,
          structured_content={"late": True},
          content=[],
        )

    session = _CallerCancellingSession()
    manager._tool_to_server = {"slow_tool": "portfolio-reads-mcp"}
    manager._prefixed_to_original = {"slow_tool": "slow_tool"}
    manager._servers = {
      "portfolio-reads-mcp": _server_state(session),
    }

    try:
      await manager.call_tool("slow_tool", {"ticker": "MSFT"})
    except asyncio.CancelledError:
      return session.cancelled
    return False

  assert _run(_run_race()) is True
