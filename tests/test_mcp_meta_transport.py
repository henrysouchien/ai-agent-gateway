# ruff: noqa: E402

import asyncio
from datetime import timedelta
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.mcp_client as mcp_client_module
import agent_gateway.tool_dispatcher as tool_dispatcher_module
from agent_gateway import AgentRunner, EventLog
from agent_gateway.approval_policy import RunContext
from agent_gateway.dispatcher_factory import (
  GatewayDispatcherDeps,
  InvocationPrincipal,
  build_tool_dispatcher,
)
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.mcp_client_connections import McpClientSession
from agent_gateway.session import GatewaySession
from agent_gateway.session_capabilities import TEAM_WORKSPACE_WRITE_CAPABILITY
from agent_gateway.tool_dispatcher import ToolDispatcher
from agent_gateway.tool_policy_registry import PreparedToolCall
from tests.capability_execution_test_support import (
  stub_runner_capability_execution,
)


def _run(coro):
  return asyncio.run(coro)


def _dispatch(
  dispatcher: ToolDispatcher,
  tool_call_id: str,
  tool_name: str,
  tool_input: dict[str, Any],
  **kwargs: Any,
):
  return dispatcher.dispatch(
    tool_call_id,
    tool_name,
    tool_input,
    advertised_tool_names=frozenset({tool_name}),
    **kwargs,
  )


def _community_session(*, risk_user_id: int = 900001) -> GatewaySession:
  now = int(time.time())
  return GatewaySession(
    session_id="sess-community",
    api_key_hash="hash-community",
    created_at=now,
    expires_at=now + 300,
    user_id="sia-community",
    risk_user_id=risk_user_id,
    role="invite",
    capabilities=frozenset({TEAM_WORKSPACE_WRITE_CAPABILITY}),
    channel="discord",
  )


class _FakeMcpClient(McpClientManager):
  def __init__(
    self,
    server_name: str = "portfolio-reads-mcp",
    *,
    tool_name: str = "portfolio_tool",
    original_names: dict[str, str] | None = None,
  ) -> None:
    super().__init__(config_path=None)
    self.server_name = server_name
    self.tool_name = tool_name
    self.original_names = original_names or {}
    self.calls: list[dict[str, Any]] = []

  def is_mcp_tool(self, name: str) -> bool:
    return name == self.tool_name

  def get_server_for_tool(self, name: str) -> str | None:
    return self.server_name if name == self.tool_name else None

  def get_original_tool_name(self, name: str) -> str:
    return self.original_names.get(name, name)

  async def call_tool(
    self,
    name: str,
    tool_input: dict[str, Any] | PreparedToolCall,
    meta: dict[str, Any] | None = None,
    abort_event: asyncio.Event | None = None,
    gateway_session: object | None = None,
    allow_uncertain_replay: bool = True,
    trusted_dispatch_scope: Mapping[str, object] | None = None,
  ):
    self.calls.append({"name": name, "tool_input": tool_input, "meta": meta})
    return {"ok": True}, None

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return [_portfolio_tool_def(name=self.tool_name)]



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

class _FakeToolCallResult:
  def __init__(
    self,
    *,
    isError: bool,
    structuredContent: object | None,
    content: object,
  ) -> None:
    self.isError = isError
    self.structuredContent = structuredContent
    self.content = content


class _FakeSession:
  def __init__(self) -> None:
    self.calls: list[dict[str, Any]] = []

  async def call_tool(
    self,
    name: str,
    arguments: dict[str, object],
    *,
    read_timeout_seconds: timedelta,
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
    return _FakeToolCallResult(
      isError=False,
      structuredContent={"ok": True},
      content=None,
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


class _RoundTripResearchMcpClient(McpClientManager):
  def __init__(self) -> None:
    super().__init__(config_path=None)
    self.rows_by_user: dict[str, dict[str, Any]] = {}
    self.calls: list[dict[str, Any]] = []

  def is_mcp_tool(self, name: str) -> bool:
    return name in {"thesis_create", "thesis_read"}

  def get_server_for_tool(self, name: str) -> str | None:
    if name == "thesis_create":
      return "portfolio-writes-mcp"
    if name == "thesis_read":
      return "research-corpus-mcp"
    return None

  def get_original_tool_name(self, name: str) -> str:
    return name

  async def call_tool(
    self,
    name: str,
    tool_input: dict[str, Any] | PreparedToolCall,
    meta: dict[str, Any] | None = None,
    abort_event: asyncio.Event | None = None,
    gateway_session: object | None = None,
    allow_uncertain_replay: bool = True,
    trusted_dispatch_scope: Mapping[str, object] | None = None,
  ):
    assert isinstance(tool_input, dict)
    user_id = str((meta or {}).get("user_id") or "")
    self.calls.append({"name": name, "tool_input": dict(tool_input), "meta": meta})
    if name == "thesis_create":
      row = {"research_file_id": tool_input["research_file_id"], "statement": tool_input["statement"]}
      self.rows_by_user[user_id] = row
      return {"status": "created", "thesis": row}, None
    if name == "thesis_read":
      row = self.rows_by_user.get(user_id)
      if row is None:
        return None, {"code": "not_found", "message": "thesis not found"}
      return {"status": "ok", "thesis": row}, None
    return None, {"code": "unknown_tool", "message": name}


def _dispatch_scope(**overrides: Any) -> dict[str, Any]:
  return {
    "kind": "portfolio",
    "source": "user_selected",
    "portfolio_name": "taxable_combined",
    "portfolio_id": "portfolio-123",
    "display_name": "Taxable Combined",
    **overrides,
  }


def _portfolio_tool_def(
  name: str = "portfolio_tool",
  *,
  properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
  return {
    "name": name,
    "description": "Portfolio tool",
    "input_schema": {
      "type": "object",
      "properties": properties
      if properties is not None
      else {
        "format": {"type": "string"},
        "portfolio_id": {"type": "string"},
        "portfolio_name": {"type": "string"},
      },
    },
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
  monkeypatch: pytest.MonkeyPatch,
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
  monkeypatch.setattr(
    tool_dispatcher_module,
    "authority_policy_denies_tool",
    lambda **_kwargs: False,
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

@pytest.mark.parametrize(
  ("server_name", "tool_name"),
  [
    ("portfolio-writes-mcp", "thesis_create"),
    ("portfolio-producers-mcp", "build_model"),
  ],
)
def test_community_write_mcp_meta_uses_authenticated_team_identity(
  server_name: str,
  tool_name: str,
) -> None:
  mcp = _FakeMcpClient(server_name=server_name, tool_name=tool_name)
  session = _community_session()
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session_id=session.session_id,
    user_id=session.user_id,
    risk_user_id=session.risk_user_id,
    channel=session.channel,
    role=session.role,
    session=session,
    run_context=RunContext(
      user_id=session.user_id,
      request_id="request-community-write",
      session_id=session.session_id,
      profile="community",
      channel="discord",
      run_id="skill-run-community-write",
    ),
    mcp_meta_inject_servers=frozenset({"portfolio-producers-mcp", "portfolio-writes-mcp"}),
  )

  result, error = _run(_dispatch(
    dispatcher,
    "call-1",
    tool_name,
    {"ticker": "MSFT"},
    skill_run_id=(
      "skill-run-community-write" if tool_name == "build_model" else None
    ),
  ))

  assert error is None
  assert result == {"ok": True}
  assert mcp.calls[0]["meta"]["user_id"] == "900001"
  assert mcp.calls[0]["meta"]["user_id"] != "1"


def test_community_thesis_create_then_read_uses_team_store_identity() -> None:
  mcp = _RoundTripResearchMcpClient()
  session = _community_session()
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    session_id=session.session_id,
    user_id=session.user_id,
    risk_user_id=session.risk_user_id,
    channel=session.channel,
    role=session.role,
    session=session,
    run_context=RunContext(
      user_id=session.user_id,
      request_id="request-community-round-trip",
      session_id=session.session_id,
      profile="community",
      channel="discord",
    ),
    mcp_meta_inject_servers=frozenset({"portfolio-writes-mcp", "research-corpus-mcp"}),
    allowed_mcp_tools_by_server={
      "portfolio-writes-mcp": {"thesis_create"},
      "research-corpus-mcp": {"thesis_read"},
    },
  )

  created, create_error = _run(
    _dispatch(
      dispatcher,
      "call-1",
      "thesis_create",
      {"research_file_id": 101, "statement": "Team thesis."},
    )
  )
  read, read_error = _run(_dispatch(dispatcher, "call-2", "thesis_read", {"research_file_id": 101}))

  assert create_error is None
  assert read_error is None
  assert created is not None
  assert read is not None
  assert created["thesis"] == read["thesis"]
  assert set(mcp.rows_by_user) == {"900001"}
  assert all(call["meta"]["user_id"] == "900001" for call in mcp.calls)


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
    session=SimpleNamespace(dispatch_scope=_dispatch_scope(portfolio_id=None)),
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
  assert session.calls[-1]["read_timeout_seconds"].total_seconds() == 300

  result, error = _run(manager.call_tool("portfolio_summary", {}))
  assert error is None
  assert result == {"ok": True}
  assert session.calls[-1]["read_timeout_seconds"].total_seconds() == 120


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
      read_timeout_seconds: timedelta,
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
      return _FakeToolCallResult(
        isError=False,
        structuredContent={"late": True},
        content=None,
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
  assert session.calls[0]["read_timeout_seconds"].total_seconds() == 0.01


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
      read_timeout_seconds: timedelta,
      meta: dict[str, object] | None = None,
    ):
      _ = name, arguments, read_timeout_seconds, meta
      self.started.set()
      try:
        await asyncio.sleep(60)
      except asyncio.CancelledError:
        self.cancelled = True
        raise
      return _FakeToolCallResult(
        isError=False,
        structuredContent={"late": True},
        content=None,
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
        read_timeout_seconds: timedelta,
        meta: dict[str, object] | None = None,
      ):
        _ = name, arguments, read_timeout_seconds, meta
        try:
          await asyncio.sleep(60)
        except asyncio.CancelledError:
          self.cancelled = True
          caller_task.cancel()
          raise
        return _FakeToolCallResult(
          isError=False,
          structuredContent={"late": True},
          content=None,
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
