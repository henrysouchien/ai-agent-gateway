# ruff: noqa: E402

from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND
import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import McpClientManager, ToolDispatcher
from agent_gateway import tool_dispatcher_runtime as runtime
from agent_gateway.mcp_client import McpInvocationReport
from agent_gateway.tool_policy_registry import PreparedToolCall
from agent_gateway.tool_dispatcher_helpers import ToolResult


class _TestMcpClient(McpClientManager):
  """Nominal MCP manager base for dispatcher-focused test doubles."""

  def __init__(self) -> None:
    super().__init__(config_path=None)



class _ValidationErrorMcpClient(_TestMcpClient):
  def __init__(self, *, exposed_name: str = "get_price_target", original_name: str = "get_price_target") -> None:
    super().__init__()
    self.exposed_name = exposed_name
    self.original_name = original_name

  def is_mcp_tool(self, name: str) -> bool:
    return name == self.exposed_name

  def get_server_for_tool(self, name: str) -> str | None:
    return "portfolio-reads-mcp" if self.is_mcp_tool(name) else None

  def get_original_tool_name(self, name: str) -> str:
    return self.original_name if name == self.exposed_name else name

  async def call_tool(
    self,
    name: str,
    tool_input: dict[str, Any] | PreparedToolCall,
    meta: object | None = None,
    abort_event: asyncio.Event | None = None,
    gateway_session: object | None = None,
    allow_uncertain_replay: bool = True,
    trusted_dispatch_scope: object | None = None,
  ) -> ToolResult:
    assert isinstance(tool_input, dict)
    assert name == self.exposed_name
    assert tool_input == {"ticker": "MSCI"}
    return None, {
      "code": "mcp_tool_error",
      "sub_code": "unknown",
      "message": (
        "2 validation errors for call[get_price_target]\n"
        "research_file_id\n Missing required argument [type=missing_argument]\n"
        "ticker\n Unexpected keyword argument [type=unexpected_keyword_argument]"
      ),
    }


class _NonValidationErrorMcpClient(_ValidationErrorMcpClient):
  async def call_tool(
    self,
    name: str,
    tool_input: dict[str, Any] | PreparedToolCall,
    meta: object | None = None,
    abort_event: asyncio.Event | None = None,
    gateway_session: object | None = None,
    allow_uncertain_replay: bool = True,
    trusted_dispatch_scope: object | None = None,
  ) -> ToolResult:
    assert isinstance(tool_input, dict)
    assert name == self.exposed_name
    assert tool_input == {"ticker": "MSCI"}
    return None, {
      "code": "mcp_tool_error",
      "sub_code": "server_unavailable",
      "message": "MCP server unavailable: portfolio-reads-mcp",
    }


class _ScopedCorpusMcpClient(_TestMcpClient):
  def __init__(self) -> None:
    super().__init__()
    self.calls: list[tuple[str, dict[str, Any]]] = []
    self.call_kwargs: list[dict[str, Any]] = []

  def is_mcp_tool(self, name: str) -> bool:
    return name in {"corpus_search", "corpus_write"}

  def get_server_for_tool(self, name: str) -> str | None:
    return "research-corpus-mcp" if self.is_mcp_tool(name) else None

  async def call_tool(
    self,
    name: str,
    tool_input: dict[str, Any] | PreparedToolCall,
    meta: object | None = None,
    abort_event: asyncio.Event | None = None,
    gateway_session: object | None = None,
    allow_uncertain_replay: bool = True,
    trusted_dispatch_scope: object | None = None,
  ) -> ToolResult:
    assert isinstance(tool_input, dict)
    self.calls.append((name, dict(tool_input)))
    self.call_kwargs.append(
      {"allow_uncertain_replay": allow_uncertain_replay}
    )
    return {"ok": name}, None


class _ReplayControlledMcpClient(_ScopedCorpusMcpClient):
  def __init__(self) -> None:
    super().__init__()
    self.replay_controls: list[bool] = []

  async def call_tool(
    self,
    name: str,
    tool_input: dict[str, Any] | PreparedToolCall,
    meta: object | None = None,
    abort_event: asyncio.Event | None = None,
    gateway_session: object | None = None,
    allow_uncertain_replay: bool = True,
    trusted_dispatch_scope: object | None = None,
  ) -> ToolResult:
    self.replay_controls.append(allow_uncertain_replay)
    return await super().call_tool(
      name,
      tool_input,
      meta,
      abort_event,
      gateway_session,
      allow_uncertain_replay,
      trusted_dispatch_scope,
    )


def test_runtime_normalizes_needs_approval_arities() -> None:
  one_arg = runtime.normalize_needs_approval(lambda name: name == "one")
  two_args = runtime.normalize_needs_approval(lambda name, tool_input: name == tool_input["name"])
  three_args = runtime.normalize_needs_approval(lambda name, _tool_input, qualifier: name == qualifier)

  assert runtime.normalize_needs_approval(None)("any", {}, "") is False
  assert one_arg("one", {"name": "other"}, "") is True
  assert two_args("two", {"name": "two"}, "") is True
  assert three_args("three", {}, "three") is True


def test_tool_dispatcher_forwards_the_exact_replay_control_to_every_client() -> None:
  strict = _ReplayControlledMcpClient()
  strict_dispatcher = ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=strict,
    local_tool_handlers={},
    role="owner",
    get_tool_definitions=lambda: [{"name": "corpus_search"}],
    allowed_mcp_tools_by_server={"research-corpus-mcp": {"corpus_search"}},
  )
  kwargs_only = _ScopedCorpusMcpClient()
  kwargs_dispatcher = ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=kwargs_only,
    local_tool_handlers={},
    role="owner",
    get_tool_definitions=lambda: [{"name": "corpus_search"}],
    allowed_mcp_tools_by_server={"research-corpus-mcp": {"corpus_search"}},
  )

  assert asyncio.run(strict_dispatcher.dispatch(
    "strict",
    "corpus_search",
    {},
    advertised_tool_names=frozenset({"corpus_search"}),
  )) == ({"ok": "corpus_search"}, None)
  assert asyncio.run(strict_dispatcher.dispatch(
    "strict-forbidden",
    "corpus_search",
    {},
    advertised_tool_names=frozenset({"corpus_search"}),
    allow_uncertain_mcp_replay=False,
  )) == ({"ok": "corpus_search"}, None)
  assert asyncio.run(kwargs_dispatcher.dispatch(
    "kwargs",
    "corpus_search",
    {},
    advertised_tool_names=frozenset({"corpus_search"}),
    allow_uncertain_mcp_replay=False,
  )) == ({"ok": "corpus_search"}, None)
  # A forbidden replay reaches the client verbatim, whatever shape its
  # signature has — a client that only names **kwargs must not silently
  # fall back to the permissive default.
  assert strict.replay_controls == [True, False]
  assert kwargs_only.calls == [("corpus_search", {})]
  assert kwargs_only.call_kwargs[-1]["allow_uncertain_replay"] is False


def test_dispatcher_adds_argument_guidance_to_direct_mcp_validation_errors() -> None:
  dispatcher = ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=_ValidationErrorMcpClient(),
    local_tool_handlers={},
    role="owner",
    get_tool_definitions=lambda: [{"name": "get_price_target"}],
  )

  result, error = asyncio.run(
    dispatcher.dispatch(
      "call-1",
      "get_price_target",
      {"ticker": "MSCI"},
      advertised_tool_names=frozenset({"get_price_target"}),
    )
  )

  assert result is None
  assert error is not None
  assert error["code"] == "mcp_tool_error"
  assert "research_file_id" in error["tool_usage_hint"]
  assert "Do not pass ticker" in error["tool_usage_hint"]


def test_dispatcher_does_not_add_argument_guidance_to_non_validation_mcp_errors() -> None:
  dispatcher = ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=_NonValidationErrorMcpClient(),
    local_tool_handlers={},
    role="owner",
    get_tool_definitions=lambda: [{"name": "get_price_target"}],
  )

  result, error = asyncio.run(
    dispatcher.dispatch(
      "call-1",
      "get_price_target",
      {"ticker": "MSCI"},
      advertised_tool_names=frozenset({"get_price_target"}),
    )
  )

  assert result is None
  assert error is not None
  assert error["code"] == "mcp_tool_error"
  assert "tool_usage_hint" not in error


def test_dispatcher_uses_original_tool_name_for_prefixed_mcp_argument_guidance() -> None:
  dispatcher = ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=_ValidationErrorMcpClient(
      exposed_name="portfolio_get_price_target",
      original_name="get_price_target",
    ),
    local_tool_handlers={},
    role="owner",
    get_tool_definitions=lambda: [{"name": "portfolio_get_price_target"}],
  )

  result, error = asyncio.run(
    dispatcher.dispatch(
      "call-1",
      "portfolio_get_price_target",
      {"ticker": "MSCI"},
      advertised_tool_names=frozenset({"portfolio_get_price_target"}),
    )
  )

  assert result is None
  assert error is not None
  assert error["code"] == "mcp_tool_error"
  assert "research_file_id" in error["tool_usage_hint"]
  assert "Do not pass ticker" in error["tool_usage_hint"]


@pytest.mark.parametrize("cancelled", [False, True])
def test_call_mcp_tool_forwards_invocation_report_unchanged(monkeypatch, cancelled) -> None:
  async def scenario():
    manager = McpClientManager(config_path=None)
    report = McpInvocationReport(certainty="settled")
    dispatcher = ToolDispatcher(
      capability_bind=TEST_CAPABILITY_BIND,
      mcp_client=manager,
      local_tool_handlers={},
      role="owner",
      get_tool_definitions=lambda: [],
    )
    seen = []

    async def call_tool(_name, _tool_input, *, invocation_report, **_kwargs):
      seen.append(invocation_report)
      invocation_report.certainty = "unknown"
      if cancelled:
        raise asyncio.CancelledError()
      invocation_report.certainty = "settled"
      return {"ok": True}, None

    monkeypatch.setattr(manager, "call_tool_reporting", call_tool)
    if cancelled:
      with pytest.raises(asyncio.CancelledError):
        await dispatcher._call_mcp_tool("read", {}, invocation_report=report)
      assert report.certainty == "unknown"
    else:
      assert await dispatcher._call_mcp_tool(
        "read", {}, invocation_report=report,
      ) == ({"ok": True}, None)
      assert report.certainty == "settled"
    assert seen == [report]
    assert seen[0] is report

  asyncio.run(scenario())
