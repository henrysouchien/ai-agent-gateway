# ruff: noqa: E402

import asyncio
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import ToolDispatcher
from agent_gateway import tool_dispatcher_runtime as runtime


class _NullMcpClient:
  def is_mcp_tool(self, _tool_name: str) -> bool:
    return False

  def get_server_for_tool(self, _tool_name: str) -> str | None:
    return None

  async def call_tool(self, _tool_name: str, _tool_input: dict[str, Any]):
    raise AssertionError("MCP should not execute in runtime helper tests")


class _ValidationErrorMcpClient:
  def __init__(self, *, exposed_name: str = "get_price_target", original_name: str = "get_price_target") -> None:
    self.exposed_name = exposed_name
    self.original_name = original_name

  def is_mcp_tool(self, tool_name: str) -> bool:
    return tool_name == self.exposed_name

  def get_server_for_tool(self, tool_name: str) -> str | None:
    return "portfolio-reads-mcp" if self.is_mcp_tool(tool_name) else None

  def get_original_tool_name(self, tool_name: str) -> str:
    return self.original_name if tool_name == self.exposed_name else tool_name

  async def call_tool(self, tool_name: str, tool_input: dict[str, Any], **_kwargs: Any):
    assert tool_name == self.exposed_name
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
  async def call_tool(self, tool_name: str, tool_input: dict[str, Any], **_kwargs: Any):
    assert tool_name == self.exposed_name
    assert tool_input == {"ticker": "MSCI"}
    return None, {
      "code": "mcp_tool_error",
      "sub_code": "server_unavailable",
      "message": "MCP server unavailable: portfolio-reads-mcp",
    }


class _ScopedCorpusMcpClient:
  def __init__(self) -> None:
    self.calls: list[tuple[str, dict[str, Any]]] = []

  def is_mcp_tool(self, tool_name: str) -> bool:
    return tool_name in {"corpus_search", "corpus_write"}

  def get_server_for_tool(self, tool_name: str) -> str | None:
    return "research-corpus-mcp" if self.is_mcp_tool(tool_name) else None

  async def call_tool(self, tool_name: str, tool_input: dict[str, Any], **_kwargs: Any):
    self.calls.append((tool_name, dict(tool_input)))
    return {"ok": tool_name}, None


def test_runtime_normalizes_needs_approval_arities() -> None:
  one_arg = runtime.normalize_needs_approval(lambda name: name == "one")
  two_args = runtime.normalize_needs_approval(lambda name, tool_input: name == tool_input["name"])
  three_args = runtime.normalize_needs_approval(lambda name, _tool_input, qualifier: name == qualifier)

  assert runtime.normalize_needs_approval(None)("any", {}, "") is False
  assert one_arg("one", {"name": "other"}, "") is True
  assert two_args("two", {"name": "two"}, "") is True
  assert three_args("three", {}, "three") is True


def test_runtime_callable_accepts_keyword_direct_or_kwargs() -> None:
  def direct(*, abort_event=None):
    return abort_event

  def accepts_kwargs(**kwargs):
    return kwargs

  def no_match(value):
    return value

  assert runtime.callable_accepts_kw(direct, "abort_event") is True
  assert runtime.callable_accepts_kw(accepts_kwargs, "abort_event") is True
  assert runtime.callable_accepts_kw(no_match, "abort_event") is False
  assert runtime.callable_accepts_kw(None, "abort_event") is False


def test_runtime_approval_cache_respects_qualifier_and_denylist() -> None:
  calls: list[tuple[str, dict[str, Any], str]] = []

  def needs_approval(name: str, tool_input: dict[str, Any], qualifier: str) -> bool:
    calls.append((name, tool_input, qualifier))
    return True

  assert runtime.should_request_approval(
    "write_file",
    {"path": "x"},
    "tmp",
    session_cache_denied=frozenset(),
    approved_tool_types={"write_file:tmp"},
    needs_approval=needs_approval,
  ) is False
  assert calls == []

  assert runtime.should_request_approval(
    "write_file",
    {"path": "x"},
    "tmp",
    session_cache_denied=frozenset({"write_file"}),
    approved_tool_types={"write_file:tmp"},
    needs_approval=needs_approval,
  ) is True
  assert calls == [("write_file", {"path": "x"}, "tmp")]

  assert runtime.tool_was_cache_hit(
    "write_file",
    "tmp",
    session_cache_denied=frozenset(),
    approved_tool_types={"write_file:tmp"},
  ) is True
  assert runtime.tool_was_cache_hit(
    "write_file",
    "tmp",
    session_cache_denied=frozenset({"write_file"}),
    approved_tool_types={"write_file:tmp"},
  ) is False


def test_runtime_mcp_scope_error_messages() -> None:
  assert runtime.mcp_scope_error(
    "browser_snapshot",
    "browser",
    allowed_mcp_tools_by_server=None,
  ) is None

  assert runtime.mcp_scope_error(
    "browser_snapshot",
    "browser",
    allowed_mcp_tools_by_server={"browser": {"browser_snapshot"}},
  ) is None

  expected_skill_serverless = "MCP tool 'unknown' is not allowed in this scoped child run."
  missing_server = runtime.mcp_scope_error(
    "unknown",
    None,
    allowed_mcp_tools_by_server={"browser": {"browser_snapshot"}},
  )
  assert missing_server is not None
  assert missing_server["code"] == "mcp_tool_not_allowed"
  assert missing_server["sub_code"] == "skill_scope"
  assert missing_server["message"] == expected_skill_serverless

  expected_skill_scoped = (
    "MCP tool 'filesystem.filesystem_read' is not allowed in this scoped child run. "
    "Use one of the MCP tools declared by the active skill."
  )
  wrong_server = runtime.mcp_scope_error(
    "filesystem_read",
    "filesystem",
    allowed_mcp_tools_by_server={"browser": {"browser_snapshot"}},
  )
  assert wrong_server is not None
  assert wrong_server["code"] == "mcp_tool_not_allowed"
  assert wrong_server["sub_code"] == "skill_scope"
  assert wrong_server["message"] == expected_skill_scoped

  explicit_skill = runtime.mcp_scope_error(
    "filesystem_read",
    "filesystem",
    allowed_mcp_tools_by_server={"browser": {"browser_snapshot"}},
    scope_context="skill",
  )
  assert explicit_skill is not None
  assert explicit_skill["message"] == expected_skill_scoped

  profile_with_closure = runtime.mcp_scope_error(
    "filesystem_read",
    "filesystem",
    allowed_mcp_tools_by_server={"browser": {"browser_snapshot"}},
    scope_context="profile",
    describe_scope_block=lambda server_name, tool_name: f"profile closure: {server_name}.{tool_name}",
  )
  assert profile_with_closure is not None
  assert profile_with_closure["code"] == "mcp_tool_not_allowed"
  assert profile_with_closure["sub_code"] == "profile_scope"
  assert profile_with_closure["message"] == "profile closure: filesystem.filesystem_read"

  profile_fallback = runtime.mcp_scope_error(
    "filesystem_read",
    "filesystem",
    allowed_mcp_tools_by_server={"browser": {"browser_snapshot"}},
    scope_context="profile",
  )
  assert profile_fallback is not None
  assert profile_fallback["code"] == "mcp_tool_not_allowed"
  assert profile_fallback["sub_code"] == "profile_scope"
  assert profile_fallback["message"] == (
    "MCP tool 'filesystem.filesystem_read' is not in this session's active MCP tool scope. "
    "Call load_tools(servers=['filesystem']) to arm that server's available tools, then retry; "
    "if it stays blocked after loading, it is not available on this channel."
  )

  profile_serverless = runtime.mcp_scope_error(
    "unknown",
    None,
    allowed_mcp_tools_by_server={"browser": {"browser_snapshot"}},
    scope_context="profile",
  )
  assert profile_serverless is not None
  assert profile_serverless["code"] == "mcp_tool_not_allowed"
  assert profile_serverless["sub_code"] == "profile_scope"
  assert "session instructions' deferred servers and tool packs list" in profile_serverless["message"]
  assert "load_tools(servers=[" not in profile_serverless["message"]


def test_tool_dispatcher_runtime_wrappers_preserve_parent_override_seams() -> None:
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    needs_approval=lambda _name, _tool_input, _qualifier: True,
    approved_tool_types={"custom-key"},
  )
  calls: list[tuple[str, str]] = []

  def custom_qualified_key(tool_name: str, qualifier: str) -> str:
    calls.append((tool_name, qualifier))
    return "custom-key"

  dispatcher._qualified_key = custom_qualified_key  # type: ignore[method-assign]

  assert dispatcher._should_request_approval("write_file", {"path": "x"}, "tmp") is False
  assert dispatcher._tool_was_cache_hit("write_file", "tmp") is True
  assert calls == [("write_file", "tmp"), ("write_file", "tmp")]


def test_tool_dispatcher_mcp_scope_wrapper_uses_instance_allowlist() -> None:
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    allowed_mcp_tools_by_server={"browser": {"browser_snapshot"}},
  )

  assert dispatcher._mcp_scope_error("browser_snapshot", "browser") is None
  error = dispatcher._mcp_scope_error("filesystem_read", "filesystem")
  assert error is not None
  assert error["code"] == "mcp_tool_not_allowed"
  assert error["sub_code"] == "skill_scope"
  assert error["message"] == (
    "MCP tool 'filesystem.filesystem_read' is not allowed in this scoped child run. "
    "Use one of the MCP tools declared by the active skill."
  )


def test_tool_dispatcher_mcp_scope_wrapper_passes_profile_context_and_describer() -> None:
  calls: list[tuple[str | None, str]] = []

  def describe(server_name: str | None, tool_name: str) -> str:
    calls.append((server_name, tool_name))
    return f"profile block for {server_name}.{tool_name}"

  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    allowed_mcp_tools_by_server={"browser": {"browser_snapshot"}},
    mcp_scope_context="profile",
    describe_mcp_scope_block=describe,
  )

  error = dispatcher._mcp_scope_error("filesystem_read", "filesystem")

  assert error is not None
  assert error["code"] == "mcp_tool_not_allowed"
  assert error["sub_code"] == "profile_scope"
  assert error["message"] == "profile block for filesystem.filesystem_read"
  assert calls == [("filesystem", "filesystem_read")]


def test_tool_dispatcher_observes_post_construction_mutable_allowlist_updates() -> None:
  mcp = _ScopedCorpusMcpClient()
  allowlist = {"research-corpus-mcp": {"corpus_search"}}
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    get_tool_definitions=lambda: [
      {"name": "corpus_search"},
      {"name": "corpus_write"},
    ],
    allowed_mcp_tools_by_server=allowlist,
  )

  blocked_result, blocked_error = asyncio.run(
    dispatcher.dispatch(
      "call-1",
      "corpus_write",
      {"ticker": "MSFT"},
      advertised_tool_names=frozenset({"corpus_search", "corpus_write"}),
    )
  )
  allowlist["research-corpus-mcp"].add("corpus_write")
  allowed_result, allowed_error = asyncio.run(
    dispatcher.dispatch(
      "call-2",
      "corpus_write",
      {"ticker": "MSFT"},
      advertised_tool_names=frozenset({"corpus_search", "corpus_write"}),
    )
  )

  assert dispatcher._allowed_mcp_tools_by_server is allowlist
  assert blocked_result is None
  assert blocked_error is not None
  assert blocked_error["code"] == "mcp_tool_not_allowed"
  assert allowed_error is None
  assert allowed_result == {"ok": "corpus_write"}
  assert mcp.calls == [("corpus_write", {"ticker": "MSFT"})]


def test_tool_dispatcher_keeps_one_wire_snapshot_for_all_calls_in_provider_response() -> None:
  mcp = _ScopedCorpusMcpClient()
  advertised = {"corpus_search"}
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    get_tool_definitions=lambda: [
      {"name": tool_name, "input_schema": {"type": "object"}}
      for tool_name in sorted(advertised)
    ],
    allowed_mcp_tools_by_server={
      "research-corpus-mcp": {"corpus_search", "corpus_write"},
    },
  )

  first_result, first_error = asyncio.run(
    dispatcher.dispatch(
      "call-1",
      "corpus_search",
      {},
      call_index=0,
      advertised_tool_names=frozenset({"corpus_search"}),
    )
  )
  advertised.add("corpus_write")
  same_response_result, same_response_error = asyncio.run(
    dispatcher.dispatch(
      "call-2",
      "corpus_write",
      {"ticker": "MSFT"},
      call_index=1,
      advertised_tool_names=frozenset({"corpus_search"}),
    )
  )
  next_response_result, next_response_error = asyncio.run(
    dispatcher.dispatch(
      "call-3",
      "corpus_write",
      {"ticker": "MSFT"},
      call_index=0,
      advertised_tool_names=frozenset({"corpus_search", "corpus_write"}),
    )
  )

  assert first_error is None
  assert first_result == {"ok": "corpus_search"}
  assert same_response_result is None
  assert same_response_error is not None
  assert same_response_error["code"] == "mcp_tool_not_allowed"
  assert next_response_error is None
  assert next_response_result == {"ok": "corpus_write"}
  assert mcp.calls == [
    ("corpus_search", {}),
    ("corpus_write", {"ticker": "MSFT"}),
  ]


def test_tool_dispatcher_snapshots_before_local_load_mutates_catalog() -> None:
  mcp = _ScopedCorpusMcpClient()
  advertised = {"load_tools", "corpus_search"}

  async def load_tools(_tool_input, **_kwargs):
    advertised.add("corpus_write")
    return {"loaded_tools": ["corpus_write"]}, None

  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={"load_tools": load_tools},
    role="owner",
    get_tool_definitions=lambda: [
      {"name": tool_name, "input_schema": {"type": "object"}}
      for tool_name in sorted(advertised)
    ],
    allowed_mcp_tools_by_server={
      "research-corpus-mcp": {"corpus_search", "corpus_write"},
    },
  )

  load_result, load_error = asyncio.run(
    dispatcher.dispatch(
      "call-load",
      "load_tools",
      {"servers": ["research-corpus-mcp"]},
      call_index=0,
      advertised_tool_names=frozenset({"load_tools", "corpus_search"}),
    )
  )
  same_response_result, same_response_error = asyncio.run(
    dispatcher.dispatch(
      "call-write",
      "corpus_write",
      {"ticker": "MSFT"},
      call_index=1,
      advertised_tool_names=frozenset({"load_tools", "corpus_search"}),
    )
  )

  assert load_error is None
  assert load_result == {"loaded_tools": ["corpus_write"]}
  assert same_response_result is None
  assert same_response_error is not None
  assert same_response_error["code"] == "mcp_tool_not_allowed"
  assert mcp.calls == []


def test_tool_dispatcher_uses_request_snapshot_when_live_catalog_mutates_before_dispatch() -> None:
  mcp = _ScopedCorpusMcpClient()
  live_catalog = {"corpus_search"}
  request_snapshot = frozenset(live_catalog)

  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    get_tool_definitions=lambda: [
      {"name": tool_name} for tool_name in sorted(live_catalog)
    ],
    allowed_mcp_tools_by_server={
      "research-corpus-mcp": {"corpus_search", "corpus_write"},
    },
  )
  live_catalog.add("corpus_write")

  result, error = asyncio.run(
    dispatcher.dispatch(
      "call-1",
      "corpus_write",
      {},
      call_index=0,
      advertised_tool_names=request_snapshot,
    )
  )

  assert result is None
  assert error is not None
  assert error["code"] == "mcp_tool_not_allowed"
  assert mcp.calls == []


def test_tool_dispatcher_denies_mcp_when_advertisement_getter_is_absent() -> None:
  mcp = _ScopedCorpusMcpClient()
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    allowed_mcp_tools_by_server={
      "research-corpus-mcp": {"corpus_search"},
    },
  )

  result, error = asyncio.run(
    dispatcher.dispatch("call-1", "corpus_search", {}, call_index=0)
  )

  assert result is None
  assert error == {
    "code": "mcp_tool_not_allowed",
    "sub_code": "advertisement_unavailable",
    "message": "The advertised MCP tool snapshot is unavailable; dispatch was denied.",
  }
  assert mcp.calls == []


def test_tool_dispatcher_still_copies_non_dict_allowlist_mappings() -> None:
  mcp = _ScopedCorpusMcpClient()
  source = {"research-corpus-mcp": {"corpus_search"}}
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    allowed_mcp_tools_by_server=MappingProxyType(source),
  )

  source["research-corpus-mcp"].add("corpus_write")
  result, error = asyncio.run(
    dispatcher.dispatch(
      "call-1",
      "corpus_write",
      {"ticker": "MSFT"},
      advertised_tool_names=frozenset({"corpus_write"}),
    )
  )

  assert dispatcher._allowed_mcp_tools_by_server is not source
  assert result is None
  assert error is not None
  assert error["code"] == "mcp_tool_not_allowed"
  assert mcp.calls == []


def test_tool_dispatcher_enforces_mcp_scope_with_empty_identity_overrides() -> None:
  mcp = _ScopedCorpusMcpClient()
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    local_tool_handlers={},
    mcp_identity_overrides={},
    allowed_mcp_tools_by_server={"research-corpus-mcp": {"corpus_search"}},
  )

  result, error = asyncio.run(
    dispatcher.dispatch(
      "call-1",
      "corpus_write",
      {"ticker": "MSFT"},
      advertised_tool_names=frozenset({"corpus_write"}),
    )
  )

  assert result is None
  assert error is not None
  assert error["code"] == "mcp_tool_not_allowed"
  assert mcp.calls == []


def test_dispatcher_adds_argument_guidance_to_direct_mcp_validation_errors() -> None:
  dispatcher = ToolDispatcher(
    mcp_client=_ValidationErrorMcpClient(),  # type: ignore[arg-type]
    local_tool_handlers={},
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
    mcp_client=_NonValidationErrorMcpClient(),  # type: ignore[arg-type]
    local_tool_handlers={},
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
    mcp_client=_ValidationErrorMcpClient(
      exposed_name="portfolio_get_price_target",
      original_name="get_price_target",
    ),  # type: ignore[arg-type]
    local_tool_handlers={},
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
