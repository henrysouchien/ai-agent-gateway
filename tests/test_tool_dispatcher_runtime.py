# ruff: noqa: E402

from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
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


_CORPUS_STATE_DEFINITION = {
  "name": "corpus_state",
  "input_schema": {
    "type": "object",
    "required": ["research_file_id"],
    "properties": {
      "research_file_id": {"type": "integer"},
      "view": {"type": "string"},
      "roles": {"type": "array", "items": {"type": "string"}},
      "selector": {
        "type": "object",
        "properties": {"accession": {"type": "string"}},
      },
    },
  },
}


class _CorpusStateMcpClient(_TestMcpClient):
  def __init__(self) -> None:
    super().__init__()
    self.calls: list[tuple[str, dict[str, Any]]] = []

  def is_mcp_tool(self, name: str) -> bool:
    return name == "corpus_state"

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
    return {"ok": name}, None


def _corpus_dispatcher(client, *, session=None) -> ToolDispatcher:
  return ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=client,
    local_tool_handlers={},
    role="owner",
    session=session,
    get_tool_definitions=lambda: [dict(_CORPUS_STATE_DEFINITION)],
  )


def _dispatch_corpus(dispatcher, tool_input):
  return asyncio.run(dispatcher.dispatch(
    "call-1",
    "corpus_state",
    tool_input,
    advertised_tool_names=frozenset({"corpus_state"}),
  ))


def test_mcp_call_without_research_file_id_defaults_to_session_active_file() -> None:
  from agent_gateway.policy_imports import configure_server_policy, load_server_policy_module

  client = _CorpusStateMcpClient()
  session = SimpleNamespace(role="owner", admitted_job_context=SimpleNamespace(
    research_binding=SimpleNamespace(research_file_id=42, ticker="NFLX"),
  ))
  dispatcher = _corpus_dispatcher(client, session=session)
  previous_policy = load_server_policy_module()
  configure_server_policy(SimpleNamespace(
    authority_denies_tool=lambda *_args, **_kwargs: False,
    get_forbidden_tools_for_session=lambda *_args, **_kwargs: frozenset(),
    get_server_for_policy_tool=lambda _name: None,
    get_tool_class=lambda *_args, **_kwargs: None,
    resolve_effective_role=lambda _role: "owner",
  ))
  try:
    assert _dispatch_corpus(dispatcher, {"view": "overview"}) == ({"ok": "corpus_state"}, None)
    assert client.calls == [("corpus_state", {"view": "overview", "research_file_id": 42})]

    # An explicit id is the caller's, never overwritten by the session default.
    assert _dispatch_corpus(dispatcher, {"research_file_id": 7}) == ({"ok": "corpus_state"}, None)
    assert client.calls[-1] == ("corpus_state", {"research_file_id": 7})
  finally:
    configure_server_policy(previous_policy)


def test_stringified_array_and_object_inputs_are_coerced_at_dispatch() -> None:
  client = _CorpusStateMcpClient()
  dispatcher = _corpus_dispatcher(client)

  result, error = _dispatch_corpus(dispatcher, {
    "roles": '["is.revenue", "is.cost"]',
    "selector": '{"accession": "0001"}',
  })

  assert error is None
  assert result == {"ok": "corpus_state"}
  assert client.calls == [("corpus_state", {
    "roles": ["is.revenue", "is.cost"],
    "selector": {"accession": "0001"},
  })]


def test_unparsable_or_wrong_typed_stringified_input_passes_through_to_its_owner() -> None:
  client = _CorpusStateMcpClient()
  dispatcher = _corpus_dispatcher(client)
  arguments = {"roles": '["is.revenue",', "selector": "[1, 2]"}

  assert _dispatch_corpus(dispatcher, arguments) == ({"ok": "corpus_state"}, None)
  assert client.calls == [("corpus_state", arguments)]


@pytest.mark.parametrize("local_name", ["model_document", "check_model"])
def test_local_required_file_and_mcp_omission_use_same_session_binding(
  local_name, monkeypatch,
) -> None:
  from agent_gateway import policy_imports
  from agent_gateway.session import session_research_binding
  from agent_workflow_contracts.models import AdmittedResearchBinding

  monkeypatch.setattr(policy_imports, "_server_policy", SimpleNamespace(
    authority_denies_tool=lambda *_args, **_kwargs: False,
    get_forbidden_tools_for_session=lambda *_args, **_kwargs: frozenset(),
    get_server_for_policy_tool=lambda _name: None,
    get_tool_class=lambda *_args, **_kwargs: None,
    resolve_effective_role=lambda _role: "owner",
  ))
  session = SimpleNamespace(
    role="owner",
    admitted_job_context=SimpleNamespace(
      research_binding=AdmittedResearchBinding(research_file_id=None, ticker=None),
    ),
    tool_research_binding=AdmittedResearchBinding(research_file_id=42, ticker="NFLX"),
  )
  client = _CorpusStateMcpClient()
  local_calls = []

  async def local_handler(tool_input, **_kwargs):
    local_calls.append(tool_input)
    expected = session_research_binding(session).research_file_id
    if tool_input["research_file_id"] != expected:
      return None, {"code": "research_file_id_mismatch", "expected": expected}
    return {"ok": local_name}, None

  local_definition = {
    "name": local_name,
    "input_schema": {
      "type": "object",
      "required": ["research_file_id", "ticker"],
      "properties": {
        "research_file_id": {"type": "integer"},
        "ticker": {"type": "string"},
      },
    },
  }
  dispatcher = ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=client,
    local_tool_handlers={local_name: local_handler},
    session=session,
    role="owner",
    get_tool_definitions=lambda: [local_definition, _CORPUS_STATE_DEFINITION],
  )

  assert asyncio.run(dispatcher.dispatch("local", local_name, {})) == (
    {"ok": local_name}, None,
  )
  assert local_calls[-1] == {"research_file_id": 42, "ticker": "NFLX"}
  assert _dispatch_corpus(dispatcher, {}) == ({"ok": "corpus_state"}, None)
  assert client.calls[-1] == ("corpus_state", {"research_file_id": 42})
  # Explicit values reach the mismatch owner unchanged, not the default.
  assert asyncio.run(dispatcher.dispatch(
    "explicit", local_name, {"research_file_id": 99, "ticker": "OTHER"},
  )) == (None, {"code": "research_file_id_mismatch", "expected": 42})
  assert local_calls[-1] == {"research_file_id": 99, "ticker": "OTHER"}
  # Admitted context supersedes ordinary-chat selection for both routes.
  session.admitted_job_context.research_binding = AdmittedResearchBinding(
    research_file_id=43, ticker="MSCI",
  )
  assert asyncio.run(dispatcher.dispatch("next", local_name, {})) == (
    {"ok": local_name}, None,
  )
  assert local_calls[-1] == {"research_file_id": 43, "ticker": "MSCI"}
  assert _dispatch_corpus(dispatcher, {}) == ({"ok": "corpus_state"}, None)
  assert client.calls[-1] == ("corpus_state", {"research_file_id": 43})


@pytest.mark.parametrize(
  ("properties", "arguments"),
  [
    ({"portfolio_id": {"type": "string"}}, {}),
    ({"ticker": {"type": "string"}}, {"ticker": "NFLX"}),
    (
      {"research_file_id": {"type": "integer"}, "ticker": {"type": "string"}},
      {"research_file_id": 42, "ticker": "NFLX"},
    ),
  ],
)
def test_prepare_resolves_research_scope_only_for_an_omitted_scope_parameter(
  monkeypatch, properties, arguments,
) -> None:
  dispatcher = ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=_TestMcpClient(),
    session=SimpleNamespace(dispatch_scope={"kind": "portfolio"}, role="owner"),
    get_tool_definitions=lambda: [{
      "name": "example", "input_schema": {"type": "object", "properties": properties},
    }],
  )

  def unexpected_context_read():
    raise AssertionError("tool preparation has no omitted research scope to resolve")

  monkeypatch.setattr(dispatcher, "_executing_context", unexpected_context_read)
  assert dispatcher.prepare_tool_call("example", arguments).materialize_input() == arguments


def _scoped_dispatcher(definitions, *, binding=None, local_names=()):
  from agent_workflow_contracts.models import AdmittedResearchBinding

  async def handler(tool_input, **_kwargs):
    return tool_input, None

  return ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=_TestMcpClient(),
    local_tool_handlers={name: handler for name in local_names},
    session=SimpleNamespace(
      admitted_job_context=None,
      tool_research_binding=binding or AdmittedResearchBinding(research_file_id=2, ticker="AAPL"),
    ),
    get_tool_definitions=lambda: definitions,
  )


def _scoped_definition(name, properties, required=()):
  return {
    "name": name,
    "input_schema": {
      "type": "object",
      "required": list(required),
      "properties": {key: {"type": kind} for key, kind in properties.items()},
    },
  }


@pytest.mark.parametrize("name", [
  "idea_capture_manual", "valuation_ready_batch_dispatch", "filings_search", "get_quote",
])
def test_tool_without_research_file_id_is_never_scoped(name) -> None:
  # A ticker-only tool names a company, not a file: no ticker, no file id.
  definition = _scoped_definition(name, {"ticker": "string"}, required=["ticker"])
  dispatcher = _scoped_dispatcher([definition], local_names=[name])

  for arguments in ({}, {"ticker": ""}, {"ticker": "MSFT"}):
    assert dispatcher.prepare_tool_call(name, arguments).materialize_input() == arguments


@pytest.mark.parametrize(
  ("tool_name", "arguments"),
  [
    ("workflow_run", {"action": "status", "workflow_run_id": "wf-1"}),
    ("workflow_run", {"action": "start", "intent": "Generic research"}),
    ("run_agent", {"objective": "Research MSFT margins", "ticker": "MSFT"}),
    ("run_agent", {"objective": "Research the sector"}),
    ("run_agent", {"interrupt": "task-1"}),
    ("run_agent", {"resume": "task-1"}),
    ("run_agent", {"resume": "task-1", "additional_context": "carry on"}),
  ],
)
def test_workflow_run_and_other_company_run_agent_keep_their_arguments(tool_name, arguments) -> None:
  from agent.shared.tool_defs.workflow import WORKFLOW_RUN_TOOL_DEF
  from agent_gateway.sub_agent_helpers import make_run_agent_tool_def

  dispatcher = _scoped_dispatcher(
    [WORKFLOW_RUN_TOOL_DEF, make_run_agent_tool_def()], local_names=[tool_name],
  )

  assert dispatcher.prepare_tool_call(tool_name, arguments).materialize_input() == arguments


def test_explicit_file_id_never_gets_another_files_ticker() -> None:
  definition = _scoped_definition(
    "model_document", {"research_file_id": "integer", "ticker": "string"},
    required=["research_file_id", "ticker"],
  )
  dispatcher = _scoped_dispatcher([definition], local_names=["model_document"])

  assert dispatcher.prepare_tool_call(
    "model_document", {"research_file_id": 7},
  ).materialize_input() == {"research_file_id": 7}
  assert dispatcher.prepare_tool_call(
    "model_document", {"research_file_id": 2},
  ).materialize_input() == {"research_file_id": 2, "ticker": "AAPL"}
  assert dispatcher.prepare_tool_call(
    "model_document", {},
  ).materialize_input() == {"research_file_id": 2, "ticker": "AAPL"}


@pytest.mark.parametrize(("name", "arguments", "expected"), [
  ("model_semantics", {}, {"research_file_id": 2}),
  ("fms_persist_correspondence", {"changes": []}, {"changes": [], "research_file_id": 2}),
  ("model_semantics", {"research_file_id": 7}, {"research_file_id": 7}),
  ("model_semantics", {"model_path": "m.xlsx"}, {"model_path": "m.xlsx"}),
])
def test_optional_research_file_id_defaults_unless_model_path_selects(name, arguments, expected) -> None:
  definition = _scoped_definition(
    name,
    {"research_file_id": "integer", "model_path": "string", "changes": "array"},
  )
  dispatcher = _scoped_dispatcher([definition], local_names=[name])

  assert dispatcher.prepare_tool_call(name, arguments).materialize_input() == expected
