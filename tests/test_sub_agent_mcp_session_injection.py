import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.sub_agent import (  # noqa: E402
  _effective_mcp_session_inject_servers,
  make_run_agent_handler,
)
from tests.capability_execution_test_support import (  # noqa: E402
  stub_capability_execution_resolver,
)


def _run(coro):
  return asyncio.run(coro)


async def _dummy_tool(_tool_input, **_kwargs):
  return {"ok": True}, None


class _StubMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  def get_server_for_tool(self, _name: str) -> str | None:
    return None

  async def call_tool(self, _name: str, _tool_input: dict[str, Any]):
    raise AssertionError("unexpected MCP tool dispatch")


class _StubRunner:
  def __init__(self) -> None:
    self._full_session_id = "session-sub-agent"
    self.calls: list[dict[str, Any]] = []

  async def spawn_sub_agent(self, task: str, **kwargs: Any):
    self.calls.append({"task": task, **kwargs})
    return {"response": "ok"}, None

  def _get_tool_definitions(self) -> list[dict[str, str]]:
    return [{"name": "file_read"}]


def test_make_run_agent_handler_bounds_session_injection_to_admitted_scope() -> None:
  runner = _StubRunner()
  capability_execution_resolver = stub_capability_execution_resolver()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=None,
    mcp_client=_StubMcpClient(),
    mcp_session_inject_servers={"browser"},
    local_tool_handlers={"file_read": _dummy_tool},
    capability_execution_resolver=capability_execution_resolver,
  )

  result, error = _run(handler({
    "background": False,
    "objective": "Collect page state",
  }))

  assert error is None
  assert result == {"response": "ok"}
  dispatcher = runner.calls[0]["dispatcher"]
  assert dispatcher._mcp_session_inject_servers == set()
  assert dispatcher._interceptors == []


def test_make_run_agent_handler_forwards_exact_interceptor_sequence() -> None:
  runner = _StubRunner()

  async def first_interceptor(_context):
    raise AssertionError("interceptor should not run during handler assembly")

  async def second_interceptor(_context):
    raise AssertionError("interceptor should not run during handler assembly")

  interceptors = (first_interceptor, second_interceptor)
  handler = make_run_agent_handler(
    [runner],
    skill_loader=None,
    mcp_client=_StubMcpClient(),
    interceptors=interceptors,
    local_tool_handlers={"file_read": _dummy_tool},
    capability_execution_resolver=stub_capability_execution_resolver(),
  )

  result, error = _run(handler({
    "background": False,
    "objective": "Collect page state",
  }))

  assert error is None
  assert result == {"response": "ok"}
  forwarded = runner.calls[0]["dispatcher"]._interceptors
  assert len(forwarded) == 2
  assert forwarded[0] is first_interceptor
  assert forwarded[1] is second_interceptor


def test_make_run_agent_handler_explicit_none_keeps_empty_interceptors() -> None:
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=None,
    mcp_client=_StubMcpClient(),
    interceptors=None,
    local_tool_handlers={"file_read": _dummy_tool},
    capability_execution_resolver=stub_capability_execution_resolver(),
  )

  result, error = _run(handler({
    "background": False,
    "objective": "Collect page state",
  }))

  assert error is None
  assert result == {"response": "ok"}
  assert runner.calls[0]["dispatcher"]._interceptors == []


def test_named_child_does_not_inherit_loaded_servers_for_session_injection() -> None:
  effective = _effective_mcp_session_inject_servers(
    admitted_mcp_scope={},
    configured_servers={"market-data-mcp"},
  )

  assert effective == set()


def test_named_child_session_injection_requires_declaration_and_scope() -> None:
  effective = _effective_mcp_session_inject_servers(
    admitted_mcp_scope={"browser": {"browser_open"}},
    configured_servers={"browser", "outside-scope", "loaded-only"},
  )
  unrestricted = _effective_mcp_session_inject_servers(
    admitted_mcp_scope={"browser": {"browser_open"}},
    configured_servers=None,
  )

  assert effective == {"browser"}
  assert unrestricted == {"browser"}


def test_make_run_agent_handler_forwards_meta_user_context_to_dispatcher() -> None:
  runner = _StubRunner()
  capability_execution_resolver = stub_capability_execution_resolver()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=None,
    mcp_client=_StubMcpClient(),
    mcp_meta_inject_servers=frozenset({"portfolio-reads-mcp", "research-corpus-mcp"}),
    user_id="42",
    credentials_resolver_active=True,
    local_tool_handlers={"file_read": _dummy_tool},
    capability_execution_resolver=capability_execution_resolver,
  )

  result, error = _run(handler({
    "background": False,
    "objective": "Collect portfolio state",
  }))

  assert error is None
  assert result == {"response": "ok"}
  dispatcher = runner.calls[0]["dispatcher"]
  assert dispatcher._mcp_meta_inject_servers == frozenset({"portfolio-reads-mcp", "research-corpus-mcp"})
  assert dispatcher._user_id == "42"
  assert dispatcher._credentials_resolver_active is True
