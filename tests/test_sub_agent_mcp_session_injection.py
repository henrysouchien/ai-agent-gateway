import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.sub_agent import make_run_agent_handler


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


def test_make_run_agent_handler_forwards_mcp_session_inject_servers_to_dispatcher() -> None:
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=None,
    mcp_client=_StubMcpClient(),
    mcp_session_inject_servers={"browser"},
    local_tool_handlers={"keep_tool": _dummy_tool},
  )

  result, error = _run(handler({"task": "Collect page state"}))

  assert error is None
  assert result == {"response": "ok"}
  assert runner.calls[0]["dispatcher"]._mcp_session_inject_servers == {"browser"}


def test_make_run_agent_handler_forwards_meta_user_context_to_dispatcher() -> None:
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=None,
    mcp_client=_StubMcpClient(),
    mcp_meta_inject_servers=frozenset({"portfolio-mcp"}),
    user_id="42",
    credentials_resolver_active=True,
    local_tool_handlers={"keep_tool": _dummy_tool},
  )

  result, error = _run(handler({"task": "Collect portfolio state"}))

  assert error is None
  assert result == {"response": "ok"}
  dispatcher = runner.calls[0]["dispatcher"]
  assert dispatcher._mcp_meta_inject_servers == frozenset({"portfolio-mcp"})
  assert dispatcher._user_id == "42"
  assert dispatcher._credentials_resolver_active is True
