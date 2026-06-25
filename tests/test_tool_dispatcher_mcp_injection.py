import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import ToolDispatcher


def _run(coro):
  return asyncio.run(coro)


class _CaptureMcpClient:
  def __init__(self) -> None:
    self.calls: list[tuple[str, dict[str, Any]]] = []
    self.servers = {
      "browser_snapshot": "browser",
      "filesystem_read": "filesystem",
    }

  def is_mcp_tool(self, name: str) -> bool:
    return name in self.servers

  def get_server_for_tool(self, name: str) -> str | None:
    return self.servers.get(name)

  async def call_tool(self, name: str, tool_input: dict[str, Any]):
    self.calls.append((name, dict(tool_input)))
    return {"ok": True}, None


def test_dispatch_injects_session_id_only_for_configured_mcp_servers() -> None:
  mcp_client = _CaptureMcpClient()
  dispatcher = ToolDispatcher(
    mcp_client=mcp_client,
    local_tool_handlers={},
    session_id="sess_browser_123",
    mcp_session_inject_servers={"browser"},
  )

  browser_result, browser_error = _run(
    dispatcher.dispatch("tool_1", "browser_snapshot", {"url": "https://example.com"})
  )
  fs_result, fs_error = _run(
    dispatcher.dispatch("tool_2", "filesystem_read", {"path": "/tmp/example.txt"})
  )

  assert browser_error is None
  assert browser_result == {"ok": True}
  assert fs_error is None
  assert fs_result == {"ok": True}
  assert mcp_client.calls == [
    ("browser_snapshot", {"url": "https://example.com", "_session_id": "sess_browser_123"}),
    ("filesystem_read", {"path": "/tmp/example.txt"}),
  ]


def test_dispatch_rejects_mcp_tool_outside_scoped_allowlist() -> None:
  mcp_client = _CaptureMcpClient()
  dispatcher = ToolDispatcher(
    mcp_client=mcp_client,
    local_tool_handlers={},
    allowed_mcp_tools_by_server={"browser": {"browser_snapshot"}},
  )

  allowed_result, allowed_error = _run(
    dispatcher.dispatch("tool_1", "browser_snapshot", {"url": "https://example.com"})
  )
  denied_result, denied_error = _run(
    dispatcher.dispatch("tool_2", "filesystem_read", {"path": "/tmp/example.txt"})
  )

  assert allowed_error is None
  assert allowed_result == {"ok": True}
  assert denied_result is None
  assert denied_error is not None
  assert denied_error["code"] == "mcp_tool_not_allowed"
  assert "filesystem.filesystem_read" in denied_error["message"]
  assert mcp_client.calls == [("browser_snapshot", {"url": "https://example.com"})]
