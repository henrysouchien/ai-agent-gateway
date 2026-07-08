# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import ToolDispatcher
from agent_gateway import tool_dispatcher
from agent_gateway import tool_dispatcher_skill_tools as skill_tools


class _NullMcpClient:
  def is_mcp_tool(self, _tool_name: str) -> bool:
    return False

  def get_server_for_tool(self, _tool_name: str) -> str | None:
    return None

  async def call_tool(self, _tool_name: str, _tool_input: dict[str, Any]):
    raise AssertionError("MCP should not execute")


async def _handler(_tool_input: dict[str, Any], **_kwargs: Any):
  return {"ok": True}, None


def test_skill_tool_helper_installs_active_skill_handler() -> None:
  local_handlers: dict[str, Any] = {}
  session = SimpleNamespace(
    gateway_local_skill_tools=[
      {"skill_name": "other", "handlers": {"dynamic_tool": object()}},
      {"skill_name": "active-skill", "handlers": {"dynamic_tool": _handler}},
    ]
  )

  assert skill_tools.ensure_gateway_local_tool_handler(
    " dynamic_tool ",
    local_handlers=local_handlers,
    session=session,
    current_skill_fn=lambda: "active-skill",
  )

  assert local_handlers == {"dynamic_tool": _handler}


def test_skill_tool_helper_rejects_missing_or_non_callable_handler() -> None:
  local_handlers: dict[str, Any] = {}
  session = SimpleNamespace(
    gateway_local_skill_tools={"skill_name": "active-skill", "handlers": {"dynamic_tool": "not-callable"}}
  )

  assert not skill_tools.ensure_gateway_local_tool_handler(
    "dynamic_tool",
    local_handlers=local_handlers,
    session=session,
    current_skill_fn=lambda: "active-skill",
  )
  assert local_handlers == {}


def test_tool_dispatcher_skill_tool_wrapper_preserves_parent_current_skill_seam(monkeypatch) -> None:
  session = SimpleNamespace(
    gateway_local_skill_tools={"skill_name": "patched-skill", "handlers": {"dynamic_tool": _handler}}
  )
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    session=session,
  )
  monkeypatch.setattr(tool_dispatcher, "current_skill", lambda: "patched-skill")

  assert dispatcher.ensure_gateway_local_tool_handler("dynamic_tool")
  assert dispatcher._local["dynamic_tool"] is _handler
