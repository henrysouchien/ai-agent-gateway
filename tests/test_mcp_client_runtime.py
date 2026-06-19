# ruff: noqa: E402

import asyncio
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.mcp_client as mcp_client_module
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.mcp_client_runtime import extract_text, read_claude_config


class _CaptureLogger:
  def __init__(self) -> None:
    self.debugs: list[tuple[object, ...]] = []
    self.warnings: list[tuple[object, ...]] = []

  def debug(self, message, *args) -> None:
    self.debugs.append((message, *args))

  def warning(self, message, *args) -> None:
    self.warnings.append((message, *args))


def test_extract_text_matches_parent_private_wrapper() -> None:
  content = [
    {"text": " alpha "},
    {"text": "   "},
    SimpleNamespace(text="beta"),
    SimpleNamespace(text=None),
  ]

  assert extract_text(content) == "alpha \nbeta"
  assert McpClientManager._extract_text(content) == "alpha \nbeta"


def test_parent_close_contexts_routes_suppression_and_logger(monkeypatch) -> None:
  logger = _CaptureLogger()
  events: list[str] = []

  @contextmanager
  def fake_suppress():
    events.append("enter")
    try:
      yield
    finally:
      events.append("exit")

  class _FailingContext:
    async def __aexit__(self, exc_type, exc, tb):
      events.append("close")
      raise RuntimeError("close failed")

  monkeypatch.setattr(mcp_client_module, "log", logger)
  monkeypatch.setattr(
    mcp_client_module,
    "_suppress_mcp_stdio_termination_fallback_warnings",
    fake_suppress,
  )

  asyncio.run(McpClientManager._close_contexts([_FailingContext()]))

  assert events == ["enter", "close", "exit"]
  assert logger.debugs[0][0] == "MCP context close failed: %s"
  assert str(logger.debugs[0][1]) == "close failed"


def test_parent_suppression_wrapper_uses_parent_filter_alias_and_logger(monkeypatch) -> None:
  events: list[object] = []

  class _FakeLogger:
    def addFilter(self, log_filter):
      events.append(("add", log_filter))

    def removeFilter(self, log_filter):
      events.append(("remove", log_filter))

  class _FakeFilter:
    def __init__(self):
      events.append("filter_init")

  def fake_get_logger(name):
    events.append(("logger", name))
    return _FakeLogger()

  monkeypatch.setattr(mcp_client_module, "logging", SimpleNamespace(getLogger=fake_get_logger))
  monkeypatch.setattr(mcp_client_module, "_McpStdioTerminationFallbackFilter", _FakeFilter)

  with mcp_client_module._suppress_mcp_stdio_termination_fallback_warnings():
    events.append("inside")

  assert events[0] == ("logger", "mcp.os.posix.utilities")
  assert events[1] == "filter_init"
  assert events[2][0] == "add"
  assert events[3] == "inside"
  assert events[4][0] == "remove"
  assert events[2][1] is events[4][1]


def test_parent_cancel_tool_call_uses_parent_consume_callback(monkeypatch) -> None:
  consumed: list[asyncio.Task] = []

  async def stubborn_task() -> str:
    try:
      await asyncio.sleep(60)
    except asyncio.CancelledError:
      await asyncio.sleep(0.02)
      return "late"

  async def run_cancel() -> None:
    task = asyncio.create_task(stubborn_task())
    await asyncio.sleep(0)
    await McpClientManager._cancel_mcp_tool_call(task, tool_name="slow", reason="test")
    assert await task == "late"
    await asyncio.sleep(0)

  monkeypatch.setattr(mcp_client_module, "_MCP_TOOL_CANCEL_GRACE_SECONDS", 0.001)
  monkeypatch.setattr(
    mcp_client_module,
    "_consume_mcp_tool_call_result",
    lambda task: consumed.append(task),
  )

  asyncio.run(run_cancel())

  assert len(consumed) == 1
  assert consumed[0].done() is True


def test_parent_read_claude_config_routes_parent_logger(monkeypatch, tmp_path: Path) -> None:
  logger = _CaptureLogger()
  config_path = tmp_path / "mcp.json"
  config_path.write_text("{not json", encoding="utf-8")
  manager = McpClientManager(config_path=config_path)
  monkeypatch.setattr(mcp_client_module, "log", logger)

  assert manager._read_claude_config() == {}
  assert logger.warnings
  assert logger.warnings[0][0] == "Failed to read %s: %s"
  assert logger.warnings[0][1] == config_path


def test_runtime_read_claude_config_accepts_only_dict_payloads(tmp_path: Path) -> None:
  logger = _CaptureLogger()
  config_path = tmp_path / "mcp.json"

  config_path.write_text("[]", encoding="utf-8")
  assert read_claude_config(config_path, json_load=lambda f: __import__("json").load(f), logger=logger) == {}

  config_path.write_text('{"mcpServers": {}}', encoding="utf-8")
  assert read_claude_config(config_path, json_load=lambda f: __import__("json").load(f), logger=logger) == {
    "mcpServers": {}
  }
