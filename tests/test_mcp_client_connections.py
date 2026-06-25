import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.mcp_client as mcp_client_module  # noqa: E402
from agent_gateway.mcp_client import McpClientManager  # noqa: E402


def test_connect_stdio_wrapper_uses_parent_module_runtime(monkeypatch) -> None:
  captured: dict[str, object] = {}

  class _FakeServerParameters:
    def __init__(self, **kwargs):
      captured["server_params"] = kwargs

  class _FakeStdioContext:
    async def __aenter__(self):
      captured["stdio_entered"] = True
      return object(), object()

    async def __aexit__(self, exc_type, exc, tb):
      captured["stdio_exited"] = True
      return None

  class _FakeClientSession:
    def __init__(self, read_stream, write_stream):
      captured["session_streams"] = (read_stream, write_stream)

    async def __aenter__(self):
      captured["session_entered"] = True
      return self

    async def __aexit__(self, exc_type, exc, tb):
      captured["session_exited"] = True
      return None

    async def initialize(self):
      captured["initialized"] = True
      return None

    async def list_tools(self, cursor=None):
      captured.setdefault("list_tool_cursors", []).append(cursor)
      return SimpleNamespace(
        tools=[
          SimpleNamespace(
            name="patched_tool",
            description="Patched tool",
            inputSchema={"type": "object", "properties": {"x": {"type": "string"}}},
          )
        ],
        nextCursor=None,
      )

  def _fake_stdio_client(server_params, errlog=None):
    captured["stdio_client_params"] = server_params
    captured["errlog"] = errlog
    return _FakeStdioContext()

  monkeypatch.setattr(mcp_client_module, "StdioServerParameters", _FakeServerParameters)
  monkeypatch.setattr(mcp_client_module, "stdio_client", _fake_stdio_client)
  monkeypatch.setattr(mcp_client_module, "ClientSession", _FakeClientSession)
  monkeypatch.setattr(mcp_client_module, "_build_mcp_env", lambda env: {"PATCHED": str(env["raw"])})
  monkeypatch.setattr(mcp_client_module, "_stdio_connect_stabilize_delay", lambda: 0)

  manager = McpClientManager(config_path=None)
  state = asyncio.run(
    manager._connect_stdio(
      "demo",
      {"command": "fake-server", "args": ["--serve"], "env": {"raw": "env"}},
    )
  )

  assert captured["server_params"] == {
    "command": "fake-server",
    "args": ["--serve"],
    "env": {"PATCHED": "env"},
    "cwd": None,
  }
  assert captured["stdio_entered"] is True
  assert captured["session_entered"] is True
  assert captured["initialized"] is True
  assert captured["list_tool_cursors"] == [None, None]
  assert captured["errlog"] is not None
  assert captured["errlog"].closed is False
  assert state.name == "demo"
  assert state.config == {"command": "fake-server", "args": ["--serve"], "env": {"raw": "env"}}
  assert state.tool_names == {"patched_tool"}

  asyncio.run(manager._close_contexts(state.exit_contexts))
  assert captured["session_exited"] is True
  assert captured["stdio_exited"] is True
  assert captured["errlog"].closed is True


def test_build_http_auth_wrapper_uses_parent_module_path_factory(monkeypatch) -> None:
  captured: dict[str, object] = {}

  class _FakePath:
    def __init__(self, value):
      self.value = str(value)

    @classmethod
    def home(cls):
      return cls("/patched-home")

    def __truediv__(self, child):
      return _FakePath(f"{self.value}/{child}")

    def expanduser(self):
      captured["expanded_path"] = self.value
      return self

    def __str__(self):
      return self.value

  class _FakeStorage:
    def __init__(self, path):
      captured["storage_path"] = str(path)

  class _FakeOAuth:
    def __init__(self, **kwargs):
      captured["oauth_kwargs"] = kwargs

  monkeypatch.delenv("AGENT_GATEWAY_MCP_OAUTH_CACHE_DIR", raising=False)
  monkeypatch.setattr(mcp_client_module, "Path", _FakePath)
  monkeypatch.setattr(mcp_client_module, "_JsonFileKeyValue", _FakeStorage)
  monkeypatch.setattr(mcp_client_module, "FastMCPOAuth", _FakeOAuth)
  monkeypatch.setattr(mcp_client_module, "FASTMCP_OAUTH_IMPORT_ERROR", None)

  manager = McpClientManager(config_path=None)
  auth = manager._build_http_auth(
    "finance-cli",
    "https://cashnerd.ai/mcp",
    {"oauth": True},
  )

  assert isinstance(auth, _FakeOAuth)
  assert captured["expanded_path"] == "/patched-home/.cache/agent-gateway/mcp-oauth/finance-cli.json"
  assert captured["storage_path"] == "/patched-home/.cache/agent-gateway/mcp-oauth/finance-cli.json"
  assert captured["oauth_kwargs"]["token_storage"].__class__ is _FakeStorage
