import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.mcp_client as mcp_client_module
from agent_gateway.mcp_client import McpClientManager, _JsonFileKeyValue


def _run(coro):
  return asyncio.run(coro)


class _FakeTimeout:
  def __init__(self, timeout, *, read):
    self.timeout = timeout
    self.read = read


class _FakeAsyncClient:
  def __init__(self, **kwargs):
    self.kwargs = kwargs
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    return self

  async def __aexit__(self, exc_type, exc, tb):
    self.exited = True
    return None


class _FakeStreamContext:
  def __init__(self, captured: dict[str, object]):
    self.captured = captured
    self.exited = False

  async def __aenter__(self):
    self.captured["stream_entered"] = True
    return object(), object(), lambda: "mcp-session-1"

  async def __aexit__(self, exc_type, exc, tb):
    self.exited = True
    self.captured["stream_exited"] = True
    return None


class _FakeClientSession:
  def __init__(self, read_stream, write_stream):
    self.read_stream = read_stream
    self.write_stream = write_stream

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc, tb):
    return None

  async def initialize(self):
    return None

  async def list_tools(self, cursor=None):
    assert cursor is None
    return SimpleNamespace(
      tools=[
        SimpleNamespace(
          name="remote_tool",
          description="Remote tool",
          inputSchema={"type": "object", "properties": {"ticker": {"type": "string"}}},
        )
      ],
      nextCursor=None,
    )


def test_startup_allows_streamable_http_server_type(tmp_path) -> None:
  config_path = tmp_path / "claude.json"
  config_path.write_text(
    '{"mcpServers": {"finance-cli": {"type": "streamable-http", "url": "https://cashnerd.ai/mcp"}}}',
    encoding="utf-8",
  )
  manager = McpClientManager(config_path=config_path, allowed_servers={"finance-cli"})
  calls: list[str] = []

  async def _fake_connect_or_warn(name, config):
    calls.append(name)
    return None

  async def _main():
    original = manager._connect_or_warn
    manager._connect_or_warn = _fake_connect_or_warn
    try:
      await manager.startup()
    finally:
      manager._connect_or_warn = original

  _run(_main())

  assert calls == ["finance-cli"]


def test_connect_streamable_http_uses_url_headers_and_lists_tools(monkeypatch) -> None:
  captured: dict[str, object] = {}
  http_clients: list[_FakeAsyncClient] = []

  class _FakeHttpx:
    Timeout = _FakeTimeout

    @staticmethod
    def AsyncClient(**kwargs):
      client = _FakeAsyncClient(**kwargs)
      http_clients.append(client)
      return client

  def _fake_streamable_http_client(url, *, http_client, terminate_on_close=True):
    captured["url"] = url
    captured["http_client"] = http_client
    captured["terminate_on_close"] = terminate_on_close
    return _FakeStreamContext(captured)

  monkeypatch.setattr(mcp_client_module, "HTTPX_IMPORT_ERROR", None)
  monkeypatch.setattr(mcp_client_module, "httpx", _FakeHttpx)
  monkeypatch.setattr(mcp_client_module, "streamable_http_client", _fake_streamable_http_client)
  monkeypatch.setattr(mcp_client_module, "ClientSession", _FakeClientSession)
  monkeypatch.setenv("CASHNERD_MCP_TOKEN", "secret-token")

  manager = McpClientManager(config_path=None, startup_timeout=1)
  state = _run(
    manager._connect(
      "finance-cli",
      {
        "type": "streamable-http",
        "url": "https://cashnerd.ai/mcp",
        "headers": {"Authorization": "Bearer ${CASHNERD_MCP_TOKEN}"},
        "timeout": 7,
        "sse_read_timeout": 45,
        "terminate_on_close": False,
      },
    )
  )

  assert captured["url"] == "https://cashnerd.ai/mcp"
  assert captured["terminate_on_close"] is False
  assert http_clients[0].entered is True
  assert http_clients[0].kwargs["headers"] == {"Authorization": "Bearer secret-token"}
  assert http_clients[0].kwargs["timeout"].timeout == 7
  assert http_clients[0].kwargs["timeout"].read == 45
  assert state.tool_names == {"remote_tool"}
  assert state.tool_definitions[0]["input_schema"]["properties"]["ticker"]["type"] == "string"

  _run(manager._close_contexts(state.exit_contexts))
  assert http_clients[0].exited is True
  assert captured["stream_exited"] is True


def test_oauth_auth_uses_persistent_json_storage(monkeypatch, tmp_path) -> None:
  captured: dict[str, object] = {}

  class _FakeOAuth:
    def __init__(self, **kwargs):
      captured.update(kwargs)

  monkeypatch.setattr(mcp_client_module, "FASTMCP_OAUTH_IMPORT_ERROR", None)
  monkeypatch.setattr(mcp_client_module, "FastMCPOAuth", _FakeOAuth)

  cache_path = tmp_path / "oauth.json"
  manager = McpClientManager(config_path=None)
  auth = manager._build_http_auth(
    "finance-cli",
    "https://cashnerd.ai/mcp",
    {
      "oauth": {
        "cache_path": str(cache_path),
        "scopes": ["openid", "email"],
        "callback_port": 8765,
        "client_name": "advisor",
      }
    },
  )

  assert isinstance(auth, _FakeOAuth)
  assert captured["mcp_url"] == "https://cashnerd.ai/mcp"
  assert captured["scopes"] == ["openid", "email"]
  assert captured["callback_port"] == 8765
  assert captured["client_name"] == "advisor"
  assert isinstance(captured["token_storage"], _JsonFileKeyValue)
