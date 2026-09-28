import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest
from mcp.shared.auth import OAuthToken

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.mcp_client as mcp_client_module
from agent_gateway.mcp_client import McpClientManager


def _run(coro):
  return asyncio.run(coro)


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


@pytest.mark.parametrize("terminate_on_close", [False, True])
def test_connect_streamable_http_lists_catalog_pages_and_closes_session(
  monkeypatch, terminate_on_close,
) -> None:
  requests: list[httpx2.Request] = []
  cursors: list[str | None] = []
  http_clients: list[httpx2.AsyncClient] = []

  async def respond(request: httpx2.Request) -> httpx2.Response:
    requests.append(request)
    assert request.headers["Authorization"] == "Bearer secret-token"
    if request.method == "GET":
      return httpx2.Response(405)
    if request.method == "DELETE":
      return httpx2.Response(200)
    payload = json.loads(request.content)
    if payload["method"] == "notifications/initialized":
      return httpx2.Response(202)
    if payload["method"] == "initialize":
      result = {
        "protocolVersion": "2025-11-25",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "paged-finance", "version": "1"},
      }
    else:
      assert payload["method"] == "tools/list"
      cursor = payload.get("params", {}).get("cursor")
      cursors.append(cursor)
      if cursor is None:
        result = {
          "tools": [{
            "name": "remote_tool",
            "description": "Remote tool",
            "inputSchema": {"type": "object", "properties": {"ticker": {"type": "string"}}},
            "_meta": {"audience": ["assistant"]},
          }],
          "nextCursor": "page-2",
        }
      else:
        assert cursor == "page-2"
        result = {
          "tools": [{
            "name": "second_tool",
            "inputSchema": {"type": "object", "properties": {}},
          }],
        }
    return httpx2.Response(
      200,
      headers={"Mcp-Session-Id": "mcp-session-1"},
      json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
    )

  def http_client(**kwargs):
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond), **kwargs)
    http_clients.append(client)
    return client

  monkeypatch.setattr(
    mcp_client_module, "httpx2",
    SimpleNamespace(AsyncClient=http_client, Timeout=httpx2.Timeout),
  )
  monkeypatch.setenv("CASHNERD_MCP_TOKEN", "secret-token")

  async def scenario():
    manager = McpClientManager(config_path=None, startup_timeout=1)
    state = await manager._connect(
      "finance-cli",
      {
        "type": "streamable-http",
        "url": "https://cashnerd.ai/mcp",
        "headers": {"Authorization": "Bearer ${CASHNERD_MCP_TOKEN}"},
        "timeout": 7,
        "sse_read_timeout": 45,
        "terminate_on_close": terminate_on_close,
      },
    )
    try:
      assert state.tool_names == {"remote_tool", "second_tool"}
      assert state.tool_definitions[0]["input_schema"]["properties"]["ticker"]["type"] == "string"
      assert state.tool_metadata["remote_tool"] == {"audience": ["assistant"]}
      assert all("_meta" not in tool and "meta" not in tool for tool in state.tool_definitions)
      assert cursors == [None, "page-2"]
    finally:
      await manager._close_contexts(state.exit_contexts)
    assert http_clients[0].is_closed
    assert sum(request.method == "DELETE" for request in requests) == int(terminate_on_close)

  _run(scenario())


def test_oauth_auth_reuses_persistent_tokens_with_httpx2(tmp_path) -> None:
  cache_path = tmp_path / "oauth.json"
  config = {
    "oauth": {
      "cache_path": str(cache_path),
      "scopes": ["openid", "email"],
      "callback_port": 8765,
      "client_name": "advisor",
    }
  }
  manager = McpClientManager(config_path=None)

  async def scenario():
    original = manager._build_http_auth("finance-cli", "https://cashnerd.ai/mcp", config)
    assert original is not None
    await original.context.storage.set_tokens(
      OAuthToken(access_token="persisted-token", token_type="Bearer"),
    )
    restored = manager._build_http_auth("finance-cli", "https://cashnerd.ai/mcp", config)
    different_endpoint = manager._build_http_auth(
      "finance-cli", "https://cashnerd.ai/other-mcp", config,
    )
    assert different_endpoint is not None
    assert await different_endpoint.context.storage.get_tokens() is None

    async def respond(request: httpx2.Request) -> httpx2.Response:
      assert request.headers["Authorization"] == "Bearer persisted-token"
      return httpx2.Response(200, json={"authenticated": True})

    async with httpx2.AsyncClient(
      auth=restored, transport=httpx2.MockTransport(respond),
    ) as client:
      response = await client.get("https://cashnerd.ai/mcp")
    assert response.json() == {"authenticated": True}

  _run(scenario())
