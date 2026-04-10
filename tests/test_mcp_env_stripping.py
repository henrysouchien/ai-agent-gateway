import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.mcp_client as mcp_client_module
from agent_gateway.mcp_client import McpClientManager, _build_mcp_env


def _run(coro):
  return asyncio.run(coro)


def test_build_mcp_env_keeps_allowlist_only_for_empty_server_env(monkeypatch) -> None:
  monkeypatch.setattr(
    mcp_client_module.os,
    "environ",
    {
      "PATH": "/usr/bin:/bin",
      "HOME": "/tmp/home",
      "LANG": "en_US.UTF-8",
      "PYTHONPATH": "/workspace/lib",
      "OPENAI_API_KEY": "openai-secret",
      "ANTHROPIC_API_KEY": "anthropic-secret",
      "GOOGLE_API_KEY": "google-secret",
      "AWS_SECRET_ACCESS_KEY": "aws-secret",
      "UNRELATED_VAR": "drop-me",
    },
  )

  env = _build_mcp_env({})

  assert env == {
    "PATH": "/usr/bin:/bin",
    "HOME": "/tmp/home",
    "LANG": "en_US.UTF-8",
    "PYTHONPATH": "/workspace/lib",
  }
  assert "ANTHROPIC_API_KEY" not in env
  assert "OPENAI_API_KEY" not in env


def test_build_mcp_env_adds_explicit_server_env_and_overrides_allowlist(monkeypatch) -> None:
  monkeypatch.setattr(
    mcp_client_module.os,
    "environ",
    {
      "PATH": "/usr/bin:/bin",
      "HOME": "/tmp/home",
      "PYTHONPATH": "/workspace/lib",
      "OPENAI_API_KEY": "openai-secret",
    },
  )

  env = _build_mcp_env(
    {
      "ANTHROPIC_API_KEY": "explicit-anthropic",
      "PATH": "/custom/bin",
      "EXTRA_FLAG": "enabled",
      "IGNORED_NONE": None,
    }
  )

  assert env["ANTHROPIC_API_KEY"] == "explicit-anthropic"
  assert env["PATH"] == "/custom/bin"
  assert env["HOME"] == "/tmp/home"
  assert env["PYTHONPATH"] == "/workspace/lib"
  assert env["EXTRA_FLAG"] == "enabled"
  assert "OPENAI_API_KEY" not in env
  assert "IGNORED_NONE" not in env


def test_connect_passes_filtered_env_to_stdio_server_parameters(monkeypatch) -> None:
  monkeypatch.setattr(mcp_client_module, "MCP_IMPORT_ERROR", None)
  monkeypatch.setattr(
    mcp_client_module.os,
    "environ",
    {
      "PATH": "/usr/bin:/bin",
      "HOME": "/tmp/home",
      "USER": "tester",
      "OPENAI_API_KEY": "openai-secret",
      "AWS_SECRET_ACCESS_KEY": "aws-secret",
    },
  )

  captured: dict[str, object] = {}

  class _FakeServerParameters:
    def __init__(self, **kwargs):
      captured.update(kwargs)

  class _FakeStdioContext:
    async def __aenter__(self):
      return object(), object()

    async def __aexit__(self, exc_type, exc, tb):
      return None

  class _FakeListedTools:
    tools = []
    nextCursor = None

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
      return _FakeListedTools()

  def _fake_stdio_client(server_params, errlog=None):
    captured["server_params"] = server_params
    captured["errlog"] = errlog
    return _FakeStdioContext()

  monkeypatch.setattr(mcp_client_module, "StdioServerParameters", _FakeServerParameters)
  monkeypatch.setattr(mcp_client_module, "stdio_client", _fake_stdio_client)
  monkeypatch.setattr(mcp_client_module, "ClientSession", _FakeClientSession)

  manager = McpClientManager(config_path=None)
  state = _run(
    manager._connect(
      "browser",
      {
        "command": "fake-mcp-server",
        "args": ["--serve"],
        "env": {"ANTHROPIC_API_KEY": "explicit-anthropic", "PATH": "/custom/bin"},
      },
    )
  )

  assert captured["command"] == "fake-mcp-server"
  assert captured["args"] == ["--serve"]
  assert captured["cwd"] is None
  assert captured["env"] == {
    "PATH": "/custom/bin",
    "HOME": "/tmp/home",
    "USER": "tester",
    "ANTHROPIC_API_KEY": "explicit-anthropic",
  }

  _run(manager._close_contexts(state.exit_contexts))
