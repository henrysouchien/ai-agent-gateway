import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.mcp_client as mcp_client_module  # noqa: E402
from agent_gateway.mcp_client import McpClientManager, _build_mcp_env  # noqa: E402


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
      "PYTHONNOUSERSITE": "1",
      "OPENAI_API_KEY": "openai-secret",
      "ANTHROPIC_API_KEY": "anthropic-secret",
      "XAI_API_KEY": "xai-secret",
      "XAI_AUTH_TOKEN": "xai-oauth-secret",
      "XAI_REFRESH_TOKEN": "xai-refresh-secret",
      "XAI_TOKEN_EXPIRES_AT": "1234567890",
      "CLAUDE_CODE_OAUTH_TOKEN": "claude-setup-token-secret",
      "AGENT_API_USER_CLAIM_HMAC_KEY": "claim-key-secret",
      "AGENT_GATEWAY_CLAIM_SIGNING_AUTHORITY": "fd-v1",
      "INVESTMENT_CAPABILITY_CLAIM_ED25519_PRIVATE_KEY": "private-signing-key",
      "INVESTMENT_CAPABILITY_CLAIM_ED25519_PUBLIC_KEY": "public-verification-key",
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
    "PYTHONNOUSERSITE": "1",
  }
  assert "ANTHROPIC_API_KEY" not in env
  assert "OPENAI_API_KEY" not in env
  assert "XAI_API_KEY" not in env
  assert "XAI_AUTH_TOKEN" not in env
  assert "XAI_REFRESH_TOKEN" not in env
  assert "XAI_TOKEN_EXPIRES_AT" not in env
  assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
  assert "AGENT_API_USER_CLAIM_HMAC_KEY" not in env
  assert "AGENT_GATEWAY_CLAIM_SIGNING_AUTHORITY" not in env
  assert "INVESTMENT_CAPABILITY_CLAIM_ED25519_PRIVATE_KEY" not in env
  assert "INVESTMENT_CAPABILITY_CLAIM_ED25519_PUBLIC_KEY" not in env
  assert (
    "INVESTMENT_CAPABILITY_CLAIM_ED25519_PRIVATE_KEY"
    not in mcp_client_module._DEFAULT_ENV_ALLOWLIST
  )
  assert (
    "INVESTMENT_CAPABILITY_CLAIM_ED25519_PUBLIC_KEY"
    not in mcp_client_module._DEFAULT_ENV_ALLOWLIST
  )


def test_build_mcp_env_expands_explicit_investment_claim_public_key_only(
  monkeypatch,
) -> None:
  monkeypatch.setattr(
    mcp_client_module.os,
    "environ",
    {"INVESTMENT_CAPABILITY_CLAIM_ED25519_PUBLIC_KEY": "server-public-key"},
  )

  env = _build_mcp_env({
    "INVESTMENT_CAPABILITY_CLAIM_ED25519_PUBLIC_KEY": (
      "${INVESTMENT_CAPABILITY_CLAIM_ED25519_PUBLIC_KEY}"
    ),
  })

  assert env == {
    "INVESTMENT_CAPABILITY_CLAIM_ED25519_PUBLIC_KEY": "server-public-key",
  }


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
  assert "PYTHONPATH" not in env
  assert env["EXTRA_FLAG"] == "enabled"
  assert "OPENAI_API_KEY" not in env
  assert "IGNORED_NONE" not in env


def test_build_mcp_env_expands_single_env_ref(monkeypatch) -> None:
  monkeypatch.setattr(mcp_client_module.os, "environ", {"FMP_API_KEY": "secret"})

  env = _build_mcp_env({"FMP_API_KEY": "${FMP_API_KEY}"})

  assert env["FMP_API_KEY"] == "secret"


def test_build_mcp_env_expands_risk_module_dotenv_pointer(monkeypatch) -> None:
  monkeypatch.setattr(
    mcp_client_module.os,
    "environ",
    {"RISK_MODULE_DOTENV": "/tmp/synthetic-risk-module.env"},
  )

  env = _build_mcp_env({"RISK_MODULE_DOTENV": "${RISK_MODULE_DOTENV}"})

  assert env["RISK_MODULE_DOTENV"] == "/tmp/synthetic-risk-module.env"

  monkeypatch.setattr(mcp_client_module.os, "environ", {})
  env = _build_mcp_env({"RISK_MODULE_DOTENV": "${RISK_MODULE_DOTENV}"})

  assert env["RISK_MODULE_DOTENV"] == ""


@pytest.mark.parametrize(
  "env_name",
  [
    "FMP_API_KEY",
    "SEC_USER_AGENT",
    "GMAIL_TOKEN_PATH",
    "GMAIL_CREDENTIALS_PATH",
    "INVESTMENT_TOOLS_DISABLE_RESEARCH_IDEAS",
  ],
)
def test_server_env_passthrough_uses_parent_value_without_literal(
  tmp_path, monkeypatch, env_name
) -> None:
  config_path = tmp_path / "mcp.json"
  config_path.write_text(
    json.dumps(
      {
        "mcpServers": {
          "jobs-mcp": {
            "command": "jobs-server",
            "env": {env_name: "must-not-reach-child"},
          },
          "idea-workbench-mcp": {"command": "workbench-server", "env": {}},
        }
      }
    ),
    encoding="utf-8",
  )
  monkeypatch.setattr(mcp_client_module, "MCP_IMPORT_ERROR", None)
  monkeypatch.setattr(mcp_client_module.os, "environ", {env_name: "parent-secret"})

  manager = McpClientManager(
    config_path=config_path,
    server_env_passthrough={"jobs-mcp": {env_name}},
  )
  captured: dict[str, dict[str, object]] = {}

  async def _capture(name, config):
    captured[name] = config
    return None

  manager._connect_or_warn = _capture
  _run(manager.startup())

  jobs_spec = captured["jobs-mcp"]["env"]
  assert jobs_spec == {env_name: f"${{{env_name}}}"}
  assert _build_mcp_env(jobs_spec)[env_name] == "parent-secret"
  assert captured["idea-workbench-mcp"]["env"] == {}
  assert "must-not-reach-child" not in json.dumps(captured)


@pytest.mark.parametrize(
  ("configured_value", "expected_value"),
  [
    ("materialized-value", "materialized-value"),
    ("self-ref", None),
    (None, None),
  ],
)
@pytest.mark.parametrize(
  "env_name",
  [
    "FMP_API_KEY",
    "SEC_USER_AGENT",
    "GMAIL_TOKEN_PATH",
    "GMAIL_CREDENTIALS_PATH",
    "INVESTMENT_TOOLS_DISABLE_RESEARCH_IDEAS",
  ],
)
def test_server_env_passthrough_handles_config_when_parent_absent(
  tmp_path, monkeypatch, env_name, configured_value, expected_value
) -> None:
  server_env = {}
  if configured_value == "self-ref":
    server_env[env_name] = f"${{{env_name}}}"
  elif configured_value is not None:
    server_env[env_name] = configured_value
  config_path = tmp_path / "mcp.json"
  config_path.write_text(
    json.dumps(
      {
        "mcpServers": {
          "jobs-mcp": {
            "command": "jobs-server",
            "env": server_env,
          }
        }
      }
    ),
    encoding="utf-8",
  )
  monkeypatch.setattr(mcp_client_module, "MCP_IMPORT_ERROR", None)
  monkeypatch.setattr(mcp_client_module.os, "environ", {})

  manager = McpClientManager(
    config_path=config_path,
    server_env_passthrough={"jobs-mcp": {env_name}},
  )
  captured: dict[str, object] = {}

  async def _capture(_name, config):
    captured.update(config)
    return None

  manager._connect_or_warn = _capture
  _run(manager.startup())

  expected_env = {} if expected_value is None else {env_name: expected_value}
  assert captured["env"] == expected_env


def test_server_env_passthrough_preserves_literal_when_parent_value_is_empty(
  tmp_path, monkeypatch
) -> None:
  config_path = tmp_path / "mcp.json"
  config_path.write_text(
    json.dumps(
      {
        "mcpServers": {
          "jobs-mcp": {
            "command": "jobs-server",
            "env": {"FMP_API_KEY": "materialized-value"},
          }
        }
      }
    ),
    encoding="utf-8",
  )
  monkeypatch.setattr(mcp_client_module, "MCP_IMPORT_ERROR", None)
  monkeypatch.setattr(mcp_client_module.os, "environ", {"FMP_API_KEY": ""})

  manager = McpClientManager(
    config_path=config_path,
    server_env_passthrough={"jobs-mcp": {"FMP_API_KEY"}},
  )
  captured: dict[str, object] = {}

  async def _capture(_name, config):
    captured.update(config)
    return None

  manager._connect_or_warn = _capture
  _run(manager.startup())

  assert captured["env"] == {"FMP_API_KEY": "materialized-value"}


def test_build_mcp_env_expands_multiple_env_refs(monkeypatch) -> None:
  monkeypatch.setattr(mcp_client_module.os, "environ", {"FOO": "a", "BAR": "b"})

  env = _build_mcp_env({"COMBO": "${FOO}/${BAR}"})

  assert env["COMBO"] == "a/b"


def test_build_mcp_env_preserves_literal_without_refs(monkeypatch) -> None:
  monkeypatch.setattr(mcp_client_module.os, "environ", {})

  env = _build_mcp_env({"PLAIN": "no-refs-here"})

  assert env["PLAIN"] == "no-refs-here"


def test_build_mcp_env_preserves_literal_dollars_outside_refs(monkeypatch) -> None:
  monkeypatch.setattr(mcp_client_module.os, "environ", {"FOO": "x"})

  env = _build_mcp_env({"PWD": "pa$$word-$5-${FOO}"})

  assert env["PWD"] == "pa$$word-$5-x"


def test_build_mcp_env_expands_undefined_ref_to_empty_string(monkeypatch) -> None:
  monkeypatch.setattr(mcp_client_module.os, "environ", {})

  env = _build_mcp_env({"MISSING": "${UNDEFINED_XYZ}"})

  assert env["MISSING"] == ""


def test_build_mcp_env_expands_refs_once_without_recursion(monkeypatch) -> None:
  monkeypatch.setattr(
    mcp_client_module.os,
    "environ",
    {"FOO": "${BAR}", "BAR": "should-not-appear"},
  )

  env = _build_mcp_env({"NEST": "${FOO}"})

  assert env["NEST"] == "${BAR}"


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
  monkeypatch.setattr(
    mcp_client_module, "_preflight_stdio_executable", lambda _command, _args, _env: None
  )

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
