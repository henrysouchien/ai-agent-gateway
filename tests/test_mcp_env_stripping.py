import asyncio
import json
import sys
from pathlib import Path

import pytest
from mcp.types import ListToolsResult, PaginatedRequestParams

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.mcp_client as mcp_client_module  # noqa: E402
from agent_gateway.mcp_client import McpClientManager, _build_mcp_env  # noqa: E402
from agent_gateway.mcp_client_config import McpEnvExpansionError  # noqa: E402


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
      "VIRTUAL_ENV": "/workspace/host-venv",
      "PYTHONNOUSERSITE": "1",
      "HANK_MODEL_AUTHORITY_DIR": "/etc/hank/model-authority",
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
      "SEC_BUDGET_SITE": "prod",
      "SEC_USER_AGENT": "host-agent contact@example.com",
      "SEC_EXPECTED_MACHINE_IDENTITY": "host.example.test",
      "AWS_SECRET_ACCESS_KEY": "aws-secret",
      "UNRELATED_VAR": "drop-me",
    },
  )

  env = _build_mcp_env({}, environ={})

  assert env == {
    "PATH": "/usr/bin:/bin",
    "HOME": "/tmp/home",
    "LANG": "en_US.UTF-8",
    "PYTHONNOUSERSITE": "1",
    "HANK_MODEL_AUTHORITY_DIR": "/etc/hank/model-authority",
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
  assert "SEC_BUDGET_SITE" not in env
  assert "SEC_USER_AGENT" not in env
  assert "SEC_EXPECTED_MACHINE_IDENTITY" not in env
  assert (
    "INVESTMENT_CAPABILITY_CLAIM_ED25519_PRIVATE_KEY"
    not in mcp_client_module._DEFAULT_ENV_ALLOWLIST
  )
  assert (
    "INVESTMENT_CAPABILITY_CLAIM_ED25519_PUBLIC_KEY"
    not in mcp_client_module._DEFAULT_ENV_ALLOWLIST
  )
  assert "SEC_BUDGET_SITE" not in mcp_client_module._DEFAULT_ENV_ALLOWLIST
  assert "SEC_USER_AGENT" not in mcp_client_module._DEFAULT_ENV_ALLOWLIST
  assert (
    "SEC_EXPECTED_MACHINE_IDENTITY" not in mcp_client_module._DEFAULT_ENV_ALLOWLIST
  )


def test_build_mcp_env_expands_explicit_investment_claim_public_key_only(
  monkeypatch,
) -> None:
  monkeypatch.setattr(
    mcp_client_module.os,
    "environ",
    {"INVESTMENT_CAPABILITY_CLAIM_ED25519_PUBLIC_KEY": "server-public-key"},
  )

  spec = {
    "INVESTMENT_CAPABILITY_CLAIM_ED25519_PUBLIC_KEY": (
      "${INVESTMENT_CAPABILITY_CLAIM_ED25519_PUBLIC_KEY}"
    ),
  }
  with pytest.raises(TypeError, match="environ"):
    _build_mcp_env(spec)  # pyright: ignore[reportCallIssue]  # negative: projection is required
  with pytest.raises(McpEnvExpansionError, match="INVESTMENT_CAPABILITY_CLAIM_ED25519_PUBLIC_KEY"):
    _build_mcp_env(spec, environ={})
  env = _build_mcp_env(
    spec,
    environ={"INVESTMENT_CAPABILITY_CLAIM_ED25519_PUBLIC_KEY": "server-public-key"},
  )

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
    },
    environ={},
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

  with pytest.raises(TypeError, match="environ"):
    _build_mcp_env({"FMP_API_KEY": "${FMP_API_KEY}"})  # pyright: ignore[reportCallIssue]  # negative: projection is required
  with pytest.raises(McpEnvExpansionError, match="FMP_API_KEY"):
    _build_mcp_env({"FMP_API_KEY": "${FMP_API_KEY}"}, environ={})
  env = _build_mcp_env(
    {"FMP_API_KEY": "${FMP_API_KEY}"}, environ={"FMP_API_KEY": "secret"},
  )

  assert env["FMP_API_KEY"] == "secret"


def test_build_mcp_env_expands_risk_module_dotenv_pointer(monkeypatch) -> None:
  monkeypatch.setattr(
    mcp_client_module.os,
    "environ",
    {"RISK_MODULE_DOTENV": "/tmp/synthetic-risk-module.env"},
  )

  with pytest.raises(TypeError, match="environ"):
    _build_mcp_env({"RISK_MODULE_DOTENV": "${RISK_MODULE_DOTENV}"})  # pyright: ignore[reportCallIssue]  # negative: projection is required
  with pytest.raises(McpEnvExpansionError, match="RISK_MODULE_DOTENV"):
    _build_mcp_env({"RISK_MODULE_DOTENV": "${RISK_MODULE_DOTENV}"}, environ={})
  env = _build_mcp_env(
    {"RISK_MODULE_DOTENV": "${RISK_MODULE_DOTENV}"},
    environ={"RISK_MODULE_DOTENV": "/tmp/synthetic-risk-module.env"},
  )

  assert env["RISK_MODULE_DOTENV"] == "/tmp/synthetic-risk-module.env"

  monkeypatch.setattr(mcp_client_module.os, "environ", {})
  env = _build_mcp_env(
    {"RISK_MODULE_DOTENV": "${RISK_MODULE_DOTENV}"},
    environ={"RISK_MODULE_DOTENV": ""},
  )

  assert env["RISK_MODULE_DOTENV"] == ""


@pytest.mark.parametrize(
  "env_name",
  [
    "FMP_API_KEY",
    "SEC_BUDGET_SITE",
    "SEC_USER_AGENT",
    "SEC_EXPECTED_MACHINE_IDENTITY",
    "GMAIL_TOKEN_PATH",
    "GMAIL_CREDENTIALS_PATH",
    "INVESTMENT_TOOLS_DISABLE_RESEARCH_IDEAS",
  ],
)
def test_declared_jobs_config_comes_from_the_run_mapping_not_a_config_literal(
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
  monkeypatch.setattr(mcp_client_module.os, "environ", {env_name: "run-scoped-value"})
  monkeypatch.setattr(mcp_client_module, "load_user_runtime_credentials", lambda _env: None)

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
  assert isinstance(jobs_spec, dict)
  assert jobs_spec == {env_name: f"${{{env_name}}}"}
  scoped, _absent = manager._scoped_server_env(
    "jobs-mcp",
    captured["jobs-mcp"],
    deployment_environ={env_name: "run-scoped-value", "ANTHROPIC_API_KEY": "ambient"},
  )
  assert scoped[env_name] == "run-scoped-value"
  assert "ANTHROPIC_API_KEY" not in scoped
  assert _build_mcp_env(
    jobs_spec, environ=scoped, base_environ={},
  )[env_name] == "run-scoped-value"
  assert "must-not-reach-child" not in json.dumps(captured)
  assert captured["idea-workbench-mcp"]["env"] == {}


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
    "SEC_BUDGET_SITE",
    "SEC_USER_AGENT",
    "SEC_EXPECTED_MACHINE_IDENTITY",
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
  monkeypatch.setattr(mcp_client_module.os, "environ", {})

  manager = McpClientManager(
    config_path=config_path,
    server_env_passthrough={"jobs-mcp": {env_name}},
  )
  captured: dict[str, object] = {}

  async def _capture(name, config):
    _ = name
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
  monkeypatch.setattr(mcp_client_module.os, "environ", {"FMP_API_KEY": ""})

  manager = McpClientManager(
    config_path=config_path,
    server_env_passthrough={"jobs-mcp": {"FMP_API_KEY"}},
  )
  captured: dict[str, object] = {}

  async def _capture(name, config):
    _ = name
    captured.update(config)
    return None

  manager._connect_or_warn = _capture
  _run(manager.startup())

  assert captured["env"] == {"FMP_API_KEY": "materialized-value"}


def test_build_mcp_env_expands_multiple_env_refs(monkeypatch) -> None:
  monkeypatch.setattr(mcp_client_module.os, "environ", {"FOO": "a", "BAR": "b"})

  env = _build_mcp_env({"COMBO": "${FOO}/${BAR}"}, environ={"FOO": "a", "BAR": "b"})

  assert env["COMBO"] == "a/b"


def test_build_mcp_env_preserves_literal_without_refs(monkeypatch) -> None:
  monkeypatch.setattr(mcp_client_module.os, "environ", {})

  env = _build_mcp_env({"PLAIN": "no-refs-here"}, environ={})

  assert env["PLAIN"] == "no-refs-here"


def test_build_mcp_env_preserves_literal_dollars_outside_refs(monkeypatch) -> None:
  monkeypatch.setattr(mcp_client_module.os, "environ", {"FOO": "x"})

  env = _build_mcp_env({"PWD": "pa$$word-$5-${FOO}"}, environ={"FOO": "x"})

  assert env["PWD"] == "pa$$word-$5-x"


def test_build_mcp_env_refuses_undefined_ref_but_accepts_declared_empty_value(monkeypatch) -> None:
  monkeypatch.setattr(mcp_client_module.os, "environ", {})

  with pytest.raises(McpEnvExpansionError, match="UNDEFINED_XYZ"):
    _build_mcp_env({"MISSING": "${UNDEFINED_XYZ}"}, environ={})
  env = _build_mcp_env(
    {"MISSING": "${UNDEFINED_XYZ}"}, environ={"UNDEFINED_XYZ": ""},
  )

  assert env["MISSING"] == ""


def test_build_mcp_env_expands_refs_once_without_recursion(monkeypatch) -> None:
  monkeypatch.setattr(
    mcp_client_module.os,
    "environ",
    {"FOO": "${BAR}", "BAR": "should-not-appear"},
  )

  env = _build_mcp_env(
    {"NEST": "${FOO}"}, environ={"FOO": "${BAR}", "BAR": "should-not-appear"},
  )

  assert env["NEST"] == "${BAR}"


def test_connect_passes_filtered_env_to_stdio_server_parameters(monkeypatch) -> None:
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

    async def list_tools(self, *, params: PaginatedRequestParams | None = None) -> ListToolsResult:
      return ListToolsResult(tools=[])

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


_INVESTMENT_DESK_SERVERS = ("idea-workbench-mcp", "macro-mcp", "positioning-mcp")


def _local_template_child_env(server_name: str, deployment_environ: dict[str, str]) -> dict[str, str]:
  """The env the gateway gives one shipped local-template server, as its spawn builds it."""
  servers = json.loads((ROOT / "deploy/mcp.local.template.json").read_text(encoding="utf-8"))["mcpServers"]
  config = servers[server_name]
  manager = McpClientManager(config_path=None)
  scoped, absent = manager._scoped_server_env(server_name, config, deployment_environ=deployment_environ)
  return _build_mcp_env(
    config.get("env"),
    environ=scoped,
    base_environ=deployment_environ,
    absent=absent,
  )


@pytest.mark.parametrize("server_name", _INVESTMENT_DESK_SERVERS)
def test_placed_runtime_investment_children_inherit_the_desk_and_shared_children_do_not(
  tmp_path, server_name
) -> None:
  from agent_gateway.runtime_placement import investment_state_dir, runtime_state_environment
  from hank_model_transport.user_runtime_credentials import USER_STORAGE_ID_ENV

  launch = {
    "PATH": "/usr/bin:/bin",
    "LOCAL_GATEWAY_PYTHON": "/runtime/python",
    "LOCAL_GATEWAY_RUNTIME_ROOT": "/runtime",
    "LOCAL_GATEWAY_RUNTIME_VERSION_ROOT": "/runtime/v1",
  }
  user_data_dir = tmp_path / "placement" / "user-data"
  placed = _local_template_child_env(
    server_name, {**launch, **runtime_state_environment(user_data_dir, "7")}
  )
  assert placed[USER_STORAGE_ID_ENV] == "7"
  assert placed["INVESTMENT_TOOLS_STATE_DIR"] == str(investment_state_dir(user_data_dir, "7"))

  shared = _local_template_child_env(server_name, launch)
  assert USER_STORAGE_ID_ENV not in shared
  assert "INVESTMENT_TOOLS_STATE_DIR" not in shared


_UNSET_DECLARED_CONFIG = {
  "command": "fake-mcp-server",
  "run_scoped_env": ["SET_NAME", "UNSET_NAME", "EMPTY_NAME"],
}


def test_scoped_server_env_omits_declared_names_unset_in_the_deployment_env() -> None:
  manager = McpClientManager(config_path=None)
  scoped, absent = manager._scoped_server_env(
    "child",
    _UNSET_DECLARED_CONFIG,
    deployment_environ={"SET_NAME": "value", "EMPTY_NAME": ""},
  )

  assert scoped == {"SET_NAME": "value", "EMPTY_NAME": ""}
  assert absent == frozenset({"UNSET_NAME"})


def test_scoped_server_env_delivers_declared_passthrough_from_the_run_mapping() -> None:
  manager = McpClientManager(
    config_path=None, server_env_passthrough={"child": {"PASS_NAME"}},
  )

  assert manager._scoped_server_env("child", {"command": "x"}, deployment_environ={}) == (
    {}, frozenset({"PASS_NAME"}),
  )
  assert manager._scoped_server_env(
    "child", {"command": "x"}, deployment_environ={"PASS_NAME": "run-scoped"},
  ) == ({"PASS_NAME": "run-scoped"}, frozenset())
  assert manager._scoped_server_env(
    "child",
    {"command": "x"},
    deployment_environ={"PASS_NAME": "run-scoped", "ANTHROPIC_API_KEY": "ambient"},
    projected_env={"PASS_NAME": "projected"},
  ) == ({"PASS_NAME": "projected"}, frozenset())


def test_jobs_oauth_requires_the_bound_run_owner(monkeypatch) -> None:
  from hank_model_transport.providers.anthropic_oauth import ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES

  manager = McpClientManager(
    config_path=None,
    server_env_passthrough={"jobs-mcp": set(ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES)},
  )
  environ = {
    "ANTHROPIC_AUTH_TOKEN": "owner1-seat",
    "USER_DATA_DIR": "/mnt/hank-data",
    "ANTHROPIC_API_KEY": "payg-must-not-enter",
  }
  config = {
    "command": "jobs-server",
    "run_scoped_env": ["ANTHROPIC_AUTH_TOKEN", "USER_DATA_DIR"],
    "env": {
      "ANTHROPIC_AUTH_TOKEN": "${ANTHROPIC_AUTH_TOKEN}",
      "USER_DATA_DIR": "${USER_DATA_DIR}",
    },
  }
  monkeypatch.setattr(mcp_client_module, "load_user_runtime_credentials", lambda _env: None)
  monkeypatch.setattr(
    mcp_client_module, "scope_admits_operator_seats",
    lambda scope, *, environ: str(scope) == "7",
  )
  assert manager._jobs_oauth_projection(environ) == {}
  unbound, _ = manager._scoped_server_env("jobs-mcp", config, deployment_environ=environ)
  assert "ANTHROPIC_AUTH_TOKEN" not in unbound
  assert unbound["USER_DATA_DIR"] == "/mnt/hank-data"

  class _Identity:
    def __init__(self, role: str, risk_user_id: int) -> None:
      self.role = role
      self.risk_user_id = risk_user_id

  class _Bundle:
    def __init__(self, role: str, risk_user_id: int) -> None:
      self.user = _Identity(role, risk_user_id)

  monkeypatch.setattr(
    mcp_client_module, "load_user_runtime_credentials",
    lambda _env: _Bundle("invite", 9),
  )
  assert manager._jobs_oauth_projection(environ) == {}
  monkeypatch.setattr(
    mcp_client_module, "load_user_runtime_credentials",
    lambda _env: _Bundle("owner", 8),
  )
  assert manager._jobs_oauth_projection(environ) == {}
  invite_scoped, _ = manager._scoped_server_env(
    "jobs-mcp", config, deployment_environ=environ,
  )
  assert "ANTHROPIC_AUTH_TOKEN" not in invite_scoped
  assert invite_scoped["USER_DATA_DIR"] == "/mnt/hank-data"

  monkeypatch.setattr(
    mcp_client_module, "load_user_runtime_credentials",
    lambda _env: _Bundle("owner", 7),
  )
  projected = manager._jobs_oauth_projection(environ)
  assert projected == {"ANTHROPIC_AUTH_TOKEN": "owner1-seat"}
  assert "USER_DATA_DIR" not in projected
  admitted, _ = manager._scoped_server_env(
    "jobs-mcp", config, deployment_environ=environ, projected_env=projected,
  )
  built = _build_mcp_env(config["env"], environ=admitted, base_environ={"PATH": "/usr/bin"})
  assert built["ANTHROPIC_AUTH_TOKEN"] == "owner1-seat"
  assert built["USER_DATA_DIR"] == "/mnt/hank-data"
  assert "payg-must-not-enter" not in built.values()


def test_non_owner_keeps_declared_user_data_dir_and_jobs_config() -> None:
  manager = McpClientManager(
    config_path=None,
    server_env_passthrough={"jobs-mcp": {
      "GMAIL_TOKEN_PATH",
      "GMAIL_CREDENTIALS_PATH",
      "INVESTMENT_TOOLS_DISABLE_RESEARCH_IDEAS",
      "ANTHROPIC_AUTH_TOKEN",
      "USER_DATA_DIR",
    }},
  )
  deployment = {
    "GMAIL_TOKEN_PATH": "/run/gmail-token",
    "GMAIL_CREDENTIALS_PATH": "/run/gmail-credentials",
    "INVESTMENT_TOOLS_DISABLE_RESEARCH_IDEAS": "1",
    "USER_DATA_DIR": "/mnt/hank-data",
    "ANTHROPIC_AUTH_TOKEN": "owner1-seat",
    "ANTHROPIC_API_KEY": "ambient-payg",
  }
  config = {
    "command": "jobs-server",
    "run_scoped_env": ["USER_DATA_DIR"],
    "env": {
      "GMAIL_TOKEN_PATH": "${GMAIL_TOKEN_PATH}",
      "GMAIL_CREDENTIALS_PATH": "${GMAIL_CREDENTIALS_PATH}",
      "INVESTMENT_TOOLS_DISABLE_RESEARCH_IDEAS": "${INVESTMENT_TOOLS_DISABLE_RESEARCH_IDEAS}",
      "USER_DATA_DIR": "${USER_DATA_DIR}",
      "ANTHROPIC_AUTH_TOKEN": "${ANTHROPIC_AUTH_TOKEN}",
    },
  }
  scoped, absent = manager._scoped_server_env(
    "jobs-mcp", config, deployment_environ=deployment,
  )
  assert scoped["GMAIL_TOKEN_PATH"] == "/run/gmail-token"
  assert scoped["GMAIL_CREDENTIALS_PATH"] == "/run/gmail-credentials"
  assert scoped["INVESTMENT_TOOLS_DISABLE_RESEARCH_IDEAS"] == "1"
  assert scoped["USER_DATA_DIR"] == "/mnt/hank-data"
  assert "ANTHROPIC_AUTH_TOKEN" in absent
  assert "ANTHROPIC_API_KEY" not in scoped
  built = _build_mcp_env(config["env"], environ=scoped, base_environ={}, absent=absent)
  assert built["USER_DATA_DIR"] == "/mnt/hank-data"
  assert "ANTHROPIC_AUTH_TOKEN" not in built
  assert "owner1-seat" not in built.values()


def test_build_mcp_env_omits_entries_referencing_an_absent_declared_name() -> None:
  environ = {"SET_NAME": "value"}
  env = _build_mcp_env(
    {
      "UNSET_NAME": "${UNSET_NAME}",
      "MIXED": "${SET_NAME}/x:${UNSET_NAME}",
      "SET_NAME": "${SET_NAME}",
      "LITERAL": "plain",
    },
    environ=environ,
    base_environ={},
    absent=frozenset({"UNSET_NAME"}),
  )

  assert env == {"SET_NAME": "value", "LITERAL": "plain"}


def test_build_mcp_env_still_refuses_undeclared_reference_with_absent_names() -> None:
  with pytest.raises(McpEnvExpansionError, match="NOT_DECLARED"):
    _build_mcp_env(
      {"X": "${NOT_DECLARED}"},
      environ={},
      base_environ={},
      absent=frozenset({"UNSET_NAME"}),
    )


def test_stdio_child_env_lacks_declared_unset_names_and_keeps_empty_values(monkeypatch) -> None:
  monkeypatch.setattr(
    mcp_client_module.os,
    "environ",
    {"PATH": "/usr/bin", "SET_NAME": "value", "EMPTY_NAME": ""},
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

  class _FakeClientSession:
    def __init__(self, read_stream, write_stream):
      pass

    async def __aenter__(self):
      return self

    async def __aexit__(self, exc_type, exc, tb):
      return None

    async def initialize(self):
      return None

    async def list_tools(self, *, params: PaginatedRequestParams | None = None) -> ListToolsResult:
      return ListToolsResult(tools=[])

  monkeypatch.setattr(mcp_client_module, "StdioServerParameters", _FakeServerParameters)
  monkeypatch.setattr(
    mcp_client_module, "stdio_client", lambda _params, errlog=None: _FakeStdioContext()
  )
  monkeypatch.setattr(mcp_client_module, "ClientSession", _FakeClientSession)
  monkeypatch.setattr(
    mcp_client_module, "_preflight_stdio_executable", lambda _command, _args, _env: None
  )

  manager = McpClientManager(config_path=None)
  state = _run(
    manager._connect(
      "child",
      {
        **_UNSET_DECLARED_CONFIG,
        "env": {
          "SET_NAME": "${SET_NAME}",
          "UNSET_NAME": "${UNSET_NAME}",
          "EMPTY_NAME": "${EMPTY_NAME}",
          "MIXED": "${SET_NAME}/x:${UNSET_NAME}",
        },
      },
    )
  )

  assert captured["env"] == {
    "PATH": "/usr/bin",
    "SET_NAME": "value",
    "EMPTY_NAME": "",
  }
  _run(manager._close_contexts(state.exit_contexts))
