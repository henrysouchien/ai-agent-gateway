# ruff: noqa: E402

from collections import UserDict
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp import MCPError
from mcp.types import CONNECTION_CLOSED, INTERNAL_ERROR, REQUEST_TIMEOUT

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.mcp_client as mcp_client_module
from agent_gateway import mcp_client_config
from agent_gateway.mcp_client import (
  _build_http_headers,
  _build_mcp_env,
  _resolve_mcp_config_path,
  _stdio_connect_retry_delay,
)
from agent_gateway.mcp_client_config import MCP_CHILD_FIXED_ENV, McpEnvExpansionError

FASTMCP_CHILD_ENV = {
  "FASTMCP_SHOW_SERVER_BANNER": "false",
  "FASTMCP_CHECK_FOR_UPDATES": "off",
}


def test_every_child_env_turns_off_the_fastmcp_banner_and_update_check() -> None:
  assert MCP_CHILD_FIXED_ENV == FASTMCP_CHILD_ENV
  assert mcp_client_config.build_mcp_env(None, environ={}, base_environ={}) == {
    "HANK_RUN_ROUTE_FILE": "", **FASTMCP_CHILD_ENV,
  }
  # An entry that asks for the banner explicitly still gets it.
  assert mcp_client_config.build_mcp_env(
    {"FASTMCP_SHOW_SERVER_BANNER": "true"}, environ={}, base_environ={},
  )["FASTMCP_SHOW_SERVER_BANNER"] == "true"


def test_config_helper_builds_env_and_headers_from_explicit_environ() -> None:
  environ = {
    "PATH": "/usr/bin",
    "HOME": "/tmp/home",
    "TOKEN": "secret",
    "DROP_ME": "nope",
  }

  assert mcp_client_config.build_mcp_env(
    {
      "AUTH": "Bearer ${TOKEN}",
      "OMIT": None,
    },
    environ=environ,
  ) == {
    "PATH": "/usr/bin",
    "HOME": "/tmp/home",
    "HANK_RUN_ROUTE_FILE": "",
    **FASTMCP_CHILD_ENV,
    "AUTH": "Bearer secret",
  }
  assert mcp_client_config.build_http_headers(
    {
      "Authorization": "Bearer ${TOKEN}",
      "Ignored": None,
    },
    environ=environ,
  ) == {"Authorization": "Bearer secret"}


def test_parent_wrappers_use_parent_module_environ(monkeypatch) -> None:
  monkeypatch.setattr(
    mcp_client_module,
    "os",
    SimpleNamespace(
      environ={
        "MCP_CONFIG_PATH": "/tmp/custom-mcp.json",
        "PATH": "/bin",
        "HEADER_TOKEN": "abc",
      }
    ),
  )

  assert _resolve_mcp_config_path() == Path("/tmp/custom-mcp.json")
  with pytest.raises(TypeError, match="environ"):
    _build_mcp_env({"TOKEN": "${HEADER_TOKEN}"})  # pyright: ignore[reportCallIssue]  # negative: projection is required
  with pytest.raises(TypeError, match="environ"):
    _build_http_headers({"Authorization": "Bearer ${HEADER_TOKEN}"})  # pyright: ignore[reportCallIssue]  # negative: projection is required
  with pytest.raises(McpEnvExpansionError, match="HEADER_TOKEN"):
    _build_mcp_env({"TOKEN": "${HEADER_TOKEN}"}, environ={})
  with pytest.raises(McpEnvExpansionError, match="HEADER_TOKEN"):
    _build_http_headers({"Authorization": "Bearer ${HEADER_TOKEN}"}, environ={})
  assert _build_mcp_env(
    {"TOKEN": "${HEADER_TOKEN}"}, environ={"HEADER_TOKEN": "abc"},
  ) == {
    "PATH": "/bin",
    "HANK_RUN_ROUTE_FILE": "",
    **FASTMCP_CHILD_ENV,
    "TOKEN": "abc",
  }
  assert _build_http_headers(
    {"Authorization": "Bearer ${HEADER_TOKEN}"}, environ={"HEADER_TOKEN": "abc"},
  ) == {"Authorization": "Bearer abc"}


def test_parent_retry_delay_uses_parent_env_and_random(monkeypatch) -> None:
  monkeypatch.setattr(
    mcp_client_module,
    "os",
    SimpleNamespace(environ={"MCP_STDIO_CONNECT_BACKOFF_S": "0.5"}),
  )
  monkeypatch.setattr(mcp_client_module, "random", SimpleNamespace(uniform=lambda low, high: high))

  assert _stdio_connect_retry_delay(3) == 2.5


def test_parent_config_path_does_not_resolve_home_for_none_or_explicit_path(monkeypatch) -> None:
  def _fail_home():
    raise AssertionError("Path.home should not be called")

  monkeypatch.setattr(mcp_client_module, "Path", SimpleNamespace(home=_fail_home))
  monkeypatch.setattr(mcp_client_module, "os", SimpleNamespace(environ={}))

  assert _resolve_mcp_config_path(None) is None
  assert _resolve_mcp_config_path("/tmp/mcp.json") == Path("/tmp/mcp.json")


def test_env_and_header_helpers_ignore_non_dict_mappings(monkeypatch) -> None:
  monkeypatch.setattr(
    mcp_client_module,
    "os",
    SimpleNamespace(environ={"PATH": "/bin", "TOKEN": "secret"}),
  )
  mapping = UserDict({"Authorization": "Bearer ${TOKEN}"})

  assert mcp_client_config.build_mcp_env(mapping, environ={"PATH": "/bin", "TOKEN": "secret"}) == {
    "PATH": "/bin", "HANK_RUN_ROUTE_FILE": "", **FASTMCP_CHILD_ENV,
  }
  assert mcp_client_config.build_http_headers(mapping, environ={"TOKEN": "secret"}) == {}
  assert _build_mcp_env(mapping, environ={}) == {"PATH": "/bin", "HANK_RUN_ROUTE_FILE": "", **FASTMCP_CHILD_ENV}  # pyright: ignore[reportArgumentType]  # negative: non-dict environment mapping ignored
  assert _build_http_headers(mapping, environ={}) == {}  # pyright: ignore[reportArgumentType]  # negative: non-dict header mapping ignored


def test_config_helper_invalid_numeric_logs_warning() -> None:
  warnings = []

  class _Logger:
    @staticmethod
    def warning(*args):
      warnings.append(args)

  assert mcp_client_config.env_nonnegative_int(
    "MCP_STARTUP_CONCURRENCY",
    4,
    environ={"MCP_STARTUP_CONCURRENCY": "invalid"},
    logger=_Logger(),
  ) == 4
  assert warnings == [("Ignoring invalid %s=%r; using %d", "MCP_STARTUP_CONCURRENCY", "invalid", 4)]


@pytest.mark.parametrize(
  ("exc", "retry_connect", "retry_startup"),
  [
    (MCPError(CONNECTION_CLOSED, "peer disappeared"), True, True),
    (MCPError(REQUEST_TIMEOUT, "response deadline elapsed"), False, True),
    (MCPError(INTERNAL_ERROR, "connection closed"), False, False),
  ],
)
def test_stdio_reconnect_classification_uses_mcp_codes(
  exc, retry_connect, retry_startup,
) -> None:
  grouped = ExceptionGroup("stdio failed", [exc])
  assert mcp_client_config.is_retryable_stdio_connect_error(grouped) is retry_connect
  assert mcp_client_config.is_retryable_stdio_startup_error(grouped) is retry_startup


def test_startup_concurrency_defaults_to_bounded_fanout() -> None:
  assert mcp_client_config.startup_concurrency_limit(environ={}) == 4


def test_startup_concurrency_preserves_explicit_override() -> None:
  assert mcp_client_config.startup_concurrency_limit(
    environ={"MCP_STARTUP_CONCURRENCY": "2"}
  ) == 2


def test_startup_concurrency_preserves_explicit_unbounded_escape_hatch() -> None:
  assert mcp_client_config.startup_concurrency_limit(
    environ={"MCP_STARTUP_CONCURRENCY": "0"}
  ) == 0


def test_startup_concurrency_rejects_invalid_or_negative_values() -> None:
  for value in ("invalid", "-1"):
    warnings = []

    class _Logger:
      @staticmethod
      def warning(*args):
        warnings.append(args)

    assert mcp_client_config.startup_concurrency_limit(
      environ={"MCP_STARTUP_CONCURRENCY": value},
      logger=_Logger(),
    ) == 4
    assert warnings == [
      ("Ignoring invalid %s=%r; using %d", "MCP_STARTUP_CONCURRENCY", value, 4)
    ]


def test_resolve_missing_stdio_executable_decisive_shapes(tmp_path: Path) -> None:
  present = tmp_path / "gsheets-mcp"
  present.write_text("#!/bin/sh\n", encoding="utf-8")
  present.chmod(0o755)
  missing = tmp_path / "deleted-venv" / "bin" / "gsheets-mcp"
  script = 'exec "$GSHEETS_MCP_EXECUTABLE" serve'

  # Existing executable target: no finding.
  assert mcp_client_config.resolve_missing_stdio_executable(
    "bash", ["-c", script], {"GSHEETS_MCP_EXECUTABLE": str(present)}
  ) is None
  # Deleted target behind the env var: decisive, names the path and the var.
  finding = mcp_client_config.resolve_missing_stdio_executable(
    "bash", ["-c", script], {"GSHEETS_MCP_EXECUTABLE": str(missing)}
  )
  assert finding is not None
  assert str(missing) in finding
  assert "GSHEETS_MCP_EXECUTABLE" in finding
  # Unset/empty env reference: decisive.
  finding = mcp_client_config.resolve_missing_stdio_executable(
    "bash", ["-c", script], {}
  )
  assert finding is not None
  assert "GSHEETS_MCP_EXECUTABLE" in finding
  # Present-but-not-executable target: decisive.
  plain = tmp_path / "not-executable"
  plain.write_text("", encoding="utf-8")
  plain.chmod(0o644)
  finding = mcp_client_config.resolve_missing_stdio_executable(
    str(plain), [], {}
  )
  assert finding is not None
  assert str(plain) in finding
  # Direct argv PATH lookup against the spawn env PATH.
  assert mcp_client_config.resolve_missing_stdio_executable(
    "gsheets-mcp", ["serve"], {"PATH": str(tmp_path)}
  ) is None
  finding = mcp_client_config.resolve_missing_stdio_executable(
    "definitely-not-a-real-mcp-server", [], {"PATH": str(tmp_path)}
  )
  assert finding is not None
  assert "definitely-not-a-real-mcp-server" in finding


def test_resolve_missing_stdio_executable_keeps_ambiguous_shapes_silent(
  tmp_path: Path,
) -> None:
  # Unrecognized shell scripts stay ambiguous: current behavior unchanged.
  assert mcp_client_config.resolve_missing_stdio_executable(
    "bash", ["-c", "some-wrapper --flag && exec server"], {}
  ) is None
  # Shell without -c payload stays ambiguous.
  assert mcp_client_config.resolve_missing_stdio_executable(
    "bash", ["script.sh"], {"PATH": "/usr/bin:/bin"}
  ) is None
  # Relative paths (cwd-dependent) stay ambiguous.
  assert mcp_client_config.resolve_missing_stdio_executable(
    "./relative/server", [], {}
  ) is None
  assert mcp_client_config.resolve_missing_stdio_executable(
    "bash",
    ["-c", 'exec "$SERVER_EXEC" serve'],
    {"SERVER_EXEC": "relative/server"},
  ) is None
  # Empty command stays ambiguous (existing missing-command handling owns it).
  assert mcp_client_config.resolve_missing_stdio_executable("", [], {}) is None


def test_allowed_tools_parser_preserves_exact_canonical_order() -> None:
  assert mcp_client_config.parse_allowed_tools(
    ["list_investment_capabilities", "get_investment_artifact"]
  ) == ("list_investment_capabilities", "get_investment_artifact")
  assert mcp_client_config.parse_allowed_tools(None) is None


@pytest.mark.parametrize(
  "value",
  (
    [],
    "get_investment_artifact",
    [""],
    [" get_investment_artifact"],
    ["get_investment_artifact", "get_investment_artifact"],
    [1],
  ),
)
def test_allowed_tools_parser_rejects_malformed_lists(value) -> None:
  with pytest.raises(ValueError, match="allowed_tools"):
    mcp_client_config.parse_allowed_tools(value)


def test_scoped_url_expansion_refuses_unset_or_empty_risk_api_url() -> None:
  """An HTTP url ref to an unset or empty RISK_API_URL is undeclared.

  An unset name is absent from the scoped env; an empty one is carried as "".
  The HTTP url expander refuses both instead of producing a relative URL.
  """
  manager = mcp_client_module.McpClientManager(config_path=None)
  config = {
    "type": "streamable-http",
    "url": "${RISK_API_URL}/api/mcp/portfolio-reads-mcp/",
    "per_user": True,
    "run_scoped_env": ["RISK_API_URL"],
  }
  for environ in ({}, {"RISK_API_URL": ""}):
    scoped, _ = manager._scoped_server_env(
      "portfolio-reads-mcp",
      config,
      deployment_environ=environ,
    )
    assert scoped.get("RISK_API_URL") == environ.get("RISK_API_URL")
    with pytest.raises(
      McpEnvExpansionError,
      match="undeclared environment variable: RISK_API_URL",
    ):
      mcp_client_config.expand_http_template(config["url"], environ=scoped)
    with pytest.raises(
      McpEnvExpansionError,
      match="undeclared environment variable: RISK_API_URL",
    ):
      mcp_client_config.build_http_headers(
        {"X-Risk-Origin": "${RISK_API_URL}"},
        environ=scoped,
      )
  assert mcp_client_config.expand_env_refs(
    "${RISK_API_URL}",
    environ={"RISK_API_URL": ""},
  ) == ""
  present, _ = manager._scoped_server_env(
    "portfolio-reads-mcp",
    config,
    deployment_environ={"RISK_API_URL": "https://risk.example.test"},
  )
  assert mcp_client_config.expand_http_template(config["url"], environ=present) == (
    "https://risk.example.test/api/mcp/portfolio-reads-mcp/"
  )


def test_scope_run_env_scopes_set_names_and_reports_the_rest_absent() -> None:
  scoped, absent = mcp_client_config.scope_run_env(
    ["SET_NAME", "UNSET_NAME", "EMPTY_NAME", "PROJECTED_NAME"],
    {"SET_NAME": "value", "EMPTY_NAME": "", "UNDECLARED": "ambient"},
    projected={"PROJECTED_NAME": "projected", "EXTRA": "extra"},
  )

  assert scoped == {
    "SET_NAME": "value",
    "EMPTY_NAME": "",
    "PROJECTED_NAME": "projected",
    "EXTRA": "extra",
  }
  assert absent == frozenset({"UNSET_NAME"})
