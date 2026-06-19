# ruff: noqa: E402

from collections import UserDict
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
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
  assert _build_mcp_env({"TOKEN": "${HEADER_TOKEN}"}) == {
    "PATH": "/bin",
    "TOKEN": "abc",
  }
  assert _build_http_headers({"Authorization": "Bearer ${HEADER_TOKEN}"}) == {
    "Authorization": "Bearer abc"
  }


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
    "PATH": "/bin"
  }
  assert mcp_client_config.build_http_headers(mapping, environ={"TOKEN": "secret"}) == {}
  assert _build_mcp_env(mapping) == {"PATH": "/bin"}
  assert _build_http_headers(mapping) == {}


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
