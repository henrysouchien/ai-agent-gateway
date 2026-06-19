# ruff: noqa: E402

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.mcp_client as mcp_client_module
from agent_gateway import mcp_client_errors


def test_error_classifiers_match_parent_private_wrappers() -> None:
  cases = [
    (asyncio.TimeoutError(), "operation timed out", "timeout"),
    (ConnectionError("connection refused"), "connection refused", "connection_error"),
    (RuntimeError("other"), "other", "unknown"),
  ]

  for exc, message, expected in cases:
    assert mcp_client_errors.classify_exception(exc, message) == expected
    assert mcp_client_module._classify_exception(exc, message) == expected

  mcp_error_cases = [
    ("filing not found", "not_found"),
    ("malformed payload", "parse_error"),
    ("request timed out", "timeout"),
    ("unexpected", "unknown"),
  ]
  for message, expected in mcp_error_cases:
    assert mcp_client_errors.classify_mcp_error(message) == expected
    assert mcp_client_module._classify_mcp_error(message) == expected


def test_startup_failure_helper_classifies_timeout_config_and_default() -> None:
  def never_retryable(_exc: BaseException) -> bool:
    return False

  assert mcp_client_errors.startup_failure_from_exception(
    asyncio.TimeoutError(),
    is_retryable_stdio_connect_error=never_retryable,
  ) == {
    "category": "transient_timeout",
    "retryable": True,
    "message": "TimeoutError",
    "error_type": "TimeoutError",
  }
  assert mcp_client_errors.startup_failure_from_exception(
    ValueError("missing url"),
    is_retryable_stdio_connect_error=never_retryable,
  ) == {
    "category": "config_error",
    "retryable": False,
    "message": "missing url",
    "error_type": "ValueError",
  }
  assert mcp_client_errors.startup_failure_from_exception(
    RuntimeError("boom"),
    is_retryable_stdio_connect_error=never_retryable,
  ) == {
    "category": "startup_error",
    "retryable": False,
    "message": "boom",
    "error_type": "RuntimeError",
  }


def test_parent_startup_failure_wrapper_uses_parent_retryable_checker(monkeypatch) -> None:
  checked: list[BaseException] = []
  exc = RuntimeError("closed transport")

  def fake_retryable(candidate: BaseException) -> bool:
    checked.append(candidate)
    return candidate is exc

  monkeypatch.setattr(mcp_client_module, "_is_retryable_stdio_connect_error", fake_retryable)

  assert mcp_client_module._startup_failure_from_exception(exc) == {
    "category": "transient_transport",
    "retryable": True,
    "message": "closed transport",
    "error_type": "RuntimeError",
  }
  assert checked == [exc]
