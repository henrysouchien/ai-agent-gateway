import builtins
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.runner_tool_audit import get_tool_risk_value, redact_tool_input_for_event  # noqa: E402


def test_runner_preserves_tool_audit_helper_aliases() -> None:
  assert gateway_runner._get_tool_risk_value is get_tool_risk_value
  assert gateway_runner._redact_tool_input_for_event is redact_tool_input_for_event


def test_tool_risk_value_fallback_classifies_common_patterns(monkeypatch) -> None:
  original_import = builtins.__import__

  def import_without_tool_risk(name: str, *args: Any, **kwargs: Any):
    if name == "api.agent.shared.tool_risk":
      raise ImportError("forced missing tool risk module")
    return original_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", import_without_tool_risk)

  assert get_tool_risk_value("file_read") == "read_only"
  assert get_tool_risk_value(" list_reports ") == "read_only"
  assert get_tool_risk_value("memory_write") == "idempotent_write"
  assert get_tool_risk_value("delete_everything") == "side_effecting"


def test_tool_risk_value_returns_side_effecting_when_registry_raises(monkeypatch) -> None:
  def broken_get_tool_risk(_tool_name: str):
    raise RuntimeError("broken registry")

  original_import = builtins.__import__

  def import_broken_tool_risk(name: str, globals=None, locals=None, fromlist=(), level=0):
    if name == "api.agent.shared.tool_risk" and "get_tool_risk" in fromlist:
      return type("_BrokenToolRiskModule", (), {"get_tool_risk": broken_get_tool_risk})()
    return original_import(name, globals, locals, fromlist, level)

  monkeypatch.setattr(builtins, "__import__", import_broken_tool_risk)

  assert get_tool_risk_value("file_read") == "side_effecting"


def test_redact_tool_input_fails_closed_when_policy_import_fails(monkeypatch) -> None:
  original_import = builtins.__import__

  def import_without_redaction(name: str, *args: Any, **kwargs: Any):
    if name == "agent.shared.tool_redaction":
      raise ImportError("forced missing redaction module")
    return original_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", import_without_redaction)

  original = {"secret": "sk-ant-api03-CODEX-WAVE0-CANARY-DO-NOT-USE-8f21d7"}
  redacted = redact_tool_input_for_event("web_fetch", original)

  assert redacted == {"_boundary_error": "<secret-sanitization-failed>"}


def test_redact_tool_input_uses_packaged_projection_only_when_host_is_absent(
  monkeypatch,
) -> None:
  original_import = builtins.__import__

  def import_without_host_redaction(name: str, *args: Any, **kwargs: Any):
    if name == "agent.shared.tool_redaction":
      raise ModuleNotFoundError(
        "No module named 'agent'",
        name="agent",
      )
    return original_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", import_without_host_redaction)

  secret = "sk-ant-api03-CODEX-WAVE0-CANARY-DO-NOT-USE-8f21d7"
  original = {
    "symbol": "AAPL",
    "start_date": "2025-01-01",
    "end_date": "2025-01-31",
    "credential_note": secret,
  }

  redacted = redact_tool_input_for_event("data_historical_prices", original)

  assert redacted == {
    "symbol": "AAPL",
    "start_date": "2025-01-01",
    "end_date": "2025-01-31",
    "credential_note": "<redacted-secret>",
  }
  assert original["credential_note"] == secret


def test_redact_tool_input_does_not_fallback_for_host_transitive_import_failure(
  monkeypatch,
) -> None:
  original_import = builtins.__import__

  def import_with_broken_host_dependency(name: str, *args: Any, **kwargs: Any):
    if name == "agent.shared.tool_redaction":
      raise ModuleNotFoundError(
        "No module named 'host_redaction_dependency'",
        name="host_redaction_dependency",
      )
    return original_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", import_with_broken_host_dependency)

  assert redact_tool_input_for_event(
    "data_historical_prices",
    {"symbol": "AAPL"},
  ) == {"_boundary_error": "<secret-sanitization-failed>"}


def test_redact_tool_input_isolates_raw_input_from_selected_host_redactor(
  monkeypatch,
) -> None:
  def mutating_redactor(
    _tool_name: str,
    tool_input: dict[str, Any],
    *,
    deployment_secret: bytes,
  ) -> dict[str, Any]:
    assert deployment_secret == b"test-audit-secret"
    tool_input["symbol"] = "MUTATED"
    return tool_input

  original_import = builtins.__import__

  def import_mutating_redactor(name: str, globals=None, locals=None, fromlist=(), level=0):
    if name == "agent.shared.tool_redaction":
      return type(
        "_MutatingRedactionModule",
        (),
        {
          "get_audit_hmac_secret": staticmethod(lambda: b"test-audit-secret"),
          "redact_tool_input": staticmethod(mutating_redactor),
        },
      )()
    return original_import(name, globals, locals, fromlist, level)

  monkeypatch.setattr(builtins, "__import__", import_mutating_redactor)
  original = {"symbol": "AAPL"}

  assert redact_tool_input_for_event(
    "data_historical_prices",
    original,
  ) == {"symbol": "MUTATED"}
  assert original == {"symbol": "AAPL"}
