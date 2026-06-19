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


def test_redact_tool_input_falls_back_to_shallow_copy(monkeypatch) -> None:
  original_import = builtins.__import__

  def import_without_redaction(name: str, *args: Any, **kwargs: Any):
    if name == "agent.shared.tool_redaction":
      raise ImportError("forced missing redaction module")
    return original_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", import_without_redaction)

  original = {"secret": "token", "nested": {"kept": True}}
  redacted = redact_tool_input_for_event("web_fetch", original)

  assert redacted == original
  assert redacted is not original
  assert redacted["nested"] is original["nested"]
