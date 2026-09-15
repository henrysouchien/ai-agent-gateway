from types import SimpleNamespace

import pytest

from agent_gateway.runner_tool_audit import get_tool_risk_value, redact_tool_input_for_event
import agent_gateway.runner_tool_audit as audit


def test_tool_risk_does_not_guess_from_tool_names(monkeypatch) -> None:
  monkeypatch.setattr(audit, "load_server_policy_module", lambda: None)
  assert get_tool_risk_value("read_and_delete_everything") == "side_effecting"


def test_tool_risk_uses_bound_product_authority(monkeypatch) -> None:
  policy = SimpleNamespace(get_tool_risk_value=lambda name: {"unusual_read": "read_only"}[name])
  monkeypatch.setattr(audit, "load_server_policy_module", lambda: policy)
  assert get_tool_risk_value("unusual_read") == "read_only"
  with pytest.raises(KeyError):
    get_tool_risk_value("unknown")


def test_redaction_preserves_data_without_leaking_secrets() -> None:
  secret = "sk-ant-api03-CODEX-WAVE0-CANARY-DO-NOT-USE-8f21d7"
  original = {"symbol": "AAPL", "credential_note": secret}
  assert redact_tool_input_for_event("data_historical_prices", original) == {
    "symbol": "AAPL", "credential_note": "<redacted-secret>",
  }
  assert original["credential_note"] == secret


def test_redaction_failure_never_leaks_raw_input(monkeypatch) -> None:
  def broken_redactor(*args, **kwargs):
    raise ValueError("unredacted value")
  monkeypatch.setattr(audit, "redact_tool_input", broken_redactor)
  assert redact_tool_input_for_event("lookup", {"credential": "raw"}) == {
    "_boundary_error": "<secret-sanitization-failed>",
  }


def test_redaction_isolates_nested_raw_input(monkeypatch) -> None:
  def mutating_redactor(_name, payload, **kwargs):
    payload["nested"]["symbol"] = "MUTATED"
    return payload
  monkeypatch.setattr(audit, "redact_tool_input", mutating_redactor)
  original = {"nested": {"symbol": "AAPL"}}
  assert redact_tool_input_for_event("lookup", original) == {"nested": {"symbol": "MUTATED"}}
  assert original == {"nested": {"symbol": "AAPL"}}
