from __future__ import annotations

from agent_gateway.artifact_readback import readback_artifact_ready_event


def _success_result(**artifact_overrides):
  artifact = {
    "contract_name": "RiskFingerprint",
    "data_source": "live",
    # On risk-module readbacks source_path can point at notes markdown — it must
    # never become the event's artifact_path.
    "source_path": "notes/skills/PCTY-quantifying-risk.md",
    "typed_outputs": {"risk_fingerprint": {}},
  }
  artifact.update(artifact_overrides)
  return {
    "status": "success",
    "ticker": "PCTY",
    "skill": "quantifying-risk",
    "artifact_id": "2026-07-01T152641.981-abc",
    "artifact": artifact,
  }


def test_success_synthesizes_readback_event() -> None:
  event = readback_artifact_ready_event("get_skill_artifact", _success_result(), "toolu_123")
  assert event["type"] == "artifact_ready"
  assert event["skill_run_id"] == "readback-toolu_123"
  assert event["ticker"] == "PCTY"
  assert event["skill"] == "quantifying-risk"
  assert event["artifact_id"] == "2026-07-01T152641.981-abc"
  assert event["artifact_path"] == "artifacts/PCTY/quantifying-risk/2026-07-01T152641.981-abc.json"
  assert event["binary_artifact_path"] is None
  assert event["contract_name"] == "RiskFingerprint"
  assert event["data_source"] == "live"
  assert event["origin"] == "readback"
  assert event["scope"] == "ticker"
  assert isinstance(event["ts"], float)


def test_sidecar_artifact_path_wins_over_convention() -> None:
  result = _success_result(artifact_path="artifacts/PCTY/quantifying-risk/custom.json")
  event = readback_artifact_ready_event("get_skill_artifact", result, "t1")
  assert event["artifact_path"] == "artifacts/PCTY/quantifying-risk/custom.json"


def test_missing_optional_fields_default_conservatively() -> None:
  result = _success_result()
  del result["artifact"]["contract_name"]
  del result["artifact"]["data_source"]
  event = readback_artifact_ready_event("get_skill_artifact", result, "t1")
  assert event is not None
  assert event["contract_name"] == ""
  assert event["data_source"] == "live"


def test_non_readback_tools_and_bad_results_return_none() -> None:
  ok = _success_result()
  assert readback_artifact_ready_event("get_positions", ok, "t1") is None
  assert readback_artifact_ready_event("get_skill_artifact", None, "t1") is None
  assert readback_artifact_ready_event("get_skill_artifact", "text", "t1") is None
  assert readback_artifact_ready_event("get_skill_artifact", {"status": "not_found"}, "t1") is None

  not_a_dict_artifact = _success_result()
  not_a_dict_artifact["artifact"] = "not-a-dict"
  assert readback_artifact_ready_event("get_skill_artifact", not_a_dict_artifact, "t1") is None

  blank_ticker = _success_result()
  blank_ticker["ticker"] = "  "
  assert readback_artifact_ready_event("get_skill_artifact", blank_ticker, "t1") is None

  missing_skill = _success_result()
  del missing_skill["skill"]
  assert readback_artifact_ready_event("get_skill_artifact", missing_skill, "t1") is None


def test_missing_tool_call_id_still_emits() -> None:
  event = readback_artifact_ready_event("get_skill_artifact", _success_result(), None)
  assert event["skill_run_id"] == "readback-unknown"
