# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.tool_dispatcher import ToolDispatcher
from agent_gateway import tool_dispatcher_source_pack as source_pack_helpers


def _planner_payload(**overrides):
  payload = {
    "matched_intent": "revenue",
    "required_reads": ["10-K"],
    "rationale": "matched filing evidence",
  }
  payload.update(overrides)
  return payload


def test_source_pack_helper_extracts_nested_planner_payload() -> None:
  result = {"result": _planner_payload(form_type="10-K")}

  payload = source_pack_helpers.planner_result_payload(result)

  assert isinstance(payload, SimpleNamespace)
  assert payload.matched_intent == "revenue"
  assert payload.form_type == "10-K"


def test_tool_dispatcher_source_pack_private_wrappers_match_helper_outputs() -> None:
  result = {"planner_trace": _planner_payload(year=2025, quarter=2)}

  assert ToolDispatcher._planner_result_payload(result).__dict__ == (
    source_pack_helpers.planner_result_payload(result).__dict__
  )
  assert ToolDispatcher._planner_result_candidates(result) == (
    source_pack_helpers.planner_result_candidates(result)
  )
  assert ToolDispatcher._looks_like_source_pack_payload(_planner_payload())
  assert ToolDispatcher._payload_get({"form_type": "10-Q"}, "form_type") == "10-Q"
  assert (
    ToolDispatcher._derive_fiscal_period({"ticker": "MSFT"}, ToolDispatcher._planner_result_payload(result))
    == "FY2025 Q2"
  )


def test_tool_dispatcher_source_pack_parent_wrappers_honor_nested_overrides() -> None:
  class CustomDispatcher(ToolDispatcher):
    @classmethod
    def _planner_result_candidates(cls, result):
      assert result == "raw"
      return ["candidate"]

    @staticmethod
    def _looks_like_source_pack_payload(payload):
      return payload == "candidate"

    @staticmethod
    def _coerce_planner_result_payload(payload):
      return {"coerced": payload}

    @staticmethod
    def _payload_get(payload, key):
      return payload.get(f"patched_{key}")

  assert CustomDispatcher._planner_result_payload("raw") == {"coerced": "candidate"}
  assert CustomDispatcher._derive_fiscal_period(
    {},
    {"patched_year": 2028, "patched_quarter": 3},
  ) == "FY2028 Q3"


def test_source_pack_fiscal_period_prefers_explicit_tool_input() -> None:
  planner_result = SimpleNamespace(year=2024, quarter=1, fiscal_period="FY2024")

  assert source_pack_helpers.derive_fiscal_period(
    {"fiscal_period": "FY2026 Q3"},
    planner_result,
  ) == "FY2026 Q3"
  assert source_pack_helpers.derive_fiscal_period(
    {"fiscal_year": 2027, "fiscal_quarter": "4"},
    SimpleNamespace(),
  ) == "FY2027 Q4"


def test_capture_filing_source_pack_adapts_and_stores_pack(monkeypatch) -> None:
  stored = {}

  class FakeSourcePack:
    @classmethod
    def from_planner_result(
      cls,
      planner_result,
      *,
      ticker,
      fiscal_period,
      form_type,
    ):
      return {
        "planner_result": planner_result,
        "ticker": ticker,
        "fiscal_period": fiscal_period,
        "form_type": form_type,
      }

  fake_source_pack_session = SimpleNamespace(
    store=lambda session, pack: stored.update(session=session, pack=pack)
  )
  fake_agent = ModuleType("agent")
  fake_agent_shared = ModuleType("agent.shared")
  fake_agent_shared.source_pack_session = fake_source_pack_session
  fake_schema = ModuleType("schema")
  fake_schema_source_pack = ModuleType("schema.source_pack")
  fake_schema_source_pack.SourcePack = FakeSourcePack
  monkeypatch.setitem(sys.modules, "agent", fake_agent)
  monkeypatch.setitem(sys.modules, "agent.shared", fake_agent_shared)
  monkeypatch.setitem(sys.modules, "schema", fake_schema)
  monkeypatch.setitem(sys.modules, "schema.source_pack", fake_schema_source_pack)
  logger = SimpleNamespace(warnings=[])
  logger.warning = lambda *args: logger.warnings.append(args)

  source_pack_helpers.capture_filing_source_pack(
    "source-session",
    {"source_pack": _planner_payload(form_type="10-K", year=2025, quarter=2)},
    {"ticker": "MSFT"},
    logger,
  )

  assert logger.warnings == []
  assert stored == {
    "session": "source-session",
    "pack": {
      "planner_result": ToolDispatcher._planner_result_payload(
        {"source_pack": _planner_payload(form_type="10-K", year=2025, quarter=2)}
      ),
      "ticker": "MSFT",
      "fiscal_period": "FY2025 Q2",
      "form_type": "10-K",
    },
  }


def test_call_mcp_tool_captures_get_filing_evidence_source_pack(monkeypatch) -> None:
  class FakeMcp:
    def get_server_for_tool(self, _tool_name):
      return "edgar-parser-mcp"

    async def call_tool(self, tool_name, tool_input, **_kwargs):
      return {"source_pack": _planner_payload()}, None

  captured = {}
  dispatcher = ToolDispatcher(
    mcp_client=FakeMcp(),
    local_tool_handlers={},
    session="session",
  )

  def fake_capture(
    source_pack_session_target,
    result,
    tool_input,
    logger,
    *,
    planner_result_payload_fn,
    derive_fiscal_period_fn,
    payload_get_fn,
  ):
    captured["source_pack_session_target"] = source_pack_session_target
    captured["result"] = result
    captured["tool_input"] = tool_input
    captured["logger"] = logger
    captured["planner_result_payload_fn"] = planner_result_payload_fn
    captured["derive_fiscal_period_fn"] = derive_fiscal_period_fn
    captured["payload_get_fn"] = payload_get_fn

  monkeypatch.setattr(source_pack_helpers, "capture_filing_source_pack", fake_capture)

  import asyncio

  result, error = asyncio.run(
    dispatcher._call_mcp_tool(
      "get_filing_evidence",
      {"ticker": "MSFT"},
    )
  )

  assert error is None
  assert result == {"source_pack": _planner_payload()}
  assert captured["source_pack_session_target"] == "session"
  assert captured["result"] == result
  assert captured["tool_input"] == {"ticker": "MSFT"}
  assert captured["planner_result_payload_fn"].__func__ is dispatcher._planner_result_payload.__func__
  assert captured["derive_fiscal_period_fn"].__func__ is dispatcher._derive_fiscal_period.__func__
  assert captured["payload_get_fn"] == dispatcher._payload_get
