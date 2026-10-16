"""Derived-result lineage: a retained producer value keeps the sources it came with.

`hank.call` results carry their reference, `hank.input` selects from them and
`hank.derive` publishes a computed value as `derived` on the cell's result. The
sticky-kernel journey runs real interpreters through the real dispatcher.
"""
import asyncio
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
for path in (PKG_DIR, Path(__file__).resolve().parent):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from agent_gateway import SessionStore, ToolDispatcher, tool_dispatch_declarations
from agent_gateway.code_execution import CodeExecutionConfig, build_code_execution, cleanup_code_execution
from agent_gateway.code_execution._frame_codec import decode_frame, encode_frame
from agent_gateway.code_execution._frame_reader import lookup_active_cell
from agent_gateway.tool_dispatch_declarations import ToolDispatchSemantics
from agent_gateway.tool_dispatcher_helpers import ToolResult
from agent_workflow_contracts.tool_registration import RegisteredToolIdentity, VersionedPolicyRef
from test_code_execution_bridge import (
  FRED_SERIES, GET_FILINGS, _CODE_CALLABLE, _DISPATCH_SEMANTICS, _CaptureMcp, _bind, _filings_extractor, _policy,
)

_MONTHS = 60


def _fred_url(series_id: str) -> str:
  return f"https://fred.stlouisfed.org/series/{series_id}"


class _FredMcp(_CaptureMcp):
  """FRED-like producer: dated observations plus the producer's own source row."""

  async def call_tool(self, name: str, tool_input: Any, *args: Any, **kwargs: Any) -> ToolResult:
    if name != FRED_SERIES:
      return await super().call_tool(name, tool_input, *args, **kwargs)
    self.calls.append((name, dict(tool_input)))
    series_id = tool_input["series_id"]
    base = 250.0 if series_id == "CPIAUCSL" else 4.0
    observations = [
      {"date": f"{2021 + index // 12}-{index % 12 + 1:02d}-01", "value": round(base + index * 1.5, 3)}
      for index in range(_MONTHS)
    ]
    return {
      "status": "success",
      "series_id": series_id,
      "observations": observations,
      "sources": [{
        "source_kind": "vendor", "provider": "fred", "endpoint": "fred_get_series",
        "key_fields": {"series_id": series_id}, "source_url": _fred_url(series_id),
      }],
    }, None


@pytest.fixture(autouse=True)
def _fred_source_identity(monkeypatch: pytest.MonkeyPatch, _policy: Any) -> None:
  # The registration owner declares FRED's producer-issued source identity, as it
  # does for filings; the dispatcher settles `sources` from that declaration.
  monkeypatch.setitem(_DISPATCH_SEMANTICS, RegisteredToolIdentity(
    route_kind="mcp", logical_server_id="fred-mcp", logical_name="fred_get_series",
  ), ToolDispatchSemantics(
    idempotent=True,
    outcome_policy=VersionedPolicyRef(
      kind="outcome", policy_id="current-declared-status-equals", version="v1",
      parameters={"field": "status", "values": ("success",)},
    ),
    source_identity_policy=VersionedPolicyRef(
      kind="source_identity", policy_id="producer_vendor_sources", version="v1", parameters={},
    ),
  ))
  monkeypatch.setattr(tool_dispatch_declarations, "_CACHED_DECLARATIONS", None)


def _session(tmp_path: Path) -> tuple[Any, Any, ToolDispatcher]:
  session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
  config = CodeExecutionConfig(register_kernel=True, register_docker=False, work_dir_root=str(tmp_path))
  bundle = build_code_execution(session, config=config)
  mcp = _FredMcp(frame_reader=bundle.frame_reader)
  return session, bundle, _bind(bundle, mcp, extract_sources=_filings_extractor)


async def _cell(dispatcher: ToolDispatcher, call_id: str, code: str, host: str = "kernel") -> dict[str, Any]:
  result, error = await dispatcher.dispatch(
    call_id, "code_execute", {"code": textwrap.dedent(code), "host": host},
  )
  assert error is None and result is not None, error
  assert result["return_code"] == 0, result["stderr"]
  return result


def _row(sources: list[dict[str, Any]], series_id: str) -> dict[str, Any]:
  [row] = [source for source in sources if source.get("key_fields") == {"series_id": series_id}]
  return row


def test_later_kernel_cell_derives_cpi_changes_from_retained_observations(tmp_path: Path) -> None:
  async def run() -> None:
    session, _bundle, dispatcher = _session(tmp_path)
    try:
      first = await _cell(dispatcher, "cell-1", f"""
        cpi = hank.call({FRED_SERIES!r}, series_id="CPIAUCSL", tail={_MONTHS})
        unused = hank.call({FRED_SERIES!r}, series_id="UNRATE", tail={_MONTHS})
        print(len(cpi["observations"]))
      """)
      cpi_row = _row(first["sources"], "CPIAUCSL")
      unused_row = _row(first["sources"], "UNRATE")
      assert [call["nested_call_id"] for call in first["nested_calls"]] == ["cell-1.1", "cell-1.2"]
      cpi_refs = first["nested_calls"][0]["source_refs"]
      assert len(cpi_refs) == 1 and cpi_refs != first["nested_calls"][1]["source_refs"]
      assert "derived" not in first

      second = await _cell(dispatcher, "cell-2", """
        obs = cpi["observations"]
        latest = hank.input(obs, -1)
        four_years_ago = hank.input(obs, -49)
        two_years_ago = hank.input(obs, -25)
        four_year = latest["value"]["value"] / four_years_ago["value"]["value"] - 1
        two_year = latest["value"]["value"] / two_years_ago["value"]["value"] - 1
        hank.derive("cpi_4y_change", round(four_year, 4),
                    expression="obs[-1].value / obs[-49].value - 1", inputs=[latest, four_years_ago])
        hank.derive("cpi_2y_change", round(two_year, 4),
                    expression="obs[-1].value / obs[-25].value - 1", inputs=[latest, two_years_ago])
      """)
      observations = [
        {"date": f"{2021 + index // 12}-{index % 12 + 1:02d}-01", "value": round(250.0 + index * 1.5, 3)}
        for index in range(_MONTHS)
      ]
      assert "sources" not in second and "nested_calls" not in second
      four, two = second["derived"]
      assert (four["name"], four["status"], four["findings"]) == ("cpi_4y_change", "derived", [])
      assert four["expression"] == "obs[-1].value / obs[-49].value - 1"
      assert four["value"] == round(observations[-1]["value"] / observations[-49]["value"] - 1, 4)
      assert four["inputs"] == [
        {"nested_call_id": "cell-1.1", "path": ["observations", 59], "value": observations[-1], "sources": [cpi_row]},
        {"nested_call_id": "cell-1.1", "path": ["observations", 11], "value": observations[-49], "sources": [cpi_row]},
      ]
      assert (two["name"], two["status"], two["findings"]) == ("cpi_2y_change", "derived", [])
      assert two["inputs"][1] == {
        "nested_call_id": "cell-1.1", "path": ["observations", 35], "value": observations[-25], "sources": [cpi_row],
      }
      assert unused_row not in [source for row in (four, two) for item in row["inputs"] for source in item["sources"]]
      assert cpi_row["produced_by_tool"] == "fred_get_series"
      assert json.loads(json.dumps(second["derived"])) == second["derived"]
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


@pytest.mark.parametrize("host", ["subprocess", "kernel"])
def test_repeated_selection_in_one_cell_yields_equivalent_inputs(tmp_path: Path, host: str) -> None:
  async def run() -> None:
    session, _bundle, dispatcher = _session(tmp_path)
    try:
      result = await _cell(dispatcher, "cell-1", f"""
        obs = hank.call({FRED_SERIES!r}, series_id="CPIAUCSL", tail={_MONTHS})["observations"]
        for _ in range(2):
            latest, prior = hank.input(obs, -1), hank.input(obs, -13)
            hank.derive("yoy", latest["value"]["value"] / prior["value"]["value"] - 1,
                        expression="latest / prior - 1", inputs=[latest, prior])
      """, host)
      first, second = result["derived"]
      assert first["inputs"] == second["inputs"] and first["findings"] == second["findings"] == []
      assert [item["path"] for item in first["inputs"]] == [["observations", 59], ["observations", 47]]
      assert all(item["sources"] == [_row(result["sources"], "CPIAUCSL")] for item in first["inputs"])
      assert len(result["sources"]) == 1 and len(result["nested_calls"]) == 1
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_inputs_without_a_producer_reference_are_typed_findings(tmp_path: Path) -> None:
  async def run() -> None:
    session, _bundle, dispatcher = _session(tmp_path)
    try:
      result = await _cell(dispatcher, "cell-1", f"""
        series = hank.call({FRED_SERIES!r}, series_id="CPIAUCSL", tail={_MONTHS})
        sourced = hank.input(series["observations"], -1)
        typed = hank.input({{"value": 99.0}}, "value")
        hank.derive("mixed", 1.0, expression="a + 0.5 + 0.5", inputs=[sourced, typed, 0.5])
        # A reference this session never issued resolves to no source.
        forged = hank.input(series, "series_id")
        forged["source_refs"] = ["src_0000000000000000"]
        hank.derive("forged", 2.0, expression="x", inputs=[forged])
      """)
      mixed, forged = result["derived"]
      assert [(finding["code"], finding["input"]) for finding in mixed["findings"]] == [
        ("input_unsourced", 1), ("input_unsourced", 2),
      ]
      assert mixed["inputs"][0]["sources"] == [_row(result["sources"], "CPIAUCSL")]
      assert [(item["value"], item["sources"], item["nested_call_id"]) for item in mixed["inputs"][1:]] == [
        (99.0, [], None), (0.5, [], None),
      ]
      assert forged["status"] == "derived" and forged["inputs"][0]["sources"] == []
      assert [finding["code"] for finding in forged["findings"]] == ["input_unsourced"]
      assert "sources" in result and len(result["sources"]) == 1  # Derived rows never become sources.
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_carrier_behaves_as_the_producer_json(tmp_path: Path) -> None:
  async def run() -> None:
    session, _bundle, dispatcher = _session(tmp_path)
    try:
      result = await _cell(dispatcher, "cell-1", f"""
        import copy, json
        import pandas as pd
        cpi = hank.call({FRED_SERIES!r}, series_id="CPIAUCSL", tail={_MONTHS})
        plain = json.loads(json.dumps(cpi))
        obs = cpi["observations"]
        copied = copy.deepcopy(obs)
        frame = pd.DataFrame(obs)
        print(json.dumps({{
            "types": [isinstance(cpi, dict), isinstance(obs, list), isinstance(obs[0], dict)],
            "equal": [cpi == plain, obs == plain["observations"], copied == obs],
            "same_json": json.dumps(cpi, sort_keys=True) == json.dumps(plain, sort_keys=True),
            "repr": repr(obs[:2]) == repr(plain["observations"][:2]),
            "frame": [frame.shape[0], float(frame["value"].iloc[-1])],
            "slice_value": hank.input(obs, slice(-2, None))["value"] == plain["observations"][-2:],
            "numpy": hank.derive("n", frame["value"].iloc[-1], expression="last", inputs=[obs[-1]])["value"],
        }}))
      """)
      checks = json.loads(result["stdout"])
      assert checks["types"] == [True, True, True] and checks["equal"] == [True, True, True]
      assert checks["same_json"] and checks["repr"] and checks["slice_value"]
      assert checks["frame"] == [_MONTHS, 250.0 + (_MONTHS - 1) * 1.5] and checks["numpy"] == checks["frame"][1]
      [derived] = result["derived"]
      assert derived["inputs"][0]["path"] == ["observations", _MONTHS - 1] and derived["findings"] == []
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_native_settled_call_issues_refs_that_a_later_cell_derives_from(tmp_path: Path) -> None:
  async def run() -> None:
    session, bundle, _dispatcher = _session(tmp_path)
    reader = bundle.frame_reader
    native = reader.open_cell("native-cell", abort_event=None, active=True)
    later = reader.open_cell("later-cell", abort_event=None, active=True)
    try:
      active = lookup_active_cell(native.token)
      assert active is not None
      entry = active.start_native(GET_FILINGS, {"cik": "AAPL"})
      active.settle_native(entry, result={
        "status": "success",
        "filings": [{"document_id": "edgar:0000320193-24-000123", "accession": "0000320193-24-000123"}],
      }, error=None)
      assert len(entry["source_refs"]) == 1
      reply = decode_frame(await later.handle_frame(encode_frame({
        "type": "derive", "call_id": "d1", "cell_token": later.token, "parent_call_id": later.outer_call_id,
        "name": "form_count", "value": 1, "expression": "len(filings)",
        "inputs": [{
          "nested_call_id": entry["nested_call_id"], "source_refs": entry["source_refs"],
          "path": ["filings"], "value": [{"form": "10-K"}],
        }],
      })))
      assert reply["ok"], reply
      [row] = later.summary()["derived"]
      assert row["findings"] == [] and row["inputs"][0]["nested_call_id"] == "native-cell.1"
      assert [source["document_id"] for source in row["inputs"][0]["sources"]] == ["edgar:0000320193-24-000123"]
      assert "sources" not in later.summary() and "nested_calls" not in later.summary()
      assert reply["result"] == row
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_native_result_carried_in_a_cell_derives_from_the_native_settlement_in_a_later_cell(tmp_path: Path) -> None:
  async def run() -> None:
    session, bundle, dispatcher = _session(tmp_path)
    native = bundle.frame_reader.open_cell("native-cell", abort_event=None, active=True)
    try:
      active = lookup_active_cell(native.token)
      assert active is not None
      entry = active.start_native(GET_FILINGS, {"cik": "AAPL"})
      result = {
        "status": "success",
        "filings": [{"document_id": "edgar:0000320193-24-000123", "accession": "0000320193-24-000123"}],
      }
      active.settle_native(entry, result=result, error=None)
      assert len(entry["source_refs"]) == 1
      # The settle response carries `entry["source_refs"]`; the native client hands the
      # plain result and that reference to `hank.carry`, which `_risk` is constructed with.
      first = await _cell(dispatcher, "cell-1", f"""
        import json
        plain = json.loads({json.dumps(result)!r})
        risk = hank.carry(plain, {entry["nested_call_id"]!r}, {entry["source_refs"]!r})
        print(json.dumps([risk == plain, json.dumps(risk) == json.dumps(plain), type(risk) is dict]))
      """)
      assert json.loads(first["stdout"]) == [True, True, False]
      second = await _cell(dispatcher, "cell-2", """
        filing = hank.input(risk["filings"], 0)
        hank.derive("accession", filing["value"]["accession"], expression="filings[0].accession", inputs=[filing])
      """)
      [row] = second["derived"]
      assert row["findings"] == [] and row["inputs"][0]["nested_call_id"] == "native-cell.1"
      assert row["inputs"][0]["path"] == ["filings", 0]
      assert [source["document_id"] for source in row["inputs"][0]["sources"]] == ["edgar:0000320193-24-000123"]
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_sliced_carrier_list_keeps_lineage_with_the_slice_in_its_path(tmp_path: Path) -> None:
  async def run() -> None:
    session, _bundle, dispatcher = _session(tmp_path)
    try:
      result = await _cell(dispatcher, "cell-1", f"""
        obs = hank.call({FRED_SERIES!r}, series_id="CPIAUCSL", tail={_MONTHS})["observations"]
        window = obs[-12:]
        mean = sum(row["value"] for row in window) / len(window)
        hank.derive("trailing_mean", mean, expression="mean(obs[-12:].value)", inputs=[window])
        hank.derive("last_in_window", window[-1]["value"], expression="obs[-12:][-1].value",
                    inputs=[hank.input(window, -1)])
        plain = [row for row in obs][-12:]
        hank.derive("plain_copy", 0, expression="not a carrier", inputs=[plain])
        print(type(window).__name__, window == plain, len(window))
      """)
      assert result["stdout"].split() == ["_HankRefList", "True", "12"]
      observations = [
        {"date": f"{2021 + index // 12}-{index % 12 + 1:02d}-01", "value": round(250.0 + index * 1.5, 3)}
        for index in range(_MONTHS)
      ]
      cpi_row = _row(result["sources"], "CPIAUCSL")
      window_row, last_row, plain_row = result["derived"]
      window_path = ["observations", {"slice": [_MONTHS - 12, _MONTHS, 1]}]
      assert window_row["findings"] == [] and last_row["findings"] == []
      assert window_row["inputs"] == [{
        "nested_call_id": "cell-1.1", "path": window_path, "value": observations[-12:], "sources": [cpi_row],
      }]
      assert last_row["inputs"][0]["path"] == [*window_path, 11]
      assert last_row["inputs"][0]["value"] == observations[-1]
      assert [finding["code"] for finding in plain_row["findings"]] == ["input_unsourced"]
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())
