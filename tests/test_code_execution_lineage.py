"""Cell lineage: producer values are plain, and the host ledger keeps their sources.

`hank.call` returns plain `dict`/`list` values; a cell computes on them with
ordinary Python and returns plain `dict`/`list` values. There is no
`hank.derive`: provenance is recorded host-side on the cell's nested-call
ledger (`nested_calls` with `source_refs`, and `sources`), and lineage into a
model flows only through `model_document` formulas / `fms_persist`. The
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
  config = CodeExecutionConfig(work_dir_root=str(tmp_path))
  bundle = build_code_execution(session, config=config)
  mcp = _FredMcp(frame_reader=bundle.frame_reader)
  return session, bundle, _bind(bundle, mcp, extract_sources=_filings_extractor)


async def _cell(dispatcher: ToolDispatcher, call_id: str, code: str) -> dict[str, Any]:
  result, error = await dispatcher.dispatch(
    call_id, "code_execute", {"code": textwrap.dedent(code)},
  )
  assert error is None and result is not None, error
  assert result["return_code"] == 0, result["stderr"]
  return result


def _row(sources: list[dict[str, Any]], series_id: str) -> dict[str, Any]:
  [row] = [source for source in sources if source.get("key_fields") == {"series_id": series_id}]
  return row


def _observations() -> list[dict[str, Any]]:
  return [
    {"date": f"{2021 + index // 12}-{index % 12 + 1:02d}-01", "value": round(250.0 + index * 1.5, 3)}
    for index in range(_MONTHS)
  ]


def test_retained_producer_values_are_plain_and_a_later_cell_returns_plain_dicts(tmp_path: Path) -> None:
  async def run() -> None:
    session, _bundle, dispatcher = _session(tmp_path)
    try:
      first = await _cell(dispatcher, "cell-1", f"""
        cpi = hank.call({FRED_SERIES!r}, series_id="CPIAUCSL", tail={_MONTHS})
        unused = hank.call({FRED_SERIES!r}, series_id="UNRATE", tail={_MONTHS})
        assert type(cpi) is dict and type(cpi["observations"]) is list and type(cpi["observations"][0]) is dict
        print(len(cpi["observations"]))
      """)
      cpi_row = _row(first["sources"], "CPIAUCSL")
      unused_row = _row(first["sources"], "UNRATE")
      assert [call["nested_call_id"] for call in first["nested_calls"]] == ["cell-1.1", "cell-1.2"]
      cpi_refs = first["nested_calls"][0]["source_refs"]
      assert len(cpi_refs) == 1 and cpi_refs != first["nested_calls"][1]["source_refs"]
      assert cpi_row["produced_by_tool"] == "fred_get_series"
      assert "derived" not in first

      second = await _cell(dispatcher, "cell-2", """
        import json
        obs = cpi["observations"]
        four_year = round(obs[-1]["value"] / obs[-49]["value"] - 1, 4)
        two_year = round(obs[-1]["value"] / obs[-25]["value"] - 1, 4)
        result = {"cpi_4y_change": four_year, "cpi_2y_change": two_year, "points": [obs[-1], obs[-49]]}
        assert type(result) is dict and type(result["points"]) is list
        print(json.dumps(result))
      """)
      observations = _observations()
      returned = json.loads(second["stdout"])
      assert returned["cpi_4y_change"] == round(observations[-1]["value"] / observations[-49]["value"] - 1, 4)
      assert returned["cpi_2y_change"] == round(observations[-1]["value"] / observations[-25]["value"] - 1, 4)
      assert returned["points"] == [observations[-1], observations[-49]]
      # A cell that made no call records no ledger and publishes no `derived` row.
      assert "sources" not in second and "nested_calls" not in second and "derived" not in second
      assert unused_row != cpi_row
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_hank_has_no_derive_and_calling_it_raises_attribute_error(tmp_path: Path) -> None:
  async def run() -> None:
    session, _bundle, dispatcher = _session(tmp_path)
    try:
      result = await _cell(dispatcher, "cell-1", """
        import json
        absent = not hasattr(hank, "derive")
        try:
          hank.derive("x", 1.0, expression="x", inputs=[])
          raised = None
        except AttributeError as error:
          raised = str(error)
        print(json.dumps([absent, raised]))
      """)
      absent, raised = json.loads(result["stdout"])
      assert absent is True and raised == "derive"
      # A rejected attribute is not an operation: nothing was framed to the host.
      assert "nested_calls" not in result and "sources" not in result and "derived" not in result
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_repeated_reads_of_one_call_keep_one_ledger_entry_and_one_source(tmp_path: Path) -> None:
  async def run() -> None:
    session, _bundle, dispatcher = _session(tmp_path)
    try:
      first = await _cell(dispatcher, "cell-1", f"""
        obs = hank.call({FRED_SERIES!r}, series_id="CPIAUCSL", tail={_MONTHS})["observations"]
        assert type(obs) is list and type(obs[0]) is dict
      """)
      assert len(first["sources"]) == 1 and len(first["nested_calls"]) == 1
      result = await _cell(dispatcher, "cell-2", """
        import json
        rows = []
        for _ in range(2):
          rows.append({"yoy": obs[-1]["value"] / obs[-13]["value"] - 1, "latest": obs[-1]})
        print(json.dumps(rows[0] == rows[1]))
      """)
      assert json.loads(result["stdout"]) is True
      assert "sources" not in result and "nested_calls" not in result
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_a_second_call_in_a_later_cell_ledgers_its_own_sources_only(tmp_path: Path) -> None:
  async def run() -> None:
    session, _bundle, dispatcher = _session(tmp_path)
    try:
      first = await _cell(dispatcher, "cell-1", f"""
        cpi = hank.call({FRED_SERIES!r}, series_id="CPIAUCSL", tail={_MONTHS})
      """)
      second = await _cell(dispatcher, "cell-2", f"""
        rate = hank.call({FRED_SERIES!r}, series_id="UNRATE", tail={_MONTHS})
        spread = cpi["observations"][-1]["value"] - rate["observations"][-1]["value"]
        assert type(spread) is float
      """)
      assert [source["key_fields"] for source in first["sources"]] == [{"series_id": "CPIAUCSL"}]
      assert [source["key_fields"] for source in second["sources"]] == [{"series_id": "UNRATE"}]
      assert [call["nested_call_id"] for call in second["nested_calls"]] == ["cell-2.1"]
      assert second["nested_calls"][0]["source_refs"] != first["nested_calls"][0]["source_refs"]
      assert "derived" not in first and "derived" not in second
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_producer_values_behave_as_plain_json(tmp_path: Path) -> None:
  async def run() -> None:
    session, _bundle, dispatcher = _session(tmp_path)
    try:
      first = await _cell(dispatcher, "cell-1", f"""
        import copy, json
        import pandas as pd
        cpi = hank.call({FRED_SERIES!r}, series_id="CPIAUCSL", tail={_MONTHS})
        plain = json.loads(json.dumps(cpi))
        obs = cpi["observations"]
        copied = copy.deepcopy(obs)
        frame = pd.DataFrame(obs)
        print(json.dumps({{
            "types": [type(cpi) is dict, type(obs) is list, type(obs[0]) is dict, type(obs[:2]) is list],
            "equal": [cpi == plain, obs == plain["observations"], copied == obs],
            "same_json": json.dumps(cpi, sort_keys=True) == json.dumps(plain, sort_keys=True),
            "repr": repr(obs[:2]) == repr(plain["observations"][:2]),
            "frame": [frame.shape[0], float(frame["value"].iloc[-1])],
            "no_derive": not hasattr(hank, "derive"),
        }}))
      """)
      result = await _cell(dispatcher, "cell-2", """
        import json
        print(json.dumps(float(frame["value"].iloc[-1])))
      """)
      checks = json.loads(first["stdout"])
      assert checks["types"] == [True, True, True, True] and checks["equal"] == [True, True, True]
      assert checks["same_json"] and checks["repr"] and checks["no_derive"]
      assert checks["frame"] == [_MONTHS, 250.0 + (_MONTHS - 1) * 1.5]
      assert json.loads(result["stdout"]) == checks["frame"][1]
      assert "derived" not in first and "derived" not in result
      assert _row(first["sources"], "CPIAUCSL")["produced_by_tool"] == "fred_get_series"
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_native_settled_call_issues_source_refs_on_the_host_ledger(tmp_path: Path) -> None:
  async def run() -> None:
    session, bundle, _dispatcher = _session(tmp_path)
    native = bundle.frame_reader.open_cell("native-cell", abort_event=None, active=True)
    try:
      active = lookup_active_cell(native.token)
      assert active is not None
      entry = active.start_native(GET_FILINGS, {"cik": "AAPL"})
      active.settle_native(entry, result={
        "status": "success",
        "filings": [{"document_id": "edgar:0000320193-24-000123", "accession": "0000320193-24-000123"}],
      }, error=None)
      assert len(entry["source_refs"]) == 1
      summary = native.summary()
      assert [call["nested_call_id"] for call in summary["nested_calls"]] == ["native-cell.1"]
      assert summary["nested_calls"][0]["source_refs"] == entry["source_refs"]
      assert [source["document_id"] for source in summary["sources"]] == ["edgar:0000320193-24-000123"]
      assert "derived" not in summary
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_native_result_carried_into_a_cell_is_plain_and_the_ledger_keeps_its_sources(tmp_path: Path) -> None:
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
      cell = await _cell(dispatcher, "cell-1", f"""
        import json
        plain = json.loads({json.dumps(result)!r})
        accession = plain["filings"][0]["accession"]
        print(json.dumps([type(plain) is dict, type(plain["filings"]) is list, accession]))
      """)
      assert json.loads(cell["stdout"]) == [True, True, "0000320193-24-000123"]
      # The cell only handled a plain value: no `derived` row, no ledger of its own.
      assert "derived" not in cell and "sources" not in cell and "nested_calls" not in cell
      summary = native.summary()
      assert [source["document_id"] for source in summary["sources"]] == ["edgar:0000320193-24-000123"]
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_sliced_producer_list_is_a_plain_list(tmp_path: Path) -> None:
  async def run() -> None:
    session, _bundle, dispatcher = _session(tmp_path)
    try:
      first = await _cell(dispatcher, "cell-1", f"""
        obs = hank.call({FRED_SERIES!r}, series_id="CPIAUCSL", tail={_MONTHS})["observations"]
        assert type(obs) is list and type(obs[-12:]) is list
      """)
      result = await _cell(dispatcher, "cell-2", """
        import json
        window = obs[-12:]
        assert type(window) is list and len(window) == 12
        mean = sum(row["value"] for row in window) / len(window)
        print(json.dumps({"mean": mean, "window": window, "last": window[-1]}))
      """)
      observations = _observations()
      out = json.loads(result["stdout"])
      assert out["window"] == observations[-12:] and out["last"] == observations[-1]
      assert out["mean"] == sum(row["value"] for row in observations[-12:]) / 12
      assert "sources" not in result and "nested_calls" not in result and "derived" not in result
      assert _row(first["sources"], "CPIAUCSL")["produced_by_tool"] == "fred_get_series"
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())
