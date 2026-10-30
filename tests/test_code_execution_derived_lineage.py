"""Derived observations travel through the real sticky kernel and bridge frames."""
import asyncio
import json
from typing import Any

import pytest

from test_code_execution_bridge import (
  FRED_SERIES, GET_FILINGS, _filings_extractor, _policy, _run_cell, _setup,
  _CaptureMcp, _DISPATCH_SEMANTICS, _bind,
  cleanup_code_execution,
)

from agent_gateway import tool_dispatch_declarations
from agent_gateway.tool_dispatch_declarations import ToolDispatchSemantics
from agent_workflow_contracts.tool_registration import RegisteredToolIdentity, VersionedPolicyRef


class _SeriesMcp(_CaptureMcp):
  async def call_tool(self, name: str, tool_input: Any, *args: Any, **kwargs: Any) -> Any:
    if name != FRED_SERIES:
      return await super().call_tool(name, tool_input, *args, **kwargs)
    self.calls.append((name, dict(tool_input)))
    series_id = tool_input["series_id"]
    return {
      "status": "success", "series_id": series_id,
      "observations": [{"date": "2025-01-01", "value": 100},
                       {"date": "2026-01-01", "value": 125}],
      "sources": [{
        "source_kind": "vendor", "provider": "fred", "endpoint": "fred_get_series",
        "key_fields": {"series_id": series_id},
        "source_url": f"https://fred.stlouisfed.org/series/{series_id}",
      }],
    }, None


@pytest.fixture
def series_policy(monkeypatch: pytest.MonkeyPatch, _policy: Any) -> None:
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


def test_retained_bridge_list_publishes_selected_derived_observation() -> None:
  async def run() -> None:
    session, _bundle, mcp, dispatcher = _setup(extract_sources=_filings_extractor)
    try:
      fetched, error = await _run_cell(dispatcher, f'''
        filings = hank.call({GET_FILINGS!r}, cik="0000320193", limit=2)
        retained = filings["filings"]
        print(len(retained))
      ''')
      assert error is None, error
      assert fetched["return_code"] == 0, fetched["stderr"]
      assert fetched["stdout"] == "2\n"
      result, error = await _run_cell(dispatcher, '''
        import json
        selected = hank.input(retained[0], "form")
        print(json.dumps(selected))
        hank.derive("selected_form", selected["value"], expression="filings[0].form", inputs=[selected])
      ''')
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      selected = json.loads(result["stdout"])
      assert selected["nested_call_id"] == fetched["nested_calls"][0]["nested_call_id"]
      assert selected["source_refs"] == fetched["nested_calls"][0]["source_refs"]
      assert selected["path"] == ["filings", 0, "form"]
      row, = result["derived"]
      assert row["name"] == "selected_form"
      assert row["value"] == "10-K"
      assert row["expression"] == "filings[0].form"
      assert row["status"] == "derived"
      operand, = row["inputs"]
      assert operand["nested_call_id"] == fetched["nested_calls"][0]["nested_call_id"]
      assert operand["path"] == ["filings", 0, "form"]
      assert operand["value"] == "10-K"
      assert operand["sources"] == fetched["sources"]
      assert row["findings"] == []
      assert len(mcp.calls) == 1
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_literal_derive_publishes_unsourced_finding() -> None:
  async def run() -> None:
    session, _bundle, mcp, dispatcher = _setup()
    try:
      result, error = await _run_cell(dispatcher, '''
        hank.derive("literal", 3, expression="1 + 2", inputs=[1, 2])
      ''')
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      row, = result["derived"]
      assert row["status"] == "derived" and row["value"] == 3
      assert [(item["code"], item["input"]) for item in row["findings"]] == [
        ("input_unsourced", 0), ("input_unsourced", 1),
      ]
      assert [(item["value"], item["sources"]) for item in row["inputs"]] == [(1, []), (2, [])]
      assert mcp.calls == []
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


@pytest.mark.parametrize("split_cells", [False, True])
def test_selected_inputs_match_in_one_or_two_sticky_cells(split_cells: bool) -> None:
  async def run() -> None:
    session, _bundle, mcp, dispatcher = _setup(extract_sources=_filings_extractor)
    fetch = f'''
        filings = hank.call({GET_FILINGS!r}, cik="0000320193", limit=2)
        unused = hank.call({FRED_SERIES!r}, series_id="UNRATE")
        retained = filings["filings"]
    '''
    derive = '''
        selected = hank.input(retained, 0)
        hank.derive("filing", selected["value"]["form"],
                    expression="filings[0].form", inputs=[selected])
    '''
    try:
      if split_cells:
        fetched, error = await _run_cell(dispatcher, fetch)
        assert error is None, error
        assert fetched["return_code"] == 0, fetched["stderr"]
        result, error = await _run_cell(dispatcher, derive)
      else:
        result, error = await _run_cell(dispatcher, fetch + derive)
        fetched = result
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      row, = result["derived"]
      assert (row["name"], row["value"], row["expression"], row["status"], row["findings"]) == (
        "filing", "10-K", "filings[0].form", "derived", [],
      )
      assert row["inputs"] == [{
        "nested_call_id": fetched["nested_calls"][0]["nested_call_id"],
        "path": ["filings", 0],
        "value": {"document_id": "edgar:0000320193-24-000123",
                  "accession": "0000320193-24-000123",
                  "accession_number": "0000320193-24-000123", "form": "10-K"},
        "sources": fetched["sources"],
      }]
      assert row["inputs"][0]["nested_call_id"] != fetched["nested_calls"][1]["nested_call_id"]
      assert all(source.get("produced_by_tool") != "fred_get_series"
                 for source in row["inputs"][0]["sources"])
      assert json.loads(json.dumps(row)) == row
      assert len(mcp.calls) == 2
      assert all(source.get("status") != "derived" for source in fetched["sources"])
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_unissued_input_reference_is_a_finding_not_a_refusal() -> None:
  async def run() -> None:
    session, _bundle, mcp, dispatcher = _setup(extract_sources=_filings_extractor)
    try:
      result, error = await _run_cell(dispatcher, f'''
        filings = hank.call({GET_FILINGS!r}, cik="0000320193")
        selected = hank.input(filings["filings"], 0)
        literal = hank.input({{"value": 99}}, "value")
        hank.derive("mixed", 100, expression="selected + literal + 1",
                    inputs=[selected, literal, 1])
        forged = hank.input(filings, "ticker")
        forged["source_refs"] = ["src_never_issued"]
        hank.derive("unissued", 2, expression="x", inputs=[forged])
      ''')
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      mixed, unissued = result["derived"]
      assert mixed["status"] == unissued["status"] == "derived"
      assert [(item["code"], item["input"]) for item in mixed["findings"]] == [
        ("input_unsourced", 1), ("input_unsourced", 2),
      ]
      assert mixed["inputs"][0]["sources"] == result["sources"]
      assert [(item["value"], item["sources"]) for item in mixed["inputs"][1:]] == [(99, []), (1, [])]
      assert unissued["inputs"][0]["sources"] == []
      assert [item["code"] for item in unissued["findings"]] == ["input_unsourced"]
      assert len(result["sources"]) == len(result["nested_calls"][0]["source_refs"])
      assert len(mcp.calls) == 1
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


@pytest.mark.parametrize("split_cells", [False, True])
def test_growth_selects_exact_observations_and_excludes_unused_series(
  series_policy: None, split_cells: bool,
) -> None:
  async def run() -> None:
    session, bundle, _mcp, _dispatcher = _setup()
    mcp = _SeriesMcp(frame_reader=bundle.frame_reader)
    dispatcher = _bind(bundle, mcp)
    fetch = f'''
        cpi = hank.call({FRED_SERIES!r}, series_id="CPIAUCSL")
        unused = hank.call({FRED_SERIES!r}, series_id="UNRATE")
        observations = cpi["observations"]
        print(observations[-1]["value"] / observations[0]["value"] - 1)
    '''
    derive = '''
        latest = hank.input(observations, -1)
        prior = hank.input(observations, 0)
        hank.derive("growth", latest["value"]["value"] / prior["value"]["value"] - 1,
                    expression="observations[-1].value / observations[0].value - 1",
                    inputs=[latest, prior])
    '''
    try:
      if split_cells:
        fetched, error = await _run_cell(dispatcher, fetch)
        assert error is None, error
        assert fetched["return_code"] == 0, fetched["stderr"]
        result, error = await _run_cell(dispatcher, derive)
      else:
        result, error = await _run_cell(dispatcher, fetch + derive)
        fetched = result
      assert error is None, error
      assert fetched["stdout"] == "0.25\n"
      assert result["return_code"] == 0, result["stderr"]
      cpi_source, = [source for source in fetched["sources"]
                     if source.get("key_fields") == {"series_id": "CPIAUCSL"}]
      unused_source, = [source for source in fetched["sources"]
                        if source.get("key_fields") == {"series_id": "UNRATE"}]
      first, second = fetched["nested_calls"]
      assert first["source_refs"] and second["source_refs"]
      assert first["source_refs"] != second["source_refs"]
      row, = result["derived"]
      assert row["value"] == 0.25 and row["status"] == "derived" and row["findings"] == []
      assert row["expression"] == "observations[-1].value / observations[0].value - 1"
      assert row["inputs"] == [
        {"nested_call_id": first["nested_call_id"], "path": ["observations", 1],
         "value": {"date": "2026-01-01", "value": 125}, "sources": [cpi_source]},
        {"nested_call_id": first["nested_call_id"], "path": ["observations", 0],
         "value": {"date": "2025-01-01", "value": 100}, "sources": [cpi_source]},
      ]
      assert unused_source not in [source for operand in row["inputs"] for source in operand["sources"]]
      assert len(mcp.calls) == 2
      assert len(fetched["sources"]) == 2
      assert "derived" not in fetched or fetched is result
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_carriers_preserve_child_identity_mutation_and_live_dict_views() -> None:
  async def run() -> None:
    session, _bundle, mcp, dispatcher = _setup(extract_sources=_filings_extractor)
    try:
      first, error = await _run_cell(dispatcher, f'''
        filings = hank.call({GET_FILINGS!r}, cik="0000320193")
        retained = filings["filings"]
        first_row = retained[0]
        items_view = filings.items()
        values_view = filings.values()
        assert retained is filings["filings"]
        assert first_row is retained[0]
        assert dict(items_view)["filings"] is retained
        assert next(value for value in values_view if isinstance(value, list)) is retained
        first_row["note"] = "ordinary mutation"
        assert filings["filings"][0]["note"] == "ordinary mutation"
        filings["added"] = "visible through views"
        assert ("added", "visible through views") in items_view
        assert "visible through views" in values_view
        assert len(items_view) == len(values_view) == len(filings)
        assert next(reversed(items_view)) == ("added", "visible through views")
        assert next(reversed(values_view)) == "visible through views"
      ''')
      assert error is None, error
      assert first["return_code"] == 0, first["stderr"]
      second, error = await _run_cell(dispatcher, '''
        assert retained is filings["filings"]
        assert first_row is retained[0]
        assert first_row["note"] == "ordinary mutation"
        del filings["added"]
        assert "visible through views" not in values_view
        assert all(key != "added" for key, value in items_view)
      ''')
      assert error is None, error
      assert second["return_code"] == 0, second["stderr"]
      assert len(mcp.calls) == 1
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_retained_slice_normalizes_bounds_and_selected_index() -> None:
  async def run() -> None:
    session, _bundle, mcp, dispatcher = _setup(extract_sources=_filings_extractor)
    try:
      first, error = await _run_cell(dispatcher, f'''
        filings = hank.call({GET_FILINGS!r}, cik="0000320193")
        retained = filings["filings"]
        window = retained[-2:]
        assert isinstance(window, list)
        assert window == retained
        assert window[0] is retained[0]
      ''')
      assert error is None, error
      assert first["return_code"] == 0, first["stderr"]
      result, error = await _run_cell(dispatcher, '''
        selected = hank.input(window, -1)
        hank.derive("last_form", selected["value"]["form"],
                    expression="filings[-2:][-1].form", inputs=[selected])
      ''')
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      row, = result["derived"]
      assert row["findings"] == []
      operand, = row["inputs"]
      assert operand["nested_call_id"] == first["nested_calls"][0]["nested_call_id"]
      assert operand["path"] == ["filings", 1]
      assert operand["value"]["form"] == row["value"] == "10-Q"
      assert operand["sources"] == first["sources"]
      assert len(mcp.calls) == 1
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


@pytest.mark.parametrize(("mutation", "selection", "expected"), [
  ('filings["added"] = 99', 'hank.input(filings, "added")', 99),
  ('filings["added"] = {"value": 99}', 'hank.input(filings, "added")', {"value": 99}),
  ('filings["ticker"] = "MSFT"', 'hank.input(filings, "ticker")', "MSFT"),
  ('rows.append(99)', 'hank.input(rows, -1)', 99),
  ('rows.append({"value": 99})', 'hank.input(rows, -1)', {"value": 99}),
  ('rows.insert(0, 99)', 'hank.input(rows, 0)', 99),
  ('rows.insert(0, {"value": 99})', 'hank.input(rows, 0)', {"value": 99}),
])
def test_mutated_selected_values_publish_unsourced_findings(
  mutation: str, selection: str, expected: Any,
) -> None:
  async def run() -> None:
    session, _bundle, mcp, dispatcher = _setup(extract_sources=_filings_extractor)
    try:
      fetched, error = await _run_cell(dispatcher, f'''
        filings = hank.call({GET_FILINGS!r}, cik="0000320193")
        rows = filings["filings"]
        {mutation}
      ''')
      assert error is None, error
      assert fetched["return_code"] == 0, fetched["stderr"]
      assert fetched["sources"]
      result, error = await _run_cell(dispatcher, f'''
        selected = {selection}
        hank.derive("changed", selected["value"], expression="selected", inputs=[selected])
      ''')
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      row, = result["derived"]
      assert row["status"] == "derived"
      assert row["value"] == expected
      operand, = row["inputs"]
      assert operand["value"] == expected
      assert operand["sources"] == []
      assert [(item["code"], item["input"]) for item in row["findings"]] == [
        ("input_unsourced", 0),
      ]
      assert len(mcp.calls) == 1
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


@pytest.mark.parametrize("mutation", [
  'rows.insert(0, {"form": "invented"})',
  'rows.reverse()',
  'rows[:] = rows[::-1]',
  'rows.sort(key=lambda row: row["form"], reverse=True)',
])
def test_original_container_moves_keep_their_issued_origin(mutation: str) -> None:
  async def run() -> None:
    session, _bundle, mcp, dispatcher = _setup(extract_sources=_filings_extractor)
    try:
      fetched, error = await _run_cell(dispatcher, f'''
        filings = hank.call({GET_FILINGS!r}, cik="0000320193")
        rows = filings["filings"]
        original = rows[0]
        {mutation}
      ''')
      assert error is None, error
      assert fetched["return_code"] == 0, fetched["stderr"]
      result, error = await _run_cell(dispatcher, '''
        index = next(i for i, row in enumerate(rows) if row is original)
        selected = hank.input(rows, index - len(rows))
        hank.derive("original", selected["value"]["form"],
                    expression="original.form", inputs=[selected])
      ''')
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      row, = result["derived"]
      operand, = row["inputs"]
      assert row["value"] == "10-K"
      assert row["findings"] == []
      assert operand["sources"] == fetched["sources"]
      assert operand["nested_call_id"] == fetched["nested_calls"][0]["nested_call_id"]
      assert operand["path"] == ["filings", 0]
      assert len(mcp.calls) == 1
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


@pytest.mark.parametrize("selection", [
  'hank.input(filings, "filings")',
  'hank.input(rows, 0)',
  'hank.input(rows[0], "form")',
])
def test_changed_nested_values_do_not_keep_original_container_sources(selection: str) -> None:
  async def run() -> None:
    session, _bundle, mcp, dispatcher = _setup(extract_sources=_filings_extractor)
    try:
      result, error = await _run_cell(dispatcher, f'''
        filings = hank.call({GET_FILINGS!r}, cik="0000320193")
        rows = filings["filings"]
        rows[0]["form"] = "invented"
        selected = {selection}
        hank.derive("changed_nested", selected["value"], expression="selected", inputs=[selected])
      ''')
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      row, = result["derived"]
      assert row["status"] == "derived"
      assert row["inputs"][0]["sources"] == []
      assert [(item["code"], item["input"]) for item in row["findings"]] == [
        ("input_unsourced", 0),
      ]
      assert len(mcp.calls) == 1
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_equal_scalar_list_slots_travel_with_origin_through_mutation_and_slicing() -> None:
  class ScalarSlotsMcp(_CaptureMcp):
    async def call_tool(self, name: str, tool_input: Any, *args: Any, **kwargs: Any) -> Any:
      result, error = await super().call_tool(name, tool_input, *args, **kwargs)
      assert error is None, error
      assert isinstance(result, dict)
      return {**result, "slots": [7, 7]}, error

  async def run() -> None:
    session, bundle, _mcp, _dispatcher = _setup(extract_sources=_filings_extractor)
    mcp = ScalarSlotsMcp(frame_reader=bundle.frame_reader)
    dispatcher = _bind(bundle, mcp, extract_sources=_filings_extractor)
    try:
      fetched, error = await _run_cell(dispatcher, f'''
        filings = hank.call({GET_FILINGS!r}, cik="0000320193")
        slots = filings["slots"]
        slots.insert(0, 7)
        slots.append(7)
        slots.reverse()
        window = slots[:]
      ''')
      assert error is None, error
      assert fetched["return_code"] == 0, fetched["stderr"]
      result, error = await _run_cell(dispatcher, '''
        for index in range(len(window)):
            selected = hank.input(window, index - len(window))
            hank.derive("slot_" + str(index), selected["value"],
                        expression="window[index]", inputs=[selected])
      ''')
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      appended, second, first, inserted = result["derived"]
      for row in (appended, inserted):
        assert row["value"] == 7
        assert row["inputs"][0]["sources"] == []
        assert [(item["code"], item["input"]) for item in row["findings"]] == [
          ("input_unsourced", 0),
        ]
      for row, original_index in ((first, 0), (second, 1)):
        assert row["value"] == 7
        assert row["findings"] == []
        operand, = row["inputs"]
        assert operand["path"] == ["slots", original_index]
        assert operand["sources"] == fetched["sources"]
        assert operand["nested_call_id"] == fetched["nested_calls"][0]["nested_call_id"]
      assert len(mcp.calls) == 1
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


@pytest.mark.parametrize(("selection", "mutation", "expected"), [
  ('hank.input(filings["filings"], 0)', 'selected["value"]["form"] = "changed"',
   "changed"),
  ('hank.input(filings, "ticker")', 'selected["value"] = "MSFT"', "MSFT"),
])
def test_changed_selected_input_envelope_publishes_unsourced_finding(
  selection: str, mutation: str, expected: str,
) -> None:
  async def run() -> None:
    session, _bundle, mcp, dispatcher = _setup(extract_sources=_filings_extractor)
    try:
      fetched, error = await _run_cell(dispatcher, f'''
        filings = hank.call({GET_FILINGS!r}, cik="0000320193")
        selected = {selection}
      ''')
      assert error is None, error
      assert fetched["return_code"] == 0, fetched["stderr"]
      result, error = await _run_cell(dispatcher, f'''
        {mutation}
        value = selected["value"]
        if isinstance(value, dict):
            value = value["form"]
        hank.derive("changed_selection", value, expression="selected", inputs=[selected])
      ''')
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      row, = result["derived"]
      assert row["status"] == "derived"
      assert row["value"] == expected
      assert row["inputs"][0]["sources"] == []
      assert [(item["code"], item["input"]) for item in row["findings"]] == [
        ("input_unsourced", 0),
      ]
      assert len(mcp.calls) == 1
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


@pytest.mark.parametrize(("mutation", "normalized"), [
  ("", [0, 2, 1]),
  ("rows.reverse()", [1, -1, -1]),
])
def test_original_whole_slice_retains_normalized_slice_lineage(
  mutation: str, normalized: list[int],
) -> None:
  async def run() -> None:
    session, _bundle, mcp, dispatcher = _setup(extract_sources=_filings_extractor)
    try:
      fetched, error = await _run_cell(dispatcher, f'''
        filings = hank.call({GET_FILINGS!r}, cik="0000320193")
        rows = filings["filings"]
        {mutation}
        window = rows[-2:]
      ''')
      assert error is None, error
      assert fetched["return_code"] == 0, fetched["stderr"]
      result, error = await _run_cell(dispatcher, '''
        hank.derive("original_window", len(window),
                    expression="len(window)", inputs=[window])
      ''')
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      row, = result["derived"]
      assert row["value"] == 2
      assert row["findings"] == []
      operand, = row["inputs"]
      assert operand["path"] == ["filings", {"slice": normalized}]
      assert operand["sources"] == fetched["sources"]
      assert operand["nested_call_id"] == fetched["nested_calls"][0]["nested_call_id"]
      original_forms = ["10-K", "10-Q"]
      selected_indices = range(normalized[0], normalized[1], normalized[2])
      assert [value["form"] for value in operand["value"]] == [
        original_forms[index] for index in selected_indices
      ]
      assert len(mcp.calls) == 1
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


@pytest.mark.parametrize("carrier", ["filings", 'filings["filings"]', 'filings["filings"][0]'])
def test_pickle_roundtrip_retains_nested_carrier_origins(carrier: str) -> None:
  async def run() -> None:
    session, _bundle, mcp, dispatcher = _setup(extract_sources=_filings_extractor)
    try:
      fetched, error = await _run_cell(dispatcher, f'''
        import pickle
        filings = hank.call({GET_FILINGS!r}, cik="0000320193")
        filings["filings"][0]["invented"] = 99
        original = {carrier}
        restored = pickle.loads(pickle.dumps(original))
        assert restored == original and restored is not original
        if isinstance(restored, list):
            restored_row = restored[0]
            assert restored_row is not original[0]
        elif "filings" in restored:
            restored_row = restored["filings"][0]
            assert restored["filings"] is not original["filings"]
            assert restored_row is not original["filings"][0]
        else:
            restored_row = restored
        selected = hank.input(restored_row, "form")
        hank.derive("restored", selected["value"], expression="restored.form", inputs=[selected])
        selected = hank.input(restored_row, "invented")
        hank.derive("before_pickle", selected["value"], expression="invented", inputs=[selected])
      ''')
      assert error is None, error
      assert fetched["return_code"] == 0, fetched["stderr"]
      result, error = await _run_cell(dispatcher, '''
        restored_row["added"] = "invented"
        selected = hank.input(restored_row, "added")
        hank.derive("added", selected["value"], expression="restored.added", inputs=[selected])
        original_row = original[0] if isinstance(original, list) else original.get("filings", [original])[0]
        assert "added" not in original_row
        selected = hank.input(original_row, "form")
        hank.derive("original", selected["value"], expression="original.form", inputs=[selected])
      ''')
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      for row in (fetched["derived"][0], result["derived"][1]):
        assert row["value"] == "10-K"
        assert row["findings"] == []
        operand, = row["inputs"]
        assert operand["path"] == ["filings", 0, "form"]
        assert operand["nested_call_id"] == fetched["nested_calls"][0]["nested_call_id"]
        assert operand["sources"] == fetched["sources"]
      added = result["derived"][0]
      assert added["inputs"][0]["sources"] == []
      assert [(item["code"], item["input"]) for item in added["findings"]] == [
        ("input_unsourced", 0),
      ]
      before_pickle = fetched["derived"][1]
      assert before_pickle["value"] == 99
      assert before_pickle["inputs"][0]["sources"] == []
      assert [(item["code"], item["input"]) for item in before_pickle["findings"]] == [
        ("input_unsourced", 0),
      ]
      assert len(mcp.calls) == 1
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


@pytest.mark.parametrize("protocol", ["0", "pickle.HIGHEST_PROTOCOL"])
def test_scalar_list_pickle_preserves_equal_original_slots_only(protocol: str) -> None:
  class ScalarSlotsMcp(_CaptureMcp):
    async def call_tool(self, name: str, tool_input: Any, *args: Any, **kwargs: Any) -> Any:
      result, error = await super().call_tool(name, tool_input, *args, **kwargs)
      assert error is None, error
      assert isinstance(result, dict)
      return {**result, "slots": [7, 7]}, error

  async def run() -> None:
    session, bundle, _mcp, _dispatcher = _setup(extract_sources=_filings_extractor)
    mcp = ScalarSlotsMcp(frame_reader=bundle.frame_reader)
    dispatcher = _bind(bundle, mcp, extract_sources=_filings_extractor)
    try:
      fetched, error = await _run_cell(dispatcher, f'''
        import pickle
        filings = hank.call({GET_FILINGS!r}, cik="0000320193")
        slots = filings["slots"]
        slots.insert(0, 7)
        restored = pickle.loads(pickle.dumps(slots, protocol={protocol}))
        assert restored == slots and restored is not slots
      ''')
      assert error is None, error
      assert fetched["return_code"] == 0, fetched["stderr"]
      result, error = await _run_cell(dispatcher, '''
        restored.append(7)
        restored.reverse()
        assert slots == [7, 7, 7]
        for index in range(len(restored)):
            selected = hank.input(restored, index)
            hank.derive("slot_" + str(index), selected["value"],
                        expression="restored[index]", inputs=[selected])
      ''')
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      appended, second, first, inserted = result["derived"]
      for row, original_index in ((first, 0), (second, 1)):
        assert row["value"] == 7
        assert row["findings"] == []
        operand, = row["inputs"]
        assert operand["path"] == ["slots", original_index]
        assert operand["nested_call_id"] == fetched["nested_calls"][0]["nested_call_id"]
        assert operand["sources"] == fetched["sources"]
      for row in (appended, inserted):
        assert row["value"] == 7
        assert row["inputs"][0]["sources"] == []
        assert [(item["code"], item["input"]) for item in row["findings"]] == [
          ("input_unsourced", 0),
        ]
      assert len(mcp.calls) == 1
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_deepcopy_nested_carriers_and_selection_preserves_origins_and_memo() -> None:
  class ScalarSlotsMcp(_CaptureMcp):
    async def call_tool(self, name: str, tool_input: Any, *args: Any, **kwargs: Any) -> Any:
      result, error = await super().call_tool(name, tool_input, *args, **kwargs)
      assert error is None, error
      assert isinstance(result, dict)
      return {**result, "slots": [7, 7]}, error

  async def run() -> None:
    session, bundle, _mcp, _dispatcher = _setup(extract_sources=_filings_extractor)
    mcp = ScalarSlotsMcp(frame_reader=bundle.frame_reader)
    dispatcher = _bind(bundle, mcp, extract_sources=_filings_extractor)
    try:
      fetched, error = await _run_cell(dispatcher, f'''
        import copy
        filings = hank.call({GET_FILINGS!r}, cik="0000320193")
        selected = hank.input(filings["filings"], 0)
        graph = {{"filings": filings, "rows": filings["filings"], "selected": selected}}
        graph["cycle"] = graph
        copied = copy.deepcopy(graph)
        assert copied is not graph and copied["cycle"] is copied
        assert copied["rows"] is copied["filings"]["filings"]
        assert copied["rows"] is not filings["filings"]
        assert copied["rows"][0] is not filings["filings"][0]
        assert copied["selected"]["value"] is copied["rows"][0]
      ''')
      assert error is None, error
      assert fetched["return_code"] == 0, fetched["stderr"]
      result, error = await _run_cell(dispatcher, '''
        selections = [
            ("nested", hank.input(copied["rows"][0], "form")),
            ("first", hank.input(copied["filings"]["slots"], 0)),
            ("second", hank.input(copied["filings"]["slots"], 1)),
            ("selection", copied["selected"]),
        ]
        for name, operand in selections:
            hank.derive(name, operand["value"], expression=name, inputs=[operand])
        copied["rows"][0]["form"] = "changed"
        copied["filings"]["slots"].append(7)
        assert filings["filings"][0]["form"] == "10-K"
        assert filings["slots"] == [7, 7]
        changed = hank.input(copied["rows"][0], "form")
        added = hank.input(copied["filings"]["slots"], 2)
        for name, operand in (("changed", changed), ("added", added),
                              ("changed_selection", copied["selected"])):
            hank.derive(name, operand["value"], expression=name, inputs=[operand])
      ''')
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      rows = {row["name"]: row for row in result["derived"]}
      for name, path in (
        ("nested", ["filings", 0, "form"]),
        ("first", ["slots", 0]), ("second", ["slots", 1]),
        ("selection", ["filings", 0]),
      ):
        row = rows[name]
        assert row["findings"] == []
        operand, = row["inputs"]
        assert operand["path"] == path
        assert operand["nested_call_id"] == fetched["nested_calls"][0]["nested_call_id"]
        assert operand["sources"] == fetched["sources"]
      assert rows["first"]["value"] == rows["second"]["value"] == 7
      for name in ("changed", "added", "changed_selection"):
        row = rows[name]
        assert row["inputs"][0]["sources"] == []
        assert [(item["code"], item["input"]) for item in row["findings"]] == [
          ("input_unsourced", 0),
        ]
      assert len(mcp.calls) == 1
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_shallow_copy_owns_dict_and_equal_scalar_list_slot_metadata() -> None:
  class ScalarSlotsMcp(_CaptureMcp):
    async def call_tool(self, name: str, tool_input: Any, *args: Any, **kwargs: Any) -> Any:
      result, error = await super().call_tool(name, tool_input, *args, **kwargs)
      assert error is None, error
      assert isinstance(result, dict)
      return {**result, "slots": [7, 7]}, error

  async def run() -> None:
    session, bundle, _mcp, _dispatcher = _setup(extract_sources=_filings_extractor)
    mcp = ScalarSlotsMcp(frame_reader=bundle.frame_reader)
    dispatcher = _bind(bundle, mcp, extract_sources=_filings_extractor)
    try:
      fetched, error = await _run_cell(dispatcher, f'''
        import copy
        filings = hank.call({GET_FILINGS!r}, cik="0000320193")
        copied = copy.copy(filings)
        assert copied is not filings and copied == filings
        assert copied["filings"] is filings["filings"]
        keys = copied.keys()
        copied["added"] = 7
        assert "added" in keys and "added" not in filings
        copied["ticker"] = "MSFT"
        slots = filings["slots"]
        copied_slots = copy.copy(slots)
        assert copied_slots is not slots and copied_slots == slots
        copied_slots.insert(0, 7)
        copied_slots.append(7)
        copied_slots.reverse()
        assert slots == [7, 7]
        copied_rows = copy.copy(filings["filings"])
        assert copied_rows[0] is filings["filings"][0]
      ''')
      assert error is None, error
      assert fetched["return_code"] == 0, fetched["stderr"]
      result, error = await _run_cell(dispatcher, '''
        selections = [
            ("original_ticker", hank.input(filings, "ticker")),
            ("changed_ticker", hank.input(copied, "ticker")),
            ("added_key", hank.input(copied, "added")),
            ("original_0", hank.input(slots, 0)),
            ("original_1", hank.input(slots, 1)),
        ]
        selections.extend(("copy_" + str(i), hank.input(copied_slots, i)) for i in range(4))
        for name, selected in selections:
            hank.derive(name, selected["value"], expression=name, inputs=[selected])
      ''')
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      rows = {row["name"]: row for row in result["derived"]}
      for name, path in (
        ("original_ticker", ["ticker"]),
        ("original_0", ["slots", 0]), ("original_1", ["slots", 1]),
        ("copy_1", ["slots", 1]), ("copy_2", ["slots", 0]),
      ):
        row = rows[name]
        assert row["findings"] == []
        operand, = row["inputs"]
        assert operand["path"] == path
        assert operand["nested_call_id"] == fetched["nested_calls"][0]["nested_call_id"]
        assert operand["sources"] == fetched["sources"]
      for name in ("changed_ticker", "added_key", "copy_0", "copy_3"):
        row = rows[name]
        assert row["inputs"][0]["sources"] == []
        assert [(item["code"], item["input"]) for item in row["findings"]] == [
          ("input_unsourced", 0),
        ]
      assert rows["changed_ticker"]["value"] == "MSFT"
      assert all(rows[name]["value"] == 7 for name in (
        "original_0", "original_1", "copy_0", "copy_1", "copy_2", "copy_3",
      ))
      assert len(mcp.calls) == 1
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


@pytest.mark.parametrize("operation", ["pickle.loads(pickle.dumps(selected))", "copy.copy(selected)"])
def test_selected_input_lifecycle_preserves_origin_and_independent_value(operation: str) -> None:
  async def run() -> None:
    session, _bundle, mcp, dispatcher = _setup(extract_sources=_filings_extractor)
    try:
      fetched, error = await _run_cell(dispatcher, f'''
        import copy
        import pickle
        filings = hank.call({GET_FILINGS!r}, cik="0000320193")
        selected = hank.input(filings, "ticker")
        restored = {operation}
        assert restored == selected and restored is not selected
        hank.derive("restored", restored["value"], expression="restored", inputs=[restored])
      ''')
      assert error is None, error
      assert fetched["return_code"] == 0, fetched["stderr"]
      result, error = await _run_cell(dispatcher, '''
        restored["value"] = "MSFT"
        hank.derive("changed", restored["value"], expression="restored", inputs=[restored])
        hank.derive("original", selected["value"], expression="selected", inputs=[selected])
      ''')
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      for row in (fetched["derived"][0], result["derived"][1]):
        assert row["findings"] == []
        operand, = row["inputs"]
        assert operand["path"] == ["ticker"]
        assert operand["sources"] == fetched["sources"]
        assert operand["nested_call_id"] == fetched["nested_calls"][0]["nested_call_id"]
      changed = result["derived"][0]
      assert changed["value"] == "MSFT"
      assert changed["inputs"][0]["sources"] == []
      assert [(item["code"], item["input"]) for item in changed["findings"]] == [
        ("input_unsourced", 0),
      ]
      assert len(mcp.calls) == 1
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())
