import copy
import json
from types import SimpleNamespace

from hank_model_transport.policy_imports import configure_server_policy, load_server_policy_module
from agent_gateway.secret_boundary import SecretBoundary, sanitize_boundary_value
from agent_gateway.tool_result_compaction import (
  compact_model_tool_result_entry,
  truncate_model_tool_result_content,
)
from agent_gateway.tool_result_spill import SpillCapabilities, SpillSink

MAX_CHARS = 60_000
DRIVER_COUNT = 36


def _driver_keys() -> list[str]:
  return [f"driver_{index:02d}_segment_revenue_growth" for index in range(DRIVER_COUNT)]


def _forecast_context(*, anchor_chars: int) -> dict:
  keys = _driver_keys()
  return {
    "status": "success",
    "view": "forecast_assumptions_context",
    "required_drivers": [
      {
        "driver_key": key,
        "label": f"Driver {index}",
        "unit": "percent",
        "model_scale": "percent_points",
        "value_semantics": "growth_rate",
        "latest_value": 0.04 + index / 1000,
        "latest_period_id": "FY2025",
        "missing_reason": None,
        "description": "d" * 900,
        "anchors": [{"period_id": f"FY{2020 + n}", "note": "a" * 400} for n in range(6)],
      }
      for index, key in enumerate(keys)
    ],
    "semantic_keys": {
      "drivers": keys,
      "periods": [f"FY{year}" for year in range(2018, 2031)],
      "gaps": [],
    },
    "latest_actual_anchors": {
      "rows": [{"label": f"row-{index}", "text": "x" * 1_500} for index in range(anchor_chars // 1_500)],
    },
  }


def test_forecast_context_navigation_survives_a_forced_truncation() -> None:
  context = _forecast_context(anchor_chars=150_000)
  content = json.dumps(context)
  assert len(content) > 2 * MAX_CHARS

  truncated, was_truncated = truncate_model_tool_result_content(
    content,
    tool_name="get_forecast_assumptions_context",
    max_chars=MAX_CHARS,
  )

  assert was_truncated is True
  assert len(truncated) <= MAX_CHARS
  projection = json.loads(truncated)["content_projection"]
  keys = _driver_keys()
  # The anchors are bulk and are cut ...
  assert "_elided_items" in projection["latest_actual_anchors"]["rows"][-1]
  # ... but every issued key survives, in the index and in its descriptor row.
  assert projection["semantic_keys"]["drivers"] == keys
  assert [row["driver_key"] for row in projection["required_drivers"]] == keys
  assert "_elided_items" not in json.dumps(projection["required_drivers"])
  assert "_elided_items" not in json.dumps(projection["semantic_keys"])
  for row in projection["required_drivers"]:
    assert row["unit"] == "percent"
    assert row["latest_period_id"] == "FY2025"


def test_navigation_rows_mark_the_bulk_they_drop() -> None:
  content = json.dumps(_forecast_context(anchor_chars=150_000))

  truncated, _ = truncate_model_tool_result_content(
    content,
    tool_name="get_forecast_assumptions_context",
    max_chars=MAX_CHARS,
  )

  rows = json.loads(truncated)["content_projection"]["required_drivers"]
  assert all(row.get("_elided_keys") == 1 for row in rows)
  assert all("anchors" not in row for row in rows)


def test_navigation_keys_survive_a_wide_dict() -> None:
  wide: dict[str, object] = {f"field_{index:03d}": "v" * 20 for index in range(600)}
  wide["valid_keys"] = ["alpha", "beta", "gamma"]
  content = json.dumps({"payload": wide, "filler": "z" * 80_000})

  truncated, was_truncated = truncate_model_tool_result_content(
    content,
    tool_name="lookup",
    max_chars=20_000,
  )

  assert was_truncated is True
  payload = json.loads(truncated)["content_projection"]["payload"]
  assert payload["valid_keys"] == ["alpha", "beta", "gamma"]


def test_navigation_keys_come_from_the_policy_hook() -> None:
  previous = load_server_policy_module()
  configure_server_policy(SimpleNamespace(
    preserve_tool_result_navigation_key=lambda key: key == "ledger_index",
  ))
  try:
    ledger = [f"entry-{index:03d}" for index in range(300)]
    content = json.dumps({"ledger_index": ledger, "other": ledger, "filler": "z" * 80_000})

    truncated, was_truncated = truncate_model_tool_result_content(
      content,
      tool_name="lookup",
      max_chars=20_000,
    )
  finally:
    configure_server_policy(previous)

  assert was_truncated is True
  projection = json.loads(truncated)["content_projection"]
  assert projection["ledger_index"] == ledger
  assert len(projection["other"]) < len(ledger)


def _with_navigation_key(key: str, run):
  previous = load_server_policy_module()
  configure_server_policy(SimpleNamespace(
    preserve_tool_result_navigation_key=lambda candidate: candidate == key,
  ))
  try:
    return run()
  finally:
    configure_server_policy(previous)


def test_oversized_navigation_container_keeps_envelope_instead_of_omission_stub() -> None:
  rows = [
    {"observation_key": f"obs_{index:04d}", "label": "l" * 100, "unit": "usd"}
    for index in range(2_000)
  ]
  content = json.dumps({
    "status": "ok",
    "outline": "observations by period",
    "next_call": {"tool": "read_observations", "cursor": "obs_0000"},
    "observation_index": rows,
  })
  assert len(content) > 4 * MAX_CHARS

  truncated, was_truncated = _with_navigation_key(
    "observation_index",
    lambda: truncate_model_tool_result_content(content, tool_name="lookup", max_chars=MAX_CHARS),
  )

  assert was_truncated is True
  assert len(truncated) <= MAX_CHARS
  projection = json.loads(truncated)["content_projection"]
  assert projection["status"] == "ok"
  assert projection["outline"] == "observations by period"
  assert projection["next_call"] == {"tool": "read_observations", "cursor": "obs_0000"}
  index = projection["observation_index"]
  assert index[0]["observation_key"] == "obs_0000"
  assert index[-1]["_elided_items"] == 2_000 - (len(index) - 1)


def test_navigation_rows_keep_every_item_when_they_fit_at_bare_width() -> None:
  rows = [{"key": f"row_{index:04d}", "note": "n" * 150} for index in range(500)]
  content = json.dumps({"valid_keys": rows, "filler": "z" * 80_000})

  truncated, was_truncated = truncate_model_tool_result_content(
    content,
    tool_name="lookup",
    max_chars=MAX_CHARS,
  )

  assert was_truncated is True
  assert len(truncated) <= MAX_CHARS
  kept = json.loads(truncated)["content_projection"]["valid_keys"]
  assert [row["key"] for row in kept] == [row["key"] for row in rows]
  assert "_elided_items" not in json.dumps(kept)
  assert all(row["note"].startswith("n" * 40) and len(row["note"]) < 150 for row in kept)


def test_oversized_navigation_list_keeps_an_ordered_prefix_and_counts_the_rest() -> None:
  keys = [f"key_{index:04d}_segment_revenue" for index in range(5_000)]
  content = json.dumps({"status": "ok", "valid_keys": keys})
  assert len(content) > 2 * MAX_CHARS

  truncated, was_truncated = truncate_model_tool_result_content(
    content,
    tool_name="lookup",
    max_chars=MAX_CHARS,
  )

  assert was_truncated is True
  assert len(truncated) <= MAX_CHARS
  projection = json.loads(truncated)["content_projection"]
  assert projection["status"] == "ok"
  kept = projection["valid_keys"]
  prefix, marker = kept[:-1], kept[-1]
  assert prefix == keys[: len(prefix)]
  assert 0 < len(prefix) < len(keys)
  assert marker == {"_elided_items": len(keys) - len(prefix)}


def test_navigation_rows_stay_whole_when_they_fit_at_full_width() -> None:
  rows = [{"key": f"row_{index:02d}", "note": "n" * 150} for index in range(60)]
  content = json.dumps({"valid_keys": rows, "filler": "z" * 80_000})

  truncated, _ = truncate_model_tool_result_content(
    content,
    tool_name="lookup",
    max_chars=MAX_CHARS,
  )

  kept = json.loads(truncated)["content_projection"]["valid_keys"]
  assert kept == rows


class _NullLogger:
  def info(self, *args, **kwargs) -> None:
    return None


def test_all_fms_persist_doors_project_to_delta_receipt_without_bulk_readback(tmp_path) -> None:
  doors = {
    "fms_persist_historical_foundation": "historical_foundation_stage_receipt",
    "fms_persist_forecast_assumption_set": "forecast_assumption_set_stage_receipt",
    "fms_persist_valuation_policy": "valuation_policy_stage_receipt",
    "fms_persist_scenario_set": "scenario_set_stage_receipt",
  }
  for tool_name, receipt_key in doors.items():
    content = json.dumps(
      {
        "status": "staged",
        "gate_code": "PROCEED",
        "artifact_ref": f"artifacts/PCTY/{tool_name}.json",
        "proposal_id": "proposal-1",
        "verdict_echo": {
          "verdict": "ACCEPTED_CONSTRUCTED",
          "confidence": "HIGH",
          "revision": "rev-1",
          "validation": {"rows": ["v" * 1_000] * 40},
        },
        "readback": {
          "typed_outputs": {
            receipt_key: {"status": "staged"},
            "bulk_body": {"rows": ["b" * 1_000] * 40},
          },
        },
      }
    )
    assert len(content) > 50_000
    result_entry = {"type": "tool_result", "tool_use_id": f"tool-{tool_name}", "content": content}

    live_entry, durable_entry = compact_model_tool_result_entry(
      result_entry,
      tool_name=tool_name,
      spill_dir_provider=lambda: str(tmp_path),
      log_session_id="sess-delta",
      logger=_NullLogger(),
    )

    assert live_entry == durable_entry
    assert len(live_entry["content"]) < 2_000
    projection = json.loads(live_entry["content"])
    assert "readback" not in projection
    assert "verdict_echo" not in projection
    assert projection["status"] == "staged"
    assert projection["gate_code"] == "PROCEED"
    assert projection["artifact_ref"] == f"artifacts/PCTY/{tool_name}.json"
    assert projection["proposal_id"] == "proposal-1"
    assert projection["verdict"] == "ACCEPTED_CONSTRUCTED"
    assert projection["confidence"] == "HIGH"
    assert projection["revision"] == "rev-1"
    assert projection["stage_receipt_status"] == "staged"
    assert "findings" not in projection


def _persist_projection(payload: dict, tmp_path) -> dict:
  entry = {"type": "tool_result", "tool_use_id": "tool-undo", "content": json.dumps(payload)}
  live_entry, durable_entry = compact_model_tool_result_entry(
    entry,
    tool_name="fms_persist_scenario_multiple_pricing",
    spill_dir_provider=lambda: str(tmp_path),
    log_session_id="sess-undo",
    logger=_NullLogger(),
  )
  assert live_entry == durable_entry
  return json.loads(live_entry["content"])


def _applied_multiple_pricing_payload(receipt: dict) -> dict:
  return {
    "status": "applied",
    "gate_code": "PROCEED",
    "artifact_ref": "artifacts/PCTY/scenario_multiple_pricing.json",
    "proposal_id": "proposal-undo",
    "verdict_echo": {"verdict": "ACCEPTED_CONSTRUCTED", "confidence": "HIGH", "revision": "rev-2"},
    "receipt": receipt,
  }


def test_applied_persist_projection_keeps_the_undo_handle_and_drops_the_signed_token(tmp_path) -> None:
  signed_token = "SIGNED-UNDO-TOKEN-" + "s" * 400
  payload = _applied_multiple_pricing_payload(
    {
      "undo_available": True,
      "undo_token_id": "undo-123",
      "undo_expires_at": 1_790_000_000,
      "undo_token": signed_token,
    }
  )

  content_before = json.dumps(payload)
  projection = _persist_projection(payload, tmp_path)

  assert projection["status"] == "applied"
  assert projection["undo_available"] is True
  assert projection["undo_token_id"] == "undo-123"
  assert projection["undo_expires_at"] == 1_790_000_000
  assert "undo_token" not in projection
  assert signed_token not in json.dumps(projection)
  assert "receipt" not in projection
  assert json.dumps(payload) == content_before


def test_persist_projection_does_not_synthesize_undo_fields(tmp_path) -> None:
  for receipt in (
    {},
    {"receipt_id": "r-1"},
  ):
    projection = _persist_projection(_applied_multiple_pricing_payload(receipt), tmp_path)
    assert not [key for key in projection if key.startswith("undo_")], receipt

  staged = _applied_multiple_pricing_payload({})
  del staged["receipt"]
  staged["status"] = "staged"
  projection = _persist_projection(staged, tmp_path)
  assert projection["status"] == "staged"
  assert not [key for key in projection if key.startswith("undo_")]
  assert "receipt" not in projection

  explicit_no_undo = _persist_projection(
    _applied_multiple_pricing_payload({"undo_available": False}), tmp_path
  )
  assert explicit_no_undo["undo_available"] is False
  assert "undo_token_id" not in explicit_no_undo
  assert "undo_expires_at" not in explicit_no_undo


_SECRET = "CUSTOM-ACTIVE-CREDENTIAL-TOOL-RESULT-8f21d7"
_CUT_PREFIX = _SECRET[:10]
_MODEL_CUT_CHARS = 4_000


def _secret_entry(content: str) -> dict:
  return {"type": "tool_result", "tool_use_id": "tool-secret", "content": content}


def _retained(entry: dict) -> dict:
  safe = sanitize_boundary_value(
    entry,
    sink="tool_result",
    boundary=SecretBoundary((_SECRET,)),
  )
  assert isinstance(safe, dict) and safe is not entry
  return safe


def _secret_across_the_model_cut() -> dict:
  # The model ladder cuts this string at 2000 chars, ten chars into the secret.
  return _secret_entry(
    json.dumps(
      {
        "status": "ok",
        "public": "public-result-marker",
        "blob": "p" * 1_990 + _SECRET + "q" * 6_000,
      }
    )
  )


def test_durable_preview_is_cut_from_the_projected_content_and_live_stays_raw() -> None:
  entry = _secret_across_the_model_cut()
  retained = _retained(entry)
  entry_before, retained_before = copy.deepcopy(entry), copy.deepcopy(retained)
  prior_live, prior_durable = compact_model_tool_result_entry(
    entry,
    tool_name="lookup",
    log_session_id="sess-secret",
    logger=_NullLogger(),
    max_chars=_MODEL_CUT_CHARS,
  )
  # Without a retained entry the durable copy is the live one: the raw cut.
  assert prior_durable is prior_live
  assert _CUT_PREFIX in prior_live["content"]

  live_entry, durable_entry = compact_model_tool_result_entry(
    entry,
    tool_name="lookup",
    log_session_id="sess-secret",
    logger=_NullLogger(),
    max_chars=_MODEL_CUT_CHARS,
    retained_entry=retained,
  )

  assert entry == entry_before and retained == retained_before
  assert live_entry == prior_live
  assert _CUT_PREFIX in live_entry["content"]
  assert durable_entry is not live_entry
  assert durable_entry["tool_use_id"] == "tool-secret"
  durable_payload = json.loads(durable_entry["content"])
  assert durable_payload["_runner_truncated"] is True
  assert durable_payload["original_chars"] == len(retained["content"])
  assert durable_payload["content_projection"]["public"] == "public-result-marker"
  assert _CUT_PREFIX not in durable_entry["content"]
  # The cut now falls inside the redaction marker, not inside the secret.
  assert "p" * 1_990 + "<redacted-" in durable_entry["content"]


def test_spill_set_is_written_from_the_projected_content_and_live_keeps_the_raw_preview(
  tmp_path,
) -> None:
  entry = _secret_across_the_model_cut()
  retained = _retained(entry)
  prior_live, _ = compact_model_tool_result_entry(
    entry,
    tool_name="lookup",
    log_session_id="sess-secret",
    logger=_NullLogger(),
    max_chars=_MODEL_CUT_CHARS,
  )
  publications = []

  live_entry, durable_entry = compact_model_tool_result_entry(
    entry,
    tool_name="lookup",
    spill_sink=SpillSink(
      root_provider=lambda: str(tmp_path),
      capabilities=SpillCapabilities(code_execute=True, spill_read=True),
    ),
    log_session_id="sess-secret",
    logger=_NullLogger(),
    max_chars=_MODEL_CUT_CHARS,
    on_spill_publication=publications.append,
    retained_entry=retained,
  )

  live_payload = json.loads(live_entry["content"])
  prior_payload = json.loads(prior_live["content"])
  assert live_payload.pop("spill_ref").startswith("spill:v1:")
  assert live_payload.pop("spill_summary")["payload_kind"]
  assert live_payload == prior_payload
  assert _CUT_PREFIX in live_entry["content"]
  (publication,) = publications
  assert publication.spill_ref in live_entry["content"]
  assert _CUT_PREFIX not in durable_entry["content"]
  assert "spill_ref" not in json.loads(durable_entry["content"])
  spill_files = [path for path in tmp_path.rglob("*") if path.is_file()]
  assert spill_files
  spilled = [path.read_text(encoding="utf-8") for path in spill_files]
  assert all(_CUT_PREFIX not in text for text in spilled)
  assert any(json.loads(text)["public"] == "public-result-marker" for text in spilled)
  assert any("<redacted-secret>" in text for text in spilled)


def test_untruncated_entry_keeps_the_live_object_and_returns_the_retained_one() -> None:
  entry = _secret_entry(json.dumps({"public": "public-result-marker", "echo": _SECRET}))
  retained = _retained(entry)

  live_entry, durable_entry = compact_model_tool_result_entry(
    entry,
    tool_name="lookup",
    log_session_id="sess-secret",
    logger=_NullLogger(),
    max_chars=_MODEL_CUT_CHARS,
    retained_entry=retained,
  )

  assert live_entry is entry
  assert durable_entry is retained
  assert _SECRET in live_entry["content"]
  assert _SECRET not in durable_entry["content"]
  assert json.loads(durable_entry["content"])["public"] == "public-result-marker"


def test_non_text_entry_returns_the_live_object_and_the_retained_one() -> None:
  entry = {"type": "tool_result", "tool_use_id": "tool-secret", "content": [{"type": "text", "text": _SECRET}]}
  retained = _retained(entry)

  live_entry, durable_entry = compact_model_tool_result_entry(
    entry,
    tool_name="lookup",
    log_session_id="sess-secret",
    logger=_NullLogger(),
    retained_entry=retained,
  )

  assert live_entry is entry
  assert durable_entry is retained
  assert _SECRET not in json.dumps(durable_entry)


def test_fms_terminal_projection_is_projected_per_copy() -> None:
  envelope = {
    "status": "staged",
    "gate_code": "PROCEED",
    "artifact_ref": "artifacts/PCTY/business-model.json",
    "proposal_id": f"proposal-{_SECRET}",
    "error": None,
    "verdict": {
      "verdict": "BM_CONSTRUCTED",
      "confidence": "MEDIUM",
      "revision": "pcty-business-model-rev-1",
      "validation": {"large": "v" * 5_000},
    },
    "readback": {
      "typed_outputs": {
        "business_model_stage_receipt": {"status": "accepted"},
        "business_model": {"segments": ["x" * 5_000]},
      }
    },
  }
  entry = _secret_entry(json.dumps(envelope))
  retained = _retained(entry)

  live_entry, durable_entry = compact_model_tool_result_entry(
    entry,
    tool_name="fms_persist_business_model",
    log_session_id="sess-secret",
    logger=_NullLogger(),
    retained_entry=retained,
  )

  live_projection = json.loads(live_entry["content"])
  durable_projection = json.loads(durable_entry["content"])
  assert "readback" not in live_projection and "readback" not in durable_projection
  assert live_projection["proposal_id"] == f"proposal-{_SECRET}"
  assert durable_projection["proposal_id"] == "proposal-<redacted-secret>"
  assert durable_projection["gate_code"] == live_projection["gate_code"] == "PROCEED"
  assert durable_entry["tool_use_id"] == live_entry["tool_use_id"] == "tool-secret"
  assert _SECRET not in durable_entry["content"]


def test_redaction_that_lengthens_content_spills_the_projected_text_and_leaves_live_raw(tmp_path) -> None:
  # An 8-char secret becomes the 17-char marker: the raw content fits the limit
  # while the retained copy does not.
  short_secret = "ab12cd34"
  max_chars = 1_500
  entry = _secret_entry(
    json.dumps({"public": "public-result-marker", "echo": (short_secret + " ") * 100})
  )
  assert len(entry["content"]) < max_chars
  retained = sanitize_boundary_value(
    entry,
    sink="tool_result",
    boundary=SecretBoundary((short_secret,)),
  )
  assert len(retained["content"]) > max_chars
  publications = []

  live_entry, durable_entry = compact_model_tool_result_entry(
    entry,
    tool_name="lookup",
    spill_sink=SpillSink(
      root_provider=lambda: str(tmp_path),
      capabilities=SpillCapabilities(code_execute=True, spill_read=True),
    ),
    log_session_id="sess-secret",
    logger=_NullLogger(),
    max_chars=max_chars,
    on_spill_publication=publications.append,
    retained_entry=retained,
  )

  assert live_entry is entry
  assert short_secret in live_entry["content"]
  assert durable_entry is not retained
  assert durable_entry["tool_use_id"] == "tool-secret"
  durable_payload = json.loads(durable_entry["content"])
  assert durable_payload["_runner_truncated"] is True
  assert durable_payload["original_chars"] == len(retained["content"])
  assert "spill_ref" not in durable_payload
  assert short_secret not in durable_entry["content"]
  (publication,) = publications
  spill_files = [path for path in tmp_path.rglob("*") if path.is_file()]
  assert len(spill_files) == 1
  spilled = spill_files[0].read_text(encoding="utf-8")
  assert json.loads(spilled) == json.loads(retained["content"])
  assert short_secret not in spilled
  assert publication.spill_ref.startswith(f"spill:v1:{spill_files[0].name}:")


def test_report_door_receipt_omits_source_map_instruction_and_artifact_path(tmp_path) -> None:
  """PCTY turn 2 copied the door path and the source-map quote instruction.

  The model-visible receipt must not carry either, at any size: a small
  envelope used to pass through whole, and a large one kept both inside the
  shape-preserving truncation the model was told it may report as-is.
  """
  warning = {
    "message": "source-map ref S17 quote was not found in the verified handle text",
    "reason": "quote_not_found",
    "data": {
      "outcome": "handle_only",
      "fix": "Replace the submitted quote with one contiguous exact quote from the cited source.",
      "next_action": "Copy one contiguous exact quote from the canonical source text and retry the same source_ref with that quote.",
      "citation_readiness": "quote_mintable",
    },
  }
  payload = {
    "status": "applied",
    "subcommand": "report_business_quality",
    "mutation_mode": "thesis_writer",
    "ticker": "PCTY",
    "artifact_ref": "artifacts/PCTY/business-quality-assessment/2026-11-05T023057.655-93bac72e.json",
    "typed_artifact_ref": "artifacts/PCTY/business-quality-assessment/2026-11-05T023057.655-93bac72e.json",
    "gate_code": "PROCEED",
    "artifact_id": "2026-11-05T023057.655-93bac72e",
    "sidecar_hash": "sha256:36cddbf3195e90f8d5a093ac612c882562b4a493",
    "next_actions": [
      "Before the imminent Q1 FY27 print, build the GAAP-to-adjusted margin bridge."
    ],
    "verdict": {
      "verdict": "HIGH_QUALITY",
      "confidence": "MEDIUM",
      "composite": 4.2,
      "summary": "Five reinforcing pillars; the quality rating stands.",
      "recommended_next_action": "Build the GAAP-to-adjusted margin bridge before the print.",
      "data_gap_count": 6,
      "source_count": 18,
      # R23 GOOGL shape: the door admitted with findings naming the failing source-map
      # refs; the receipt keeps the typed facts (code, field, refs), never the prose
      # message with its "source-map ref" instruction or the next_call.
      "findings": [
        {"field": "sources", "code": "quote_not_found",
         "message": "source-map ref S1515 quote was not found in the verified handle text",
         "next_call": {"tool_name": "thesis_read", "arguments": {}}},
        {"field": "sources", "code": "quote_not_found",
         "message": "source-map ref S1514 quote was not found in the verified handle text",
         "next_call": {"tool_name": "thesis_read", "arguments": {}}},
        {"field": "thesis_projection.thesis", "code": "field_withheld",
         "message": "the typed Thesis ThesisField does not hold conviction, date_initiated",
         "next_call": {"tool_name": "thesis_read", "arguments": {}}},
      ],
    },
    "readback": {
      "citation_degradation_warnings": [warning, warning],
      "decisions_log": [
        {
          "thesis_id": "4d459e6d-027b-4535-9a9b-67f28377982a",
          "artifact_refs": [
            {"artifact_path": "artifacts/PCTY/business-quality-assessment/2026-11-05T023057.655-93bac72e.json"}
          ],
        }
      ],
      "typed_outputs": {
        "thesis_projection": {"business_overview": {"description": "Payroll and HCM"}},
        "bulk": "x" * 80_000,
      },
    },
  }
  content = json.dumps(payload)
  assert len(content) > 60_000
  entry = {"type": "tool_result", "tool_use_id": "tool-bq", "content": content}

  live_entry, durable_entry = compact_model_tool_result_entry(
    entry,
    tool_name="fms_report_business_quality_assessment",
    spill_dir_provider=lambda: str(tmp_path),
    log_session_id="sess-report",
    logger=_NullLogger(),
  )

  assert live_entry == durable_entry
  assert live_entry is not entry
  text = live_entry["content"]
  assert "artifacts/" not in text
  assert "source-map ref" not in text
  assert "handle_only" not in text
  assert "quote_mintable" not in text
  assert "citation_use" not in text
  assert "contiguous exact quote" not in text
  assert "4d459e6d" not in text
  assert "may be reported as-is" not in text
  receipt = json.loads(text)
  assert receipt["status"] == "applied"
  assert receipt["verdict"] == "HIGH_QUALITY"
  assert receipt["confidence"] == "MEDIUM"
  assert receipt["findings"] == [
    {"code": "quote_not_found", "field": "sources", "source_refs": ["S1515"]},
    {"code": "quote_not_found", "field": "sources", "source_refs": ["S1514"]},
    {"code": "field_withheld", "field": "thesis_projection.thesis"},
  ]
  assert "thesis_read" not in text
  assert receipt["composite"] == 4.2
  assert "bridge" in receipt["recommended_next_action"]
  assert receipt["next_actions"] == payload["next_actions"]
  assert receipt["decision_log_appended"] is True
  assert receipt["business_overview_projected"] is True
  assert "readback" not in receipt
  assert "artifact_ref" not in receipt
  assert "typed_artifact_ref" not in receipt
  assert receipt["artifact_id"] == payload["artifact_id"]
  assert "/" not in receipt["artifact_id"]
  assert "citation_degradation_warnings" not in receipt
