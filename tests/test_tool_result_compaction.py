import json
from types import SimpleNamespace

from agent_gateway.policy_imports import configure_server_policy, load_server_policy_module
from agent_gateway.tool_result_compaction import truncate_model_tool_result_content

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
