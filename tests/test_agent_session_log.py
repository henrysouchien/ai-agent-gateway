import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentSessionLog, AgentSessionRef, QueryCursor, resolve_agent_session_id
from agent_gateway.agent_session_log import agent_session_logical_path_for_jsonl, slugify


def _run(coro):
  return asyncio.run(coro)


def _load_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  for raw_line in path.read_text(encoding="utf-8").splitlines():
    if raw_line.strip():
      rows.append(json.loads(raw_line))
  return rows


def _build_sample_log(tmp_path: Path) -> tuple[AgentSessionLog, list]:
  log = AgentSessionLog(tmp_path / "sample.jsonl")
  entries = []
  sample_events = [
    {"type": "attach", "runner_id": "run-0", "role": "writer", "client_kind": "cron"},
    {
      "type": "assistant_message",
      "runner_id": "run-1",
      "role": "writer",
      "content_blocks": [{"type": "text", "text": "Alpha briefing"}],
    },
    {
      "type": "tool_call_start",
      "tool_name": "screen_estimate_revisions",
      "tool_call_id": "tc-1",
      "sub_agent_id": "sub-1",
      "runner_id": "run-1",
      "role": "sub_agent",
      "tool_input": {"query": "Alpha ticker scan"},
    },
    {
      "type": "tool_call_complete",
      "tool_name": "screen_estimate_revisions",
      "tool_call_id": "tc-1",
      "sub_agent_id": "sub-1",
      "runner_id": "run-1",
      "role": "sub_agent",
      "result": {"text": "Alpha results"},
    },
    {
      "type": "tool_call_complete",
      "tool_name": "get_news",
      "tool_call_id": "tc-2",
      "sub_agent_id": "sub-2",
      "runner_id": "run-2",
      "role": "writer",
      "error": {"code": "quota", "message": "Quota exceeded"},
      "details": {"text": "quota failure"},
    },
    {
      "type": "summary",
      "runner_id": "run-2",
      "role": "writer",
      "text": "Macro wrap-up for beta",
    },
  ]

  for event in sample_events:
    entries.append(_run(log.append(event)))
    time.sleep(0.01)
  return log, entries


def test_agent_session_log_creates_parent_dirs_and_file_from_session_ref(tmp_path: Path) -> None:
  base_dir = tmp_path / "api" / "sessions"
  session_ref = AgentSessionRef(
    user_id="Henry",
    agent_id="Analyst",
    agent_session_id=resolve_agent_session_id("Henry", "Analyst"),
  )

  log = AgentSessionLog(session_ref=session_ref, base_dir=base_dir)

  assert log.path == base_dir / "analyst" / "agentsess_analyst_henry.jsonl"
  assert log.path.parent.is_dir()
  assert log.path.is_file()


def test_append_assigns_seq_timestamp_and_schema_version(tmp_path: Path) -> None:
  log = AgentSessionLog(tmp_path / "append.jsonl")

  first = _run(log.append({"type": "attach", "runner_id": "run-1"}))
  second = _run(log.append({"type": "assistant_message", "runner_id": "run-2"}))

  assert first.seq == 1
  assert second.seq == 2
  assert first.timestamp > 0
  assert second.timestamp >= first.timestamp
  assert first.event["event_schema_version"] == 1
  assert second.event["event_schema_version"] == 1
  assert _run(log.latest_seq()) == 2

  iterated = list(_run(_collect_async(log.iter_from(after_seq=1))))
  assert [entry.seq for entry in iterated] == [2]

  rows = _load_jsonl(log.path)
  assert [row["seq"] for row in rows] == [1, 2]
  assert rows[0]["event"]["event_schema_version"] == 1
  assert rows[1]["event"]["event_schema_version"] == 1


async def _collect_async(iterator):
  return [entry async for entry in iterator]


def test_query_supports_all_filters_and_combined_filters(tmp_path: Path) -> None:
  log, entries = _build_sample_log(tmp_path)

  only_assistant, _ = _run(log.query(event_types={"assistant_message"}))
  assert [entry.seq for entry in only_assistant] == [entries[1].seq]

  by_tool_name, _ = _run(log.query(tool_name="screen_estimate_revisions"))
  assert [entry.seq for entry in by_tool_name] == [entries[2].seq, entries[3].seq]

  by_tool_call_id, _ = _run(log.query(tool_call_id="tc-2"))
  assert [entry.seq for entry in by_tool_call_id] == [entries[4].seq]

  by_sub_agent, _ = _run(log.query(sub_agent_id="sub-1"))
  assert [entry.seq for entry in by_sub_agent] == [entries[2].seq, entries[3].seq]

  by_runner, _ = _run(log.query(runner_id="run-2"))
  assert [entry.seq for entry in by_runner] == [entries[4].seq, entries[5].seq]

  by_role, _ = _run(log.query(role="sub_agent"))
  assert [entry.seq for entry in by_role] == [entries[2].seq, entries[3].seq]

  after_seq_entries, _ = _run(log.query(after_seq=entries[3].seq))
  assert [entry.seq for entry in after_seq_entries] == [entries[3].seq, entries[4].seq, entries[5].seq]

  before_seq_entries, _ = _run(log.query(before_seq=entries[2].seq))
  assert [entry.seq for entry in before_seq_entries] == [entries[0].seq, entries[1].seq, entries[2].seq]

  after_ts_entries, _ = _run(log.query(after_ts=entries[2].timestamp))
  assert [entry.seq for entry in after_ts_entries] == [entries[2].seq, entries[3].seq, entries[4].seq, entries[5].seq]

  before_ts_entries, _ = _run(log.query(before_ts=entries[4].timestamp))
  assert [entry.seq for entry in before_ts_entries] == [
    entries[0].seq,
    entries[1].seq,
    entries[2].seq,
    entries[3].seq,
    entries[4].seq,
  ]

  by_text, _ = _run(log.query(contains_text="quota"))
  assert [entry.seq for entry in by_text] == [entries[4].seq]

  error_only, _ = _run(log.query(has_error=True))
  assert [entry.seq for entry in error_only] == [entries[4].seq]

  no_error, _ = _run(log.query(has_error=False))
  assert [entry.seq for entry in no_error] == [entry.seq for entry in entries if entry.seq != entries[4].seq]

  combined, _ = _run(
    log.query(
      event_types={"tool_call_complete"},
      tool_name="screen_estimate_revisions",
      sub_agent_id="sub-1",
      has_error=False,
    )
  )
  assert [entry.seq for entry in combined] == [entries[3].seq]

  text_and_role, _ = _run(log.query(role="sub_agent", contains_text="alpha"))
  assert [entry.seq for entry in text_and_role] == [entries[2].seq, entries[3].seq]


def test_query_honors_asc_desc_ordering_and_pagination(tmp_path: Path) -> None:
  log, entries = _build_sample_log(tmp_path)

  asc_entries, _ = _run(log.query(order="asc"))
  desc_entries, _ = _run(log.query(order="desc"))

  assert [entry.seq for entry in asc_entries] == [entry.seq for entry in entries]
  assert [entry.seq for entry in desc_entries] == [entry.seq for entry in reversed(entries)]

  first_page, cursor = _run(log.query(order="asc", limit=2))
  assert [entry.seq for entry in first_page] == [entries[0].seq, entries[1].seq]
  assert cursor == QueryCursor(after_seq=entries[1].seq, direction="asc")

  second_page, second_cursor = _run(log.query(order="asc", limit=2, cursor=cursor))
  assert [entry.seq for entry in second_page] == [entries[2].seq, entries[3].seq]
  assert second_cursor == QueryCursor(after_seq=entries[3].seq, direction="asc")

  third_page, third_cursor = _run(log.query(order="asc", limit=2, cursor=second_cursor))
  assert [entry.seq for entry in third_page] == [entries[4].seq, entries[5].seq]
  assert third_cursor is None


def test_desc_query_uses_reverse_scan_and_stops_early(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  log = AgentSessionLog(tmp_path / "reverse.jsonl")
  for idx in range(1200):
    _run(log.append({"type": "assistant_message", "index": idx, "text": f"event-{idx}"}))

  seen_lines = 0
  original_reverse = log._iter_lines_reverse

  def _counting_reverse(*args, **kwargs):
    nonlocal seen_lines
    for item in original_reverse(*args, **kwargs):
      seen_lines += 1
      yield item

  def _unexpected_forward(*_args, **_kwargs):
    raise AssertionError("descending queries should not use the forward scan path")

  monkeypatch.setattr(log, "_iter_lines_reverse", _counting_reverse)
  monkeypatch.setattr(log, "_query_asc_sync", _unexpected_forward)

  entries, _ = _run(log.query(order="desc", limit=1))

  assert len(entries) == 1
  assert entries[0].seq == 1200
  assert entries[0].event["index"] == 1199
  assert seen_lines <= 3


def test_query_past_active_tail_uses_fast_bounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  path = tmp_path / "large-active.jsonl"
  rows = [
    json.dumps(
      {
        "seq": seq,
        "timestamp": float(seq),
        "event": {"type": "assistant_message", "index": seq},
      },
      separators=(",", ":"),
    )
    for seq in range(1, 5001)
  ]
  path.write_text("\n".join(rows) + "\n", encoding="utf-8")
  log = AgentSessionLog(path)

  parsed_entries = 0
  original_parse_entry = log._parse_entry

  def _counting_parse_entry(raw: bytes, *, is_last_line: bool):
    nonlocal parsed_entries
    parsed_entries += 1
    return original_parse_entry(raw, is_last_line=is_last_line)

  monkeypatch.setattr(log, "_parse_entry", _counting_parse_entry)

  entries, _ = _run(log.query(after_seq=5001, order="asc"))

  assert entries == []
  assert parsed_entries <= 4


def test_active_offset_cache_invalidates_after_external_rotation(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  path = tmp_path / "rotated-by-peer.jsonl"
  cached_reader = AgentSessionLog(path)
  rotating_writer = AgentSessionLog(path)

  _run(cached_reader.append({"type": "assistant_message", "text": "first"}))
  first_entries, _ = _run(cached_reader.query(after_seq=1, order="asc"))
  assert [entry.seq for entry in first_entries] == [1]

  _run(rotating_writer.append({"type": "assistant_message", "text": "second"}))
  second_entries, _ = _run(cached_reader.query(after_seq=2, order="asc"))

  assert [entry.seq for entry in second_entries] == [2]


def test_truncated_trailing_line_is_skipped_with_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
  path = tmp_path / "truncated.jsonl"
  valid_row = {
    "seq": 1,
    "timestamp": 123.0,
    "event": {"type": "attach", "runner_id": "run-1", "event_schema_version": 1},
  }
  path.write_text(json.dumps(valid_row) + "\n" + '{"seq": 2, "timestamp": 124.0, "event": {"type": "', encoding="utf-8")
  log = AgentSessionLog(path)

  with caplog.at_level(logging.WARNING, logger="agent_gateway.agent_session_log"):
    entries, _ = _run(log.query(order="asc"))

  assert [entry.seq for entry in entries] == [1]
  assert _run(log.latest_seq()) == 1
  assert "Skipping truncated trailing JSONL line" in caplog.text


def test_concurrent_appends_remain_well_formed_jsonl(tmp_path: Path) -> None:
  log = AgentSessionLog(tmp_path / "concurrent.jsonl")

  async def _append_many() -> None:
    await asyncio.gather(*(log.append({"type": "assistant_message", "index": idx}) for idx in range(200)))

  _run(_append_many())

  rows = _load_jsonl(log.path)
  assert len(rows) == 200
  assert [row["seq"] for row in rows] == list(range(1, 201))
  assert all(row["event"]["event_schema_version"] == 1 for row in rows)
  assert _run(log.latest_seq()) == 200


def test_rotation_keeps_query_pagination_iter_and_latest_seq(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  log = AgentSessionLog(tmp_path / "rotating.jsonl")

  for idx in range(5):
    _run(log.append({"type": "assistant_message", "index": idx, "text": f"event-{idx}"}))

  manifest = json.loads(log.manifest_path.read_text(encoding="utf-8"))
  assert manifest["latest_seq"] == 5
  assert len(manifest["segments"]) == 4
  assert [item["first_seq"] for item in manifest["segments"]] == [1, 2, 3, 4]
  assert [item["last_seq"] for item in manifest["segments"]] == [1, 2, 3, 4]
  assert [row["seq"] for row in _load_jsonl(log.path)] == [5]

  asc_entries, _ = _run(log.query(order="asc"))
  desc_entries, _ = _run(log.query(order="desc"))
  assert [entry.seq for entry in asc_entries] == [1, 2, 3, 4, 5]
  assert [entry.seq for entry in desc_entries] == [5, 4, 3, 2, 1]

  desc_page, desc_cursor = _run(log.query(order="desc", limit=2))
  assert [entry.seq for entry in desc_page] == [5, 4]
  assert desc_cursor == QueryCursor(after_seq=4, direction="desc")
  desc_second_page, desc_second_cursor = _run(log.query(order="desc", limit=2, cursor=desc_cursor))
  assert [entry.seq for entry in desc_second_page] == [3, 2]
  assert desc_second_cursor == QueryCursor(after_seq=2, direction="desc")

  first_page, cursor = _run(log.query(order="asc", limit=3))
  assert [entry.seq for entry in first_page] == [1, 2, 3]
  assert cursor == QueryCursor(after_seq=3, direction="asc")
  second_page, second_cursor = _run(log.query(order="asc", limit=3, cursor=cursor))
  assert [entry.seq for entry in second_page] == [4, 5]
  assert second_cursor is None

  filtered, _ = _run(log.query(contains_text="event-3", order="asc"))
  assert [entry.seq for entry in filtered] == [4]
  iterated = list(_run(_collect_async(log.iter_from(after_seq=2))))
  assert [entry.seq for entry in iterated] == [3, 4, 5]
  assert _run(log.latest_seq()) == 5


def test_rotation_writes_v2_active_and_segment_sidecars(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  ref = AgentSessionRef(
    user_id="alice",
    agent_id="analyst",
    agent_session_id=resolve_agent_session_id("alice", "analyst"),
  )
  log = AgentSessionLog(session_ref=ref, base_dir=tmp_path / "sessions")

  _run(log.append({"type": "assistant_message", "text": "first"}))
  _run(log.append({"type": "assistant_message", "text": "second"}))

  active_meta = json.loads(log.path.with_suffix(".meta.json").read_text(encoding="utf-8"))
  manifest = json.loads(log.manifest_path.read_text(encoding="utf-8"))
  segment_path = log.segments_dir / manifest["segments"][0]["path"]
  segment_meta = json.loads(segment_path.with_suffix(".meta.json").read_text(encoding="utf-8"))

  assert active_meta["schema_version"] == 2
  assert active_meta["file_role"] == "active"
  assert active_meta["active_generation"] == 1
  assert active_meta["agent_session_id"] == ref.agent_session_id
  assert segment_meta["schema_version"] == 2
  assert segment_meta["file_role"] == "segment"
  assert segment_meta["segment_id"] == manifest["segments"][0]["segment_id"]
  assert segment_meta["first_seq"] == 1
  assert segment_meta["last_seq"] == 1
  assert segment_meta["rotated_from_source_id"] == manifest["segments"][0]["rotated_from_source_id"]
  assert set(segment_meta["rotated_from_file_identity"]) == {"mtime_ns", "size", "st_dev", "st_ino"}
  assert agent_session_logical_path_for_jsonl(log.path) == log.path.resolve()
  assert agent_session_logical_path_for_jsonl(segment_path) == log.path.resolve()

  orphan_segment = log.segments_dir / "orphan.jsonl"
  orphan_segment.write_text("", encoding="utf-8")
  assert agent_session_logical_path_for_jsonl(orphan_segment) is None


def test_repair_rebuilds_missing_manifest_and_segment_sidecar(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  ref = AgentSessionRef(
    user_id="alice",
    agent_id="analyst",
    agent_session_id=resolve_agent_session_id("alice", "analyst"),
  )
  log = AgentSessionLog(session_ref=ref, base_dir=tmp_path / "sessions")
  _run(log.append({"type": "assistant_message", "text": "first"}))
  _run(log.append({"type": "assistant_message", "text": "second"}))
  manifest = json.loads(log.manifest_path.read_text(encoding="utf-8"))
  segment_path = log.segments_dir / manifest["segments"][0]["path"]

  log.manifest_path.unlink()
  segment_path.with_suffix(".meta.json").unlink()

  repaired = AgentSessionLog(path=log.path)
  repaired_manifest = json.loads(repaired.manifest_path.read_text(encoding="utf-8"))
  repaired_segment_meta = json.loads(segment_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
  entries, _ = _run(repaired.query(order="asc"))

  assert [entry.seq for entry in entries] == [1, 2]
  assert repaired_manifest["active_generation"] == 1
  assert repaired_manifest["latest_seq"] == 2
  assert repaired_manifest["segments"][0]["segment_id"] == "000000000001-000000000001-g000000"
  assert repaired_segment_meta["schema_version"] == 2
  assert repaired_segment_meta["file_role"] == "segment"
  assert repaired_segment_meta["agent_session_id"] == ref.agent_session_id
  assert repaired_segment_meta["agent_id"] == ref.agent_id
  assert repaired_segment_meta["user_id"] == ref.user_id


def test_repair_rewrites_missing_segment_sidecar_and_recreates_active(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  ref = AgentSessionRef(
    user_id="alice",
    agent_id="analyst",
    agent_session_id=resolve_agent_session_id("alice", "analyst"),
  )
  log = AgentSessionLog(session_ref=ref, base_dir=tmp_path / "sessions")
  _run(log.append({"type": "assistant_message", "text": "first"}))
  log._rotate_active_if_needed_locked()
  manifest = json.loads(log.manifest_path.read_text(encoding="utf-8"))
  segment_path = log.segments_dir / manifest["segments"][0]["path"]

  segment_path.with_suffix(".meta.json").unlink()
  log.path.unlink()
  log.path.with_suffix(".meta.json").unlink()

  repaired = AgentSessionLog(path=log.path)
  assert repaired.path.exists()
  assert repaired.path.read_text(encoding="utf-8") == ""
  assert segment_path.with_suffix(".meta.json").exists()
  active_meta = json.loads(repaired.path.with_suffix(".meta.json").read_text(encoding="utf-8"))
  assert active_meta["schema_version"] == 2
  assert active_meta["file_role"] == "active"
  assert active_meta["active_generation"] == 1

  second = _run(repaired.append({"type": "assistant_message", "text": "second"}))
  entries, _ = _run(repaired.query(order="asc"))
  assert second.seq == 2
  assert [entry.seq for entry in entries] == [1, 2]


def test_slugify_normalizes_and_rejects_empty() -> None:
  assert slugify("Henry C.") == "henry_c_"
  assert slugify("Already_OK-123") == "already_ok-123"
  assert slugify("A" * 80) == "a" * 64
  with pytest.raises(ValueError):
    slugify("")


def test_resolve_agent_session_id_uses_canonical_format() -> None:
  assert resolve_agent_session_id("Henry C.", "Analyst") == "agentsess_analyst_henry_c_"
