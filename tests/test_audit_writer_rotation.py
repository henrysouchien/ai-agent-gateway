"""Day-file segment rotation for the JSONL approval audit writer.

The capacity caps are resource ceilings, not reasons to stop recording:
a full segment rolls to ``{day}.{n}.jsonl`` instead of refusing the append.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_gateway import audit_writer as audit_writer_module
from agent_gateway.approval_audit import ApprovalAuditEntry
from agent_gateway.audit_writer import JSONLAuditWriter


_TS = datetime(2026, 8, 22, 12, 0, 0, tzinfo=UTC)


def _entry(entry_id: str, *, reason: str = "ok") -> ApprovalAuditEntry:
  return ApprovalAuditEntry(
    entry_id=entry_id,
    approval_id=f"approval-{entry_id}",
    request_id=f"request-{entry_id}",
    tool_call_id=f"call-{entry_id}",
    parent_approval_id=None,
    approval_chain_id=f"chain-{entry_id}",
    pending_tools_nonce=None,
    ts=_TS,
    event_type="tool_executed_success",
    user_id="user-1",
    profile="default",
    channel="chat",
    session_id=None,
    run_id=None,
    skill=None,
    tool_name="write_file",
    tool_class="state_write",
    tool_args_redacted={"path": "x"},
    args_hash="a" * 64,
    args_hash_version="sha256",
    decider_id="owner-1",
    decider_role="owner",
    decision_reason=reason,
    decision_latency_ms=1,
    outcome="success",
    error_summary=None,
    policy_id="policy-1",
    policy_version="1",
    policy_bundle_hash="b" * 64,
  )


def _entry_ids(path: Path) -> list[str]:
  return [
    json.loads(line)["entry_id"]
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
  ]


def test_record_capacity_rolls_to_next_segment(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(audit_writer_module, "APPROVAL_AUDIT_MAX_RECORDS", 2)
  writer = JSONLAuditWriter(tmp_path)
  for index in range(5):
    asyncio.run(writer.write(_entry(f"entry-{index}")))

  base = writer._segment_path(_TS, 0)
  first = writer._segment_path(_TS, 1)
  second = writer._segment_path(_TS, 2)
  assert _entry_ids(base) == ["entry-0", "entry-1"]
  assert _entry_ids(first) == ["entry-2", "entry-3"]
  assert _entry_ids(second) == ["entry-4"]
  assert not writer._segment_path(_TS, 3).exists()

  persisted, cursor = asyncio.run(writer.query(limit=10, order="asc"))
  assert cursor is None
  assert [entry.entry_id for entry in persisted] == [
    f"entry-{index}" for index in range(5)
  ]


def test_byte_capacity_rolls_to_next_segment(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  writer = JSONLAuditWriter(tmp_path)
  asyncio.run(writer.write(_entry("entry-0")))
  base = writer._segment_path(_TS, 0)
  monkeypatch.setattr(
    audit_writer_module,
    "APPROVAL_AUDIT_MAX_FILE_BYTES",
    base.stat().st_size + 1,
  )
  asyncio.run(writer.write(_entry("entry-1")))
  assert _entry_ids(base) == ["entry-0"]
  assert _entry_ids(writer._segment_path(_TS, 1)) == ["entry-1"]


def test_replay_into_full_segment_stays_idempotent(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(audit_writer_module, "APPROVAL_AUDIT_MAX_RECORDS", 1)
  writer = JSONLAuditWriter(tmp_path)
  asyncio.run(writer.write(_entry("entry-0")))
  asyncio.run(writer.write(_entry("entry-1")))
  # Replaying an entry that lives in a now-full segment is recognized there
  # and never duplicated into a later segment.
  asyncio.run(writer.write(_entry("entry-0")))

  assert _entry_ids(writer._segment_path(_TS, 0)) == ["entry-0"]
  assert _entry_ids(writer._segment_path(_TS, 1)) == ["entry-1"]
  assert not writer._segment_path(_TS, 2).exists()


def test_entry_id_reuse_with_different_content_still_raises(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(audit_writer_module, "APPROVAL_AUDIT_MAX_RECORDS", 1)
  writer = JSONLAuditWriter(tmp_path)
  asyncio.run(writer.write(_entry("entry-0")))
  asyncio.run(writer.write(_entry("entry-1")))
  with pytest.raises(RuntimeError, match="entry_id was reused"):
    asyncio.run(writer.write(_entry("entry-0", reason="tampered")))


def test_oversized_record_is_still_refused(tmp_path: Path) -> None:
  writer = JSONLAuditWriter(tmp_path)
  huge = _entry("entry-huge", reason="x" * (512 * 1024))
  with pytest.raises(RuntimeError, match="record exceeds its byte limit"):
    asyncio.run(writer.write(huge))
  assert list(tmp_path.glob("*.jsonl")) == []


def test_historical_rows_project_through_current_schema(tmp_path: Path) -> None:
  writer = JSONLAuditWriter(tmp_path)
  asyncio.run(writer.write(_entry("entry-0")))
  path = writer._segment_path(_TS, 0)

  old_row = _entry("entry-old").to_json_dict()
  del old_row["approval_constraint"]  # field added after this row was written
  new_row = _entry("entry-new").to_json_dict()
  new_row["field_from_the_future"] = "carried in durable bytes"
  with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(old_row, sort_keys=True) + "\n")
    handle.write(json.dumps(new_row, sort_keys=True) + "\n")

  persisted, _ = asyncio.run(writer.query(limit=10, order="asc"))
  assert sorted(entry.entry_id for entry in persisted) == [
    "entry-0",
    "entry-new",
    "entry-old",
  ]
  by_id = {entry.entry_id: entry for entry in persisted}
  assert by_id["entry-old"].approval_constraint == "legacy_unknown"


def test_retention_refuses_unknown_retention_class(tmp_path: Path) -> None:
  writer = JSONLAuditWriter(tmp_path)
  path = writer._segment_path(_TS, 0)
  row = _entry("entry-0").to_json_dict()
  row["retention_class"] = "quarantine_hold"
  path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

  with pytest.raises(KeyError):
    asyncio.run(writer.apply_retention())
  assert path.exists()
