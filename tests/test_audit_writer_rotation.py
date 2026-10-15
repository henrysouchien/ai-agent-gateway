"""Day-file segment rotation for the JSONL approval audit writer.

The capacity caps are resource ceilings, not reasons to stop recording:
a full segment rolls to ``{day}.{n}.jsonl`` instead of refusing the append.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_gateway import audit_writer as audit_writer_module
from agent_gateway.approval_audit import ApprovalAuditEntry, RetentionClass
from agent_gateway.audit_writer import JSONLAuditWriter, owner_key


_TS = datetime(2026, 8, 22, 12, 0, 0, tzinfo=UTC)


def _entry(
  entry_id: str,
  *,
  reason: str = "ok",
  user_id: str = "user-1",
  ts: datetime = _TS,
  retention_class: RetentionClass = "operational",
  legal_hold: bool = False,
) -> ApprovalAuditEntry:
  return ApprovalAuditEntry(
    entry_id=entry_id,
    approval_id=f"approval-{entry_id}",
    request_id=f"request-{entry_id}",
    tool_call_id=f"call-{entry_id}",
    parent_approval_id=None,
    approval_chain_id=f"chain-{entry_id}",
    pending_tools_nonce=None,
    ts=ts,
    event_type="tool_executed_success",
    user_id=user_id,
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
    retention_class=retention_class,
    legal_hold=legal_hold,
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

  base = writer._segment_path(_TS, 0, user_id="user-1")
  first = writer._segment_path(_TS, 1, user_id="user-1")
  second = writer._segment_path(_TS, 2, user_id="user-1")
  assert _entry_ids(base) == ["entry-0", "entry-1"]
  assert _entry_ids(first) == ["entry-2", "entry-3"]
  assert _entry_ids(second) == ["entry-4"]
  assert not writer._segment_path(_TS, 3, user_id="user-1").exists()

  persisted, cursor = asyncio.run(writer.query(owner_user_id="user-1", limit=10, order="asc"))
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
  base = writer._segment_path(_TS, 0, user_id="user-1")
  monkeypatch.setattr(
    audit_writer_module,
    "APPROVAL_AUDIT_MAX_FILE_BYTES",
    base.stat().st_size + 1,
  )
  asyncio.run(writer.write(_entry("entry-1")))
  assert _entry_ids(base) == ["entry-0"]
  assert _entry_ids(writer._segment_path(_TS, 1, user_id="user-1")) == ["entry-1"]


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

  assert _entry_ids(writer._segment_path(_TS, 0, user_id="user-1")) == ["entry-0"]
  assert _entry_ids(writer._segment_path(_TS, 1, user_id="user-1")) == ["entry-1"]
  assert not writer._segment_path(_TS, 2, user_id="user-1").exists()


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
  assert list(writer._owner_directory("user-1").glob("*.jsonl")) == []


def test_historical_rows_project_through_current_schema(tmp_path: Path) -> None:
  writer = JSONLAuditWriter(tmp_path)
  asyncio.run(writer.write(_entry("entry-0")))
  path = writer._segment_path(_TS, 0, user_id="user-1")

  old_row = _entry("entry-old").to_json_dict()
  del old_row["approval_constraint"]  # field added after this row was written
  new_row = _entry("entry-new").to_json_dict()
  new_row["field_from_the_future"] = "carried in durable bytes"
  with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(old_row, sort_keys=True) + "\n")
    handle.write(json.dumps(new_row, sort_keys=True) + "\n")

  persisted, _ = asyncio.run(writer.query(owner_user_id="user-1", limit=10, order="asc"))
  assert sorted(entry.entry_id for entry in persisted) == [
    "entry-0",
    "entry-new",
    "entry-old",
  ]
  by_id = {entry.entry_id: entry for entry in persisted}
  assert by_id["entry-old"].approval_constraint == "legacy_unknown"


def test_retention_refuses_unknown_retention_class(tmp_path: Path) -> None:
  writer = JSONLAuditWriter(tmp_path)
  path = writer._segment_path(_TS, 0, user_id="user-1")
  path.parent.mkdir(mode=0o700)
  row = _entry("entry-0").to_json_dict()
  row["retention_class"] = "quarantine_hold"
  path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

  with pytest.raises(KeyError):
    asyncio.run(writer.apply_retention(owner_user_id="user-1"))
  assert path.exists()


_OLD = datetime(2020, 1, 1, tzinfo=UTC)


def _seal(path: Path) -> None:
  path.chmod(0o000)


def _unseal(path: Path) -> None:
  path.chmod(0o700)


def test_query_is_owner_scoped_and_does_not_open_other_owner(tmp_path: Path) -> None:
  writer = JSONLAuditWriter(tmp_path)
  asyncio.run(writer.write(_entry("entry-a", user_id="user-a")))
  asyncio.run(writer.write(_entry("entry-a2", user_id="user-a")))
  asyncio.run(writer.write(_entry("entry-b", user_id="user-b")))
  a_path = writer._segment_path(_TS, 0, user_id="user-a")
  b_dir = writer._owner_directory("user-b")
  assert a_path.parent.name == owner_key("user-a")
  assert a_path.parent.name == "u-" + hashlib.sha256(b"user-a").hexdigest()
  assert "/" not in a_path.parent.name

  _seal(b_dir)
  try:
    page, cursor = asyncio.run(writer.query(owner_user_id="user-a", limit=1, order="asc"))
    assert cursor is not None
    with pytest.raises(ValueError, match="cursor"):
      asyncio.run(writer.query(owner_user_id="user-b", limit=1, order="asc", cursor=cursor))
    rows, _cursor = asyncio.run(writer.query(owner_user_id="user-a", limit=10, order="asc"))
  finally:
    _unseal(b_dir)

  assert [row.entry_id for row in page] == ["entry-a"]
  assert [row.entry_id for row in rows] == ["entry-a", "entry-a2"]
  assert [row.user_id for row in rows] == ["user-a", "user-a"]


def test_retention_deletes_only_the_requesting_owners_segment(tmp_path: Path) -> None:
  writer = JSONLAuditWriter(tmp_path)
  asyncio.run(writer.write(_entry("entry-a", user_id="user-a", ts=_OLD, retention_class="dev")))
  asyncio.run(writer.write(_entry("entry-b", user_id="user-b", ts=_OLD, retention_class="dev")))
  a_path = writer._segment_path(_OLD, 0, user_id="user-a")
  b_path = writer._segment_path(_OLD, 0, user_id="user-b")
  b_dir = b_path.parent

  _seal(b_dir)
  try:
    deleted = asyncio.run(writer.apply_retention(owner_user_id="user-a"))
  finally:
    _unseal(b_dir)

  assert deleted == 1
  assert not a_path.exists()
  assert b_path.is_file()


def test_export_packs_none_of_another_owners_records(tmp_path: Path) -> None:
  writer = JSONLAuditWriter(tmp_path)
  asyncio.run(writer.write(_entry("entry-a", user_id="user-a")))
  asyncio.run(writer.write(_entry("entry-b", user_id="user-b")))
  b_dir = writer._owner_directory("user-b")

  _seal(b_dir)
  try:
    archive = asyncio.run(writer.export_for_legal_hold(
      owner_user_id="user-a",
      case_id="case-a",
      filter={},
      destination=str(tmp_path / "exports"),
    ))
  finally:
    _unseal(b_dir)

  packed = _archive_bytes(archive)
  assert b"entry-a" in packed
  assert b"entry-b" not in packed
  assert b"user-b" not in packed


def test_legacy_mixed_file_is_filtered_and_not_deleted(tmp_path: Path) -> None:
  writer = JSONLAuditWriter(tmp_path)
  legacy = tmp_path / "2019-01-01.jsonl"
  a_line = json.dumps(
    _entry("legacy-a", user_id="user-a", ts=_OLD, retention_class="dev").to_json_dict(),
    sort_keys=True,
  )
  b_line = json.dumps(
    _entry("legacy-b", user_id="user-b", ts=_OLD, retention_class="dev").to_json_dict(),
    sort_keys=True,
  )
  original = a_line + "\n" + b_line + "\n"
  legacy.write_text(original, encoding="utf-8")
  # Another owner's partition exists and must stay unopened.
  asyncio.run(writer.write(_entry("live-b", user_id="user-b", ts=_OLD, retention_class="dev")))
  b_dir = writer._owner_directory("user-b")

  _seal(b_dir)
  try:
    rows, _cursor = asyncio.run(writer.query(owner_user_id="user-a", order="asc"))
    archive = asyncio.run(writer.export_for_legal_hold(
      owner_user_id="user-a",
      case_id="case-legacy",
      filter={},
      destination=str(tmp_path / "exports"),
    ))
    deleted = asyncio.run(writer.apply_retention(owner_user_id="user-a"))
  finally:
    _unseal(b_dir)

  assert [row.entry_id for row in rows] == ["legacy-a"]
  assert deleted == 0
  assert legacy.read_text(encoding="utf-8") == original

  member_name = "legacy/2019-01-01.jsonl"
  with tarfile.open(archive, "r:gz") as tar:
    extracted = tar.extractfile(member_name)
    assert extracted is not None
    member = extracted.read()
    manifest_file = tar.extractfile("manifest.json")
    assert manifest_file is not None
    manifest = json.loads(manifest_file.read())
  assert b"legacy-a" in member
  assert b"legacy-b" not in member
  assert b"user-b" not in member
  digest = hashlib.sha256(member).hexdigest()
  matched = [item for item in manifest["files"] if item["path"] == str(legacy)]
  assert matched == [{"path": str(legacy), "sha256": digest}]


def test_legacy_owner_only_file_follows_retention_rule(tmp_path: Path) -> None:
  writer = JSONLAuditWriter(tmp_path)
  legacy = tmp_path / "2019-06-01.jsonl"
  line = json.dumps(
    _entry("only-a", user_id="user-a", ts=_OLD, retention_class="dev").to_json_dict(),
    sort_keys=True,
  )
  legacy.write_text(line + "\n", encoding="utf-8")

  assert asyncio.run(writer.apply_retention(owner_user_id="user-a")) == 1
  assert not legacy.exists()


def test_blank_owner_raises(tmp_path: Path) -> None:
  writer = JSONLAuditWriter(tmp_path)
  for owner in ("", "   "):
    with pytest.raises(ValueError):
      asyncio.run(writer.query(owner_user_id=owner))
    with pytest.raises(ValueError):
      asyncio.run(writer.apply_retention(owner_user_id=owner))
    with pytest.raises(ValueError):
      asyncio.run(writer.export_for_legal_hold(
        owner_user_id=owner,
        case_id="case-blank",
        filter={},
        destination=str(tmp_path / "exports"),
      ))


def _archive_bytes(archive: str) -> bytes:
  chunks: list[bytes] = []
  with tarfile.open(archive, "r:gz") as tar:
    for member in tar.getmembers():
      if not member.isfile():
        continue
      extracted = tar.extractfile(member)
      assert extracted is not None
      chunks.append(extracted.read())
  return b"\n".join(chunks)
