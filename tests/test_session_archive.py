from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from agent_gateway import session_archive
from agent_gateway.tool_result_spill import (
  SPILL_LANE_FILE_TOOLS,
  SpillCapabilities,
  SpillSink,
  write_spill_set,
)


def _json(path: Path, value: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _event(value: dict[str, Any], seq: int = 1) -> dict[str, Any]:
  return {"seq": seq, "timestamp": "2026-10-16T00:00:00+00:00", "event": value}


@pytest.fixture
def archive_tree(tmp_path: Path) -> dict[str, Any]:
  session_id = "sess_example"
  records = tmp_path / "records"
  chat = records / "state/logs/chat"
  sessions = records / "state/logs/sessions"
  auto = tmp_path / "autonomous"
  auto.mkdir()
  surface = tmp_path / "surface_content.jsonl"
  prompts = ["First system prompt.\n", "Second prompt — changed mid-session.\n"]
  hashes = [f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}" for text in prompts]
  _jsonl(chat / f"{session_id}.jsonl", [{"role": "user", "content": "hello"}])
  _json(chat / f"{session_id}.meta.json", {"user_id": "alice"})
  root = tmp_path / "spill-root"
  root.mkdir(mode=0o700)
  payload = json.dumps({"result": "λ" * 65001}, ensure_ascii=False).encode("utf-8")
  spill = root / "result.json"
  spill.write_bytes(payload)
  ref = f"spill:v1:{spill.name}:{hashlib.sha256(payload).hexdigest()}"
  active = sessions / "analyst" / f"agentsess_{session_id}--openai-era_alice.jsonl"
  _jsonl(active, [
    *[_event({"type": "context_manifest", "system_prompt_hash": value}, seq) for seq, value in enumerate(hashes, 1)],
    _event({"type": "tool_call_complete", "spill_publication": {"spill_ref": ref, "producer_root": str(root)}}, 3),
  ])
  _json(active.with_suffix(".meta.json"), {"user_id": "alice"})
  _jsonl(surface, [{"content_hash": key, "surface_type": "rendered_system_prompt", "content": value} for key, value in zip(hashes, prompts)])
  for run_id, parent in (("bg_1", session_id), ("bg_2", None)):
    _json(auto / f"{run_id}.task.json", {
      "session_id": run_id,
      "user_id": "alice",
      "admitted_context": {"lineage": {"parent_job_id": parent}} if parent else None,
    })
    _jsonl(auto / f"{run_id}.events.jsonl", [{"type": "text_delta", "text": "child output"}])
    (auto / f"{run_id}.log").write_text("child log\n", encoding="utf-8")
    _jsonl(auto / f"{run_id}.operator-messages.jsonl", [{"text": "continue"}])
  return dict(session_id=session_id, records=records, sessions=sessions, auto=auto, surface=surface, prompts=prompts, hashes=hashes, payload=payload, ref=ref, active=active, root=root)


def _archive(tree: dict[str, Any], output: Path, **kwargs: Any) -> session_archive.ArchiveResult:
  return session_archive.archive_session(
    tree["session_id"], output,
    records_root=tree["records"], autonomous_log_dir=tree["auto"],
    surface_content_path=tree["surface"], **kwargs,
  )


def test_archive_restores_prompts_spills_and_children(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive_tree: dict[str, Any],
) -> None:
  # Force actual sink-free read_spill_result pagination, rather than mocking it.
  monkeypatch.setattr(session_archive, "SPILL_READ_DEFAULT_FILE_MAX_BYTES", 32_000)
  result = _archive(archive_tree, tmp_path / "output")
  assert result.output_dir == tmp_path / "output" / archive_tree["session_id"]
  manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
  assert manifest["session_id"] == archive_tree["session_id"]
  assert result.files_archived == len(manifest["files"])
  for entry in manifest["files"]:
    data = (result.output_dir / entry["arcname"]).read_bytes()
    assert entry["sha256"] == hashlib.sha256(data).hexdigest()
    assert entry["bytes"] == len(data)
    assert Path(entry["source_path"]).is_absolute()
  for key, text in zip(archive_tree["hashes"], archive_tree["prompts"]):
    assert (result.output_dir / "prompts" / f"{key.removeprefix('sha256:')}.txt").read_bytes() == text.encode("utf-8")
  assert set(result.prompts) == {(key, True) for key in archive_tree["hashes"]}
  assert (result.output_dir / "spills/result.json").read_bytes() == archive_tree["payload"]
  assert result.spills == [(archive_tree["ref"], "spills/result.json", False)]
  for suffix in ("task.json", "events.jsonl", "log", "operator-messages.jsonl"):
    assert (result.output_dir / "autonomous/bg_1" / f"bg_1.{suffix}").read_bytes() == (archive_tree["auto"] / f"bg_1.{suffix}").read_bytes()
  assert not (result.output_dir / "autonomous/unknown-parent").exists()
  assert any("Unknown parent" in value and "bg_2" in value for value in result.warnings)
  assert (result.output_dir / "session" / f"{archive_tree['session_id']}.jsonl").is_file()
  assert (result.output_dir / "session/analyst" / archive_tree["active"].name).is_file()


def test_archive_orphans_are_explicitly_bucketed(tmp_path: Path, archive_tree: dict[str, Any]) -> None:
  result = _archive(archive_tree, tmp_path / "output", include_orphans=True)
  for suffix in ("task.json", "events.jsonl", "log", "operator-messages.jsonl"):
    assert (result.output_dir / "autonomous/unknown-parent" / f"bg_2.{suffix}").is_file()
  assert not (result.output_dir / "autonomous/bg_2").exists()


def test_archive_copies_all_provider_eras_and_segments(tmp_path: Path, archive_tree: dict[str, Any]) -> None:
  second = archive_tree["active"].with_name(f"agentsess_{archive_tree['session_id']}--openai-other_alice.jsonl")
  _jsonl(second, [_event({"type": "text", "text": "other era"})])
  segments = second.with_name(f"{second.stem}.segments")
  _jsonl(segments / "000001.jsonl", [_event({"type": "text", "text": "rotated"})])
  _json(segments / "manifest.json", {"segments": ["000001.jsonl"]})
  result = _archive(archive_tree, tmp_path / "output")
  assert (result.output_dir / "session/analyst" / second.name).is_file()
  assert (result.output_dir / "session/analyst" / segments.name / "000001.jsonl").is_file()
  assert (result.output_dir / "session/analyst" / segments.name / "manifest.json").is_file()


def test_archive_missing_sources_are_warnings(tmp_path: Path, archive_tree: dict[str, Any]) -> None:
  archive_tree["surface"].unlink()
  (archive_tree["root"] / "result.json").unlink()
  result = _archive(archive_tree, tmp_path / "output")
  assert all(not present for _, present in result.prompts)
  assert result.spills == [(archive_tree["ref"], None, True)]
  assert any("Lost spill" in value for value in result.warnings)
  assert result.manifest_path.is_file()


def test_archive_missing_transcript_ambiguous_users(tmp_path: Path, archive_tree: dict[str, Any]) -> None:
  chat = archive_tree["records"] / "state/logs/chat"
  for path in chat.iterdir():
    path.unlink()
  other = archive_tree["active"].with_name(f"agentsess_{archive_tree['session_id']}_bob.jsonl")
  _jsonl(other, [_event({"type": "text", "text": "not Alice"})])
  result = _archive(archive_tree, tmp_path / "output")
  assert any("Ambiguous" in value and "alice" in value and "bob" in value for value in result.warnings)
  assert not (result.output_dir / "session/analyst").exists()


def test_archive_reconstructs_file_tools_manifest(tmp_path: Path, archive_tree: dict[str, Any]) -> None:
  sink = SpillSink(
    root_provider=lambda: str(archive_tree["root"]),
    capabilities=SpillCapabilities(file_read=True),
  )
  assert sink.lane == SPILL_LANE_FILE_TOOLS
  payload = {"result": "large result\\n" * 6000}
  # A ~200-char JSON-quoted tmp path costs 6 * 200 + 1024 + 2000 = 4224 chars.
  # min(48_000, 20_000 * 0.8) - 4224 = 11_776, well above the 768-char minimum.
  publication = write_spill_set(
    sink=sink, tool_name="lookup", tool_use_id="call_1",
    content=json.dumps(payload), model_max_chars=20_000,
  )
  _jsonl(archive_tree["active"], [_event({
    "type": "tool_call_complete",
    "spill_publication": {"spill_ref": publication.spill_ref, "producer_root": str(archive_tree["root"])},
  })])
  result = _archive(archive_tree, tmp_path / "output")
  ref, arcname, lost = result.spills[0]
  assert ref == publication.spill_ref and not lost and arcname is not None
  assert json.loads((result.output_dir / arcname).read_text(encoding="utf-8")) == payload


def test_archive_collects_child_v1_and_linked_v2_streams(tmp_path: Path, archive_tree: dict[str, Any]) -> None:
  child_v1 = archive_tree["sessions"] / "autonomous/agentsess_bg_1--openai-era_alice.jsonl"
  _jsonl(child_v1, [_event({"type": "session_recap", "session_id": "bg_1--openai-era"})])
  v2_dir = archive_tree["sessions"] / ".session-log-v2/s-example"
  v2_active = v2_dir / "agentsess_s-example_alice.jsonl"
  _json(v2_active.with_suffix(".meta.json"), {"schema_version": 2, "user_id": "alice"})
  _jsonl(v2_active, [_event({"type": "text", "text": "active"})])
  v2_segments = v2_active.with_name(f"{v2_active.stem}.segments")
  _jsonl(v2_segments / "000001.jsonl", [_event({"type": "session_recap", "session_id": "bg_1"})])
  unrelated = v2_dir.parent / "s-unrelated/agentsess_s-unrelated_alice.jsonl"
  _jsonl(unrelated, [_event({"type": "session_recap", "session_id": "bg_99"})])
  _json(unrelated.with_suffix(".meta.json"), {"schema_version": 2, "user_id": "alice"})
  result = _archive(archive_tree, tmp_path / "output")
  assert (result.output_dir / "session/autonomous" / child_v1.name).is_file()
  assert (result.output_dir / "session/.session-log-v2/s-example" / v2_active.name).is_file()
  assert (result.output_dir / "session/.session-log-v2/s-example" / v2_segments.name / "000001.jsonl").is_file()
  assert not (result.output_dir / "session/.session-log-v2/s-unrelated").exists()


def test_archive_discovers_user_without_transcript(tmp_path: Path, archive_tree: dict[str, Any]) -> None:
  for path in (archive_tree["records"] / "state/logs/chat").iterdir():
    path.unlink()
  result = _archive(archive_tree, tmp_path / "output")
  assert (result.output_dir / "session/analyst" / archive_tree["active"].name).is_file()
  assert not any("Ambiguous" in value for value in result.warnings)


def test_archive_resolves_chat_logs_directly_under_gateway_log_dir(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive_tree: dict[str, Any],
) -> None:
  gateway_log_dir = tmp_path / "gateway-logs"
  chat_logs = gateway_log_dir / "chat_logs"
  session_id = archive_tree["session_id"]
  _jsonl(chat_logs / f"{session_id}.jsonl", [{"role": "user", "content": "from gateway_log_dir"}])
  _json(chat_logs / f"{session_id}.meta.json", {"user_id": "alice"})
  monkeypatch.delenv("GATEWAY_CHAT_LOG_DIR", raising=False)
  monkeypatch.setenv("GATEWAY_LOG_DIR", str(gateway_log_dir))
  result = session_archive.archive_session(
    session_id,
    tmp_path / "output",
    session_log_base=archive_tree["sessions"],
    autonomous_log_dir=archive_tree["auto"],
    surface_content_path=archive_tree["surface"],
  )
  assert (result.output_dir / "session" / f"{session_id}.jsonl").is_file()
  assert (result.output_dir / "session" / f"{session_id}.meta.json").is_file()
  assert not any("logs/chat_logs" in value for value in result.warnings)
