from __future__ import annotations

import errno
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from api import local_tools  # noqa: E402
from agent_gateway.tool_result_compaction import compact_model_tool_result_entry  # noqa: E402
import agent_gateway.tool_result_spill as spill_module  # noqa: E402
from agent_gateway.tool_result_spill import (  # noqa: E402
  SPILL_ROOT_CONTROL_FILES,
  SpillBudget,
  SpillCapabilities,
  SpillError,
  SpillLimitExceeded,
  SpillSink,
  reconstruct_spill_manifest,
  write_spill_set,
)


class _Logger:
  def __init__(self) -> None:
    self.warnings: list[tuple[str, tuple[Any, ...]]] = []

  def info(self, _message: str, *_args: Any, **_kwargs: Any) -> None:
    return None

  def warning(self, message: str, *args: Any, **_kwargs: Any) -> None:
    self.warnings.append((message, args))


def _sink(
  root: Path,
  *,
  capabilities: SpillCapabilities | None = None,
  budget: SpillBudget | None = None,
  max_file_bytes: int | None = None,
) -> SpillSink:
  return SpillSink(
    root_provider=lambda: str(root),
    capabilities=capabilities or SpillCapabilities(file_read=True, file_grep=True),
    budget=budget or SpillBudget(),
    max_file_bytes=max_file_bytes,
  )


def _disk_spill_bytes(root: Path) -> int:
  return sum(
    path.stat().st_size
    for path in root.iterdir()
    if path.name not in SPILL_ROOT_CONTROL_FILES and path.is_file() and not path.is_symlink()
  )


def _reader_json(result: dict[str, Any]) -> str:
  return json.dumps(result, default=str)


def test_multiline_json_sidecar_round_trips_through_real_readers(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("AGENT_GATEWAY_MAX_MODEL_TOOL_RESULT_CHARS", "60000")
  root = tmp_path / "spill-😀"
  root.mkdir()
  lines = [f"line-{index:04d} {'x' * 80}" for index in range(2_000)]
  lines[-3] = "LATE_SENTINEL exact fact beyond the first sixty thousand characters"
  markdown = "\n".join(lines)
  content = json.dumps({"status": "success", "nested": {"markdown": markdown}})

  publication = write_spill_set(
    sink=_sink(root),
    tool_name="thesis_read",
    tool_use_id="tool-1",
    content=content,
    model_max_chars=60_000,
  )

  manifest_path = Path(publication.abspath)
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  sidecar_name = next(member["filename"] for member in manifest["members"] if member["role"] == "sidecar")
  sidecar = root / sidecar_name
  assert sidecar.suffix == ".md"
  assert sidecar.read_text(encoding="utf-8") == markdown
  assert reconstruct_spill_manifest(manifest_path) == json.loads(content)

  read_result, read_error = local_tools.file_read(
    {"file_path": str(sidecar), "offset": len(lines) - 3, "limit": 3}
  )
  assert read_error is None
  assert read_result is not None
  assert "LATE_SENTINEL" in read_result["content"]
  assert len(_reader_json(read_result)) <= 60_000

  grep_result, grep_error = local_tools.file_grep(
    {
      "pattern": "LATE_SENTINEL",
      "path": str(sidecar),
      "max_results": 1,
      "context_lines": 1,
    }
  )
  assert grep_error is None
  assert grep_result is not None
  assert grep_result["matches"][0]["line"].startswith("LATE_SENTINEL")
  serialized_grep = _reader_json(grep_result)
  assert len(serialized_grep) <= 60_000

  entry = {"type": "tool_result", "tool_use_id": "reader", "content": serialized_grep}
  live, durable = compact_model_tool_result_entry(
    entry,
    tool_name="file_grep",
    log_session_id="reader-proof",
    logger=_Logger(),
  )
  assert live is entry
  assert durable is entry


def test_single_line_megabyte_json_uses_bounded_chunks_and_round_trips(
  tmp_path: Path,
) -> None:
  root = tmp_path / "emoji-😀-spill"
  root.mkdir()
  markdown = ("😀abc" * 270_000) + " LATE_MEGABYTE_SENTINEL"
  content = json.dumps({"markdown": markdown})

  publication = write_spill_set(
    sink=_sink(root),
    tool_name="thesis_read",
    tool_use_id="one-line",
    content=content,
    model_max_chars=60_000,
  )

  manifest_path = Path(publication.abspath)
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  chunk_name = next(member["filename"] for member in manifest["members"] if member["role"] == "chunks")
  chunk_path = root / chunk_name
  chunk_lines = chunk_path.read_text(encoding="utf-8").splitlines()
  assert len(chunk_lines) > 100
  assert reconstruct_spill_manifest(manifest_path) == json.loads(content)

  line_budget = spill_module._reader_line_budget(chunk_path, model_max_chars=60_000)
  assert max(spill_module._reader_encoded_line_chars(line) for line in chunk_lines) <= line_budget

  grep_result, grep_error = local_tools.file_grep(
    {
      "pattern": "LATE_MEGABYTE_SENTINEL",
      "path": str(chunk_path),
      "max_results": 1,
      "context_lines": 1,
    }
  )
  assert grep_error is None
  assert grep_result is not None
  assert grep_result["count"] == 1
  assert len(_reader_json(grep_result)) <= 60_000

  final_line = int(grep_result["matches"][0]["line_number"])
  read_result, read_error = local_tools.file_read(
    {"file_path": str(chunk_path), "offset": final_line, "limit": 1}
  )
  assert read_error is None
  assert read_result is not None
  assert "LATE_MEGABYTE_SENTINEL" in read_result["content"]
  assert len(_reader_json(read_result)) <= 60_000


@pytest.mark.parametrize(
  "value",
  [
    "literal\\n versus real\nnewline",
    "windows\r\nline endings\r\n",
    "unicode separators \u2028 and \u2029 survive",
    "emoji 😀 offsets are code points, not UTF-8 bytes",
    "trailing newline\n",
    "",
  ],
)
def test_chunk_codec_round_trips_exact_unicode_code_points(value: str) -> None:
  encoded = spill_module._encode_chunk_records(value, "c0-deadbeef", 220)

  assert spill_module._decode_chunk_records(encoded, "c0-deadbeef") == value
  assert all(
    spill_module._reader_encoded_line_chars(line) <= 220
    for line in encoded.splitlines()
  )


@pytest.mark.parametrize(
  "capabilities",
  [
    SpillCapabilities(code_execute=True),
    SpillCapabilities(file_read=True),
    SpillCapabilities(file_grep=True),
    SpillCapabilities(),
  ],
)
def test_non_json_spills_preserve_exact_utf8_bytes_in_every_lane(
  tmp_path: Path,
  capabilities: SpillCapabilities,
) -> None:
  root = tmp_path / capabilities.lane
  root.mkdir()
  content = "first\r\nliteral\\n 😀\nlast\n"

  publication = write_spill_set(
    sink=_sink(root, capabilities=capabilities),
    tool_name="plain",
    tool_use_id="bytes",
    content=content,
    model_max_chars=60_000,
  )

  if capabilities.file_read or capabilities.file_grep:
    manifest_path = Path(publication.abspath)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical_name = next(member["filename"] for member in manifest["members"] if member["role"] == "canonical")
    canonical = root / canonical_name
    assert reconstruct_spill_manifest(manifest_path) == content
  else:
    canonical = Path(publication.abspath)
  assert canonical.read_bytes() == content.encode("utf-8")


@pytest.mark.parametrize(
  ("capabilities", "present", "absent"),
  [
    (SpillCapabilities(code_execute=True), ("code_execute",), ("file_grep(pattern=",)),
    (SpillCapabilities(file_read=True), ("file_read(file_path=",), ("file_grep(pattern=", "code_execute")),
    (SpillCapabilities(file_grep=True), ("file_grep(pattern=",), ("file_read(file_path=", "code_execute")),
    (
      SpillCapabilities(file_read=True, file_grep=True),
      ("file_read(file_path=", "file_grep(pattern="),
      ("code_execute",),
    ),
    (SpillCapabilities(), ("no file/code reader",), ("file_read(file_path=", "file_grep(pattern=", "code_execute")),
  ],
)
def test_spill_hints_name_only_effective_reader_capabilities(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capabilities: SpillCapabilities,
  present: tuple[str, ...],
  absent: tuple[str, ...],
) -> None:
  monkeypatch.setenv("AGENT_GATEWAY_MAX_MODEL_TOOL_RESULT_CHARS", "60000")
  root = tmp_path / capabilities.lane
  root.mkdir()
  content = json.dumps({"markdown": "x" * 80_000})
  entry = {"type": "tool_result", "tool_use_id": "hint", "content": content}

  live, durable = compact_model_tool_result_entry(
    entry,
    tool_name="lookup",
    spill_sink=_sink(root, capabilities=capabilities),
    log_session_id="hint-proof",
    logger=_Logger(),
  )

  live_payload = json.loads(live["content"])
  hint = live_payload["spill_hint"]
  for expected in present:
    assert expected in hint
  for unexpected in absent:
    assert unexpected not in hint
  assert "spill_file" not in json.loads(durable["content"])
  if capabilities.file_read or capabilities.file_grep:
    assert live_payload["spill_file"].endswith(".manifest.json")
    assert set(live_payload["spill_summary"]) == {"member_count", "total_chars", "largest_members"}
    assert len(live_payload["spill_summary"]["largest_members"]) <= 3


def test_shared_budget_is_atomic_under_concurrent_file_set_writers(tmp_path: Path) -> None:
  worker_count = 12
  budget = SpillBudget(max_bytes=120_000)
  sink = _sink(tmp_path, budget=budget)
  barrier = threading.Barrier(worker_count)
  content = json.dumps({"markdown": "\n".join(f"line-{index} {'x' * 80}" for index in range(180))})

  def _write(index: int) -> bool:
    barrier.wait()
    try:
      write_spill_set(
        sink=sink,
        tool_name="bulk",
        tool_use_id=f"tool-{index}",
        content=content,
        model_max_chars=60_000,
      )
      return True
    except SpillLimitExceeded:
      return False

  with ThreadPoolExecutor(max_workers=worker_count) as pool:
    results = list(pool.map(_write, range(worker_count)))

  assert any(results)
  assert not all(results)
  assert budget.used_bytes <= 120_000
  assert budget.used_bytes == _disk_spill_bytes(tmp_path)
  assert len(list(tmp_path.glob("*.manifest.json"))) == sum(results)


def test_budget_rehydrates_only_committed_sets_and_removes_orphans(tmp_path: Path) -> None:
  first_sink = _sink(tmp_path, budget=SpillBudget(max_bytes=2_000_000))
  write_spill_set(
    sink=first_sink,
    tool_name="first",
    tool_use_id="one",
    content=json.dumps({"markdown": "\n".join(["a" * 100] * 100)}),
    model_max_chars=60_000,
  )
  (tmp_path / "orphan.index.json").write_text("orphan", encoding="utf-8")
  (tmp_path / ".orphan.tmp").write_text("temp", encoding="utf-8")
  (tmp_path / ".lease").write_text("control", encoding="utf-8")
  (tmp_path / ".spill_fresh").write_text("control", encoding="utf-8")

  rehydrated = SpillBudget(max_bytes=2_000_000)
  second_sink = _sink(tmp_path, budget=rehydrated)
  write_spill_set(
    sink=second_sink,
    tool_name="second",
    tool_use_id="two",
    content=json.dumps({"markdown": "\n".join(["b" * 100] * 100)}),
    model_max_chars=60_000,
  )

  assert not (tmp_path / "orphan.index.json").exists()
  assert not (tmp_path / ".orphan.tmp").exists()
  assert (tmp_path / ".lease").read_text(encoding="utf-8") == "control"
  assert (tmp_path / ".spill_fresh").read_text(encoding="utf-8") == "control"
  assert rehydrated.used_bytes == _disk_spill_bytes(tmp_path)
  assert len(list(tmp_path.glob("*.manifest.json"))) == 2


@pytest.mark.parametrize("collision_role", ["index", "sidecar", "manifest"])
def test_file_set_collision_reformats_and_republishes_one_fresh_stem(
  tmp_path: Path,
  collision_role: str,
) -> None:
  sink = _sink(tmp_path)
  sink.budget.reserve(tmp_path, sink.lane, 0)
  stem = "lookup_tool-1"
  pointer_hash = hashlib.sha256(b"/markdown").hexdigest()[:8]
  collision_names = {
    "index": f"{stem}.index.json",
    "sidecar": f"{stem}.s0.{pointer_hash}.md",
    "manifest": f"{stem}.manifest.json",
  }
  collision = tmp_path / collision_names[collision_role]
  collision.write_bytes(b"old")
  content = json.dumps({"markdown": "\n".join(["line" * 40] * 150)})

  publication = write_spill_set(
    sink=sink,
    tool_name="lookup",
    tool_use_id="tool-1",
    content=content,
    model_max_chars=60_000,
    uuid_factory=lambda: SimpleNamespace(hex="cafebabedeadbeef"),
  )

  assert collision.read_bytes() == b"old"
  assert publication.filename == "lookup_tool-1_cafebabe.manifest.json"
  assert reconstruct_spill_manifest(publication.abspath) == json.loads(content)
  assert not (tmp_path / f"{stem}.index.json").exists() or collision_role == "index"


def test_concurrent_same_tool_id_never_clobbers_a_committed_set(tmp_path: Path) -> None:
  sink = _sink(tmp_path)
  barrier = threading.Barrier(2)
  content = json.dumps({"markdown": "\n".join(["same" * 40] * 150)})

  def _write() -> str:
    barrier.wait()
    return write_spill_set(
      sink=sink,
      tool_name="lookup",
      tool_use_id="same-id",
      content=content,
      model_max_chars=60_000,
    ).abspath

  with ThreadPoolExecutor(max_workers=2) as pool:
    manifests = list(pool.map(lambda _index: _write(), range(2)))

  assert len(set(manifests)) == 2
  assert all(reconstruct_spill_manifest(path) == json.loads(content) for path in manifests)
  assert sink.budget.used_bytes == _disk_spill_bytes(tmp_path)


def test_partial_set_failure_removes_owned_members_and_rolls_back_budget(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  sink = _sink(tmp_path)
  sink.budget.reserve(tmp_path, sink.lane, 0)
  original_publish = spill_module._publish_no_clobber

  def _fail_sidecar(root: Path, final_path: Path, data: bytes, *, uuid_factory: Any):
    if ".s0." in final_path.name:
      raise OSError("injected sidecar failure")
    return original_publish(root, final_path, data, uuid_factory=uuid_factory)

  monkeypatch.setattr(spill_module, "_publish_no_clobber", _fail_sidecar)
  content = json.dumps({"markdown": "\n".join(["line" * 40] * 150)})

  with pytest.raises(OSError, match="injected sidecar failure"):
    write_spill_set(
      sink=sink,
      tool_name="lookup",
      tool_use_id="partial",
      content=content,
      model_max_chars=60_000,
    )

  assert list(tmp_path.iterdir()) == []
  assert sink.budget.used_bytes == 0


def test_partial_cleanup_never_unlinks_a_replaced_foreign_member(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  sink = _sink(tmp_path)
  sink.budget.reserve(tmp_path, sink.lane, 0)
  original_publish = spill_module._publish_no_clobber
  first_path: Path | None = None
  first_publication: Any = None

  def _replace_then_fail(root: Path, final_path: Path, data: bytes, *, uuid_factory: Any):
    nonlocal first_path, first_publication
    if first_path is None:
      first_publication = original_publish(root, final_path, data, uuid_factory=uuid_factory)
      first_path = final_path
      return first_publication
    assert first_path is not None
    first_path.unlink()
    pinned = spill_module.os.fstat(first_publication.guard_fd)
    assert (pinned.st_dev, pinned.st_ino) == first_publication.identity
    first_path.write_text("foreign replacement", encoding="utf-8")
    raise OSError("injected failure after replacement")

  monkeypatch.setattr(spill_module, "_publish_no_clobber", _replace_then_fail)
  content = json.dumps({"markdown": "\n".join(["line" * 40] * 150)})

  with pytest.raises(OSError, match="after replacement"):
    write_spill_set(
      sink=sink,
      tool_name="lookup",
      tool_use_id="replacement",
      content=content,
      model_max_chars=60_000,
    )

  assert first_path is not None
  assert first_path.read_text(encoding="utf-8") == "foreign replacement"
  with pytest.raises(OSError):
    spill_module.os.fstat(first_publication.guard_fd)
  assert sink.budget.used_bytes == 0


def test_publication_guard_failure_removes_the_linked_member(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  sink = _sink(tmp_path)
  sink.budget.reserve(tmp_path, sink.lane, 0)
  original_open = spill_module.os.open

  def _fail_guard(path: Any, flags: int, *args: Any, **kwargs: Any):
    if flags & spill_module.os.O_ACCMODE == spill_module.os.O_RDONLY:
      raise OSError("injected guard failure")
    return original_open(path, flags, *args, **kwargs)

  monkeypatch.setattr(spill_module.os, "open", _fail_guard)

  with pytest.raises(OSError, match="injected guard failure"):
    write_spill_set(
      sink=sink,
      tool_name="lookup",
      tool_use_id="guard-failure",
      content=json.dumps({"markdown": "x" * 10_000}),
      model_max_chars=60_000,
    )

  assert list(tmp_path.iterdir()) == []
  assert sink.budget.used_bytes == 0


def test_hard_link_unsupported_falls_back_to_plain_truncation_without_temp_or_pointer(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("AGENT_GATEWAY_MAX_MODEL_TOOL_RESULT_CHARS", "4000")

  def _unsupported(*_args: Any, **_kwargs: Any) -> None:
    raise OSError(errno.EPERM, "hard links unsupported")

  monkeypatch.setattr(spill_module.os, "link", _unsupported)
  logger = _Logger()
  content = json.dumps({"payload": "x" * 10_000})
  entry = {"type": "tool_result", "tool_use_id": "no-link", "content": content}

  live, durable = compact_model_tool_result_entry(
    entry,
    tool_name="lookup",
    spill_sink=_sink(tmp_path, capabilities=SpillCapabilities(code_execute=True)),
    log_session_id="no-link",
    logger=logger,
  )

  assert live == durable
  assert "spill_file" not in json.loads(live["content"])
  assert list(tmp_path.iterdir()) == []
  assert logger.warnings


def test_per_file_cap_and_overlong_index_fail_before_publication(tmp_path: Path) -> None:
  capped_root = tmp_path / "capped"
  capped_root.mkdir()
  capped_sink = _sink(capped_root, max_file_bytes=100)
  with pytest.raises(SpillLimitExceeded, match="per-file cap"):
    write_spill_set(
      sink=capped_sink,
      tool_name="lookup",
      tool_use_id="capped",
      content=json.dumps({"payload": "x" * 1_000}),
      model_max_chars=60_000,
    )
  assert list(capped_root.iterdir()) == []
  assert capped_sink.budget.used_bytes == 0

  index_root = tmp_path / "index"
  index_root.mkdir()
  pathological = json.dumps({"k" * 50_000: "short", "payload": "x" * 70_000})
  with pytest.raises(SpillLimitExceeded, match="index contains an overlong reader line"):
    write_spill_set(
      sink=_sink(index_root),
      tool_name="lookup",
      tool_use_id="index",
      content=pathological,
      model_max_chars=60_000,
    )
  assert list(index_root.iterdir()) == []


def test_ambiguous_user_marker_is_detected_and_symlink_root_is_rejected(tmp_path: Path) -> None:
  marker_root = tmp_path / "marker"
  marker_root.mkdir()
  pointer_hash = hashlib.sha256(b"/markdown").hexdigest()[:8]
  ref_id = f"c0-{pointer_hash}"
  payload = {
    "markdown": "x" * 100_000,
    "legitimate_user_object": {"$spill_ref": ref_id},
  }
  with pytest.raises(SpillError, match="ambiguous spill ref marker"):
    write_spill_set(
      sink=_sink(marker_root),
      tool_name="lookup",
      tool_use_id="marker",
      content=json.dumps(payload),
      model_max_chars=60_000,
    )
  assert list(marker_root.iterdir()) == []

  target = tmp_path / "target"
  target.mkdir()
  linked = tmp_path / "linked"
  linked.symlink_to(target, target_is_directory=True)
  with pytest.raises(SpillError, match="non-symlink directory"):
    write_spill_set(
      sink=_sink(linked, capabilities=SpillCapabilities(code_execute=True)),
      tool_name="lookup",
      tool_use_id="symlink",
      content="x" * 100,
      model_max_chars=60_000,
    )
  assert list(target.iterdir()) == []


def test_disk_reconstruction_rejects_manifest_count_and_ref_metadata_tampering(tmp_path: Path) -> None:
  content = json.dumps({"markdown": "x" * 100_000})
  publication = write_spill_set(
    sink=_sink(tmp_path),
    tool_name="lookup",
    tool_use_id="tamper",
    content=content,
    model_max_chars=60_000,
  )
  manifest_path = Path(publication.abspath)
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  manifest["refs"][0]["chars"] += 1
  manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

  with pytest.raises(SpillError, match="ref character count mismatch"):
    reconstruct_spill_manifest(manifest_path)

  manifest["refs"][0]["chars"] -= 1
  manifest["member_count"] += 1
  manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
  with pytest.raises(SpillError, match="members missing"):
    reconstruct_spill_manifest(manifest_path)


def test_overlong_json_pointer_is_bounded_in_manifest_without_affecting_reconstruction(
  tmp_path: Path,
) -> None:
  leaf: Any = {"markdown": "\n".join(["value"] * 4_000)}
  key = "nested-segment-" + ("k" * 48)
  for _index in range(240):
    leaf = {key: leaf}
  content = json.dumps(leaf)

  publication = write_spill_set(
    sink=_sink(tmp_path),
    tool_name="lookup",
    tool_use_id="long-pointer",
    content=content,
    model_max_chars=60_000,
  )

  manifest = json.loads(Path(publication.abspath).read_text(encoding="utf-8"))
  ref = manifest["refs"][0]
  assert "pointer" not in ref
  assert len(ref["pointer_prefix"]) <= 256
  assert len(ref["pointer_sha256"]) == 64
  assert reconstruct_spill_manifest(publication.abspath) == leaf
