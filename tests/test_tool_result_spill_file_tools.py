from __future__ import annotations

import errno
import hashlib
import json
import sys
import threading
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.tool_result_compaction import compact_model_tool_result_entry  # noqa: E402
from agent_gateway.model_bound_wire import model_bound_wire_size  # noqa: E402
import agent_gateway.tool_result_spill as spill_module  # noqa: E402
from agent_gateway.tool_result_spill import (  # noqa: E402
  SPILL_ROOT_CONTROL_FILES,
  SpillBudget,
  SpillCapabilities,
  SpillError,
  SpillLimitExceeded,
  SpillReadAccessDenied,
  SpillReadInvalidInput,
  SpillReadUnavailable,
  SpillSink,
  make_tool_result_read_handler,
  read_spill_result,
  reconstruct_spill_manifest,
  write_spill_set,
  job_spill_sink,
)
from agent_gateway.admission import ParentContext  # noqa: E402
from agent_gateway.approval_route import NoApprovalRoute  # noqa: E402
from agent_workflow_contracts.models import (  # noqa: E402
  AdmittedResearchBinding,
  AdmittedTenant,
)
from agent_gateway.agent_session_log import AgentSessionLog  # noqa: E402
from gateway_test_support.resumer_custody_test_support import (  # noqa: E402
  canonical_read_custody_task,
  custody_registration,
)
from gateway_test_support.spill_custody_test_support import publish_custody_spill  # noqa: E402


def _job(job_id: str, *, handle_grants: frozenset[str] = frozenset()) -> ParentContext:
  return ParentContext(
    tenant=AdmittedTenant(tenant_id=None, storage_id="alice", role="owner"),
    research_binding=AdmittedResearchBinding(),
    children=None,
    tool_ids=None,
    withheld=frozenset(),
    handle_grants=handle_grants,
    remaining_usd=None,
    job_id=job_id,
    depth=0,
    approval_route=NoApprovalRoute(),
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
    capabilities=capabilities or SpillCapabilities(
      file_read=True,
      file_grep=True,
      spill_read=True,
    ),
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


def test_multiline_json_sidecar_round_trips_through_exact_sink_reader(
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

  sink = _sink(root)
  publication = write_spill_set(
    sink=sink,
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

  read_result = read_spill_result(
    sink,
    spill_ref=publication.spill_ref,
    member_ref=sidecar_name,
    offset=markdown.index("LATE_SENTINEL") - 100,
    max_chars=400,
  )
  assert "LATE_SENTINEL" in read_result["content"]
  assert len(_reader_json(read_result)) <= 60_000

  grep_result = read_spill_result(
    sink,
    spill_ref=publication.spill_ref,
    member_ref=sidecar_name,
    query="LATE_SENTINEL",
    max_results=1,
    context_chars=100,
  )
  assert "LATE_SENTINEL" in grep_result["matches"][0]["content"]
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

  sink = _sink(root)
  publication = write_spill_set(
    sink=sink,
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

  grep_result = read_spill_result(
    sink,
    spill_ref=publication.spill_ref,
    member_ref=chunk_name,
    query="LATE_MEGABYTE_SENTINEL",
    max_results=1,
    context_chars=100,
  )
  assert len(grep_result["matches"]) == 1
  assert len(_reader_json(grep_result)) <= 60_000

  read_result = read_spill_result(
    sink,
    spill_ref=publication.spill_ref,
    member_ref=chunk_name,
    offset=grep_result["matches"][0]["start_offset"],
    max_chars=500,
  )
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
  "capabilities",
  [
    SpillCapabilities(code_execute=True, spill_read=True),
    SpillCapabilities(file_read=True, spill_read=True),
    SpillCapabilities(file_grep=True, spill_read=True),
    SpillCapabilities(file_read=True, file_grep=True, spill_read=True),
    SpillCapabilities(),
  ],
)
def test_spill_ref_is_published_only_with_exact_reader_capability(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capabilities: SpillCapabilities,
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
  assert "spill_file" not in live_payload
  assert "spill_abspath" not in live_payload
  assert "spill_hint" not in live_payload
  assert "spill_ref" not in json.loads(durable["content"])
  if capabilities.spill_read:
    assert live_payload["spill_ref"].startswith("spill:v1:")
    assert set(live_payload["spill_summary"]) == {
      "payload_kind",
      "member_count",
      "total_chars",
      "largest_members",
    }
    assert len(live_payload["spill_summary"]["largest_members"]) <= 3
    root_page = read_spill_result(
      _sink(root, capabilities=capabilities),
      spill_ref=live_payload["spill_ref"],
      max_chars=200,
    )
    assert root_page["content"]
  else:
    assert "spill_ref" not in live_payload
    assert "spill_summary" not in live_payload


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


def test_rehydrated_budget_counts_retained_child_job_spills(tmp_path: Path) -> None:
  content = json.dumps({"markdown": "\n".join(["c" * 100] * 100)})
  first = _sink(tmp_path, budget=SpillBudget(max_bytes=2_000_000))
  child = job_spill_sink(first, "bg_0")
  write_spill_set(
    sink=child,
    tool_name="child",
    tool_use_id="one",
    content=content,
    model_max_chars=60_000,
  )
  child_bytes = _disk_spill_bytes(Path(child()))
  assert child_bytes > 0

  # A restarted run rebuilds its sink and budget over the same tree root; the
  # child's retained bytes already fill this budget.
  rehydrated = SpillBudget(max_bytes=child_bytes)
  restarted = _sink(tmp_path, budget=rehydrated)
  with pytest.raises(SpillLimitExceeded):
    write_spill_set(
      sink=job_spill_sink(restarted, "bg_1"),
      tool_name="next",
      tool_use_id="two",
      content=content,
      model_max_chars=60_000,
    )
  assert rehydrated.used_bytes == child_bytes


@pytest.mark.parametrize("linked", ["jobs", "job"])
def test_job_spill_reader_never_follows_a_symlinked_jobs_boundary(
  tmp_path: Path,
  linked: str,
) -> None:
  tree_a = tmp_path / "a"
  tree_b = tmp_path / "b"
  tree_a.mkdir()
  tree_b.mkdir()
  foreign = job_spill_sink(_sink(tree_a), "bg_0")
  spill_ref = _spill(foreign, "foreign-call")
  foreign_root = Path(foreign())
  if linked == "jobs":
    (tree_b / "jobs").symlink_to(tree_a / "jobs")
  else:
    (tree_b / "jobs").mkdir()
    (tree_b / "jobs" / foreign_root.name).symlink_to(foreign_root)
  # Job ids repeat across trees, so tree B's ``bg_0`` names A's directory.
  reader = job_spill_sink(_sink(tree_b), "bg_0")
  assert Path(reader()).name == foreign_root.name

  assert spill_module.resolve_spill_producer(reader, spill_ref) is None
  with pytest.raises(SpillReadUnavailable):
    read_spill_result(reader, spill_ref=spill_ref)
  with pytest.raises(SpillError):
    _spill(reader, "through-link")


def test_job_spill_reads_locate_without_creating_directories(tmp_path: Path) -> None:
  tree = tmp_path / "tree"
  elsewhere = tmp_path / "elsewhere"
  tree.mkdir()
  elsewhere.mkdir()
  spill_ref = _spill(_sink(elsewhere), "elsewhere-call")
  worker = job_spill_sink(job_spill_sink(_sink(tree), "bg_0"), "lead_0")

  assert spill_module.resolve_spill_producer(worker, spill_ref) is None
  with pytest.raises(SpillReadUnavailable):
    read_spill_result(worker, spill_ref=spill_ref)
  assert list(tree.iterdir()) == []


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

  # spill_module.os IS the global os module and the guard raises for every
  # O_RDONLY open — exactly what tmp_path retention rmtree does at teardown.
  # Scope the patch so cleanup never runs against it.
  with monkeypatch.context() as scoped:
    scoped.setattr(spill_module.os, "open", _fail_guard)

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
    spill_sink=_sink(
      tmp_path,
      capabilities=SpillCapabilities(code_execute=True, spill_read=True),
    ),
    log_session_id="no-link",
    logger=logger,
  )

  assert live == durable
  live_payload = json.loads(live["content"])
  assert "spill_file" not in live_payload
  assert "spill_ref" not in live_payload
  assert "spill_summary" not in live_payload
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


def test_exact_reader_denies_paths_control_files_symlinks_and_unlisted_members(
  tmp_path: Path,
) -> None:
  sink = _sink(tmp_path)
  publication = write_spill_set(
    sink=sink,
    tool_name="lookup",
    tool_use_id="deny",
    content=json.dumps({"markdown": "x" * 100_000}),
    model_max_chars=60_000,
  )
  (tmp_path / "unlisted.txt").write_text("not admitted", encoding="utf-8")
  (tmp_path / ".lease").write_text("control", encoding="utf-8")
  (tmp_path / "linked.txt").symlink_to(tmp_path / "unlisted.txt")

  with pytest.raises(SpillReadInvalidInput, match="malformed"):
    read_spill_result(sink, spill_ref="/tmp/not-a-ref")
  for member_ref in ("../unlisted.txt", ".lease", "linked.txt", "unlisted.txt"):
    with pytest.raises(SpillReadAccessDenied):
      read_spill_result(
        sink,
        spill_ref=publication.spill_ref,
        member_ref=member_ref,
      )

  pointer_path = Path(publication.abspath)
  pointer_path.unlink()
  pointer_path.symlink_to(tmp_path / "unlisted.txt")
  with pytest.raises(SpillReadAccessDenied):
    read_spill_result(sink, spill_ref=publication.spill_ref)


def test_exact_reader_fails_closed_on_missing_replaced_or_tampered_members(
  tmp_path: Path,
) -> None:
  pointer_root = tmp_path / "pointer"
  pointer_root.mkdir()
  pointer_sink = _sink(
    pointer_root,
    capabilities=SpillCapabilities(code_execute=True, spill_read=True),
  )
  pointer_publication = write_spill_set(
    sink=pointer_sink,
    tool_name="lookup",
    tool_use_id="pointer",
    content='{"value": "exact"}',
    model_max_chars=60_000,
  )
  Path(pointer_publication.abspath).write_text(
    '{"value": "replacement"}',
    encoding="utf-8",
  )
  with pytest.raises(SpillReadUnavailable, match="integrity mismatch"):
    read_spill_result(pointer_sink, spill_ref=pointer_publication.spill_ref)

  member_root = tmp_path / "member"
  member_root.mkdir()
  member_sink = _sink(member_root)
  member_publication = write_spill_set(
    sink=member_sink,
    tool_name="lookup",
    tool_use_id="member",
    content=json.dumps({"markdown": "x" * 100_000}),
    model_max_chars=60_000,
  )
  manifest = json.loads(Path(member_publication.abspath).read_text(encoding="utf-8"))
  member_ref = next(
    member["filename"]
    for member in manifest["members"]
    if member["role"] in {"sidecar", "chunks"}
  )
  (member_root / member_ref).write_text("tampered", encoding="utf-8")
  with pytest.raises(SpillReadUnavailable, match="integrity mismatch"):
    read_spill_result(
      member_sink,
      spill_ref=member_publication.spill_ref,
      member_ref=member_ref,
    )

  Path(member_publication.abspath).unlink()
  with pytest.raises(SpillReadUnavailable, match="missing or expired"):
    read_spill_result(member_sink, spill_ref=member_publication.spill_ref)


@pytest.mark.asyncio
async def test_exact_reader_handler_reports_expired_reference_without_fallback(
  tmp_path: Path,
) -> None:
  sink = _sink(
    tmp_path,
    capabilities=SpillCapabilities(code_execute=True, spill_read=True),
  )
  publication = write_spill_set(
    sink=sink,
    tool_name="lookup",
    tool_use_id="expired",
    content="expired content",
    model_max_chars=60_000,
  )
  Path(publication.abspath).unlink()

  result, error = await make_tool_result_read_handler(
    lambda: sink,
    caller=lambda: _job("root"),
    journal=lambda: None,
  )({
    "spill_ref": publication.spill_ref,
  })

  assert result is None
  assert error == {
    "code": "spill_unavailable",
    "message": "spill is missing or expired",
  }


def _spill(sink: SpillSink, tool_use_id: str) -> str:
  return write_spill_set(
    sink=sink,
    tool_name="lookup",
    tool_use_id=tool_use_id,
    content=f"content of {tool_use_id}",
    model_max_chars=60_000,
  ).spill_ref


@pytest.mark.asyncio
async def test_spill_reads_admit_ancestors_and_passed_handles_only(
  tmp_path: Path,
) -> None:
  root = _sink(
    tmp_path,
    capabilities=SpillCapabilities(code_execute=True, spill_read=True),
  )
  lead = job_spill_sink(root, "bg_0")
  worker = job_spill_sink(lead, "lead_0")
  sibling = job_spill_sink(root, "bg_1")
  root_ref = _spill(root, "root-call")
  worker_ref = _spill(worker, "worker-call")

  async def _read(sink: SpillSink, reader: ParentContext, spill_ref: str):
    return await make_tool_result_read_handler(
      lambda: sink,
      caller=lambda: reader,
      journal=lambda: None,
    )({"spill_ref": spill_ref})

  # Ancestors read a descendant's spill: the lead and the root.
  for sink, job_id in ((lead, "bg_0"), (root, "root")):
    result, error = await _read(sink, _job(job_id), worker_ref)
    assert error is None and result is not None
    assert "content of worker-call" in json.dumps(result)
  # A sibling, or a child reading its parent's spill, holds no grant.
  for sink, job_id, ref in ((sibling, "bg_1", worker_ref), (lead, "bg_0", root_ref)):
    result, error = await _read(sink, _job(job_id), ref)
    assert result is None and error is not None
    assert error["code"] == "handle_not_granted"
  # The sibling reads it once the handle was passed to it at spawn.
  result, error = await _read(
    sibling,
    _job("bg_1", handle_grants=frozenset({worker_ref})),
    worker_ref,
  )
  assert error is None and result is not None

@pytest.mark.asyncio
async def test_copied_root_reader_rebound_to_a_child_reads_as_that_child(
  tmp_path: Path,
) -> None:
  from agent_gateway.job_caller import JobScope, rebind_job_handlers

  root = _sink(
    tmp_path,
    capabilities=SpillCapabilities(code_execute=True, spill_read=True),
  )
  sibling = job_spill_sink(root, "bg_0")
  child = job_spill_sink(root, "bg_1")
  sibling_ref = _spill(sibling, "sibling-call")
  child_ref = _spill(child, "child-call")
  root_handlers = {
    "tool_result_read": make_tool_result_read_handler(
      lambda: root,
      caller=lambda: _job("root"),
      journal=lambda: None,
    ),
  }
  # The child's dispatcher starts from the root's map and rebinds it.
  child_handlers = dict(root_handlers)
  rebind_job_handlers(
    child_handlers,
    JobScope(caller=lambda: _job("bg_1"), spill_sink=lambda: child),
  )

  result, error = await child_handlers["tool_result_read"]({"spill_ref": child_ref})
  assert error is None and result is not None
  result, error = await child_handlers["tool_result_read"]({"spill_ref": sibling_ref})
  assert result is None and error is not None
  assert error["code"] == "handle_not_granted"
  # The root's own handler still reads its descendants' spills.
  result, error = await root_handlers["tool_result_read"]({"spill_ref": sibling_ref})
  assert error is None and result is not None



@pytest.mark.asyncio
async def test_reader_cannot_name_the_producer_root(tmp_path: Path) -> None:
  root = _sink(
    tmp_path,
    capabilities=SpillCapabilities(code_execute=True, spill_read=True),
  )
  child = job_spill_sink(root, "bg_0")
  root_ref = _spill(root, "root-call")

  result, error = await make_tool_result_read_handler(
    lambda: child,
    caller=lambda: _job("bg_0", handle_grants=frozenset({root_ref})),
    journal=lambda: None,
  )({"spill_ref": root_ref, "producer_root": str(tmp_path / "elsewhere")})

  # The server resolves the producer by content; a model-supplied root is
  # overwritten, so the granted read still serves the real bytes.
  assert error is None and result is not None
  assert "content of root-call" in json.dumps(result)


def test_exact_reader_enforces_page_and_search_response_budgets(
  tmp_path: Path,
) -> None:
  sink = _sink(
    tmp_path,
    capabilities=SpillCapabilities(code_execute=True, spill_read=True),
  )
  content = ("needle-" + "x" * 2_000) * 200
  publication = write_spill_set(
    sink=sink,
    tool_name="lookup",
    tool_use_id="budget",
    content=content,
    model_max_chars=60_000,
  )

  page = read_spill_result(
    sink,
    spill_ref=publication.spill_ref,
    max_chars=40_000,
  )
  assert len(page["content"]) == 40_000
  assert page["truncated"] is True
  assert len(_reader_json(page)) < 48_000

  search = read_spill_result(
    sink,
    spill_ref=publication.spill_ref,
    query="needle",
    max_results=20,
    context_chars=500,
  )
  assert len(search["matches"]) == 20
  assert search["truncated"] is True
  assert len(_reader_json(search)) < 48_000
  clamped_page = read_spill_result(
    sink,
    spill_ref=publication.spill_ref,
    max_chars=40_001,
  )
  assert len(clamped_page["content"]) == 40_000
  assert clamped_page["notes"] == [
    "max_chars=40001 exceeds the reader limit; served with max_chars=40000"
  ]
  page_with_search_limits = read_spill_result(
    sink,
    spill_ref=publication.spill_ref,
    max_results=2,
  )
  assert "content" in page_with_search_limits
  assert page_with_search_limits["notes"] == [
    "max_results and context_chars apply to a query; ignored for this page read"
  ]
  search_with_page_size = read_spill_result(
    sink,
    spill_ref=publication.spill_ref,
    query="needle",
    max_chars=100,
  )
  assert search_with_page_size["matches"]
  assert search_with_page_size["notes"] == ["max_chars applies to a page read; ignored for this query"]


def test_exact_reader_clamps_context_chars_with_a_note(tmp_path: Path) -> None:
  """PCTY 2026-10-06 seq 12: ``context_chars=1500`` was refused ``must be <= 500``."""
  sink = _sink(
    tmp_path,
    capabilities=SpillCapabilities(code_execute=True, spill_read=True),
  )
  content = ("x" * 2_000) + '"semantic_keys"' + ("y" * 2_000)
  publication = write_spill_set(
    sink=sink,
    tool_name="get_forecast_assumptions_context",
    tool_use_id="context-clamp",
    content=content,
    model_max_chars=1_000,
  )

  search = read_spill_result(
    sink,
    spill_ref=publication.spill_ref,
    query='"semantic_keys"',
    max_results=2,
    context_chars=1_500,
  )

  [match] = search["matches"]
  assert match["start_offset"] == 2_000 - 500
  assert match["end_offset"] == 2_000 + len('"semantic_keys"') + 500
  assert search["notes"] == [
    "context_chars=1500 exceeds the reader limit; served with context_chars=500"
  ]


@pytest.mark.parametrize("escaped", ["\x00", "\\", '"', "😀"])
def test_exact_reader_bounds_serialized_page_and_search_results(
  tmp_path: Path,
  escaped: str,
) -> None:
  root = tmp_path / hashlib.sha256(escaped.encode("utf-8")).hexdigest()[:8]
  root.mkdir()
  sink = _sink(
    root,
    capabilities=SpillCapabilities(code_execute=True, spill_read=True),
  )
  content = ((escaped * 600) + "needle") * 100
  publication = write_spill_set(
    sink=sink,
    tool_name="lookup",
    tool_use_id="escaped-budget",
    content=content,
    model_max_chars=60_000,
  )

  page = read_spill_result(
    sink,
    spill_ref=publication.spill_ref,
    max_chars=40_000,
  )
  assert model_bound_wire_size(page) <= 48_000
  assert page["truncated"] is True
  assert page["next_offset"] == len(page["content"])

  search = read_spill_result(
    sink,
    spill_ref=publication.spill_ref,
    query="needle",
    max_results=20,
    context_chars=500,
  )
  assert model_bound_wire_size(search) <= 48_000
  assert search["truncated"] is True
  assert search["next_offset"] is not None


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


async def _read_custody_spill(
  journal: AgentSessionLog,
  reader: ParentContext,
  spill_ref: str,
  *,
  sink: SpillSink | None = None,
  **arguments: Any,
) -> tuple[Any, Any]:
  return await make_tool_result_read_handler(
    lambda: sink, caller=lambda: reader, journal=lambda: journal,
  )({"spill_ref": spill_ref, **arguments})


@pytest.mark.asyncio
async def test_cold_spill_custody_crosses_physical_roots_without_inheriting_actor_custody(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # File-tools publication reserves path-aware reader wrappers; a 4k model
  # budget cannot fit them. Keep these payloads larger than the 16k budget.
  monkeypatch.setenv("AGENT_GATEWAY_MAX_MODEL_TOOL_RESULT_CHARS", "16000")
  journal_path = tmp_path / "session.jsonl"
  journal = AgentSessionLog(journal_path)
  roots = [tmp_path / name for name in ("A", "B", "C")]
  for root in roots:
    root.mkdir()
  a, b, c = (_sink(root) for root in roots)
  source = canonical_read_custody_task("source")
  successor = canonical_read_custody_task(
    "successor", predecessor=source, resumer_job_id="nested-actor",
  )
  later = canonical_read_custody_task(
    "later", predecessor=successor, resumer_job_id="later-actor",
  )
  unrelated = canonical_read_custody_task("unrelated", parent_job_id="unrelated-owner")
  tasks = (
    source,
    canonical_read_custody_task("source-child", parent_job_id="source"),
    canonical_read_custody_task("old-sibling"),
    canonical_read_custody_task("actor-parent", parent_job_id="actor-root"),
    canonical_read_custody_task("nested-actor", parent_job_id="actor-parent"),
    successor,
    canonical_read_custody_task("successor-child", parent_job_id="successor"),
    canonical_read_custody_task("later-actor", parent_job_id="later-root"),
    later,
    canonical_read_custody_task("later-child", parent_job_id="later"),
    unrelated,
    canonical_read_custody_task(
      "unrelated-successor", predecessor=unrelated, resumer_job_id="successor",
    ),
  )
  for task in tasks:
    await journal.append(custody_registration(task))
  source_sink = job_spill_sink(a, "source")
  successor_sink = job_spill_sink(b, "successor")
  later_sink = job_spill_sink(c, "later")
  producers = {
    "source": source_sink,
    "source-child": job_spill_sink(source_sink, "source-child"),
    "old-sibling": job_spill_sink(a, "old-sibling"),
    "successor": successor_sink,
    "successor-child": job_spill_sink(successor_sink, "successor-child"),
    "later": later_sink,
    "later-child": job_spill_sink(later_sink, "later-child"),
    "unrelated": job_spill_sink(a, "unrelated"),
    "unrelated-successor": job_spill_sink(b, "unrelated-successor"),
  }
  refs: dict[str, str] = {}
  for job_id, sink in producers.items():
    refs[job_id], durable, _live, _provider = await publish_custody_spill(
      sink, _job(job_id), journal,
      tool_use_id=f"call-{job_id}", payload={"payload": job_id + "😀" * 20000},
    )
    assert durable["spill_publication"]["producer_job_id"] == job_id
  expected = {
    "original-owner": {
      "source", "source-child", "old-sibling", "successor",
      "successor-child", "later", "later-child",
    },
    "nested-actor": {
      "source", "source-child", "successor", "successor-child", "later", "later-child",
    },
    "later-actor": {"successor", "successor-child", "later", "later-child"},
    "source": {"source", "source-child", "successor", "successor-child", "later", "later-child"},
    "successor": {
      "successor", "successor-child", "later", "later-child",
      "unrelated", "unrelated-successor",
    },
    "actor-parent": set(),
    "actor-root": set(),
    "later-root": set(),
    "old-sibling": {"old-sibling"},
    "session-alias": set(),
    "runner-alias": set(),
  }

  def _spill_state() -> dict[Path, tuple[int, int, int]]:
    return {
      path: (path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_size)
      for root in roots for path in (root, *root.rglob("*"))
    }

  before = _spill_state()
  # All reads reopen the authenticated journal and have no current sink.
  # No producer writer, lease or freshness callback is needed or reacquired.
  del producers, source_sink, successor_sink, later_sink, a, b, c, journal
  replay = AgentSessionLog(journal_path)
  for reader_job_id, readable_jobs in expected.items():
    for producer_job_id, spill_ref in refs.items():
      locations = await spill_module.resolve_job_spill_producer(
        None, spill_ref, durable_log=replay,
      )
      assert locations
      result, error = await _read_custody_spill(replay, _job(reader_job_id), spill_ref)
      if producer_job_id in readable_jobs:
        assert error is None and result is not None
        assert result["content"] == (
          locations[0] / spill_ref.split(":", 3)[2]
        ).read_text(encoding="utf-8")
      else:
        assert result is None and error is not None
        assert error["code"] == "handle_not_granted"
  assert _spill_state() == before


@pytest.mark.asyncio
async def test_explicit_cross_root_spill_grants_retain_tenant_storage_and_read_limits(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("AGENT_GATEWAY_MAX_MODEL_TOOL_RESULT_CHARS", "16000")
  producer_root = tmp_path / "producer"
  reader_root = tmp_path / "reader"
  producer_root.mkdir()
  reader_root.mkdir()
  journal = AgentSessionLog(tmp_path / "session.jsonl")
  spill_ref, durable, _live, _provider = await publish_custody_spill(
    _sink(producer_root), _job("producer"), journal,
    tool_use_id="explicit", payload={"payload": "α😀" * 10000},
  )
  ungranted = _job("reader")
  granted = replace(ungranted, handle_grants=frozenset({spill_ref}))
  assert granted.tenant is not None
  result, error = await _read_custody_spill(journal, ungranted, spill_ref)
  assert result is None and error["code"] == "handle_not_granted"
  # A current sink is optional, and its write lane is not read custody.
  for sink in (None, _sink(reader_root, capabilities=SpillCapabilities())):
    result, error = await _read_custody_spill(
      journal, granted, spill_ref, sink=sink, max_chars=1,
    )
    assert error is None and len(result["content"]) == 1
  same_storage_role = replace(
    granted, tenant=granted.tenant.model_copy(update={"role": "member"}),
  )
  result, error = await _read_custody_spill(journal, same_storage_role, spill_ref)
  assert error is None and result is not None
  for namespace, tenant in (
    ("bob", granted.tenant.model_copy(update={"storage_id": "bob"})),
    ("other-tenant", granted.tenant.model_copy(update={"tenant_id": "other-tenant"})),
  ):
    # The authenticated caller receives its own tenant/storage journal,
    # never the producer's journal with a reader-local tenant comparison.
    other_journal = AgentSessionLog(tmp_path / namespace / "session.jsonl")
    result, error = await _read_custody_spill(
      other_journal, replace(granted, tenant=tenant), spill_ref,
    )
    assert result is None and error["code"] == "spill_unavailable"
  result, error = await _read_custody_spill(
    journal, granted, spill_ref, sink=_sink(reader_root, max_file_bytes=8),
  )
  assert result is None and error["code"] == "spill_unavailable"
  # Authored result content cannot replace the server publication record.
  isolated = AgentSessionLog(tmp_path / "other-journal.jsonl")
  await isolated.append({
    "type": "tool_call_complete",
    "result": {"spill_publication": durable["spill_publication"]},
  })
  result, error = await _read_custody_spill(isolated, granted, spill_ref)
  assert result is None and error["code"] == "spill_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", [
  "pointer_bytes", "pointer_missing", "pointer_symlink", "member_bytes",
  "member_symlink", "invalid_utf8", "unlisted_member", "wrong_ref", "root_symlink",
])
async def test_cold_custody_reader_preserves_publication_integrity(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  damage: str,
) -> None:
  monkeypatch.setenv("AGENT_GATEWAY_MAX_MODEL_TOOL_RESULT_CHARS", "16000")
  root = tmp_path / "producer"
  root.mkdir()
  journal_path = tmp_path / "session.jsonl"
  journal = AgentSessionLog(journal_path)
  text = "α😀\n" * 6000
  spill_ref, durable, _live, _provider = await publish_custody_spill(
    _sink(root), _job("producer"), journal,
    tool_use_id="integrity", payload={"payload": text},
  )
  reader = _job("passed-reader", handle_grants=frozenset({spill_ref}))
  pointer = root / spill_ref.split(":", 3)[2]
  pointer_bytes = pointer.read_bytes()
  assert hashlib.sha256(pointer_bytes).hexdigest() == spill_ref.rsplit(":", 1)[1]
  manifest = json.loads(pointer_bytes)
  member_ref = next(entry["filename"] for entry in manifest["members"] if entry["role"] == "sidecar")
  member = root / member_ref
  member_bytes = member.read_bytes()
  assert member_bytes.decode("utf-8") == text
  result, error = await _read_custody_spill(
    journal, reader, spill_ref, member_ref=member_ref, offset=1, max_chars=7,
  )
  assert error is None and result["content"] == text[1:8]
  assert pointer.read_bytes() == pointer_bytes and member.read_bytes() == member_bytes
  durable_content = json.loads(durable["final_tool_result_blocks"][0]["content"])
  assert "spill_ref" not in durable_content and "spill_summary" not in durable_content
  arguments: dict[str, Any] = {"member_ref": member_ref}
  if damage == "pointer_bytes":
    pointer.write_bytes(b"changed")
  elif damage == "pointer_missing":
    pointer.unlink()
  elif damage == "pointer_symlink":
    target = root / "pointer-copy"
    target.write_bytes(pointer_bytes)
    pointer.unlink()
    pointer.symlink_to(target)
  elif damage == "member_bytes":
    member.write_bytes(b"changed")
  elif damage == "member_symlink":
    target = root / "member-copy"
    target.write_bytes(member_bytes)
    member.unlink()
    member.symlink_to(target)
  elif damage == "invalid_utf8":
    member.write_bytes(b"\xff" * len(member_bytes))
  elif damage == "unlisted_member":
    (root / "unlisted.txt").write_text("not admitted", encoding="utf-8")
    arguments["member_ref"] = "unlisted.txt"
  elif damage == "wrong_ref":
    spill_ref = spill_ref.rsplit(":", 1)[0] + ":" + "0" * 64
    reader = replace(reader, handle_grants=frozenset({spill_ref}))
  elif damage == "root_symlink":
    moved = tmp_path / "moved-producer"
    root.rename(moved)
    root.symlink_to(moved, target_is_directory=True)
  result, error = await _read_custody_spill(
    AgentSessionLog(journal_path), reader, spill_ref, **arguments,
  )
  assert result is None
  assert error["code"] in {"spill_unavailable", "spill_access_denied"}
