from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


SPILL_LANE_CODE_EXECUTE = "code_execute"
SPILL_LANE_FILE_TOOLS = "file_tools"
SPILL_LANE_NO_READER = "no_reader"
SPILL_MANIFEST_VERSION = 1
SPILL_READER_PATTERN_RESERVE = 1024
SPILL_READER_TARGET_MAX_CHARS = 48_000
SPILL_READER_TARGET_FRACTION = 0.8
SPILL_ROOT_CONTROL_FILES = frozenset({".lease", ".spill_fresh"})


class SpillError(RuntimeError):
  pass


class SpillLimitExceeded(SpillError):
  pass


class SpillPublicationUnsupported(SpillError):
  pass


@dataclass(frozen=True)
class SpillCapabilities:
  code_execute: bool = False
  file_read: bool = False
  file_grep: bool = False

  @property
  def lane(self) -> str:
    if self.code_execute:
      return SPILL_LANE_CODE_EXECUTE
    if self.file_read or self.file_grep:
      return SPILL_LANE_FILE_TOOLS
    return SPILL_LANE_NO_READER


class SpillBudget:
  """A thread-safe, lazily rehydrated byte budget shared by one run tree."""

  def __init__(self, max_bytes: int | None = None) -> None:
    self.max_bytes = max_bytes if max_bytes is None else max(0, int(max_bytes))
    self.used_bytes = 0
    self._initialized_key: tuple[str, str] | None = None
    self._lock = threading.RLock()

  def reserve(self, root: Path, lane: str, amount: int) -> bool:
    if amount < 0:
      raise ValueError("spill reservation must be non-negative")
    with self._lock:
      self._ensure_initialized(root, lane)
      if self.max_bytes is not None and self.used_bytes + amount > self.max_bytes:
        return False
      self.used_bytes += amount
      return True

  def rollback(self, amount: int) -> None:
    if amount <= 0:
      return
    with self._lock:
      self.used_bytes = max(0, self.used_bytes - amount)

  def _ensure_initialized(self, root: Path, lane: str) -> None:
    key = (str(root), lane)
    if self._initialized_key == key:
      return
    if self._initialized_key is not None and self._initialized_key != key:
      raise SpillError("one SpillBudget cannot span multiple spill roots or lanes")
    self.used_bytes = _scan_existing_spill_bytes(root, lane)
    self._initialized_key = key


@dataclass
class SpillSink:
  root_provider: Callable[[], str]
  capabilities: SpillCapabilities = field(default_factory=SpillCapabilities)
  budget: SpillBudget = field(default_factory=SpillBudget)
  max_file_bytes: int | None = None
  after_commit: Callable[[], None] | None = None

  def __call__(self) -> str:
    return self.root_provider()

  @property
  def lane(self) -> str:
    return self.capabilities.lane


@dataclass(frozen=True)
class SpillPublication:
  filename: str
  abspath: str
  lane: str
  payload_kind: str
  member_count: int
  total_chars: int
  largest_members: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _PreparedMember:
  filename: str
  role: str
  data: bytes

  @property
  def chars(self) -> int:
    return len(self.data.decode("utf-8"))

  def manifest_entry(self) -> dict[str, Any]:
    return {
      "filename": self.filename,
      "role": self.role,
      "bytes": len(self.data),
      "chars": self.chars,
      "sha256": hashlib.sha256(self.data).hexdigest(),
    }


@dataclass(frozen=True)
class _PreparedSet:
  stem: str
  payload_kind: str
  members: tuple[_PreparedMember, ...]
  commit_member: _PreparedMember | None

  @property
  def publish_order(self) -> tuple[_PreparedMember, ...]:
    if self.commit_member is None:
      return self.members
    return (*self.members, self.commit_member)


def normalize_spill_sink(value: Callable[[], str] | SpillSink | None) -> SpillSink | None:
  if value is None or isinstance(value, SpillSink):
    return value
  if not callable(value):
    raise TypeError("spill provider must be callable")
  return SpillSink(
    root_provider=value,
    capabilities=SpillCapabilities(code_execute=True),
    budget=SpillBudget(max_bytes=None),
    max_file_bytes=None,
  )


def write_spill_set(
  *,
  sink: SpillSink,
  tool_name: str,
  tool_use_id: Any,
  content: str,
  model_max_chars: int,
  uuid_factory: Callable[[], Any] | None = None,
) -> SpillPublication:
  uuid_factory = uuid_factory or uuid.uuid4
  root = _validated_root(Path(sink()))
  raw = f"{tool_name}_{tool_use_id or uuid_factory().hex}"
  base_stem = re.sub(r"[^A-Za-z0-9._-]", "_", raw)[:120]

  last_collision: FileExistsError | None = None
  for attempt in range(2):
    stem = base_stem if attempt == 0 else f"{base_stem}_{uuid_factory().hex[:8]}"
    prepared = _prepare_set(
      root=root,
      stem=stem,
      content=content,
      lane=sink.lane,
      model_max_chars=model_max_chars,
    )
    emitted = prepared.publish_order
    if sink.max_file_bytes is not None:
      oversized = next((member for member in emitted if len(member.data) > sink.max_file_bytes), None)
      if oversized is not None:
        raise SpillLimitExceeded(
          f"spill member {oversized.filename} exceeds per-file cap ({len(oversized.data)} > {sink.max_file_bytes})"
        )
    reserved_total = sum(len(member.data) for member in emitted)
    if not sink.budget.reserve(root, sink.lane, reserved_total):
      raise SpillLimitExceeded(
        f"spill run budget exceeded ({sink.budget.used_bytes} + {reserved_total} > {sink.budget.max_bytes})"
      )

    linked: list[tuple[Path, int, tuple[int, int]]] = []
    try:
      for member in emitted:
        final_path = _contained_member_path(root, member.filename)
        identity = _publish_no_clobber(root, final_path, member.data, uuid_factory=uuid_factory)
        linked.append((final_path, len(member.data), identity))
    except FileExistsError as exc:
      retained = _remove_attempt_members(linked)
      sink.budget.rollback(reserved_total - retained)
      last_collision = exc
      if attempt == 0:
        continue
      raise
    except Exception:
      retained = _remove_attempt_members(linked)
      sink.budget.rollback(reserved_total - retained)
      raise

    if sink.after_commit is not None:
      sink.after_commit()
    pointer = prepared.commit_member or prepared.members[0]
    summary_members = sorted(
      (member.manifest_entry() for member in emitted),
      key=lambda item: (-int(item["chars"]), str(item["filename"])),
    )[:3]
    return SpillPublication(
      filename=pointer.filename,
      abspath=str(_contained_member_path(root, pointer.filename)),
      lane=sink.lane,
      payload_kind=prepared.payload_kind,
      member_count=len(emitted),
      total_chars=sum(member.chars for member in emitted),
      largest_members=tuple(
        {
          "filename": item["filename"],
          "role": item["role"],
          "chars": item["chars"],
        }
        for item in summary_members
      ),
    )
  if last_collision is not None:
    raise last_collision
  raise FileExistsError(base_stem)


def reconstruct_spill_manifest(manifest_path: str | Path) -> Any:
  """Reconstruct and verify a committed file-tools spill using disk only."""

  path = Path(manifest_path)
  if path.is_symlink() or not _is_regular_file(path):
    raise SpillError("spill manifest must be a regular non-symlink file")
  manifest = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(manifest, dict) or manifest.get("schema_version") != SPILL_MANIFEST_VERSION:
    raise SpillError("unsupported spill manifest")
  if manifest.get("lane") != SPILL_LANE_FILE_TOOLS:
    raise SpillError("spill manifest lane mismatch")
  stem = manifest.get("stem")
  if not isinstance(stem, str) or path.name != f"{stem}.manifest.json":
    raise SpillError("spill manifest stem mismatch")
  payload_kind = manifest.get("payload_kind")
  if payload_kind not in {"json", "text"}:
    raise SpillError("unsupported spill payload kind")
  root = path.parent.resolve()
  member_entries = manifest.get("members")
  if (
    not isinstance(member_entries, list)
    or manifest.get("member_count") != len(member_entries)
  ):
    raise SpillError("spill manifest members missing")
  members: dict[str, dict[str, Any]] = {}
  for entry in member_entries:
    if not isinstance(entry, dict) or not isinstance(entry.get("filename"), str):
      raise SpillError("invalid spill manifest member")
    filename = entry["filename"]
    role = entry.get("role")
    if (
      filename in members
      or not filename.startswith(f"{stem}.")
      or role not in {"index", "sidecar", "chunks", "canonical"}
      or not isinstance(entry.get("bytes"), int)
      or not isinstance(entry.get("chars"), int)
      or not isinstance(entry.get("sha256"), str)
    ):
      raise SpillError("invalid spill manifest member")
    member_path = _contained_member_path(root, filename)
    if member_path.is_symlink() or not _is_regular_file(member_path):
      raise SpillError(f"spill member is not a regular file: {filename}")
    data = member_path.read_bytes()
    if len(data) != entry.get("bytes") or hashlib.sha256(data).hexdigest() != entry.get("sha256"):
      raise SpillError(f"spill member integrity mismatch: {filename}")
    try:
      member_text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
      raise SpillError(f"spill member is not UTF-8: {filename}") from exc
    if len(member_text) != entry.get("chars"):
      raise SpillError(f"spill member character count mismatch: {filename}")
    members[filename] = {**entry, "path": member_path, "data": data, "text": member_text}

  role_counts = {
    role: sum(1 for entry in members.values() if entry.get("role") == role)
    for role in ("index", "sidecar", "chunks", "canonical")
  }
  if payload_kind == "json" and (role_counts["index"] != 1 or role_counts["canonical"] != 0):
    raise SpillError("JSON spill member roles are invalid")
  if payload_kind == "text" and (
    role_counts["canonical"] != 1
    or role_counts["index"] != 0
    or role_counts["sidecar"] != 0
  ):
    raise SpillError("text spill member roles are invalid")

  refs = manifest.get("refs")
  if not isinstance(refs, list):
    raise SpillError("spill manifest refs missing")
  ref_values: dict[str, str] = {}
  referenced_projection_members: set[str] = set()
  for ref in refs:
    if not isinstance(ref, dict):
      raise SpillError("invalid spill ref")
    ref_id = ref.get("ref_id")
    filename = ref.get("member_filename")
    role = ref.get("role")
    if not isinstance(ref_id, str) or not isinstance(filename, str) or filename not in members:
      raise SpillError("spill ref target missing")
    if ref_id in ref_values or role != members[filename].get("role") or not isinstance(ref.get("chars"), int):
      raise SpillError("invalid or duplicate spill ref")
    data = members[filename]["data"]
    if role in {"sidecar", "canonical"}:
      ref_values[ref_id] = data.decode("utf-8")
    elif role == "chunks":
      ref_values[ref_id] = _decode_chunk_records(data.decode("utf-8"), ref_id)
    else:
      raise SpillError(f"unsupported spill ref role: {role}")
    if len(ref_values[ref_id]) != ref.get("chars"):
      raise SpillError(f"spill ref character count mismatch: {ref_id}")
    if role in {"sidecar", "chunks"}:
      referenced_projection_members.add(filename)

  if payload_kind == "text":
    if len(ref_values) != 1 or next(iter(refs), {}).get("role") != "canonical":
      raise SpillError("text spill must contain exactly one ref")
    return next(iter(ref_values.values()))

  expected_projection_members = {
    filename
    for filename, entry in members.items()
    if entry.get("role") in {"sidecar", "chunks"}
  }
  if referenced_projection_members != expected_projection_members:
    raise SpillError("JSON spill projection refs do not match members")

  index_entry = next((entry for entry in members.values() if entry.get("role") == "index"), None)
  if index_entry is None:
    raise SpillError("JSON spill index missing")
  index = json.loads(index_entry["data"].decode("utf-8"))
  counts = {ref_id: 0 for ref_id in ref_values}

  def _restore(value: Any) -> Any:
    if isinstance(value, dict):
      if set(value) == {"$spill_ref"} and value.get("$spill_ref") in ref_values:
        ref_id = str(value["$spill_ref"])
        counts[ref_id] += 1
        return ref_values[ref_id]
      return {key: _restore(item) for key, item in value.items()}
    if isinstance(value, list):
      return [_restore(item) for item in value]
    return value

  restored = _restore(index)
  mismatched = {ref_id: count for ref_id, count in counts.items() if count != 1}
  if mismatched:
    raise SpillError(f"spill ref occurrence mismatch: {mismatched}")
  return restored


def _prepare_set(
  *,
  root: Path,
  stem: str,
  content: str,
  lane: str,
  model_max_chars: int,
) -> _PreparedSet:
  try:
    parsed = json.loads(content)
  except Exception:
    parsed = None
    is_json = False
  else:
    is_json = True

  if lane != SPILL_LANE_FILE_TOOLS:
    if is_json:
      canonical = json.dumps(parsed, indent=1, ensure_ascii=False, default=str).encode("utf-8")
      member = _PreparedMember(f"{stem}.json", "canonical", canonical)
      return _PreparedSet(stem, "json", (member,), None)
    member = _PreparedMember(f"{stem}.txt", "canonical", content.encode("utf-8"))
    return _PreparedSet(stem, "text", (member,), None)

  line_budget = _reader_line_budget(
    _contained_member_path(root, f"{stem}.c99999999.ffffffff.chunks.txt"),
    model_max_chars=model_max_chars,
  )
  if is_json:
    return _prepare_file_tools_json(root, stem, parsed, content, line_budget)
  return _prepare_file_tools_text(root, stem, content, line_budget)


def _prepare_file_tools_json(
  root: Path,
  stem: str,
  parsed: Any,
  original_content: str,
  line_budget: int,
) -> _PreparedSet:
  refs: list[dict[str, Any]] = []
  members: list[_PreparedMember] = []

  def _walk(value: Any, pointer: str, field_name: str) -> Any:
    if isinstance(value, str) and _string_requires_externalization(value, line_budget):
      ordinal = len(refs)
      pointer_hash = hashlib.sha256(pointer.encode("utf-8")).hexdigest()[:8]
      raw_lines = value.splitlines() or [value]
      use_sidecar = "\n" in value or "\r" in value
      use_sidecar = use_sidecar and all(_reader_encoded_line_chars(line) <= line_budget for line in raw_lines)
      if use_sidecar:
        ref_id = f"s{ordinal}-{pointer_hash}"
        lowered = field_name.lower()
        ext = "md" if lowered == "markdown" or lowered.endswith(("_markdown", "_md")) else "txt"
        filename = f"{stem}.s{ordinal}.{pointer_hash}.{ext}"
        role = "sidecar"
        data = value.encode("utf-8")
      else:
        ref_id = f"c{ordinal}-{pointer_hash}"
        filename = f"{stem}.c{ordinal}.{pointer_hash}.chunks.txt"
        role = "chunks"
        data = _encode_chunk_records(value, ref_id, line_budget).encode("utf-8")
      members.append(_PreparedMember(filename, role, data))
      refs.append(
        {
          "ref_id": ref_id,
          "member_filename": filename,
          "role": role,
          "chars": len(value),
          "pointer_value": pointer,
        }
      )
      return {"$spill_ref": ref_id}
    if isinstance(value, dict):
      return {
        key: _walk(item, f"{pointer}/{_json_pointer_escape(str(key))}", str(key))
        for key, item in value.items()
      }
    if isinstance(value, list):
      return [_walk(item, f"{pointer}/{index}", str(index)) for index, item in enumerate(value)]
    return value

  index_value = _walk(parsed, "", "")
  for ref in refs:
    if _count_ref_markers(index_value, str(ref["ref_id"])) != 1:
      raise SpillError(f"ambiguous spill ref marker: {ref['ref_id']}")
  index_text = json.dumps(index_value, indent=1, ensure_ascii=False, default=str)
  _require_reader_lines(index_text, line_budget, role="index")
  index_member = _PreparedMember(f"{stem}.index.json", "index", index_text.encode("utf-8"))
  members.insert(0, index_member)
  manifest_member = _build_manifest_member(
    root=root,
    stem=stem,
    payload_kind="json",
    original_chars=len(original_content),
    members=members,
    refs=refs,
    line_budget=line_budget,
  )
  return _PreparedSet(stem, "json", tuple(members), manifest_member)


def _prepare_file_tools_text(root: Path, stem: str, content: str, line_budget: int) -> _PreparedSet:
  canonical = _PreparedMember(f"{stem}.txt", "canonical", content.encode("utf-8"))
  members = [canonical]
  ref_id = f"r0-{hashlib.sha256(stem.encode('utf-8')).hexdigest()[:8]}"
  any_overlong = any(
    _reader_encoded_line_chars(line) > line_budget
    for line in (content.splitlines() or [content])
  )
  ref_role = "canonical"
  ref_filename = canonical.filename
  if any_overlong:
    chunks = _PreparedMember(
      f"{stem}.c0.{hashlib.sha256(stem.encode('utf-8')).hexdigest()[:8]}.chunks.txt",
      "chunks",
      _encode_chunk_records(content, ref_id, line_budget).encode("utf-8"),
    )
    members.append(chunks)
  refs = [
    {
      "ref_id": ref_id,
      "member_filename": ref_filename,
      "role": ref_role,
      "chars": len(content),
      "pointer_value": "$",
    }
  ]
  manifest_member = _build_manifest_member(
    root=root,
    stem=stem,
    payload_kind="text",
    original_chars=len(content),
    members=members,
    refs=refs,
    line_budget=line_budget,
  )
  return _PreparedSet(stem, "text", tuple(members), manifest_member)


def _build_manifest_member(
  *,
  root: Path,
  stem: str,
  payload_kind: str,
  original_chars: int,
  members: Iterable[_PreparedMember],
  refs: list[dict[str, Any]],
  line_budget: int,
) -> _PreparedMember:
  manifest_refs: list[dict[str, Any]] = []
  for raw_ref in refs:
    pointer = str(raw_ref["pointer_value"])
    ref = {key: value for key, value in raw_ref.items() if key != "pointer_value"}
    ref["pointer"] = pointer
    probe = json.dumps(ref, ensure_ascii=False, default=str)
    if _reader_encoded_line_chars(probe) > line_budget:
      prefix_chars = max(16, min(256, line_budget // 8))
      ref.pop("pointer", None)
      ref["pointer_prefix"] = pointer[:prefix_chars]
      ref["pointer_sha256"] = hashlib.sha256(pointer.encode("utf-8")).hexdigest()
    manifest_refs.append(ref)
  member_entries = [member.manifest_entry() for member in members]
  payload = {
    "schema_version": SPILL_MANIFEST_VERSION,
    "stem": stem,
    "lane": SPILL_LANE_FILE_TOOLS,
    "payload_kind": payload_kind,
    "original_chars": original_chars,
    "member_count": len(member_entries),
    "members": member_entries,
    "refs": manifest_refs,
  }
  text = json.dumps(payload, indent=1, ensure_ascii=False, default=str)
  try:
    _require_reader_lines(text, line_budget, role="manifest")
  except SpillLimitExceeded:
    for ref in manifest_refs:
      pointer = ref.pop("pointer", None)
      if pointer is None:
        continue
      prefix_chars = max(16, min(64, line_budget // 12))
      ref["pointer_prefix"] = str(pointer)[:prefix_chars]
      ref["pointer_sha256"] = hashlib.sha256(str(pointer).encode("utf-8")).hexdigest()
    text = json.dumps(payload, indent=1, ensure_ascii=False, default=str)
    _require_reader_lines(text, line_budget, role="manifest")
  filename = f"{stem}.manifest.json"
  _contained_member_path(root, filename)
  return _PreparedMember(filename, "manifest", text.encode("utf-8"))


def _reader_line_budget(final_path: Path, *, model_max_chars: int) -> int:
  target = min(
    SPILL_READER_TARGET_MAX_CHARS,
    int(max(0, model_max_chars) * SPILL_READER_TARGET_FRACTION),
  )
  path_chars = len(json.dumps(str(final_path), ensure_ascii=True))
  wrapper_overhead = (6 * path_chars) + SPILL_READER_PATTERN_RESERVE + 2_000
  remaining = target - wrapper_overhead
  if remaining < 768:
    raise SpillLimitExceeded("spill reader wrapper overhead exceeds model-safe target")
  return remaining // 3


def _reader_encoded_line_chars(line: str) -> int:
  return len(json.dumps(line, ensure_ascii=True))


def _string_requires_externalization(value: str, line_budget: int) -> bool:
  return len(json.dumps(value, ensure_ascii=True)) + 128 > line_budget


def _require_reader_lines(text: str, line_budget: int, *, role: str) -> None:
  for line in text.splitlines() or [text]:
    if _reader_encoded_line_chars(line) > line_budget:
      raise SpillLimitExceeded(f"{role} contains an overlong reader line")


def _encode_chunk_records(value: str, ref_id: str, line_budget: int) -> str:
  records: list[str] = []
  start = 0
  while start < len(value):
    low = start + 1
    high = len(value)
    best: tuple[int, str] | None = None
    while low <= high:
      end = (low + high) // 2
      segment = value[start:end]
      record = f"@{ref_id} {start}-{end} {json.dumps(segment, ensure_ascii=True)}"
      if _reader_encoded_line_chars(record) <= line_budget:
        best = (end, record)
        low = end + 1
      else:
        high = end - 1
    if best is None:
      raise SpillLimitExceeded("chunk line budget cannot encode one code point")
    end, record = best
    records.append(record)
    start = end
  if not records:
    records.append(f"@{ref_id} 0-0 \"\"")
  return "\n".join(records) + "\n"


def _decode_chunk_records(text: str, expected_ref_id: str) -> str:
  cursor = 0
  segments: list[str] = []
  pattern = re.compile(r"^@(\S+) (\d+)-(\d+) (.+)$")
  for line in text.splitlines():
    match = pattern.fullmatch(line)
    if match is None:
      raise SpillError("invalid spill chunk record")
    ref_id, start_raw, end_raw, literal = match.groups()
    start = int(start_raw)
    end = int(end_raw)
    if ref_id != expected_ref_id or start != cursor or end < start:
      raise SpillError("non-contiguous spill chunk record")
    segment = json.loads(literal)
    if not isinstance(segment, str) or len(segment) != end - start:
      raise SpillError("spill chunk offset mismatch")
    segments.append(segment)
    cursor = end
  return "".join(segments)


def _json_pointer_escape(value: str) -> str:
  return value.replace("~", "~0").replace("/", "~1")


def _count_ref_markers(value: Any, ref_id: str) -> int:
  if isinstance(value, dict):
    own = 1 if set(value) == {"$spill_ref"} and value.get("$spill_ref") == ref_id else 0
    return own + sum(_count_ref_markers(item, ref_id) for item in value.values())
  if isinstance(value, list):
    return sum(_count_ref_markers(item, ref_id) for item in value)
  return 0


def _validated_root(path: Path) -> Path:
  expanded = path.expanduser()
  try:
    if expanded.is_symlink() or not expanded.is_dir():
      raise SpillError("spill root must be an existing non-symlink directory")
    resolved = expanded.resolve(strict=True)
  except OSError as exc:
    raise SpillError(f"spill root unavailable: {expanded}") from exc
  return resolved


def _contained_member_path(root: Path, filename: str) -> Path:
  if Path(filename).name != filename or filename in {"", ".", ".."}:
    raise SpillError("invalid spill member filename")
  path = root / filename
  if path.parent.resolve() != root.resolve():
    raise SpillError("spill member escapes root")
  return path


def _publish_no_clobber(
  root: Path,
  final_path: Path,
  data: bytes,
  *,
  uuid_factory: Callable[[], Any],
) -> tuple[int, int]:
  _ = uuid_factory
  temp_path = _contained_member_path(root, f".{final_path.name}.{uuid.uuid4().hex}.tmp")
  flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
  if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
  fd: int | None = None
  try:
    fd = os.open(temp_path, flags, 0o600)
    with os.fdopen(fd, "wb", closefd=True) as handle:
      fd = None
      handle.write(data)
      handle.flush()
      os.fsync(handle.fileno())
    temp_info = os.lstat(temp_path)
    if not stat.S_ISREG(temp_info.st_mode):
      raise SpillPublicationUnsupported("spill temp publication member is not a regular file")
    try:
      os.link(temp_path, final_path, follow_symlinks=False)
    except OSError as exc:
      if exc.errno in {errno.EPERM, getattr(errno, "EOPNOTSUPP", errno.EPERM)}:
        raise SpillPublicationUnsupported("spill filesystem does not support no-clobber hard-link publication") from exc
      raise
    return (temp_info.st_dev, temp_info.st_ino)
  finally:
    if fd is not None:
      os.close(fd)
    try:
      temp_path.unlink()
    except FileNotFoundError:
      pass


def _remove_attempt_members(linked: list[tuple[Path, int, tuple[int, int]]]) -> int:
  retained = 0
  for path, size, identity in reversed(linked):
    try:
      visible = path.lstat()
      if (visible.st_dev, visible.st_ino) != identity:
        continue
      path.unlink()
    except FileNotFoundError:
      pass
    except OSError:
      retained += size
  return retained


def _scan_existing_spill_bytes(root: Path, lane: str) -> int:
  if not root.exists():
    return 0
  if lane != SPILL_LANE_FILE_TOOLS:
    return _scan_canonical_spill_bytes(root)

  committed: set[str] = set()
  total = 0
  for manifest_path in root.glob("*.manifest.json"):
    if manifest_path.is_symlink() or not _is_regular_file(manifest_path):
      continue
    try:
      payload = json.loads(manifest_path.read_text(encoding="utf-8"))
      entries = payload["members"]
      if payload.get("schema_version") != SPILL_MANIFEST_VERSION or not isinstance(entries, list):
        raise ValueError("invalid manifest")
      names = [entry["filename"] for entry in entries]
      if not all(isinstance(name, str) and Path(name).name == name for name in names):
        raise ValueError("invalid member name")
      paths = [_contained_member_path(root, name) for name in names]
      if not all(_is_regular_file(path) and not path.is_symlink() for path in paths):
        raise ValueError("missing member")
    except Exception:
      continue
    committed.add(manifest_path.name)
    committed.update(names)
    total += manifest_path.stat().st_size + sum(path.stat().st_size for path in paths)

  for path in root.iterdir():
    if path.name in SPILL_ROOT_CONTROL_FILES or path.name in committed or path.is_dir():
      continue
    if path.is_symlink() or not _is_regular_file(path):
      continue
    size = path.stat().st_size
    try:
      path.unlink()
    except OSError:
      total += size
  return total


def _scan_canonical_spill_bytes(root: Path) -> int:
  total = 0
  for path in root.iterdir():
    if path.name in SPILL_ROOT_CONTROL_FILES or path.is_dir() or path.is_symlink():
      continue
    if not _is_regular_file(path):
      continue
    if path.name.endswith(".tmp"):
      try:
        path.unlink()
      except OSError:
        total += path.stat().st_size
      continue
    total += path.stat().st_size
  return total


def _is_regular_file(path: Path) -> bool:
  try:
    return stat.S_ISREG(path.lstat().st_mode)
  except OSError:
    return False


__all__ = [
  "SPILL_LANE_CODE_EXECUTE",
  "SPILL_LANE_FILE_TOOLS",
  "SPILL_LANE_NO_READER",
  "SpillBudget",
  "SpillCapabilities",
  "SpillError",
  "SpillLimitExceeded",
  "SpillPublication",
  "SpillPublicationUnsupported",
  "SpillSink",
  "normalize_spill_sink",
  "reconstruct_spill_manifest",
  "write_spill_set",
]
