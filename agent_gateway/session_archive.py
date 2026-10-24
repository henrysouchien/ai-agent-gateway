"""Gather a self-contained session archive from durable stores, without a server."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from glob import escape
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterator

from .agent_session_log_records import slugify
from .tool_result_spill import (
  SPILL_READ_DEFAULT_FILE_MAX_BYTES,
  _SPILL_REF_RE,
  read_spill_result,
  reconstruct_spill_manifest,
)


@dataclass
class ArchiveResult:
  output_dir: Path
  manifest_path: Path
  files_archived: int
  prompts: list[tuple[str, bool]]
  spills: list[tuple[str, str | None, bool]]
  warnings: list[str]


def _configured_path(name: str, default: Path) -> Path:
  return Path(os.environ.get(name, "").strip() or default).expanduser()


def _default_chat_dir() -> Path:
  # Precedence: records_root/state/logs/chat (selected by archive_session),
  # GATEWAY_CHAT_LOG_DIR, GATEWAY_LOG_DIR/chat_logs, then the production
  # data/gateway_logs/chat_logs convention below. Isolated launcher proof:
  # scripts/local_gateway_isolated.py:290 defines state/logs/chat; :1351 exports it.
  # Resolve offline without gateway_log_root's directory-creation side effects.
  configured = os.environ.get("GATEWAY_CHAT_LOG_DIR", "").strip()
  if configured:
    path = Path(configured).expanduser()
    if not path.is_absolute():
      raise ValueError("GATEWAY_CHAT_LOG_DIR must be an absolute path")
    return path
  log_root = os.environ.get("GATEWAY_LOG_DIR", "").strip()
  if log_root:
    return Path(log_root).expanduser() / "chat_logs"
  return Path.home() / ".local/state/hank/data/gateway_logs/chat_logs"


def _json_object(path: Path, warnings: list[str]) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
      raise ValueError("expected a JSON object")
    return value
  except (OSError, ValueError, UnicodeError) as exc:
    warnings.append(f"Cannot read {path}: {exc}")
    return {}


def _json_lines(path: Path, warnings: list[str]) -> Iterator[dict[str, Any]]:
  try:
    with path.open(encoding="utf-8") as handle:
      for number, line in enumerate(handle, 1):
        if not line.strip():
          continue
        try:
          value = json.loads(line)
          if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        except ValueError as exc:
          warnings.append(f"Cannot read {path}:{number}: {exc}")
          continue
        yield value
  except (OSError, UnicodeError) as exc:
    warnings.append(f"Cannot read {path}: {exc}")


def archive_session(
  session_id: str,
  output_dir: Path,
  *,
  records_root: Path | None = None,
  include_orphans: bool = False,
  chat_log_dir: Path | None = None,
  session_log_base: Path | None = None,
  autonomous_log_dir: Path | None = None,
  surface_content_path: Path | None = None,
) -> ArchiveResult:
  """Copy a session's records and resolved payloads to ``output_dir/session_id``.

  Missing, malformed, ambiguous, or expired source records are reported in the
  manifest rather than aborting collection. Destination I/O errors propagate.
  Explicit store paths take precedence over records-root and environment defaults.
  """
  if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", session_id):
    raise ValueError("session_id must be a filename-safe session identifier")
  records = Path(records_root).expanduser() if records_root is not None else None
  chat_dir = Path(chat_log_dir).expanduser() if chat_log_dir is not None else (
    records / "state/logs/chat" if records else _default_chat_dir()
  )
  session_base = Path(session_log_base).expanduser() if session_log_base is not None else (
    records / "state/logs/sessions" if records else _configured_path("AGENT_SESSION_LOG_BASE_DIR", Path("api/sessions"))
  )
  auto_default = _configured_path("AGENTS_MCP_LOG_DIR", Path.home() / ".cache/agent-gateway/autonomous")
  if os.environ.get("GATEWAY_LOG_DIR", "").strip():
    auto_default = Path(os.environ["GATEWAY_LOG_DIR"]).expanduser() / "autonomous"
  auto_dir = Path(autonomous_log_dir).expanduser() if autonomous_log_dir is not None else _configured_path(
    "AGENT_GATEWAY_AUTONOMOUS_LOG_DIR", auto_default,
  )
  surface_path = Path(surface_content_path).expanduser() if surface_content_path is not None else _configured_path(
    "GATEWAY_SURFACE_CONTENT_PATH",
    Path.home() / ".local/state/hank/data/telemetry/surface_content.jsonl",
  )
  destination = Path(output_dir).expanduser() / session_id
  destination.mkdir(parents=True, exist_ok=False)
  warnings: list[str] = []
  files: list[dict[str, Any]] = []
  copied: dict[Path, str] = {}
  event_logs: list[Path] = []

  def write_payload(arcname: str, source: Path, data: bytes) -> None:
    target = destination / arcname
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    files.append({
      "arcname": arcname,
      "source_path": str(source.absolute()),
      "sha256": hashlib.sha256(data).hexdigest(),
      "bytes": len(data),
    })

  def copy_file(source: Path, arcname: str, *, events: bool = False) -> None:
    if source in copied:
      return
    try:
      data = source.read_bytes()
    except OSError as exc:
      warnings.append(f"Cannot copy {source}: {exc}")
      return
    write_payload(arcname, source, data)
    copied[source] = arcname
    if events:
      event_logs.append(destination / arcname)

  def copy_stream(active: Path, prefix: Path) -> None:
    if active.is_file():
      copy_file(active, (prefix / active.name).as_posix(), events=True)
    meta = active.with_suffix(".meta.json")
    if meta.is_file():
      copy_file(meta, (prefix / meta.name).as_posix())
    segments = active.with_name(f"{active.stem}.segments")
    try:
      for member in sorted(segments.rglob("*")) if segments.is_dir() else []:
        if member.is_file():
          copy_file(member, (prefix / member.relative_to(active.parent)).as_posix(), events=member.suffix == ".jsonl")
    except OSError as exc:
      warnings.append(f"Cannot enumerate {segments}: {exc}")

  transcript = chat_dir / f"{session_id}.jsonl"
  transcript_meta = chat_dir / f"{session_id}.meta.json"
  copy_file(transcript, f"session/{transcript.name}", events=True)
  copy_file(transcript_meta, f"session/{transcript_meta.name}")
  metadata = _json_object(transcript_meta, warnings) if transcript_meta.is_file() else {}
  user_id = metadata.get("user_id")
  pattern = f"agentsess_{escape(session_id)}*"
  if isinstance(user_id, str) and user_id:
    pattern += f"_{escape(user_id)}"
  pattern += ".jsonl"
  try:
    streams = set(session_base.glob(f"*/{pattern}"))
    # A rotation may have moved the active JSONL away before its replacement.
    for segments in session_base.glob(f"*/{pattern[:-6]}.segments"):
      streams.add(segments.with_name(f"{segments.name[:-9]}.jsonl"))
  except OSError as exc:
    warnings.append(f"Cannot enumerate session logs in {session_base}: {exc}")
    streams = set()
  if not streams:
    warnings.append(f"No v1 session logs for {session_id} in {session_base}; interactive v2 logs have no session-id index and cannot be resolved by guessing a digest")
  if not isinstance(user_id, str) or not user_id:
    users = set()
    for active in streams:
      sidecar = active.with_suffix(".meta.json")
      identity = _json_object(sidecar, warnings) if sidecar.is_file() else {}
      users.add(str(identity.get("user_id") or active.stem.rsplit("_", 1)[-1]))
    if len(users) > 1:
      warnings.append(f"Ambiguous session {session_id}: session logs name multiple users: {', '.join(sorted(users))}")
      # An ambiguous filename match is evidence, not permission to attach users.
      streams = set()
  for active in sorted(streams):
    copy_stream(active, Path("session") / active.parent.relative_to(session_base))

  try:
    tasks = sorted(auto_dir.glob("bg_*.task.json"))
  except OSError as exc:
    warnings.append(f"Cannot enumerate autonomous records in {auto_dir}: {exc}")
    tasks = []
  child_owners: dict[str, set[str]] = {}
  for task_path in tasks:
    task = _json_object(task_path, warnings)
    context = task.get("admitted_context")
    lineage = context.get("lineage") if isinstance(context, dict) else None
    parent = lineage.get("parent_job_id") if isinstance(lineage, dict) else None
    orphan = not parent
    if orphan:
      warnings.append(f"Unknown parent for autonomous run {task_path.name}")
    if parent != session_id and not (orphan and include_orphans):
      continue
    run_id = task_path.name[:-len(".task.json")]
    prefix = Path("autonomous") / ("unknown-parent" if orphan else run_id)
    # Lineage is the durable join. User/time guesses would silently attach
    # unrelated sessions, so null-lineage records remain explicitly unassigned.
    for suffix in ("task.json", "events.jsonl", "log", "operator-messages.jsonl"):
      member = task_path.with_name(f"{run_id}.{suffix}")
      if member.is_file() or suffix == "task.json":
        copy_file(member, (prefix / member.name).as_posix(), events=suffix == "events.jsonl")
    owner = task.get("owner_user_id") or task.get("user_id")
    if isinstance(owner, str) and owner.strip():
      child_owners.setdefault(owner, set()).add(run_id)
      child_pattern = f"agentsess_{escape(slugify(run_id))}*_{escape(slugify(owner))}.jsonl"
      try:
        child_streams = set((session_base / "autonomous").glob(child_pattern))
        for segments in (session_base / "autonomous").glob(f"{child_pattern[:-6]}.segments"):
          child_streams.add(segments.with_name(f"{segments.name[:-9]}.jsonl"))
      except OSError as exc:
        warnings.append(f"Cannot enumerate session logs for {run_id}: {exc}")
        child_streams = set()
      for active in sorted(child_streams):
        copy_stream(active, Path("session/autonomous"))

  # v2 streams are shared by workload identity, not named for bg_N. Only
  # autonomous children are resolved here: an event's recorded session_id must
  # establish the join; profile/user matches alone cannot attach a stream.
  v2_base = session_base / ".session-log-v2"
  if child_owners and v2_base.is_dir():
    try:
      v2_sidecars = sorted(v2_base.glob("*/*.meta.json"))
    except OSError as exc:
      warnings.append(f"Cannot enumerate autonomous v2 logs in {v2_base}: {exc}")
      v2_sidecars = []
    for sidecar in v2_sidecars:
      identity = _json_object(sidecar, warnings)
      owner = identity.get("user_id")
      if not isinstance(owner, str) or owner not in child_owners:
        continue
      active = sidecar.with_name(f"{sidecar.name[:-len('.meta.json')]}.jsonl")
      segments = active.with_name(f"{active.stem}.segments")
      try:
        candidates = ([active] if active.is_file() else []) + sorted(segments.glob("*.jsonl"))
      except OSError as exc:
        warnings.append(f"Cannot enumerate {segments}: {exc}")
        continue
      matched = False
      for candidate in candidates:
        for row in _json_lines(candidate, warnings):
          event = row.get("event", row)
          linked_id = event.get("session_id") if isinstance(event, dict) else None
          if isinstance(linked_id, str) and any(
            linked_id == run_id or linked_id.startswith(f"{run_id}--")
            for run_id in child_owners[owner]
          ):
            matched = True
            break
        if matched:
          break
      if matched:
        copy_stream(active, Path("session") / active.parent.relative_to(session_base))

  prompt_hashes: set[str] = set()
  publications: dict[str, str | None] = {}
  for log_path in event_logs:
    for row in _json_lines(log_path, warnings):
      event = row.get("event", row)
      if not isinstance(event, dict):
        continue
      if event.get("type") == "context_manifest":
        prompt_hash = event.get("system_prompt_hash")
        if isinstance(prompt_hash, str):
          prompt_hashes.add(prompt_hash)
      publication = event.get("spill_publication")
      if event.get("type") == "spill_publication":
        publication = event
      if isinstance(publication, dict) and isinstance(publication.get("spill_ref"), str):
        root = publication.get("producer_root")
        ref = publication["spill_ref"]
        if ref not in publications or publications[ref] is None:
          publications[ref] = root if isinstance(root, str) else None

  prompt_content: dict[str, str] = {}
  if prompt_hashes or not surface_path.is_file():
    for row in _json_lines(surface_path, warnings):
      content_hash, content = row.get("content_hash"), row.get("content")
      if isinstance(content_hash, str) and content_hash in prompt_hashes and isinstance(content, str):
        prompt_content[content_hash] = content
  prompts: list[tuple[str, bool]] = []
  for prompt_hash in sorted(prompt_hashes):
    content = prompt_content.get(prompt_hash)
    valid_hash = re.fullmatch(r"(?:sha256:)?([0-9a-f]{64})", prompt_hash)
    present = content is not None and valid_hash is not None
    if present and valid_hash is not None and content is not None:
      write_payload(f"prompts/{valid_hash.group(1)}.txt", surface_path, content.encode("utf-8"))
    else:
      warnings.append(f"System prompt {prompt_hash} is missing or invalid in {surface_path}")
    prompts.append((prompt_hash, present))

  spills: list[tuple[str, str | None, bool]] = []
  spill_names: set[str] = set()
  for ref, root in sorted(publications.items()):
    arcname = None
    match = _SPILL_REF_RE.fullmatch(ref)
    try:
      if match is None or not root:
        raise ValueError("spill publication has no valid reference/producer_root")
      filename = match.group(1)
      pointer = Path(root) / filename
      if filename.endswith(".manifest.json"):
        # Reconstruction validates members; the reader binds the manifest
        # itself to the digest in the durable publication's opaque reference.
        read_spill_result(None, spill_ref=ref, producer_root=Path(root), max_chars=1)
        data = json.dumps(reconstruct_spill_manifest(pointer), ensure_ascii=False).encode("utf-8")
      else:
        pages: list[str] = []
        offset = 0
        while True:
          page = read_spill_result(None, spill_ref=ref, producer_root=Path(root), offset=offset, max_chars=SPILL_READ_DEFAULT_FILE_MAX_BYTES)
          pages.append(page["content"])
          next_offset = page["next_offset"]
          if next_offset is None:
            break
          if next_offset <= offset:
            raise ValueError("spill pagination made no progress")
          offset = next_offset
        data = "".join(pages).encode("utf-8")
      arcname = f"spills/{filename}"
      if arcname in spill_names:
        arcname = f"spills/{match.group(2)}/{filename}"
    except Exception as exc:
      warnings.append(f"Lost spill {ref} from {root}: {exc}")
      spills.append((ref, None, True))
      continue
    write_payload(arcname, pointer, data)
    spill_names.add(arcname)
    spills.append((ref, arcname, False))

  manifest = {
    "session_id": session_id,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "files": sorted(files, key=lambda entry: entry["arcname"]),
    "prompts": [{"hash": value, "present": present} for value, present in prompts],
    "spills": [{"ref": ref, "arcname": arcname, "lost": lost} for ref, arcname, lost in spills],
    "warnings": warnings,
  }
  manifest_path = destination / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
  return ArchiveResult(destination, manifest_path, len(files), prompts, spills, warnings)
