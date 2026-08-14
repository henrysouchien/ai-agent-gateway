from __future__ import annotations

import codecs
import errno
import hashlib
import io
import json
import os
import re
import stat
import uuid
from collections.abc import Iterator
from pathlib import Path

from agent_workflow_contracts import ContentHandle, TERMINAL_NARRATIVE_CONTRACT

from .sub_agent_result_contract import (
  FinalNarrativeArtifactReference,
)


FINAL_NARRATIVE_ARTIFACT_VERSION = 1
_ARTIFACT_DIRECTORY = "_sub_agent_final_narratives"
_CONTENT_INDEX_DIRECTORY = "_by_content_sha256"
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_TEXT_CHUNK_CHARS = 256 * 1024
_FILE_CHUNK_BYTES = 1024 * 1024


class FinalNarrativeArtifactError(RuntimeError):
  """A canonical child terminal narrative could not be stored or verified."""


def publish_final_narrative(
  *,
  workspace_dir: str | Path,
  sub_agent_id: str,
  terminal_event_seq: int,
  text: str,
) -> FinalNarrativeArtifactReference:
  """Atomically publish one exact visible terminal assistant message."""

  if type(sub_agent_id) is not str or not sub_agent_id.strip():
    raise FinalNarrativeArtifactError("sub_agent_id must be canonical text")
  if type(terminal_event_seq) is not int or terminal_event_seq < 1:
    raise FinalNarrativeArtifactError(
      "terminal_event_seq must be a positive integer"
    )
  if type(text) is not str or not text:
    raise FinalNarrativeArtifactError("final narrative text must be non-empty")
  try:
    content_sha256, content_bytes = _measure_utf8_text(text)
  except UnicodeEncodeError as exc:
    raise FinalNarrativeArtifactError(
      "final narrative text must be valid UTF-8"
    ) from exc
  workspace = _workspace_root(workspace_dir)
  artifact_root = _artifact_root(workspace, create=True)
  source_sub_agent_sha256 = hashlib.sha256(
    sub_agent_id.strip().encode("utf-8")
  ).hexdigest()
  identity_sha256 = hashlib.sha256(
    b"sub-agent-final-narrative.v1\0"
    + bytes.fromhex(source_sub_agent_sha256)
    + b"\0"
    + str(terminal_event_seq).encode("ascii")
    + b"\0"
    + bytes.fromhex(content_sha256)
  ).hexdigest()
  prefix = _safe_child_directory(artifact_root, identity_sha256[:2], create=True)
  final_dir = prefix / identity_sha256
  payload_name = "payload.txt"
  metadata_name = "artifact.json"
  artifact_id = f"sha256:{identity_sha256}"
  artifact_ref = (
    Path("artifacts")
    / _ARTIFACT_DIRECTORY
    / identity_sha256[:2]
    / identity_sha256
    / metadata_name
  ).as_posix()
  reference = FinalNarrativeArtifactReference(
    artifact_id=artifact_id,
    artifact_ref=artifact_ref,
    content_sha256=content_sha256,
    content_chars=len(text),
    content_bytes=content_bytes,
    terminal_event_seq=terminal_event_seq,
  )
  metadata = {
    "artifact_id": artifact_id,
    "artifact_kind": "sub_agent_final_narrative",
    "content_bytes": content_bytes,
    "content_chars": len(text),
    "content_sha256": content_sha256,
    "content_type": "text/plain; charset=utf-8",
    "contract_name": "SubAgentFinalNarrativeArtifact",
    "payload_ref": payload_name,
    "retention": "durable",
    "schema_version": FINAL_NARRATIVE_ARTIFACT_VERSION,
    "source_sub_agent_sha256": source_sub_agent_sha256,
    "terminal_event_seq": terminal_event_seq,
  }
  metadata_bytes = (
    json.dumps(
      metadata,
      ensure_ascii=False,
      separators=(",", ":"),
      sort_keys=True,
    )
    + "\n"
  ).encode("utf-8")
  if final_dir.exists() or final_dir.is_symlink():
    _validate_committed_directory(
      final_dir,
      expected_metadata=metadata_bytes,
      expected_content_bytes=content_bytes,
      expected_content_sha256=content_sha256,
    )
    _publish_content_index(workspace, reference)
    return reference

  temp_dir = prefix / f".{identity_sha256}.{uuid.uuid4().hex}.tmp"
  temp_identity: tuple[int, int] | None = None
  try:
    temp_dir.mkdir(mode=0o700)
    info = temp_dir.lstat()
    temp_identity = (info.st_dev, info.st_ino)
    _write_new_utf8_file(
      temp_dir / payload_name,
      text,
      expected_bytes=content_bytes,
      expected_sha256=content_sha256,
    )
    _write_new_file(temp_dir / metadata_name, metadata_bytes)
    _fsync_directory(temp_dir)
    try:
      os.rename(temp_dir, final_dir)
      temp_identity = None
    except OSError as exc:
      if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
        raise
    if temp_identity is not None:
      _remove_owned_temp_directory(temp_dir, temp_identity)
      temp_identity = None
    _fsync_directory(prefix)
    _validate_committed_directory(
      final_dir,
      expected_metadata=metadata_bytes,
      expected_content_bytes=content_bytes,
      expected_content_sha256=content_sha256,
    )
    _publish_content_index(workspace, reference)
    return reference
  except FinalNarrativeArtifactError:
    if temp_identity is not None:
      _remove_owned_temp_directory(temp_dir, temp_identity)
    raise
  except Exception as exc:
    if temp_identity is not None:
      _remove_owned_temp_directory(temp_dir, temp_identity)
    raise FinalNarrativeArtifactError(
      f"final narrative publication failed: {type(exc).__name__}"
    ) from exc


def read_final_narrative(
  *,
  workspace_dir: str | Path,
  reference: FinalNarrativeArtifactReference,
) -> str:
  """Load and verify one exact final narrative from its trusted reference."""

  if not isinstance(reference, FinalNarrativeArtifactReference):
    raise FinalNarrativeArtifactError(
      "final narrative read requires a typed reference"
    )
  workspace = _workspace_root(workspace_dir)
  identity_sha256 = reference.artifact_id.removeprefix("sha256:")
  if not _HASH_RE.fullmatch(identity_sha256):
    raise FinalNarrativeArtifactError("final narrative artifact id is invalid")
  expected_ref = (
    Path("artifacts")
    / _ARTIFACT_DIRECTORY
    / identity_sha256[:2]
    / identity_sha256
    / "artifact.json"
  ).as_posix()
  if reference.artifact_ref != expected_ref:
    raise FinalNarrativeArtifactError(
      "final narrative artifact path does not match its identity"
    )
  artifact_root = _artifact_root(workspace, create=False)
  final_dir = _safe_child_directory(
    _safe_child_directory(artifact_root, identity_sha256[:2], create=False),
    identity_sha256,
    create=False,
  )
  metadata_path = _regular_file(final_dir / "artifact.json")
  payload_path = _regular_file(final_dir / "payload.txt")
  try:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise FinalNarrativeArtifactError(
      "final narrative metadata is unavailable or invalid"
    ) from exc
  expected_metadata = {
    "artifact_id": reference.artifact_id,
    "artifact_kind": "sub_agent_final_narrative",
    "content_bytes": reference.content_bytes,
    "content_chars": reference.content_chars,
    "content_sha256": reference.content_sha256,
    "content_type": "text/plain; charset=utf-8",
    "contract_name": "SubAgentFinalNarrativeArtifact",
    "payload_ref": "payload.txt",
    "retention": "durable",
    "schema_version": FINAL_NARRATIVE_ARTIFACT_VERSION,
    "terminal_event_seq": reference.terminal_event_seq,
  }
  if not isinstance(metadata, dict) or any(
    metadata.get(key) != value for key, value in expected_metadata.items()
  ):
    raise FinalNarrativeArtifactError(
      "final narrative metadata does not match its reference"
    )
  source_sub_agent_sha256 = metadata.get("source_sub_agent_sha256")
  if (
    not isinstance(source_sub_agent_sha256, str)
    or not _HASH_RE.fullmatch(source_sub_agent_sha256)
  ):
    raise FinalNarrativeArtifactError(
      "final narrative source identity is invalid"
    )
  expected_identity_sha256 = hashlib.sha256(
    b"sub-agent-final-narrative.v1\0"
    + bytes.fromhex(source_sub_agent_sha256)
    + b"\0"
    + str(reference.terminal_event_seq).encode("ascii")
    + b"\0"
    + bytes.fromhex(reference.content_sha256)
  ).hexdigest()
  if identity_sha256 != expected_identity_sha256:
    raise FinalNarrativeArtifactError(
      "final narrative artifact identity failed verification"
    )
  try:
    text, content_sha256, content_bytes, content_chars = (
      _read_utf8_file(payload_path, expected_bytes=reference.content_bytes)
    )
  except OSError as exc:
    raise FinalNarrativeArtifactError(
      "final narrative payload is unavailable"
    ) from exc
  except UnicodeDecodeError as exc:
    raise FinalNarrativeArtifactError(
      "final narrative payload is not UTF-8"
    ) from exc
  if content_bytes != reference.content_bytes:
    raise FinalNarrativeArtifactError("final narrative byte size changed")
  if content_sha256 != reference.content_sha256:
    raise FinalNarrativeArtifactError("final narrative content hash changed")
  if content_chars != reference.content_chars:
    raise FinalNarrativeArtifactError("final narrative character size changed")
  return text


def read_final_narrative_by_content_handle(
  *,
  workspace_dir: str | Path,
  content: ContentHandle,
) -> str:
  """Resolve and verify exact terminal prose without exposing storage paths."""

  if not isinstance(content, ContentHandle):
    raise FinalNarrativeArtifactError(
      "final narrative read requires a typed ContentHandle"
    )
  if (
    content.contract != TERMINAL_NARRATIVE_CONTRACT
    or not content.media_type.lower().startswith("text/")
    or content.encoding != "utf-8"
    or content.retention != "durable"
  ):
    raise FinalNarrativeArtifactError(
      "content handle is not a terminal narrative"
    )
  workspace = _workspace_root(workspace_dir)
  index_path = _content_index_path(
    workspace,
    content.content_sha256,
    create=False,
  )
  try:
    payload = json.loads(_regular_file(index_path).read_text(encoding="utf-8"))
  except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise FinalNarrativeArtifactError(
      "final narrative content index is unavailable or invalid"
    ) from exc
  if not isinstance(payload, dict):
    raise FinalNarrativeArtifactError("final narrative content index is invalid")
  try:
    reference = FinalNarrativeArtifactReference.model_validate(
      payload.get("reference")
    )
  except Exception as exc:
    raise FinalNarrativeArtifactError(
      "final narrative content index reference is invalid"
    ) from exc
  expected = {
    "schema_version": 1,
    "content_id": content.content_id,
    "content_sha256": content.content_sha256,
    "content_chars": content.content_chars,
    "content_bytes": content.content_bytes,
  }
  if any(payload.get(key) != value for key, value in expected.items()):
    raise FinalNarrativeArtifactError(
      "final narrative content index does not match its handle"
    )
  if (
    reference.content_sha256 != content.content_sha256
    or reference.content_chars != content.content_chars
    or reference.content_bytes != content.content_bytes
  ):
    raise FinalNarrativeArtifactError(
      "final narrative reference does not match its content handle"
    )
  return read_final_narrative(
    workspace_dir=workspace,
    reference=reference,
  )


def _publish_content_index(
  workspace: Path,
  reference: FinalNarrativeArtifactReference,
) -> None:
  index_path = _content_index_path(
    workspace,
    reference.content_sha256,
    create=True,
  )
  payload = {
    "schema_version": 1,
    "content_id": f"sha256:{reference.content_sha256}",
    "content_sha256": reference.content_sha256,
    "content_chars": reference.content_chars,
    "content_bytes": reference.content_bytes,
    "reference": reference.model_dump(mode="json"),
  }
  encoded = (
    json.dumps(
      payload,
      ensure_ascii=False,
      separators=(",", ":"),
      sort_keys=True,
    )
    + "\n"
  ).encode("utf-8")
  try:
    _write_new_file(index_path, encoded)
    _fsync_directory(index_path.parent)
  except FileExistsError:
    try:
      existing = _regular_file(index_path).read_bytes()
    except OSError as exc:
      raise FinalNarrativeArtifactError(
        "final narrative content index is unavailable"
      ) from exc
    # Identical terminal bytes may be published by different attempts. The
    # first immutable reference remains the canonical resolver alias; validate
    # its exact content identity rather than overwriting it.
    try:
      existing_payload = json.loads(existing.decode("utf-8"))
      existing_reference = FinalNarrativeArtifactReference.model_validate(
        existing_payload.get("reference")
      )
    except Exception as exc:
      raise FinalNarrativeArtifactError(
        "final narrative content index is invalid"
      ) from exc
    if (
      existing_payload.get("schema_version") != 1
      or existing_payload.get("content_id")
      != f"sha256:{reference.content_sha256}"
      or existing_payload.get("content_sha256") != reference.content_sha256
      or existing_payload.get("content_chars") != reference.content_chars
      or existing_payload.get("content_bytes") != reference.content_bytes
      or existing_reference.content_sha256 != reference.content_sha256
      or existing_reference.content_chars != reference.content_chars
      or existing_reference.content_bytes != reference.content_bytes
    ):
      raise FinalNarrativeArtifactError(
        "final narrative content index conflicts with publication"
      )


def _content_index_path(
  workspace: Path,
  content_sha256: str,
  *,
  create: bool,
) -> Path:
  if not _HASH_RE.fullmatch(content_sha256):
    raise FinalNarrativeArtifactError("final narrative content hash is invalid")
  artifact_root = _artifact_root(workspace, create=create)
  index_root = _safe_child_directory(
    artifact_root,
    _CONTENT_INDEX_DIRECTORY,
    create=create,
  )
  prefix = _safe_child_directory(
    index_root,
    content_sha256[:2],
    create=create,
  )
  return prefix / f"{content_sha256}.json"


def _workspace_root(workspace_dir: str | Path) -> Path:
  try:
    requested = Path(workspace_dir).expanduser()
    requested_info = requested.lstat()
    root = requested.resolve(strict=True)
    info = root.lstat()
  except (OSError, TypeError, ValueError) as exc:
    raise FinalNarrativeArtifactError(
      "durable workspace root is unavailable"
    ) from exc
  if (
    stat.S_ISLNK(requested_info.st_mode)
    or stat.S_ISLNK(info.st_mode)
    or not stat.S_ISDIR(info.st_mode)
  ):
    raise FinalNarrativeArtifactError(
      "durable workspace root must be a non-symlink directory"
    )
  return root


def _artifact_root(workspace: Path, *, create: bool) -> Path:
  artifacts = _safe_child_directory(workspace, "artifacts", create=create)
  return _safe_child_directory(artifacts, _ARTIFACT_DIRECTORY, create=create)


def _safe_child_directory(parent: Path, name: str, *, create: bool) -> Path:
  path = parent / name
  try:
    if create:
      path.mkdir(mode=0o700, exist_ok=True)
    info = path.lstat()
  except OSError as exc:
    raise FinalNarrativeArtifactError(
      "final narrative artifact directory is unavailable"
    ) from exc
  if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
    raise FinalNarrativeArtifactError(
      "final narrative artifact directory is unsafe"
    )
  try:
    path.resolve(strict=True).relative_to(parent.resolve(strict=True))
  except (OSError, ValueError) as exc:
    raise FinalNarrativeArtifactError(
      "final narrative artifact directory escapes its parent"
    ) from exc
  return path


def _regular_file(path: Path) -> Path:
  try:
    info = path.lstat()
  except OSError as exc:
    raise FinalNarrativeArtifactError(
      "final narrative artifact member is unavailable"
    ) from exc
  if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
    raise FinalNarrativeArtifactError(
      "final narrative artifact member is unsafe"
    )
  return path


def _write_new_file(path: Path, data: bytes) -> None:
  flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
  if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
  fd = os.open(path, flags, 0o600)
  try:
    _write_all(fd, data)
    os.fsync(fd)
  finally:
    os.close(fd)


def _measure_utf8_text(text: str) -> tuple[str, int]:
  digest = hashlib.sha256()
  content_bytes = 0
  for chunk in _iter_utf8_chunks(text):
    digest.update(chunk)
    content_bytes += len(chunk)
  return digest.hexdigest(), content_bytes


def _iter_utf8_chunks(text: str) -> Iterator[bytes]:
  for offset in range(0, len(text), _TEXT_CHUNK_CHARS):
    yield text[offset : offset + _TEXT_CHUNK_CHARS].encode("utf-8")


def _write_new_utf8_file(
  path: Path,
  text: str,
  *,
  expected_bytes: int,
  expected_sha256: str,
) -> None:
  flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
  if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
  fd = os.open(path, flags, 0o600)
  digest = hashlib.sha256()
  content_bytes = 0
  try:
    for chunk in _iter_utf8_chunks(text):
      digest.update(chunk)
      content_bytes += len(chunk)
      _write_all(fd, chunk)
    if (
      content_bytes != expected_bytes
      or digest.hexdigest() != expected_sha256
    ):
      raise FinalNarrativeArtifactError(
        "final narrative changed during publication"
      )
    os.fsync(fd)
  finally:
    os.close(fd)


def _write_all(fd: int, data: bytes) -> None:
  view = memoryview(data)
  offset = 0
  while offset < len(view):
    written = os.write(fd, view[offset:])
    if written <= 0:
      raise OSError("final narrative write made no progress")
    offset += written


def _open_regular_file_for_reading(path: Path) -> tuple[int, int]:
  _regular_file(path)
  flags = os.O_RDONLY
  if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
  fd = os.open(path, flags)
  try:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
      raise FinalNarrativeArtifactError(
        "final narrative artifact member is unsafe"
      )
    return fd, info.st_size
  except Exception:
    os.close(fd)
    raise


def _file_fingerprint(
  path: Path,
  *,
  expected_bytes: int,
) -> tuple[str, int]:
  fd, file_size = _open_regular_file_for_reading(path)
  if file_size != expected_bytes:
    os.close(fd)
    return "", file_size
  digest = hashlib.sha256()
  content_bytes = 0
  try:
    while chunk := os.read(fd, _FILE_CHUNK_BYTES):
      digest.update(chunk)
      content_bytes += len(chunk)
  finally:
    os.close(fd)
  return digest.hexdigest(), content_bytes


def _read_utf8_file(
  path: Path,
  *,
  expected_bytes: int,
) -> tuple[str, str, int, int]:
  fd, file_size = _open_regular_file_for_reading(path)
  if file_size != expected_bytes:
    os.close(fd)
    return "", "", file_size, 0
  digest = hashlib.sha256()
  decoder = codecs.getincrementaldecoder("utf-8")("strict")
  output = io.StringIO()
  content_bytes = 0
  content_chars = 0
  try:
    while chunk := os.read(fd, _FILE_CHUNK_BYTES):
      digest.update(chunk)
      content_bytes += len(chunk)
      decoded = decoder.decode(chunk, final=False)
      output.write(decoded)
      content_chars += len(decoded)
    decoded = decoder.decode(b"", final=True)
    output.write(decoded)
    content_chars += len(decoded)
  finally:
    os.close(fd)
  return output.getvalue(), digest.hexdigest(), content_bytes, content_chars


def _validate_committed_directory(
  directory: Path,
  *,
  expected_metadata: bytes,
  expected_content_bytes: int,
  expected_content_sha256: str,
) -> None:
  info = directory.lstat()
  if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
    raise FinalNarrativeArtifactError(
      "committed final narrative artifact is unsafe"
    )
  if {path.name for path in directory.iterdir()} != {
    "artifact.json",
    "payload.txt",
  }:
    raise FinalNarrativeArtifactError(
      "committed final narrative artifact has unexpected members"
    )
  metadata_path = _regular_file(directory / "artifact.json")
  if metadata_path.read_bytes() != expected_metadata:
    raise FinalNarrativeArtifactError(
      "committed final narrative artifact does not match publication"
    )
  payload_path = _regular_file(directory / "payload.txt")
  content_sha256, content_bytes = _file_fingerprint(
    payload_path,
    expected_bytes=expected_content_bytes,
  )
  if (
    content_bytes != expected_content_bytes
    or content_sha256 != expected_content_sha256
  ):
    raise FinalNarrativeArtifactError(
      "committed final narrative artifact does not match publication"
    )


def _remove_owned_temp_directory(
  directory: Path,
  identity: tuple[int, int],
) -> None:
  try:
    info = directory.lstat()
  except FileNotFoundError:
    return
  if (
    stat.S_ISLNK(info.st_mode)
    or not stat.S_ISDIR(info.st_mode)
    or (info.st_dev, info.st_ino) != identity
  ):
    return
  for child in directory.iterdir():
    try:
      child_info = child.lstat()
    except FileNotFoundError:
      continue
    if stat.S_ISREG(child_info.st_mode) and not stat.S_ISLNK(child_info.st_mode):
      child.unlink()
  try:
    directory.rmdir()
  except OSError:
    pass


def _fsync_directory(path: Path) -> None:
  flags = os.O_RDONLY
  if hasattr(os, "O_DIRECTORY"):
    flags |= os.O_DIRECTORY
  fd = os.open(path, flags)
  try:
    os.fsync(fd)
  finally:
    os.close(fd)


__all__ = [
  "FINAL_NARRATIVE_ARTIFACT_VERSION",
  "FinalNarrativeArtifactError",
  "publish_final_narrative",
  "read_final_narrative",
]
