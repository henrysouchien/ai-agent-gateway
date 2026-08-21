"""Descriptor-bound inventory for the selected durable session-log layout.

The filesystem layout admits streams.  Metadata is read only after admission
and may classify a stream, but it can never introduce a path into inventory.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Literal, Mapping

from .agent_session_log import (
  AgentSessionLogEnumerationError,
  AgentSessionLogLocation,
  enumerate_agent_session_log_paths,
)
from .agent_session_log_layout import (
  AgentSessionLogLayoutError,
  SESSION_LOG_LAYOUT_V1,
  SESSION_LOG_LAYOUT_V2,
  SESSION_LOG_V2_DIRECTORY,
  derive_v2_agent_session_log_paths,
  resolve_agent_session_log_layout,
  validate_v2_sidecar_payload,
)
from .agent_session_log_records import (
  _SEGMENT_FILE_RE,
  resolve_agent_session_id,
  slugify,
)
from .descriptor_paths import DirectoryChainSecurityError, absolute_lexical_path, open_directory_chain
from .agent_session_log_sidecars import (
  V2_ACTIVE_SIDECAR_FIELDS,
  V2_SEGMENT_SIDECAR_FIELDS,
  V2_STORAGE_IDENTITY_FIELDS,
)


_MAX_SIDECAR_BYTES = 64 * 1024
_SUPPORTED_SIDECAR_SCHEMAS = frozenset({1, 2})
_FILE_IDENTITY_FIELDS = frozenset({"st_dev", "st_ino", "size", "mtime_ns"})
_SIDECAR_SLUG_RE = re.compile(r"^[a-z0-9-]+$")
_V2_MANIFEST_FIELDS = frozenset({
  "schema_version", "logical_stream_id", "agent_session_id", "active_path",
  "active_generation", "active_telemetry_source_id", "segments",
  "min_seq_available", "latest_seq", *V2_STORAGE_IDENTITY_FIELDS,
})
_V2_MANIFEST_SEGMENT_FIELDS = frozenset({
  "segment_id", "path", "first_seq", "last_seq", "bytes",
  "telemetry_source_id", "rotated_from_source_id", "rotated_from_path",
  "rotated_from_file_identity", "created_at", "closed_at",
})

SessionLogStorageLayout = Literal["v1", "v2"]
SessionLogStreamKind = Literal["canonical", "batch", "pipeline", "ephemeral", "unknown"]
SessionLogFileRole = Literal["active", "segment"]
SessionLogSidecarStatus = Literal["valid", "missing", "invalid", "unsupported"]


class SessionLogInventoryError(RuntimeError):
  """The selected session-log namespace could not be classified safely."""


@dataclass(frozen=True, slots=True)
class SessionLogFileIdentity:
  device: int
  inode: int
  size: int
  mtime_ns: int


@dataclass(frozen=True, slots=True)
class SessionLogPhysicalFile:
  path: Path
  parent_identity: tuple[tuple[int, int], ...]
  file_identity: SessionLogFileIdentity
  role: SessionLogFileRole
  sidecar_path: Path
  sidecar_identity: SessionLogFileIdentity | None
  sidecar_payload: Mapping[str, Any] | None
  classification_payload: Mapping[str, Any] | None
  sidecar_status: SessionLogSidecarStatus


@dataclass(frozen=True, slots=True)
class SelectedAgentSessionLog:
  location: AgentSessionLogLocation
  storage_layout: SessionLogStorageLayout
  stream_kind: SessionLogStreamKind
  sidecar_path: Path
  sidecar_identity: SessionLogFileIdentity | None
  sidecar_payload: Mapping[str, Any] | None
  sidecar_status: SessionLogSidecarStatus
  files: tuple[SessionLogPhysicalFile, ...]
  manifest_path: Path | None = None
  manifest_identity: SessionLogFileIdentity | None = None
  manifest_payload: Mapping[str, Any] | None = None
  manifest_status: SessionLogSidecarStatus = "missing"

  @property
  def path(self) -> Path:
    return self.location.path


def read_session_log_physical_range(
  physical: SessionLogPhysicalFile,
  *,
  offset_lo: int,
  offset_hi: int,
) -> bytes:
  """Read one exact inventoried file range without reopening a free path."""

  if type(physical) is not SessionLogPhysicalFile:
    raise SessionLogInventoryError("session-log physical file must be exact")
  if (
    isinstance(offset_lo, bool)
    or isinstance(offset_hi, bool)
    or not isinstance(offset_lo, int)
    or not isinstance(offset_hi, int)
    or offset_lo < 0
    or offset_hi < offset_lo
    or offset_hi > physical.file_identity.size
  ):
    raise SessionLogInventoryError("session-log physical range is invalid")
  try:
    parent_descriptor, parent_identity = open_directory_chain(
      physical.path.parent
    )
  except DirectoryChainSecurityError as exc:
    raise SessionLogInventoryError(
      "session-log physical parent is unavailable"
    ) from exc
  descriptor = -1
  try:
    if parent_identity != physical.parent_identity:
      raise SessionLogInventoryError(
        "session-log physical parent identity changed"
      )
    named = os.stat(
      physical.path.name,
      dir_fd=parent_descriptor,
      follow_symlinks=False,
    )
    _require_safe_file(named, target="session-log physical file")
    descriptor = os.open(
      physical.path.name,
      os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
      dir_fd=parent_descriptor,
    )
    opened = os.fstat(descriptor)
    _require_safe_file(opened, target="session-log physical file")
    expected = physical.file_identity
    observed = _identity(opened)
    if observed != expected or (
      opened.st_dev,
      opened.st_ino,
    ) != (named.st_dev, named.st_ino):
      raise SessionLogInventoryError(
        "session-log physical file identity changed"
      )
    os.lseek(descriptor, offset_lo, os.SEEK_SET)
    remaining = offset_hi - offset_lo
    chunks: list[bytes] = []
    while remaining:
      chunk = os.read(descriptor, remaining)
      if not chunk:
        raise SessionLogInventoryError(
          "session-log physical file ended before its bound range"
        )
      chunks.append(chunk)
      remaining -= len(chunk)
    return b"".join(chunks)
  except FileNotFoundError as exc:
    raise SessionLogInventoryError(
      "session-log physical file is missing"
    ) from exc
  except OSError as exc:
    raise SessionLogInventoryError(
      "session-log physical range is unavailable"
    ) from exc
  finally:
    if descriptor >= 0:
      os.close(descriptor)
    os.close(parent_descriptor)


def _identity(info: os.stat_result) -> SessionLogFileIdentity:
  return SessionLogFileIdentity(
    device=int(info.st_dev),
    inode=int(info.st_ino),
    size=int(info.st_size),
    mtime_ns=int(info.st_mtime_ns),
  )


def _require_safe_directory(info: os.stat_result, *, target: str) -> None:
  if (
    not stat.S_ISDIR(info.st_mode)
    or info.st_uid != os.geteuid()
    or stat.S_IMODE(info.st_mode) & 0o022
  ):
    raise SessionLogInventoryError(f"{target} is unsafe")


def _require_safe_file(info: os.stat_result, *, target: str) -> None:
  if (
    not stat.S_ISREG(info.st_mode)
    or info.st_uid != os.geteuid()
    or info.st_nlink != 1
    or stat.S_IMODE(info.st_mode) & 0o022
  ):
    raise SessionLogInventoryError(f"{target} is unsafe")


def _open_bound_parent(location: AgentSessionLogLocation) -> int:
  try:
    descriptor, identity = open_directory_chain(location.path.parent)
  except DirectoryChainSecurityError as exc:
    raise SessionLogInventoryError("session-log parent is unavailable") from exc
  try:
    _require_safe_directory(os.fstat(descriptor), target="session-log parent")
    if identity != location.parent_identity:
      raise SessionLogInventoryError("session-log parent identity changed")
  except Exception:
    os.close(descriptor)
    raise
  return descriptor


def _read_json_at(
  parent_descriptor: int,
  name: str,
  *,
  path: Path,
  require_schema: bool = True,
) -> tuple[Mapping[str, Any] | None, SessionLogFileIdentity | None, SessionLogSidecarStatus]:
  descriptor = -1
  try:
    try:
      named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
      return None, None, "missing"
    try:
      _require_safe_file(named, target="session-log sidecar")
      descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_descriptor,
      )
      opened = os.fstat(descriptor)
      _require_safe_file(opened, target="session-log sidecar")
      if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
        raise SessionLogInventoryError("session-log sidecar identity changed")
      chunks: list[bytes] = []
      total = 0
      while True:
        chunk = os.read(descriptor, min(8192, _MAX_SIDECAR_BYTES + 1 - total))
        if not chunk:
          break
        chunks.append(chunk)
        total += len(chunk)
        if total > _MAX_SIDECAR_BYTES:
          return None, _identity(opened), "invalid"
      def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
          if key in result:
            raise ValueError("duplicate JSON object field")
          result[key] = value
        return result

      try:
        payload = json.loads(
          b"".join(chunks).decode("utf-8"),
          object_pairs_hook=reject_duplicate_pairs,
        )
      except (UnicodeDecodeError, ValueError):
        return None, _identity(opened), "invalid"
      if not isinstance(payload, dict):
        return None, _identity(opened), "invalid"
      if require_schema:
        schema = payload.get("schema_version")
        if type(schema) is not int or schema < 1:
          return payload, _identity(opened), "invalid"
        if schema not in _SUPPORTED_SIDECAR_SCHEMAS:
          return payload, _identity(opened), "unsupported"
      return payload, _identity(opened), "valid"
    except SessionLogInventoryError:
      raise
    except OSError as exc:
      raise SessionLogInventoryError(
        f"session-log sidecar is unavailable: {path}"
      ) from exc
  finally:
    if descriptor >= 0:
      os.close(descriptor)


def _open_physical_at(
  parent_descriptor: int,
  parent_identity: tuple[tuple[int, int], ...],
  *,
  path: Path,
  expected_identity: tuple[int, int] | None,
  role: SessionLogFileRole,
  require_absent_when_unbound: bool = False,
) -> SessionLogPhysicalFile | None:
  descriptor = -1
  try:
    try:
      named = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
      if expected_identity is None:
        return None
      raise SessionLogInventoryError("session-log file disappeared") from None
    if expected_identity is None and require_absent_when_unbound:
      raise SessionLogInventoryError("session-log file appeared after enumeration")
    _require_safe_file(named, target="session-log file")
    if expected_identity is not None and (named.st_dev, named.st_ino) != expected_identity:
      raise SessionLogInventoryError("session-log file identity changed")
    descriptor = os.open(
      path.name,
      os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
      dir_fd=parent_descriptor,
    )
    opened = os.fstat(descriptor)
    _require_safe_file(opened, target="session-log file")
    if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
      raise SessionLogInventoryError("session-log file identity changed")
    sidecar_path = path.with_suffix(".meta.json")
    payload, sidecar_identity, sidecar_status = _read_json_at(
      parent_descriptor,
      sidecar_path.name,
      path=sidecar_path,
    )
    return SessionLogPhysicalFile(
      path=path,
      parent_identity=parent_identity,
      file_identity=_identity(opened),
      role=role,
      sidecar_path=sidecar_path,
      sidecar_identity=sidecar_identity,
      sidecar_payload=payload,
      classification_payload=payload,
      sidecar_status=sidecar_status,
    )
  except OSError as exc:
    raise SessionLogInventoryError("session-log file is unavailable") from exc
  finally:
    if descriptor >= 0:
      os.close(descriptor)


def _physical_files(location: AgentSessionLogLocation) -> tuple[SessionLogPhysicalFile, ...]:
  parent_descriptor = _open_bound_parent(location)
  files: list[SessionLogPhysicalFile] = []
  try:
    active = _open_physical_at(
      parent_descriptor,
      location.parent_identity,
      path=location.path,
      expected_identity=location.active_identity,
      role="active",
      require_absent_when_unbound=True,
    )
    if active is not None:
      files.append(active)
    if location.segments_identity is None:
      return tuple(files)
    segments_name = f"{location.path.stem}.segments"
    named = os.stat(segments_name, dir_fd=parent_descriptor, follow_symlinks=False)
    if stat.S_ISLNK(named.st_mode):
      raise SessionLogInventoryError("session-log segment directory is unsafe")
    _require_safe_directory(named, target="session-log segment directory")
    if (named.st_dev, named.st_ino) != location.segments_identity:
      raise SessionLogInventoryError("session-log segment directory identity changed")
    segment_descriptor = os.open(
      segments_name,
      os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
      dir_fd=parent_descriptor,
    )
    try:
      opened = os.fstat(segment_descriptor)
      _require_safe_directory(opened, target="session-log segment directory")
      if (opened.st_dev, opened.st_ino) != location.segments_identity:
        raise SessionLogInventoryError("session-log segment directory identity changed")
      segment_parent_identity = location.parent_identity + (location.segments_identity,)
      for name in sorted(os.listdir(segment_descriptor)):
        if not name.endswith(".jsonl"):
          continue
        if _SEGMENT_FILE_RE.fullmatch(name) is None:
          raise SessionLogInventoryError("session-log segment filename is invalid")
        named_segment = os.stat(
          name,
          dir_fd=segment_descriptor,
          follow_symlinks=False,
        )
        segment = _open_physical_at(
          segment_descriptor,
          segment_parent_identity,
          path=location.path.with_name(segments_name) / name,
          expected_identity=(named_segment.st_dev, named_segment.st_ino),
          role="segment",
        )
        assert segment is not None
        files.append(segment)
    finally:
      os.close(segment_descriptor)
  except SessionLogInventoryError:
    raise
  except OSError as exc:
    raise SessionLogInventoryError("session-log physical inventory changed") from exc
  finally:
    os.close(parent_descriptor)
  return tuple(files)


def _classify_sidecar(payload: Mapping[str, Any] | None) -> SessionLogStreamKind:
  if payload is None:
    return "unknown"
  file_kind = payload.get("file_kind")
  run_kind = payload.get("run_kind")
  if file_kind in {"batch", "pipeline"}:
    if run_kind not in {None, file_kind}:
      return "unknown"
    return file_kind
  if file_kind == "ephemeral":
    return "ephemeral" if run_kind is None else "unknown"
  if file_kind != "canonical":
    return "unknown"
  if run_kind is None:
    return "canonical"
  if run_kind in {"batch", "pipeline"}:
    return run_kind
  return "unknown"


def _active_sidecar(
  location: AgentSessionLogLocation,
  files: tuple[SessionLogPhysicalFile, ...],
) -> tuple[Path, SessionLogFileIdentity | None, Mapping[str, Any] | None, SessionLogSidecarStatus]:
  for item in files:
    if item.role == "active":
      return item.sidecar_path, item.sidecar_identity, item.sidecar_payload, item.sidecar_status
  parent_descriptor = _open_bound_parent(location)
  sidecar_path = location.path.with_suffix(".meta.json")
  try:
    payload, identity, status = _read_json_at(
      parent_descriptor,
      sidecar_path.name,
      path=sidecar_path,
    )
    return sidecar_path, identity, payload, status
  finally:
    os.close(parent_descriptor)


def _manifest_for_location(
  location: AgentSessionLogLocation,
) -> tuple[Path | None, SessionLogFileIdentity | None, Mapping[str, Any] | None, SessionLogSidecarStatus]:
  if location.segments_identity is None:
    return None, None, None, "missing"
  parent_descriptor = _open_bound_parent(location)
  segment_descriptor = -1
  manifest_path = location.path.with_name(f"{location.path.stem}.segments") / "manifest.json"
  try:
    segments_name = manifest_path.parent.name
    named = os.stat(segments_name, dir_fd=parent_descriptor, follow_symlinks=False)
    _require_safe_directory(named, target="session-log segment directory")
    if (named.st_dev, named.st_ino) != location.segments_identity:
      raise SessionLogInventoryError("session-log segment directory identity changed")
    segment_descriptor = os.open(
      segments_name,
      os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
      dir_fd=parent_descriptor,
    )
    payload, identity, status = _read_json_at(
      segment_descriptor,
      manifest_path.name,
      path=manifest_path,
      require_schema=False,
    )
    return manifest_path, identity, payload, status
  except SessionLogInventoryError:
    raise
  except OSError as exc:
    raise SessionLogInventoryError("session-log manifest is unavailable") from exc
  finally:
    if segment_descriptor >= 0:
      os.close(segment_descriptor)
    os.close(parent_descriptor)


def _require_v2_canonical(
  base_dir: Path,
  location: AgentSessionLogLocation,
  trusted_product_id: str,
  payload: Mapping[str, Any] | None,
  status: SessionLogSidecarStatus,
  files: tuple[SessionLogPhysicalFile, ...],
  manifest_payload: Mapping[str, Any] | None,
  manifest_status: SessionLogSidecarStatus,
  *,
  strict_rotated: bool,
  allow_missing_manifest_segments: bool,
) -> None:
  if status != "valid" or payload is None:
    raise SessionLogInventoryError("v2 canonical stream requires valid metadata")
  if set(payload) != V2_ACTIVE_SIDECAR_FIELDS:
    raise SessionLogInventoryError("v2 canonical metadata violates its closed field contract")
  if (
    payload.get("schema_version") != 2
    or payload.get("storage_layout") != 2
    or payload.get("file_role") != "active"
    or _classify_sidecar(payload) != "canonical"
    or payload.get("logical_stream_id") != str(location.path)
    or payload.get("tenant_id") != trusted_product_id
    or payload.get("product_id") != trusted_product_id
  ):
    raise SessionLogInventoryError("v2 canonical metadata contradicts storage")
  try:
    root, active, meta, digest = derive_v2_agent_session_log_paths(
      base_dir,
      tenant=trusted_product_id,
      owner=str(payload["user_id"]),
      workload_profile=str(payload["workload_profile"]),
      provider=str(payload["provider"]),
      provider_session_epoch=(
        None
        if payload["provider_session_epoch"] is None
        else str(payload["provider_session_epoch"])
      ),
    )
  except (TypeError, ValueError) as exc:
    raise SessionLogInventoryError("v2 canonical metadata identity is invalid") from exc
  if (
    root != location.path.parent
    or active != location.path
    or meta != location.path.with_suffix(".meta.json")
    or payload.get("storage_identity_digest") != digest
  ):
    raise SessionLogInventoryError("v2 canonical metadata identity does not match its path")
  try:
    validate_v2_sidecar_payload(
      payload,
      active=location.path,
      digest=digest,
      tenant=trusted_product_id,
      owner=str(payload["user_id"]),
      workload_profile=str(payload["workload_profile"]),
      provider=str(payload["provider"]),
      provider_session_epoch=(
        None
        if payload["provider_session_epoch"] is None
        else str(payload["provider_session_epoch"])
      ),
    )
  except (AgentSessionLogLayoutError, TypeError, ValueError) as exc:
    raise SessionLogInventoryError("v2 canonical metadata is invalid") from exc
  if not strict_rotated:
    return
  identity_fields = {
    "storage_layout": 2,
    "tenant_id": payload["tenant_id"],
    "user_id": payload["user_id"],
    "workload_profile": payload["workload_profile"],
    "provider": payload["provider"],
    "provider_session_epoch": payload["provider_session_epoch"],
    "storage_identity_digest": digest,
  }
  segment_names: set[str] = set()
  for physical in files:
    if physical.role != "segment":
      continue
    segment_names.add(physical.path.name)
    segment_payload = physical.sidecar_payload
    if physical.sidecar_status != "valid" or segment_payload is None:
      raise SessionLogInventoryError("v2 segment requires valid metadata")
    if set(segment_payload) != V2_SEGMENT_SIDECAR_FIELDS:
      raise SessionLogInventoryError("v2 segment metadata violates its closed field contract")
    if any(segment_payload.get(key) != value for key, value in identity_fields.items()):
      raise SessionLogInventoryError("v2 segment metadata identity is inconsistent")
    shared_fields = V2_ACTIVE_SIDECAR_FIELDS - {
      "file_role", "telemetry_source_id", "active_generation",
    }
    if any(segment_payload.get(key) != payload.get(key) for key in shared_fields):
      raise SessionLogInventoryError("v2 segment classification is inconsistent")
    identity = segment_payload.get("rotated_from_file_identity")
    expected_identity = {
      "st_dev": physical.file_identity.device,
      "st_ino": physical.file_identity.inode,
      "size": physical.file_identity.size,
      "mtime_ns": physical.file_identity.mtime_ns,
    }
    if (
      not isinstance(identity, Mapping)
      or set(identity) != _FILE_IDENTITY_FIELDS
      or any(type(identity.get(key)) is not int for key in _FILE_IDENTITY_FIELDS)
      or any(identity.get(key) != value for key, value in expected_identity.items())
    ):
      raise SessionLogInventoryError("v2 segment metadata does not match its physical file")
    filename_match = _SEGMENT_FILE_RE.fullmatch(physical.path.name)
    assert filename_match is not None
    segment_id = physical.path.stem
    generation = int(filename_match.group("generation"))
    stream_prefix = str(payload["telemetry_source_id"]).rsplit(":", 2)[0]
    if (
      segment_payload.get("schema_version") != 2
      or segment_payload.get("file_role") != "segment"
      or segment_payload.get("logical_stream_id") != str(location.path)
      or segment_payload.get("rotated_from_path") != str(location.path)
      or segment_payload.get("segment_id") != segment_id
      or segment_payload.get("first_seq") != int(filename_match.group("first"))
      or segment_payload.get("last_seq") != int(filename_match.group("last"))
      or segment_payload.get("active_generation") != generation
      or segment_payload.get("telemetry_source_id") != f"{stream_prefix}:segment:{segment_id}"
      or segment_payload.get("rotated_from_source_id") != f"{stream_prefix}:active:{generation:06d}"
      or _classify_sidecar(segment_payload) != "canonical"
    ):
      raise SessionLogInventoryError("v2 segment metadata contradicts storage")
  if location.segments_identity is None:
    return
  if manifest_status != "valid" or manifest_payload is None:
    raise SessionLogInventoryError("v2 rotated storage requires a valid manifest")
  if set(manifest_payload) != _V2_MANIFEST_FIELDS:
    raise SessionLogInventoryError("v2 manifest violates its closed field contract")
  manifest_identity_fields = {
    key: payload[key]
    for key in V2_STORAGE_IDENTITY_FIELDS
  }
  if any(manifest_payload.get(key) != value for key, value in manifest_identity_fields.items()):
    raise SessionLogInventoryError("v2 manifest identity is inconsistent")
  if (
    manifest_payload.get("schema_version") != 1
    or manifest_payload.get("logical_stream_id") != str(location.path)
    or manifest_payload.get("agent_session_id") != payload.get("agent_session_id")
    or manifest_payload.get("active_path") != f"../{location.path.name}"
    or manifest_payload.get("active_generation") != payload.get("active_generation")
    or manifest_payload.get("active_telemetry_source_id") != payload.get("telemetry_source_id")
    or not isinstance(manifest_payload.get("segments"), list)
  ):
    raise SessionLogInventoryError("v2 manifest contradicts storage")
  manifest_names: set[str] = set()
  physical_by_name = {
    physical.path.name: physical
    for physical in files
    if physical.role == "segment"
  }
  for descriptor in manifest_payload["segments"]:
    if not isinstance(descriptor, Mapping):
      raise SessionLogInventoryError("v2 manifest segment descriptor is invalid")
    if set(descriptor) != _V2_MANIFEST_SEGMENT_FIELDS:
      raise SessionLogInventoryError("v2 manifest segment violates its closed field contract")
    name = descriptor.get("path")
    if not isinstance(name, str) or _SEGMENT_FILE_RE.fullmatch(name) is None:
      raise SessionLogInventoryError("v2 manifest segment path is invalid")
    if name in manifest_names:
      raise SessionLogInventoryError("v2 manifest contains a duplicate segment")
    manifest_names.add(name)
    physical = physical_by_name.get(name)
    if physical is None:
      if allow_missing_manifest_segments:
        continue
      raise SessionLogInventoryError("v2 manifest names absent physical storage")
    segment_payload = physical.sidecar_payload
    assert segment_payload is not None
    if (
      descriptor.get("segment_id") != segment_payload.get("segment_id")
      or descriptor.get("first_seq") != segment_payload.get("first_seq")
      or descriptor.get("last_seq") != segment_payload.get("last_seq")
      or descriptor.get("bytes") != physical.file_identity.size
      or descriptor.get("telemetry_source_id") != segment_payload.get("telemetry_source_id")
      or descriptor.get("rotated_from_source_id") != segment_payload.get("rotated_from_source_id")
      or descriptor.get("rotated_from_file_identity") != segment_payload.get("rotated_from_file_identity")
      or descriptor.get("rotated_from_path") not in {
        str(location.path), f"../{location.path.name}",
      }
    ):
      raise SessionLogInventoryError("v2 manifest segment contradicts physical storage")
  if (
    not segment_names.issubset(manifest_names)
    or (not allow_missing_manifest_segments and manifest_names != segment_names)
  ):
    raise SessionLogInventoryError("v2 manifest does not describe exact segment storage")


def _require_flat_stream(
  item: SelectedAgentSessionLog,
  *,
  strict_rotated: bool,
  allow_missing_manifest_segments: bool,
) -> str:
  payload = item.sidecar_payload
  if item.sidecar_status != "valid" or payload is None or item.stream_kind == "unknown":
    raise SessionLogInventoryError("selected flat session log cannot be classified")
  for field in ("agent_session_id", "agent_id", "user_id", "product_id"):
    if not isinstance(payload.get(field), str) or not payload.get(field):
      raise SessionLogInventoryError("retained flat metadata has no exact identity")
  if (
    payload.get("tenant_id") not in {None, payload.get("product_id")}
  ):
    raise SessionLogInventoryError("retained flat metadata has ambiguous tenant attribution")
  if payload.get("schema_version") == 2 and (
    payload.get("file_role") != "active"
    or payload.get("logical_stream_id") != str(item.path)
  ):
    raise SessionLogInventoryError("retained flat metadata contradicts its physical stream")
  if slugify(str(payload["agent_id"])) != item.path.parent.name:
    raise SessionLogInventoryError("retained flat metadata does not match its family directory")
  if item.stream_kind in {"canonical", "batch", "pipeline"}:
    if payload.get("agent_session_id") != item.path.stem:
      raise SessionLogInventoryError("retained run metadata does not match its active path")
    expected = resolve_agent_session_id(str(payload["user_id"]), str(payload["agent_id"]))
    if expected != item.path.stem:
      raise SessionLogInventoryError("retained run metadata does not match its stream identity")
  if item.stream_kind in {"batch", "pipeline"}:
    family_field = "batch_id" if item.stream_kind == "batch" else "pipeline_id"
    if (
      isinstance(payload.get("run_seq"), bool)
      or not isinstance(payload.get("run_seq"), int)
      or payload.get(family_field) in {None, ""}
    ):
      raise SessionLogInventoryError("retained run metadata has no exact family attribution")
  elif item.stream_kind == "ephemeral":
    if (
      not isinstance(payload.get("channel"), str)
      or _SIDECAR_SLUG_RE.fullmatch(str(payload["channel"])) is None
    ):
      raise SessionLogInventoryError("retained ephemeral metadata has no channel")
    if (
      not isinstance(payload.get("profile"), str)
      or _SIDECAR_SLUG_RE.fullmatch(str(payload["profile"])) is None
    ):
      raise SessionLogInventoryError("retained ephemeral metadata has no profile")
    raw_session_id = str(payload["agent_session_id"])
    epoch_suffix = ""
    session_slug = raw_session_id
    if "--openai-" in raw_session_id:
      session_slug, raw_epoch = raw_session_id.rsplit("--openai-", 1)
      if not session_slug or not raw_epoch:
        raise SessionLogInventoryError(
          "retained ephemeral metadata has an invalid provider epoch"
        )
      epoch_suffix = f"--openai-{raw_epoch}"
    expected_stem = (
      f"agentsess_{slugify(session_slug)}_"
      f"{slugify(str(payload['user_id']))}{epoch_suffix}"
    )
    if item.path.stem != expected_stem:
      raise SessionLogInventoryError(
        "retained ephemeral metadata does not match its active path"
      )
  if not strict_rotated:
    return str(payload["product_id"])
  segment_names: set[str] = set()
  for physical in item.files:
    if physical.role != "segment":
      continue
    segment_names.add(physical.path.name)
    segment_payload = physical.sidecar_payload
    if physical.sidecar_status != "valid" or segment_payload is None:
      raise SessionLogInventoryError(
        "retained flat segment requires valid metadata"
      )
    for field in (
      "agent_session_id", "agent_id", "user_id", "product_id",
      "file_kind", "channel", "profile",
    ):
      if field in segment_payload and segment_payload.get(field) != payload.get(field):
        raise SessionLogInventoryError(
          "retained flat segment classification is inconsistent"
        )
    if segment_payload.get("tenant_id") not in {
      None,
      payload.get("product_id"),
    }:
      raise SessionLogInventoryError(
        "retained flat segment tenant attribution is ambiguous"
      )
    if (
      segment_payload.get("run_kind") not in {None, payload.get("run_kind")}
      or segment_payload.get("schema_version") != 2
      or segment_payload.get("file_role") != "segment"
      or segment_payload.get("logical_stream_id") != str(item.path)
      or segment_payload.get("rotated_from_path") != str(item.path)
      or segment_payload.get("segment_id") != physical.path.stem
    ):
      raise SessionLogInventoryError(
        "retained flat segment metadata contradicts its stream"
      )
    filename_match = _SEGMENT_FILE_RE.fullmatch(physical.path.name)
    assert filename_match is not None
    expected_identity = {
      "st_dev": physical.file_identity.device,
      "st_ino": physical.file_identity.inode,
      "size": physical.file_identity.size,
      "mtime_ns": physical.file_identity.mtime_ns,
    }
    if (
      segment_payload.get("first_seq") != int(filename_match.group("first"))
      or segment_payload.get("last_seq") != int(filename_match.group("last"))
      or segment_payload.get("rotated_from_file_identity")
      != expected_identity
    ):
      raise SessionLogInventoryError(
        "retained flat segment metadata does not match physical storage"
      )
  if item.location.segments_identity is not None:
    manifest = item.manifest_payload
    if item.manifest_status != "valid" or manifest is None:
      raise SessionLogInventoryError(
        "retained flat rotated storage requires a valid manifest"
      )
    if (
      manifest.get("logical_stream_id") != str(item.path)
      or manifest.get("active_path") != f"../{item.path.name}"
      or not isinstance(manifest.get("segments"), list)
    ):
      raise SessionLogInventoryError(
        "retained flat manifest contradicts its stream"
      )
    manifest_names = {
      descriptor.get("path")
      for descriptor in manifest["segments"]
      if isinstance(descriptor, Mapping)
      and isinstance(descriptor.get("path"), str)
    }
    if (
      len(manifest_names) != len(manifest["segments"])
      or not segment_names.issubset(manifest_names)
      or (
        not allow_missing_manifest_segments
        and manifest_names != segment_names
      )
    ):
      raise SessionLogInventoryError(
        "retained flat manifest does not describe exact segments"
      )
    physical_by_name = {
      physical.path.name: physical
      for physical in item.files
      if physical.role == "segment"
    }
    for descriptor in manifest["segments"]:
      assert isinstance(descriptor, Mapping)
      physical = physical_by_name.get(str(descriptor["path"]))
      if physical is None:
        continue
      segment_payload = physical.sidecar_payload
      assert segment_payload is not None
      if (
        descriptor.get("segment_id") != segment_payload.get("segment_id")
        or descriptor.get("first_seq") != segment_payload.get("first_seq")
        or descriptor.get("last_seq") != segment_payload.get("last_seq")
        or descriptor.get("bytes") != physical.file_identity.size
        or descriptor.get("telemetry_source_id")
        != segment_payload.get("telemetry_source_id")
      ):
        raise SessionLogInventoryError(
          "retained flat manifest contradicts physical segments"
        )
  return str(payload["product_id"])


def _bind_flat_file_classification(
  item: SelectedAgentSessionLog,
) -> SelectedAgentSessionLog:
  active = item.sidecar_payload
  assert active is not None
  classification_fields = (
    "agent_session_id", "agent_id", "user_id", "product_id", "tenant_id",
    "file_kind", "run_kind", "channel", "profile", "batch_id",
    "pipeline_id", "run_seq", "ticker", "stage", "skill",
  )
  files: list[SessionLogPhysicalFile] = []
  for physical in item.files:
    if physical.role == "active":
      files.append(physical)
      continue
    merged = dict(physical.sidecar_payload or {})
    for field in classification_fields:
      if field in active:
        merged[field] = active[field]
      else:
        merged.pop(field, None)
    files.append(replace(physical, classification_payload=merged))
  return replace(item, files=tuple(files))


def _inventory_item(
  location: AgentSessionLogLocation,
  *,
  storage_layout: SessionLogStorageLayout,
  base_dir: Path,
  trusted_product_id: str | None = None,
  strict_rotated: bool = True,
  allow_missing_manifest_segments: bool = False,
) -> SelectedAgentSessionLog:
  files = _physical_files(location)
  sidecar_path, sidecar_identity, payload, status = _active_sidecar(location, files)
  manifest_path, manifest_identity, manifest_payload, manifest_status = (
    _manifest_for_location(location)
  )
  kind = _classify_sidecar(payload) if status == "valid" else "unknown"
  if storage_layout == SESSION_LOG_LAYOUT_V2:
    if trusted_product_id is None:
      raise SessionLogInventoryError("v2 inventory requires a configured tenant")
    _require_v2_canonical(
      base_dir,
      location,
      trusted_product_id,
      payload,
      status,
      files,
      manifest_payload,
      manifest_status,
      strict_rotated=strict_rotated,
      allow_missing_manifest_segments=allow_missing_manifest_segments,
    )
    kind = "canonical"
  return SelectedAgentSessionLog(
    location=location,
    storage_layout=storage_layout,
    stream_kind=kind,
    sidecar_path=sidecar_path,
    sidecar_identity=sidecar_identity,
    sidecar_payload=payload,
    sidecar_status=status,
    files=files,
    manifest_path=manifest_path,
    manifest_identity=manifest_identity,
    manifest_payload=manifest_payload,
    manifest_status=manifest_status,
  )


def enumerate_selected_agent_session_logs(
  base_dir: str | Path,
  *,
  layout: SessionLogStorageLayout | None = None,
  trusted_product_id: str | None,
  allowed_product_ids: frozenset[str],
  allowed_stream_kinds: frozenset[SessionLogStreamKind],
  strict_rotated: bool = True,
  allow_missing_manifest_segments: bool = False,
) -> tuple[SelectedAgentSessionLog, ...]:
  """Return the streams selected by the current v1/v2 operational choice."""

  base = absolute_lexical_path(base_dir)
  selected_layout = resolve_agent_session_log_layout() if layout is None else layout
  if selected_layout not in {SESSION_LOG_LAYOUT_V1, SESSION_LOG_LAYOUT_V2}:
    raise SessionLogInventoryError("session-log layout is unsupported")
  supported_kinds = frozenset({"canonical", "batch", "pipeline", "ephemeral"})
  if not allowed_stream_kinds or not allowed_stream_kinds.issubset(supported_kinds):
    raise SessionLogInventoryError("selected session-log stream kinds are invalid")
  if (
    not trusted_product_id
    or not allowed_product_ids
    or trusted_product_id not in allowed_product_ids
    or any(not isinstance(product_id, str) or not product_id for product_id in allowed_product_ids)
  ):
    raise SessionLogInventoryError("session-log inventory requires exact allowed tenants")
  try:
    flat_locations = enumerate_agent_session_log_paths(base)
  except AgentSessionLogEnumerationError as exc:
    raise SessionLogInventoryError("flat session-log inventory is unavailable") from exc
  items: list[SelectedAgentSessionLog] = []
  for location in flat_locations:
    item = _inventory_item(
      location,
      storage_layout=SESSION_LOG_LAYOUT_V1,
      base_dir=base,
      allow_missing_manifest_segments=allow_missing_manifest_segments,
    )
    product_id = _require_flat_stream(
      item,
      strict_rotated=strict_rotated,
      allow_missing_manifest_segments=allow_missing_manifest_segments,
    )
    item = _bind_flat_file_classification(item)
    if product_id not in allowed_product_ids:
      continue
    if selected_layout == SESSION_LOG_LAYOUT_V1:
      if item.stream_kind in allowed_stream_kinds:
        items.append(item)
      continue
    if item.stream_kind != "canonical":
      if item.stream_kind in allowed_stream_kinds:
        items.append(item)
  if selected_layout == SESSION_LOG_LAYOUT_V2:
    v2_root = base / SESSION_LOG_V2_DIRECTORY
    try:
      v2_locations = enumerate_agent_session_log_paths(v2_root)
    except AgentSessionLogEnumerationError as exc:
      raise SessionLogInventoryError("v2 session-log inventory is unavailable") from exc
    for location in v2_locations:
      item = _inventory_item(
        location,
        storage_layout=SESSION_LOG_LAYOUT_V2,
        base_dir=base,
        trusted_product_id=trusted_product_id,
        strict_rotated=strict_rotated,
        allow_missing_manifest_segments=allow_missing_manifest_segments,
      )
      if item.stream_kind in allowed_stream_kinds:
        items.append(item)
  by_path: dict[Path, SelectedAgentSessionLog] = {}
  for item in items:
    if item.path in by_path:
      raise SessionLogInventoryError("selected session-log inventory is ambiguous")
    by_path[item.path] = item
  return tuple(by_path[path] for path in sorted(by_path))


__all__ = [
  "SelectedAgentSessionLog",
  "SessionLogFileIdentity",
  "SessionLogInventoryError",
  "SessionLogPhysicalFile",
  "enumerate_selected_agent_session_logs",
]
