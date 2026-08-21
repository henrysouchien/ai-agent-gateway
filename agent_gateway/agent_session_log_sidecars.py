from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable

from .agent_session_log_records import _atomic_write_sidecar, _now_iso
from .descriptor_paths import DirectoryIdentity, open_directory_chain


V2_STORAGE_IDENTITY_FIELDS = (
  "storage_layout",
  "tenant_id",
  "workload_profile",
  "provider",
  "provider_session_epoch",
  "storage_identity_digest",
)
FLAT_STREAM_IDENTITY_FIELDS = (
  "run_kind",
  "run_seq",
  "batch_id",
  "pipeline_id",
  "ticker",
  "stage",
  "skill",
)
V2_ACTIVE_SIDECAR_FIELDS = frozenset({
  "schema_version",
  "agent_session_id",
  "agent_id",
  "user_id",
  "product_id",
  "tenant_id",
  "file_kind",
  "channel",
  "profile",
  "created_at",
  "file_role",
  "logical_stream_id",
  "telemetry_source_id",
  "active_generation",
  *V2_STORAGE_IDENTITY_FIELDS,
})
V2_SEGMENT_SIDECAR_FIELDS = frozenset({
  *V2_ACTIVE_SIDECAR_FIELDS,
  "segment_id",
  "first_seq",
  "last_seq",
  "rotated_from_source_id",
  "rotated_from_path",
  "rotated_from_file_identity",
})

class AgentSessionLogSidecarError(RuntimeError):
  """A mandatory v2 session-log sidecar is unsafe or contradictory."""


def logical_stream_id(path: Path) -> str:
  return str(path.resolve())


def stream_hash(*, logical_stream_id_fn: Callable[[], str]) -> str:
  return hashlib.sha1(logical_stream_id_fn().encode("utf-8")).hexdigest()[:16]


def telemetry_source_id(role: str, suffix: str, *, stream_hash_fn: Callable[[], str]) -> str:
  return f"agent_session_log:{stream_hash_fn()}:{role}:{suffix}"


def active_sidecar_payload(
  base: dict[str, Any],
  *,
  active_generation: int,
  logical_stream_id_fn: Callable[[], str],
  telemetry_source_id_fn: Callable[[str, str], str],
) -> dict[str, Any]:
  payload = dict(base)
  payload.update(
    {
      "schema_version": 2,
      "file_role": "active",
      "logical_stream_id": logical_stream_id_fn(),
      "telemetry_source_id": telemetry_source_id_fn("active", f"{active_generation:06d}"),
      "active_generation": active_generation,
    }
  )
  return payload


def write_meta_sidecar(
  path: Path,
  session_ref: Any,
  *,
  active_sidecar_payload_fn: Callable[[dict[str, Any]], dict[str, Any]],
  atomic_write_sidecar_fn: Callable[[Path, dict[str, Any]], None] = _atomic_write_sidecar,
  now_iso_fn: Callable[[], str] = _now_iso,
  logger: Any | None = None,
) -> None:
  meta_path = path.with_suffix(".meta.json")
  if meta_path.exists():
    return
  try:
    from .product_config import gateway_product_id

    atomic_write_sidecar_fn(
      meta_path,
      active_sidecar_payload_fn(
        {
          "agent_session_id": session_ref.agent_session_id,
          "agent_id": session_ref.agent_id,
          "user_id": session_ref.user_id,
          "product_id": gateway_product_id() or None,
          "file_kind": "canonical",
          "channel": None,
          "profile": None,
          "created_at": now_iso_fn(),
        },
      ),
    )
  except Exception:
    if logger is not None:
      logger.warning("Sidecar write failed for %s (telemetry-only)", meta_path, exc_info=True)


def load_sidecar_payload(path: Path) -> dict[str, Any] | None:
  meta_path = path.with_suffix(".meta.json")
  if not meta_path.exists():
    return None
  try:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return None
  return payload if isinstance(payload, dict) else None


def _strict_json_object(raw: bytes) -> dict[str, Any]:
  def pairs_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
      if key in result:
        raise AgentSessionLogSidecarError(
          "v2 session-log sidecar contains duplicate fields"
        )
      result[key] = value
    return result

  try:
    payload = json.loads(
      raw.decode("utf-8"),
      object_pairs_hook=pairs_object,
    )
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise AgentSessionLogSidecarError(
      "v2 session-log sidecar is invalid"
    ) from exc
  if not isinstance(payload, dict):
    raise AgentSessionLogSidecarError(
      "v2 session-log sidecar must be an object"
    )
  return payload


def validate_v2_active_sidecar(
  *,
  active_path: Path,
  meta_path: Path,
  parent_identity: DirectoryIdentity,
  meta_device: int,
  meta_inode: int,
  max_bytes: int = 64 * 1024,
) -> dict[str, Any]:
  """Read and validate the exact mandatory v2 active sidecar.

  The sidecar classifies the already authenticated storage stream; it never
  admits a path. The directory and file identities must match the signed and
  descriptor-bound authority before any payload field is trusted.
  """

  active = Path(active_path)
  meta = Path(meta_path)
  if (
    not active.is_absolute()
    or not meta.is_absolute()
    or meta != active.with_suffix(".meta.json")
  ):
    raise AgentSessionLogSidecarError(
      "v2 session-log sidecar path is invalid"
    )
  parent_descriptor = descriptor = -1
  try:
    parent_descriptor, opened_identity = open_directory_chain(meta.parent)
    if opened_identity != parent_identity:
      raise AgentSessionLogSidecarError(
        "v2 session-log sidecar parent identity changed"
      )
    named = os.stat(
      meta.name,
      dir_fd=parent_descriptor,
      follow_symlinks=False,
    )
    descriptor = os.open(
      meta.name,
      os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
      dir_fd=parent_descriptor,
    )
    opened = os.fstat(descriptor)
    for info in (named, opened):
      if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o600
      ):
        raise AgentSessionLogSidecarError(
          "v2 session-log sidecar file is unsafe"
        )
    expected_identity = (meta_device, meta_inode)
    if (
      (named.st_dev, named.st_ino) != expected_identity
      or (opened.st_dev, opened.st_ino) != expected_identity
    ):
      raise AgentSessionLogSidecarError(
        "v2 session-log sidecar identity changed"
      )
    chunks: list[bytes] = []
    total = 0
    while True:
      chunk = os.read(descriptor, min(8192, max_bytes + 1 - total))
      if not chunk:
        break
      chunks.append(chunk)
      total += len(chunk)
      if total > max_bytes:
        raise AgentSessionLogSidecarError(
          "v2 session-log sidecar exceeds its byte bound"
        )
    payload = _strict_json_object(b"".join(chunks))
  except AgentSessionLogSidecarError:
    raise
  except OSError as exc:
    raise AgentSessionLogSidecarError(
      "v2 session-log sidecar is unavailable"
    ) from exc
  finally:
    if descriptor >= 0:
      os.close(descriptor)
    if parent_descriptor >= 0:
      os.close(parent_descriptor)

  # Semantic classification is deliberately owned by
  # agent_session_log_layout.validate_v2_sidecar_payload. Keeping this helper
  # limited to the descriptor-bound read avoids a second schema authority in
  # the rotation module's dependency direction.
  return payload


def segment_sidecar_payload(
  path: Path,
  base: dict[str, Any],
  *,
  segment_id: str,
  first_seq: int,
  last_seq: int,
  active_generation: int,
  rotated_from_file_identity: dict[str, int],
  logical_stream_id_fn: Callable[[], str],
  telemetry_source_id_fn: Callable[[str, str], str],
  now_iso_fn: Callable[[], str] = _now_iso,
) -> dict[str, Any]:
  payload = {
    "schema_version": 2,
    "agent_session_id": str(base.get("agent_session_id") or path.stem),
    "agent_id": base.get("agent_id"),
    "user_id": base.get("user_id"),
    "product_id": base.get("product_id"),
    "file_kind": base.get("file_kind") or "canonical",
    "channel": base.get("channel"),
    "profile": base.get("profile"),
    "created_at": base.get("created_at") or now_iso_fn(),
    "file_role": "segment",
    "logical_stream_id": logical_stream_id_fn(),
    "telemetry_source_id": telemetry_source_id_fn("segment", segment_id),
    "active_generation": active_generation,
    "segment_id": segment_id,
    "first_seq": first_seq,
    "last_seq": last_seq,
    "rotated_from_source_id": telemetry_source_id_fn("active", f"{active_generation:06d}"),
    "rotated_from_path": str(path),
    "rotated_from_file_identity": rotated_from_file_identity,
  }
  payload.update({
    key: base.get(key)
    for key in (*V2_STORAGE_IDENTITY_FIELDS, *FLAT_STREAM_IDENTITY_FIELDS)
    if key in base
  })
  return payload


def fallback_sidecar_base(path: Path, *, now_iso_fn: Callable[[], str] = _now_iso) -> dict[str, Any]:
  user_id = "unknown"
  if path.stem.startswith("agentsess_"):
    remainder = path.stem[len("agentsess_") :]
    if "_" in remainder:
      user_id = remainder.rsplit("_", 1)[1] or "unknown"
  return {
    "agent_session_id": path.stem,
    "agent_id": path.parent.name or "unknown",
    "user_id": user_id,
    "product_id": None,
    "file_kind": "canonical",
    "channel": None,
    "profile": None,
    "created_at": now_iso_fn(),
  }


def sidecar_base_from_segment_meta(meta: dict[str, Any] | None) -> dict[str, Any] | None:
  if not isinstance(meta, dict):
    return None
  base = {
    key: meta.get(key)
    for key in (
      "agent_session_id",
      "agent_id",
      "user_id",
      "product_id",
      "file_kind",
      "channel",
      "profile",
      "created_at",
      *FLAT_STREAM_IDENTITY_FIELDS,
      *V2_STORAGE_IDENTITY_FIELDS,
    )
    if key in meta
  }
  return base or None


def sidecar_base_for_repair(
  segment_metas: list[dict[str, Any]],
  *,
  load_sidecar_payload_fn: Callable[[], dict[str, Any] | None],
  sidecar_base_from_segment_meta_fn: Callable[[dict[str, Any] | None], dict[str, Any] | None],
  fallback_sidecar_base_fn: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
  active_base = load_sidecar_payload_fn()
  if active_base is not None:
    return active_base
  for meta in segment_metas:
    segment_base = sidecar_base_from_segment_meta_fn(meta)
    if segment_base is not None:
      return segment_base
  return fallback_sidecar_base_fn()


def file_identity(path: Path) -> dict[str, int]:
  stat = path.stat()
  return {
    "st_dev": int(stat.st_dev),
    "st_ino": int(stat.st_ino),
    "size": int(stat.st_size),
    "mtime_ns": int(stat.st_mtime_ns),
  }


def new_manifest(
  path: Path,
  *,
  manifest_schema_version: int,
  logical_stream_id_fn: Callable[[], str],
  telemetry_source_id_fn: Callable[[str, str], str],
) -> dict[str, Any]:
  return {
    "schema_version": manifest_schema_version,
    "logical_stream_id": logical_stream_id_fn(),
    "active_path": f"../{path.name}",
    "active_generation": 0,
    "active_telemetry_source_id": telemetry_source_id_fn("active", "000000"),
    "segments": [],
    "min_seq_available": 1,
    "latest_seq": 0,
  }


def load_manifest(manifest_path: Path, log: Any) -> dict[str, Any] | None:
  if not manifest_path.exists():
    return None
  try:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    log.warning("Ignoring unreadable AgentSessionLog manifest: %s", manifest_path)
    return None
  if not isinstance(payload, dict):
    log.warning("Ignoring malformed AgentSessionLog manifest: %s", manifest_path)
    return None
  return payload
