from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .agent_session_log_records import _now_iso


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


def load_sidecar_payload(path: Path) -> dict[str, Any] | None:
  meta_path = path.with_suffix(".meta.json")
  if not meta_path.exists():
    return None
  try:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return None
  return payload if isinstance(payload, dict) else None


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
  return {
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
    for key in ("agent_session_id", "agent_id", "user_id", "product_id", "file_kind", "channel", "profile", "created_at")
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
