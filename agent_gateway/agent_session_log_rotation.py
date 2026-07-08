from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

from .agent_session_log_records import (
  _MANIFEST_SCHEMA_VERSION,
  _atomic_write_json,
  _fsync_parent_dir,
  _now_iso,
)


class _RotationOwner(Protocol):
  path: Path
  segments_dir: Path
  _max_active_bytes: int | None

  def _seq_bounds_for_file(self, path: Path) -> tuple[int, int]: ...

  def _load_manifest(self) -> dict[str, Any] | None: ...

  def _new_manifest(self) -> dict[str, Any]: ...

  def _file_identity(self, path: Path) -> dict[str, int]: ...

  def _load_sidecar_payload(self) -> dict[str, Any] | None: ...

  def _segment_sidecar_payload(
    self,
    base: dict[str, Any],
    *,
    segment_id: str,
    first_seq: int,
    last_seq: int,
    active_generation: int,
    rotated_from_file_identity: dict[str, int],
  ) -> dict[str, Any]: ...

  def _telemetry_source_id(self, role: str, suffix: str) -> str: ...

  def _logical_stream_id(self) -> str: ...

  def _write_manifest(self, manifest: dict[str, Any]) -> None: ...

  def _active_sidecar_payload(self, base: dict[str, Any], *, active_generation: int) -> dict[str, Any]: ...

  def _clear_active_cache(self) -> None: ...


def rotate_active_if_needed_locked(owner: _RotationOwner) -> None:
  if owner._max_active_bytes is None:
    return
  try:
    active_size = owner.path.stat().st_size
  except FileNotFoundError:
    owner.path.touch(exist_ok=True)
    return
  if active_size == 0 or active_size <= owner._max_active_bytes:
    return

  first_seq, last_seq = owner._seq_bounds_for_file(owner.path)
  if first_seq <= 0 or last_seq <= 0:
    return

  manifest = owner._load_manifest() or owner._new_manifest()
  active_generation = int(manifest.get("active_generation") or 0)
  segment_id = f"{first_seq:012d}-{last_seq:012d}-g{active_generation:06d}"
  owner.segments_dir.mkdir(parents=True, exist_ok=True)
  segment_path = owner.segments_dir / f"{segment_id}.jsonl"
  file_identity = owner._file_identity(owner.path)
  base_meta = owner._load_sidecar_payload() or {}

  os.replace(owner.path, segment_path)
  _fsync_parent_dir(owner.segments_dir)
  segment_meta_path = segment_path.with_suffix(".meta.json")
  _atomic_write_json(
    segment_meta_path,
    owner._segment_sidecar_payload(
      base_meta,
      segment_id=segment_id,
      first_seq=first_seq,
      last_seq=last_seq,
      active_generation=active_generation,
      rotated_from_file_identity=file_identity,
    ),
  )

  segments = [item for item in manifest.get("segments", []) if isinstance(item, dict)]
  segments.append(
    {
      "segment_id": segment_id,
      "path": segment_path.name,
      "first_seq": first_seq,
      "last_seq": last_seq,
      "bytes": active_size,
      "telemetry_source_id": owner._telemetry_source_id("segment", segment_id),
      "rotated_from_source_id": owner._telemetry_source_id("active", f"{active_generation:06d}"),
      "rotated_from_path": f"../{owner.path.name}",
      "rotated_from_file_identity": file_identity,
      "created_at": _now_iso(),
      "closed_at": _now_iso(),
    }
  )
  next_generation = active_generation + 1
  manifest.update(
    {
      "schema_version": _MANIFEST_SCHEMA_VERSION,
      "logical_stream_id": owner._logical_stream_id(),
      "active_path": f"../{owner.path.name}",
      "active_generation": next_generation,
      "active_telemetry_source_id": owner._telemetry_source_id("active", f"{next_generation:06d}"),
      "segments": segments,
      "min_seq_available": int(manifest.get("min_seq_available") or 1),
      "latest_seq": max(int(manifest.get("latest_seq") or 0), last_seq),
    }
  )
  owner._write_manifest(manifest)

  owner.path.touch(exist_ok=True)
  if base_meta:
    _atomic_write_json(
      owner.path.with_suffix(".meta.json"),
      owner._active_sidecar_payload(base_meta, active_generation=next_generation),
    )
  owner._clear_active_cache()


def update_manifest_latest_seq_locked(owner: _RotationOwner, seq: int) -> None:
  manifest = owner._load_manifest()
  if manifest is None:
    return
  if seq <= int(manifest.get("latest_seq") or 0):
    return
  manifest["latest_seq"] = seq
  owner._write_manifest(manifest)
