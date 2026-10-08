from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any


AGENT_CODE_EXECUTE_WORK_DIR_ENV = "AGENT_CODE_EXECUTE_WORK_DIR"

_PROVENANCE_ROOT_NAME = ".agent_provenance"
_MAX_SIDECAR_ENTRIES = 16
_MAX_SIDECAR_BYTES = 8 * 1024
_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_provenance_component(value: Any) -> str:
  text = str(value or "").strip()
  if not text:
    return ""
  return _SAFE_COMPONENT_RE.sub("_", text)[:128]


def _sidecar_dir(work_dir: str | Path, tool_call_id: Any) -> Path | None:
  safe_tool_call_id = _sanitize_provenance_component(tool_call_id)
  if not safe_tool_call_id or safe_tool_call_id in {".", ".."}:
    return None
  return Path(work_dir) / _PROVENANCE_ROOT_NAME / safe_tool_call_id


def delete_computation_sidecar_dir(work_dir: str | Path | None, tool_call_id: Any) -> None:
  """Delete the consumed sidecar directory for one tool call, if it exists."""

  if work_dir is None:
    return
  directory = _sidecar_dir(work_dir, tool_call_id)
  if directory is None:
    return
  shutil.rmtree(directory, ignore_errors=True)


def _valid_computation_entry(value: Any) -> Mapping[str, Any] | None:
  if not isinstance(value, Mapping):
    return None
  for key in ("function", "tool_version", "output_sha256"):
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
      return None
  return value


def _valid_sidecar_entry(value: Any) -> dict[str, Any] | None:
  if not isinstance(value, dict) or value.get("schema_version") != 1:
    return None
  return value if _valid_computation_entry(value) is not None else None


def _read_sidecar_json(path: Path) -> Any:
  with path.open("r", encoding="utf-8") as handle:
    return json.load(handle)


def collect_computation_sidecars(
  result: dict[str, Any],
  *,
  work_dir: str | Path | None,
  tool_call_id: Any,
) -> dict[str, Any]:
  """Attach native ledger provenance followed by bounded legacy sidecars."""

  computations: list[dict[str, Any]] = []
  native_keys: set[tuple[str, str]] = set()
  for nested_call in result.get("nested_calls", []):
    if not isinstance(nested_call, Mapping):
      continue
    entry = _valid_computation_entry(nested_call.get("provenance"))
    if entry is None:
      continue
    computations.append({**entry, "nested_call_id": nested_call["nested_call_id"]})
    native_keys.add((entry["function"], entry["output_sha256"]))
  if computations:
    result["computations"] = computations
  if work_dir is None:
    return result
  directory = _sidecar_dir(work_dir, tool_call_id)
  if directory is None or not directory.is_dir():
    return result

  sidecar_count = 0
  dropped = 0
  try:
    files = sorted(path for path in directory.glob("*.json") if path.is_file())
    for path in files:
      try:
        if path.stat().st_size > _MAX_SIDECAR_BYTES:
          dropped += 1
          continue
        entry = _valid_sidecar_entry(_read_sidecar_json(path))
      except Exception:
        entry = None
      if entry is None:
        dropped += 1
        continue
      if (entry["function"], entry["output_sha256"]) in native_keys:
        continue
      if sidecar_count >= _MAX_SIDECAR_ENTRIES:
        dropped += 1
        continue
      sidecar_count += 1
      computations.append(entry)
  finally:
    shutil.rmtree(directory, ignore_errors=True)

  if computations:
    result["computations"] = computations
  if dropped:
    result["computations_dropped"] = dropped
  return result

