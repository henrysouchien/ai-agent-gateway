from __future__ import annotations

from collections.abc import Mapping
from typing import Any


AGENT_CODE_EXECUTE_WORK_DIR_ENV = "AGENT_CODE_EXECUTE_WORK_DIR"

def _valid_computation_entry(value: Any) -> Mapping[str, Any] | None:
  if not isinstance(value, Mapping):
    return None
  for key in ("function", "tool_version", "output_sha256"):
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
      return None
  return value


def collect_native_computations(result: dict[str, Any]) -> None:
  """Project settled native-call provenance without filesystem discovery."""
  computations: list[dict[str, Any]] = []
  for nested_call in result.get("nested_calls", []):
    if not isinstance(nested_call, Mapping):
      continue
    entry = _valid_computation_entry(nested_call.get("provenance"))
    if entry is None:
      continue
    computations.append({**entry, "nested_call_id": nested_call["nested_call_id"]})
  if computations:
    result["computations"] = computations

