"""Access to the packaged UI blocks contract."""

from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
from typing import Any

_MANIFEST_FILE = "ui_blocks_manifest.v1.json"
_ENVELOPE_SCHEMA_FILE = "schemas/hank_ui_blocks.v1.schema.json"


def packaged_contract_directory() -> Path:
  """Return the installed UI blocks contract directory."""

  return Path(str(files("agent_gateway") / "contracts" / "ui-blocks-v1"))


def _read_json(path: Path) -> Any:
  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise ValueError(f"invalid UI blocks contract JSON: {path}") from exc


def manifest() -> dict[str, Any]:
  """Return the UI blocks manifest."""

  value = _read_json(packaged_contract_directory() / _MANIFEST_FILE)
  if not isinstance(value, dict):
    raise ValueError("UI blocks manifest must be an object")
  return value


def contract_version() -> int:
  """Return the supported contract version."""

  contract = manifest().get("contract")
  if not isinstance(contract, dict) or not isinstance(
    contract.get("contract_version"), int
  ):
    raise ValueError("UI blocks contract version is invalid")
  return contract["contract_version"]


def envelope_schema() -> dict[str, Any]:
  """Return the submitted-payload envelope schema."""

  value = _read_json(packaged_contract_directory() / _ENVELOPE_SCHEMA_FILE)
  if not isinstance(value, dict):
    raise ValueError("UI blocks envelope schema must be an object")
  return value


def fallback_projection_table() -> dict[str, Any]:
  """Return the normative fallback projection table."""

  value = manifest().get("fallback_projections")
  if not isinstance(value, dict):
    raise ValueError("UI blocks fallback projection table is invalid")
  return value
