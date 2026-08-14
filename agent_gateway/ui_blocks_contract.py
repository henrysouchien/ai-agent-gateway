"""Access to the packaged UI blocks contract."""

from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
from typing import Any

_MANIFEST_FILE = "ui_blocks_manifest.v1.json"
_ENVELOPE_SCHEMA_FILE = "schemas/hank_ui_blocks.v1.schema.json"
_FIXTURES_DIRECTORY = "fixtures"


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


def fixtures() -> list[dict[str, Any]]:
  """Return all positive and negative fixtures by file name."""

  directory = packaged_contract_directory()
  fixture_directory = directory / _FIXTURES_DIRECTORY
  fixture_paths = sorted(
    fixture_directory.glob("*.json"), key=lambda path: path.name.encode("utf-8")
  )
  if not fixture_paths:
    raise ValueError("UI blocks fixtures are absent")
  loaded: list[dict[str, Any]] = []
  for path in fixture_paths:
    fixture = _read_json(path)
    if not isinstance(fixture, dict):
      raise ValueError(f"UI blocks fixture must be an object: {path.name}")
    if fixture.get("expectation") == "reject" and not isinstance(
      fixture.get("expected_code"), str
    ):
      raise ValueError(f"UI blocks negative fixture lacks expected_code: {path.name}")
    loaded.append(fixture)
  return loaded


def fallback_projection_table() -> dict[str, Any]:
  """Return the normative fallback projection table."""

  value = manifest().get("fallback_projections")
  if not isinstance(value, dict):
    raise ValueError("UI blocks fallback projection table is invalid")
  return value
