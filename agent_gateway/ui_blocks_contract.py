"""Access to the vendored UI blocks contract bundle."""

from __future__ import annotations

import hashlib
from importlib.resources import files
import json
from pathlib import Path
from typing import Any


_DIGEST_FILE = "digest.json"
_MANIFEST_FILE = "ui_blocks_manifest.v1.json"
_ENVELOPE_SCHEMA_FILE = "schemas/hank_ui_blocks.v1.schema.json"
_FIXTURES_DIRECTORY = "fixtures"


def packaged_contract_directory() -> Path:
  """Return the installed UI blocks contract directory."""

  return Path(str(files("agent_gateway") / "contracts" / "ui-blocks-v1"))


def _computed_bundle_digest(directory: Path) -> str:
  """Compute the normative digest without reading ``digest.json``."""

  if not directory.is_dir():
    raise ValueError(f"UI blocks contract directory is absent: {directory}")
  listing = bytearray()
  contract_files = sorted(
    (
      path
      for path in directory.rglob("*")
      if path.is_file() and path.relative_to(directory).as_posix() != _DIGEST_FILE
    ),
    key=lambda path: path.relative_to(directory).as_posix().encode("utf-8"),
  )
  for path in contract_files:
    relative_path = path.relative_to(directory).as_posix()
    file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    listing.extend(f"{relative_path}\n{file_digest}\n".encode("utf-8"))
  return "sha256:" + hashlib.sha256(listing).hexdigest()


def _read_json(path: Path) -> Any:
  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise ValueError(f"invalid UI blocks contract JSON: {path}") from exc


def verify_contract_directory(directory: Path) -> str:
  """Verify bundle self-consistency and return its declared digest."""

  digest_document = _read_json(directory / _DIGEST_FILE)
  if not isinstance(digest_document, dict):
    raise ValueError("UI blocks digest document must be an object")
  declared_digest = digest_document.get("digest")
  if not isinstance(declared_digest, str):
    raise ValueError("UI blocks declared digest is invalid")
  computed_digest = _computed_bundle_digest(directory)
  if declared_digest != computed_digest:
    raise ValueError(
      "UI blocks contract digest mismatch: "
      f"declared {declared_digest}, computed {computed_digest}"
    )
  return declared_digest


def _verified_json(relative_path: str) -> Any:
  directory = packaged_contract_directory()
  verify_contract_directory(directory)
  return _read_json(directory / relative_path)


def bundle_digest() -> str:
  """Return the verified bundle digest."""

  return verify_contract_directory(packaged_contract_directory())


def manifest() -> dict[str, Any]:
  """Return the verified UI blocks manifest."""

  value = _verified_json(_MANIFEST_FILE)
  if not isinstance(value, dict):
    raise ValueError("UI blocks manifest must be an object")
  return value


def contract_version() -> int:
  """Return the verified contract version."""

  contract = manifest().get("contract")
  if not isinstance(contract, dict) or not isinstance(
    contract.get("contract_version"), int
  ):
    raise ValueError("UI blocks contract version is invalid")
  return contract["contract_version"]


def envelope_schema() -> dict[str, Any]:
  """Return the verified submitted-payload envelope schema."""

  value = _verified_json(_ENVELOPE_SCHEMA_FILE)
  if not isinstance(value, dict):
    raise ValueError("UI blocks envelope schema must be an object")
  return value


def fixtures() -> list[dict[str, Any]]:
  """Return all verified positive and negative fixtures by file name."""

  directory = packaged_contract_directory()
  verify_contract_directory(directory)
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
  """Return the verified normative fallback projection table."""

  value = manifest().get("fallback_projections")
  if not isinstance(value, dict):
    raise ValueError("UI blocks fallback projection table is invalid")
  return value


def compatible_digests() -> set[str]:
  """Return foreign bundle digests declared compatible by this bundle."""

  contract = manifest().get("contract")
  if not isinstance(contract, dict):
    raise ValueError("UI blocks contract metadata is invalid")
  value = contract.get("compatible_digests")
  if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
    raise ValueError("UI blocks compatible digests map is invalid")
  return set(value)


def renderable(foreign_digest: str) -> bool:
  """Return whether a foreign bundle can interoperate with this bundle."""

  return foreign_digest == bundle_digest() or foreign_digest in compatible_digests()
