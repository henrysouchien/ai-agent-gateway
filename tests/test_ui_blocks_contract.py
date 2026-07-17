from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from agent_gateway import ui_blocks_contract


EXPECTED_DIGEST = (
  "sha256:b8291d9644995448a8d330d40e900fb1a7f930770d933be823d360bf4ef5ba2d"
)


def test_checked_in_bundle_verifies_and_accessors_match() -> None:
  directory = ui_blocks_contract.packaged_contract_directory()

  assert ui_blocks_contract.verify_contract_directory(directory) == EXPECTED_DIGEST
  assert ui_blocks_contract.bundle_digest() == EXPECTED_DIGEST
  assert ui_blocks_contract.contract_version() == 1
  assert ui_blocks_contract.manifest()["contract"]["contract_version"] == 1
  assert ui_blocks_contract.envelope_schema()["$id"] == "hank_ui_blocks.v1"
  assert ui_blocks_contract.fallback_projection_table()["projections"]
  assert ui_blocks_contract.fixtures()
  assert all(
    "expected_code" in fixture
    for fixture in ui_blocks_contract.fixtures()
    if fixture["expectation"] == "reject"
  )


def test_tampered_bundle_fails_closed(tmp_path: Path) -> None:
  copied = tmp_path / "ui-blocks-v1"
  shutil.copytree(ui_blocks_contract.packaged_contract_directory(), copied)
  manifest_path = copied / "ui_blocks_manifest.v1.json"
  manifest_path.write_bytes(manifest_path.read_bytes() + b" ")

  with pytest.raises(ValueError, match="digest mismatch"):
    ui_blocks_contract.verify_contract_directory(copied)


def test_digest_document_is_not_a_hash_input(tmp_path: Path) -> None:
  copied = tmp_path / "ui-blocks-v1"
  shutil.copytree(ui_blocks_contract.packaged_contract_directory(), copied)
  computed_before = ui_blocks_contract._computed_bundle_digest(copied)
  digest_path = copied / "digest.json"
  digest_document = json.loads(digest_path.read_text(encoding="utf-8"))
  digest_document["digest"] = "sha256:" + "0" * 64
  digest_path.write_text(json.dumps(digest_document), encoding="utf-8")

  assert ui_blocks_contract._computed_bundle_digest(copied) == computed_before
  with pytest.raises(ValueError, match="digest mismatch"):
    ui_blocks_contract.verify_contract_directory(copied)


def test_renderable_truth_table(monkeypatch: pytest.MonkeyPatch) -> None:
  compatible = "sha256:" + "1" * 64
  unknown = "sha256:" + "2" * 64
  monkeypatch.setattr(ui_blocks_contract, "compatible_digests", lambda: {compatible})

  assert ui_blocks_contract.renderable(EXPECTED_DIGEST)
  assert ui_blocks_contract.renderable(compatible)
  assert not ui_blocks_contract.renderable(unknown)
