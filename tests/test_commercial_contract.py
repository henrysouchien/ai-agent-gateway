from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from agent_gateway.commercial_contract import (
  CONTRACT_FILES,
  USAGE_V3_CONTRACT_FILES,
  canonical_usage_payload_sha256,
  packaged_contract_directory,
  packaged_usage_v3_contract_directory,
  verify_contract_directory,
  verify_usage_v3_contract_directory,
)


def test_packaged_cross_repository_contract_digests_and_cases_pass() -> None:
  digests = verify_contract_directory(packaged_contract_directory())
  assert set(digests) == CONTRACT_FILES


def test_packaged_commercial_usage_v3_contract_digest_and_schema_pass() -> None:
  digests = verify_usage_v3_contract_directory(
    packaged_usage_v3_contract_directory()
  )
  assert set(digests) == USAGE_V3_CONTRACT_FILES
  pyproject = Path(__file__).parents[1] / "pyproject.toml"
  assert '"contracts/commercial-usage-v3/*.json"' in pyproject.read_text(
    encoding="utf-8"
  )


def test_commercial_usage_v3_contract_tamper_fails_closed(tmp_path: Path) -> None:
  target = tmp_path / "commercial-usage-v3"
  shutil.copytree(packaged_usage_v3_contract_directory(), target)
  schema = target / "commercial-usage-event-v3.schema.json"
  schema.write_text(schema.read_text(encoding="utf-8") + " ", encoding="utf-8")
  with pytest.raises(ValueError, match="digest mismatch"):
    verify_usage_v3_contract_directory(target)


def test_contract_digest_tamper_fails_closed(tmp_path: Path) -> None:
  source = packaged_contract_directory()
  target = tmp_path / "contract"
  shutil.copytree(source, target)
  fixture = target / "commercial-claim-verification-v1.json"
  fixture.write_text(fixture.read_text(encoding="utf-8") + " ", encoding="utf-8")
  with pytest.raises(ValueError, match="digest mismatch"):
    verify_contract_directory(target)


def test_usage_missing_lineage_fails_even_with_updated_digest(tmp_path: Path) -> None:
  source = packaged_contract_directory()
  target = tmp_path / "contract"
  shutil.copytree(source, target)
  path = target / "commercial-v1-fixtures.json"
  payload = json.loads(path.read_text(encoding="utf-8"))
  del payload["usage"]["hank_funded"]["execution_context_id"]
  path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
  _refresh_manifest_digest(target, path)
  with pytest.raises(ValueError, match="schema mismatch"):
    verify_contract_directory(target)


@pytest.mark.parametrize(
  ("field", "value"),
  (("usage_state", "invented"), ("billable_output_tokens", -1)),
)
def test_usage_schema_rejects_invalid_enum_and_type_with_updated_digest(
  tmp_path: Path, field: str, value
) -> None:
  source = packaged_contract_directory()
  target = tmp_path / "contract"
  shutil.copytree(source, target)
  path = target / "commercial-v1-fixtures.json"
  payload = json.loads(path.read_text(encoding="utf-8"))
  payload["usage"]["hank_funded"][field] = value
  path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
  _refresh_manifest_digest(target, path)
  with pytest.raises(ValueError, match="schema mismatch"):
    verify_contract_directory(target)


def test_usage_forged_source_digest_fails_with_updated_manifest(tmp_path: Path) -> None:
  source = packaged_contract_directory()
  target = tmp_path / "contract"
  shutil.copytree(source, target)
  path = target / "commercial-v1-fixtures.json"
  payload = json.loads(path.read_text(encoding="utf-8"))
  payload["usage"]["hank_funded"]["source_payload_sha256"] = "sha256:" + "0" * 64
  path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
  _refresh_manifest_digest(target, path)
  with pytest.raises(ValueError, match="canonical digest mismatch"):
    verify_contract_directory(target)


def test_schema_permitted_numeric_money_uses_shared_float_canonicalization(
  tmp_path: Path,
) -> None:
  source = packaged_contract_directory()
  target = tmp_path / "contract"
  shutil.copytree(source, target)
  path = target / "commercial-v1-fixtures.json"
  payload = json.loads(path.read_text(encoding="utf-8"))
  event = payload["usage"]["hank_funded"]
  event["producer_estimated_cost_usd"] = 1e-7
  event["source_payload_sha256"] = canonical_usage_payload_sha256(event)
  path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
  _refresh_manifest_digest(target, path)
  verify_contract_directory(target)


def test_invalid_idempotency_submission_fails_with_updated_manifest(tmp_path: Path) -> None:
  source = packaged_contract_directory()
  target = tmp_path / "contract"
  shutil.copytree(source, target)
  path = target / "commercial-v1-fixtures.json"
  payload = json.loads(path.read_text(encoding="utf-8"))
  del payload["idempotency"]["submissions"][1]["funding_route_id"]
  path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
  _refresh_manifest_digest(target, path)
  with pytest.raises(ValueError, match="schema mismatch"):
    verify_contract_directory(target)


def test_corrupt_canonical_vector_fails_with_updated_manifest(tmp_path: Path) -> None:
  source = packaged_contract_directory()
  target = tmp_path / "contract"
  shutil.copytree(source, target)
  path = target / "canonicalization-v1-vectors.json"
  payload = json.loads(path.read_text(encoding="utf-8"))
  payload["vectors"][0]["sha256"] = "sha256:" + "0" * 64
  path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
  _refresh_manifest_digest(target, path)
  with pytest.raises(ValueError, match="canonicalization vector mismatch"):
    verify_contract_directory(target)


def _refresh_manifest_digest(directory: Path, artifact: Path) -> None:
  manifest_path = directory / "commercial-contract-digests-v1.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  manifest["digests"][artifact.name] = (
    "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
  )
  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
