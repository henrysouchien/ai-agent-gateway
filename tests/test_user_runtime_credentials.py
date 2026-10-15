from pathlib import Path

import pytest

from agent_gateway.user_runtime_credentials import (
  USER_RUNTIME_CREDENTIALS_FILE_ENV, DeskReplica, UserRuntimeIdentity,
  bundle_from_json, bundle_to_json, load_user_runtime_credentials,
  mint_user_runtime_credentials, write_user_runtime_credentials,
)


def make_bundle():
  return mint_user_runtime_credentials(
    org_claim_key="org-secret", storage_id="alice-storage",
    identity=UserRuntimeIdentity("alice-secret", "cli", "alice", "alice@example.com", 1, "owner", ()),
    desk=DeskReplica("file:///replica", None, None, None, None),
  )


def test_mint_write_load_round_trip(tmp_path: Path) -> None:
  bundle = make_bundle()
  path = tmp_path / "bundle.json"
  write_user_runtime_credentials(bundle, path)
  assert path.stat().st_mode & 0o777 == 0o600
  assert load_user_runtime_credentials({USER_RUNTIME_CREDENTIALS_FILE_ENV: str(path)}) == bundle
  assert bundle_from_json(bundle_to_json(bundle)) == bundle


def test_public_file_mode_rejected(tmp_path: Path) -> None:
  path = tmp_path / "bundle.json"
  write_user_runtime_credentials(make_bundle(), path)
  path.chmod(0o644)
  with pytest.raises(ValueError, match="mode 0644"):
    load_user_runtime_credentials({USER_RUNTIME_CREDENTIALS_FILE_ENV: str(path)})


@pytest.mark.parametrize("field,value", [("scheme", "wrong"), ("user_id", "bob")])
def test_delegation_mismatch_rejected(field: str, value: str) -> None:
  payload = bundle_to_json(make_bundle())
  claim = payload["claim_signing"]
  assert isinstance(claim, dict)
  claim[field] = value
  with pytest.raises(ValueError, match=f"claim_signing.{field}"):
    bundle_from_json(payload)


def test_key_material_redacted() -> None:
  bundle = make_bundle()
  for value in (bundle, bundle.user, bundle.claim_signing):
    assert "alice-secret" not in repr(value)
    assert bundle.claim_signing.key not in repr(value)
  desk = DeskReplica("s3://replica", "access-secret", "aws-secret", "session-secret", "2026-10-13")
  for secret in ("access-secret", "aws-secret", "session-secret"):
    assert secret not in repr(desk)


def test_no_bundle_environment() -> None:
  assert load_user_runtime_credentials({}) is None


def test_existing_file_permissions_are_private(tmp_path: Path) -> None:
  path = tmp_path / "bundle.json"
  path.write_text("old")
  path.chmod(0o644)
  write_user_runtime_credentials(make_bundle(), path)
  assert path.stat().st_mode & 0o777 == 0o600
