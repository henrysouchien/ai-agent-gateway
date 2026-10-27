from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException, Request
from hank_contracts.agent_api import AGENT_API_CLAIM_ENV_VARS, AGENT_API_CLAIM_HEADERS
import pytest

from agent_gateway import server_artifact_helpers
from agent_gateway.claim_signing_authority import GatewayClaimSigningAuthority
from agent_gateway.user_runtime_credentials import (
  USER_RUNTIME_CREDENTIALS_FILE_ENV, DeskReplica, UserRuntimeIdentity,
  bundle_from_json, bundle_storage_id, bundle_to_json, load_user_runtime_credentials,
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


def test_bundle_storage_id_accepts_only_the_bundle_slug_or_email() -> None:
  bundle = make_bundle()
  assert bundle_storage_id(bundle, "alice", None) == "alice-storage"
  assert bundle_storage_id(bundle, "", " Alice@Example.COM ") == "alice-storage"
  assert bundle_storage_id(bundle, "bob", "bob@example.com") is None
  assert bundle_storage_id(bundle, "alice-storage", None) is None
  assert bundle_storage_id(bundle, "", None) is None


def _signed_request(authority: GatewayClaimSigningAuthority, signer: GatewayClaimSigningAuthority,
                    user_id: str, user_email: str) -> Request:
  claim = signer.sign_user_claim(user_id=user_id, user_email=user_email, ttl_seconds=30)
  headers = [
    (AGENT_API_CLAIM_HEADERS[field].lower().encode("latin-1"), claim[env].encode("latin-1"))
    for field, env in AGENT_API_CLAIM_ENV_VARS.items()
  ]
  app = SimpleNamespace(state=SimpleNamespace(gateway_claim_signing_authority=authority))
  return Request({"type": "http", "method": "GET", "path": "/", "headers": headers, "app": app})


@pytest.mark.parametrize(
  "user_id,user_email,signed_by_configured_key,expected",
  [
    ("alice", "alice@example.com", True, "alice-storage"),
    ("alice-renamed", "Alice@Example.com", True, "alice-storage"),
    ("bob", "bob@example.com", True, None),
    ("alice", "alice@example.com", False, None),
  ],
)
def test_artifact_auth_verifies_real_signed_claim_then_maps_through_the_bundle(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
  user_id: str, user_email: str, signed_by_configured_key: bool, expected: str | None,
) -> None:
  bundle = make_bundle()
  path = tmp_path / "bundle.json"
  write_user_runtime_credentials(bundle, path)
  monkeypatch.setenv(USER_RUNTIME_CREDENTIALS_FILE_ENV, str(path))
  configured = GatewayClaimSigningAuthority(bundle.claim_signing.key)
  signer = configured if signed_by_configured_key else GatewayClaimSigningAuthority("x" * 64)
  request = _signed_request(configured, signer, user_id, user_email)
  if expected is None:
    with pytest.raises(HTTPException) as excinfo:
      server_artifact_helpers._artifact_auth_dependency(request)
    assert excinfo.value.status_code == 401
  else:
    assert server_artifact_helpers._artifact_auth_dependency(request) == expected


def test_artifact_auth_without_a_signed_claim_is_refused_under_a_bundle(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
  bundle = make_bundle()
  path = tmp_path / "bundle.json"
  write_user_runtime_credentials(bundle, path)
  monkeypatch.setenv(USER_RUNTIME_CREDENTIALS_FILE_ENV, str(path))
  app = SimpleNamespace(state=SimpleNamespace(
    gateway_claim_signing_authority=GatewayClaimSigningAuthority(bundle.claim_signing.key),
  ))
  request = Request({"type": "http", "method": "GET", "path": "/", "headers": [], "app": app})
  with pytest.raises(HTTPException) as excinfo:
    server_artifact_helpers._artifact_auth_dependency(request)
  assert excinfo.value.status_code == 401
