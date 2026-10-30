import hashlib
import hmac
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException, Request
from hank_contracts.agent_api import AGENT_API_CLAIM_ENV_VARS, AGENT_API_CLAIM_HEADERS
import pytest

from agent_gateway import server_artifact_helpers
from agent_gateway.claim_signing_authority import GatewayClaimSigningAuthority
from hank_model_transport.user_runtime_credentials import (
  APPROVAL_CLAIM_KEY_SCHEME, USER_RUNTIME_CREDENTIALS_FILE_ENV, DeskReplica, UserRuntimeIdentity,
  bundle_from_json, bundle_storage_id, bundle_to_json, derive_approval_claim_key,
  load_user_runtime_credentials, mint_user_runtime_credentials, write_user_runtime_credentials,
)


def make_bundle(generation: int = 1):
  return mint_user_runtime_credentials(
    org_claim_key="org-secret", storage_id="alice-storage", generation=generation,
    identity=UserRuntimeIdentity("alice-secret", "cli", "alice", "alice@example.com", 1, "owner", ()),
    desk=DeskReplica("file:///replica", None, None, None, None),
  )


@pytest.mark.parametrize("generation", [1, 2, 2**63])
def test_mint_write_load_round_trip(tmp_path: Path, generation: int) -> None:
  bundle = make_bundle(generation)
  assert bundle.generation == generation
  assert bundle_to_json(bundle)["generation"] == generation
  assert bundle.claim_signing.scheme == "user-claim-hmac-derived-v2"
  assert bundle.approval_claim_signing.scheme == "user-claim-approval.v1"
  assert bundle.approval_claim_signing.user_id == bundle.user.slug
  assert bundle.approval_claim_signing.user_email == bundle.user.email
  path = tmp_path / "bundle.json"
  write_user_runtime_credentials(bundle, path)
  assert path.stat().st_mode & 0o777 == 0o600
  assert load_user_runtime_credentials({USER_RUNTIME_CREDENTIALS_FILE_ENV: str(path)}) == bundle
  assert bundle_from_json(bundle_to_json(bundle)) == bundle


@pytest.mark.parametrize("generation", [None, True, False, 0, -1, 1.0, "1", "", [], {}])
def test_bundle_parser_rejects_invalid_generation(generation: object) -> None:
  payload = bundle_to_json(make_bundle())
  payload["generation"] = generation
  with pytest.raises(ValueError, match="generation must be a positive integer"):
    bundle_from_json(payload)


def test_bundle_parser_requires_generation() -> None:
  payload = bundle_to_json(make_bundle())
  del payload["generation"]
  with pytest.raises(ValueError, match="generation must be a positive integer"):
    bundle_from_json(payload)


def test_public_file_mode_rejected(tmp_path: Path) -> None:
  path = tmp_path / "bundle.json"
  write_user_runtime_credentials(make_bundle(), path)
  path.chmod(0o644)
  with pytest.raises(ValueError, match="mode 0644"):
    load_user_runtime_credentials({USER_RUNTIME_CREDENTIALS_FILE_ENV: str(path)})


@pytest.mark.parametrize("delegation", ["claim_signing", "approval_claim_signing"])
@pytest.mark.parametrize(
  "field,value",
  [("scheme", "wrong"), ("scheme", "user-claim-hmac-derived-v3"),
   ("user_id", "bob"), ("user_email", "bob@example.com"), ("key", "")],
)
def test_delegation_mismatch_rejected(delegation: str, field: str, value: str) -> None:
  payload = bundle_to_json(make_bundle())
  claim = payload[delegation]
  assert isinstance(claim, dict)
  claim[field] = value
  with pytest.raises(ValueError, match=f"{delegation}.{field}"):
    bundle_from_json(payload)


def test_bundle_parser_requires_approval_delegation() -> None:
  payload = bundle_to_json(make_bundle())
  del payload["approval_claim_signing"]
  with pytest.raises(ValueError, match="approval_claim_signing must be an object"):
    bundle_from_json(payload)


def test_approval_delegation_does_not_accept_generic_scheme() -> None:
  payload = bundle_to_json(make_bundle())
  payload["approval_claim_signing"] = payload["claim_signing"]
  with pytest.raises(ValueError, match="approval_claim_signing.scheme"):
    bundle_from_json(payload)


def test_key_material_redacted() -> None:
  bundle = make_bundle()
  for value in (bundle, bundle.user, bundle.claim_signing, bundle.approval_claim_signing):
    assert "alice-secret" not in repr(value)
    assert bundle.claim_signing.key not in repr(value)
    assert bundle.approval_claim_signing.key not in repr(value)
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


def test_generic_claim_key_remains_stable_across_runtime_generations(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
  old_bundle = make_bundle(1)
  replacement = make_bundle(2)
  assert old_bundle.claim_signing == replacement.claim_signing
  path = tmp_path / "bundle.json"
  write_user_runtime_credentials(old_bundle, path)
  monkeypatch.setenv(USER_RUNTIME_CREDENTIALS_FILE_ENV, str(path))
  old_authority = GatewayClaimSigningAuthority(old_bundle.claim_signing.key)
  current_authority = GatewayClaimSigningAuthority(replacement.claim_signing.key)
  old_request = _signed_request(old_authority, old_authority, "alice", "alice@example.com")
  assert server_artifact_helpers._artifact_auth_dependency(old_request) == "alice-storage"
  write_user_runtime_credentials(replacement, path)
  old_request.app.state.gateway_claim_signing_authority = current_authority
  assert server_artifact_helpers._artifact_auth_dependency(old_request) == "alice-storage"
  current_request = _signed_request(current_authority, current_authority, "alice", "alice@example.com")
  assert server_artifact_helpers._artifact_auth_dependency(current_request) == "alice-storage"


def test_approval_claim_key_uses_direct_org_hmac_not_generic_delegation() -> None:
  expected = hmac.new(
    b"org-secret",
    b"user-claim-approval.v1\n5:alice\n17:alice@example.com\n1",
    hashlib.sha256,
  ).hexdigest()
  bundle = make_bundle()
  assert APPROVAL_CLAIM_KEY_SCHEME == "user-claim-approval.v1"
  assert derive_approval_claim_key("org-secret", "alice", "alice@example.com", 1) == expected
  assert bundle.approval_claim_signing.key == expected
  assert bundle.claim_signing.key != expected
  assert derive_approval_claim_key(
    bundle.claim_signing.key, "alice", "alice@example.com", 1,
  ) != expected


def test_approval_claim_key_binds_org_identity_and_generation() -> None:
  key = derive_approval_claim_key("org-secret", "alice", "alice@example.com", 1)
  assert len(key) == 64
  assert derive_approval_claim_key("other-org", "alice", "alice@example.com", 1) != key
  assert derive_approval_claim_key("org-secret", "bob", "alice@example.com", 1) != key
  assert derive_approval_claim_key("org-secret", "alice", "bob@example.com", 1) != key
  assert derive_approval_claim_key("org-secret", "alice", "alice@example.com", 2) != key
  assert derive_approval_claim_key("org-secret", "a\nb", "c", 1) != derive_approval_claim_key(
    "org-secret", "a", "b\nc", 1,
  )


def test_approval_claim_key_length_prefixes_unicode_bytes() -> None:
  expected = hmac.new(
    "clé-secrète".encode("utf-8"),
    "user-claim-approval.v1\n10:用户-123\n14:用户@例.com\n1".encode("utf-8"),
    hashlib.sha256,
  ).hexdigest()
  assert derive_approval_claim_key("clé-secrète", "用户-123", "用户@例.com", 1) == expected


@pytest.mark.parametrize("invalid", [None, 123, b"secret", "", " ", " secret", "secret ", "sec\x00ret"])
@pytest.mark.parametrize("field", ["slug", "email"])
def test_bundle_parser_rejects_noncanonical_identity(field: str, invalid: object) -> None:
  payload = bundle_to_json(make_bundle())
  user = payload["user"]
  assert isinstance(user, dict)
  user[field] = invalid
  with pytest.raises(ValueError, match=f"user.{field}"):
    bundle_from_json(payload)


def test_replacement_approval_delegation_rejects_previous_generation_claim() -> None:
  old_bundle = make_bundle(1)
  replacement = make_bundle(2)
  assert old_bundle.approval_claim_signing.key != replacement.approval_claim_signing.key
  old_authority = GatewayClaimSigningAuthority(old_bundle.approval_claim_signing.key)
  current_authority = GatewayClaimSigningAuthority(replacement.approval_claim_signing.key)
  def mint_claim(authority: GatewayClaimSigningAuthority) -> dict[str, str]:
    signed = authority.sign_user_claim(
      user_id="alice", user_email="alice@example.com", ttl_seconds=30,
    )
    return {field: signed[env] for field, env in AGENT_API_CLAIM_ENV_VARS.items()}

  old_claim = mint_claim(old_authority)
  assert old_authority.verify_user_claim(old_claim) is not None
  assert current_authority.verify_user_claim(old_claim) is None
  current_claim = mint_claim(current_authority)
  assert current_authority.verify_user_claim(current_claim) is not None
  generic_authority = GatewayClaimSigningAuthority(replacement.claim_signing.key)
  assert generic_authority.verify_user_claim(current_claim) is None
  generic_claim = mint_claim(generic_authority)
  assert current_authority.verify_user_claim(generic_claim) is None


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
