from __future__ import annotations

from dataclasses import replace
from uuid import UUID, uuid4

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import jwt
import pytest

from agent_gateway.commercial_claims import VerifiedCommercialClaim
from agent_gateway.commercial_work_authorization import (
  WORK_AUTHORIZATION_AUDIENCE,
  WORK_AUTHORIZATION_ISSUER,
  WorkAuthorizationError,
  WorkAuthorizationTrustSnapshot,
  WorkAuthorizationVerifier,
)


NOW = 1_780_000_000
CONTEXT_ID = uuid4()
WORKFLOW_ID = uuid4()
FUNDING_ROUTE_ID = uuid4()
RESERVATION_ID = uuid4()


def _keypair():
  private = Ed25519PrivateKey.generate()
  public_pem = private.public_key().public_bytes(
    serialization.Encoding.PEM,
    serialization.PublicFormat.SubjectPublicKeyInfo,
  )
  return private, public_pem


def _claim(**changes) -> VerifiedCommercialClaim:
  claim = VerifiedCommercialClaim(
    schema_version=1,
    key_id="commercial-signing-v1",
    subject="user:123",
    environment="prod",
    surface="hp1",
    commercial_account_id=uuid4(),
    agreement_id=uuid4(),
    agreement_terms_revision=2,
    offer_code="hp1_pro",
    effective_scopes=("read", "trade-preview"),
    entitlement_revision=42,
    payer_policy_version="payer@v1",
    budget_policy_version="budget@v1",
    shadow_rate_version="rates@v1",
    manifest_version="manifest@v1",
    authorized_work_start_deadline=NOW + 300,
    usage_accept_until=NOW + 3600,
    issued_at=NOW,
    expires_at=NOW + 300,
    context_id=CONTEXT_ID,
  )
  return replace(claim, **changes)


def _payload(**changes) -> dict:
  payload = {
    "schema_version": 1,
    "kid": "work-signing-v1",
    "iss": WORK_AUTHORIZATION_ISSUER,
    "aud": WORK_AUTHORIZATION_AUDIENCE,
    "jti": str(uuid4()),
    "environment": "prod",
    "execution_claim_jti": str(CONTEXT_ID),
    "workflow_run_id": str(WORKFLOW_ID),
    "workflow_attempt_group_id": str(WORKFLOW_ID),
    "workflow_attempt_number": 1,
    "retry_of_workflow_run_id": None,
    "workflow_attempt_kind": "initial",
    "primary_inference_observability": "hank_metered",
    "funding_route_id": str(FUNDING_ROUTE_ID),
    "provider": "anthropic",
    "billing_mode": "metered",
    "reservation_id": str(RESERVATION_ID),
    "operation": "messages.create",
    "capability_id": "portfolio.review",
    "request_id": "request-1",
    "session_id": "session-1",
    "iat": NOW,
    "exp": NOW + 120,
  }
  payload.update(changes)
  return payload


def _token(private, payload=None, *, headers=None):
  return jwt.encode(
    payload or _payload(),
    private,
    algorithm="EdDSA",
    headers=headers or {"kid": "work-signing-v1", "typ": "JWT"},
  )


def _trust(public_pem: bytes, **changes) -> WorkAuthorizationTrustSnapshot:
  facts = {
    "public_keys_by_id": {"work-signing-v1": public_pem},
    "environment": "prod",
    "manifest_version": "manifest@v1",
    "capability_required_scopes": {
      "portfolio.review": frozenset({"read"}),
    },
    "capability_allowed_operations": {
      "portfolio.review": frozenset({"messages.create"}),
    },
    "retired_at_by_key_id": {"work-signing-v1": NOW + 60},
  }
  facts.update(changes)
  return WorkAuthorizationTrustSnapshot(**facts)


def _verifier(public_pem: bytes, *, now: int = NOW + 1, **trust_changes):
  return WorkAuthorizationVerifier(
    _trust(public_pem, **trust_changes),
    clock=lambda: now,
  )


def _verify(verifier, token, **changes):
  facts = {
    "execution_claim": _claim(),
    "request_id": "request-1",
    "session_id": "session-1",
    "operation": "messages.create",
    "provider": "anthropic",
    "billing_mode": "metered",
    "capability_id": "portfolio.review",
  }
  facts.update(changes)
  return verifier.verify_for_attach(token, **facts)


def test_verifier_accepts_exact_intersection_without_retaining_raw_token() -> None:
  private, public = _keypair()
  token = _token(private)
  verified = _verify(_verifier(public), token)

  assert verified.execution_context_id == CONTEXT_ID
  assert verified.workflow_run_id == WORKFLOW_ID
  assert verified.reservation_id == RESERVATION_ID
  assert verified.billing_mode == "metered"
  assert verified.token_sha256.startswith("sha256:")
  assert token not in repr(verified)
  assert not hasattr(verified, "token")


@pytest.mark.parametrize(
  ("changes", "expected"),
  (
    ({"request_id": "request-2"}, "request_mismatch"),
    ({"session_id": "session-2"}, "session_mismatch"),
    ({"operation": "responses.create"}, "operation_mismatch"),
    ({"provider": "openai"}, "provider_mismatch"),
    ({"billing_mode": "byok"}, "billing_mode_mismatch"),
    ({"capability_id": "trade.execute"}, "capability_mismatch"),
  ),
)
def test_verifier_rejects_request_path_drift(changes, expected) -> None:
  private, public = _keypair()
  with pytest.raises(WorkAuthorizationError, match=expected):
    _verify(_verifier(public), _token(private), **changes)


def test_verifier_intersects_execution_claim_manifest_and_scopes() -> None:
  private, public = _keypair()
  verifier = _verifier(public)
  token = _token(private)
  cases = (
    (_claim(context_id=uuid4()), "execution_claim_mismatch"),
    (_claim(environment="staging"), "execution_claim_environment_mismatch"),
    (_claim(manifest_version="manifest@old"), "manifest_skew"),
    (_claim(effective_scopes=("trade-preview",)), "capability_not_authorized"),
    (_claim(expires_at=NOW + 1), "execution_claim_start_authority_expired"),
    (
      _claim(authorized_work_start_deadline=NOW + 60),
      "execution_claim_window_mismatch",
    ),
  )
  for claim, expected in cases:
    with pytest.raises(WorkAuthorizationError, match=expected):
      _verify(verifier, token, execution_claim=claim)

  unmanifested_operation = _token(
    private,
    _payload(operation="trades.execute"),
  )
  with pytest.raises(WorkAuthorizationError, match="operation_not_authorized"):
    _verify(verifier, unmanifested_operation, operation="trades.execute")


def test_verifier_enforces_key_retirement_iat_and_token_window() -> None:
  private, public = _keypair()
  with pytest.raises(WorkAuthorizationError, match="key_retired"):
    _verify(
      _verifier(public, now=NOW + 62),
      _token(private, _payload(iat=NOW + 61, exp=NOW + 120)),
    )
  with pytest.raises(WorkAuthorizationError, match="start_authority_expired"):
    _verify(
      _verifier(public, now=NOW),
      _token(private, _payload(iat=NOW - 120, exp=NOW)),
    )
  with pytest.raises(WorkAuthorizationError, match="authorization_not_yet_valid"):
    _verify(
      _verifier(public, now=NOW + 1),
      _token(private, _payload(iat=NOW + 2, exp=NOW + 120)),
    )


@pytest.mark.parametrize(
  ("payload_changes", "expected"),
  (
    ({"kid": "other-key"}, "key_id_mismatch"),
    ({"workflow_attempt_number": 2}, "invalid_attempt_shape"),
    ({"reservation_id": None}, "invalid_authority_shape"),
    ({"primary_inference_observability": "hank_byok_observed"},
     "invalid_authority_shape"),
    ({"request_id": "prompt text\nsecret"}, "invalid_request_id"),
  ),
)
def test_verifier_rejects_signed_but_invalid_authority_shape(
  payload_changes, expected
) -> None:
  private, public = _keypair()
  with pytest.raises(WorkAuthorizationError, match=expected):
    _verify(
      _verifier(public),
      _token(private, _payload(**payload_changes)),
    )


def test_verifier_rejects_tamper_unknown_key_and_unsafe_headers() -> None:
  private, public = _keypair()
  wrong_private, _ = _keypair()
  verifier = _verifier(public)
  cases = (
    _token(wrong_private),
    _token(private, headers={"kid": "unknown-key", "typ": "JWT"}),
    _token(
      private,
      headers={"kid": "work-signing-v1", "typ": "JWT", "jku": "https://bad"},
    ),
  )
  for token in cases:
    with pytest.raises(WorkAuthorizationError):
      _verify(verifier, token)

  valid = _token(private)
  header, payload, signature = valid.split(".")
  malleated = f"{header}.{payload}.{signature}="
  with pytest.raises(WorkAuthorizationError, match="invalid_compact_token"):
    _verify(verifier, malleated)


@pytest.mark.parametrize("token", (123, "x" * 4097, "é.y.z", "one.two", "a..c"))
def test_verifier_bounds_untrusted_compact_input(token) -> None:
  _, public = _keypair()
  with pytest.raises(
    WorkAuthorizationError,
    match="invalid_token_type|invalid_compact_token",
  ):
    _verify(_verifier(public), token)


def test_trust_inventory_requires_exact_retirement_and_manifest_maps() -> None:
  _, public = _keypair()
  with pytest.raises(ValueError, match="retirement inventory"):
    _trust(public, retired_at_by_key_id={})
  with pytest.raises(ValueError, match="capability map"):
    _trust(
      public,
      capability_required_scopes={},
      capability_allowed_operations={},
    )
  with pytest.raises(ValueError, match="operation map"):
    _trust(public, capability_allowed_operations={})
  with pytest.raises(ValueError, match="retirement cutoff"):
    _trust(public, retired_at_by_key_id={"work-signing-v1": True})


def test_verifier_freezes_mutable_trust_inputs_at_construction() -> None:
  private, public = _keypair()
  trust = _trust(public)
  verifier = WorkAuthorizationVerifier(trust, clock=lambda: NOW + 62)

  trust.retired_at_by_key_id["work-signing-v1"] = NOW + 600
  trust.capability_required_scopes["portfolio.review"] = frozenset({"trade"})
  trust.capability_allowed_operations["portfolio.review"] = frozenset({
    "trades.execute"
  })

  with pytest.raises(WorkAuthorizationError, match="key_retired"):
    _verify(
      verifier,
      _token(private, _payload(iat=NOW + 61, exp=NOW + 120)),
    )
  with pytest.raises(WorkAuthorizationError, match="operation_not_authorized"):
    _verify(
      verifier,
      _token(private, _payload(operation="trades.execute")),
      operation="trades.execute",
    )


def test_verified_uuid_fields_use_canonical_types() -> None:
  private, public = _keypair()
  verified = _verify(_verifier(public), _token(private))
  assert isinstance(verified.authorization_id, UUID)
  assert isinstance(verified.funding_route_id, UUID)
