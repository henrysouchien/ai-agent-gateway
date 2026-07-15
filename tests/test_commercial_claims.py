from __future__ import annotations

import sys
from pathlib import Path
from uuid import UUID, uuid4

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import jwt
import pytest


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.commercial_claims import (  # noqa: E402
  COMMERCIAL_CLAIM_AUDIENCE,
  COMMERCIAL_CLAIM_ISSUER,
  CommercialClaimError,
  CommercialClaimTrustSnapshot,
  CommercialClaimVerifier,
  CommercialContextState,
)
from agent_gateway.commercial_authority_cache import (  # noqa: E402
  CommercialAuthoritySnapshot,
  CommercialAuthorityStateCache,
)
from agent_gateway.commercial_authority_subscriber import (  # noqa: E402
  CommercialAuthoritySubscriber,
)


NOW = 1_780_000_000
AGREEMENT_ID = uuid4()
CONTEXT_ID = uuid4()


def _keypair():
  private = Ed25519PrivateKey.generate()
  public_pem = private.public_key().public_bytes(
    serialization.Encoding.PEM,
    serialization.PublicFormat.SubjectPublicKeyInfo,
  )
  return private, public_pem


def _payload(**changes) -> dict:
  payload = {
    "schema_version": 1,
    "kid": "commercial-signing-v1",
    "iss": COMMERCIAL_CLAIM_ISSUER,
    "aud": COMMERCIAL_CLAIM_AUDIENCE,
    "sub": "user:123",
    "environment": "prod",
    "surface": "hp1",
    "commercial_account_id": str(uuid4()),
    "agreement_id": str(AGREEMENT_ID),
    "agreement_terms_revision": 2,
    "offer_code": "hp1_pro",
    "effective_scopes": ["read", "trade-preview"],
    "entitlement_revision": 42,
    "payer_policy_version": "hp1_customer_host@v1",
    "budget_policy_version": "hp1_pro_budget@v1",
    "shadow_rate_version": "commercial_rates@2026-09-01",
    "manifest_version": "mcp_exposure@2026-09-01",
    "authorized_work_start_deadline": NOW + 300,
    "usage_accept_until": NOW + 3600,
    "iat": NOW,
    "exp": NOW + 300,
    "jti": str(CONTEXT_ID),
  }
  payload.update(changes)
  return payload


def _token(private, payload=None, *, key_id="commercial-signing-v1", algorithm="EdDSA"):
  return jwt.encode(
    payload or _payload(), private, algorithm=algorithm,
    headers={"kid": key_id, "typ": "JWT"},
  )


def _trust(public_pem: bytes, **changes) -> CommercialClaimTrustSnapshot:
  facts = {
    "public_keys_by_id": {"commercial-signing-v1": public_pem},
    "environment": "prod",
    "manifest_versions": frozenset({"mcp_exposure@2026-09-01"}),
    "payer_policy_versions": frozenset({"hp1_customer_host@v1"}),
    "budget_policy_versions": frozenset({"hp1_pro_budget@v1"}),
    "shadow_rate_versions": frozenset({"commercial_rates@2026-09-01"}),
    "current_agreement_terms_revision": lambda agreement_id: (
      2 if agreement_id == AGREEMENT_ID else None
    ),
  }
  facts.update(changes)
  return CommercialClaimTrustSnapshot(**facts)


def test_pushed_event_changes_irreversible_verifier_decision(tmp_path) -> None:
  import asyncio

  private, public = _keypair()
  cache = CommercialAuthorityStateCache(lambda context_id: CommercialAuthoritySnapshot(
    context_id=context_id, active=True, entitlement_revision=42,
    commercial_account_id=7, token_id=None,
  ))
  verifier = CommercialClaimVerifier(_trust(
    public, resolve_context_state=cache.resolve_context_state,
  ))
  claim = verifier.verify_for_work_start(_token(private), now=NOW + 1)
  verifier.recheck_verified_for_irreversible_submission(claim, now=NOW + 1)

  class Client:
    def fetch(self, cursor):
      return {"events": [{
        "sequence_id": 1, "environment": "prod", "kind": "context",
        "commercial_account_id": 7, "entitlement_revision": 43,
        "context_id": str(CONTEXT_ID), "token_id": None,
        "occurred_at": "2026-07-12T12:00:00Z",
      }], "next_sequence": 1, "high_water_sequence": 1}

  subscriber = CommercialAuthoritySubscriber(
    client=Client(), cache=cache, cursor_path=tmp_path / "cursor.json",
  )
  asyncio.run(subscriber.catch_up())
  assert subscriber.cache is cache
  with pytest.raises(CommercialClaimError, match="execution_context_revoked"):
    verifier.recheck_verified_for_irreversible_submission(claim, now=NOW + 1)


def test_verifier_accepts_valid_offline_claim_and_overlapping_keys() -> None:
  old_private, old_public = _keypair()
  _, new_public = _keypair()
  trust = _trust(
    old_public,
    public_keys_by_id={
      "commercial-signing-v1": old_public,
      "commercial-signing-v2": new_public,
    },
  )
  claim = CommercialClaimVerifier(trust).verify_for_work_start(
    _token(old_private), now=NOW + 1
  )
  assert claim.context_id == CONTEXT_ID
  assert claim.effective_scopes == ("read", "trade-preview")
  assert claim.agreement_terms_revision == 2


@pytest.mark.parametrize(
  ("payload_changes", "trust_changes", "now", "expected"),
  (
    ({"schema_version": 2}, {}, NOW + 1, "unknown_schema"),
    ({"environment": "staging"}, {}, NOW + 1, "environment_mismatch"),
    ({"manifest_version": "mcp_exposure@old"}, {}, NOW + 1, "manifest_skew"),
    ({"shadow_rate_version": "unknown@v1"}, {}, NOW + 1, "unknown_rate_policy"),
    ({"agreement_terms_revision": 3}, {}, NOW + 1, "agreement_terms_stale"),
    ({"effective_scopes": ["trade-preview", "read"]}, {}, NOW + 1, "invalid_scopes"),
    ({"iat": NOW + 60, "exp": NOW + 300}, {}, NOW + 1, "claim_not_yet_valid"),
    ({}, {}, NOW + 300, "start_authority_expired"),
    ({"exp": NOW + 301, "authorized_work_start_deadline": NOW + 301}, {}, NOW + 1,
     "invalid_claim_lifetime"),
  ),
)
def test_verifier_fails_closed_on_skew_stale_and_unknown_lineage(
  payload_changes, trust_changes, now, expected
) -> None:
  private, public = _keypair()
  verifier = CommercialClaimVerifier(_trust(public, **trust_changes))
  with pytest.raises(CommercialClaimError, match=expected):
    verifier.verify_for_work_start(
      _token(private, _payload(**payload_changes)), now=now
    )


def test_verifier_rejects_wrong_key_tamper_audience_and_missing_lineage() -> None:
  private, public = _keypair()
  wrong_private, _ = _keypair()
  verifier = CommercialClaimVerifier(_trust(public))

  cases = [
    _token(wrong_private),
    _token(private, _payload(aud="other-gateway")),
    _token(private, {key: value for key, value in _payload().items() if key != "jti"}),
  ]
  valid = _token(private)
  header, body, signature = valid.split(".")
  replacement = "A" if body[5] != "A" else "B"
  cases.append(f"{header}.{body[:5]}{replacement}{body[6:]}.{signature}")
  for token in cases:
    with pytest.raises(CommercialClaimError):
      verifier.verify_for_work_start(token, now=NOW + 1)


@pytest.mark.parametrize("token", (123, "x" * 4097, "é.y.z", "one.two", "a..c"))
def test_verifier_bounds_untrusted_compact_input_before_jwt_parsing(token) -> None:
  _, public = _keypair()
  with pytest.raises(CommercialClaimError, match="invalid_token_type|invalid_compact_token"):
    CommercialClaimVerifier(_trust(public)).verify_for_work_start(token, now=NOW + 1)


def test_verifier_rejects_unsupported_or_critical_protected_headers() -> None:
  private, public = _keypair()
  verifier = CommercialClaimVerifier(_trust(public))
  for headers in (
    {"kid": "commercial-signing-v1", "typ": "JWT", "crit": ["custom"], "custom": True},
    {"kid": "commercial-signing-v1", "typ": "JWT", "jku": "https://example.invalid/jwks"},
  ):
    token = jwt.encode(_payload(), private, algorithm="EdDSA", headers=headers)
    with pytest.raises(CommercialClaimError, match="unsupported_protected_header"):
      verifier.verify_for_work_start(token, now=NOW + 1)


def test_historical_terms_are_rejected_when_current_revision_advanced() -> None:
  private, public = _keypair()
  verifier = CommercialClaimVerifier(_trust(
    public,
    current_agreement_terms_revision=lambda agreement_id: (
      3 if agreement_id == AGREEMENT_ID else None
    ),
  ))
  with pytest.raises(CommercialClaimError, match="agreement_terms_stale"):
    verifier.verify_for_work_start(_token(private), now=NOW + 1)


def test_irreversible_submission_requires_current_live_context() -> None:
  private, public = _keypair()
  token = _token(private)

  revoked = CommercialClaimVerifier(_trust(
    public,
    resolve_context_state=lambda _: CommercialContextState(
      active=False, entitlement_revision=42
    ),
  ))
  with pytest.raises(CommercialClaimError, match="execution_context_revoked"):
    revoked.verify_for_irreversible_submission(token, now=NOW + 1)

  stale = CommercialClaimVerifier(_trust(
    public,
    resolve_context_state=lambda _: CommercialContextState(
      active=True, entitlement_revision=43
    ),
  ))
  with pytest.raises(CommercialClaimError, match="entitlement_revision_stale"):
    stale.verify_for_irreversible_submission(token, now=NOW + 1)

  current = CommercialClaimVerifier(_trust(
    public,
    resolve_context_state=lambda _: CommercialContextState(
      active=True, entitlement_revision=42
    ),
  )).verify_for_irreversible_submission(token, now=NOW + 1)
  assert current.context_id == UUID(str(CONTEXT_ID))


def test_token_free_irreversible_recheck_uses_current_live_context() -> None:
  private, public = _keypair()
  claim = CommercialClaimVerifier(_trust(public)).verify_for_work_start(
    _token(private), now=NOW + 1
  )
  verifier = CommercialClaimVerifier(_trust(
    public,
    resolve_context_state=lambda _: CommercialContextState(
      active=True, entitlement_revision=42
    ),
  ))

  assert verifier.recheck_verified_for_irreversible_submission(
    claim, now=NOW + 2
  ) is None

  stale = CommercialClaimVerifier(_trust(
    public,
    resolve_context_state=lambda _: CommercialContextState(
      active=True, entitlement_revision=43
    ),
  ))
  with pytest.raises(CommercialClaimError, match="entitlement_revision_stale"):
    stale.recheck_verified_for_irreversible_submission(claim, now=NOW + 2)
