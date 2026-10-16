"""Default-off request gate for one authorized commercial provider-work start."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from hank_contracts.commercial_claims import (
  CommercialClaimError,
  VerifiedCommercialClaim,
)
from .commercial_claims import (
  CommercialClaimVerifier,
)
from hank_contracts.commercial_work import (
  COMMERCIAL_CLAIM_HEADER,
  COMMERCIAL_WORK_AUTHORIZATION_HEADER,
  VerifiedWorkAuthorization,
  WorkAuthorizationError,
  CommercialWorkStartError,
  CommercialWorkStartFacts,
  VerifiedCommercialWorkStartClaim,
  PendingCommercialWorkStart,
  CommercialWorkStartContext,
  CommercialWorkStartFactsResolver,
  CommercialWorkStartPreflight,
)
from .commercial_work_authorization import (
  WorkAuthorizationVerifier,
)
from .work_authorization_consumption import (
  WorkAuthorizationAlreadyAttached,
  WorkAuthorizationConsumptionConflict,
  WorkAuthorizationConsumptionError,
  WorkAuthorizationConsumptionRecord,
  WorkAuthorizationConsumptionStore,
)


def require_commercial_child_account(
  work_start: CommercialWorkStartContext | None,
  credential_principal: str,
) -> None:
  """Bind delegated work to the root authorization's credential principal.

  Money (rulings-index WA-ACCOUNT-BUDGET): a child may spend on any provider
  or model. It must spend under the same credential principal, or the call
  would be attributed to a different account.
  """
  if work_start is None:
    return
  if not isinstance(work_start, CommercialWorkStartContext):
    raise CommercialWorkStartError(
      "commercial_child_authority_invalid",
      "Commercial child authority is invalid.",
      status_code=503,
    )
  authorized_principal = work_start.authorization.credential_principal
  if credential_principal != authorized_principal:
    raise CommercialWorkStartError(
      "commercial_child_credential_principal_mismatch",
      "Commercial child credential principal differs from authorized work: "
      f"bind the child to the {authorized_principal!r} principal or start it "
      f"under a work authorization for {credential_principal!r}.",
      status_code=403,
    )




def commercial_budget_stop_fields(producer: object | None) -> dict[str, int] | None:
  """Metered reserved spend already reached, else None.

  The ledger lives on the work-start context the usage producer holds.
  Children share that producer, so they share the ledger. A byok ticket
  has no reservation and never stops. rulings-index WA-ACCOUNT-BUDGET.
  """
  work_start = (
    getattr(producer, "_work_start", None) if producer is not None else None
  )
  ledger = (
    getattr(work_start, "spend_ledger", None)
    if work_start is not None else None
  )
  if ledger is None or not ledger.reached():
    return None
  reserved = ledger.reserved_spend_microusd
  if not isinstance(reserved, int):
    return None
  return {
    "spent_microusd": ledger.spent_microusd,
    "reserved_spend_microusd": reserved,
  }



class CommercialWorkStartGate:
  """Verify request headers and consume one work authorization exactly once."""

  def __init__(
    self,
    *,
    enabled: bool,
    claim_verifier: CommercialClaimVerifier | None = None,
    authorization_verifier: WorkAuthorizationVerifier | None = None,
    consumption_store: WorkAuthorizationConsumptionStore | None = None,
    facts_resolver: CommercialWorkStartFactsResolver | None = None,
    pre_consume: CommercialWorkStartPreflight | None = None,
  ) -> None:
    self.enabled = enabled
    self._claim_verifier = claim_verifier
    self._authorization_verifier = authorization_verifier
    self._consumption_store = consumption_store
    self._facts_resolver = facts_resolver
    self._pre_consume = pre_consume
    self._binding = object()
    if enabled and any(
      dependency is None
      for dependency in (
        claim_verifier,
        authorization_verifier,
        consumption_store,
        facts_resolver,
      )
    ):
      raise ValueError(
        "enabled commercial work start requires both verifiers, a consumption "
        "store, and a trusted facts resolver"
      )

  def verify_claim(
    self,
    headers: Mapping[str, str],
    *,
    session: Any,
  ) -> VerifiedCommercialWorkStartClaim | None:
    if not self.enabled:
      # Headers carry no spend authority on their own: without a gate that
      # verifies them the work runs on the session's own credentials.
      return None
    claim_token = headers.get(COMMERCIAL_CLAIM_HEADER)
    authorization_token = headers.get(COMMERCIAL_WORK_AUTHORIZATION_HEADER)
    if not claim_token or not authorization_token:
      raise CommercialWorkStartError(
        "commercial_work_authority_required",
        "Both commercial work-start headers are required.",
        status_code=401,
      )
    claim_verifier = self._claim_verifier
    if claim_verifier is None:
      raise CommercialWorkStartError(
        "commercial_work_start_unavailable",
        "Commercial work-start verification is unavailable.",
        status_code=503,
      )
    try:
      claim = claim_verifier.verify_for_work_start(claim_token)
    except CommercialClaimError as exc:
      raise CommercialWorkStartError(
        "commercial_work_authority_invalid",
        f"Commercial work-start claim was refused by its verifier: {exc.code}.",
        status_code=403,
      ) from exc
    owner_user_id = str(getattr(session, "owner_user_id", "") or "").strip()
    if (
      not owner_user_id.isdecimal()
      or owner_user_id.startswith("0")
      or claim.subject != f"user:{owner_user_id}"
    ):
      raise CommercialWorkStartError(
        "commercial_work_subject_mismatch",
        "Commercial work-start authority is bound to a different session owner.",
        status_code=403,
      )
    return VerifiedCommercialWorkStartClaim(
      claim=claim,
      _authorization_token=authorization_token,
      _gate_binding=self._binding,
    )

  def authorize_request(
    self,
    verified_claim: VerifiedCommercialWorkStartClaim | None,
    *,
    session: Any,
    request: Any,
    channel: str | None,
  ) -> PendingCommercialWorkStart | None:
    if not self.enabled or verified_claim is None:
      return None
    authorization_verifier = self._authorization_verifier
    facts_resolver = self._facts_resolver
    if authorization_verifier is None or facts_resolver is None:
      raise CommercialWorkStartError(
        "commercial_work_start_unavailable",
        "Commercial work-start verification is unavailable.",
        status_code=503,
      )
    try:
      facts = facts_resolver(session, request, channel)
    except Exception as exc:
      raise CommercialWorkStartError(
        "commercial_work_start_unavailable",
        "Commercial work-start facts could not be resolved.",
        status_code=503,
      ) from exc
    if not isinstance(facts, CommercialWorkStartFacts):
      raise CommercialWorkStartError(
        "commercial_work_start_unavailable",
        "Commercial work-start facts are invalid.",
        status_code=503,
      )
    try:
      authorization = authorization_verifier.verify_for_attach(
        verified_claim._authorization_token,
        execution_claim=verified_claim.claim,
        request_id=request.request_id,
        session_id=session.session_id,
        operation=facts.operation,
        billing_mode=facts.billing_mode,
        capability_id=facts.capability_id,
        credential_principal=facts.credential_principal,
      )
    except WorkAuthorizationError as exc:
      raise CommercialWorkStartError(
        "commercial_work_authority_invalid",
        "Commercial work-start authorization was refused by its verifier: "
        f"{exc.code}.",
        status_code=403,
      ) from exc
    return PendingCommercialWorkStart(
      claim=verified_claim.claim,
      authorization=authorization,
      _gate_binding=verified_claim._gate_binding,
    )

  def consume(
    self,
    pending: PendingCommercialWorkStart | None,
  ) -> CommercialWorkStartContext | None:
    if pending is None:
      return None
    if not self.enabled or pending._gate_binding is not self._binding:
      raise CommercialWorkStartError(
        "commercial_work_authority_invalid",
        "Commercial work-start authority is not bound to this request gate.",
        status_code=403,
      )
    store = self._consumption_store
    if store is None:
      raise CommercialWorkStartError(
        "commercial_work_start_unavailable",
        "Commercial work-start consumption is unavailable.",
        status_code=503,
      )
    if self._pre_consume is not None:
      try:
        self._pre_consume(pending)
      except Exception as exc:
        raise CommercialWorkStartError(
          "commercial_work_start_unavailable",
          "Commercial work-start durability preflight failed.",
          status_code=503,
        ) from exc
    try:
      record = store.attach_once(pending.authorization)
    except WorkAuthorizationAlreadyAttached as exc:
      raise CommercialWorkStartError(
        "commercial_work_authority_already_consumed",
        "Commercial work-start authority has already been consumed.",
        status_code=409,
      ) from exc
    except WorkAuthorizationConsumptionConflict as exc:
      raise CommercialWorkStartError(
        "commercial_work_authority_conflict",
        "Commercial work-start authority conflicts with durable evidence.",
        status_code=409,
      ) from exc
    except WorkAuthorizationConsumptionError as exc:
      raise CommercialWorkStartError(
        "commercial_work_start_unavailable",
        "Commercial work-start authority could not be durably consumed.",
        status_code=503,
      ) from exc
    return CommercialWorkStartContext(
      claim=pending.claim,
      authorization=pending.authorization,
      consumption=record,
    )

  def recheck_irreversible(self, context: CommercialWorkStartContext) -> None:
    """Fail closed on current context/entitlement drift after user approval."""
    verifier = self._claim_verifier
    if not self.enabled or verifier is None:
      raise CommercialWorkStartError(
        "commercial_irreversible_authority_unavailable",
        "Fresh commercial authority is unavailable.",
        status_code=503,
      )
    try:
      verifier.recheck_verified_for_irreversible_submission(context.claim)
    except CommercialClaimError as exc:
      raise CommercialWorkStartError(
        "commercial_irreversible_authority_invalid",
        "Fresh commercial authority is invalid or expired.",
        status_code=403,
      ) from exc

  def uses_context_resolver(self, resolver: Callable) -> bool:
    """Return whether live checks use this exact authority resolver."""
    verifier = self._claim_verifier
    return bool(verifier is not None and verifier.uses_context_resolver(resolver))


__all__ = [
  "CommercialWorkStartGate",
  "require_commercial_child_account",
  "commercial_budget_stop_fields",
]
