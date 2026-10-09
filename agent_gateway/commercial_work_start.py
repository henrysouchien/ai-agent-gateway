"""Default-off request gate for one authorized commercial provider-work start."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .commercial_claims import (
  CommercialClaimError,
  CommercialClaimVerifier,
  VerifiedCommercialClaim,
)
from .commercial_work_authorization import (
  VerifiedWorkAuthorization,
  WorkAuthorizationError,
  WorkAuthorizationVerifier,
)
from .work_authorization_consumption import (
  WorkAuthorizationAlreadyAttached,
  WorkAuthorizationConsumptionConflict,
  WorkAuthorizationConsumptionError,
  WorkAuthorizationConsumptionRecord,
  WorkAuthorizationConsumptionStore,
)


COMMERCIAL_CLAIM_HEADER = "X-Hank-Commercial-Claim"
COMMERCIAL_WORK_AUTHORIZATION_HEADER = "X-Hank-Work-Authorization"


class CommercialWorkStartError(RuntimeError):
  """A commercial request cannot safely proceed to runtime construction."""

  def __init__(self, code: str, message: str, *, status_code: int) -> None:
    self.code = code
    self.status_code = status_code
    super().__init__(message)


@dataclass(frozen=True)
class CommercialWorkStartFacts:
  operation: str
  provider: str
  billing_mode: str
  capability_id: str | None


@dataclass(frozen=True)
class VerifiedCommercialWorkStartClaim:
  """Verified account identity awaiting facts from the prepared request."""

  claim: VerifiedCommercialClaim
  _authorization_token: str = field(repr=False, compare=False)
  _gate_binding: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class PendingCommercialWorkStart:
  """Token-free verified authority awaiting durable one-time consumption."""

  claim: VerifiedCommercialClaim
  authorization: VerifiedWorkAuthorization
  _gate_binding: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class CommercialWorkStartContext:
  """Token-free request context available only after durable consumption."""

  claim: VerifiedCommercialClaim
  authorization: VerifiedWorkAuthorization
  consumption: WorkAuthorizationConsumptionRecord


CommercialWorkStartFactsResolver = Callable[
  [Any, Any, str | None], CommercialWorkStartFacts
]
CommercialWorkStartPreflight = Callable[[PendingCommercialWorkStart], None]


def require_commercial_child_provider(
  work_start: CommercialWorkStartContext | None,
  provider: str | None,
) -> None:
  """Bind delegated provider work to the root authorization before dispatch.

  Money: the child resolves its own provider bind, which the root
  authorization's request-path provider check never sees, and a child on
  another provider would spend under a funding route authorized for one.
  """
  if work_start is None:
    return
  if not isinstance(work_start, CommercialWorkStartContext):
    raise CommercialWorkStartError(
      "commercial_child_authority_invalid",
      "Commercial child authority is invalid.",
      status_code=503,
    )
  resolved_provider = str(provider or "").strip().lower()
  if not resolved_provider:
    raise CommercialWorkStartError(
      "commercial_child_provider_unavailable",
      "Commercial child provider is unavailable.",
      status_code=503,
    )
  if resolved_provider != work_start.authorization.provider:
    raise CommercialWorkStartError(
      "commercial_child_provider_mismatch",
      "Commercial child provider differs from authorized work: bind the "
      f"child to {work_start.authorization.provider!r} or start it under a "
      f"work authorization for {resolved_provider!r}.",
      status_code=403,
    )


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
        provider=facts.provider,
        billing_mode=facts.billing_mode,
        capability_id=facts.capability_id,
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
  "COMMERCIAL_CLAIM_HEADER",
  "COMMERCIAL_WORK_AUTHORIZATION_HEADER",
  "CommercialWorkStartContext",
  "CommercialWorkStartError",
  "CommercialWorkStartFacts",
  "CommercialWorkStartFactsResolver",
  "CommercialWorkStartPreflight",
  "CommercialWorkStartGate",
  "PendingCommercialWorkStart",
  "require_commercial_child_provider",
]
