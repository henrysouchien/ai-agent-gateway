"""Default-off request gate for one authorized commercial provider-work start."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from hank_model_transport.capability_execution import MaterializedCredential
from model_authority.binding import CredentialHandle

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


@dataclass(frozen=True)
class MeteredSeat:
  """One turn's Hank-paid paid seat, minted only from a verified metered V4.

  The product's admission owner mints it (``metered_seat_minter``); the turn
  binds it as its ``service`` handle and attaches the same authorization.
  """

  authorization: VerifiedWorkAuthorization
  handle: CredentialHandle
  _materialize: Callable[[], MaterializedCredential] = field(repr=False, compare=False)

  def __post_init__(self) -> None:
    if (
      type(self.authorization) is not VerifiedWorkAuthorization
      or self.authorization.billing_mode != "metered"
      or self.handle.principal != "service"
      or self.handle.provider != self.authorization.provider
    ):
      raise ValueError("a metered seat is a service handle for its authorized provider")

  @property
  def provider(self) -> str:
    return self.handle.provider

  def materialize(self) -> MaterializedCredential:
    materialized = self._materialize()
    if type(materialized) is not MaterializedCredential or materialized.handle != self.handle:
      raise RuntimeError("metered seat materializer returned a different credential handle")
    return materialized


MeteredSeatMinter = Callable[[VerifiedWorkAuthorization, Any], MeteredSeat | None]


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
    metered_seat_minter: MeteredSeatMinter | None = None,
  ) -> None:
    self.enabled = enabled
    self._claim_verifier = claim_verifier
    self._authorization_verifier = authorization_verifier
    self._consumption_store = consumption_store
    self._facts_resolver = facts_resolver
    self._pre_consume = pre_consume
    self._metered_seat_minter = metered_seat_minter
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
    if claim_token is None and authorization_token is None:
      # No commercial headers: an ordinary authenticated turn on the session's
      # own credentials. It creates no commercial account or spend authority.
      return None
    if not claim_token or not authorization_token:
      raise CommercialWorkStartError(
        "commercial_work_authority_required",
        "Both commercial work-start headers are required.",
        status_code=401,
      )
    claim = self._verify_owner_claim(
      claim_token,
      owner_user_id=str(getattr(session, "owner_user_id", "") or "").strip(),
    )
    return VerifiedCommercialWorkStartClaim(
      claim=claim,
      _authorization_token=authorization_token,
      _gate_binding=self._binding,
    )

  def verify_session_claim(
    self,
    headers: Mapping[str, str],
    *,
    owner_user_id: str,
  ) -> VerifiedCommercialClaim | None:
    """The commercial claim a keyless session start presents, else None.

    Verified by the same claim verifier and owner check as ``verify_claim``.
    The claim names the session owner as a commercial account holder; it is
    not work authorization. Only a turn's verified, consumed metered V4 binds
    a credential (``metered_seat``).
    """
    if not self.enabled:
      return None
    claim_token = headers.get(COMMERCIAL_CLAIM_HEADER)
    if claim_token is None:
      return None
    if not claim_token:
      raise CommercialWorkStartError(
        "commercial_work_authority_required",
        "The commercial claim header is empty.",
        status_code=401,
      )
    return self._verify_owner_claim(claim_token, owner_user_id=owner_user_id)

  def _verify_owner_claim(self, claim_token: str, *, owner_user_id: str) -> VerifiedCommercialClaim:
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
    return claim

  def metered_seat(
    self,
    verified_claim: VerifiedCommercialWorkStartClaim | None,
    *,
    session: Any,
    request_id: str,
  ) -> MeteredSeat | None:
    """Mint the turn's metered paid seat from its verified authorization.

    Runs before the turn binds: a metered authorization decides the turn's
    credential, and ``authorize_request`` then attaches it only to that bind.
    A byok authorization, or one the minter leaves on today's path, is None.
    """
    minter = self._metered_seat_minter
    if not self.enabled or verified_claim is None or minter is None:
      return None
    if verified_claim._gate_binding is not self._binding:
      raise CommercialWorkStartError(
        "commercial_work_authority_invalid",
        "Commercial work-start authority is not bound to this request gate.",
        status_code=403,
      )
    authorization_verifier = self._authorization_verifier
    if authorization_verifier is None:
      raise CommercialWorkStartError(
        "commercial_work_start_unavailable",
        "Commercial work-start verification is unavailable.",
        status_code=503,
      )
    try:
      authorization = authorization_verifier.verify_token(
        verified_claim._authorization_token,
        execution_claim=verified_claim.claim,
        request_id=request_id,
        session_id=session.session_id,
      )
    except WorkAuthorizationError as exc:
      raise CommercialWorkStartError(
        "commercial_work_authority_invalid",
        "Commercial work-start authorization was refused by its verifier: "
        f"{exc.code}.",
        status_code=403,
      ) from exc
    if authorization.billing_mode != "metered":
      return None
    seat = minter(authorization, session)
    if seat is not None and (
      type(seat) is not MeteredSeat or seat.authorization != authorization
    ):
      raise CommercialWorkStartError(
        "commercial_work_start_unavailable",
        "The metered seat minter returned a seat for different authority.",
        status_code=503,
      )
    return seat

  def authorize_request(
    self,
    verified_claim: VerifiedCommercialWorkStartClaim | None,
    *,
    session: Any,
    request: Any,
    channel: str | None,
    metered_seat: MeteredSeat | None = None,
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
    if metered_seat is not None:
      bind = getattr(request, "capability_bind", None)
      if (
        authorization.token_sha256 != metered_seat.authorization.token_sha256
        or getattr(bind, "credential_principal", None) != "service"
        or getattr(bind, "credential_ref", None) != metered_seat.handle.handle_id
      ):
        raise CommercialWorkStartError(
          "commercial_metered_seat_unbound",
          "The metered paid seat is not the credential this turn bound.",
          status_code=403,
        )
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
  "MeteredSeat",
  "MeteredSeatMinter",
  "require_commercial_child_account",
  "commercial_budget_stop_fields",
]
