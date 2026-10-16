from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
import json
import os
from typing import Any, Mapping, Protocol

from agent_gateway.approval_continuation import ContinueStatus
from agent_gateway.approval_policy import (
  approval_is_executable,
  approval_request_from_projection,
  approval_request_projection,
  utc_now,
)
from agent_gateway.autonomous_runner import (
  AutonomousRegistry,
  AutonomousTask,
)
from agent_gateway.autonomous_approval_channel import (
  AutonomousApprovalChannelParent,
  AutonomousApprovalChannelAuthority,
)
from agent_gateway.autonomous_claim_broker import AutonomousClaimBroker
from agent_gateway.approval_store import (
  ApprovalExecutableBindingError,
  _restore_executable_result,
)
from agent_gateway.tool_policy_registry import PreparedToolCall
from agent_gateway.providers.anthropic_oauth import is_anthropic_generation_ref
from agent_gateway.admission import root_context
from agent_gateway.agent_session_log import AgentSessionLog
from agent_gateway.agent_session_log_records import AgentSessionRef, resolve_agent_session_id
from agent_gateway.openai_history_fence import scope_provider_session_id


class _EnsureAutonomousApprovalDeliveryAudited(Protocol):
  async def __call__(
    self,
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
  ) -> dict[str, Any]: ...


class _AutonomousApprovalDeliveryTransaction(Protocol):
  def __call__(
    self,
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
    approved: bool,
  ) -> AbstractContextManager[None]: ...


class _RecordAutonomousApprovalDeliveryFailure(Protocol):
  async def __call__(
    self,
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
    error: str,
  ) -> dict[str, Any]: ...


def autonomous_run_accepts_approval_decisions(
  record: AutonomousTask,
) -> bool:
  if record.state not in {"running", "approval_pending", "remediating", "awaiting_approval", "continued"}:
    return False
  return record.approval_continuation is not None or (
    record.proc is None or record.proc.returncode is None
  )


def autonomous_decision_unavailable(
  record: AutonomousTask,
) -> str | None:
  if not autonomous_run_accepts_approval_decisions(record):
    return "Autonomous run is not running"
  if record.approval_continuation is not None:
    return None
  if type(getattr(
    record,
    "approval_channel",
    None,
  )) is not AutonomousApprovalChannelParent:
    return "Autonomous approval channel unavailable"
  return None


def autonomous_approval_delivery_context(
  record: AutonomousTask,
  *,
  tool_call_id: str,
  nonce: str,
) -> dict[str, str]:
  return {
    "task_id": record.task_id,
    "control_run_id": record.control_run_id,
    "session_id": record.session_id,
    "channel_id": record.channel_id,
    "tool_call_id": tool_call_id,
    "nonce": nonce,
  }


def autonomous_approval_authoritative_identity(
  record: AutonomousTask,
  *,
  approval_id: str,
  tool_call_id: str,
) -> dict[str, str | None]:
  owner_user_id = str(record.owner_user_id or "").strip()
  if not owner_user_id:
    raise ValueError("autonomous run owner_user_id is required")
  return {
    "approval_id": approval_id,
    "tool_call_id": tool_call_id,
    "user_id": owner_user_id,
    "request_id": record.control_run_id,
    "run_id": record.control_run_id,
    "session_id": record.session_id,
    "channel": record.channel,
  }

@dataclass(frozen=True, slots=True, repr=False)
class DelegatedAutonomousExecutableCapture:
  """Original operands correlated once against the authenticated child launch."""

  approval_id: str
  tool_call_id: str
  nonce: str
  tool_name: str
  job_id: str
  tenant_storage_id: str
  credential_generation_ref: str | None
  prepared_call: PreparedToolCall
  dispatch_context: Mapping[str, object]
  session_log_path: str | None = None
  pack_cursor: Mapping[str, Any] | None = None


def admit_delegated_executable_capture(
  record: AutonomousTask,
  payload: dict[str, Any],
) -> DelegatedAutonomousExecutableCapture:
  """Read child identity from the verified launch, never from private operands."""

  broker = record.claim_broker
  channel = record.approval_channel
  if (
    type(broker) is not AutonomousClaimBroker
    or type(channel) is not AutonomousApprovalChannelParent
  ):
    raise ApprovalExecutableBindingError("call_identity_mismatch")
  envelope = broker.envelope
  expected_channel = AutonomousApprovalChannelAuthority(
    launch_nonce=envelope.nonce,
    task_id=envelope.task_id,
    control_run_id=envelope.control_run_id,
    session_id=envelope.session_authority.ordinary_authority.session_id,
    channel_id=envelope.channel_id,
  )
  if (
    channel.authority != expected_channel
    or envelope.task_id != record.task_id
    or envelope.control_run_id != record.control_run_id
    or envelope.channel_id != record.channel_id
    or envelope.owner_user_id != record.owner_user_id
    or record.event_channel is None
    or record.event_channel.channel_id != envelope.channel_id
  ):
    raise ApprovalExecutableBindingError("call_identity_mismatch")
  launch = record.launch_approval_continuation
  if envelope.admitted is not None:
    tenant = envelope.admitted[0].tenant
    job_id = launch["job_id"] if launch is not None else envelope.task_id
  else:
    context = root_context(
      envelope.session_authority.to_gateway_session(),
      withheld_tools=frozenset(),
      research_file_id=envelope.workload.research_file_id,
    )
    tenant = context.tenant
    job_id = launch["job_id"] if launch is not None else context.job_id
  generation_ref = (
    envelope.bind.credential_ref
    if is_anthropic_generation_ref(envelope.bind.credential_ref)
    else None
  )
  if (
    tenant is None
    or job_id is None
    or payload.get("job_id") != job_id
    or payload.get("tenant_storage_id") != tenant.storage_id
    or tenant.storage_id != envelope.owner_user_id
    or payload.get("credential_generation_ref") != generation_ref
    or any(
      type(payload.get(key)) is not str or not payload[key]
      for key in ("approval_id", "tool_call_id", "nonce", "tool_name")
    )
  ):
    raise ApprovalExecutableBindingError("call_identity_mismatch")
  try:
    prepared_call = PreparedToolCall(
      prepared_input=payload["prepared_input"],
      exact_backend=payload["exact_backend"],
    )
    dispatch_context = PreparedToolCall(
      prepared_input=payload["dispatch_context"],
    ).prepared_input
  except (KeyError, TypeError, ValueError):
    raise ApprovalExecutableBindingError("operand_mismatch") from None
  session_log_path = payload.get("session_log_path")
  if session_log_path is not None:
    authority = envelope.workload.session_log_authority
    if authority is None:
      raise ApprovalExecutableBindingError("call_identity_mismatch")
    if authority.layout == "v2":
      expected_path = authority.active_path
    else:
      agent_id = scope_provider_session_id(
        envelope.workload.profile,
        provider=envelope.bind.provider,
        durable=True,
        openai_epoch=authority.provider_session_epoch,
      )
      expected_path = str(AgentSessionLog.path_for_session(
        authority.base_path,
        AgentSessionRef(
          user_id=tenant.storage_id,
          agent_id=agent_id,
          agent_session_id=resolve_agent_session_id(tenant.storage_id, agent_id),
        ),
      ))
    if session_log_path != expected_path:
      raise ApprovalExecutableBindingError("call_identity_mismatch")
  return DelegatedAutonomousExecutableCapture(
    approval_id=payload["approval_id"],
    tool_call_id=payload["tool_call_id"],
    nonce=payload["nonce"],
    tool_name=payload["tool_name"],
    job_id=job_id,
    tenant_storage_id=tenant.storage_id,
    credential_generation_ref=generation_ref,
    prepared_call=prepared_call,
    dispatch_context=dispatch_context,
    session_log_path=session_log_path,
    pack_cursor=payload.get("pack_cursor"),
  )



async def create_delegated_autonomous_approval(
  *,
  store: Any,
  policy: Any | None,
  record: AutonomousTask,
  event: dict[str, Any],
  private_executable_capture: DelegatedAutonomousExecutableCapture | None = None,
) -> Any:
  """Record the durable row for a request a delegated child authored.

  The child owns no ledger, so this is the only writer of its approval request.
  Every identity field is stamped from the run record before the row exists: a
  child cannot name another run, another session or another user's approval.
  """

  if not isinstance(event, dict):
    raise RuntimeError("autonomous approval request event is invalid")
  approval_id = str(event.get("approval_id") or "").strip()
  tool_call_id = str(event.get("tool_call_id") or "").strip()
  nonce = str(event.get("nonce") or "").strip()
  if not approval_id or not tool_call_id or not nonce:
    raise RuntimeError(
      "autonomous approval request event is missing its identity"
    )
  try:
    request = approval_request_from_projection(event.get("durable_request"))
  except (TypeError, ValueError) as exc:
    raise RuntimeError(
      "autonomous approval request projection is invalid"
    ) from exc
  if (
    request.approval_id != approval_id
    or request.tool_call_id != tool_call_id
  ):
    raise RuntimeError(
      "autonomous approval request projection disagrees with its event"
    )
  if private_executable_capture is not None and (
    private_executable_capture.approval_id != approval_id
    or private_executable_capture.tool_call_id != tool_call_id
    or private_executable_capture.nonce != nonce
    or private_executable_capture.tool_name != request.tool_name
    or event.get("tool_name") != request.tool_name
  ):
    raise ApprovalExecutableBindingError("call_identity_mismatch")
  authoritative_identity = autonomous_approval_authoritative_identity(
    record,
    approval_id=approval_id,
    tool_call_id=tool_call_id,
  )
  broker = record.claim_broker
  if type(broker) is not AutonomousClaimBroker:
    raise ApprovalExecutableBindingError("call_identity_mismatch")
  # This row retains the signed child's concrete channel. The run's optional
  # channel remains an operator restriction, where None means unrestricted.
  authoritative_identity["channel"] = (
    broker.envelope.session_authority.ordinary_authority.channel
  )
  requested_at = utc_now()
  request = replace(
    request,
    **authoritative_identity,
    state="pending_user",
    authorization_mode="HUMAN",
    cache_reference=None,
    grant_reference=None,
    requested_at=requested_at,
    expires_at=None,
    decided_at=None,
    decider_id=None,
    decider_role=None,
    decision=None,
    decision_reason=None,
    state_version=0,
    # The child projects these, and before this they were written into the parent's
    # ledger verbatim. They are not descriptive: approvals.py mints a durable
    # PersistentGrant carrying `args_predicate` when the constraint permits reuse, so a
    # child could name the predicate its own future writes would be pre-approved under.
    # The parent neutralises every reuse/grant field; a delegated approval is decided
    # once, for one write, and grants nothing forward.
    approval_reuse_mode="legacy",
    approval_reuse_key=None,
    persistent_grant_scope=None,
    args_predicate=None,
    # The child must not choose whether its own write needs a frozen-owner approval.
    # Pinned to the value delegated requests effectively carry today, so no live flow
    # changes; the point is that the PARENT decides it. `legacy_unknown` would make
    # every delegated approval un-approvable (409) and `fresh_human_owner` would add
    # owner-role checks to every decision, so neither is a safe unilateral default.
    approval_constraint="standard",
    required_owner_user_id=None,
    policy_id=str(
      getattr(policy, "policy_id", None) or request.policy_id
    ),
    policy_version=str(
      getattr(policy, "policy_version", None) or request.policy_version
    ),
  )
  stored, _created = await store.create_or_get_by_tool_call_id(request)
  if private_executable_capture is not None:
    await store.record_executable_binding(
      approval_id=stored.approval_id,
      job_id=private_executable_capture.job_id,
      tool_call_id=private_executable_capture.tool_call_id,
      nonce=private_executable_capture.nonce,
      tenant_storage_id=private_executable_capture.tenant_storage_id,
      tool_name=private_executable_capture.tool_name,
      prepared_call=private_executable_capture.prepared_call,
      dispatch_context=private_executable_capture.dispatch_context,
      credential_generation_ref=private_executable_capture.credential_generation_ref,
    )
    if private_executable_capture.session_log_path is not None:
      record.approval_continuation = {
        "approval_id": stored.approval_id,
        "job_id": private_executable_capture.job_id,
        "tool_call_id": private_executable_capture.tool_call_id,
        "nonce": private_executable_capture.nonce,
        "tenant_storage_id": private_executable_capture.tenant_storage_id,
        "session_log_path": private_executable_capture.session_log_path,
        "pack_cursor": (
          dict(private_executable_capture.pack_cursor)
          if private_executable_capture.pack_cursor is not None else None
        ),
      }
  return stored


def decided_approval_continuation(
  *,
  parked: Mapping[str, Any],
  binding: Any,
  decision: Any,
) -> dict[str, Any] | ContinueStatus:
  """Sign the decided original call into the successor's launch, or say why not.

  The parent alone reads the ledger. An approval carries the claim token its
  successor's launch will commit, but nothing is claimed here: the claim is
  the call's one execution attempt, and it is committed by
  ``claim_continued_executable_attempt`` only once the successor can run.
  """

  claim_token: str | None = None
  if decision.state in {"denied", "auto_denied", "expired"}:
    verdict = "denied"
  elif approval_is_executable(decision):
    verdict = "approved"
    claim_token = os.urandom(16).hex()
  else:
    return "not_restorable"
  return {
    "approval_id": binding.approval_id,
    "job_id": binding.job_id,
    "tool_call_id": binding.tool_call_id,
    "nonce": binding.nonce,
    "tenant_storage_id": binding.tenant_storage_id,
    "tool_name": binding.tool_name,
    "decision": verdict,
    "decision_reason": decision.decision_reason,
    "decision_request": approval_request_projection(decision),
    "decision_state_version": decision.state_version,
    "claim_token": claim_token,
    "prepared_input": PreparedToolCall(
      prepared_input=binding.prepared_input,
      exact_backend=binding.exact_backend,
    ).materialize_input(),
    "exact_backend": binding.exact_backend,
    "dispatch_context": PreparedToolCall(
      prepared_input=binding.dispatch_context,
    ).materialize_input(),
    "credential_generation_ref": binding.credential_generation_ref,
    "operand_digest": binding.operand_digest,
    "session_log_path": parked["session_log_path"],
    "pack_cursor": parked["pack_cursor"],
  }


async def claim_continued_executable_attempt(
  *,
  store: Any,
  continuation: Mapping[str, Any],
) -> None:
  """Commit the approved call's one execution attempt for its successor.

  The caller is the launch's final consumer frontier: the successor exists but
  holds no credential yet, so it cannot run the call until this returns. Any
  failure before this point leaves the ledger unclaimed and the decision
  redeliverable; a committed claim with no recorded result reads back as
  ``execution_outcome_unknown`` and is never released.
  """

  if continuation["decision"] != "approved":
    return
  binding, _decision = await store._read_executable_binding_for_executor(
    job_id=continuation["job_id"],
    tool_call_id=continuation["tool_call_id"],
    nonce=continuation["nonce"],
    tenant_storage_id=continuation["tenant_storage_id"],
  )
  completed = await store._claim_executable_binding_for_executor(
    binding=binding,
    decision_state_version=continuation["decision_state_version"],
    claim_token=continuation["claim_token"],
  )
  if completed is not None:
    raise ApprovalExecutableBindingError("execution_outcome_unknown")


async def record_continued_executable_result(
  *,
  store: Any,
  record: AutonomousTask,
  completed: Any,
) -> None:
  """Record the exact result a successor returned for its parent-claimed call.

  Identity comes from this run's signed launch, never from the child's frame.
  """

  launch = record.launch_approval_continuation
  if (
    store is None
    or launch is None
    or launch.get("decision") != "approved"
    or type(completed) is not dict
    or any(
      completed.get(key) != launch[key]
      for key in ("approval_id", "job_id", "tool_call_id", "nonce")
    )
  ):
    raise ApprovalExecutableBindingError("call_identity_mismatch")
  result_json = completed.get("result_json")
  try:
    if type(result_json) is not str:
      raise ValueError("completed executable result is not JSON text")
    outcome, error = json.loads(result_json)
    tool_result = (_restore_executable_result(outcome), error)
  except (TypeError, ValueError, ApprovalExecutableBindingError):
    raise ApprovalExecutableBindingError("result_mismatch") from None
  binding, _decision = await store._read_executable_binding_for_executor(
    job_id=launch["job_id"],
    tool_call_id=launch["tool_call_id"],
    nonce=launch["nonce"],
    tenant_storage_id=launch["tenant_storage_id"],
  )
  await store._complete_executable_binding_for_executor(
    binding=binding,
    decision_state_version=launch["decision_state_version"],
    claim_token=launch["claim_token"],
    tool_result=tool_result,
  )



def require_matching_autonomous_delivery(
  record: AutonomousTask,
  delivery: Mapping[str, Any],
  request_record: Any,
  *,
  approval_id: str,
  tool_call_id: str,
  nonce: str,
  approved: bool,
) -> None:
  """Refuse an outbox row whose identity diverges from the durable request.

  Delivering such a row would hand one request's or run's decision to another
  run, so a mismatch is treated as corruption, never delivered.
  """
  from .approvals import _channel_matches

  expected_delivery = {
    "approval_id": approval_id,
    "tool_call_id": tool_call_id,
    "nonce": nonce,
    "task_id": record.task_id,
    "control_run_id": record.control_run_id,
    "session_id": record.session_id,
    "channel_id": record.channel_id,
    "allow_tool_type": False,
  }
  if any(
    delivery.get(field_name) != expected
    for field_name, expected in expected_delivery.items()
  ):
    raise RuntimeError(
      "Autonomous approval delivery outbox identity mismatch"
    )
  if delivery.get("approved") != approved:
    raise ValueError(
      "Approval decision conflicts with the durable autonomous decision"
    )
  expected_request = {
    **autonomous_approval_authoritative_identity(
      record,
      approval_id=approval_id,
      tool_call_id=tool_call_id,
    ),
    "decider_id": str(record.owner_user_id or "").strip(),
  }
  if any(
    (
      not _channel_matches(expected, getattr(request_record, field_name, None))
      if field_name == "channel"
      else getattr(request_record, field_name, None) != expected
    )
    for field_name, expected in expected_request.items()
  ):
    raise RuntimeError(
      "Autonomous approval durable identity mismatch"
    )
  expected_state = "approved" if approved else "denied"
  if getattr(request_record, "state", None) != expected_state:
    raise RuntimeError(
      "Autonomous approval delivery disagrees with durable state"
    )


async def deliver_autonomous_approval_outbox(
  *,
  registry: AutonomousRegistry,
  store: Any,
  record: AutonomousTask,
  request_record: Any,
  delivery: Mapping[str, Any],
  approval_id: str,
  tool_call_id: str,
  nonce: str,
  approved: bool,
  user_id: str,
  channel: str | None,
) -> None:
  require_matching_autonomous_delivery(
    record,
    delivery,
    request_record,
    approval_id=approval_id,
    tool_call_id=tool_call_id,
    nonce=nonce,
    approved=approved,
  )
  delivery_state = delivery.get("state")
  if delivery_state in {"published", "acknowledged"}:
    return
  if delivery_state != "pending":
    raise RuntimeError(
      "Autonomous approval delivery outbox state is invalid"
    )
  ensure_audited: _EnsureAutonomousApprovalDeliveryAudited | None = getattr(
    store,
    "ensure_autonomous_approval_delivery_audited",
    None,
  )
  if not callable(ensure_audited):
    raise RuntimeError(
      "Autonomous approval delivery audit gate unavailable"
    )
  delivery = await ensure_audited(
    approval_id,
    tool_call_id=tool_call_id,
    nonce=nonce,
  )
  require_matching_autonomous_delivery(
    record,
    delivery,
    request_record,
    approval_id=approval_id,
    tool_call_id=tool_call_id,
    nonce=nonce,
    approved=approved,
  )
  if delivery.get("audit_state") != "ready":
    raise RuntimeError(
      "Autonomous approval delivery audit receipt is not ready"
    )
  if delivery.get("state") != "pending":
    if delivery.get("state") in {"published", "acknowledged"}:
      return
    raise RuntimeError(
      "Autonomous approval delivery outbox state is invalid"
    )
  unavailable = autonomous_decision_unavailable(record)
  if unavailable is not None:
    raise RuntimeError(unavailable)
  append_transaction: _AutonomousApprovalDeliveryTransaction | None = getattr(
    store,
    "autonomous_approval_delivery_append_transaction",
    None,
  )
  if not callable(append_transaction):
    raise RuntimeError(
      "Autonomous approval cancellation fence unavailable"
    )
  duplicate_transaction: _AutonomousApprovalDeliveryTransaction | None = getattr(
    store,
    "autonomous_approval_delivery_duplicate_transaction",
    None,
  )
  if not callable(duplicate_transaction):
    raise RuntimeError(
      "Autonomous approval duplicate recovery unavailable"
    )

  async def publish_to_child_inbox() -> Any:
    if (
      record.approval_continuation is not None
      and record.approval_continuation["approval_id"] == approval_id
    ):
      continued = await registry.continue_approval(
        approval_id=approval_id, tenant_storage_id=user_id,
      )
      if continued is None or continued.status not in {"delivered", "already_delivered"}:
        raise RuntimeError("Autonomous approval continuation is not restorable")
      with append_transaction(
        approval_id, tool_call_id=tool_call_id, nonce=nonce, approved=approved,
      ):
        pass
      return continued
    return await registry.send_approval_decision(
      record.control_run_id,
      user_id=user_id,
      channel=channel,
      approval_id=approval_id,
      tool_call_id=tool_call_id,
      nonce=nonce,
      approved=approved,
      decided_at_ns=int(delivery["decided_at_ns"]),
      delivery_sequence=int(delivery["delivery_sequence"]),
      publication_transaction=lambda: append_transaction(
        approval_id,
        tool_call_id=tool_call_id,
        nonce=nonce,
        approved=approved,
      ),
      sent_reconciliation=lambda: duplicate_transaction(
        approval_id,
        tool_call_id=tool_call_id,
        nonce=nonce,
        approved=approved,
      ),
    )

  try:
    await publish_to_child_inbox()
  except BaseException as exc:
    record_failure: _RecordAutonomousApprovalDeliveryFailure | None = getattr(
      store,
      "record_autonomous_approval_delivery_failure",
      None,
    )
    if callable(record_failure) and isinstance(exc, Exception):
      await record_failure(
        approval_id,
        tool_call_id=tool_call_id,
        nonce=nonce,
        error=f"{type(exc).__name__}: {exc}",
      )
    raise


__all__ = [
  "autonomous_approval_authoritative_identity",
  "create_delegated_autonomous_approval",
  "autonomous_approval_delivery_context",
  "autonomous_decision_unavailable",
  "autonomous_run_accepts_approval_decisions",
  "claim_continued_executable_attempt",
  "decided_approval_continuation",
  "deliver_autonomous_approval_outbox",
  "record_continued_executable_result",
  "require_matching_autonomous_delivery",
]
