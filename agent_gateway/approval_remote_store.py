"""Consumer approval store backed by the placement-bound approval client.

Placement composition owns transport selection and authentication. The wire
client owns serialization, error reconstruction and publication reservations.
The runtime audit adapter preserves the approval request's original audit identity.
"""
from __future__ import annotations

from typing import Any

from agent_gateway.approval_remote import (
  ExecutionAuditOutcome,
  RemoteExecutionAuditEmitter,
  BoundApprovalRemoteClient,
)
from agent_gateway.approval_policy import ApprovalRequest


class RuntimeExecutionAuditEmitter(RemoteExecutionAuditEmitter):
  """Reuse the request's original hash and redaction at the edge audit owner."""

  async def emit_execution_outcome(
    self,
    request: ApprovalRequest,
    raw_tool_args: dict[str, Any],
    outcome: ExecutionAuditOutcome,
    error_summary: str | None = None,
  ) -> None:
    await super().emit_execution_outcome(
      request=request, raw_tool_args={}, outcome=outcome,
      error_summary=error_summary,
    )


class RemoteApprovalRequestStore:
  """Expose only the consumer store surface, not wire-internal operations.

  Bind the client's typed methods directly so their signatures, sync/async
  behavior and domain results stay identical without a second RPC dispatcher.
  In particular, both publication context managers remain synchronous: append
  reserves before yielding and commits or aborts on exit; duplicate publication
  reconciles before yielding. Neither requires the caller's event loop.
  """

  def __init__(
    self,
    client: BoundApprovalRemoteClient,
    *,
    audit_emitter: RuntimeExecutionAuditEmitter,
  ) -> None:
    self._audit_emitter = audit_emitter
    self._client = client
    self.create = client.create
    self.create_or_get_by_tool_call_id = client.create_or_get_by_tool_call_id
    self.locate_request = client.locate_request
    self._locate_request_for_decision = client._locate_request_for_decision
    self.record_executable_binding = client.record_executable_binding
    self.park_executable_binding = client.park_executable_binding
    self._read_executable_binding_for_executor = client._read_executable_binding_for_executor
    self._read_completed_executable_result_for_executor = client._read_completed_executable_result_for_executor
    self._claim_executable_binding_for_executor = client._claim_executable_binding_for_executor
    self._complete_executable_binding_for_executor = client._complete_executable_binding_for_executor
    self.get = client.get
    self.get_by_tool_call_id = client.get_by_tool_call_id
    self.update_request = client.update_request
    self.transition_state = client.transition_state
    self.terminalize_pending_for_cancellation = client.terminalize_pending_for_cancellation
    self.abort_unpublished_approval = client.abort_unpublished_approval
    self.record_vote = client.record_vote
    self.create_persistent_grant = client.create_persistent_grant
    self.find_persistent_grant = client.find_persistent_grant
    self.revoke_persistent_grant = client.revoke_persistent_grant
    self.revoke_persistent_grants_for_approval = client.revoke_persistent_grants_for_approval
    self.fence_persistent_grants_for_cancellation = client.fence_persistent_grants_for_cancellation
    self.create_delegation_grant = client.create_delegation_grant
    self.get_delegation_grant = client.get_delegation_grant
    self.claim_delegation_grant = client.claim_delegation_grant
    self.revoke_delegation_grant = client.revoke_delegation_grant
    self.reconcile_prepared_business_model_change = client.reconcile_prepared_business_model_change
    self.create_raw_patch_authorization = client.create_raw_patch_authorization
    self.create_or_get_with_prepared_business_model_change = client.create_or_get_with_prepared_business_model_change
    self.record_vote_with_autonomous_delivery = client.record_vote_with_autonomous_delivery
    self.list_parked_requests = client.list_parked_requests
    self.record_policy_generation = client.record_policy_generation
    self.record_root_turn = client.record_root_turn
    self._read_root_turn_for_executor = client._read_root_turn_for_executor
    self._locate_root_turn_for_approval = client._locate_root_turn_for_approval
    self.link_root_turn_executable_binding = client.link_root_turn_executable_binding
    self.reissue_parked_approval = client.reissue_parked_approval
    self.get_prepared_business_model_change = client.get_prepared_business_model_change
    self.transition_prepared_business_model_change = client.transition_prepared_business_model_change
    self.enqueue_pending_approval_notification = client.enqueue_pending_approval_notification
    self.get_autonomous_approval_delivery = client.get_autonomous_approval_delivery
    self.ensure_autonomous_approval_delivery_audited = client.ensure_autonomous_approval_delivery_audited
    self.acknowledge_autonomous_approval_delivery = client.acknowledge_autonomous_approval_delivery
    self.record_autonomous_approval_delivery_failure = client.record_autonomous_approval_delivery_failure
    self.quarantine_autonomous_approval_delivery = client.quarantine_autonomous_approval_delivery
    self.autonomous_approval_delivery_recovery_window = client.autonomous_approval_delivery_recovery_window
    self.autonomous_approval_delivery_append_transaction = client.autonomous_approval_delivery_append_transaction
    self.autonomous_approval_delivery_duplicate_transaction = client.autonomous_approval_delivery_duplicate_transaction
    self.retry_failed_approval_notifications = client.retry_failed_approval_notifications
    self.resolve_raw_patch_authorization_reference = client.resolve_raw_patch_authorization_reference
    self.get_raw_patch_prepared_authorization = client.get_raw_patch_prepared_authorization
    self.consume_raw_patch_authorization = client.consume_raw_patch_authorization
    self.claim_raw_patch_authorization = client.claim_raw_patch_authorization
    self.release_raw_patch_authorization_claim = client.release_raw_patch_authorization_claim
    self.heartbeat_raw_patch_authorization_claim = client.heartbeat_raw_patch_authorization_claim
    self.get_raw_patch_authorization_consumption = client.get_raw_patch_authorization_consumption

  @property
  def audit_emitter(self) -> RuntimeExecutionAuditEmitter:
    return self._audit_emitter

  async def list_pending_autonomous_approval_deliveries(
    self, *, after_sequence: int = 0, through_sequence: int | None = None,
  ) -> list[dict[str, Any]]:
    return await self._client.list_pending_autonomous_approval_deliveries(
      after_sequence=after_sequence, through_sequence=through_sequence,
    )
