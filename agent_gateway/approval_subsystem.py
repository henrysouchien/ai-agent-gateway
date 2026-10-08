"""The one composition of the approval ledger and policy.

A process that decides its own approvals holds both together: the durable
ledger that records the request and the policy that decides it. The gateway
server composes it at startup; a standalone in-process run
(`scripts/run_idea_to_thesis.py`) composes the same one instead of deciding
approvals by a rule of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .approval_audit import ApprovalAuditEmitter
from .approval_notifications import (
  build_env_approval_notification_destination_resolver,
  build_env_telegram_approval_notification_sender,
)
from .approval_resolver import resolve_policy
from .approval_store import SQLiteApprovalStore, resolve_approval_db_path
from .audit_resolver import resolve_audit_writer
from .tool_redaction import get_audit_hmac_key_id, get_audit_hmac_secret


@dataclass(frozen=True, slots=True)
class ApprovalSubsystem:
  """The ledger, its audit trail and the policy."""

  audit_writer: Any
  audit_emitter: ApprovalAuditEmitter
  store: SQLiteApprovalStore
  policy: Any


def build_approval_subsystem(
  *,
  audit_hmac_secret: bytes | None = None,
  audit_hmac_key_id: str | None = None,
  tool_input_redactor: Callable[..., dict[str, Any]] | None = None,
) -> ApprovalSubsystem:
  audit_writer = resolve_audit_writer()
  audit_emitter = ApprovalAuditEmitter(
    writer=audit_writer,
    deployment_secret=(
      get_audit_hmac_secret() if audit_hmac_secret is None else audit_hmac_secret
    ),
    key_id=(
      get_audit_hmac_key_id() if audit_hmac_key_id is None else audit_hmac_key_id
    ),
    tool_input_redactor=tool_input_redactor,
  )
  approval_db_path = resolve_approval_db_path()
  store = SQLiteApprovalStore(
    path=approval_db_path,
    audit_emitter=audit_emitter,
    notification_destination_resolver=build_env_approval_notification_destination_resolver(),
    notification_sender=build_env_telegram_approval_notification_sender(),
  )
  return ApprovalSubsystem(
    audit_writer=audit_writer,
    audit_emitter=audit_emitter,
    store=store,
    policy=resolve_policy(store=store),
  )


__all__ = ["ApprovalSubsystem", "build_approval_subsystem"]
