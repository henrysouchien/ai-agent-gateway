"""The one composition of the approval ledger and policy.

A process that decides its own approvals holds both together: the durable
ledger that records the request and the policy that decides it.

`build_approval_subsystem` takes every input explicitly and reads no
environment, so one composer serves every caller: the gateway server, a
standalone in-process run (`scripts/run_idea_to_thesis.py`), and the edge's
per-storage approval registry, which composes one subsystem per placed user.
`build_approval_subsystem_from_environment` is the one place a process's own
environment becomes those inputs; the process entries call it once.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict
from typing_extensions import TypeVar

from .approval_audit import ApprovalAuditEmitter
from .approval_notifications import (
  build_env_approval_notification_destination_resolver,
  build_env_telegram_approval_notification_sender,
)
from .approval_resolver import resolve_policy
from .approval_store import ApprovalRequestStore, SQLiteApprovalStore, resolve_approval_db_path
from .audit_resolver import resolve_audit_writer
from .audit_writer import AuditWriter
from .tool_redaction import get_audit_hmac_key_id, get_audit_hmac_secret
from .approval_remote import (
  ApprovalStoreAuditEmitter,
  AsyncApprovalsEgressTransport,
  APPROVALS_EGRESS_URL_ENV,
  BlockingApprovalsEgressTransport,
  BoundApprovalRemoteClient,
)
from hank_model_transport.user_runtime_credentials import UserRuntimeCredentials, load_user_runtime_credentials

StoreT = TypeVar(
  "StoreT", bound=ApprovalRequestStore, default=SQLiteApprovalStore, covariant=True,
)


@dataclass(frozen=True, slots=True)
class ApprovalSubsystem(Generic[StoreT]):
  """The ledger, its audit trail and the policy."""

  audit_writer: Any
  audit_emitter: ApprovalStoreAuditEmitter | None
  store: StoreT
  policy: Any
  remote: bool = False


class PlacedApprovalConfiguration(BaseModel):
  """The placed process's parsed approval authority."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  credentials: UserRuntimeCredentials
  approvals_egress_url: AnyHttpUrl
  transport: Literal["local", "agentcore"]


def load_placed_approval_configuration(
  environ: Mapping[str, str],
) -> PlacedApprovalConfiguration | None:
  credentials = load_user_runtime_credentials(environ)
  if credentials is None:
    return None
  return PlacedApprovalConfiguration.model_validate({
    "credentials": credentials,
    "approvals_egress_url": environ.get(APPROVALS_EGRESS_URL_ENV),
    "transport": environ.get("HANK_APPROVAL_TRANSPORT"),
  })


def build_runtime_approval_subsystem(
  *,
  audit_hmac_secret: bytes | None = None,
  audit_hmac_key_id: str | None = None,
  tool_input_redactor: Callable[..., dict[str, Any]] | None = None,
) -> ApprovalSubsystem[ApprovalRequestStore]:
  """Compose the process entry; placed processes never open a local ledger."""
  configuration = load_placed_approval_configuration(os.environ)
  if configuration is None:
    return build_approval_subsystem_from_environment(
      audit_hmac_secret=audit_hmac_secret,
      audit_hmac_key_id=audit_hmac_key_id,
      tool_input_redactor=tool_input_redactor,
    )
  from .approval_remote_store import RemoteApprovalRequestStore, RuntimeExecutionAuditEmitter
  from .claim_signing_authority import GatewayClaimSigningAuthority

  base_url = str(configuration.approvals_egress_url)
  if configuration.transport == "agentcore":
    client = BoundApprovalRemoteClient(
      sync_transport=BlockingApprovalsEgressTransport(base_url),
      async_transport=AsyncApprovalsEgressTransport(base_url),
    )
  else:
    delegation = configuration.credentials.approval_claim_signing
    authority = GatewayClaimSigningAuthority(delegation.key)
    client = BoundApprovalRemoteClient(
      sync_transport=BlockingApprovalsEgressTransport(
        base_url,
        mint_claim=lambda: authority.sign_user_claim(
          user_id=delegation.user_id,
          user_email=delegation.user_email,
        ),
      ),
    )
  emitter = RuntimeExecutionAuditEmitter(client)
  store = RemoteApprovalRequestStore(client, audit_emitter=emitter)
  return ApprovalSubsystem(
    audit_writer=None,
    audit_emitter=store.audit_emitter,
    store=store,
    policy=resolve_policy(store=store),
    remote=True,
  )


def build_approval_subsystem(
  *,
  db_path: Path,
  audit_writer: AuditWriter,
  audit_hmac_secret: bytes,
  audit_hmac_key_id: str,
  notification_sender: Any | None,
  notification_destination_resolver: Any | None,
  tool_input_redactor: Callable[..., dict[str, Any]] | None = None,
) -> ApprovalSubsystem[SQLiteApprovalStore]:
  audit_emitter = ApprovalAuditEmitter(
    writer=audit_writer,
    deployment_secret=audit_hmac_secret,
    key_id=audit_hmac_key_id,
    tool_input_redactor=tool_input_redactor,
  )
  store = SQLiteApprovalStore(
    path=db_path,
    audit_emitter=audit_emitter,
    notification_destination_resolver=notification_destination_resolver,
    notification_sender=notification_sender,
  )
  return ApprovalSubsystem(
    audit_writer=audit_writer,
    audit_emitter=audit_emitter,
    store=store,
    policy=resolve_policy(store=store),
  )


def build_approval_subsystem_from_environment(
  *,
  audit_hmac_secret: bytes | None = None,
  audit_hmac_key_id: str | None = None,
  tool_input_redactor: Callable[..., dict[str, Any]] | None = None,
) -> ApprovalSubsystem[SQLiteApprovalStore]:
  """Compose this process's own approvals from its environment."""

  return build_approval_subsystem(
    db_path=resolve_approval_db_path(),
    audit_writer=resolve_audit_writer(),
    audit_hmac_secret=get_audit_hmac_secret() if audit_hmac_secret is None else audit_hmac_secret,
    audit_hmac_key_id=get_audit_hmac_key_id() if audit_hmac_key_id is None else audit_hmac_key_id,
    notification_sender=build_env_telegram_approval_notification_sender(),
    notification_destination_resolver=build_env_approval_notification_destination_resolver(),
    tool_input_redactor=tool_input_redactor,
  )


__all__ = [
  "ApprovalSubsystem",
  "PlacedApprovalConfiguration",
  "load_placed_approval_configuration",
  "build_runtime_approval_subsystem",
  "build_approval_subsystem",
  "build_approval_subsystem_from_environment",
]
