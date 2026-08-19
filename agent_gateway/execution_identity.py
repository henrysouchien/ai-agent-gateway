"""One identity, one source (D-B6-1).

Before B-6 an execution's identity was assembled at the point of use.  The
child dispatcher took five separate identity arguments (``session``,
``user_id``, ``risk_user_id``, ``channel``, ``credentials_resolver_active``);
the fork handoff picked a credential out of a three-tier fallback chain
(session handle → runner attribute → ``auth_config`` row).  Nothing named the
identity an operation actually executes under, so nothing could be compared.

This module is that name.  :class:`ExecutionIdentity` — the wire value carried
on :class:`~agent_workflow_contracts.ResolvedAuthority` — is derived in exactly
one place, :func:`execution_identity_from_session`, from exactly one fact: the
session's own credential binding.  There is no fallback.  A session with no
bound credential has no credential handle in its identity, and callers that
require one say so out loud rather than reaching for a substitute.

:class:`DispatchIdentity` is the dispatcher-facing bundle: the resolved
``ExecutionIdentity`` plus the authenticated session the per-user MCP
projection reads (AVGO-LIVE-02).  The five separate constructor arguments
collapse into it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_workflow_contracts import ExecutionIdentity


def _text(value: Any) -> str:
  return str(value or "").strip()


def execution_identity_from_session(
  session: Any,
  *,
  bind: Any | None = None,
) -> ExecutionIdentity | None:
  """Resolve the one identity a session executes under.

  Single tier by construction: the session's credential binding is the only
  source consulted.  ``credential_handle_id`` is populated only from a bound
  :class:`~agent_gateway.capability_binding.CredentialHandle` that matches
  ``bind`` exactly (handle id, provider, principal) — a handle that does not
  match the capability bind is not a weaker identity, it is a different one,
  and the resolution fails rather than degrading.

  Returns ``None`` when the session carries no tenant at all.
  """

  if session is None:
    return None
  handle = getattr(session, "session_credential_handle", None)
  if handle is None:
    tenant_id = _text(getattr(session, "tenant_id", None))
    return ExecutionIdentity(tenant_id=tenant_id) if tenant_id else None

  handle_id = _text(getattr(handle, "handle_id", None))
  tenant_id = _text(getattr(handle, "tenant_id", None))
  if not handle_id or not tenant_id:
    return None
  if bind is not None:
    if (
      handle_id != getattr(bind, "credential_ref", None)
      or _text(getattr(handle, "provider", None)).lower()
      != _text(getattr(bind, "provider", None)).lower()
      or _text(getattr(handle, "principal", None)).lower()
      != _text(getattr(bind, "credential_principal", None)).lower()
    ):
      return None
  return ExecutionIdentity(
    tenant_id=tenant_id,
    credential_handle_id=handle_id,
  )


def resolved_execution_identity(runner: Any) -> ExecutionIdentity | None:
  """The identity one runner executes under — one read, no fallback chain."""

  session = (
    getattr(runner, "_gateway_session", None)
    or getattr(getattr(runner, "_dispatcher", None), "_session", None)
  )
  execution = getattr(runner, "_capability_execution", None)
  return execution_identity_from_session(
    session,
    bind=getattr(execution, "bind", None),
  )


@dataclass(frozen=True, slots=True)
class DispatchIdentity:
  """The exact identity one :class:`ToolDispatcher` executes under.

  ``execution`` is the frozen wire identity (``ResolvedAuthority.identity``);
  ``session`` is the authenticated :class:`GatewaySession` the per-user MCP
  credential projection derives its subject from.  Credential *material* never
  rides here.
  """

  execution: ExecutionIdentity | None = None
  session: Any | None = None
  user_id: str | None = None
  risk_user_id: int | None = None
  channel: str | None = None
  credentials_resolver_active: bool = False

  @property
  def tenant_id(self) -> str | None:
    return self.execution.tenant_id if self.execution is not None else None

  @property
  def credential_handle_id(self) -> str | None:
    return (
      self.execution.credential_handle_id
      if self.execution is not None
      else None
    )


def dispatch_identity(
  *,
  session: Any,
  execution: ExecutionIdentity | None = None,
  user_id: str | None = None,
  credentials_resolver_active: bool = False,
) -> DispatchIdentity:
  """Bundle one session's dispatch identity.

  ``execution`` is the authority's frozen identity when the caller resolved
  one; absent it, the same single derivation runs over the same session, so
  the dispatcher can never execute under an identity the resolver did not
  produce.
  """

  return DispatchIdentity(
    execution=(
      execution
      if execution is not None
      else execution_identity_from_session(session)
    ),
    session=session,
    user_id=(
      user_id
      if user_id is not None
      else getattr(session, "user_id", None)
    ),
    risk_user_id=getattr(session, "risk_user_id", None),
    channel=getattr(session, "channel", None),
    credentials_resolver_active=bool(credentials_resolver_active),
  )


__all__ = [
  "DispatchIdentity",
  "ExecutionIdentity",
  "dispatch_identity",
  "execution_identity_from_session",
  "resolved_execution_identity",
]
