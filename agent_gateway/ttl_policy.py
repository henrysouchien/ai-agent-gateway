"""User/key entitlement TTL seam; absent records preserve configured defaults."""
from __future__ import annotations

from typing import Callable

SESSION_TTL = "session"
CONTROL_SESSION_TTL = "control_session"
AUTONOMOUS_SESSION_AUTHORITY_TTL = "autonomous_session_authority"
GSHEETS_BROKER_SESSION_TTL = "gsheets_broker_session"
AGENT_API_CLAIM_TTL = "agent_api_claim"

TtlResolver = Callable[[str, str], int | None]


def default_ttl_resolver(subject: str, ttl_name: str) -> None:
  """No entitlement binding: retain the site's existing TTL."""
  return None


def resolve_ttl_seconds(
  resolver: TtlResolver,
  subject: str,
  ttl_name: str,
  default: int,
) -> int:
  """Use a positive integer entitlement TTL, otherwise retain the default."""
  resolved = resolver(subject, ttl_name)
  return resolved if type(resolved) is int and resolved > 0 else default
